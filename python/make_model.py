"""Make a model configuration: everything a board needs to run the first N layers of DeiT-Tiny.

Usage: .venv/bin/python python/make_model.py --board nexys --layers 2      (needs torch, timm, torchvision)

  --board nexys         Digilent Nexys A7-100T: 112 x 112 MNIST digits (50 tokens), a digit
                        classifier trained here (a linear probe on every row)
  --board opal_kelly    Opal Kelly XEM7310-A200: 224 x 224 ImageNet photos (197 tokens), the
                        model's own 1000-class head
  --layers N            1 .. 12 encoder layers
  --recalibrate         work the tensor scales out again and rewrite
                        python/calibration/<board>_scales.json (they cover all 12 layers)
  --retrain             nexys: train the digit classifier of N layers again
  --report              also print the accuracy on test images the scales and the classifier
                        never saw

Output: python/models/<board>_L<N>/: every stage of the integer pass on the test image (the
golden tensors), the descriptor program, the loads of every phase and host_config.json, which
python/run_image.py and python/ddr_check.py read (--model-dir). For opal_kelly also matmuls/,
every matmul on its own (tb_matmul_engine_opal_kelly).

Kept in the repo, made here when missing:
  python/calibration/<board>_scales.json       the scale of every tensor, all 12 layers
  python/calibration/nexys_L<N>_probe.npz      the digit classifier of N layers
  rtl/rom/<board>/                             the engine's tables (GELU, exp, 1/sqrt) for all
                                               12 layers; they are part of the bitstream. If the
                                               tables change (a recalibration), the script says
                                               so: the bitstream must then be built again.

What a board can run is fixed by its bitstream's buffer depths (BOARDS below,
the parameters of the board's top level): the script stops if a configuration
does not fit them. Any number of layers fits; the image size is the board's.
"""

import deterministic  # noqa: F401  (first: it sets the thread counts)

import argparse
import glob
import json
import os
import sys

import numpy as np
import timm
import torch
import torch.nn.functional as F

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HOST_DIR)
sys.path.insert(0, HOST_DIR)
import golden_model as golden  # noqa: E402
import layer_program  # noqa: E402
import preprocess  # noqa: E402
from int8_model import DATA_MAX, write_hex  # noqa: E402

CALIBRATION_DIR = os.path.join(HOST_DIR, "calibration")

# what each board's bitstream holds: image size, tokens, the words per bank of pool buffers 0..7
BOARDS = {
    "nexys": dict(size=112, tokens=50, weight_depth=16384,
                  depths=[1344, 16384, 5376, 1344, 1050, 2544, 1344, 16],
                  top="boards/nexys_a7/nexys_top_ddr.sv"),
    "opal_kelly": dict(size=224, tokens=197, weight_depth=55296,
                       depths=[4800, 55296, 19200, 4800, 14775, 9528, 4800, 16],
                       top="not built yet; tb_encoder_program uses these depths"),
}

# MNIST digits (nexys)
MNIST_DIR = os.path.join(HOST_DIR, "mnist")
MNIST_CALIBRATION = range(0, 128)        # train digits: the scales
MNIST_PROBE_TRAIN = range(128, 6128)     # train digits: the classifier
MNIST_EVALUATION = range(0, 2000)        # test digits: the accuracy
MNIST_TEST_IMAGE = 0                     # test digit of the golden vectors
# ImageNet sample photos (opal_kelly), from scripts/fetch_images.sh
IMAGENET_CALIBRATION = os.path.join(HOST_DIR, "images_calibration")
IMAGENET_HELDOUT = (os.path.join(HOST_DIR, "images"), os.path.join(HOST_DIR, "images_heldout"))


def progress(what, done, total):
    """One line at every quarter of a long loop."""
    if done * 4 // total != (done - 1) * 4 // total:
        print(f"  {what}: {done} of {total}", flush=True)


def load_model(board):
    size = BOARDS[board]["size"]
    model = timm.create_model(golden.MODEL_NAME, pretrained=True, img_size=size).eval()
    for block in model.blocks:
        block.attn.fused_attn = False  # so the softmax output passes through attn_drop
    assert model.pos_embed.shape[1] == BOARDS[board]["tokens"]
    if board == "nexys":
        assert tuple(model.pretrained_cfg["mean"]) == preprocess.MEAN
        assert tuple(model.pretrained_cfg["std"]) == preprocess.STD
    return model


class Data:
    """The board's images: calibration, the test image, evaluation. image() gives a torch image tensor."""

    def __init__(self, board, model):
        self.board = board
        if board == "nexys":
            import torchvision
            self.train = torchvision.datasets.MNIST(MNIST_DIR, train=True, download=True)
            self.test = torchvision.datasets.MNIST(MNIST_DIR, train=False, download=True)
            self.calibration = [("train", index) for index in MNIST_CALIBRATION]
            self.test_image = ("test", MNIST_TEST_IMAGE)
            self.evaluation = [("test", index) for index in MNIST_EVALUATION]
        else:
            self.cfg = model.pretrained_cfg
            self.calibration = sorted(glob.glob(os.path.join(IMAGENET_CALIBRATION, "*.JPEG")))
            heldout = [name for directory in IMAGENET_HELDOUT
                       for name in sorted(glob.glob(os.path.join(directory, "*.JPEG")))]
            if not self.calibration or not heldout:
                raise SystemExit("ImageNet images missing: run scripts/fetch_images.sh")
            self.test_image, self.evaluation = heldout[0], heldout
            self.synsets = timm.data.ImageNetInfo("imagenet-1k").label_names()

    def pixels(self, item):
        split, index = item
        return (self.train if split == "train" else self.test).data[index].numpy()

    def image(self, item):
        if self.board == "nexys":
            return torch.from_numpy(preprocess.image_tensor(self.pixels(item))).unsqueeze(0)
        return golden.load_image(item, self.cfg)

    def label(self, item):
        if self.board == "nexys":
            split, index = item
            return int((self.train if split == "train" else self.test).targets[index])
        return self.synsets.index(os.path.basename(item).split("_")[0])

    def name(self, item):
        return f"MNIST {item[0]} digit {item[1]}" if self.board == "nexys" else os.path.basename(item)


def embed_in(image):
    """The FP32 im2col matrix of an image tensor, row 0 zero: what the integer pass starts from."""
    return preprocess.im2col(image[0].numpy())


def calibrate(board, model, data):
    """Scale of every tensor: its largest magnitude over the calibration images / 127, for all 12 layers,
    and of the final LayerNorm and the model's head after each N (keys "head_in@N", "head_out@N")."""
    final_rows = "all" if board == "nexys" else "class"
    largest = {}

    def see(key, tensor):
        largest[key] = max(largest.get(key, 0.0), float(np.abs(tensor).max()))

    head_weight = model.head.weight.detach().numpy().astype(np.float64).T
    head_bias = model.head.bias.detach().numpy().astype(np.float64)
    for count, item in enumerate(data.calibration):
        fp = golden.capture(model, data.image(item), golden.MAX_LAYERS, final_rows)
        for key, tensor in fp.items():
            if key not in ("head_in", "head_out", "logits_full"):
                see(key, tensor)
        for layers in range(1, golden.MAX_LAYERS + 1):
            with torch.no_grad():
                head_in = model.norm(torch.from_numpy(fp[f"L{layers - 1}_x_out"]).float()).numpy().astype(np.float64)
            head_in = head_in if final_rows == "all" else head_in[0:1]
            see(f"head_in@{layers}", head_in)
            if board == "opal_kelly":
                see(f"head_out@{layers}", head_in @ head_weight + head_bias)
        progress("calibration images", count + 1, len(data.calibration))
    return {key: value / DATA_MAX for key, value in largest.items()}


def probe_file(layers):
    return os.path.join(CALIBRATION_DIR, f"nexys_L{layers}_probe.npz")


def features(parameters, data, items):
    """Final LayerNorm output of the integer pass for every item, as real numbers: (items, 50, 192)."""
    out = np.zeros((len(items), BOARDS["nexys"]["tokens"], 192))
    for position, item in enumerate(items):
        tensor, scale = golden.Forward(parameters).run(embed_in(data.image(item))).stage("final_ln")
        out[position] = tensor * scale
        progress("integer pass, digits", position + 1, len(items))
    return out


def train_probe(model, data, target, layers):
    """The digit classifier on the features of the integer N-layer pass: two linear layers 192 x 10, one
    on the mean of the 49 patch rows (applied by the engine to every row; the host sums the patch rows)
    and one on the class row. Trained with multinomial logistic regression."""
    parameters = golden.Model(model, target, layers, classifiers=[], final_all_rows=True)
    train_x = features(parameters, data, [("train", index) for index in MNIST_PROBE_TRAIN])
    train_y = data.train.targets[list(MNIST_PROBE_TRAIN)].numpy()

    def fit(x_real, y):
        mean, deviation = x_real.mean(0), x_real.std(0) + 1e-6
        x = torch.from_numpy((x_real - mean) / deviation)
        weight = torch.zeros(x.shape[1], 10, dtype=torch.float64, requires_grad=True)
        bias = torch.zeros(10, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=500, history_size=20,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(x @ weight + bias, torch.from_numpy(y)) + 1e-3 * (weight ** 2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        folded = weight.detach().numpy() / deviation[:, None]       # the standardisation folded in
        return folded, bias.detach().numpy() - mean @ folded

    mean_weight, mean_bias = fit(train_x[:, 1:].mean(1), train_y)
    class_weight, class_bias = fit(train_x[:, 0], train_y)
    # output scales: the largest score any row produces on the training digits
    mean_scale = float(np.abs(train_x @ mean_weight + mean_bias).max()) / DATA_MAX
    class_scale = float(np.abs(train_x[:, 0] @ class_weight + class_bias).max()) / DATA_MAX
    np.savez(probe_file(layers), mean_weight=mean_weight, mean_bias=mean_bias, class_weight=class_weight,
             class_bias=class_bias, mean_scale=mean_scale, class_scale=class_scale)
    print(f"  wrote {os.path.relpath(probe_file(layers), REPO_DIR)} (trained on {len(train_y)} digits)")


def load_probe(layers):
    """The digit classifier of N layers: classifiers for golden.Model and their output scales."""
    probe = np.load(probe_file(layers))
    classifiers = [("head_mean", probe["mean_weight"], probe["mean_bias"], "all", "head_mean_out", None),
                   ("head_class", probe["class_weight"], probe["class_bias"], "class", "head_class_out", None)]
    scales = {"head_mean_out": float(probe["mean_scale"]), "head_class_out": float(probe["class_scale"])}
    return classifiers, scales


def answer(board, forward):
    """The answer of a pass: nexys sums the 49 patch rows of head_mean, like the host."""
    if board == "nexys":
        return int(np.argmax(forward.stage("head_mean")[0][1:].sum(0)))
    return int(np.argmax(forward.stage("head1000")[0][0]))


def report(board, model, data, parameters, layers):
    """Accuracy of the integer pass on images the scales and the classifier never saw."""
    right = agree = 0
    for count, item in enumerate(data.evaluation):
        image = data.image(item)
        forward = golden.Forward(parameters).run(embed_in(image))
        right += answer(board, forward) == data.label(item)
        if board == "opal_kelly":
            fp = golden.capture(model, image, layers, "class")
            agree += answer(board, forward) == int(np.argmax(fp["head_out"]))
        progress("report images", count + 1, len(data.evaluation))
    count = len(data.evaluation)
    if board == "nexys":
        print(f"accuracy of the {layers}-layer integer model: {right} of {count} test digits "
              f"({100.0 * right / count:.1f}%)")
    else:
        print(f"{layers}-layer integer model: top-1 equal to the FP32 {layers}-layer model on {agree} of {count} "
              f"held-out photos, equal to the label on {right}")


def write_roms(board, model, scales, embed_in_real, forward):
    """The tables of all 12 layers into rtl/rom/<board>/; this pass's tables must be their first ones."""
    full = golden.Forward(golden.Model(model, golden.target_for(scales, golden.MAX_LAYERS), golden.MAX_LAYERS,
                                       classifiers=[])).run(embed_in_real)
    images, own = golden.rom_images(full), golden.rom_images(forward)
    for name in ("gelu_rom.hex", "exp_rom.hex"):
        used = np.nonzero(own[name][0])[0]
        assert np.array_equal(own[name][0][used], images[name][0][used]), name
    rom_dir = os.path.join(REPO_DIR, "rtl", "rom", board)
    os.makedirs(rom_dir, exist_ok=True)
    changed = []
    for name, (values, width) in images.items():
        path = os.path.join(rom_dir, name)
        old = open(path).read() if os.path.exists(path) else None
        write_hex(path, values, width)
        if open(path).read() != old:
            changed.append(name)
    if changed:
        print(f"ROM: {', '.join(changed)} in rtl/rom/{board} changed: build the bitstream again")
    else:
        print(f"ROM: rtl/rom/{board} already holds these tables, the bitstream needs no rebuild")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--board", choices=sorted(BOARDS), required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.layers <= golden.MAX_LAYERS:
        parser.error(f"--layers must be 1 .. {golden.MAX_LAYERS}")
    board, layers = args.board, args.layers
    deterministic.setup()

    model = load_model(board)
    data = Data(board, model)
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    scales_file = os.path.join(CALIBRATION_DIR, f"{board}_scales.json")
    if args.recalibrate or not os.path.exists(scales_file):
        golden.save_scales(scales_file, calibrate(board, model, data))
        print(f"calibrated: wrote {os.path.relpath(scales_file, REPO_DIR)}")
    scales = golden.load_scales(scales_file)
    target = golden.target_for(scales, layers)

    if board == "nexys":
        if args.retrain or args.recalibrate or not os.path.exists(probe_file(layers)):
            train_probe(model, data, target, layers)
        classifiers, probe_scales = load_probe(layers)
        target = dict(target, **probe_scales)
    else:
        classifiers = None
    parameters = golden.Model(model, target, layers, classifiers=classifiers)

    # the test image through the integer pass: the golden tensors and the program
    spec = BOARDS[board]
    out_dir = os.path.join(HOST_DIR, "models", f"{board}_L{layers}")
    embed_in_real = embed_in(data.image(data.test_image))
    forward = golden.Forward(parameters).run(embed_in_real)
    classifier_words = sum(layer_program.tiles(spec["tokens"] if rows == "all" else 1)
                           * parameters.linear[name][0].shape[1] for name, rows, _t, _f in parameters.classifiers)
    plan = layer_program.Plan(spec["tokens"], spec["weight_depth"], classifier_words)
    program = golden.dump(forward, data.name(data.test_image), out_dir,
                          matmul_dir=os.path.join(out_dir, "matmuls") if board == "opal_kelly" else None, plan=plan)
    print(layer_program.summary(program))
    for buffer_id, (used, depth) in enumerate(zip(program.used, spec["depths"])):
        if used > depth:
            raise SystemExit(f"does not fit the {board} bitstream: buffer {buffer_id} needs {used} words per bank, "
                             f"the board has {depth}")
    write_roms(board, model, scales, embed_in_real, forward)

    label = data.label(data.test_image)
    with open(os.path.join(out_dir, "host_config.json"), "w") as f:
        json.dump({"board": board, "layers": layers, "size": spec["size"], "tokens": spec["tokens"],
                   "input_scale": target["embed_in"], "rom_dir": f"rtl/rom/{board}", "depths": spec["depths"],
                   "test_image": data.name(data.test_image)}, f, indent=1)
    with open(os.path.join(out_dir, "expected_answer.txt"), "w") as f:
        f.write(f"{label} {answer(board, forward)}\n")
    print(f"wrote python/models/{board}_L{layers}: {layers} layers, test image {data.name(data.test_image)}, "
          f"label {label}, answer {answer(board, forward)}")
    if args.report:
        report(board, model, data, parameters, layers)


if __name__ == "__main__":
    main()
