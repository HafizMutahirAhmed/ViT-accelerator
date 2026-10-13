"""Golden vectors of the 112 x 112 MNIST configuration (50 tokens).

Usage: .venv/bin/python python/gen_mnist_forward.py [--dump-only]   (needs torch, timm, torchvision)
Before it: .venv/bin/python python/train_probe.py, which writes python/probe_mnist112.npz.
Output: python/vectors/mnist112/ (every stage, the descriptor program, the host
load images) and rtl/rom/mnist112/ (the ROM images of this configuration).

The integer model is gen_deit_forward.py's, unchanged; mnist_config.py holds
what differs: the input size, the images, and the two classifiers (a probe on
every row, summed over the patch rows by the host, and a probe on the class
row), which make the final LayerNorm cover every row. The descriptor program
has a 16,384-word weight buffer, so weights are loaded per group of matmuls
and fc1, fc2 and the patch embedding are split into two column chunks.

The report compares every stage with FP32 on the calibration digits
(in-sample) and on test digits (held-out).
"""

import json
import os
import sys

import numpy as np

import gen_deit_forward as golden
import layer_program
import mnist_config as config


def main():
    model = config.load_model()
    train_set, test_set = config.datasets()
    target = config.calibrate(model, train_set, golden.capture_all)
    classifiers, scales = config.load_probe()
    parameters = golden.Model(model, dict(target, **scales), classifiers=classifiers)

    def capture(dataset, index):
        fp = config.capture_digit(model, config.pixels_of(dataset, index), golden.capture_all)
        config.add_classifier_outputs(fp, classifiers)
        return fp

    print(f"model {golden.MODEL_NAME} at {config.preprocess.SIZE}x{config.preprocess.SIZE}, {config.TOKENS} tokens; "
          f"scales from {len(config.CALIBRATION)} MNIST train digits (max)")
    if "--dump-only" not in sys.argv:
        tables = []
        for title, dataset, indices in (("in-sample", train_set, config.CALIBRATION),
                                        ("held-out", test_set, config.COSINE_EVALUATION)):
            cosines, errors = {}, []
            for index in indices:
                fp = capture(dataset, index)
                forward = golden.Forward(parameters, golden.INTEGER_OPS).run(fp)
                for name, tensor, scale, key in forward.stages:
                    if key is not None:
                        cosines.setdefault(name, []).append(golden.cosine(tensor, fp[key]))
                    if name == "L1_x_out":
                        errors.append(float(np.abs(tensor - fp[key] / scale).mean()))
            tables.append(cosines)
            print(f"{title:9s} ({len(indices):3d} digits): layer 2 output cosine mean "
                  f"{np.mean(cosines['L1_x_out']):.5f} worst {min(cosines['L1_x_out']):.5f}, mean error "
                  f"{np.mean(errors):.3f} LSB; final LayerNorm {np.mean(cosines['final_ln']):.5f}; scores "
                  f"{np.mean(cosines['head_mean']):.5f} (all rows), {np.mean(cosines['head_class']):.5f} (class row)")
        print("\nAll integer: cosine against FP32 at every stage, mean / worst")
        print(f"  {'stage':12s} {'in-sample':>20s} {'held-out':>20s}")
        for stage in tables[0]:
            print(f"  {stage:12s} " + "  ".join(f"{np.mean(t[stage]):10.5f} {min(t[stage]):9.5f}" for t in tables))

    fp = capture(test_set, config.TEST_IMAGE)
    classifier_words = sum(layer_program.tiles(config.TOKENS if rows == "all" else 1) * weight.shape[1]
                           for _name, weight, _bias, rows, _t, _f in classifiers)
    plan = layer_program.Plan(config.TOKENS, config.WEIGHT_DEPTH, classifier_words)
    forward = golden.dump(parameters, fp, f"MNIST test digit {config.TEST_IMAGE}", forward_dir=config.VEC_DIR,
                          matmul_dir=False, rom_dir=config.ROM_DIR, plan=plan)
    scores = {name: tensor for name, tensor, _s, _k in forward.stages}
    label = int(test_set.targets[config.TEST_IMAGE])
    print(f"test digit {config.TEST_IMAGE}: label {label}, mean-of-patch-rows answer "
          f"{int(np.argmax(scores['head_mean'][1:].sum(0)))}, class-row answer {int(np.argmax(scores['head_class'][0]))}")
    # what python/run_image.py needs besides the program files
    with open(os.path.join(config.VEC_DIR, "host_config.json"), "w") as f:
        json.dump({"size": config.preprocess.SIZE, "tokens": config.TOKENS,
                   "input_scale": parameters.target["embed_in"],
                   "rom_dir": os.path.relpath(config.ROM_DIR, config.REPO_DIR),
                   "test_digit": config.TEST_IMAGE}, f, indent=1)
    with open(os.path.join(config.VEC_DIR, "expected_answer.txt"), "w") as f:
        f.write(f"{label} {int(np.argmax(scores['head_mean'][1:].sum(0)))} {int(np.argmax(scores['head_class'][0]))}\n")


if __name__ == "__main__":
    main()
