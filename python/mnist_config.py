"""The 112 x 112 MNIST configuration of the golden model (for the Arty A7 bring-up).

The same pretrained DeiT-Tiny, layers 0 and 1, on a 112 x 112 input: 49 patches
and the class token, 50 tokens, with the position embedding resampled by timm
to the 7 x 7 grid. A 28 x 28 MNIST digit is resized by 4. The classifier is a
linear probe trained by python/train_probe.py.

Images: torchvision's MNIST, downloaded to python/mnist/ (not in the repo).
  calibration   train images 0..127: the activation scales (largest value / 127)
  probe         train images 128..6127: the linear probe is trained on these
  evaluation    test images 0..1999: accuracy; the first 64 also for cosine
  test image    test image 0: the vectors
"""

import os
import sys

import numpy as np
import timm
import torch
import torchvision

from gen_deit_vectors import MODEL_NAME
from int8_ops import DATA_MAX

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(GOLDEN_DIR)
sys.path.insert(0, GOLDEN_DIR)
import preprocess  # noqa: E402

NAME = "mnist112"
TOKENS = 50
WEIGHT_DEPTH = 16384                       # words per bank of the weight buffer
MNIST_DIR = os.path.join(GOLDEN_DIR, "mnist")
VEC_DIR = os.path.join(GOLDEN_DIR, "vectors", NAME)
ROM_DIR = os.path.join(REPO_DIR, "rtl", "rom", NAME)
PROBE_FILE = os.path.join(GOLDEN_DIR, f"probe_{NAME}.npz")
SCALES_FILE = os.path.join(GOLDEN_DIR, f"scales_{NAME}.json")

CALIBRATION = range(0, 128)
PROBE_TRAIN = range(128, 6128)
EVALUATION = range(0, 2000)
COSINE_EVALUATION = range(0, 64)
TEST_IMAGE = 0


def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=True, img_size=preprocess.SIZE).eval()
    for block in model.blocks:
        block.attn.fused_attn = False  # so the softmax output passes through attn_drop
    cfg = model.pretrained_cfg
    assert tuple(cfg["mean"]) == preprocess.MEAN and tuple(cfg["std"]) == preprocess.STD
    assert model.pos_embed.shape[1] == TOKENS
    return model


def datasets():
    """(train, test) MNIST; .data is uint8 n x 28 x 28, .targets the digits."""
    return (torchvision.datasets.MNIST(MNIST_DIR, train=True, download=True),
            torchvision.datasets.MNIST(MNIST_DIR, train=False, download=True))


def pixels_of(dataset, index):
    return dataset.data[index].numpy()


def capture_digit(model, pixels, capture_all):
    """FP32 capture of one digit, with the final LayerNorm on every row."""
    image = torch.from_numpy(preprocess.image_tensor(pixels)).unsqueeze(0)
    fp = capture_all(model, image)
    assert np.allclose(fp["embed_in"], preprocess.im2col(image[0].numpy()), atol=1e-6)
    with torch.no_grad():
        block_out = torch.from_numpy(fp["L1_x_out"]).float()
        fp["head_in"] = model.norm(block_out).numpy().astype(np.float64)
    return fp


def integer_input(pixels):
    """What the integer pass needs of a digit: the im2col matrix."""
    return {"embed_in": preprocess.im2col(preprocess.image_tensor(pixels))}


def calibrate(model, train, capture_all, load_scales, recalibrate=False):
    """Scale of every tensor: its largest magnitude over the calibration digits / 127.

    The result is kept in the repo (SCALES_FILE) and read from there unless recalibrate is set.
    load_scales is gen_deit_forward.load_scales.
    """
    def work_out():
        largest = {}
        for index in CALIBRATION:
            for key, tensor in capture_digit(model, pixels_of(train, index), capture_all).items():
                largest[key] = max(largest.get(key, 0.0), float(np.abs(tensor).max()))
        return {key: value / DATA_MAX for key, value in largest.items()}

    return load_scales(SCALES_FILE, recalibrate, work_out)


def load_probe():
    """The trained probe: classifiers for Model() and their output scales."""
    if not os.path.exists(PROBE_FILE):
        raise SystemExit(f"{PROBE_FILE} is missing: run .venv/bin/python python/train_probe.py")
    probe = np.load(PROBE_FILE)
    classifiers = [("head_mean", probe["mean_weight"], probe["mean_bias"], "all", "head_mean_out", "head_mean_out"),
                   ("head_class", probe["class_weight"], probe["class_bias"], "class", "head_class_out",
                    "head_class_out")]
    scales = {"head_mean_out": float(probe["mean_scale"]), "head_class_out": float(probe["class_scale"])}
    return classifiers, scales


def add_classifier_outputs(fp, classifiers):
    """FP32 outputs of the classifiers, for the cosine report."""
    for name, weight, bias, rows, _target, fp_key in classifiers:
        features = fp["head_in"] if rows == "all" else fp["head_in"][0:1]
        fp[fp_key] = features @ weight + bias
