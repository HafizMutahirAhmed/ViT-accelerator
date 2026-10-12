"""Image preprocessing shared by the golden model and the host: no torch needed.

A digit (any greyscale image) is resized to SIZE x SIZE with bilinear
filtering, copied to three channels, normalised with the model's mean and
standard deviation and cut into 16 x 16 patches. im2col() returns the matrix
the patch-embedding matmul reads: one row per token, row 0 all zero (the class
token's row), 768 = 3 * 16 * 16 values per patch in (channel, row, column) order.
"""

import numpy as np
from PIL import Image

SIZE = 112
PATCH = 16
MEAN = (0.485, 0.456, 0.406)     # the model's preprocessing (ImageNet statistics)
STD = (0.229, 0.224, 0.225)


def image_tensor(pixels, size=SIZE):
    """uint8 greyscale H x W -> float32 (3, size, size), normalised."""
    grey = Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="L").resize((size, size), Image.BILINEAR)
    plane = np.asarray(grey, dtype=np.float32) / 255.0
    return np.stack([(plane - mean) / std for mean, std in zip(MEAN, STD)]).astype(np.float32)


def im2col(tensor, patch=PATCH):
    """(3, S, S) -> (1 + (S / patch)^2, 3 * patch * patch), row 0 zero."""
    channels, size, _ = tensor.shape
    grid = size // patch
    patches = (tensor.astype(np.float64).reshape(channels, grid, patch, grid, patch).transpose(1, 3, 0, 2, 4)
               .reshape(grid * grid, channels * patch * patch))
    return np.vstack([np.zeros((1, patches.shape[1])), patches])


def load_png(file_name):
    """Any image file as uint8 greyscale."""
    return np.asarray(Image.open(file_name).convert("L"), dtype=np.uint8)
