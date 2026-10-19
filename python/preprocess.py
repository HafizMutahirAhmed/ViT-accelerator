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


def mnist_style(pixels, box=20, side=28):
    """One digit in any greyscale picture -> side x side uint8 the way MNIST digits are made: a light
    digit on a black background (a picture with a light border is inverted), the background
    subtracted and the contrast stretched to 0..255, cropped to the digit, scaled so its longer side
    is `box` pixels, and placed so its centre of mass is the centre of the image."""
    grey = np.asarray(pixels, dtype=np.float64)
    border = np.concatenate([grey[0], grey[-1], grey[:, 0], grey[:, -1]])
    if np.median(border) > 127:              # a dark digit on a light background
        grey, border = 255 - grey, 255 - border
    grey = np.clip(grey - np.median(border), 0, None)
    if grey.max() <= 0:
        raise ValueError("no digit found: the picture is one flat colour")
    grey *= 255 / grey.max()
    rows, cols = np.nonzero(grey > 0.25 * 255)
    digit = grey[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    height, width = digit.shape
    scale = box / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    digit = np.asarray(Image.fromarray(digit.astype(np.uint8)).resize(size, Image.LANCZOS), dtype=np.float64)
    height, width = digit.shape
    mass = digit.sum()
    centre_row = (digit.sum(1) * np.arange(height)).sum() / mass
    centre_col = (digit.sum(0) * np.arange(width)).sum() / mass
    top = int(np.clip(round(side / 2 - 0.5 - centre_row), 0, side - height))
    left = int(np.clip(round(side / 2 - 0.5 - centre_col), 0, side - width))
    out = np.zeros((side, side))
    out[top:top + height, left:left + width] = digit
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
