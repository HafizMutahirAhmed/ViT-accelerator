"""Bit-exact integer models of the accelerator datapath.

Every function here does the arithmetic the RTL does, on integers only (no
floats), so its results can be compared bit-for-bit with simulation.

Fixed-point formats
  data  : signed INT8  (DATA_WIDTH  = 8)
  acc   : signed INT32 (ACC_WIDTH   = 32), wraps modulo 2**32 like the RTL adder
  bias  : signed INT32 (ACC_WIDTH   = 32), one per output column, at the same
          scale as acc; added to acc at drain, before the requantizer
  scale : signed INT8  (SCALE_WIDTH = 8), so a positive scale is 1..127
  shift : unsigned, SHIFT_WIDTH = $clog2(ACC_WIDTH) = 5 bits, 0..31
"""

import numpy as np

DATA_WIDTH = 8
ACC_WIDTH = 32
SCALE_WIDTH = 8
SHIFT_WIDTH = (ACC_WIDTH - 1).bit_length()  # $clog2(ACC_WIDTH)

DATA_MIN, DATA_MAX = -(1 << (DATA_WIDTH - 1)), (1 << (DATA_WIDTH - 1)) - 1
ACC_MIN, ACC_MAX = -(1 << (ACC_WIDTH - 1)), (1 << (ACC_WIDTH - 1)) - 1
SCALE_MIN, SCALE_MAX = -(1 << (SCALE_WIDTH - 1)), (1 << (SCALE_WIDTH - 1)) - 1
SHIFT_MAX = (1 << SHIFT_WIDTH) - 1


def _check_range(name, x, lo, hi):
    x = np.asarray(x, dtype=np.int64)
    if x.size and (x.min() < lo or x.max() > hi):
        raise ValueError(f"{name} out of range [{lo}, {hi}]: min={x.min()} max={x.max()}")
    return x


def wrap_signed(x, width):
    """Truncate to `width` bits and reinterpret as two's complement."""
    x = np.asarray(x, dtype=np.int64)
    half = 1 << (width - 1)
    return ((x + half) & ((1 << width) - 1)) - half


def matmul_acc(a, b):
    """C = A @ B with INT8 operands and an INT32 wrapping accumulator.

    a: (M, K) int8, b: (K, N) int8 -> (M, N) int32.
    Each PE computes acc <= acc + a*b over the full K dimension. Addition
    modulo 2**32 is associative, so the result does not depend on the order
    in which the array visits k, nor on how the matmul is tiled.
    """
    a = _check_range("a", a, DATA_MIN, DATA_MAX)
    b = _check_range("b", b, DATA_MIN, DATA_MAX)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"shape mismatch: {a.shape} @ {b.shape}")
    return wrap_signed(a @ b, ACC_WIDTH)


def add_bias(acc, bias):
    """acc[m][n] + bias[n], the add done at SA drain.

    acc: (M, N) int32, bias: (N,) int32 -> (M, N) int32. The adder is
    ACC_WIDTH bits wide, so the sum wraps modulo 2**32; it does not saturate.
    """
    acc = _check_range("acc", acc, ACC_MIN, ACC_MAX)
    bias = _check_range("bias", bias, ACC_MIN, ACC_MAX)
    if acc.ndim != 2 or bias.shape != (acc.shape[1],):
        raise ValueError(f"shape mismatch: acc {acc.shape}, bias {bias.shape}")
    return wrap_signed(acc + bias, ACC_WIDTH)


def linear_acc(a, b, bias):
    """Linear op before requantization: A @ B + bias[col], INT32."""
    return add_bias(matmul_acc(a, b), bias)


def requantize(acc, scale, shift):
    """out = clip((acc * scale) >>> shift) to INT8.

    The product is kept at full ACC_WIDTH + SCALE_WIDTH = 40 bits, so it never
    overflows. `>>>` is an arithmetic shift: it floors (rounds toward minus
    infinity) and adds no rounding offset. scale and shift may be scalars or
    arrays that broadcast against acc.
    """
    acc = _check_range("acc", acc, ACC_MIN, ACC_MAX)
    scale = _check_range("scale", scale, SCALE_MIN, SCALE_MAX)
    shift = _check_range("shift", shift, 0, SHIFT_MAX)
    prod = acc * scale
    shifted = np.right_shift(prod, shift)  # arithmetic on signed int64
    return np.clip(shifted, DATA_MIN, DATA_MAX)


def write_hex(path, values, width):
    """Write a $readmemh file: one two's-complement word per line, row-major."""
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{int(v) & mask:0{digits}x}\n")


def read_hex(path, width):
    """Read a file written by write_hex back into signed int64 values."""
    with open(path) as f:
        raw = np.array([int(line, 16) for line in f if line.strip()], dtype=np.int64)
    return wrap_signed(raw, width)
