"""The engine's arithmetic in Python, bit for bit: integer ops, the non-linear ops, the package parameters.

The RTL's arithmetic is defined here; golden_model.py, board_model.py and gen_vectors.py use it.
Run it (python3 python/int8_model.py) to print the descriptor layout as a Markdown table.

--- int8 ops: Bit-exact integer models of the accelerator datapath.

Every function here does the arithmetic the RTL does, on integers only (no
floats), so its results can be compared bit-for-bit with simulation.

Fixed-point formats
  data  : signed INT8  (DATA_WIDTH  = 8)
  acc   : signed INT32 (ACC_WIDTH   = 32), wraps modulo 2**32 like the RTL adder
  bias  : signed INT32 (ACC_WIDTH   = 32), one per output column, at the same
          scale as acc; added to acc at drain, before the requantizer
  scale : signed INT8  (SCALE_WIDTH = 8), so a positive scale is 1..127
  shift : unsigned, SHIFT_WIDTH = $clog2(ACC_WIDTH) = 5 bits, 0..31

--- nonlinear ops: Integer reference models of the non-linear ops: residual add, GELU, LayerNorm, softmax.

The op functions (residual_add, gelu, layernorm, softmax) work on integers
only, so RTL can be compared with them bit for bit. The table and constant
builders (pick_multiplier, add_constants, gelu_table, rsqrt_table,
layernorm_constants, exp_table, reciprocal_table) use floats: they are what the
host computes once and loads, like the weights.

Data is signed INT8 as everywhere else. Each op states its intermediate widths;
the functions check them, so a width that is too small raises an error instead
of silently wrapping.

--- pkg config: Reads rtl/core/vit_config_pkg.sv so the golden model and the RTL share one definition.

PARAMS holds every package parameter as an int. DESCRIPTOR_FIELDS lists the
fields of descriptor_t as (name, width, signed), most significant first, the
same order as the packed struct.
"""

import os
import re
from math import erf, exp, sqrt

import numpy as np


# ============================================================================
# int8 ops
# ============================================================================

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


# ============================================================================
# nonlinear ops
# ============================================================================

ROW_WIDTH = 192          # LayerNorm row length (the embedding width)


def _signed_fits(name, x, width):
    return _check_range(name, x, -(1 << (width - 1)), (1 << (width - 1)) - 1)


def _unsigned_fits(name, x, width):
    return _check_range(name, x, 0, (1 << width) - 1)


def pick_multiplier(ratio, multiplier_max=SCALE_MAX):
    """(scale, shift) with scale / 2**shift closest to ratio, scale <= multiplier_max.

    The largest shift that keeps the scale in range is used, which gives the
    scale the most significant bits.
    """
    assert ratio > 0
    for shift in range(SHIFT_MAX, -1, -1):
        scale = int(round(ratio * (1 << shift)))
        if 1 <= scale <= multiplier_max:
            return scale, shift
    raise ValueError(f"ratio {ratio} does not fit a multiplier of at most {multiplier_max} and a 5-bit shift")


# ---------------------------------------------------------------------------
# Residual add
# ---------------------------------------------------------------------------

def add_constants(scale_a, scale_b, scale_out, multiplier_max=SCALE_MAX):
    """Constants of out = (a * mult_a + b * mult_b) >> shift for real scales.

    Returns (mult_a, mult_b, shift, effective output scale, gain error of the
    less exact input). Both multipliers share one shift. The effective output
    scale is the one that makes the input with the larger multiplier exact.
    """
    ratio_a, ratio_b = scale_a / scale_out, scale_b / scale_out
    for shift in range(SHIFT_MAX, -1, -1):
        mult_a, mult_b = int(round(ratio_a * (1 << shift))), int(round(ratio_b * (1 << shift)))
        if max(mult_a, mult_b) <= multiplier_max and min(mult_a, mult_b) >= 1:
            break
    else:
        raise ValueError(f"add ratios {ratio_a}, {ratio_b} do not fit")
    if mult_a >= mult_b:
        effective = scale_a * (1 << shift) / mult_a
        gain_error = mult_b * effective / (scale_b * (1 << shift)) - 1.0
    else:
        effective = scale_b * (1 << shift) / mult_b
        gain_error = mult_a * effective / (scale_a * (1 << shift)) - 1.0
    return mult_a, mult_b, shift, effective, gain_error


def residual_add(a, b, mult_a, mult_b, shift, rounding=True, multiplier_width=8):
    """out = clip((a * mult_a + b * mult_b + round) >>> shift) to INT8.

    a, b: INT8. mult_a, mult_b: signed multiplier_width bits. The sum is
    DATA_WIDTH + multiplier_width + 1 bits signed (17 for 8-bit multipliers).
    round is 2**(shift-1) when rounding is on (0 for shift 0), else 0.
    """
    a = _check_range("a", a, DATA_MIN, DATA_MAX)
    b = _check_range("b", b, DATA_MIN, DATA_MAX)
    _signed_fits("mult_a", mult_a, multiplier_width)
    _signed_fits("mult_b", mult_b, multiplier_width)
    _check_range("shift", shift, 0, SHIFT_MAX)
    total = a * mult_a + b * mult_b
    _signed_fits("add sum", total, DATA_WIDTH + multiplier_width + 1)
    if rounding and shift > 0:
        total = total + (1 << (shift - 1))
    return np.clip(total >> shift, DATA_MIN, DATA_MAX)


# ---------------------------------------------------------------------------
# GELU
# ---------------------------------------------------------------------------

def gelu_real(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def gelu_table(scale_in, scale_out, input_width=DATA_WIDTH):
    """Table of 2**input_width INT8 entries, addressed by the input's two's complement bits.

    entry[x] = clip(round(gelu(x * scale_in) / scale_out)).
    """
    count = 1 << input_width
    table = np.zeros(count, dtype=np.int64)
    for address in range(count):
        x = address - count if address >= count // 2 else address
        table[address] = min(max(int(round(gelu_real(x * scale_in) / scale_out)), DATA_MIN), DATA_MAX)
    return table


def gelu(x, table):
    """out = table[x], the input taken as an unsigned address."""
    count = len(table)
    x = _check_range("gelu input", x, -(count // 2), count // 2 - 1)
    return table[x & (count - 1)]


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNormConfig:
    """Precision of the LayerNorm datapath.

    rsqrt        "lut": normalise the variance and look the reciprocal square
                 root up; "isqrt": integer square root, then one division.
    index_bits   lut: address width of the table (three quarters of it used)
    rsqrt_bits   lut: width of a table entry; isqrt: width of the reciprocal
    norm_frac    fraction bits of the normalised value (x - mean) / std
    gain_bits    width of the signed per-column gain (gamma / output scale)
    """

    def __init__(self, name, rsqrt="lut", index_bits=8, rsqrt_bits=12, norm_frac=8, gain_bits=8):
        self.name, self.rsqrt = name, rsqrt
        self.index_bits, self.rsqrt_bits = index_bits, rsqrt_bits
        self.norm_frac, self.gain_bits = norm_frac, gain_bits

    def table_bits(self):
        if self.rsqrt != "lut":
            return 0
        return (1 << self.index_bits) * (self.rsqrt_bits + 1)


VARIANCE_WIDTH = 30      # n * sum(x^2) - sum(x)^2 for n = 192 INT8 values is below 2**30
ISQRT_RECIP_SHIFT = 24   # isqrt: reciprocal = floor(2**24 / isqrt(variance))


def rsqrt_table(config):
    """entry[i] = round(2**rsqrt_bits / sqrt(m)), m = (i + 0.5) / 2**(index_bits - 2) in [1, 4).

    Entries are in (2**(rsqrt_bits - 1), 2**rsqrt_bits], so they need
    rsqrt_bits + 1 bits. Addresses below 2**(index_bits - 2) are never read.
    """
    count = 1 << config.index_bits
    table = np.zeros(count, dtype=np.int64)
    for index in range(count // 4, count):
        mantissa = (index + 0.5) / (count // 4)
        table[index] = int(round((1 << config.rsqrt_bits) / sqrt(mantissa)))
    return table


def layernorm_constants(gamma, beta, scale_in, scale_out, epsilon, config, row_width=ROW_WIDTH):
    """Per-column gain and offset, the output shift and the epsilon constant.

    out = (norm * gain[c] + offset[c]) >> out_shift, norm having norm_frac
    fraction bits. gain[c] = gamma[c] / scale_out * 2**(out_shift - norm_frac),
    offset[c] = beta[c] / scale_out * 2**out_shift plus half an output LSB,
    which makes the final shift round to nearest. epsilon is added to
    n * sum(x^2) - sum(x)^2, so it is epsilon * n^2 / scale_in^2.
    """
    gain_real = np.asarray(gamma, dtype=np.float64) / scale_out
    gain_max = (1 << (config.gain_bits - 1)) - 1
    out_shift = config.norm_frac + int(np.floor(np.log2(gain_max / np.abs(gain_real).max())))
    assert 0 <= out_shift <= SHIFT_MAX, out_shift
    gain = np.rint(gain_real * 2.0 ** (out_shift - config.norm_frac)).astype(np.int64)
    offset = np.rint(np.asarray(beta, dtype=np.float64) / scale_out * 2.0 ** out_shift).astype(np.int64)
    offset += (1 << (out_shift - 1)) if out_shift > 0 else 0
    epsilon_int = int(round(epsilon * row_width * row_width / (scale_in * scale_in)))
    return gain, offset, out_shift, epsilon_int


def integer_sqrt(value):
    """floor(sqrt(value)) by the bit-serial method, one result bit per step."""
    value = np.asarray(value, dtype=np.int64)
    root = np.zeros_like(value)
    remainder = value.copy()
    for bit in range(VARIANCE_WIDTH // 2, -1, -1):
        trial = (root << (bit + 1)) + (1 << (2 * bit))
        take = remainder >= trial
        remainder = np.where(take, remainder - trial, remainder)
        root = np.where(take, root + (1 << bit), root)
    return root


def layernorm(x, gain, offset, out_shift, epsilon_int, config, table=None, widths=None):
    """LayerNorm over each row of x (rows x n), INT8 in, INT8 out.

    Per row, with n the row length:
      total    = sum(x)                          signed 16 bits
      squares  = sum(x * x)                      unsigned 22 bits
      variance = n * squares - total * total     unsigned 30 bits, = n^2 * var(x)
                 + epsilon_int, at least 1
      centred  = n * x - total                   signed 17 bits, = n * (x - mean)
    so centred / sqrt(variance) = (x - mean) / std without any division by n.
      lut:   variance = m * 4**e with m in [1, 4); r = table[top index_bits of m]
             norm = (centred * r) >> (rsqrt_bits + e - norm_frac)
      isqrt: root = isqrt(variance); r = min(floor(2**24 / root), 2**rsqrt_bits - 1)
             norm = (centred * r) >> (24 - norm_frac)
      out[c] = clip((norm * gain[c] + offset[c]) >> out_shift)
    Both shifts are arithmetic (floor); the half LSB for the last one is in offset.
    widths, if given, is a dict that collects the largest magnitudes seen.
    """
    x = _check_range("x", x, DATA_MIN, DATA_MAX)
    row_width = x.shape[1]
    total = x.sum(axis=1, keepdims=True)
    squares = (x * x).sum(axis=1, keepdims=True)
    _signed_fits("total", total, 16)
    _unsigned_fits("squares", squares, 22)
    variance = np.maximum(row_width * squares - total * total + epsilon_int, 1)
    _unsigned_fits("variance", variance, VARIANCE_WIDTH)
    centred = row_width * x - total
    _signed_fits("centred", centred, 17)

    if config.rsqrt == "lut":
        # e = floor(log4(variance)); the normalised variance is in [2**28, 2**30)
        exponent = np.zeros_like(variance)
        for e in range(1, VARIANCE_WIDTH // 2):
            exponent = np.where(variance >= (1 << (2 * e)), e, exponent)
        normalised = variance << (VARIANCE_WIDTH - 2 - 2 * exponent)
        reciprocal = table[normalised >> (VARIANCE_WIDTH - config.index_bits)]
        norm = (centred * reciprocal) >> (config.rsqrt_bits + exponent - config.norm_frac)
    else:
        root = integer_sqrt(variance)
        reciprocal = np.minimum((1 << ISQRT_RECIP_SHIFT) // root, (1 << config.rsqrt_bits) - 1)
        norm = (centred * reciprocal) >> (ISQRT_RECIP_SHIFT - config.norm_frac)
    # |x - mean| / std is at most sqrt(n - 1), below 16 for n = 192
    _signed_fits("norm", norm, config.norm_frac + 5)
    scaled = norm * gain[None, :] + offset[None, :]
    if widths is not None:
        for key, value in (("reciprocal", reciprocal), ("product", centred * reciprocal), ("norm", norm),
                           ("offset", offset), ("scaled", scaled), ("variance", variance)):
            widths[key] = max(widths.get(key, 0), int(np.abs(value).max()))
    return np.clip(scaled >> out_shift, DATA_MIN, DATA_MAX)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

class SoftmaxConfig:
    """Precision of the softmax datapath.

    exp_bits     width of an exp table entry (unsigned)
    reciprocal   "divide": one exact integer division per row;
                 "lut": normalise the sum and look the reciprocal up
    index_bits   lut: address width of the reciprocal table (half of it used)
    gain_log2    the output is P * 127 * 2**gain_log2, clipped to 127
    calibrated   divide only: the output is P * numerator, clipped to 127, with
                 a numerator per head taken from the calibrated scale of P
    """

    def __init__(self, name, exp_bits=12, reciprocal="divide", index_bits=8, gain_log2=0, calibrated=False):
        self.name, self.exp_bits, self.reciprocal = name, exp_bits, reciprocal
        self.index_bits, self.gain_log2, self.calibrated = index_bits, gain_log2, calibrated

    def table_bits(self):
        bits = 256 * self.exp_bits
        if self.reciprocal == "lut":
            bits += (1 << self.index_bits) * (RECIPROCAL_BITS + 1)
        return bits


SUM_EXTRA_BITS = 8       # 197 table entries add up to less than 2**(exp_bits + 8)
QUOTIENT_FRAC = 10       # divide: quotient = floor(numerator * 2**(exp_bits + 10) / sum)
RECIPROCAL_BITS = 12     # lut: width of a reciprocal table entry


def exp_table(scale_in, config):
    """entry[d] = round((2**exp_bits - 1) * exp(-d * scale_in)) for d = 0 .. 255."""
    top = (1 << config.exp_bits) - 1
    return np.array([int(round(top * exp(-d * scale_in))) for d in range(256)], dtype=np.int64)


def reciprocal_table(config):
    """entry[i] = round(2**RECIPROCAL_BITS / m), m = (i + 0.5) / 2**(index_bits - 1) in [1, 2).

    Addresses below 2**(index_bits - 1) are never read.
    """
    count = 1 << config.index_bits
    table = np.zeros(count, dtype=np.int64)
    for index in range(count // 2, count):
        table[index] = int(round((1 << RECIPROCAL_BITS) / ((index + 0.5) / (count // 2))))
    return table


def softmax(scores, table, config, recip_table=None, widths=None, numerator=None):
    """Softmax over each row of scores (rows x n, n <= 197), INT8 in, 0..127 out.

    Per row:
      peak     = max(scores)
      distance = peak - scores[j]                0 .. 255, unsigned 8 bits
      e[j]     = table[distance]                 unsigned exp_bits
      total    = sum(e)                          unsigned exp_bits + 8
      divide: quotient = floor(numerator * 2**(exp_bits + 10) / total)
              out[j]   = min(127, (e[j] * quotient + half) >> (exp_bits + 10))
              numerator is 127 * 2**gain unless given; the output is P * numerator
      lut:    total = m * 2**k with m in [1, 2); r = recip_table[top index_bits of m]
              out[j] = min(127, (e[j] * r * 127 + half) >> (12 + k - gain))
    half is half of the last shift, so the result is rounded to nearest.
    """
    scores = _check_range("scores", scores, DATA_MIN, DATA_MAX)
    distance = scores.max(axis=1, keepdims=True) - scores
    _unsigned_fits("distance", distance, 8)
    entries = table[distance]
    total = entries.sum(axis=1, keepdims=True)
    sum_width = config.exp_bits + SUM_EXTRA_BITS
    _unsigned_fits("total", total, sum_width)

    if config.reciprocal == "divide":
        shift = config.exp_bits + QUOTIENT_FRAC
        if numerator is None:
            numerator = DATA_MAX << config.gain_log2
        quotient = (numerator << shift) // total
        product = entries * quotient
        out = (product + (1 << (shift - 1))) >> shift
    else:
        position = np.zeros_like(total)
        for k in range(1, sum_width):
            position = np.where(total >= (1 << k), k, position)
        normalised = total << (sum_width - 1 - position)
        quotient = recip_table[normalised >> (sum_width - config.index_bits)]
        product = entries * quotient * DATA_MAX
        shift = RECIPROCAL_BITS + position - config.gain_log2
        out = (product + (1 << (shift - 1))) >> shift
    if widths is not None:
        for key, value in (("quotient", quotient), ("product", product), ("total", total)):
            widths[key] = max(widths.get(key, 0), int(np.abs(value).max()))
    return np.minimum(out, DATA_MAX)


# ============================================================================
# pkg config
# ============================================================================

PACKAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "rtl", "core", "vit_config_pkg.sv")


def _clog2(value):
    return (value - 1).bit_length()


def _evaluate(expr, params, struct_widths):
    expr = re.sub(r"\$bits\((\w+)\)", lambda m: str(struct_widths[m.group(1)]), expr)
    expr = re.sub(r"\d+'([bdh])([0-9a-fA-F]+)",
                  lambda m: str(int(m.group(2), {"b": 2, "d": 10, "h": 16}[m.group(1)])), expr)
    expr = expr.replace("$clog2", "clog2").replace("/", "//")
    return int(eval(expr, {"__builtins__": {}}, {**params, "clog2": _clog2}))


def _load():
    with open(PACKAGE_FILE) as f:
        text = re.sub(r"//[^\n]*", "", f.read())
    params, structs, struct_widths = {}, {}, {}
    statement = re.compile(
        r"parameter\s+(?P<name>\w+)\s*=\s*(?P<expr>[^;]+);"
        r"|typedef\s+struct\s+packed\s*\{(?P<body>.*?)\}\s*(?P<struct>\w+)\s*;", re.S)
    field = re.compile(r"logic\s*(signed)?\s*(?:\[([^\]]+):0\])?\s*(\w+)\s*;")
    for m in statement.finditer(text):
        if m.group("name"):
            params[m.group("name")] = _evaluate(m.group("expr"), params, struct_widths)
        else:
            fields = [(name, _evaluate(msb, params, struct_widths) + 1 if msb else 1, bool(signed))
                      for signed, msb, name in field.findall(m.group("body"))]
            structs[m.group("struct")] = fields
            struct_widths[m.group("struct")] = sum(width for _, width, _ in fields)
    return params, structs


PARAMS, _STRUCTS = _load()
DESCRIPTOR_FIELDS = _STRUCTS["descriptor_t"]

assert PARAMS["DESCRIPTOR_WIDTH"] == sum(width for _, width, _ in DESCRIPTOR_FIELDS)


def pack_descriptor(**values):
    """Pack one descriptor into DESCRIPTOR_WORDS 32-bit words, word 0 least significant."""
    names = [name for name, _, _ in DESCRIPTOR_FIELDS]
    if set(values) != set(names):
        raise ValueError(f"descriptor fields differ from the package: {sorted(set(values) ^ set(names))}")
    packed = 0
    for name, width, signed in DESCRIPTOR_FIELDS:
        value = int(values[name])
        lo, hi = (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)
        if not lo <= value <= hi:
            raise ValueError(f"descriptor field {name}={value} does not fit {width} bits")
        packed = (packed << width) | (value & ((1 << width) - 1))
    return [(packed >> (32 * i)) & 0xFFFFFFFF for i in range(PARAMS["DESCRIPTOR_WORDS"])]


def unpack_descriptor(words):
    """The inverse of pack_descriptor: 32-bit words, word 0 least significant, to a dict."""
    packed = sum(int(word) << (32 * i) for i, word in enumerate(words))
    values, position = {}, PARAMS["DESCRIPTOR_WIDTH"]
    for name, width, signed in DESCRIPTOR_FIELDS:
        position -= width
        value = (packed >> position) & ((1 << width) - 1)
        if signed and value >> (width - 1):
            value -= 1 << width
        values[name] = value
    return values


# what each descriptor field means, for the table in docs/ENGINE.md
DESCRIPTOR_FIELD_NOTES = {
    "opcode": "Operation. 0 = MATMUL, 1 = ADD, 2 = GELU, 3 = LAYERNORM, 4 = SOFTMAX; any other value is skipped.",
    "rows_total": "M, rows of the activation operand and of the result.",
    "inner_dim": "K, the dimension summed over. At least 1.",
    "cols_total": "N, columns of the weight operand and of the result.",
    "row_tiles": "ceil(M / SYS_ROW), worked out by the host.",
    "col_tiles": "ceil(N / SYS_COL), worked out by the host.",
    "activation_buffer_id": "Pool buffer read as the activation operand (row banks).",
    "weight_buffer_id": "Pool buffer read as the weight operand (column banks).",
    "output_buffer_id": "Pool buffer the result is written to.",
    "activation_base": "Address of the activation operand's first word in each bank.",
    "weight_base": "Address of the weight operand's first word in each bank.",
    "output_base": "Address of the result's first word in each bank.",
    "output_stride": "Words per stored row of the result. N for column-select and M for "
                     "row-select, or more to place the result inside a wider tensor.",
    "bias_base": "Address in the bias store of the bias words for output columns 0 .. SYS_COL-1.",
    "scale": "Requantizer multiplier, signed.",
    "shift": "Requantizer right shift, arithmetic.",
    "drain_mode": "0 = row-select (result in column banks), 1 = column-select (result in row banks).",
    "bias_enable": "1 = add the bias before requantizing.",
    "last_op": "1 = stop after this op and pulse done.",
}


def descriptor_markdown():
    """The descriptor_t layout as a Markdown table, most significant field first."""
    lines = ["| Field | Bits | Width | Meaning |", "|---|---|---|---|"]
    position = PARAMS["DESCRIPTOR_WIDTH"]
    for name, width, signed in DESCRIPTOR_FIELDS:
        position -= width
        bits = f"{position + width - 1}:{position}" if width > 1 else f"{position}"
        kind = f"{width}, signed" if signed else f"{width}"
        lines.append(f"| `{name}` | {bits} | {kind} | {DESCRIPTOR_FIELD_NOTES[name]} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(descriptor_markdown())
    print(f"\n{PARAMS['DESCRIPTOR_WIDTH']} bits, stored in {PARAMS['DESCRIPTOR_WORDS']} 32-bit words "
          f"({PARAMS['DESCRIPTOR_MEMORY_WIDTH']} bits).")
