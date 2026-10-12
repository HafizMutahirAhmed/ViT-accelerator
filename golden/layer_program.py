"""The whole forward pass as one descriptor program for matmul_engine.

build() takes a finished Forward (gen_deit_forward.py) and lays its ops out on
the buffer pool: which buffer every tensor lives in, the descriptor of every
op, and what the host has to write before each phase. Nothing is copied
between ops: an op names the buffer an earlier op wrote.

Buffers (id, words per bank at 8x8):
  0 X      residual stream 197 x 192, row banks                         4,800
  1 W      weights of the running phase, column banks                  55,296
  2 BIG    im2col input, attention output O, fc1 output / GELU         19,200
  3 N      LayerNorm output, then projection output, then fc2 output    4,800
  4 S      scores and probabilities of the three heads; the addend     14,775
           of the position add; the logits
  5 QV     Q of the three heads (row banks), then V (column banks)      9,528
  6 K      K of the three heads, read as K^T                            4,800
  7        not used                                                        16

Phases, each a run of descriptors ending with last_op, started by the host
after it has loaded the phase's weights, biases and LayerNorm constants:
  embed    patch embedding (im2col matmul), class-and-position add       2 ops
  L0, L1   one encoder layer each                                       26 ops
  final    final LayerNorm on the class row, head10, head1000            3 ops

Layer order: LN1; Q, K, V of each head; S of each head; softmax of each head;
O of each head; projection; add; LN2; fc1; GELU; fc2; add.

Files written to golden/vectors/deit_forward:
  program_desc.hex     every descriptor, 5 words each, entry 0 first
  program_phases.txt   phase, first descriptor entry, number of ops
  program_ops.txt      entry, phase, name, kind, output buffer, layout (row or
                       col banks), base, stride, rows, cols, golden file
  program_loads.txt    what the host writes before a phase: phase, kind
                       (buffer, bias, layernorm), buffer id or 0, first
                       address, number of addresses, image file
  *_image.hex          one line per address. buffer: 8 bytes, bank 0 in the
                       low byte. bias: 8 INT32, bank 0 in the low word.
                       layernorm: offset (32 bits) above gain (16 bits).
"""

import os

import numpy as np

from int8_ops import DATA_WIDTH, write_hex
from pkg_config import PARAMS, pack_descriptor

LANES = 8
ROW_SELECT, COLUMN_SELECT = PARAMS["DRAIN_ROW_SELECT"], PARAMS["DRAIN_COLUMN_SELECT"]
ROW_OPCODES = {"add": 1, "gelu": 2, "layernorm": 3, "softmax": 4}

X, W, BIG, N, S, QV, K = range(7)
BUFFER_NAMES = ["X", "W", "BIG", "N", "S", "QV", "K", "unused"]
BUFFER_DEPTHS = [4800, 55296, 19200, 4800, 14775, 9528, 4800, 16]

TILES = 25                       # ceil(197 / 8)
Q_WORDS = TILES * 64             # one head's Q or K in row banks
V_WORDS = 8 * 197                # one head's V in column banks
S_WORDS = TILES * 197            # one head's scores
V_BASE = 3 * Q_WORDS


def tiles(count):
    return -(-count // LANES)


class Program:
    def __init__(self):
        self.descriptors = []     # 5 words each
        self.ops = []             # check records, one per descriptor
        self.phases = []          # (name, first entry, count)
        self.loads = []           # (phase, kind, target, base, words, file)
        self.used = [0] * len(BUFFER_DEPTHS)
        self.bias_used = 0
        self.files = {}           # file name -> (array, width)

    def use(self, buffer_id, end):
        self.used[buffer_id] = max(self.used[buffer_id], end)
        assert end <= BUFFER_DEPTHS[buffer_id], (BUFFER_NAMES[buffer_id], end)


def row_image(tensor, stride=None):
    """A rows x cols tensor in row banks: [address][bank]."""
    rows, cols = tensor.shape
    stride = cols if stride is None else stride
    image = np.zeros((tiles(rows) * stride, LANES), dtype=np.int64)
    for row in range(rows):
        image[(row // LANES) * stride:(row // LANES) * stride + cols, row % LANES] = tensor[row]
    return image


def pack_lanes(image, width):
    """One Python int per address, bank 0 in the low bits."""
    mask = (1 << width) - 1
    return [sum((int(value) & mask) << (width * lane) for lane, value in enumerate(line)) for line in image]


def build(forward, stage_file, out_dir):
    """stage_file: name of the golden file of a stage tensor, by id(tensor)."""
    program = Program()
    by_name = {op["name"]: op for op in forward.matmuls + forward.row_ops}

    def matmul(phase, name, a_buffer, a_base, weight_buffer, weight_base, out_buffer, out_base=0,
               bias_base=0, golden=None):
        op = by_name[name]
        rows, inner = op["a"].shape
        cols = op["b"].shape[1]
        column_select = op["drain_mode"] == COLUMN_SELECT
        stride = op["output_stride"]
        program.use(a_buffer, a_base + tiles(rows) * inner)
        program.use(weight_buffer, weight_base + tiles(cols) * inner)
        program.use(out_buffer, out_base + (tiles(rows) if column_select else tiles(cols)) * stride)
        program.bias_used = max(program.bias_used, bias_base + tiles(cols))
        program.descriptors.append(pack_descriptor(
            opcode=PARAMS["OPCODE_MATMUL"], rows_total=rows, inner_dim=inner, cols_total=cols,
            row_tiles=tiles(rows), col_tiles=tiles(cols),
            activation_buffer_id=a_buffer, weight_buffer_id=weight_buffer, output_buffer_id=out_buffer,
            activation_base=a_base, weight_base=weight_base, output_base=out_base, output_stride=stride,
            bias_base=bias_base, scale=op["scale"], shift=op["shift"], drain_mode=op["drain_mode"],
            bias_enable=int(op["bias"] is not None), last_op=0))
        program.ops.append(dict(phase=phase, name=name, kind="matmul", buffer=out_buffer,
                                layout="row" if column_select else "col", base=out_base, stride=stride,
                                rows=rows, cols=cols, file=golden or stage_file[id(op["y"])],
                                shape=(rows, inner, cols), column_select=column_select))
        return op

    def row_op(phase, name, a_buffer, a_base, out_buffer, out_base, b_buffer=W, b_base=0):
        op = by_name[name]
        rows, length = op["a"].shape
        program.use(a_buffer, a_base + tiles(rows) * length)
        program.use(out_buffer, out_base + tiles(rows) * length)
        if "b" in op:
            program.use(b_buffer, b_base + tiles(rows) * length)
        program.descriptors.append(pack_descriptor(
            opcode=ROW_OPCODES[op["kind"]], rows_total=rows, inner_dim=op.get("inner_dim", 0), cols_total=length,
            row_tiles=tiles(rows), col_tiles=0,
            activation_buffer_id=a_buffer, weight_buffer_id=b_buffer, output_buffer_id=out_buffer,
            activation_base=a_base, weight_base=b_base, output_base=out_base, output_stride=length,
            bias_base=op["bias_base"], scale=op.get("scale", 0), shift=op.get("shift", 0),
            drain_mode=0, bias_enable=0, last_op=0))
        program.ops.append(dict(phase=phase, name=name, kind=op["kind"], buffer=out_buffer, layout="row",
                                base=out_base, stride=length, rows=rows, cols=length,
                                file=stage_file[id(op["y"])], shape=(rows, length)))
        return op

    def weight_images(phase, placed, depth_bias):
        """placed: (op, weight base in W or None, bias base). Builds the phase's W and bias images."""
        weight_end = max([base + tiles(op["b"].shape[1]) * op["b"].shape[0] for op, base, _ in placed
                          if base is not None])
        weights = np.zeros((weight_end, LANES), dtype=np.int64)
        biases = np.zeros((depth_bias, LANES), dtype=np.int64)
        for op, base, bias_base in placed:
            inner, cols = op["b"].shape
            for col in range(cols):
                if base is not None:
                    start = base + (col // LANES) * inner
                    weights[start:start + inner, col % LANES] = op["b"][:, col]
                if op["bias"] is not None:
                    biases[bias_base + col // LANES, col % LANES] = op["bias"][col]
        program.files[f"{phase}_weights_image.hex"] = (pack_lanes(weights, DATA_WIDTH), DATA_WIDTH * LANES)
        program.files[f"{phase}_bias_image.hex"] = (pack_lanes(biases, 32), 32 * LANES)
        program.loads.append((phase, "buffer", W, 0, len(weights), f"{phase}_weights_image.hex"))
        program.loads.append((phase, "bias", 0, 0, len(biases), f"{phase}_bias_image.hex"))

    def layernorm_image(phase, names):
        base = min(by_name[name]["bias_base"] for name in names)
        words = []
        for name in names:
            assert by_name[name]["bias_base"] == base + len(words)
            constants = forward.constants[name]
            words += [((int(offset) & 0xFFFFFFFF) << 16) | (int(gain) & 0xFFFF)
                      for gain, offset in zip(constants["gain"], constants["offset"])]
        program.files[f"{phase}_layernorm_image.hex"] = (words, 48)
        program.loads.append((phase, "layernorm", 0, base, len(words), f"{phase}_layernorm_image.hex"))

    def buffer_image(phase, name, buffer_id, tensor):
        image = row_image(tensor)
        program.files[f"{name}_image.hex"] = (pack_lanes(image, DATA_WIDTH), DATA_WIDTH * LANES)
        program.loads.append((phase, "buffer", buffer_id, 0, len(image), f"{name}_image.hex"))

    def end_phase(phase, first):
        last = list(program.descriptors[-1])
        last[0] |= 1                                    # last_op is bit 0
        program.descriptors[-1] = last
        program.phases.append((phase, first, len(program.descriptors) - first))

    # --- embed: im2col input in BIG, addend in S ---------------------------
    first = len(program.descriptors)
    embed = matmul("embed", "embed", BIG, 0, W, 0, N)
    weight_images("embed", [(embed, 0, 0)], tiles(embed["b"].shape[1]))
    buffer_image("embed", "embed_in", BIG, embed["a"])
    buffer_image("embed", "addend", S, by_name["position_add"]["b"])
    row_op("embed", "position_add", N, 0, X, 0, b_buffer=S, b_base=0)
    end_phase("embed", first)

    # --- the encoder layers ------------------------------------------------
    layers = sorted({name[:2] for name in by_name if name.startswith("L")})
    for layer in layers:
        first = len(program.descriptors)
        placed = []
        row_op(layer, f"{layer}_ln1", X, 0, N, 0)
        for head in range(3):
            for part, letter in enumerate("qkv"):
                tile = part * 24 + head * 8             # column tile of this slice in the 192 x 576 qkv weight
                out_buffer, out_base = {"q": (QV, head * Q_WORDS), "k": (K, head * Q_WORDS),
                                        "v": (QV, V_BASE + head * V_WORDS)}[letter]
                op = matmul(layer, f"{layer}_h{head}_{letter}", N, 0, W, tile * 192, out_buffer, out_base, tile)
                placed.append((op, tile * 192, tile))
        for head in range(3):
            # S = Q @ K^T: K in row banks is K^T in column banks
            op = matmul(layer, f"{layer}_h{head}_s", QV, head * Q_WORDS, K, head * Q_WORDS, S, head * S_WORDS,
                        216 + head * TILES)
            placed.append((op, None, 216 + head * TILES))
        for head in range(3):
            row_op(layer, f"{layer}_h{head}_p", S, head * S_WORDS, S, head * S_WORDS)
        for head in range(3):
            name = f"{layer}_h{head}_o"
            program.files[f"{name}.hex"] = (by_name[name]["y"], DATA_WIDTH)
            op = matmul(layer, name, S, head * S_WORDS, QV, V_BASE + head * V_WORDS, BIG, head * 64,
                        291 + head * 8, golden=f"{name}.hex")
            placed.append((op, None, 291 + head * 8))
        placed.append((matmul(layer, f"{layer}_proj", BIG, 0, W, 13824, N, 0, 72), 13824, 72))
        row_op(layer, f"{layer}_add1", X, 0, X, 0, b_buffer=N, b_base=0)
        row_op(layer, f"{layer}_ln2", X, 0, N, 0)
        placed.append((matmul(layer, f"{layer}_fc1", N, 0, W, 18432, BIG, 0, 96), 18432, 96))
        row_op(layer, f"{layer}_gelu", BIG, 0, BIG, 0)
        placed.append((matmul(layer, f"{layer}_fc2", BIG, 0, W, 36864, N, 0, 192), 36864, 192))
        row_op(layer, f"{layer}_add2", X, 0, X, 0, b_buffer=N, b_base=0)
        weight_images(layer, placed, 315)
        layernorm_image(layer, [f"{layer}_ln1", f"{layer}_ln2"])
        end_phase(layer, first)

    # --- final LayerNorm on the class row, then the classifiers -------------
    first = len(program.descriptors)
    row_op("final", "final_ln", X, 0, N, 0)
    head1000 = matmul("final", "head1000", N, 0, W, 0, S, 0, 0)
    head10 = matmul("final", "head10", N, 0, W, 24000, S, 1000, 125)
    weight_images("final", [(head1000, 0, 0), (head10, 24000, 125)], 127)
    layernorm_image("final", ["final_ln"])
    end_phase("final", first)

    assert len(program.descriptors) <= PARAMS["DESCRIPTOR_COUNT"]
    assert program.bias_used <= PARAMS["BIAS_DEPTH"]

    # --- files ----------------------------------------------------------------
    write_hex(os.path.join(out_dir, "program_desc.hex"), np.array(program.descriptors, dtype=np.int64), 32)
    for name, (values, width) in program.files.items():
        if isinstance(values, np.ndarray):
            write_hex(os.path.join(out_dir, name), values, width)
        else:
            digits = (width + 3) // 4
            with open(os.path.join(out_dir, name), "w") as f:
                f.write("".join(f"{value:0{digits}x}\n" for value in values))
    with open(os.path.join(out_dir, "program_phases.txt"), "w") as f:
        for phase in program.phases:
            f.write("{} {} {}\n".format(*phase))
    with open(os.path.join(out_dir, "program_ops.txt"), "w") as f:
        for entry, op in enumerate(program.ops):
            f.write(f"{entry} {op['phase']} {op['name']} {op['kind']} {op['buffer']} {op['layout']} {op['base']} "
                    f"{op['stride']} {op['rows']} {op['cols']} {op['file']}\n")
    with open(os.path.join(out_dir, "program_loads.txt"), "w") as f:
        for load in program.loads:
            f.write("{} {} {} {} {} {}\n".format(*load))
    return program


def summary(program):
    lines = [f"descriptor program: {len(program.descriptors)} of {PARAMS['DESCRIPTOR_COUNT']} entries; phases "
             + ", ".join(f"{name} at {first} ({count} ops)" for name, first, count in program.phases),
             f"bias store: {program.bias_used} of {PARAMS['BIAS_DEPTH']} words per bank used"]
    for buffer_id, name in enumerate(BUFFER_NAMES):
        lines.append(f"  buffer {buffer_id} {name:6s} {program.used[buffer_id]:6d} of {BUFFER_DEPTHS[buffer_id]:6d} "
                     f"words per bank used")
    return "\n".join(lines)
