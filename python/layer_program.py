"""The whole forward pass as one descriptor program for matmul_engine.

build() takes a finished Forward (golden_model.py) and lays its ops out on
the buffer pool: which buffer every tensor lives in, the descriptor of every
op, and what the host has to write before each phase. Nothing is copied
between ops: an op names the buffer an earlier op wrote.

A Plan fixes the token count and the depth of the weight buffer. With n
tokens, T = ceil(n / 8) row tiles and the three heads kept at once, the
buffers are (id, name, words per bank):
  0 X      residual stream n x 192, row banks                    T * 192
  1 W      weights of the running phase, column banks            weight_depth
  2 BIG    im2col input, attention output O, fc1 / GELU          T * 768
  3 N      LayerNorm output, projection output, fc2 output       T * 192
  4 S      scores and probabilities of the three heads; the      max(3 * T * n,
           classifier outputs                                        classifier words)
  5 QV     Q of the three heads (row banks), V (column banks)    3 * T * 64 + 3 * 8 * n
  6 K      K of the three heads, read as K^T; before that the    T * 192
           constant addend of the class-and-position add
  7        not used                                              16

Order: patch embedding and class-and-position add; per layer LN1; Q, K, V of
each head; S of each head; softmax of each head; O of each head; projection;
add; LN2; fc1; GELU; fc2; add; then the final LayerNorm and the classifiers.

Weights. A matmul's weights go into W behind the ones already there. When
they do not fit, a new phase starts: the descriptor before it gets last_op,
the host loads the next weights and starts again. A matmul whose weights are
larger than W on their own is split into column chunks of equal size, each a
matmul that writes its columns into the same output tensor. A new phase also
starts at the embedding, at each layer and at the final LayerNorm. With a
weight buffer as deep as one layer's weights there is one phase per layer.

Descriptor memory. The descriptors are numbered through the whole program
(the entry), and entry e lives in slot e mod DESCRIPTOR_COUNT: the sequencer's
index wraps the same way. A program of at most DESCRIPTOR_COUNT entries is
loaded once; a longer one phase by phase, before each START (the host and the
run list do that), which only needs every phase to fit.

LayerNorm constants. The LayerNorms of a section (a layer's two, or the final
one) are loaded with its first phase, behind the previous section's; when they
would run past the end of the constants memory they start again at 0.

Files written to the output directory:
  program_desc.hex     every descriptor, 5 words each, entry 0 first
  program_phases.txt   phase, first descriptor entry, number of ops
  program_ops.txt      entry, phase, name, kind, output buffer, layout (row or
                       col banks), base, stride, rows, cols, golden file
  program_loads.txt    what the host writes before a phase: phase, kind
                       (buffer, bias, layernorm), buffer id or 0, first
                       address, number of addresses, image file
  program_plan.txt     tokens, then the words per bank of buffers 0..7
  program_survivors.txt  the entries whose output is still whole when their phase
                       ends, which is what a host can read back and compare
  *_image.hex          one line per address. buffer: 8 bytes, bank 0 in the
                       low byte. bias: 8 INT32, bank 0 in the low word.
                       layernorm: offset (32 bits) above gain (16 bits).
  <op>_c<i>.hex        golden columns of chunk i of a split matmul
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
HEADS, HEAD_DIM, EMBED, HIDDEN = 3, 64, 192, 768


def tiles(count):
    return -(-count // LANES)


class Plan:
    """Token count and weight buffer depth; everything else follows."""

    def __init__(self, tokens=197, weight_depth=55296, classifier_words=1010):
        self.tokens, self.weight_depth = tokens, weight_depth
        self.row_tiles = tiles(tokens)
        self.q_words = self.row_tiles * HEAD_DIM          # one head's Q or K in row banks
        self.v_words = tiles(HEAD_DIM) * tokens           # one head's V in column banks
        self.s_words = self.row_tiles * tokens            # one head's scores
        self.v_base = HEADS * self.q_words
        self.depths = [self.row_tiles * EMBED,
                       weight_depth,
                       self.row_tiles * HIDDEN,
                       self.row_tiles * EMBED,
                       max(HEADS * self.s_words, classifier_words),
                       HEADS * self.q_words + HEADS * self.v_words,
                       max(HEADS * self.q_words, self.row_tiles * EMBED),
                       16]


class Program:
    def __init__(self, plan):
        self.plan = plan
        self.descriptors = []     # 5 words each
        self.ops = []             # check records, one per descriptor
        self.phases = []          # [name, first entry, count]
        self.loads = []           # (phase, kind, target, base, words, file)
        self.used = [0] * len(plan.depths)
        self.bias_used = 0
        self.files = {}           # file name -> (values, width)

    def use(self, buffer_id, end):
        self.used[buffer_id] = max(self.used[buffer_id], end)
        assert end <= self.plan.depths[buffer_id], (BUFFER_NAMES[buffer_id], end, self.plan.depths[buffer_id])


def row_image(tensor):
    """A rows x cols tensor in row banks: [address][bank]."""
    rows, cols = tensor.shape
    image = np.zeros((tiles(rows) * cols, LANES), dtype=np.int64)
    for row in range(rows):
        image[(row // LANES) * cols:(row // LANES) * cols + cols, row % LANES] = tensor[row]
    return image


def pack_lanes(image, width):
    """One Python int per address, bank 0 in the low bits."""
    mask = (1 << width) - 1
    return [sum((int(value) & mask) << (width * lane) for lane, value in enumerate(line)) for line in image]


def build(forward, stage_file, out_dir, plan=None):
    """stage_file: name of the golden file of a stage tensor, by id(tensor)."""
    plan = plan or Plan()
    program = Program(plan)
    by_name = {op["name"]: op for op in forward.matmuls + forward.row_ops}
    state = {"phase": None, "section": None, "index": 0, "first": 0, "weight_end": 0, "bias_end": 0,
             "weights": None, "biases": None, "layernorm_end": 0}
    layernorm_base = {}       # LayerNorm name -> first word of its gain and offset in the constants memory

    # --- phases ---------------------------------------------------------------
    def close_phase():
        """Ends the running phase: last_op on its last descriptor, its W and bias images."""
        if state["phase"] is None:
            return
        phase = state["phase"]
        last = list(program.descriptors[-1])
        last[0] |= 1                                    # last_op is bit 0
        program.descriptors[-1] = last
        program.phases.append((phase, state["first"], len(program.descriptors) - state["first"]))
        loads = []
        if state["weight_end"]:
            program.files[f"{phase}_weights_image.hex"] = (pack_lanes(state["weights"][:state["weight_end"]],
                                                                      DATA_WIDTH), DATA_WIDTH * LANES)
            loads.append((phase, "buffer", W, 0, state["weight_end"], f"{phase}_weights_image.hex"))
        if state["bias_end"]:
            program.files[f"{phase}_bias_image.hex"] = (pack_lanes(state["biases"][:state["bias_end"]], 32),
                                                        32 * LANES)
            loads.append((phase, "bias", 0, 0, state["bias_end"], f"{phase}_bias_image.hex"))
        # the weights first, then whatever else the phase asked for
        position = next((i for i, load in enumerate(program.loads) if load[0] == phase), len(program.loads))
        program.loads[position:position] = loads
        state["phase"] = None

    def open_phase(section=None):
        close_phase()
        if section is not None:
            state["section"], state["index"] = section, 0
        else:
            state["index"] += 1
        state["phase"] = state["section"] if state["index"] == 0 else f"{state['section']}_p{state['index']}"
        state["first"] = len(program.descriptors)
        state["weight_end"] = state["bias_end"] = 0
        state["weights"] = np.zeros((plan.weight_depth, LANES), dtype=np.int64)
        state["biases"] = np.zeros((PARAMS["BIAS_DEPTH"], LANES), dtype=np.int64)

    def place_bias(bias):
        """Bias words of one matmul (or chunk) in the bias store; returns the base."""
        base = state["bias_end"]
        if bias is not None:
            for col, value in enumerate(bias):
                state["biases"][base + col // LANES, col % LANES] = value
            state["bias_end"] = base + tiles(len(bias))
            program.bias_used = max(program.bias_used, state["bias_end"])
        return base

    def place_weights(weights):
        """Weights (K x cols) of one matmul or chunk in W, in a new phase if they do not fit. Returns the base."""
        inner, cols = weights.shape
        words = tiles(cols) * inner
        assert words <= plan.weight_depth
        if state["weight_end"] + words > plan.weight_depth:
            open_phase()
        base = state["weight_end"]
        for col in range(cols):
            start = base + (col // LANES) * inner
            state["weights"][start:start + inner, col % LANES] = weights[:, col]
        state["weight_end"] = base + words
        program.use(W, state["weight_end"])
        return base

    # --- ops ----------------------------------------------------------------------
    def emit_matmul(op, name, cols_from, cols, a_buffer, a_base, weight_buffer, weight_base, out_buffer, out_base,
                    bias_base, golden):
        rows, inner = op["a"].shape
        column_select = op["drain_mode"] == COLUMN_SELECT
        stride = op["output_stride"]
        program.use(a_buffer, a_base + tiles(rows) * inner)
        program.use(weight_buffer, weight_base + tiles(cols) * inner)
        # the last stored row ends at its own length, not at the stride
        program.use(out_buffer, out_base + ((tiles(rows) - 1) * stride + cols if column_select
                                            else (tiles(cols) - 1) * stride + rows))
        program.descriptors.append(pack_descriptor(
            opcode=PARAMS["OPCODE_MATMUL"], rows_total=rows, inner_dim=inner, cols_total=cols,
            row_tiles=tiles(rows), col_tiles=tiles(cols),
            activation_buffer_id=a_buffer, weight_buffer_id=weight_buffer, output_buffer_id=out_buffer,
            activation_base=a_base, weight_base=weight_base, output_base=out_base, output_stride=stride,
            bias_base=bias_base, scale=op["scale"], shift=op["shift"], drain_mode=op["drain_mode"],
            bias_enable=int(op["bias"] is not None), last_op=0))
        program.ops.append(dict(phase=state["phase"], name=name, kind="matmul", buffer=out_buffer,
                                layout="row" if column_select else "col", base=out_base, stride=stride,
                                rows=rows, cols=cols, file=golden))

    def linear(name, a_buffer, a_base, out_buffer, out_base=0, golden=None):
        """A matmul with weights from W, split into column chunks when its weights are larger than W."""
        op = by_name[name]
        inner, cols = op["b"].shape
        chunk_count = -(-(tiles(cols) * inner) // plan.weight_depth)
        if chunk_count == 1:
            weight_base = place_weights(op["b"])
            emit_matmul(op, name, 0, cols, a_buffer, a_base, W, weight_base, out_buffer, out_base,
                        place_bias(op["bias"]), golden or stage_file[id(op["y"])])
            return
        assert op["drain_mode"] == COLUMN_SELECT, name      # a chunk is a run of result columns
        chunk_tiles = -(-tiles(cols) // chunk_count)
        for chunk in range(chunk_count):
            start = chunk * chunk_tiles * LANES
            end = min(cols, start + chunk_tiles * LANES)
            weight_base = place_weights(op["b"][:, start:end])
            chunk_name = f"{name}_c{chunk}"
            program.files[f"{chunk_name}.hex"] = (op["y"][:, start:end], DATA_WIDTH)
            emit_matmul(op, chunk_name, start, end - start, a_buffer, a_base, W, weight_base, out_buffer,
                        out_base + start, place_bias(None if op["bias"] is None else op["bias"][start:end]),
                        f"{chunk_name}.hex")

    def pool_matmul(name, a_buffer, a_base, b_buffer, b_base, out_buffer, out_base, golden=None):
        """A matmul whose second operand is a tensor in the pool (S and O)."""
        op = by_name[name]
        emit_matmul(op, name, 0, op["b"].shape[1], a_buffer, a_base, b_buffer, b_base, out_buffer, out_base,
                    place_bias(op["bias"]), golden or stage_file[id(op["y"])])

    def row_op(name, a_buffer, a_base, out_buffer, out_base, b_buffer=W, b_base=0):
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
            bias_base=layernorm_base.get(name, op["bias_base"]), scale=op.get("scale", 0), shift=op.get("shift", 0),
            drain_mode=0, bias_enable=0, last_op=0))
        program.ops.append(dict(phase=state["phase"], name=name, kind=op["kind"], buffer=out_buffer, layout="row",
                                base=out_base, stride=length, rows=rows, cols=length,
                                file=stage_file[id(op["y"])]))

    def layernorm_load(names):
        """The gain and offset of a section's LayerNorms, loaded with its first phase."""
        words = []
        for name in names:
            constants = forward.constants[name]
            layernorm_base[name] = len(words)
            words += [((int(offset) & 0xFFFFFFFF) << 16) | (int(gain) & 0xFFFF)
                      for gain, offset in zip(constants["gain"], constants["offset"])]
        depth = PARAMS["LAYERNORM_CONSTANT_DEPTH"]
        assert len(words) <= depth
        base = state["layernorm_end"] if state["layernorm_end"] + len(words) <= depth else 0
        for name in names:
            layernorm_base[name] += base
        state["layernorm_end"] = base + len(words)
        file_name = f"{state['phase']}_layernorm_image.hex"
        program.files[file_name] = (words, 48)
        program.loads.append((state["phase"], "layernorm", 0, base, len(words), file_name))

    def buffer_load(name, buffer_id, tensor):
        image = row_image(tensor)
        program.use(buffer_id, len(image))
        program.files[f"{name}_image.hex"] = (pack_lanes(image, DATA_WIDTH), DATA_WIDTH * LANES)
        program.loads.append((state["phase"], "buffer", buffer_id, 0, len(image), f"{name}_image.hex"))

    # --- embed: im2col input in BIG, addend in K ---------------------------------
    open_phase("embed")
    buffer_load("embed_in", BIG, by_name["embed"]["a"])
    buffer_load("addend", K, by_name["position_add"]["b"])
    linear("embed", BIG, 0, N)
    row_op("position_add", N, 0, X, 0, b_buffer=K, b_base=0)

    # --- the encoder layers ------------------------------------------------------
    layers = [f"L{index}" for index in sorted({int(name.split("_")[0][1:]) for name in by_name
                                               if name[0] == "L" and name.split("_")[0][1:].isdigit()})]
    for layer in layers:
        open_phase(layer)
        layernorm_load([f"{layer}_ln1", f"{layer}_ln2"])
        row_op(f"{layer}_ln1", X, 0, N, 0)
        for head in range(HEADS):
            for letter in "qkv":
                out_buffer, out_base = {"q": (QV, head * plan.q_words), "k": (K, head * plan.q_words),
                                        "v": (QV, plan.v_base + head * plan.v_words)}[letter]
                linear(f"{layer}_h{head}_{letter}", N, 0, out_buffer, out_base)
        for head in range(HEADS):
            # S = Q @ K^T: K in row banks is K^T in column banks
            pool_matmul(f"{layer}_h{head}_s", QV, head * plan.q_words, K, head * plan.q_words,
                        S, head * plan.s_words)
        for head in range(HEADS):
            row_op(f"{layer}_h{head}_p", S, head * plan.s_words, S, head * plan.s_words)
        for head in range(HEADS):
            name = f"{layer}_h{head}_o"
            program.files[f"{name}.hex"] = (by_name[name]["y"], DATA_WIDTH)
            pool_matmul(name, S, head * plan.s_words, QV, plan.v_base + head * plan.v_words,
                        BIG, head * HEAD_DIM, golden=f"{name}.hex")
        linear(f"{layer}_proj", BIG, 0, N)
        row_op(f"{layer}_add1", X, 0, X, 0, b_buffer=N, b_base=0)
        row_op(f"{layer}_ln2", X, 0, N, 0)
        linear(f"{layer}_fc1", N, 0, BIG)
        row_op(f"{layer}_gelu", BIG, 0, BIG, 0)
        linear(f"{layer}_fc2", BIG, 0, N)
        row_op(f"{layer}_add2", X, 0, X, 0, b_buffer=N, b_base=0)

    # --- final LayerNorm, then the classifiers, their outputs back to back in S -------
    open_phase("final")
    layernorm_load(["final_ln"])
    row_op("final_ln", X, 0, N, 0)
    out_base = 0
    for op in forward.matmuls:
        if op["group"] == "head":
            linear(op["name"], N, 0, S, out_base)
            out_base += tiles(op["a"].shape[0]) * op["b"].shape[1]
    close_phase()

    assert all(count <= PARAMS["DESCRIPTOR_COUNT"] for _name, _first, count in program.phases)
    assert program.bias_used <= PARAMS["BIAS_DEPTH"]

    # --- files ------------------------------------------------------------------------
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
    # an op's output can be read after its phase if no later op of the phase wrote over it
    def addresses(op):
        outer, inner = (op["rows"], op["cols"]) if op["layout"] == "row" else (op["cols"], op["rows"])
        return {op["base"] + tile * op["stride"] + position for tile in range(tiles(outer))
                for position in range(inner)}

    with open(os.path.join(out_dir, "program_survivors.txt"), "w") as f:
        for entry, op in enumerate(program.ops):
            later = [other for other in program.ops[entry + 1:]
                     if other["phase"] == op["phase"] and other["buffer"] == op["buffer"]]
            written = addresses(op)
            if not any(written & addresses(other) for other in later):
                f.write(f"{entry}\n")
    with open(os.path.join(out_dir, "program_plan.txt"), "w") as f:
        f.write(f"{plan.tokens} " + " ".join(str(depth) for depth in plan.depths) + "\n")
    return program


def summary(program):
    plan = program.plan
    lines = [f"descriptor program: {plan.tokens} tokens, {len(program.descriptors)} entries "
             f"({'loaded once' if len(program.descriptors) <= PARAMS['DESCRIPTOR_COUNT'] else 'loaded per phase'}, "
             f"{PARAMS['DESCRIPTOR_COUNT']} slots) in {len(program.phases)} phases: "
             + ", ".join(f"{name} at {first} ({count})" for name, first, count in program.phases),
             f"bias store: {program.bias_used} of {PARAMS['BIAS_DEPTH']} words per bank used"]
    for buffer_id, name in enumerate(BUFFER_NAMES):
        lines.append(f"  buffer {buffer_id} {name:6s} {program.used[buffer_id]:6d} of {plan.depths[buffer_id]:6d} "
                     f"words per bank used")
    return "\n".join(lines)
