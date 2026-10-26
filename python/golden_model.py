"""Golden model: the first N encoder layers of DeiT-Tiny in the accelerator's integer arithmetic.

Used by python/make_model.py, which picks the board (image size) and the number
of layers. Needs torch and timm (import python/deterministic.py first).

The pass is the real pretrained model: patch embedding, class token and
position add, then per layer LayerNorm, Q/K/V, S, softmax, O, projection,
residual add, LayerNorm, fc1, GELU, fc2, residual add, and at the end the final
LayerNorm and the classifiers. Each stage reads the INT8 tensor the stage
before it wrote.

Scales. Every tensor has one scale: the largest magnitude seen over the
calibration images, divided by 127. A matmul's requantizer multiplier (8-bit
scale, 5-bit shift) cannot hit a wanted ratio exactly, so the pass carries
each tensor's effective scale, the one the integers really have, and builds
biases, tables and the next multiplier from that. The three heads of O share
one tensor and keep the calibrated scale. A layer's scales do not depend on
the layers after it, so one calibration of all 12 layers serves every N; only
the final LayerNorm depends on N (key "head_in@N"); the classifier is trained
for each N (python/make_model.py).

Rounding. The engine requantizer floors. Half an output LSB is added to every
bias word, so the result is rounded to nearest; S and O, which have no bias,
get a constant one for this.

Class token and position. Row 0 of the patch-embedding input is zero, so row 0
of its output is the requantized bias. One residual add then adds a constant
tokens x 192 tensor: rows 1.. hold the position embedding and row 0 holds
class token + position 0 - that bias row.

Tables. GELU has one 256-entry table per layer, softmax one exp table per head
and layer, LayerNorm one reciprocal-square-root table; they are the engine's
ROMs (rtl/rom/input_<size>/), made by write_roms() from a 12-layer pass so that
every N finds its tables there.
"""

import glob
import json
import os

import numpy as np
import torch

import layer_program
from int8_model import ACC_MAX, ACC_WIDTH, DATA_MAX, DATA_MIN, DATA_WIDTH, add_bias, requantize, write_hex
from int8_model import (LayerNormConfig, SoftmaxConfig, add_constants, exp_table, gelu, gelu_table, layernorm,
                           layernorm_constants, pick_multiplier, residual_add, rsqrt_table, softmax)
from int8_model import PARAMS, pack_descriptor

MODEL_NAME = "deit_tiny_patch16_224.fb_in1k"
MAX_LAYERS = 12
SYS_ROW = SYS_COL = 8
BANK_DEPTH = PARAMS["BANK_DEPTH_MAX"]  # tb_matmul_engine_opal_kelly gives every pool buffer this depth
ROW_SELECT, COLUMN_SELECT = PARAMS["DRAIN_ROW_SELECT"], PARAMS["DRAIN_COLUMN_SELECT"]

# precision of the non-linear ops
LAYERNORM = LayerNormConfig("lut256/r12/n8/g16", "lut", index_bits=8, rsqrt_bits=12, norm_frac=8, gain_bits=16)
SOFTMAX = SoftmaxConfig("exp12/divide", exp_bits=12)
ADD_MULTIPLIER_WIDTH = 8
GAIN_WIDTH, OFFSET_WIDTH = 16, 32   # file widths of the LayerNorm constants

ROW_OPCODES = {"add": 1, "gelu": 2, "layernorm": 3, "softmax": 4}
for _kind, _opcode in ROW_OPCODES.items():
    assert PARAMS[f"OPCODE_{_kind.upper()}"] == _opcode, _kind


def quantize(tensor, scale):
    return np.clip(np.rint(tensor / scale), DATA_MIN, DATA_MAX).astype(np.int64)


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def capture(model, image, layers):
    """Run one image and return the FP32 tensors of every stage of the first `layers` layers.

    Tensors: embed_in/_out (row 0 zero / the bias), L<l>_qkv_in, L<l>_h<h>_q, _k, _v, _s, _p, _o,
    L<l>_proj, _fc1, _fc2 _in, _out, L<l>_x_in, _x_mid, _x_out, and head_in (the final LayerNorm on
    every row)."""
    grabbed = {}

    def grab(name):
        def hook(_module, inputs, output):
            grabbed[name + "_in"] = inputs[0].detach()[0].numpy().astype(np.float64)
            grabbed[name + "_out"] = output.detach()[0].numpy().astype(np.float64)
        return hook

    hooks = [model.patch_embed.proj.register_forward_hook(grab("conv"))]
    for layer in range(layers):
        block = model.blocks[layer]
        hooks += [block.attn.qkv.register_forward_hook(grab(f"L{layer}_qkv")),
                  block.attn.attn_drop.register_forward_hook(grab(f"L{layer}_softmax")),
                  block.attn.proj.register_forward_hook(grab(f"L{layer}_proj")),
                  block.mlp.fc1.register_forward_hook(grab(f"L{layer}_fc1")),
                  block.mlp.fc2.register_forward_hook(grab(f"L{layer}_fc2")),
                  block.norm2.register_forward_hook(grab(f"L{layer}_norm2")),
                  block.register_forward_hook(grab(f"L{layer}_block"))]
    with torch.no_grad():
        model(image)
        block_out = torch.from_numpy(grabbed[f"L{layers - 1}_block_out"]).float()
        head_in = model.norm(block_out).numpy().astype(np.float64)
    for hook in hooks:
        hook.remove()

    embed, heads, head_dim = model.embed_dim, model.blocks[0].attn.num_heads, model.blocks[0].attn.head_dim
    fp = {"head_in": head_in}

    # patch embedding as a matmul: host im2col, element order (channel, row, column), row 0 zero
    patch = model.patch_embed.patch_size[0]
    grid = image.shape[-1] // patch
    pixels = image[0].numpy().astype(np.float64)
    patches = pixels.reshape(3, grid, patch, grid, patch).transpose(1, 3, 0, 2, 4).reshape(grid * grid, -1)
    bias = model.patch_embed.proj.bias.detach().numpy().astype(np.float64)
    fp["embed_in"] = np.vstack([np.zeros((1, patches.shape[1])), patches])
    fp["embed_out"] = np.vstack([bias[None, :], grabbed["conv_out"].reshape(embed, grid * grid).T])
    weight = model.patch_embed.proj.weight.detach().numpy().astype(np.float64).reshape(embed, -1)
    assert np.allclose(fp["embed_in"] @ weight.T + bias, fp["embed_out"], atol=1e-3)

    for layer in range(layers):
        qkv = grabbed[f"L{layer}_qkv_out"]
        fp[f"L{layer}_qkv_in"] = grabbed[f"L{layer}_qkv_in"]
        for head in range(heads):
            q, k, v = (qkv[:, part * embed + head * head_dim:part * embed + (head + 1) * head_dim]
                       for part in range(3))
            h = f"L{layer}_h{head}"
            fp[f"{h}_q"], fp[f"{h}_k"], fp[f"{h}_v"] = q, k, v
            fp[f"{h}_s"] = (q @ k.T) * model.blocks[layer].attn.scale
            fp[f"{h}_p"] = grabbed[f"L{layer}_softmax_in"][head]
            fp[f"{h}_o"] = grabbed[f"L{layer}_proj_in"][:, head * head_dim:(head + 1) * head_dim]
            # the captured tensors must be what the matmul view of the model computes
            softmax_ = np.exp(fp[f"{h}_s"] - fp[f"{h}_s"].max(axis=1, keepdims=True))
            assert np.allclose(softmax_ / softmax_.sum(axis=1, keepdims=True), fp[f"{h}_p"], atol=1e-4)
            assert np.allclose(fp[f"{h}_p"] @ v, fp[f"{h}_o"], atol=1e-3)
        for name in ("proj", "fc1", "fc2"):
            fp[f"L{layer}_{name}_in"] = grabbed[f"L{layer}_{name}_in"]
            fp[f"L{layer}_{name}_out"] = grabbed[f"L{layer}_{name}_out"]
        fp[f"L{layer}_x_in"] = grabbed[f"L{layer}_block_in"]
        fp[f"L{layer}_x_mid"] = grabbed[f"L{layer}_norm2_in"]
        fp[f"L{layer}_x_out"] = grabbed[f"L{layer}_block_out"]
    return fp


class Model:
    """Quantized weights, FP32 parameters and scales of the first `layers` layers: what does not depend
    on the image."""

    def __init__(self, model, target, layers, classifiers, final_all_rows=None):
        """classifiers: (name, weight 192 x N, bias, rows, target key, FP32 key or None) per classifier head,
        rows being "class" (the class row only) or "all"; the digit classifiers of make_model.py."""
        assert 1 <= layers <= MAX_LAYERS
        self.target = target
        self.layers = layers
        self.embed = model.embed_dim
        self.heads = model.blocks[0].attn.num_heads
        self.head_dim = model.blocks[0].attn.head_dim
        self.attention_scale = float(model.blocks[0].attn.scale)
        self.linear = {}
        self.norm = {}

        def numpy(parameter):
            return parameter.detach().numpy().astype(np.float64)

        def add_linear(name, weight, bias):
            weight_scale = np.abs(weight).max() / DATA_MAX
            w_q = np.clip(np.rint(weight / weight_scale), DATA_MIN, DATA_MAX).astype(np.int64)
            self.linear[name] = (w_q, weight_scale, bias)

        add_linear("embed", numpy(model.patch_embed.proj.weight).reshape(self.embed, -1).T,
                   numpy(model.patch_embed.proj.bias))
        for layer in range(layers):
            block = model.blocks[layer]
            qkv_weight, qkv_bias = numpy(block.attn.qkv.weight).T, numpy(block.attn.qkv.bias)
            for head in range(self.heads):
                for part, letter in enumerate("qkv"):
                    col = part * self.embed + head * self.head_dim
                    add_linear(f"L{layer}_h{head}_{letter}", qkv_weight[:, col:col + self.head_dim],
                               qkv_bias[col:col + self.head_dim])
            for name, linear in (("proj", block.attn.proj), ("fc1", block.mlp.fc1), ("fc2", block.mlp.fc2)):
                add_linear(f"L{layer}_{name}", numpy(linear.weight).T, numpy(linear.bias))
            for name, norm in (("ln1", block.norm1), ("ln2", block.norm2)):
                self.norm[f"L{layer}_{name}"] = (numpy(norm.weight), numpy(norm.bias), norm.eps)
        self.norm["final_ln"] = (numpy(model.norm.weight), numpy(model.norm.bias), model.norm.eps)
        self.classifiers = []
        for name, weight, bias, rows, target_key, fp_key in classifiers:
            add_linear(name, np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64))
            self.classifiers.append((name, rows, target_key, fp_key))
        # the final LayerNorm covers every row as soon as one classifier reads every row
        self.final_all_rows = any(rows == "all" for _name, rows, _t, _f in self.classifiers)
        if final_all_rows is not None:
            self.final_all_rows = final_all_rows

        position = numpy(model.pos_embed)[0]
        self.embed_bias = numpy(model.patch_embed.proj.bias)
        self.class_row = numpy(model.cls_token)[0, 0] + position[0]
        self.position = position[1:]

    def addend_real(self, bias_row=None):
        """The constant of the class-and-position add. bias_row: what row 0 of the embedding output holds."""
        bias_row = self.embed_bias if bias_row is None else bias_row
        return np.vstack([(self.class_row - bias_row)[None, :], self.position])


def matmul(a, b):
    """int8_model.matmul_acc, but through float64 for speed. Exact: every sum is far below 2**53."""
    acc = np.rint(a.astype(np.float64) @ b.astype(np.float64)).astype(np.int64)
    assert np.abs(acc).max() <= ACC_MAX
    return acc


class Forward:
    """One image through the integer model: the arithmetic of the engine, bit for bit."""

    def __init__(self, parameters):
        self.m, self.t = parameters, parameters.target
        self.stages = []      # (name, INT8 tensor, scale, FP32 key or None)
        self.constants = {}   # non-linear op name -> dict of integer constants
        self.matmuls = []     # one dict per engine op, in order
        self.row_ops = []     # one dict per non-linear op, in order

    def keep(self, name, tensor, scale, key=None):
        self.stages.append((name, tensor, scale, key))
        return tensor, scale

    def engine_op(self, name, group, a_name, a, b, acc_scale, target_key, bias_real=None,
                  drain_mode=COLUMN_SELECT, output_base=0, output_stride=None, fixed_scale=False):
        """One matmul_engine op: acc = a @ b + bias, out = clip((acc * scale) >>> shift)."""
        scale, shift = pick_multiplier(acc_scale / self.t[target_key])
        bias = np.zeros(b.shape[1], dtype=np.int64)
        if bias_real is not None:
            bias = np.rint(bias_real / acc_scale).astype(np.int64)
        if shift > 0:
            bias = bias + int(round((1 << (shift - 1)) / scale))   # half an output LSB
        acc = add_bias(matmul(a, b), bias)
        y = requantize(acc, scale, shift)
        effective = acc_scale * (1 << shift) / scale
        natural = b.shape[1] if drain_mode == COLUMN_SELECT else a.shape[0]
        self.matmuls.append(dict(name=name, group=group, a_name=a_name, a=a, b=b, bias=bias, y=y, scale=scale,
                                 shift=shift, drain_mode=drain_mode, output_base=output_base,
                                 output_stride=natural if output_stride is None else output_stride))
        return y, (self.t[target_key] if fixed_scale else effective)

    def linear(self, name, group, a_name, a, a_scale, target_key, **layout):
        w_q, weight_scale, bias = self.m.linear[name]
        return self.engine_op(name, group, a_name, a, w_q, a_scale * weight_scale, target_key, bias, **layout)

    def add(self, name, a, a_scale, b, b_scale, target_key):
        limit = (1 << (ADD_MULTIPLIER_WIDTH - 1)) - 1
        mult_a, mult_b, shift, effective, _gain_error = add_constants(a_scale, b_scale, self.t[target_key], limit)
        self.constants[name] = dict(mult_a=mult_a, mult_b=mult_b, shift=shift)
        y = residual_add(a, b, mult_a, mult_b, shift, True, ADD_MULTIPLIER_WIDTH)
        self.row_ops.append(dict(name=name, kind="add", a=a, b=b, y=y, scale=mult_a, bias_base=mult_b & 0xFF,
                                 shift=shift))
        return y, effective

    def layernorm(self, name, x, x_scale, target_key):
        gamma, beta, epsilon = self.m.norm[name]
        scale = self.t[target_key]
        gain, offset, out_shift, epsilon_int = layernorm_constants(gamma, beta, x_scale, scale, epsilon, LAYERNORM)
        self.constants[name] = dict(gain=gain, offset=offset, out_shift=out_shift, epsilon_int=epsilon_int)
        y = layernorm(x, gain, offset, out_shift, epsilon_int, LAYERNORM, rsqrt_table(LAYERNORM))
        # bias_base: where this LayerNorm's gain and offset are in the constants memory in the per-op vectors
        # (the unit testbenches load one op at a time): one slot per LayerNorm, as many as fit, then from 0
        # again; layer_program.py places them per section for the program
        slots = PARAMS["LAYERNORM_CONSTANT_DEPTH"] // x.shape[1]
        index = sum(op["kind"] == "layernorm" for op in self.row_ops)
        self.row_ops.append(dict(name=name, kind="layernorm", a=x, y=y, shift=out_shift, inner_dim=epsilon_int,
                                 bias_base=(index % slots) * x.shape[1],
                                 files=(f"{name}_gain.hex", f"{name}_offset.hex")))
        return y, scale

    def softmax(self, name, scores, score_scale):
        table = exp_table(score_scale, SOFTMAX)
        self.constants[name] = dict(exp_table=table)
        y = softmax(scores, table, SOFTMAX)
        # bias_base: which exp table, one per head and layer
        self.row_ops.append(dict(name=name, kind="softmax", a=scores, y=y,
                                 bias_base=sum(op["kind"] == "softmax" for op in self.row_ops)))
        return y, 1.0 / DATA_MAX

    def gelu(self, name, x, x_scale, target_key):
        scale = self.t[target_key]
        table = gelu_table(x_scale, scale)
        self.constants[name] = dict(table=table)
        y = gelu(x, table)
        # bias_base: which table, one per layer
        self.row_ops.append(dict(name=name, kind="gelu", a=x, y=y,
                                 bias_base=sum(op["kind"] == "gelu" for op in self.row_ops)))
        return y, scale

    def run(self, embed_in_real):
        """embed_in_real: the FP32 im2col matrix, row 0 zero."""
        m, keep = self.m, self.keep
        embed_in, embed_in_scale = keep("embed_in", quantize(embed_in_real, self.t["embed_in"]), self.t["embed_in"])
        patches, patch_scale = keep("embed_out", *self.linear("embed", "embed", "embed_in", embed_in, embed_in_scale,
                                                              "embed_out"), "embed_out")
        # row 0 of patches is the requantized bias; the addend's row 0 makes it class token + position 0
        addend_real = m.addend_real(patches[0] * patch_scale)
        addend_scale = np.abs(addend_real).max() / DATA_MAX
        addend, _ = keep("addend", quantize(addend_real, addend_scale), addend_scale)
        x, x_scale = keep("L0_x_in", *self.add("position_add", patches, patch_scale, addend, addend_scale, "L0_x_in"),
                          "L0_x_in")

        for layer in range(m.layers):
            p = f"L{layer}"
            n1, n1_scale = keep(f"{p}_ln1", *self.layernorm(f"{p}_ln1", x, x_scale, f"{p}_qkv_in"), f"{p}_qkv_in")
            heads = []
            for head in range(m.heads):
                h = f"{p}_h{head}"
                q = keep(f"{h}_q", *self.linear(f"{h}_q", p, f"{p}_ln1", n1, n1_scale, f"{h}_q"), f"{h}_q")
                k = keep(f"{h}_k", *self.linear(f"{h}_k", p, f"{p}_ln1", n1, n1_scale, f"{h}_k"), f"{h}_k")
                v = keep(f"{h}_v", *self.linear(f"{h}_v", p, f"{p}_ln1", n1, n1_scale, f"{h}_v",
                                                drain_mode=ROW_SELECT), f"{h}_v")
                heads.append((h, q, k, v))
            probabilities = []
            for h, (q, q_scale), (k, k_scale), _v in heads:
                # S = Q @ K^T; the 1/sqrt(head_dim) is part of the requantizer multiplier
                s, s_scale = keep(f"{h}_s", *self.engine_op(f"{h}_s", p, f"{h}_q", q, k.T.copy(),
                                                            q_scale * k_scale * m.attention_scale, f"{h}_s"), f"{h}_s")
                probabilities.append(keep(f"{h}_p", *self.softmax(f"{h}_p", s, s_scale), f"{h}_p"))
            attention = np.zeros((x.shape[0], m.embed), dtype=np.int64)
            for head, ((h, _q, _k, (v, v_scale)), (prob, prob_scale)) in enumerate(zip(heads, probabilities)):
                # O = P @ V lands in its slot of the tokens x 192 tensor
                o, _ = self.engine_op(f"{h}_o", p, f"{h}_p", prob, v, prob_scale * v_scale, f"{p}_proj_in",
                                      output_base=head * m.head_dim, output_stride=m.embed, fixed_scale=True)
                attention[:, head * m.head_dim:(head + 1) * m.head_dim] = o
            attention, attention_scale = keep(f"{p}_o", attention, self.t[f"{p}_proj_in"], f"{p}_proj_in")
            proj, proj_scale = keep(f"{p}_proj", *self.linear(f"{p}_proj", p, f"{p}_o", attention, attention_scale,
                                                              f"{p}_proj_out"), f"{p}_proj_out")
            x, x_scale = keep(f"{p}_x_mid", *self.add(f"{p}_add1", x, x_scale, proj, proj_scale, f"{p}_x_mid"),
                              f"{p}_x_mid")
            n2, n2_scale = keep(f"{p}_ln2", *self.layernorm(f"{p}_ln2", x, x_scale, f"{p}_fc1_in"), f"{p}_fc1_in")
            fc1, fc1_scale = keep(f"{p}_fc1", *self.linear(f"{p}_fc1", p, f"{p}_ln2", n2, n2_scale, f"{p}_fc1_out"),
                                  f"{p}_fc1_out")
            act, act_scale = keep(f"{p}_gelu", *self.gelu(f"{p}_gelu", fc1, fc1_scale, f"{p}_fc2_in"), f"{p}_fc2_in")
            fc2, fc2_scale = keep(f"{p}_fc2", *self.linear(f"{p}_fc2", p, f"{p}_gelu", act, act_scale, f"{p}_fc2_out"),
                                  f"{p}_fc2_out")
            x, x_scale = keep(f"{p}_x_out", *self.add(f"{p}_add2", x, x_scale, fc2, fc2_scale, f"{p}_x_out"),
                              f"{p}_x_out")

        # the final LayerNorm, then the classifiers
        final_in = x if m.final_all_rows else x[0:1]
        final, final_scale = keep("final_ln", *self.layernorm("final_ln", final_in, x_scale, "head_in"), "head_in")
        for name, rows, target_key, fp_key in m.classifiers:
            if rows == "all" or not m.final_all_rows:
                a, a_name = final, "final_ln"
            else:
                a, a_name = keep("final_ln_class", final[0:1].copy(), final_scale)[0], "final_ln_class"
            keep(name, *self.linear(name, "head", a_name, a, final_scale, target_key), fp_key)
        return self

    def stage(self, name):
        """(INT8 tensor, scale) of a stage."""
        return next((tensor, scale) for stage, tensor, scale, _key in self.stages if stage == name)


def dump(forward, test_name, out_dir, matmul_dir=None, plan=None):
    """Every stage of the pass, the tables and the constants, the per-op vectors of the non-linear ops, the
    descriptor program; with matmul_dir also every matmul on its own (tb_matmul_engine_opal_kelly)."""

    def fresh(directory):
        os.makedirs(directory, exist_ok=True)
        for stale in glob.glob(os.path.join(directory, "*")):
            if os.path.isfile(stale):
                os.remove(stale)

    written = set()
    if matmul_dir:
        # one line per op in deit_ops.txt: name, group, A file, B file, bias file, Y file, descriptor file.
        # A, B, Y are INT8 row-major, bias is INT32 per column.
        fresh(matmul_dir)
        with open(os.path.join(matmul_dir, "deit_ops.txt"), "w") as manifest:
            for op in forward.matmuls:
                name, (rows, inner), cols = op["name"], op["a"].shape, op["b"].shape[1]
                row_tiles, col_tiles = -(-rows // SYS_ROW), -(-cols // SYS_COL)
                bank_rows = row_tiles if op["drain_mode"] == COLUMN_SELECT else col_tiles
                assert row_tiles * inner <= BANK_DEPTH and col_tiles * inner <= BANK_DEPTH, name
                assert op["output_base"] + bank_rows * op["output_stride"] <= BANK_DEPTH, name
                assert col_tiles <= PARAMS["BIAS_DEPTH"], name
                descriptor = pack_descriptor(
                    opcode=PARAMS["OPCODE_MATMUL"], rows_total=rows, inner_dim=inner, cols_total=cols,
                    row_tiles=row_tiles, col_tiles=col_tiles,
                    activation_buffer_id=0, weight_buffer_id=1, output_buffer_id=2,
                    activation_base=0, weight_base=0, output_base=op["output_base"],
                    output_stride=op["output_stride"], bias_base=0, scale=op["scale"], shift=op["shift"],
                    drain_mode=op["drain_mode"], bias_enable=1, last_op=1)
                a_file = f"{op['a_name']}_A.hex"
                if a_file not in written:
                    write_hex(os.path.join(matmul_dir, a_file), op["a"], DATA_WIDTH)
                    written.add(a_file)
                write_hex(os.path.join(matmul_dir, f"{name}_B.hex"), op["b"], DATA_WIDTH)
                write_hex(os.path.join(matmul_dir, f"{name}_Y.hex"), op["y"], DATA_WIDTH)
                write_hex(os.path.join(matmul_dir, f"{name}_desc_r{SYS_ROW}c{SYS_COL}.hex"), descriptor, 32)
                write_hex(os.path.join(matmul_dir, f"{name}_bias.hex"), op["bias"], ACC_WIDTH)
                manifest.write(f"{name} {op['group']} {a_file} {name}_B.hex {name}_bias.hex {name}_Y.hex "
                               f"{name}_desc_r{SYS_ROW}c{SYS_COL}.hex\n")

    fresh(out_dir)

    def write(name, values, width):
        write_hex(os.path.join(out_dir, name), values, width)
        return name

    lines = [f"# image {test_name}; layernorm {LAYERNORM.name}; softmax {SOFTMAX.name}; gelu 256 entries; "
             f"add {ADD_MULTIPLIER_WIDTH}-bit multipliers, rounded",
             "# tensor <name> <rows> <cols> <file>      INT8, row-major",
             "# matmul <op> <scale> <shift>             requantizer constants",
             f"# const  <op> <key> <value or file>       files: gain INT{GAIN_WIDTH}, offset INT{OFFSET_WIDTH}, "
             f"gelu table INT8,",
             f"#                                         exp table {SOFTMAX.exp_bits}-bit, rsqrt table "
             f"{LAYERNORM.rsqrt_bits + 1}-bit"]
    for name, tensor, _scale, _key in forward.stages:
        lines.append(f"tensor {name} {tensor.shape[0]} {tensor.shape[1]} {write(name + '.hex', tensor, DATA_WIDTH)}")
    for op in forward.matmuls:
        lines.append(f"matmul {op['name']} {op['scale']} {op['shift']}")
    table = write("rsqrt_table.hex", rsqrt_table(LAYERNORM), LAYERNORM.rsqrt_bits + 1)
    lines.append(f"const layernorm rsqrt_table {table}")
    widths = {"gain": GAIN_WIDTH, "offset": OFFSET_WIDTH, "table": DATA_WIDTH, "exp_table": SOFTMAX.exp_bits}
    for op, constants in forward.constants.items():
        for key, value in constants.items():
            if isinstance(value, np.ndarray):
                value = write(f"{op}_{key}.hex", value, widths[key])
            lines.append(f"const {op} {key} {value}")
    with open(os.path.join(out_dir, "forward_ops.txt"), "w") as manifest:
        manifest.write("\n".join(lines) + "\n")

    # the non-linear ops as engine ops. nonlinear_ops.txt has one line per op: name, kind, a file, b file or
    # "-", y file, descriptor file, and for layernorm the gain and offset files. Every op is there twice:
    # writing to a third buffer, and in place over operand a.
    stage_of = {id(tensor): name for name, tensor, _scale, _key in forward.stages}
    with open(os.path.join(out_dir, "nonlinear_ops.txt"), "w") as manifest:
        for op in forward.row_ops:
            rows, length = op["a"].shape
            for suffix, output_buffer in (("", 2), ("_inplace", 0)):
                name = op["name"] + suffix
                descriptor = pack_descriptor(
                    opcode=ROW_OPCODES[op["kind"]], rows_total=rows, inner_dim=op.get("inner_dim", 0),
                    cols_total=length, row_tiles=-(-rows // SYS_ROW), col_tiles=0,
                    activation_buffer_id=0, weight_buffer_id=1, output_buffer_id=output_buffer,
                    activation_base=0, weight_base=0, output_base=0, output_stride=length,
                    bias_base=op["bias_base"], scale=op.get("scale", 0), shift=op.get("shift", 0),
                    drain_mode=0, bias_enable=0, last_op=1)
                write(f"{name}_desc_r{SYS_ROW}c{SYS_COL}.hex", descriptor, 32)
                a_file = stage_of[id(op["a"])] + ".hex" if id(op["a"]) in stage_of else write(
                    f"{op['name']}_a.hex", op["a"], DATA_WIDTH)
                b_file = stage_of[id(op["b"])] + ".hex" if "b" in op else "-"
                extra = op.get("files", ("-", "-"))
                manifest.write(f"{name} {op['kind']} {a_file} {b_file} {stage_of[id(op['y'])]}.hex "
                               f"{name}_desc_r{SYS_ROW}c{SYS_COL}.hex {extra[0]} {extra[1]}\n")
    # the whole pass as one descriptor program
    stage_file = {tensor_id: name + ".hex" for tensor_id, name in stage_of.items()}
    return layer_program.build(forward, stage_file, out_dir, plan)


def rom_images(forward):
    """The engine's table ROMs for a pass: {file name: (values, width)}. Every table of a kind back to back,
    in the order of the ops (bias_base), unused exp tables zero."""
    gelu_tables = [forward.constants[op["name"]]["table"] for op in forward.row_ops if op["kind"] == "gelu"]
    exp_tables = [forward.constants[op["name"]]["exp_table"] for op in forward.row_ops if op["kind"] == "softmax"]
    assert len(gelu_tables) <= PARAMS["GELU_TABLE_COUNT"] and len(exp_tables) <= PARAMS["SOFTMAX_TABLE_COUNT"]
    gelu_rom = np.zeros(PARAMS["GELU_TABLE_COUNT"] * 256, dtype=np.int64)
    gelu_rom[:256 * len(gelu_tables)] = np.concatenate(gelu_tables)
    exp_rom = np.zeros(PARAMS["SOFTMAX_TABLE_COUNT"] * 256, dtype=np.int64)
    exp_rom[:256 * len(exp_tables)] = np.concatenate(exp_tables)
    return {"gelu_rom.hex": (gelu_rom, DATA_WIDTH), "exp_rom.hex": (exp_rom, SOFTMAX.exp_bits),
            "rsqrt_rom.hex": (rsqrt_table(LAYERNORM), LAYERNORM.rsqrt_bits + 1)}


def load_scales(file_name):
    with open(file_name) as f:
        return json.load(f)


def save_scales(file_name, scales):
    with open(file_name, "w") as f:
        json.dump(dict(sorted(scales.items())), f, indent=1)
        f.write("\n")


def target_for(scales, layers):
    """The scales of an N-layer pass: the shared ones, with the final LayerNorm's and the head's of N."""
    target = {key: value for key, value in scales.items() if "@" not in key}
    for key, value in scales.items():
        if key.endswith(f"@{layers}"):
            target[key.split("@")[0]] = value
    return target
