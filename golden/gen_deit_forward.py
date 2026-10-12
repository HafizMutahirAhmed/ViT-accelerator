"""Golden model of the 2-layer DeiT-Tiny forward pass in integer arithmetic.

Usage: .venv/bin/python golden/gen_deit_forward.py [--no-dump | --dump-only]   (needs torch and timm)
       --dump-only writes the vectors without the accuracy report
Images: golden/images_calibration/ and golden/images_heldout/ come from
scripts/fetch_images.sh; golden/images/ is in the repo.
Output, for the test image (the first one of golden/images/):
  golden/vectors/deit/           the 39 matmuls of the pass, for tb_matmul_engine_deit
  golden/vectors/deit_forward/   every stage, the tables and the constants
  rtl/rom/                       the ROM images of the tables (in the repo)

The pass is the real pretrained model, layers 0 and 1: patch embedding, class
token and position add, then per layer LayerNorm, Q/K/V, S, softmax, O,
projection, residual add, LayerNorm, fc1, GELU, fc2, residual add, and at the
end the final LayerNorm on the class row and the classifier. Each stage reads
the INT8 tensor the stage before it wrote.

Scales. Every tensor has one scale: the largest magnitude seen over the
calibration images, divided by 127. A matmul's requantizer multiplier (8-bit
scale, 5-bit shift) cannot hit a wanted ratio exactly, so the pass carries
each tensor's effective scale, the one the integers really have, and builds
biases, tables and the next multiplier from that. The three heads of O share
one tensor and keep the calibrated scale.

Rounding. The engine requantizer floors. Half an output LSB is added to every
bias word, so the result is rounded to nearest; S and O, which have no bias,
get a constant one for this.

Class token and position. Row 0 of the patch-embedding input is zero, so row 0
of its output is the requantized bias. One residual add then adds a constant
197 x 192 tensor: rows 1..196 hold the position embedding and row 0 holds
class token + position 0 - that bias row.

The non-linear ops are the integer models in nonlinear_ops.py. Each can also
run as a "float" reference (the real function on the dequantized INT8 input,
rounded to INT8), which separates the cost of INT8 tensors from the cost of
the integer arithmetic.

The report is printed twice: for the calibration images (in-sample) and for
images the calibration never saw (held-out).
"""

import glob
import os
import sys

import numpy as np
import timm
import torch

from gen_deit_vectors import IMAGE_DIR, LAYERS, MODEL_NAME, capture, cosine, load_image, quantize
from int8_ops import ACC_MAX, ACC_WIDTH, DATA_MAX, DATA_MIN, DATA_WIDTH, add_bias, requantize, write_hex
from nonlinear_ops import (LayerNormConfig, SoftmaxConfig, add_constants, exp_table, gelu, gelu_real,
                           gelu_table, layernorm, layernorm_constants, pick_multiplier, residual_add,
                           rsqrt_table, softmax)
from pkg_config import PARAMS, pack_descriptor

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(GOLDEN_DIR, "images_calibration")
HELDOUT_DIR = os.path.join(GOLDEN_DIR, "images_heldout")
MATMUL_DIR = os.path.join(GOLDEN_DIR, "vectors", "deit")
FORWARD_DIR = os.path.join(GOLDEN_DIR, "vectors", "deit_forward")
ROM_DIR = os.path.join(os.path.dirname(GOLDEN_DIR), "rtl", "rom")   # ROM images, kept in the repo

CALIBRATION_MIN, HELDOUT_MIN = 128, 32
SYS_ROW = SYS_COL = 8
BANK_DEPTH = PARAMS["BANK_DEPTH_MAX"]  # tb_matmul_engine_deit gives every pool buffer this depth
ROW_SELECT, COLUMN_SELECT = PARAMS["DRAIN_ROW_SELECT"], PARAMS["DRAIN_COLUMN_SELECT"]

# precision of the non-linear ops
LAYERNORM = LayerNormConfig("lut256/r12/n8/g16", "lut", index_bits=8, rsqrt_bits=12, norm_frac=8, gain_bits=16)
SOFTMAX = SoftmaxConfig("exp12/divide", exp_bits=12)
ADD_MULTIPLIER_WIDTH = 8
GAIN_WIDTH, OFFSET_WIDTH = 16, 32   # file widths of the LayerNorm constants

# descriptor opcodes of the row ops; the package may not define all of them yet
ROW_OPCODES = {"add": 1, "gelu": 2, "layernorm": 3, "softmax": 4}
for _kind, _opcode in ROW_OPCODES.items():
    assert PARAMS.get(f"OPCODE_{_kind.upper()}", _opcode) == _opcode, _kind


def capture_all(model, image):
    """capture() plus the residual stream, the logits and the head output."""
    fp = capture(model, image)
    grabbed = {}

    def grab(name):
        def hook(_module, inputs, output):
            grabbed[name + "_in"] = inputs[0].detach()[0].numpy().astype(np.float64)
            grabbed[name + "_out"] = output.detach()[0].numpy().astype(np.float64)
        return hook

    hooks = []
    for layer in range(LAYERS):
        hooks += [model.blocks[layer].register_forward_hook(grab(f"L{layer}_block")),
                  model.blocks[layer].norm2.register_forward_hook(grab(f"L{layer}_norm2"))]
    with torch.no_grad():
        fp["logits_full"] = model(image)[0].numpy().astype(np.float64)[None, :]
    for hook in hooks:
        hook.remove()
    for layer in range(LAYERS):
        fp[f"L{layer}_x_in"] = grabbed[f"L{layer}_block_in"]
        fp[f"L{layer}_x_mid"] = grabbed[f"L{layer}_norm2_in"]
        fp[f"L{layer}_x_out"] = grabbed[f"L{layer}_block_out"]
    weight = model.head.weight.detach().numpy().astype(np.float64).T
    fp["head_out"] = fp["head_in"] @ weight + model.head.bias.detach().numpy().astype(np.float64)
    # the patch embedding with a zero row 0: its output row 0 is the bias
    bias = model.patch_embed.proj.bias.detach().numpy().astype(np.float64)
    fp["embed_in"] = np.vstack([np.zeros((1, fp["embed_in"].shape[1])), fp["embed_in"]])
    fp["embed_out"] = np.vstack([bias[None, :], fp["embed_out"]])
    return fp


class Model:
    """Quantized weights, FP32 parameters and scales: everything that does not depend on the image."""

    def __init__(self, model, target):
        self.target = target
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
        for layer in range(LAYERS):
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
        head_weight, head_bias = numpy(model.head.weight).T, numpy(model.head.bias)
        add_linear("head1000", head_weight, head_bias)
        add_linear("head10", head_weight[:, :10], head_bias[:10])

        position = numpy(model.pos_embed)[0]
        self.embed_bias = numpy(model.patch_embed.proj.bias)
        self.class_row = numpy(model.cls_token)[0, 0] + position[0]
        self.position = position[1:]

    def addend_real(self, bias_row=None):
        """The constant of the class-and-position add. bias_row: what row 0 of the embedding output holds."""
        bias_row = self.embed_bias if bias_row is None else bias_row
        return np.vstack([(self.class_row - bias_row)[None, :], self.position])


def layernorm_real(x, gamma, beta, epsilon):
    mean = x.mean(axis=1, keepdims=True)
    variance = x.var(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(variance + epsilon) * gamma + beta


def softmax_real(scores):
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


GELU_REAL = np.vectorize(gelu_real)


def matmul(a, b):
    """int8_ops.matmul_acc, but through float64 for speed. Exact: every sum is far below 2**53."""
    acc = np.rint(a.astype(np.float64) @ b.astype(np.float64)).astype(np.int64)
    assert np.abs(acc).max() <= ACC_MAX
    return acc


class Forward:
    """One image through the integer model.

    options: add, gelu, layernorm, softmax are "int" or "float"; rounding is
    whether half an LSB goes into the matmul biases.
    """

    def __init__(self, parameters, options):
        self.m, self.o, self.t = parameters, options, parameters.target
        self.stages = []      # (name, INT8 tensor, scale, FP32 key or None)
        self.constants = {}   # non-linear op name -> dict of integer constants
        self.matmuls = []     # one dict per engine op, in order
        self.row_ops = []     # one dict per non-linear op, in order
        self.acc_max = 0
        self.gain_error = 0.0

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
        if self.o["rounding"] and shift > 0:
            bias = bias + int(round((1 << (shift - 1)) / scale))   # half an output LSB
        has_bias = bias_real is not None or self.o["rounding"]
        acc = matmul(a, b)
        if has_bias:
            acc = add_bias(acc, bias)
        self.acc_max = max(self.acc_max, int(np.abs(acc).max()))
        y = requantize(acc, scale, shift)
        effective = acc_scale * (1 << shift) / scale
        self.gain_error = max(self.gain_error, abs(effective / self.t[target_key] - 1.0))
        natural = b.shape[1] if drain_mode == COLUMN_SELECT else a.shape[0]
        self.matmuls.append(dict(name=name, group=group, a_name=a_name, a=a, b=b, bias=bias if has_bias else None,
                                 y=y, scale=scale, shift=shift, drain_mode=drain_mode, output_base=output_base,
                                 output_stride=natural if output_stride is None else output_stride))
        return y, (self.t[target_key] if fixed_scale else effective)

    def linear(self, name, group, a_name, a, a_scale, target_key, **layout):
        w_q, weight_scale, bias = self.m.linear[name]
        return self.engine_op(name, group, a_name, a, w_q, a_scale * weight_scale, target_key, bias, **layout)

    def add(self, name, a, a_scale, b, b_scale, target_key):
        if self.o["add"] == "float":
            scale = self.t[target_key]
            return quantize(a * a_scale + b * b_scale, scale), scale
        limit = (1 << (ADD_MULTIPLIER_WIDTH - 1)) - 1
        mult_a, mult_b, shift, effective, gain_error = add_constants(a_scale, b_scale, self.t[target_key], limit)
        self.constants[name] = dict(mult_a=mult_a, mult_b=mult_b, shift=shift)
        self.gain_error = max(self.gain_error, abs(gain_error))
        y = residual_add(a, b, mult_a, mult_b, shift, True, ADD_MULTIPLIER_WIDTH)
        self.row_ops.append(dict(name=name, kind="add", a=a, b=b, y=y, scale=mult_a, bias_base=mult_b & 0xFF,
                                 shift=shift))
        return y, effective

    def layernorm(self, name, x, x_scale, target_key):
        gamma, beta, epsilon = self.m.norm[name]
        scale = self.t[target_key]
        if self.o["layernorm"] == "float":
            return quantize(layernorm_real(x * x_scale, gamma, beta, epsilon), scale), scale
        gain, offset, out_shift, epsilon_int = layernorm_constants(gamma, beta, x_scale, scale, epsilon, LAYERNORM)
        self.constants[name] = dict(gain=gain, offset=offset, out_shift=out_shift, epsilon_int=epsilon_int)
        y = layernorm(x, gain, offset, out_shift, epsilon_int, LAYERNORM, rsqrt_table(LAYERNORM))
        # bias_base: where this LayerNorm's gain and offset start in the constants memory
        self.row_ops.append(dict(name=name, kind="layernorm", a=x, y=y, shift=out_shift, inner_dim=epsilon_int,
                                 bias_base=sum(op["kind"] == "layernorm" for op in self.row_ops) * x.shape[1],
                                 files=(f"{name}_gain.hex", f"{name}_offset.hex")))
        return y, scale

    def softmax(self, name, scores, score_scale):
        if self.o["softmax"] == "float":
            return quantize(softmax_real(scores * score_scale) * DATA_MAX, 1.0).clip(0), 1.0 / DATA_MAX
        table = exp_table(score_scale, SOFTMAX)
        self.constants[name] = dict(exp_table=table)
        y = softmax(scores, table, SOFTMAX)
        # bias_base: which exp table, one per head and layer
        self.row_ops.append(dict(name=name, kind="softmax", a=scores, y=y,
                                 bias_base=sum(op["kind"] == "softmax" for op in self.row_ops)))
        return y, 1.0 / DATA_MAX

    def gelu(self, name, x, x_scale, target_key):
        scale = self.t[target_key]
        if self.o["gelu"] == "float":
            return quantize(GELU_REAL(x * x_scale), scale), scale
        table = gelu_table(x_scale, scale)
        self.constants[name] = dict(table=table)
        y = gelu(x, table)
        # bias_base: which table, one per layer
        self.row_ops.append(dict(name=name, kind="gelu", a=x, y=y,
                                 bias_base=sum(op["kind"] == "gelu" for op in self.row_ops)))
        return y, scale

    def run(self, fp):
        m, keep = self.m, self.keep
        embed_in, embed_in_scale = keep("embed_in", quantize(fp["embed_in"], self.t["embed_in"]), self.t["embed_in"])
        patches, patch_scale = keep("embed_out", *self.linear("embed", "embed", "embed_in", embed_in, embed_in_scale,
                                                              "embed_out"), "embed_out")
        # row 0 of patches is the requantized bias; the addend's row 0 makes it class token + position 0
        addend_real = m.addend_real(patches[0] * patch_scale)
        addend_scale = np.abs(addend_real).max() / DATA_MAX
        addend, _ = keep("addend", quantize(addend_real, addend_scale), addend_scale)
        x, x_scale = keep("L0_x_in", *self.add("position_add", patches, patch_scale, addend, addend_scale, "L0_x_in"),
                          "L0_x_in")

        for layer in range(LAYERS):
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
                # O = P @ V lands in its slot of the 197 x 192 tensor
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

        # the 2-layer truncated model: final LayerNorm on the class row, then the classifier
        final, final_scale = keep("final_ln", *self.layernorm("final_ln", x[0:1], x_scale, "head_in"), "head_in")
        keep("head10", *self.linear("head10", "head", "final_ln", final, final_scale, "head_out"))
        keep("head1000", *self.linear("head1000", "head", "final_ln", final, final_scale, "head_out"), "head_out")
        return self


class Tally:
    """Error statistics of one op variant over many instances."""

    def __init__(self):
        self.max_error, self.mean_error, self.cosines = 0.0, [], []
        self.exact, self.count, self.max_step = 0, 0, 0

    def add(self, result, scale, real, ideal):
        """real: the FP32 tensor. ideal: the float op on the same INT8 input, rounded to INT8."""
        error = np.abs(result - real / scale)
        self.max_error = max(self.max_error, float(error.max()))
        self.mean_error.append(float(error.mean()))
        self.cosines.append(cosine(result, real))
        self.exact += int((result == ideal).sum())
        self.count += result.size
        self.max_step = max(self.max_step, int(np.abs(result - ideal).max()))

    def row(self, name):
        return (f"  {name:34s} {self.max_error:8.2f} {np.mean(self.mean_error):8.3f} {np.mean(self.cosines):9.5f} "
                f"{min(self.cosines):9.5f} {100.0 * self.exact / self.count:8.2f}% {self.max_step:4d}")


FLOAT_OPS = dict(add="float", gelu="float", layernorm="float", softmax="float", rounding=True)
INTEGER_OPS = dict(add="int", gelu="int", layernorm="int", softmax="int", rounding=True)
VARIANTS = [("INT8 tensors, float non-linear ops", FLOAT_OPS),
            ("all integer", INTEGER_OPS),
            ("all integer, requantizer floors", dict(INTEGER_OPS, rounding=False))]
PER_OP = ["residual add", "GELU", "LayerNorm, 197 rows", "LayerNorm, class row", "softmax"]


class Evaluation:
    """Per-op and end-to-end statistics over one set of images."""

    def __init__(self, parameters):
        self.m = parameters
        self.tallies = {(op, kind): Tally() for op in PER_OP for kind in ("float", "integer")}
        self.stage_cosines = {name: {} for name, _ in VARIANTS}
        self.final_error = {name: [] for name, _ in VARIANTS}
        self.agree = {name: [0, 0] for name, _ in VARIANTS}
        self.full_correct = self.truncated_correct = self.count = 0
        self.truncated_classes = set()
        self.acc_max, self.gain_error = 0, 0.0

    def per_op(self, fp):
        """Each op on its own real input, quantized with the calibrated scale."""
        m, target, tallies = self.m, self.m.target, self.tallies
        limit = (1 << (ADD_MULTIPLIER_WIDTH - 1)) - 1

        addend_real = m.addend_real()
        cases = [(fp["embed_out"], target["embed_out"], addend_real, np.abs(addend_real).max() / DATA_MAX, "L0_x_in")]
        for layer in range(LAYERS):
            p = f"L{layer}"
            cases += [(fp[f"{p}_x_in"], target[f"{p}_x_in"], fp[f"{p}_proj_out"], target[f"{p}_proj_out"],
                       f"{p}_x_mid"),
                      (fp[f"{p}_x_mid"], target[f"{p}_x_mid"], fp[f"{p}_fc2_out"], target[f"{p}_fc2_out"],
                       f"{p}_x_out")]
        for a_real, a_scale, b_real, b_scale, out_key in cases:
            a, b, real = quantize(a_real, a_scale), quantize(b_real, b_scale), fp[out_key]
            ideal = quantize(a * a_scale + b * b_scale, target[out_key])
            tallies["residual add", "float"].add(ideal, target[out_key], real, ideal)
            mult_a, mult_b, shift, effective, _ = add_constants(a_scale, b_scale, target[out_key], limit)
            result = residual_add(a, b, mult_a, mult_b, shift, True, ADD_MULTIPLIER_WIDTH)
            # compare at the scale the integers really have
            tallies["residual add", "integer"].add(result, effective, real,
                                                   quantize(a * a_scale + b * b_scale, effective))

        for layer in range(LAYERS):
            p = f"L{layer}"
            in_scale, out_scale, real = target[f"{p}_fc1_out"], target[f"{p}_fc2_in"], fp[f"{p}_fc2_in"]
            x = quantize(fp[f"{p}_fc1_out"], in_scale)
            ideal = quantize(GELU_REAL(x * in_scale), out_scale)
            tallies["GELU", "float"].add(ideal, out_scale, real, ideal)
            tallies["GELU", "integer"].add(gelu(x, gelu_table(in_scale, out_scale)), out_scale, real, ideal)

        cases = [("LayerNorm, class row", "final_ln", fp[f"L{LAYERS - 1}_x_out"][0:1], f"L{LAYERS - 1}_x_out",
                  "head_in")]
        for layer in range(LAYERS):
            p = f"L{layer}"
            cases += [("LayerNorm, 197 rows", f"{p}_ln1", fp[f"{p}_x_in"], f"{p}_x_in", f"{p}_qkv_in"),
                      ("LayerNorm, 197 rows", f"{p}_ln2", fp[f"{p}_x_mid"], f"{p}_x_mid", f"{p}_fc1_in")]
        table = rsqrt_table(LAYERNORM)
        for op, name, x_real, in_key, out_key in cases:
            gamma, beta, epsilon = m.norm[name]
            x, real = quantize(x_real, target[in_key]), fp[out_key]
            ideal = quantize(layernorm_real(x * target[in_key], gamma, beta, epsilon), target[out_key])
            tallies[op, "float"].add(ideal, target[out_key], real, ideal)
            gain, offset, out_shift, epsilon_int = layernorm_constants(gamma, beta, target[in_key], target[out_key],
                                                                      epsilon, LAYERNORM)
            tallies[op, "integer"].add(layernorm(x, gain, offset, out_shift, epsilon_int, LAYERNORM, table),
                                       target[out_key], real, ideal)

        for layer in range(LAYERS):
            for head in range(m.heads):
                h = f"L{layer}_h{head}"
                s, real = quantize(fp[f"{h}_s"], target[f"{h}_s"]), fp[f"{h}_p"]
                ideal = np.clip(np.rint(softmax_real(s * target[f"{h}_s"]) * DATA_MAX), 0, DATA_MAX).astype(np.int64)
                tallies["softmax", "float"].add(ideal, 1.0 / DATA_MAX, real, ideal)
                tallies["softmax", "integer"].add(softmax(s, exp_table(target[f"{h}_s"], SOFTMAX), SOFTMAX),
                                                  1.0 / DATA_MAX, real, ideal)

    def add_image(self, fp, label):
        self.count += 1
        self.per_op(fp)
        reference = int(np.argmax(fp["head_out"]))
        reference10 = int(np.argmax(fp["head_out"][0, :10]))
        self.full_correct += int(np.argmax(fp["logits_full"])) == label
        self.truncated_correct += reference == label
        self.truncated_classes.add(reference)
        final = f"L{LAYERS - 1}_x_out"
        for name, options in VARIANTS:
            forward = Forward(self.m, options).run(fp)
            for stage, tensor, scale, key in forward.stages:
                if key is not None:
                    self.stage_cosines[name].setdefault(stage, []).append(cosine(tensor, fp[key]))
                if stage == final:
                    self.final_error[name].append(float(np.abs(tensor - fp[key] / scale).mean()))
                if stage == "head1000":
                    self.agree[name][0] += int(np.argmax(tensor)) == reference
                if stage == "head10":
                    self.agree[name][1] += int(np.argmax(tensor)) == reference10
            if options is INTEGER_OPS:
                self.acc_max = max(self.acc_max, forward.acc_max)
                self.gain_error = max(self.gain_error, forward.gain_error)

    def report(self, title):
        count, final = self.count, f"L{LAYERS - 1}_x_out"
        print(f"\n=== {title}: {count} images ===")
        print("Per-op error, each op fed its own real input. Errors are in output LSBs against the FP32 tensor.")
        print("'= float' is the share of outputs equal to the real function on the same INT8 input, rounded;")
        print("'max' is the largest difference from it.")
        print(f"  {'op':34s} {'max err':>8s} {'mean err':>8s} {'cos mean':>9s} {'cos min':>9s} {'= float':>9s} "
              f"{'max':>4s}")
        for op in PER_OP:
            print(self.tallies[op, "float"].row(f"{op}, float reference"))
            print(self.tallies[op, "integer"].row(f"{op}, integer"))
        print(f"End to end. 'cos' and 'err' are the layer {LAYERS} output (197x192) against FP32, 'final LN' and "
              f"'logits' the class row")
        print("after the final LayerNorm and the 1000 logits. Top-1 is agreement with the FP32 2-layer model.")
        print(f"  {'variant':36s} {'cos mean':>9s} {'cos min':>9s} {'err LSB':>8s} {'final LN':>9s} {'logits':>8s} "
              f"{'top-1 /1000':>12s} {'top-1 /10':>11s}")
        for name, _ in VARIANTS:
            cosines = self.stage_cosines[name]
            print(f"  {name:36s} {np.mean(cosines[final]):9.5f} {min(cosines[final]):9.5f} "
                  f"{np.mean(self.final_error[name]):8.3f} {np.mean(cosines['final_ln']):9.4f} "
                  f"{np.mean(cosines['head1000']):8.4f} {self.agree[name][0]:5d} of {count:3d} "
                  f"{self.agree[name][1]:4d} of {count:3d}")
        print(f"FP32 context: full 12-layer model top-1 correct {self.full_correct} of {count}; 2-layer truncated "
              f"model {self.truncated_correct} of {count}, predicting {len(self.truncated_classes)} distinct "
              f"class(es)")
        print(f"largest |accumulator| {self.acc_max} ({100.0 * self.acc_max / 2 ** 31:.4f}% of 2^31); multipliers "
              f"at most {100 * self.gain_error:.2f}% from the wanted ratio")

    def stage_table(self):
        return self.stage_cosines["all integer"]


def dump(parameters, fp, test_name):
    """The test image: the engine ops for tb_matmul_engine_deit and every stage of the pass."""
    forward = Forward(parameters, INTEGER_OPS).run(fp)

    def fresh(directory):
        os.makedirs(directory, exist_ok=True)
        for stale in glob.glob(os.path.join(directory, "*")):
            os.remove(stale)

    # golden/vectors/deit: one line per op in deit_ops.txt, name, group, A file, B file, bias file, Y file,
    # descriptor file. A, B, Y are INT8 row-major, bias is INT32 per column.
    fresh(MATMUL_DIR)
    written = set()
    with open(os.path.join(MATMUL_DIR, "deit_ops.txt"), "w") as manifest:
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
                drain_mode=op["drain_mode"], bias_enable=int(op["bias"] is not None), last_op=1)
            a_file = f"{op['a_name']}_A.hex"
            if a_file not in written:
                write_hex(os.path.join(MATMUL_DIR, a_file), op["a"], DATA_WIDTH)
                written.add(a_file)
            write_hex(os.path.join(MATMUL_DIR, f"{name}_B.hex"), op["b"], DATA_WIDTH)
            write_hex(os.path.join(MATMUL_DIR, f"{name}_Y.hex"), op["y"], DATA_WIDTH)
            write_hex(os.path.join(MATMUL_DIR, f"{name}_desc_r{SYS_ROW}c{SYS_COL}.hex"), descriptor, 32)
            bias_file = "-"
            if op["bias"] is not None:
                bias_file = f"{name}_bias.hex"
                write_hex(os.path.join(MATMUL_DIR, bias_file), op["bias"], ACC_WIDTH)
            manifest.write(f"{name} {op['group']} {a_file} {name}_B.hex {bias_file} {name}_Y.hex "
                           f"{name}_desc_r{SYS_ROW}c{SYS_COL}.hex\n")

    # golden/vectors/deit_forward: every stage, the tables and the constants
    fresh(FORWARD_DIR)

    def write(name, values, width):
        write_hex(os.path.join(FORWARD_DIR, name), values, width)
        return name

    lines = [f"# image {test_name}; layernorm {LAYERNORM.name}; softmax {SOFTMAX.name}; gelu 256 entries; "
             f"add {ADD_MULTIPLIER_WIDTH}-bit multipliers, rounded",
             "# tensor <name> <rows> <cols> <file>      INT8, row-major",
             "# matmul <op> <scale> <shift>             requantizer constants; operands are in ../deit/",
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
    with open(os.path.join(FORWARD_DIR, "forward_ops.txt"), "w") as manifest:
        manifest.write("\n".join(lines) + "\n")

    # ROM images for the RTL: every table of a kind back to back, in the order of the ops (bias_base)
    os.makedirs(ROM_DIR, exist_ok=True)
    write_hex(os.path.join(ROM_DIR, "gelu_rom.hex"),
              np.concatenate([forward.constants[op["name"]]["table"] for op in forward.row_ops
                              if op["kind"] == "gelu"]), DATA_WIDTH)
    write_hex(os.path.join(ROM_DIR, "rsqrt_rom.hex"), rsqrt_table(LAYERNORM), LAYERNORM.rsqrt_bits + 1)
    exp_tables = [forward.constants[op["name"]]["exp_table"] for op in forward.row_ops if op["kind"] == "softmax"]
    exp_rom = np.zeros(PARAMS.get("SOFTMAX_TABLE_COUNT", 8) * 256, dtype=np.int64)   # unused tables are zero
    exp_rom[:256 * len(exp_tables)] = np.concatenate(exp_tables)
    write_hex(os.path.join(ROM_DIR, "exp_rom.hex"), exp_rom, SOFTMAX.exp_bits)

    # the non-linear ops as engine ops. nonlinear_ops.txt has one line per op: name, kind, a file, b file or
    # "-", y file, descriptor file, and for layernorm the gain and offset files. Every op is there twice:
    # writing to a third buffer, and in place over operand a.
    stage_of = {id(tensor): name for name, tensor, _scale, _key in forward.stages}
    with open(os.path.join(FORWARD_DIR, "nonlinear_ops.txt"), "w") as manifest:
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
    print(f"\nwrote {len(os.listdir(MATMUL_DIR))} files to {os.path.relpath(MATMUL_DIR, GOLDEN_DIR)} "
          f"({len(forward.matmuls)} engine ops) and {len(os.listdir(FORWARD_DIR))} files to "
          f"{os.path.relpath(FORWARD_DIR, GOLDEN_DIR)}, test image {test_name}")


def main():
    torch.manual_seed(0)
    model = timm.create_model(MODEL_NAME, pretrained=True).eval()
    for block in model.blocks:
        block.attn.fused_attn = False  # so the softmax output passes through attn_drop
    cfg = model.pretrained_cfg
    synsets = timm.data.ImageNetInfo("imagenet-1k").label_names()

    def images(*directories):
        return [name for directory in directories for name in sorted(glob.glob(os.path.join(directory, "*.JPEG")))]

    calibration, heldout = images(CALIBRATION_DIR), images(IMAGE_DIR, HELDOUT_DIR)
    assert len(calibration) >= CALIBRATION_MIN and len(heldout) >= HELDOUT_MIN, \
        "images missing: run scripts/fetch_images.sh"
    assert not {os.path.basename(f) for f in calibration} & {os.path.basename(f) for f in heldout}

    def label(file_name):
        return synsets.index(os.path.basename(file_name).split("_")[0])

    # pass 1: the scale of a tensor is its largest magnitude over the calibration images / 127
    largest = {}
    for file_name in calibration:
        for key, tensor in capture_all(model, load_image(file_name, cfg)).items():
            largest[key] = max(largest.get(key, 0.0), float(np.abs(tensor).max()))
    parameters = Model(model, {key: value / DATA_MAX for key, value in largest.items()})

    fp = capture_all(model, load_image(heldout[0], cfg))
    assert np.allclose(fp["embed_out"] + parameters.addend_real(), fp["L0_x_in"], atol=1e-4)

    print(f"model {MODEL_NAME}; scales from {len(calibration)} calibration images (max); "
          f"{len(heldout)} held-out images; layernorm {LAYERNORM.name}, softmax {SOFTMAX.name}")
    # pass 2: evaluate on both sets
    evaluations = {}
    sets = (("IN-SAMPLE (the calibration images)", calibration),
            ("HELD-OUT (never seen by the calibration)", heldout))
    for title, files in (() if "--dump-only" in sys.argv else sets):
        evaluation = Evaluation(parameters)
        for file_name in files:
            evaluation.add_image(capture_all(model, load_image(file_name, cfg)), label(file_name))
        evaluation.report(title)
        evaluations[title] = evaluation

    tables = [evaluation.stage_table() for evaluation in evaluations.values()]
    if tables:
        print("\nAll integer: cosine against FP32 at every stage, mean / worst")
        print(f"  {'stage':12s} {'in-sample':>20s} {'held-out':>20s}")
    for stage in (tables[0] if tables else ()):
        print(f"  {stage:12s} " + "  ".join(f"{np.mean(t[stage]):10.5f} {min(t[stage]):9.5f}" for t in tables))

    if "--no-dump" not in sys.argv:
        dump(parameters, fp, os.path.basename(heldout[0]))


if __name__ == "__main__":
    main()
