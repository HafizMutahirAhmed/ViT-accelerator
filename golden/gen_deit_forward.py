"""Full 2-layer DeiT-Tiny forward pass in integer arithmetic, with the non-linear ops.

Usage: .venv/bin/python golden/gen_deit_forward.py [--max] [--no-dump]   (needs torch and timm)
Output: golden/vectors/deit_forward/*.hex and forward_ops.txt.

gen_deit_vectors.py tests every matmul on its own captured input. This script
chains everything: patch embedding, position add, then per layer LayerNorm,
Q/K/V, S, softmax, O, projection, residual add, LayerNorm, fc1, GELU, fc2,
residual add, and at the end the final LayerNorm and the classifier. Each
stage reads the INT8 tensor the stage before it wrote.

Scales. Every tensor has a target scale. Calibration A is the per-tensor
percentile scale of gen_deit_vectors.py, from the same calibration images, with
the engine requantizer as it is. Calibration B (--max) takes the largest
calibration value instead and adds half an output LSB to every matmul bias, so
the requantizer rounds to nearest; it needs no engine change. The end-to-end
report covers both; the per-op report and the dump use the selected one. A matmul's
requantizer multiplier (8-bit scale, 5-bit shift) cannot hit a target ratio
exactly, so the script carries each tensor's effective scale, the one the
integers really have, and builds biases, tables and the next multiplier from
that. The three heads of O share one tensor and keep the target scale.

The non-linear ops are the integer models in nonlinear_ops.py. Each can also
run as a "float" reference (the real function on the dequantized INT8 input,
rounded to INT8), which separates the cost of INT8 tensors from the cost of
the integer approximation.

Reports: per-op error against the FP32 tensor (each op fed its own real
input), then the end-to-end comparison over all images.
"""

import glob
import os
import sys

import numpy as np
import timm
import torch

from gen_deit_vectors import (ACTIVATION_PERCENTILE, IMAGE_DIR, LAYERS, MODEL_NAME, capture, cosine,
                              load_image, quantize)
from int8_ops import ACC_MAX, ACC_WIDTH, DATA_MAX, DATA_MIN, DATA_WIDTH, add_bias, read_hex, write_hex
from nonlinear_ops import (LayerNormConfig, SoftmaxConfig, add_constants, exp_table, gelu, gelu_real,
                           gelu_table, layernorm, layernorm_constants, pick_multiplier, reciprocal_table,
                           residual_add, rsqrt_table, softmax)

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
DEIT_DIR = os.path.join(GOLDEN_DIR, "vectors", "deit")
VEC_DIR = os.path.join(GOLDEN_DIR, "vectors", "deit_forward")

LAYERNORM_OPTIONS = [
    LayerNormConfig("lut64/r8/n6/g8", "lut", index_bits=6, rsqrt_bits=8, norm_frac=6, gain_bits=8),
    LayerNormConfig("lut256/r12/n8/g8", "lut", index_bits=8, rsqrt_bits=12, norm_frac=8, gain_bits=8),
    LayerNormConfig("lut256/r12/n8/g16", "lut", index_bits=8, rsqrt_bits=12, norm_frac=8, gain_bits=16),
    LayerNormConfig("lut1024/r16/n10/g16", "lut", index_bits=10, rsqrt_bits=16, norm_frac=10, gain_bits=16),
    LayerNormConfig("isqrt/r17/n10/g16", "isqrt", rsqrt_bits=17, norm_frac=10, gain_bits=16),
]
SOFTMAX_OPTIONS = [
    SoftmaxConfig("exp8/divide", exp_bits=8),
    SoftmaxConfig("exp12/divide", exp_bits=12),
    SoftmaxConfig("exp16/divide", exp_bits=16),
    SoftmaxConfig("exp12/lut64", exp_bits=12, reciprocal="lut", index_bits=6),
    SoftmaxConfig("exp12/lut256", exp_bits=12, reciprocal="lut", index_bits=8),
    SoftmaxConfig("exp12/divide/x2", exp_bits=12, gain_log2=1),
    SoftmaxConfig("exp12/divide/cal", exp_bits=12, calibrated=True),
]
GELU_WIDTHS = [8, 10, 12]

# the configuration that is dumped
RECOMMENDED = dict(layernorm="lut256/r12/n8/g16", softmax="exp12/divide", gelu_width=8,
                   add_rounding=True, add_multiplier_width=8)


def by_name(options, name):
    return next(option for option in options if option.name == name)


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
    return fp


class Model:
    """Quantized weights, FP32 parameters and target scales: everything that does not depend on the image."""

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
        self.class_row = numpy(model.cls_token)[0] + position[0:1]
        self.position_scale = np.abs(position[1:]).max() / DATA_MAX
        self.position = quantize(position[1:], self.position_scale)
        self.position_real = position[1:]


def layernorm_real(x, gamma, beta, epsilon):
    mean = x.mean(axis=1, keepdims=True)
    variance = x.var(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(variance + epsilon) * gamma + beta


def softmax_real(scores):
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


GELU_REAL = np.vectorize(gelu_real)


def matmul(a, b):
    """matmul_acc, but through float64 for speed. Exact: every sum is far below 2**53."""
    acc = np.rint(a.astype(np.float64) @ b.astype(np.float64)).astype(np.int64)
    assert np.abs(acc).max() <= ACC_MAX
    return acc


def numerator_of(scale):
    """Softmax numerator for a calibrated scale of P: the output is P * numerator."""
    return max(DATA_MAX, int(round(1.0 / scale)))


class Forward:
    """One image through the integer model. Stages are kept in order in self.stages."""

    def __init__(self, parameters, options):
        self.m, self.o = parameters, options
        self.t = dict(parameters.target, **options.get("target", {}))   # target scales, with overrides
        self.stages = []      # (name, INT8 tensor, scale, FP32 key or None)
        self.constants = {}   # name -> dict of integer constants, for the dump
        self.acc_max = 0
        self.widths = {"layernorm": {}, "softmax": {}}

    def keep(self, name, tensor, scale, key=None):
        self.stages.append((name, tensor, scale, key))
        return tensor, scale

    def linear(self, name, a, a_scale, target_key, weight_name=None, extra_bits=0):
        w_q, weight_scale, bias = self.m.linear[weight_name or name]
        bias_q = np.rint(bias / (a_scale * weight_scale)).astype(np.int64)
        acc = add_bias(matmul(a, w_q), bias_q)
        return self.requantize(name, acc, a_scale * weight_scale, target_key, bias_q, extra_bits)

    def requantize(self, name, acc, acc_scale, target_key, bias_q=None, extra_bits=0, fixed_scale=False):
        """The engine requantizer. extra_bits > 0 keeps a wider result (GELU study only)."""
        self.acc_max = max(self.acc_max, int(np.abs(acc).max()))
        assert np.abs(acc).max() < ACC_MAX
        scale, shift = pick_multiplier(acc_scale / self.t[target_key])
        assert shift >= extra_bits, name
        if self.o["matmul_rounding"] and shift > 0:
            # half an output LSB, as an addition to every bias word
            rounding = int(round((1 << (shift - 1)) / scale))
            acc = acc + rounding
            bias_q = rounding if bias_q is None else bias_q + rounding
        low, high = DATA_MIN << extra_bits, ((DATA_MAX + 1) << extra_bits) - 1
        y = np.clip((acc * scale) >> (shift - extra_bits), low, high)
        effective = acc_scale * (1 << shift) / scale
        self.constants[name] = dict(scale=scale, shift=shift, bias=bias_q,
                                    gain_error=effective / self.t[target_key] - 1.0)
        out_scale = self.t[target_key] if fixed_scale else effective
        return y, out_scale / (1 << extra_bits)

    def add(self, name, a, a_scale, b, b_scale, target_key):
        if self.o["add"] == "float":
            scale = self.t[target_key]
            return quantize(a * a_scale + b * b_scale, scale), scale
        limit = (1 << (self.o["add_multiplier_width"] - 1)) - 1
        mult_a, mult_b, shift, effective, gain_error = add_constants(a_scale, b_scale, self.t[target_key],
                                                                     limit)
        self.constants[name] = dict(mult_a=mult_a, mult_b=mult_b, shift=shift, gain_error=gain_error)
        y = residual_add(a, b, mult_a, mult_b, shift, self.o["add_rounding"], self.o["add_multiplier_width"])
        return y, effective

    def layernorm(self, name, x, x_scale, target_key):
        gamma, beta, epsilon = self.m.norm[name]
        scale = self.t[target_key]
        config = self.o["layernorm"]
        if config == "float":
            return quantize(layernorm_real(x * x_scale, gamma, beta, epsilon), scale), scale
        gain, offset, out_shift, epsilon_int = layernorm_constants(gamma, beta, x_scale, scale, epsilon, config)
        table = rsqrt_table(config) if config.rsqrt == "lut" else None
        self.constants[name] = dict(gain=gain, offset=offset, out_shift=out_shift, epsilon_int=epsilon_int)
        y = layernorm(x, gain, offset, out_shift, epsilon_int, config, table, self.widths["layernorm"])
        return y, scale

    def softmax(self, name, scores, score_scale):
        config = self.o["softmax"]
        if config == "float":
            return quantize(softmax_real(scores * score_scale) * DATA_MAX, 1.0).clip(0), 1.0 / DATA_MAX
        table = exp_table(score_scale, config)
        recip = reciprocal_table(config) if config.reciprocal == "lut" else None
        self.constants[name] = dict(exp_table=table)
        numerator = numerator_of(self.t[name]) if config.calibrated else DATA_MAX << config.gain_log2
        self.constants[name]["numerator"] = numerator
        y = softmax(scores, table, config, recip, self.widths["softmax"], numerator)
        return y, 1.0 / numerator

    def gelu(self, name, x, x_scale, target_key):
        scale = self.t[target_key]
        if self.o["gelu"] == "float":
            return quantize(GELU_REAL(x * x_scale), scale), scale
        table = gelu_table(x_scale, scale, self.o["gelu_width"])
        self.constants[name] = dict(table=table)
        return gelu(x, table), scale

    def run(self, fp):
        m, keep = self.m, self.keep
        embed_in = quantize(fp["embed_in"], self.t["embed_in"])
        keep("embed_in", embed_in, self.t["embed_in"])
        patches, patch_scale = keep("embed_out", *self.linear("embed", embed_in, self.t["embed_in"], "embed_out"),
                                    "embed_out")
        keep("position", m.position, m.position_scale)
        body, x_scale = self.add("position_add", patches, patch_scale, m.position, m.position_scale, "L0_x_in")
        # the class row is a constant: class token + position 0, written by the host
        x = np.vstack([quantize(m.class_row, x_scale), body])
        keep("L0_x_in", x, x_scale, "L0_x_in")

        for layer in range(LAYERS):
            p = f"L{layer}"
            n1, n1_scale = keep(f"{p}_ln1", *self.layernorm(f"{p}_ln1", x, x_scale, f"{p}_qkv_in"), f"{p}_qkv_in")
            attention = np.zeros((x.shape[0], m.embed), dtype=np.int64)
            for head in range(m.heads):
                h = f"{p}_h{head}"
                q, q_scale = keep(f"{h}_q", *self.linear(f"{h}_q", n1, n1_scale, f"{h}_q"), f"{h}_q")
                k, k_scale = keep(f"{h}_k", *self.linear(f"{h}_k", n1, n1_scale, f"{h}_k"), f"{h}_k")
                v, v_scale = keep(f"{h}_v", *self.linear(f"{h}_v", n1, n1_scale, f"{h}_v"), f"{h}_v")
                # S = Q @ K^T; the 1/sqrt(head_dim) is part of the requantizer multiplier
                s, s_scale = keep(f"{h}_s", *self.requantize(f"{h}_s", matmul(q, k.T),
                                                             q_scale * k_scale * m.attention_scale, f"{h}_s"),
                                  f"{h}_s")
                prob, prob_scale = keep(f"{h}_p", *self.softmax(f"{h}_p", s, s_scale), f"{h}_p")
                o, _ = self.requantize(f"{h}_o", matmul(prob, v), prob_scale * v_scale, f"{p}_proj_in",
                                       fixed_scale=True)
                attention[:, head * m.head_dim:(head + 1) * m.head_dim] = o
            attention, attention_scale = keep(f"{p}_o", attention, self.t[f"{p}_proj_in"], f"{p}_proj_in")
            proj, proj_scale = keep(f"{p}_proj", *self.linear(f"{p}_proj", attention, attention_scale,
                                                              f"{p}_proj_out"), f"{p}_proj_out")
            x, x_scale = keep(f"{p}_x_mid", *self.add(f"{p}_add1", x, x_scale, proj, proj_scale, f"{p}_x_mid"),
                              f"{p}_x_mid")
            n2, n2_scale = keep(f"{p}_ln2", *self.layernorm(f"{p}_ln2", x, x_scale, f"{p}_fc1_in"), f"{p}_fc1_in")
            extra = self.o["gelu_width"] - DATA_WIDTH if self.o["gelu"] != "float" else 0
            fc1, fc1_scale = self.linear(f"{p}_fc1", n2, n2_scale, f"{p}_fc1_out", extra_bits=extra)
            keep(f"{p}_fc1", fc1, fc1_scale, f"{p}_fc1_out")
            act, act_scale = keep(f"{p}_gelu", *self.gelu(f"{p}_gelu", fc1, fc1_scale, f"{p}_fc2_in"),
                                  f"{p}_fc2_in")
            fc2, fc2_scale = keep(f"{p}_fc2", *self.linear(f"{p}_fc2", act, act_scale, f"{p}_fc2_out"),
                                  f"{p}_fc2_out")
            x, x_scale = keep(f"{p}_x_out", *self.add(f"{p}_add2", x, x_scale, fc2, fc2_scale, f"{p}_x_out"),
                              f"{p}_x_out")

        # the 2-layer truncated model: final LayerNorm on the class row, then the classifier
        final, final_scale = keep("final_ln", *self.layernorm("final_ln", x[0:1], x_scale, "head_in"), "head_in")
        logits, _ = keep("head1000", *self.linear("head1000", final, final_scale, "head_out"), "head_out")
        keep("head10", *self.linear("head10", final, final_scale, "head_out"))
        return x, x_scale, logits


def error_stats(result, scale, real):
    """(max |error|, mean |error|) in output LSBs, and the cosine, against a real tensor."""
    error = np.abs(result - real / scale)
    return error.max(), error.mean(), cosine(result, real)


class Tally:
    """Collects per-instance error statistics of one op variant."""

    def __init__(self):
        self.max_error, self.mean_error, self.cosines = 0.0, [], []
        self.exact, self.count, self.max_step = 0, 0, 0

    def add(self, result, scale, real, ideal):
        """ideal: the float op on the same INT8 input, rounded to INT8."""
        max_error, mean_error, cos = error_stats(result, scale, real)
        self.max_error = max(self.max_error, max_error)
        self.mean_error.append(mean_error)
        self.cosines.append(cos)
        self.exact += int((result == ideal).sum())
        self.count += result.size
        self.max_step = max(self.max_step, int(np.abs(result - ideal).max()))

    def row(self, name):
        return (f"  {name:22s} {self.max_error:8.2f} {np.mean(self.mean_error):8.3f} {np.mean(self.cosines):9.5f} "
                f"{min(self.cosines):9.5f} {100.0 * self.exact / self.count:8.2f}% {self.max_step:5d}")


TALLY_HEADER = (f"  {'variant':22s} {'max err':>8s} {'mean err':>8s} {'cos mean':>9s} {'cos min':>9s} "
                f"{'= float':>9s} {'max':>5s}")


def per_op_report(parameters, captures):
    """Each op on its own real input, quantized with the target scale, over every image."""
    m, target = parameters, parameters.target
    print("\nPer-op error, each op fed its own real input (all images, all instances).")
    print("Errors are in output LSBs against the FP32 tensor. '= float' is the share of outputs equal to")
    print("the real function applied to the same INT8 input and rounded; 'max' is the largest difference from it.")

    def section(title):
        print(f"\n{title}\n{TALLY_HEADER}")

    section("Residual add (position, attention, MLP: 5 per image)")
    variants = [("float", None, None), ("8-bit mult, floor", 8, False), ("8-bit mult, round", 8, True),
                ("16-bit mult, round", 16, True)]
    tallies = {name: Tally() for name, _, _ in variants}
    gain_errors = []
    for fp in captures:
        cases = [(fp["embed_out"], target["embed_out"], m.position_real, m.position_scale, "L0_x_in", slice(1, None))]
        for layer in range(LAYERS):
            p = f"L{layer}"
            cases += [(fp[f"{p}_x_in"], target[f"{p}_x_in"], fp[f"{p}_proj_out"], target[f"{p}_proj_out"],
                       f"{p}_x_mid", slice(None)),
                      (fp[f"{p}_x_mid"], target[f"{p}_x_mid"], fp[f"{p}_fc2_out"], target[f"{p}_fc2_out"],
                       f"{p}_x_out", slice(None))]
        for a_real, a_scale, b_real, b_scale, out_key, rows in cases:
            a, b, real = quantize(a_real, a_scale), quantize(b_real, b_scale), fp[out_key][rows]
            ideal = quantize(a * a_scale + b * b_scale, target[out_key])
            for name, width, rounding in variants:
                if width is None:
                    tallies[name].add(ideal, target[out_key], real, ideal)
                    continue
                mult_a, mult_b, shift, effective, gain_error = add_constants(
                    a_scale, b_scale, target[out_key], (1 << (width - 1)) - 1)
                if width == 8:
                    gain_errors.append(abs(gain_error))
                result = residual_add(a, b, mult_a, mult_b, shift, rounding, width)
                # compare at the scale the integers really have
                tallies[name].add(result, effective, real, quantize(a * a_scale + b * b_scale, effective))
    for name, _, _ in variants:
        print(tallies[name].row(name))
    print(f"  8-bit multipliers: largest gain error of the less exact input {100 * max(gain_errors):.2f}%")

    section("GELU (2 per image)")
    tallies = {"float": Tally()}
    tallies.update({f"{width}-bit in, {1 << width} entries": Tally() for width in GELU_WIDTHS})
    for fp in captures:
        for layer in range(LAYERS):
            p = f"L{layer}"
            real, out_scale = fp[f"{p}_fc2_in"], target[f"{p}_fc2_in"]
            for width in GELU_WIDTHS:
                in_scale = target[f"{p}_fc1_out"] / (1 << (width - DATA_WIDTH))
                limit = 1 << (width - 1)
                x = np.clip(np.rint(fp[f"{p}_fc1_out"] / in_scale), -limit, limit - 1).astype(np.int64)
                ideal = quantize(GELU_REAL(x * in_scale), out_scale)
                if width == DATA_WIDTH:
                    tallies["float"].add(ideal, out_scale, real, ideal)
                tallies[f"{width}-bit in, {1 << width} entries"].add(
                    gelu(x, gelu_table(in_scale, out_scale, width)), out_scale, real, ideal)
    for name, tally in tallies.items():
        print(tally.row(name))

    widths = {config.name: {} for config in LAYERNORM_OPTIONS}
    for title, final_only in (("LayerNorm on 197 rows (ln1 and ln2 of each layer: 4 per image)", False),
                              ("LayerNorm on the class row (the final one: 1 per image)", True)):
        section(title)
        tallies = {"float": Tally()}
        tallies.update({config.name: Tally() for config in LAYERNORM_OPTIONS})
        for fp in captures:
            cases = [("final_ln", fp[f"L{LAYERS - 1}_x_out"][0:1], f"L{LAYERS - 1}_x_out", "head_in")]
            if not final_only:
                cases = []
                for layer in range(LAYERS):
                    p = f"L{layer}"
                    cases += [(f"{p}_ln1", fp[f"{p}_x_in"], f"{p}_x_in", f"{p}_qkv_in"),
                              (f"{p}_ln2", fp[f"{p}_x_mid"], f"{p}_x_mid", f"{p}_fc1_in")]
            for name, x_real, in_key, out_key in cases:
                gamma, beta, epsilon = m.norm[name]
                x, real = quantize(x_real, target[in_key]), fp[out_key]
                ideal = quantize(layernorm_real(x * target[in_key], gamma, beta, epsilon), target[out_key])
                tallies["float"].add(ideal, target[out_key], real, ideal)
                for config in LAYERNORM_OPTIONS:
                    gain, offset, out_shift, epsilon_int = layernorm_constants(gamma, beta, target[in_key],
                                                                              target[out_key], epsilon, config)
                    table = rsqrt_table(config) if config.rsqrt == "lut" else None
                    result = layernorm(x, gain, offset, out_shift, epsilon_int, config, table, widths[config.name])
                    tallies[config.name].add(result, target[out_key], real, ideal)
        for name, tally in tallies.items():
            print(tally.row(name))
    print("  largest magnitudes seen, in bits (sign bit not counted):")
    for config in LAYERNORM_OPTIONS:
        seen = "  ".join(f"{key} {int(value).bit_length()}" for key, value in sorted(widths[config.name].items()))
        print(f"    {config.name:22s} {seen}   table {config.table_bits()} bits")

    section("Softmax (3 heads per layer, 197 rows of 197)")
    tallies = {"float": Tally()}
    tallies.update({config.name: Tally() for config in SOFTMAX_OPTIONS})
    widths = {config.name: {} for config in SOFTMAX_OPTIONS}
    for fp in captures:
        for layer in range(LAYERS):
            for head in range(m.heads):
                h = f"L{layer}_h{head}"
                s, real = quantize(fp[f"{h}_s"], target[f"{h}_s"]), fp[f"{h}_p"]
                probability = softmax_real(s * target[f"{h}_s"])
                ideal = np.clip(np.rint(probability * DATA_MAX), 0, DATA_MAX).astype(np.int64)
                tallies["float"].add(ideal, 1.0 / DATA_MAX, real, ideal)
                for config in SOFTMAX_OPTIONS:
                    recip = reciprocal_table(config) if config.reciprocal == "lut" else None
                    gain = numerator_of(target[f"{h}_p"]) if config.calibrated else DATA_MAX << config.gain_log2
                    result = softmax(s, exp_table(target[f"{h}_s"], config), config, recip, widths[config.name],
                                     gain)
                    ideal_gain = np.clip(np.rint(probability * gain), 0, DATA_MAX).astype(np.int64)
                    tallies[config.name].add(result, 1.0 / gain, real, ideal_gain)
    for name, tally in tallies.items():
        print(tally.row(name))
    print("  largest magnitudes seen, in bits:")
    for config in SOFTMAX_OPTIONS:
        seen = "  ".join(f"{key} {int(value).bit_length()}" for key, value in sorted(widths[config.name].items()))
        print(f"    {config.name:22s} {seen}   tables {config.table_bits()} bits")


def end_to_end(parameters, captures, options):
    """Runs every image. Returns per-stage cosines, 1000-class and 10-class predictions and the last run."""
    stage_cosines, predictions, predictions10 = {}, [], []
    forward = None
    for fp in captures:
        forward = Forward(parameters, options)
        forward.run(fp)
        for name, tensor, _scale, key in forward.stages:
            if key is not None:
                stage_cosines.setdefault(name, []).append(cosine(tensor, fp[key]))
            if name == "head1000":
                predictions.append(int(np.argmax(tensor)))
            if name == "head10":
                predictions10.append(int(np.argmax(tensor)))
    return stage_cosines, predictions, predictions10, forward


def options_for(**changes):
    """All non-linear ops as float references, except the ones named."""
    options = dict(add="float", gelu="float", layernorm="float", softmax="float",
                   gelu_width=8, add_rounding=True, add_multiplier_width=8, matmul_rounding=False)
    options.update(changes)
    return options


def recommended_options(**changes):
    options = options_for(add="int", gelu="int",
                          layernorm=by_name(LAYERNORM_OPTIONS, RECOMMENDED["layernorm"]),
                          softmax=by_name(SOFTMAX_OPTIONS, RECOMMENDED["softmax"]),
                          gelu_width=RECOMMENDED["gelu_width"], add_rounding=RECOMMENDED["add_rounding"],
                          add_multiplier_width=RECOMMENDED["add_multiplier_width"])
    options.update(changes)
    return options


def end_to_end_report(parameters, captures, labels, names, calibrations, selected):
    """calibrations: name -> (target scales, matmul rounding). selected: the one that is dumped."""
    reference = [int(np.argmax(fp["head_out"])) for fp in captures]
    reference10 = [int(np.argmax(fp["head_out"][0, :10])) for fp in captures]
    full = [int(np.argmax(fp["logits_full"])) for fp in captures]
    count = len(captures)
    final = f"L{LAYERS - 1}_x_out"

    print(f"\nFP32 context, {count} images:")
    print(f"  full 12-layer model, top-1 correct: {sum(a == b for a, b in zip(full, labels))} of {count}")
    print(f"  2-layer truncated model, top-1 correct: {sum(a == b for a, b in zip(reference, labels))} of {count}; "
          f"it predicts {len(set(reference))} different class(es) over the {count} images")

    header = (f"  {'variant':36s} {'cos mean':>9s} {'cos min':>9s} {'final LN':>9s} {'logits':>8s} "
              f"{'top-1 /1000':>12s} {'top-1 /10':>10s}")

    def run(name, options):
        stage_cosines, predictions, predictions10, forward = end_to_end(parameters, captures, options)
        print(f"  {name:36s} {np.mean(stage_cosines[final]):9.5f} {min(stage_cosines[final]):9.5f} "
              f"{np.mean(stage_cosines['final_ln']):9.4f} {np.mean(stage_cosines['head1000']):8.4f} "
              f"{sum(a == b for a, b in zip(predictions, reference)):6d} of {count:2d} "
              f"{sum(a == b for a, b in zip(predictions10, reference10)):4d} of {count:2d}")
        return stage_cosines, predictions, forward

    print(f"\nEnd to end over {count} images. 'cos' is the layer {LAYERS} output (197x192) against FP32, 'final LN'")
    print("and 'logits' are the class row after the final LayerNorm and the 1000 logits. Top-1 is agreement with")
    print("the FP32 2-layer truncated model.")
    results = {}
    for calibration, (target, rounding) in calibrations.items():
        def float_ops(**changes):
            return options_for(target=target, matmul_rounding=rounding, **changes)

        print(f"\n{calibration}\n{header}")
        run("INT8 tensors, float non-linear ops", float_ops())
        run("add: 8-bit mult, floor", float_ops(add="int", add_rounding=False))
        run("add: 8-bit mult, round", float_ops(add="int"))
        run("add: 16-bit mult, round", float_ops(add="int", add_multiplier_width=16))
        for width in GELU_WIDTHS:
            run(f"gelu: {width}-bit in", float_ops(gelu="int", gelu_width=width))
        for config in LAYERNORM_OPTIONS:
            run(f"layernorm: {config.name}", float_ops(layernorm=config))
        for config in SOFTMAX_OPTIONS:
            run(f"softmax: {config.name}", float_ops(softmax=config))
        results[calibration] = run("ALL INTEGER, recommended", recommended_options(target=target,
                                                                                   matmul_rounding=rounding))

    stream = {key for key in parameters.target if "_x_" in key}
    percentile, maximum = (calibrations[name][0] for name in calibrations)
    print(f"\nWhat the calibration changes (all integer, recommended precision)\n{header}")
    for name, keys, rounding in (("percentile scales, floor", set(), False),
                                 ("percentile scales, round", set(), True),
                                 ("residual stream at max, floor", stream, False),
                                 ("residual stream at max, round", stream, True),
                                 ("all tensors at max, floor", set(maximum), False),
                                 ("all tensors at max, round", set(maximum), True)):
        target = {key: (maximum[key] if key in keys else percentile[key]) for key in percentile}
        run(name, recommended_options(target=target, matmul_rounding=rounding))

    print("\nALL INTEGER, recommended: cosine against FP32 at every stage, mean / worst over the images")
    print(f"  {'stage':12s} " + "   ".join(f"{name[:28]:>28s}" for name in calibrations))
    for stage in results[selected][0]:
        print(f"  {stage:12s} " + "   ".join(
            f"{np.mean(results[name][0][stage]):18.5f} {min(results[name][0][stage]):9.5f}" for name in calibrations))

    stage_cosines, predictions, forward = results[selected]
    print(f"\nPer image, {selected}, all integer (label / FP32 12-layer / FP32 2-layer / integer 2-layer):")
    for index, file_name in enumerate(names):
        print(f"  {file_name:36s} cos {stage_cosines[final][index]:.5f}  {labels[index]:4d} {full[index]:4d} "
              f"{reference[index]:4d} {predictions[index]:4d}")
    print(f"largest |accumulator| in the chained pass: {forward.acc_max} ({100.0 * forward.acc_max / 2 ** 31:.4f}% "
          f"of 2^31)")
    gains = [abs(c["gain_error"]) for c in forward.constants.values() if "gain_error" in c]
    print(f"requantizer and add multipliers: largest distance from the wanted ratio {100 * max(gains):.2f}%")
    print("LayerNorm and softmax intermediate magnitudes in the chained pass, in bits (sign not counted): "
          + ", ".join(f"{op} {key} {int(value).bit_length()}" for op in ("layernorm", "softmax")
                      for key, value in sorted(forward.widths[op].items())))


def dump(parameters, fp, test_name, calibration, rounding):
    """Per-stage tensors, tables and constants of the test image, recommended precision."""
    forward = Forward(parameters, recommended_options(matmul_rounding=rounding))
    forward.run(fp)
    os.makedirs(VEC_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(VEC_DIR, "*")):
        os.remove(stale)

    def write(name, values, width):
        write_hex(os.path.join(VEC_DIR, name), values, width)
        return name

    lines = [f"# {calibration}",
             f"# image {test_name}; layernorm {RECOMMENDED['layernorm']}; softmax {RECOMMENDED['softmax']}; "
             f"gelu {RECOMMENDED['gelu_width']}-bit; add {RECOMMENDED['add_multiplier_width']}-bit multipliers, "
             f"rounding {int(RECOMMENDED['add_rounding'])}",
             "# tensor <name> <rows> <cols> <file>      INT8, row-major",
             "# const  <op> <key> <value or file>       files: bias INT32, gain INT16, offset INT32,",
             "#                                         gelu table INT8, exp table 12-bit, rsqrt table 13-bit"]
    for name, tensor, _scale, _key in forward.stages:
        lines.append(f"tensor {name} {tensor.shape[0]} {tensor.shape[1]} {write(name + '.hex', tensor, DATA_WIDTH)}")
    config = by_name(LAYERNORM_OPTIONS, RECOMMENDED["layernorm"])
    lines.append(f"const layernorm rsqrt_table {write('rsqrt_table.hex', rsqrt_table(config), config.rsqrt_bits + 1)}")
    file_widths = {"bias": ACC_WIDTH, "gain": 16, "offset": 32, "table": DATA_WIDTH,
                   "exp_table": by_name(SOFTMAX_OPTIONS, RECOMMENDED["softmax"]).exp_bits}
    for op, constants in forward.constants.items():
        for key, value in constants.items():
            if value is None or key == "gain_error":
                continue
            if isinstance(value, np.ndarray):
                value = write(f"{op}_{key}.hex", value, file_widths[key])
            lines.append(f"const {op} {key} {value}")
    with open(os.path.join(VEC_DIR, "forward_ops.txt"), "w") as manifest:
        manifest.write("\n".join(lines) + "\n")

    # the weights are the ones gen_deit_vectors.py already wrote
    for name, (w_q, _scale, _bias) in parameters.linear.items():
        existing = read_hex(os.path.join(DEIT_DIR, f"{name}_B.hex"), DATA_WIDTH)
        assert np.array_equal(existing, w_q.reshape(-1)), name
    print(f"\nwrote {len(os.listdir(VEC_DIR))} files to {VEC_DIR} "
          f"(weights are the *_B.hex files in {os.path.relpath(DEIT_DIR, GOLDEN_DIR)}, checked equal)")


def main():
    torch.manual_seed(0)
    model = timm.create_model(MODEL_NAME, pretrained=True).eval()
    for block in model.blocks:
        block.attn.fused_attn = False
    cfg = model.pretrained_cfg
    files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.JPEG")))
    names = [os.path.basename(file_name) for file_name in files]
    synsets = timm.data.ImageNetInfo("imagenet-1k").label_names()
    labels = [synsets.index(name.split("_")[0]) for name in names]

    captures = [capture_all(model, load_image(file_name, cfg)) for file_name in files]
    # the same calibration as gen_deit_vectors.py: every image but the first
    percentile, maximum = {}, {}
    for key in captures[0]:
        values = np.concatenate([np.abs(fp[key]).reshape(-1) for fp in captures[1:]])
        percentile[key] = float(np.percentile(values, ACTIVATION_PERCENTILE)) / DATA_MAX
        maximum[key] = float(values.max()) / DATA_MAX
    percentile["head_out"] = maximum["head_out"]   # nothing may clip before the argmax
    calibrations = {
        f"A: {ACTIVATION_PERCENTILE} percentile scales, requantizer floors": (percentile, False),
        "B: max scales, half an LSB added to every matmul bias": (maximum, True),
    }
    selected = list(calibrations)[1 if "--max" in sys.argv else 0]
    target, rounding = calibrations[selected]
    parameters = Model(model, target)

    position = model.pos_embed.detach().numpy().astype(np.float64)[0]
    assert np.allclose(captures[0]["embed_out"] + position[1:], captures[0]["L0_x_in"][1:], atol=1e-4)
    assert np.allclose(parameters.class_row, captures[0]["L0_x_in"][0:1], atol=1e-4)

    print(f"model {MODEL_NAME}, {len(files)} images, scales from images 2..{len(files)}, test image {names[0]}")
    print(f"per-op report and dump use calibration {selected}")
    per_op_report(parameters, captures)
    end_to_end_report(parameters, captures, labels, names, calibrations, selected)
    if "--no-dump" not in sys.argv:
        dump(parameters, captures[0], names[0], selected, rounding)


if __name__ == "__main__":
    main()
