"""Golden vectors from the real pretrained DeiT-Tiny (ViT-Ti/16, 224x224).

Usage: .venv/bin/python golden/gen_deit_vectors.py     (needs torch and timm)
Output: golden/vectors/deit/*.hex and deit_ops.txt, for tb_matmul_engine_deit.

One real image goes through the FP32 model. Hooks capture the real input of
every matmul in encoder layers 0 and 1, the patch embedding and the classifier
head. Every input is quantized to INT8 with a per-tensor scale taken from a
small calibration set of other real images. Each op's golden INT8 result is
then built with the hardware integer model in int8_ops.py: matmul, optional
INT32 bias, requantize with one scale and shift per op, chosen so the largest
real accumulator lands on 127.

Ops, each independent and fed its own real input:
  embed                        patch embedding, host im2col  196x768 @ 768x192
  L<l>_h<h>_q, _k, _v          per head, a slice of qkv      197x192 @ 192x64
  L<l>_h<h>_s                  S = Q @ K^T                   197x64  @ 64x197
  L<l>_h<h>_o                  O = P @ V, P = real softmax   197x197 @ 197x64
  L<l>_proj, _fc1, _fc2        197x192 @ 192x192, 197x192 @ 192x768, 197x768 @ 768x192
  head10, head1000             classifier on the CLS row     1x192 @ 192xN

Files per op (INT8 unless noted, matrices row-major):
  <tensor>_A.hex   activation operand A[row][k]; shared by ops with the same input
  <op>_B.hex       weight-side operand B[k][col]
  <op>_bias.hex    INT32 bias[col], only for ops with a bias
  <op>_Y.hex       golden requantized result
  <op>_desc_r8c8.hex   one-op descriptor table for an 8x8 array
deit_ops.txt has one line per op: name, group, A file, B file, bias file or "-",
Y file, descriptor file.
"""

import glob
import os

import numpy as np
import timm
import torch
from PIL import Image

from gen_vectors import pick_scale_shift
from int8_ops import (ACC_MAX, ACC_MIN, ACC_WIDTH, DATA_MAX, DATA_MIN, DATA_WIDTH,
                      add_bias, matmul_acc, requantize, write_hex)
from pkg_config import PARAMS, pack_descriptor

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(GOLDEN_DIR, "images")
VEC_DIR = os.path.join(GOLDEN_DIR, "vectors", "deit")

MODEL_NAME = "deit_tiny_patch16_224.fb_in1k"
LAYERS = 2
SYS_ROW = SYS_COL = 8
BANK_DEPTH = PARAMS["BANK_DEPTH_MAX"]  # tb_matmul_engine_deit gives every pool buffer this depth

# activation scale = this percentile of |x| over the calibration images / 127
ACTIVATION_PERCENTILE = 99.99
ROW_SELECT, COLUMN_SELECT = PARAMS["DRAIN_ROW_SELECT"], PARAMS["DRAIN_COLUMN_SELECT"]


def load_image(file_name, cfg):
    """The model's eval preprocessing: resize, centre crop, normalise."""
    size = cfg["input_size"][-1]
    resize = int(size / cfg["crop_pct"])
    image = Image.open(file_name).convert("RGB")
    scale = resize / min(image.size)
    image = image.resize((max(resize, round(image.width * scale)), max(resize, round(image.height * scale))),
                         Image.BICUBIC)
    left, top = (image.width - size) // 2, (image.height - size) // 2
    pixels = np.asarray(image.crop((left, top, left + size, top + size)), dtype=np.float32) / 255.0
    pixels = (pixels - np.array(cfg["mean"], dtype=np.float32)) / np.array(cfg["std"], dtype=np.float32)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0)


def capture(model, image):
    """Run one image and return the FP32 inputs and outputs of every matmul."""
    grabbed = {}

    def grab(name):
        def hook(_module, inputs, output):
            grabbed[name + "_in"] = inputs[0].detach()[0].numpy().astype(np.float64)
            grabbed[name + "_out"] = output.detach()[0].numpy().astype(np.float64)
        return hook

    hooks = [model.patch_embed.proj.register_forward_hook(grab("conv"))]
    for layer in range(LAYERS):
        block = model.blocks[layer]
        hooks += [block.attn.qkv.register_forward_hook(grab(f"L{layer}_qkv")),
                  block.attn.attn_drop.register_forward_hook(grab(f"L{layer}_softmax")),
                  block.attn.proj.register_forward_hook(grab(f"L{layer}_proj")),
                  block.mlp.fc1.register_forward_hook(grab(f"L{layer}_fc1")),
                  block.mlp.fc2.register_forward_hook(grab(f"L{layer}_fc2")),
                  block.register_forward_hook(grab(f"L{layer}_block"))]
    with torch.no_grad():
        model(image)
        # the head of the 2-layer truncated model: final norm on the layer 1 output, CLS row
        block_out = torch.from_numpy(grabbed[f"L{LAYERS - 1}_block_out"]).float()
        grabbed["head_in"] = model.norm(block_out)[0:1].numpy().astype(np.float64)
    for hook in hooks:
        hook.remove()

    embed, heads, head_dim = model.embed_dim, model.blocks[0].attn.num_heads, model.blocks[0].attn.head_dim
    tensors = {}

    # patch embedding as a matmul: host im2col, element order (channel, row, column)
    patch = model.patch_embed.patch_size[0]
    grid = image.shape[-1] // patch
    pixels = image[0].numpy().astype(np.float64)
    tensors["embed_in"] = (pixels.reshape(3, grid, patch, grid, patch).transpose(1, 3, 0, 2, 4)
                           .reshape(grid * grid, 3 * patch * patch))
    tensors["embed_out"] = grabbed["conv_out"].reshape(embed, grid * grid).T

    for layer in range(LAYERS):
        qkv = grabbed[f"L{layer}_qkv_out"]
        tensors[f"L{layer}_qkv_in"] = grabbed[f"L{layer}_qkv_in"]
        for head in range(heads):
            q, k, v = (qkv[:, part * embed + head * head_dim:part * embed + (head + 1) * head_dim]
                       for part in range(3))
            p = grabbed[f"L{layer}_softmax_in"][head]
            tensors[f"L{layer}_h{head}_q"] = q
            tensors[f"L{layer}_h{head}_k"] = k
            tensors[f"L{layer}_h{head}_v"] = v
            tensors[f"L{layer}_h{head}_s"] = (q @ k.T) * model.blocks[layer].attn.scale
            tensors[f"L{layer}_h{head}_p"] = p
            tensors[f"L{layer}_h{head}_o"] = grabbed[f"L{layer}_proj_in"][:, head * head_dim:(head + 1) * head_dim]
            # the captured tensors must be what the matmul view of the model computes
            scores = tensors[f"L{layer}_h{head}_s"]
            softmax = np.exp(scores - scores.max(axis=1, keepdims=True))
            assert np.allclose(softmax / softmax.sum(axis=1, keepdims=True), p, atol=1e-4)
            assert np.allclose(p @ v, tensors[f"L{layer}_h{head}_o"], atol=1e-3)
        for name in ("proj", "fc1", "fc2"):
            tensors[f"L{layer}_{name}_in"] = grabbed[f"L{layer}_{name}_in"]
            tensors[f"L{layer}_{name}_out"] = grabbed[f"L{layer}_{name}_out"]
    tensors["head_in"] = grabbed["head_in"]

    weight = model.patch_embed.proj.weight.detach().numpy().astype(np.float64).reshape(embed, -1)
    assert np.allclose(tensors["embed_in"] @ weight.T + model.patch_embed.proj.bias.detach().numpy(),
                       tensors["embed_out"], atol=1e-3)
    return tensors


def quantize(tensor, scale):
    return np.clip(np.rint(tensor / scale), DATA_MIN, DATA_MAX).astype(np.int64)


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def per_column_result(a_q, a_scale, weight, bias):
    """The same op with one weight scale and one requant scale/shift per output column.

    The hardware has one scale and shift per op, so this is a study only: it
    tells how much accuracy per-column scales would add.
    """
    column_scale = np.maximum(np.abs(weight).max(axis=0), 1e-12) / DATA_MAX
    w_q = np.clip(np.rint(weight / column_scale), DATA_MIN, DATA_MAX).astype(np.int64)
    acc = a_q @ w_q + np.rint(bias / (a_scale * column_scale)).astype(np.int64)
    real = acc * a_scale * column_scale
    multiplier = a_scale * column_scale / (np.abs(real).max() / DATA_MAX)   # per column, to one output scale
    shift = np.clip(np.floor(np.log2(DATA_MAX / multiplier)), 0, 31).astype(np.int64)
    scale = np.clip(np.rint(multiplier * 2.0 ** shift), 1, DATA_MAX).astype(np.int64)
    return requantize(np.clip(acc, ACC_MIN, ACC_MAX), scale[None, :], shift[None, :])


class Op:
    """One engine op with its INT8 operands, golden result and statistics."""

    def __init__(self, name, group, a_key, a_q, b_q, bias_q, fp_out, drain_mode,
                 output_base=0, output_stride=None, per_column=None):
        self.name, self.group, self.a_key = name, group, a_key
        self.a_q, self.b_q, self.bias_q = a_q, b_q, bias_q
        self.rows, self.inner = a_q.shape
        self.cols = b_q.shape[1]
        self.drain_mode, self.output_base = drain_mode, output_base
        natural = self.cols if drain_mode == COLUMN_SELECT else self.rows
        self.output_stride = natural if output_stride is None else output_stride

        acc = matmul_acc(a_q, b_q)
        self.acc_max = int(np.abs(acc).max())
        # no partial sum can exceed this, whatever order the array adds in
        self.acc_bound = int((np.abs(a_q) @ np.abs(b_q)).max())
        if bias_q is not None:
            acc = add_bias(acc, bias_q)
            self.acc_max = max(self.acc_max, int(np.abs(acc).max()))
            self.acc_bound += int(np.abs(bias_q).max())
        self.scale, self.shift = pick_scale_shift(acc, DATA_MAX)
        self.y = requantize(acc, self.scale, self.shift)
        self.saturated = float(np.mean((self.y == DATA_MIN) | (self.y == DATA_MAX))) * 100.0
        self.cosine = cosine(self.y, fp_out)
        self.cosine_per_column = None if per_column is None else cosine(per_column, fp_out)

    def descriptor(self):
        row_tiles, col_tiles = -(-self.rows // SYS_ROW), -(-self.cols // SYS_COL)
        bank_rows = row_tiles if self.drain_mode == COLUMN_SELECT else col_tiles
        assert row_tiles * self.inner <= BANK_DEPTH and col_tiles * self.inner <= BANK_DEPTH, self.name
        assert self.output_base + bank_rows * self.output_stride <= BANK_DEPTH, self.name
        assert col_tiles <= PARAMS["BIAS_DEPTH"], self.name
        return pack_descriptor(
            opcode=PARAMS["OPCODE_MATMUL"],
            rows_total=self.rows, inner_dim=self.inner, cols_total=self.cols,
            row_tiles=row_tiles, col_tiles=col_tiles,
            activation_buffer_id=0, weight_buffer_id=1, output_buffer_id=2,
            activation_base=0, weight_base=0,
            output_base=self.output_base, output_stride=self.output_stride,
            bias_base=0, scale=self.scale, shift=self.shift, drain_mode=self.drain_mode,
            bias_enable=int(self.bias_q is not None), last_op=1)


def main():
    torch.manual_seed(0)
    model = timm.create_model(MODEL_NAME, pretrained=True).eval()
    for block in model.blocks:
        block.attn.fused_attn = False  # so the softmax output passes through attn_drop
    cfg = model.pretrained_cfg
    embed, heads, head_dim = model.embed_dim, model.blocks[0].attn.num_heads, model.blocks[0].attn.head_dim

    files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.JPEG")))
    assert len(files) >= 2, f"need real images in {IMAGE_DIR}"
    test_file, calibration_files = files[0], files[1:]

    # per-tensor activation scales from the calibration images
    samples = {}
    for file_name in calibration_files:
        for key, tensor in capture(model, load_image(file_name, cfg)).items():
            samples.setdefault(key, []).append(np.abs(tensor).reshape(-1))
    act_scale = {key: float(np.percentile(np.concatenate(values), ACTIVATION_PERCENTILE)) / DATA_MAX
                 for key, values in samples.items()}

    fp = capture(model, load_image(test_file, cfg))
    q = {key: quantize(fp[key], act_scale[key]) for key in fp}

    def linear_op(name, group, a_key, weight, bias, fp_out, drain_mode, **layout):
        """weight is (K, N) FP32, quantized with one scale for the whole tensor."""
        weight_scale = np.abs(weight).max() / DATA_MAX
        w_q = np.clip(np.rint(weight / weight_scale), DATA_MIN, DATA_MAX).astype(np.int64)
        bias_q = np.rint(bias / (act_scale[a_key] * weight_scale)).astype(np.int64)
        assert ACC_MIN <= bias_q.min() and bias_q.max() <= ACC_MAX, name
        per_column = per_column_result(q[a_key], act_scale[a_key], weight, bias)
        return Op(name, group, a_key, q[a_key], w_q, bias_q, fp_out, drain_mode, per_column=per_column, **layout)

    def parameters(linear):
        return (linear.weight.detach().numpy().astype(np.float64).T,
                linear.bias.detach().numpy().astype(np.float64))

    ops = []
    weight = model.patch_embed.proj.weight.detach().numpy().astype(np.float64).reshape(embed, -1).T
    ops.append(linear_op("embed", "embed", "embed_in", weight,
                         model.patch_embed.proj.bias.detach().numpy().astype(np.float64),
                         fp["embed_out"], COLUMN_SELECT))

    for layer in range(LAYERS):
        group, block = f"L{layer}", model.blocks[layer]
        qkv_weight, qkv_bias = parameters(block.attn.qkv)
        for head in range(heads):
            for part, letter, mode in ((0, "q", COLUMN_SELECT), (1, "k", COLUMN_SELECT), (2, "v", ROW_SELECT)):
                col = part * embed + head * head_dim
                ops.append(linear_op(f"L{layer}_h{head}_{letter}", group, f"L{layer}_qkv_in",
                                     qkv_weight[:, col:col + head_dim], qkv_bias[col:col + head_dim],
                                     fp[f"L{layer}_h{head}_{letter}"], mode))
        for head in range(heads):
            prefix = f"L{layer}_h{head}"
            # S = Q @ K^T: both operands are INT8 activations, the 1/sqrt(64) is part of the scale and shift
            ops.append(Op(f"{prefix}_s", group, f"{prefix}_q", q[f"{prefix}_q"], q[f"{prefix}_k"].T.copy(),
                          None, fp[f"{prefix}_s"], COLUMN_SELECT))
        for head in range(heads):
            prefix = f"L{layer}_h{head}"
            # O = P @ V with P the real softmax output as 0..127; O lands in its slot of the 197 x 192 tensor
            p_q = np.clip(np.rint(fp[f"{prefix}_p"] * DATA_MAX), 0, DATA_MAX).astype(np.int64)
            q[f"{prefix}_p"] = p_q
            ops.append(Op(f"{prefix}_o", group, f"{prefix}_p", p_q, q[f"{prefix}_v"],
                          None, fp[f"{prefix}_o"], COLUMN_SELECT,
                          output_base=head * head_dim, output_stride=embed))
        for name, linear in (("proj", block.attn.proj), ("fc1", block.mlp.fc1), ("fc2", block.mlp.fc2)):
            weight, bias = parameters(linear)
            ops.append(linear_op(f"L{layer}_{name}", group, f"L{layer}_{name}_in", weight, bias,
                                 fp[f"L{layer}_{name}_out"], COLUMN_SELECT))

    # classifier on the CLS row of the 2-layer model; N = 10 uses the first ten ImageNet classes
    head_weight, head_bias = parameters(model.head)
    for classes in (10, 1000):
        ops.append(linear_op(f"head{classes}", "head", "head_in", head_weight[:, :classes], head_bias[:classes],
                             fp["head_in"] @ head_weight[:, :classes] + head_bias[:classes], COLUMN_SELECT))

    os.makedirs(VEC_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(VEC_DIR, "*")):
        os.remove(stale)
    written = set()
    with open(os.path.join(VEC_DIR, "deit_ops.txt"), "w") as manifest:
        for op in ops:
            a_file = f"{op.a_key}_A.hex"
            if a_file not in written:
                write_hex(os.path.join(VEC_DIR, a_file), op.a_q, DATA_WIDTH)
                written.add(a_file)
            write_hex(os.path.join(VEC_DIR, f"{op.name}_B.hex"), op.b_q, DATA_WIDTH)
            write_hex(os.path.join(VEC_DIR, f"{op.name}_Y.hex"), op.y, DATA_WIDTH)
            write_hex(os.path.join(VEC_DIR, f"{op.name}_desc_r{SYS_ROW}c{SYS_COL}.hex"), op.descriptor(), 32)
            bias_file = "-"
            if op.bias_q is not None:
                bias_file = f"{op.name}_bias.hex"
                write_hex(os.path.join(VEC_DIR, bias_file), op.bias_q, ACC_WIDTH)
            manifest.write(f"{op.name} {op.group} {a_file} {op.name}_B.hex {bias_file} {op.name}_Y.hex "
                           f"{op.name}_desc_r{SYS_ROW}c{SYS_COL}.hex\n")

    print(f"model {MODEL_NAME}, test image {os.path.basename(test_file)}, "
          f"{len(calibration_files)} calibration images, activation scale = {ACTIVATION_PERCENTILE} percentile")
    print(f"{'op':12s} {'shape':>18s} {'max|acc|':>10s} {'of 2^31':>9s} {'sum bound':>10s} "
          f"{'sat %':>6s} {'cos tensor':>10s} {'cos column':>10s}")
    for op in ops:
        shape = f"{op.rows}x{op.inner}@{op.inner}x{op.cols}"
        column = "" if op.cosine_per_column is None else f"{op.cosine_per_column:.5f}"
        print(f"{op.name:12s} {shape:>18s} {op.acc_max:10d} {100.0 * op.acc_max / 2 ** 31:8.4f}% "
              f"{op.acc_bound:10d} {op.saturated:6.3f} {op.cosine:10.5f} {column:>10s}")
    weighted = [op for op in ops if op.cosine_per_column is not None]
    print(f"largest max|acc|: {max(op.acc_max for op in ops)}, largest partial-sum bound: "
          f"{max(op.acc_bound for op in ops)}, 2^31 = {2 ** 31}")
    print(f"ops with weights: mean cosine per-tensor {np.mean([op.cosine for op in weighted]):.5f}, "
          f"per-column {np.mean([op.cosine_per_column for op in weighted]):.5f}; "
          f"worst per-tensor {min(op.cosine for op in weighted):.5f}, "
          f"worst per-column {min(op.cosine_per_column for op in weighted):.5f}")
    print(f"wrote {len(os.listdir(VEC_DIR))} files to {VEC_DIR}")


if __name__ == "__main__":
    main()
