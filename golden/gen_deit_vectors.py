"""Capture of the real pretrained DeiT-Tiny (ViT-Ti/16, 224x224) and the vectors made from it.

Usage: .venv/bin/python golden/gen_deit_vectors.py     (needs torch and timm)

This is the same generator as golden/gen_deit_forward.py: the 39 matmuls in
golden/vectors/deit/ are the engine ops of the chained integer forward pass, so
both vector directories come from one run. See that file for the scales, the
rounding and the file formats.

This module holds what the capture needs: the image preprocessing, the hooks
that grab the FP32 input and output of every matmul of encoder layers 0 and 1,
the patch embedding and the classifier head, and the INT8 quantizer.

Tensors returned by capture(), FP32:
  embed_in, embed_out          patch embedding, host im2col  196x768 @ 768x192
  L<l>_qkv_in                  input of Q, K and V           197x192
  L<l>_h<h>_q, _k, _v          per head, a slice of qkv      197x64
  L<l>_h<h>_s, _p              S = Q @ K^T / 8 and softmax   197x197
  L<l>_h<h>_o                  O = P @ V                     197x64
  L<l>_proj, _fc1, _fc2 _in, _out
  head_in                      class row after the final LayerNorm of the 2-layer model
"""

import os

import numpy as np
import timm
import torch
from PIL import Image

from int8_ops import DATA_MAX, DATA_MIN

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(GOLDEN_DIR, "images")

MODEL_NAME = "deit_tiny_patch16_224.fb_in1k"
LAYERS = 2


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


if __name__ == "__main__":
    import gen_deit_forward
    gen_deit_forward.main()
