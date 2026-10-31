# Changelog

## 1.0.0 (2026-10-31)

The first release.

### What it contains

- An INT8 engine for the encoder layers of the pretrained DeiT-Tiny: an 8 x 8
  systolic array, GELU, LayerNorm, softmax and the residual add, driven by a
  descriptor program. It runs the first N of the 12 layers, N from 1 to 12, with
  the same bitstream.
- A Nexys A7-100T build at 100 MHz: weights in DDR2, loaded by an AXI DMA, with
  the host PC on the USB-UART and one `RUN` command per image.
- A Python integer golden model, a model generator, and host software that can
  also run against a software model of the board.
- One Vivado project, `vivado/vit.xpr`: the Nexys A7 build and one simulation set
  per self-checking testbench.
- A 224 x 224 configuration for the Opal Kelly XEM7310, simulated only.

### Results

- RTL against the golden model: bit-exact, 27 of 27 simulation sets pass.
- Nexys A7 build: timing met (WNS +0.386 ns, WHS +0.021 ns), 112 of 135 block RAM
  tiles.
- MNIST demo (2,000 test digits): 90.4 % with 6 layers, 81.8 % with 12.

### Known limitations

- INT8 top-1 accuracy against FP32 has not been measured on ImageNet. Agreement
  with FP32 falls with depth: mean cosine 0.964 after 6 layers, 0.928 at the
  final LayerNorm after 12.
- The Opal Kelly configuration has not been built.
