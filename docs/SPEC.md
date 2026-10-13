# Spec

A SystemVerilog accelerator for the first two encoder layers of the pretrained
DeiT-Tiny vision transformer (ViT-Ti/16), bit-exact against an integer golden
model. Everything lives in block RAM; there is no external memory.

## Model

- Official pretrained DeiT-Tiny: embedding width 192, 3 heads of 64, MLP
  hidden 768, 16 x 16 patches. Layers 0 and 1 of 12, then the final LayerNorm
  and a linear classifier.
- INT8 tensors everywhere, residual stream included; INT32 accumulation. One
  scale per tensor, taken from calibration data and kept in the repo
  (`python/scales_*.json`).
- Requantizer: `out = clip((acc * scale) >>> shift)` with an 8-bit scale and a
  5-bit shift. Half an LSB in the bias makes it round to nearest.
- Non-linear ops in integer arithmetic: residual add with two multipliers,
  GELU by a 256-entry table, LayerNorm with a reciprocal-square-root table,
  softmax with an exp table and one division per row.

## Configurations

| | DeiT-224 | MNIST-112 |
|---|---|---|
| Input | 224 x 224, ImageNet images | 112 x 112, MNIST digits resized by 4 |
| Tokens | 197 | 50 (position embedding resampled) |
| Classifier | the model's own head on the class row | linear probe on every row, summed over the patch rows by the host |
| Weights in the engine | one layer at a time | one group of matmuls at a time; fc1, fc2 and the patch embedding in two column chunks |
| Descriptor program | 57 ops, 4 phases | 62 ops, 13 phases |
| Fits | XC7A200T (326 of 365 block RAM tiles) | XC7A100T (94 of 135) |
| Used for | simulation | the Arty A7-100T |

## Architecture

- **Engine** (`rtl/core`, frozen): an 8 x 8 output-stationary systolic array,
  one DSP48E1 per element; a pool of eight operand buffers in block RAM; a
  drain with bias add and requantizer; row units for the four non-linear ops;
  a sequencer that runs a table of op descriptors. An op names its input and
  output buffers, so nothing is copied between ops. See [ENGINE.md](ENGINE.md).
- **Host-loaded weights.** The engine holds no weights of its own. Before each
  phase of the program the host writes that phase's weights, biases and
  LayerNorm constants, starts it and waits. The GELU, exp and
  reciprocal-square-root tables are ROMs built into the bitstream
  (`rtl/rom/<configuration>/`).
- **AXI wrapper** (`rtl/interface`): AXI4-Lite for control and status and a
  64-bit AXI4-Stream pair to load any memory and read a buffer out. See
  [AXI_REGISTER_MAP.md](AXI_REGISTER_MAP.md).
- **Host link.** A byte protocol with checksums ([HOST_PROTOCOL.md](HOST_PROTOCOL.md)),
  bridged to the AXI wrapper. A board supplies only the byte transport: a UART
  on the Arty A7, FrontPanel pipes on the XEM7310 later.
- **One clock domain**, synchronous reset. 100 MHz on the Arty A7.

## Boards

- **Digilent Arty A7-100T** (XC7A100T): the MNIST-112 configuration over the
  USB-UART. See [ARTY_QUICKSTART.md](ARTY_QUICKSTART.md).
- **Opal Kelly XEM7310-A200** (XC7A200T): not built yet.

## Verification

Every module has a self-checking testbench against golden vectors that
`scripts/gen_vectors.sh` generates. The whole forward pass runs as one
descriptor program and every stage is compared with the golden model, in both
configurations; the host software's byte stream is replayed against the RTL.

## Not in scope

External memory (DDR3), a DMA, more than two encoder layers, training.
