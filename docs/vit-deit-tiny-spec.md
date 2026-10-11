# ViT (DeiT-Tiny) FPGA Accelerator — Project Spec (v2, restarted Oct 11)

## Goal
Fully Verilog/SystemVerilog implementation of a ViT encoder on the Opal Kelly XEM7310-A200 (Artix-7 XC7A200T, 740 DSPs). **Deadline: Nov 2, 2026.** No report needed. RTL completeness first; per-module golden-vector checks are how we find bugs early.

## STATUS: RESTARTED FROM SCRATCH (Oct 11)
Nothing is built. The earlier weight-stationary-style systolic array / requantizer plan is dropped. Design is now an **output-stationary (OS) systolic array**, fed from DDR3.

## Scope (unchanged)
- Official pretrained **DeiT-Tiny (ViT-Ti/16)**, 224x224, **2 of 12 encoder layers**, embed dim 192, **197 tokens** (196 patches + CLS). Standard DeiT-Tiny: 3 heads x 64, MLP hidden 768.
- INT8 everywhere incl. residuals, INT32 accumulate. No training, no custom model, no borrowed weights.
- Requantizer: `out = clip((acc * scale) >>> shift)` to int8; multiply registered 1 cycle; DATA_WIDTH=8, ACC_WIDTH=32, SHIFT_WIDTH=$clog2(ACC_WIDTH), SCALE_WIDTH=8. Applied at SA drain.
- Standard fixed-point softmax, LUT GELU, fixed-point LayerNorm (no I-ViT Shiftmax/ShiftGELU).
- Classifier: linear probe on frozen 2-layer CLS row (1x192) x (192xN), argmax; reuses SA + requant.
- Demo: (a) 2-layer intermediate activations bit-exact vs a 2-layer-truncated PyTorch/golden model; (b) MNIST test case on the hardware.
- Host comms: Opal Kelly FrontPanel (WireIn/WireOut/PipeIn/PipeOut). AXI4 only for DDR3 via MIG.

## Architecture (data path)
Host (FrontPanel PipeIn) -> DDR3 (via MIG, AXI4) -> **DMA read engine** (burst reads) -> **BRAM ping-pong tile buffers** (A tile, B tile) -> **skew feeders** -> **OS systolic array** (acc stays in PEs) -> drain -> **requantizer** -> **output BRAM buffer** -> **DMA write engine** -> DDR3 -> host (PipeOut).
- Rule: the SA is never fed directly from DDR. Compute tile t while prefetching tile t+1 (double buffering).
- Single main clock domain = MIG ui_clk for DMA + SA + all ops; the only CDC is FrontPanel okClk <-> ui_clk (async FIFOs).
- Weights are stored in DDR **pre-tiled in the exact order the SA consumes them** (host/Python generates the DDR image), so reads are long linear bursts.
- OS SA: PE(i,j) holds C[i][j] accumulator; A streams left->right (row i skewed by i cycles), B streams top->bottom (col j skewed by j cycles); loop the **full K dimension per output tile** (no partial-sum spill); then drain + requant once. Tile loops over M, N outer, K inner. Array size R x C is a parameter.

## Open decisions (resolve early)
1. Patch embedding + CLS + pos-embed: compute on host/golden and DMA the 197x192 INT8 token matrix in (recommended for time), or run patch-embed as a 196x768 @ 768x192 matmul on the SA.
2. SA tile size R x C (bandwidth needed = R + C bytes/cycle at INT8; check against MIG/BRAM-port limits).
3. Target clock (start conservative, e.g. 100 MHz, raise later).
4. DDR3 data width / MIG config for XEM7310-A200 — confirm from Opal Kelly docs and MIG settings, don't assume.

## Schedule (Oct 11 -> Nov 2)
- Oct 11-12: golden model (INT8, per-stage `.hex` dumps), repo + Claude Code setup, OS SA design.
- Oct 13-15: OS SA (PE, array, skew, drain, requant), multi-tile matmul bit-exact in sim.
- Oct 14-17 (parallel): **MIG + FrontPanel loopback on the real board** (host -> DDR3 -> host).
- Oct 16-18: DMA engines, ping-pong buffers, tiled controller; one matmul from DDR bit-exact.
- Oct 19-21: LayerNorm, softmax, GELU sims bit-exact.
- **Oct 21 HARD CHECKPOINT:** one matmul from DDR3 bit-exact on the board + LN/softmax/GELU passing sims. If missed -> switch to BRAM-only reduced fallback.
- Oct 22-25: QKV -> attention -> proj -> MLP -> residual = full layer 1, sequencer FSM, bit-exact.
- Oct 26-28: layer 2, classifier head, MNIST path.
- Oct 29-Nov 1: on-board bring-up, fixes, demo. Nov 2 deadline.

## Fallback
BRAM-only reduced demo (single layer or the smaller MNIST-only ViT) if the Oct 21 checkpoint is missed.
