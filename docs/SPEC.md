# Spec

A SystemVerilog accelerator for the first two encoder layers of the pretrained
DeiT-Tiny vision transformer (ViT-Ti/16), bit-exact against an integer golden
model. The engine works out of block RAM. The weights are moving to external
DDR memory, copied into the engine by DMA: see
[External memory and DMA](#external-memory-and-dma-planned), not built yet.

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
| Used for | simulation | the Nexys A7-100T |

## Architecture

- **Engine** (`rtl/core`, frozen): an 8 x 8 output-stationary systolic array,
  one DSP48E1 per element; a pool of eight operand buffers in block RAM; a
  drain with bias add and requantizer; row units for the four non-linear ops;
  a sequencer that runs a table of op descriptors. An op names its input and
  output buffers, so nothing is copied between ops. See [ENGINE.md](ENGINE.md).
- **Host-loaded weights.** The engine holds no weights of its own. Before each
  phase of the program the host writes that phase's weights, biases and
  LayerNorm constants, starts it and waits. Today every load crosses the host
  link; with external memory the loads come from DDR by DMA. The GELU, exp and
  reciprocal-square-root tables are ROMs built into the bitstream
  (`rtl/rom/<configuration>/`).
- **AXI wrapper** (`rtl/interface`): AXI4-Lite for control and status and a
  64-bit AXI4-Stream pair to load any memory and read a buffer out. See
  [AXI_REGISTER_MAP.md](AXI_REGISTER_MAP.md).
- **Host link.** A byte protocol with checksums ([HOST_PROTOCOL.md](HOST_PROTOCOL.md)),
  bridged to the AXI wrapper. A board supplies only the byte transport: a UART
  on the Nexys A7, FrontPanel pipes on the XEM7310 later.
- **One clock domain**, synchronous reset. 100 MHz on the Nexys A7. With
  external memory the memory controller adds a second clock (below).

## Boards

- **Digilent Nexys A7-100T** (XC7A100T): the MNIST-112 configuration over the
  USB-UART. See [NEXYS_QUICKSTART.md](NEXYS_QUICKSTART.md).
- **Opal Kelly XEM7310-A200** (XC7A200T): not built yet.

## External memory and DMA (planned)

### Why

The weights of the two layers are about 885 KB, more than the block RAM can
hold next to the operand buffers, so the program runs in phases and every
phase's weights are written into the engine before it runs. Over the
921,600-baud UART (about 92 KB/s) that is roughly ten seconds of transfer for
every image, and it is repeated for every image.

With DDR the host writes all weights once per session. Before each phase a DMA
copies that phase's loads from DDR into the engine at one 64-bit beat per
clock, about a millisecond for the whole set. Per image only the image and a
few short commands cross the host link.

### Decisions

| | Choice |
|---|---|
| Boards | Nexys A7-100T first (DDR2, 128 MiB, MNIST-112); XEM7310-A200 later (DDR3, 1 GiB, DeiT-224) |
| Vendor IP (approved) | MIG (memory controller), AXI DMA (simple mode, no scatter-gather, 64-bit streams), SmartConnect (interconnect and clock crossing), Clocking Wizard only if the MIG's reference clock needs it |
| Who drives the sequence | The host over the host link: it sends every DMA, start and readback command, as `run_image.py` does for the loads today |
| Engine | Unchanged. `rtl/core` stays frozen; the AXI wrapper's stream pair already fits the DMA's MM2S and S2MM ports |

### Blocks

```
host ── transport ── axi_host_bridge
                        │ AXI4-Lite master (32-bit address)
                        ▼
                    SmartConnect ──> vit_axi_wrapper s_axi   (registers)
                      ▲    │    ──> AXI DMA s_axi_lite       (DMA registers)
                      │    └────> MIG ──> DDR                (DDR window)
                      │
        AXI DMA M_AXI_MM2S, M_AXI_S2MM (read and write DDR)

  bridge m_axis ──┐
                  ├─ stream mux ──> vit_axi_wrapper s_axis   (loads)
  DMA MM2S ───────┘
  vit_axi_wrapper m_axis ── stream demux ──> bridge s_axis   (READ_BUFFER)
                                        └──> DMA S2MM        (store to DDR)
```

- **Stream mux and demux** (new, `rtl/interface`): the direct bridge path
  stays for debugging and for the existing tests; the DMA path is selected
  between packets, never inside one.
- **Bridge**: its AXI4-Lite address grows from 12 to 32 bits so it can reach
  the DMA registers and the DDR window as well as the wrapper.
- **Address map** (fixed at bring-up): wrapper, AXI DMA and DDR each get a
  window in the bridge's 32-bit space.

### Clocks

The engine, wrapper, bridge, DMA and stream mux stay on the 100 MHz system
clock. The MIG runs on its own user clock, set by the memory speed, and
SmartConnect does the crossing between the two. That is the only clock
crossing.

### What DDR holds

DDR holds each load exactly as the beats the wrapper's `s_axis` takes: a pool
item is one 8-byte beat, a bias address four beats, a LayerNorm pair one beat,
a descriptor three beats. A load is then one DMA transfer: set STREAM_TARGET
and STREAM_ADDRESS, then start MM2S with the DDR address and the byte count.
The program generator writes a DDR image file and a load table (phase, kind,
target, buffer address, DDR offset, bytes) next to the existing program files.

The wrapper holds TREADY low while the engine runs, so a DMA load issued then
waits. The host issues loads only between phases, as now.

### Host protocol additions

| Command | Does |
|---|---|
| `WRITE_DDR` | writes the payload into DDR from an address, once per session |
| `READ_DDR` | reads DDR back, for checks |
| `DMA_LOAD` | sets STREAM_TARGET and STREAM_ADDRESS, starts MM2S from a DDR address for a byte count, waits for the DMA to finish |
| `DMA_STORE` | optional: a pool buffer range to DDR through S2MM |

The existing commands keep working unchanged.

### Steps

1. **DDR alone.** MIG, SmartConnect and the wider bridge address;
   `WRITE_DDR` and `READ_DDR`. On the Nexys: write a pattern, read it back.
2. **DMA loads.** AXI DMA, the stream mux, `DMA_LOAD`. `run_image.py` writes
   the DDR image once, then runs every phase with `DMA_LOAD` and START. The
   scores must equal the direct-load flow byte for byte.
3. **DMA stores** (optional). The demux and `DMA_STORE`.
4. **XEM7310.** The same blocks with DDR3 and FrontPanel; DeiT-224 on hardware.

### Verification

- New RTL of our own (stream mux and demux, bridge commands) gets
  self-checking testbenches as usual, with an AXI memory model standing in
  for SmartConnect, MIG and DDR.
- The AXI DMA is checked in a system simulation with its Vivado simulation
  model and the same AXI memory model. The MIG is not simulated; step 1's
  pattern test checks it on the board.
- Bit-exactness does not change: the engine is the same, and the DDR flow
  must deliver the same bytes as the direct flow, which `run_image.py --check`
  reads back after every phase.

## Verification

Every module has a self-checking testbench against golden vectors that
`scripts/gen_vectors.sh` generates. The whole forward pass runs as one
descriptor program and every stage is compared with the golden model, in both
configurations; the host software's byte stream is replayed against the RTL.

## Not in scope

More than two encoder layers, training, a scatter-gather DMA, a soft CPU, and
for now a hardware load table (an FSM that would run the phase loads from DDR
without the host).
