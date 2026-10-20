# Opal Kelly XEM7310-A200: how to run the full 224 x 224 model

**Not built.** The Nexys A7-100T (`boards/nexys_a7/`) is the finished board,
with 112 x 112 MNIST digits. This page is for whoever takes the design to the
XEM7310-A200 (Artix-7 XC7A200T, 1 GiB DDR3, USB 3.0 FrontPanel) to run the
real DeiT-Tiny input size, 224 x 224 (197 tokens). Both boards take the same
MNIST digits; here they are resized by 8 instead of 4.

## What already exists and is tested

| Part | State |
|---|---|
| The engine (`rtl/core`) at 224 x 224 | simulated bit-exact: `tb_encoder_program` (the whole 2-layer program, 197 tokens), `tb_matmul_engine_opal_kelly` (all 39 matmuls), the non-linear unit testbenches |
| The model files | `python/make_model.py --board opal_kelly --layers N`, any N from 1 to 12 |
| The tables of all 12 layers | `rtl/rom/opal_kelly/` (in the repo) |
| The scales | `python/calibration/opal_kelly_scales.json` (in the repo) |
| The bridge, `RUN`, the AXI DMA path | `rtl/interface`, tested on the Nexys and against Xilinx's AXI DMA |
| The host protocol | `docs/HOST_PROTOCOL.md`, the same on every board |

The engine needs no change. Everything below is board work.

## Why 224 cannot run on the Nexys

The weights live in DDR, but the activations (Q, K, V, the 197 x 197
attention scores, the MLP hidden layer) live in the engine's block RAM, and
their size follows the token count. The 224 configuration needs about 330 of
the XC7A100T's 135 block RAM tiles; the XC7A200T has 365. Run
`make_model.py --board opal_kelly` and it prints how much of each buffer the
program uses.

## Steps

1. **Make the model.**
   ```sh
   .venv/bin/python python/make_model.py --board opal_kelly --layers 2
   ```
   It writes `python/models/opal_kelly_L2/`, with the digit classifier of that
   number of layers (trained the first time, kept in `python/calibration/`).
   `python3 python/run_image.py --model --model-dir python/models/opal_kelly_L2
   --digit 0` runs it on the software model of the board.

2. **The top level**, `boards/opal_kelly/opal_kelly_top.sv`. Start from
   `boards/nexys_a7/nexys_top_ddr.sv` and change:
   - the pool depths to the `opal_kelly` entry of `BOARDS` in
     `python/make_model.py` (4800, 55296, 19200, 4800, 14775, 9528, 4800, 16);
   - the ROM files to `../rom/opal_kelly/...`;
   - the transport: replace `uart_rx` and `uart_tx` with a FrontPanel adapter
     that turns a pipe-in and a pipe-out into the byte stream
     `accelerator_system` takes (`rx_data`/`rx_valid`/`rx_ready`,
     `tx_data`/`tx_valid`/`tx_ready`). Opal Kelly's FrontPanel HDL (`okHost`,
     the pipes) is proprietary: take it from an installed FrontPanel SDK, do
     not copy it into the repo;
   - the clocks and resets for the XEM7310's oscillator.

3. **The memory system.** `scripts/create_memory_system_bd.tcl` builds MIG +
   AXI DMA + SmartConnect for the Nexys. Copy it and replace the MIG
   configuration with the XEM7310's DDR3 one (Opal Kelly publishes the MIG
   settings for the board). The DMA, SmartConnect and the address map
   (DMA at 0x4040_0000, DDR at 0x8000_0000) can stay. The weights do not
   grow with the image size: 2 layers take about 1.3 MB of DDR, 12 layers
   about 6 MB, far below the 128 MiB `python/host.py` assumes (`DDR_BYTES`).

4. **Constraints.** Pins from Opal Kelly's XEM7310 constraint file, as the
   Nexys's come from Digilent's (`boards/nexys_a7/nexys_a7_ddr.xdc` shows the
   style: copied lines, the source and its checksum in the header).

5. **Build script.** Copy `scripts/create_nexys_ddr_project.tcl` and
   `scripts/build_nexys_ddr.tcl`; part `xc7a200tfbg484-1`.

6. **Host software.**
   - `python/host.py`: a FrontPanel transport, a class with
     `write(bytes)` and `read(count)` like `SerialTransport`.
   - `python/run_image.py` works as it is (it takes the image size from the
     model directory); give it the new transport where it opens the serial
     port, and drop its check that a board must be the Nexys. The protocol,
     the DDR image and the `RUN` flow in `host.py` work unchanged.

7. **Test before the board.**
   - `scripts/regress.sh`: the 224 engine tests must still pass.
   - A testbench of the new top through its pins, like
     `tb/tb_nexys_top_ddr.sv`, and a replay of the host's byte stream, like
     `tb/tb_axi_replay_run.sv` with the opal_kelly model and depths.
   - The build must meet timing (`WNS` and `WHS` at least 0). The engine alone
     reached about 150 MHz on this chip out of context.

## Things to watch

- **Block RAM is tight**: about 330 of 365 tiles before the memory system.
  The MIG and the DMA need a few more; check the utilization report early.
- **Time per image**: about 1.84 million engine cycles per layer at 197
  tokens (`python/board_model.py --program`), 18 ms at 100 MHz; 2 layers with
  the embedding and the classifiers take 4.18 million cycles, 42 ms.
- **Accuracy**: `make_model.py --report` prints it on 2,000 MNIST test digits
  the scales and the classifier never saw.
