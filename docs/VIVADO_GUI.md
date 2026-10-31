# Using the project in the Vivado GUI (Linux)

Everything in this guide is done with the mouse in Vivado 2022.2, except the last
part: sending a digit to the board takes one command in a terminal. Windows works
too, with the differences at the end.

## 1. Unzip

Unzip anywhere, for example into your home folder, so that the project is
`~/ViT-implementation/vivado/vit.xpr`. Open it from the unzipped folder, not from
inside the zip.

## 2. Open the project

Start Vivado 2022.2, then **Open Project** (on the start page, or File > Project >
Open) and choose `ViT-implementation/vivado/vit.xpr`.

The project holds:

| In the Sources window | What it is |
|---|---|
| Design Sources, top `nexys_top_ddr` | the accelerator on the Nexys A7-100T: `rtl/core` (the engine), `rtl/interface` (the AXI wrapper and the host protocol), `boards/nexys_a7` (UART and top level), the ROM tables of `rtl/rom/input_112`, the Clocking Wizard `system_clock_generator` and the block design `memory_system` (DDR2 controller, AXI DMA, SmartConnect) |
| Constraints | `boards/nexys_a7/*.xdc`: which FPGA pin is the clock, the UART, the LEDs and the DDR2, and the clock speeds |
| Simulation Sources | one simulation set per testbench (below) |

The files are not copied into the project: they are the ones in `rtl/`, `tb/` and
`boards/`, so an edit in Vivado changes the file in those folders.

The first time a simulation or the bitstream needs the IPs, Vivado generates
them (a few minutes, once).

## 3. Run a testbench

Every testbench checks the hardware against the integer golden model on vectors
in `python/` and ends with one line that starts with `PASS` or `FAIL` (with the
first mismatch).

1. In the Sources window, open **Simulation Sources**. Every set there is one
   testbench run.
2. Right-click the set, **Make Active**.
3. Flow Navigator > SIMULATION > **Run Simulation > Run Behavioral Simulation**.
4. The simulation runs to the end by itself. The result is the `PASS` or `FAIL`
   line in the **Tcl Console** at the bottom.
5. Close the simulation (the X on the blue SIMULATION bar) before running the
   next set.

The waveform window opens, but no signals are recorded, so that the long runs
stay fast. To look at signals: add them to the waveform, then Restart and Run All
(the toolbar buttons).

| Simulation set | What it checks |
|---|---|
| `systolic_array_2x2`, `_4x4`, `_3x5` | the grid of multiply-accumulate cells, three sizes |
| `matmul_stream_controller_4x4`, `_8x8` | streaming a tiled matrix multiply into the array |
| `drain_4x4`, `_8x8`, `_3x5` | reading results out, adding the bias, requantizing to INT8 |
| `requantize` | 32-bit sum to INT8 |
| `residual_add` | adding two INT8 tensors with different scales |
| `gelu_lut` | GELU by table, every GELU of the real model |
| `softmax` | softmax by table, every attention head of the real model |
| `layernorm` | LayerNorm, every LayerNorm of the real model |
| `matmul_engine_4x4`, `_8x8` | one matrix multiply through the engine's ports |
| `matmul_engine_chain_4x4`, `_8x8` | one attention head as a chain of operations |
| `matmul_engine_nonlinear` | the non-linear operations through the engine |
| `matmul_engine_opal_kelly` | every matrix multiply of the real DeiT-Tiny at 224 x 224 |
| `encoder_program` | the whole forward pass, 224 x 224 (two layers) |
| `encoder_program_nexys` | the whole forward pass, 112 x 112 (two layers) |
| `axi_replay_step` | the full system with DDR: the PC's byte stream, phase by phase |
| `axi_replay_run` | the same with the RUN command (the board runs the program by itself) |
| `axi_replay_run_nexys_L6` | the same with six layers |
| `axi_replay_run_nexys_L12` | the same with all twelve layers (the descriptors load phase by phase) |
| `nexys_top_ddr` | the top level through its pins, with a UART model |
| `dma_ip` | the system with Xilinx's real AXI DMA IP |

A set takes from seconds for the small blocks to about two hours for
`axi_replay_run_nexys_L12`.

## 4. Build the bitstream

Flow Navigator > PROGRAM AND DEBUG > **Generate Bitstream**. Vivado asks to run
synthesis and implementation first: **Yes**, then **OK**. It takes a while (the first
time longer, while Vivado generates the IPs). The bitstream is `vivado/vit.runs/impl_1/nexys_top_ddr.bit`.

To see the timing: **Open Implemented Design**, then Reports > **Report Timing
Summary**. The worst setup slack (WNS) and hold slack (WHS) must be 0 or more.

## 5. Program the board

Once per computer, Vivado's cable driver must be installed (as root):
`sudo <Vivado>/data/xicom/cable_drivers/lin64/install_script/install_drivers/install_drivers`,
where `<Vivado>` is the Vivado installation, for example `/tools/Xilinx/Vivado/2022.2`.

1. Connect the Nexys A7 by USB and switch it on.
2. Flow Navigator > **Open Hardware Manager** > **Open Target** > **Auto Connect**.
3. **Program Device**, choose the bitstream from step 4, **Program**.
4. LED **LD4** comes on within a second: the DDR2 is ready. If it stays off,
   press CPU_RESET once.

## 6. Run a digit on the board (terminal)

Once: Python 3 with numpy, Pillow and pyserial, and permission to use the serial
port (log out and in again after `usermod`):

```
sudo apt install python3-numpy python3-pil python3-serial
sudo usermod -aG dialout $USER
```

Then, in a terminal in the `ViT-implementation` folder (right-click the folder >
Open in Terminal):

```
python3 python/run_image.py --port /dev/ttyUSB1 --ping
python3 python/ddr_check.py --port /dev/ttyUSB1
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0
python3 python/run_image.py --port /dev/ttyUSB1 --digit 7 --model-dir python/models/nexys_L6
```

- The board gives two serial ports, `/dev/ttyUSB0` and `/dev/ttyUSB1`; one of
  them answers `--ping`.
- `--ping` prints the board's status; `ddr 1` means the DDR2 is ready.
- `ddr_check.py` tests the DDR2, the DMA and the RUN command; every step prints
  PASS or FAIL.
- `--digit N` runs test digit N of MNIST (0 to 9999) and prints the answer and
  the right label. The first run uploads the weights (about 12 s for two
  layers); later runs on the powered board send only the image.
- `--model-dir` picks the number of layers: `python/models/nexys_L2` (the
  default), `nexys_L6` or `nexys_L12`. The same bitstream runs all three.
- Without a board, `--model` instead of `--port /dev/ttyUSB1` runs the same on
  a software copy of the board.

## On Windows

The Vivado steps are the same. The differences:

- Unzip to a short path such as `C:\vit`: Vivado on Windows fails on file paths
  longer than 260 characters, and the DDR2 controller it generates inside the
  project has deep folders.
- The cable driver comes with the Vivado installer; nothing to install.
- Python from python.org, then `py -m pip install numpy pillow pyserial`. The
  serial port is a COM port (Device Manager > Ports), and the commands are
  `py python\run_image.py --port COM4 --digit 0` and so on.
