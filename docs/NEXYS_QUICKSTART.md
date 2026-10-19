# Nexys A7-100T quick start

From a fresh clone to a digit recognised on a Digilent Nexys A7-100T
(xc7a100tcsg324-1). The design is `nexys_top_axi` with the MNIST-112
configuration: a 28 x 28 digit is resized to 112 x 112 on the host, sent over
the USB-UART with the weights, run through two DeiT-Tiny encoder layers on the
FPGA, and the class scores come back. The constraints are
`boards/nexys_a7/nexys_a7.xdc` (lines from Digilent's Nexys-A7-100T-Master.xdc,
digilent-xdc commit 7583b4e) and `boards/nexys_a7/nexys_config.xdc`.

| Port | Pin | Nexys A7 signal |
|---|---|---|
| CLK100MHZ | E3 | 100 MHz oscillator |
| ck_rst | C12 | CPU_RESET button, active low |
| uart_txd_in | C4 | UART_TXD_IN, host to FPGA |
| uart_rxd_out | D4 | UART_RXD_OUT, FPGA to host |
| led[0..3] | H17, K15, J13, N14 | LD0 busy, LD1 done, LD2 error, LD3 byte lost |

## Prerequisites

- Vivado 2022.2 with the cable drivers; on Linux run
  `<Vivado>/data/xicom/cable_drivers/lin64/install_script/install_drivers/install_drivers` once, as root.
- Python 3 with `numpy`, `Pillow` and `pyserial`, and on Linux membership of the
  `dialout` group for the serial port.
- The program files: `python/vectors/mnist112/`, written by
  `scripts/gen_vectors.sh`, which needs `torch`, `timm` and `torchvision` in `.venv/`:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision timm
scripts/gen_vectors.sh
```

Without the board, `python3 python/sim_digit.py --digit N [--index K]` runs the K-th test image of digit N
through the simulated design (xsim, 6 to 10 min), checks every stage bit for bit against the Python model and
prints the predicted digit; `--all-digits [--jobs J]` does one image of each digit and prints a table.

## Build in the Vivado GUI

1. **Tools > Run Tcl Script...** and pick `scripts/create_nexys_project.tcl`.
   It creates `build/nexys_a7_project/nexys_a7.xpr` with top `nexys_top_axi`.
2. In the Flow Navigator: **Run Synthesis**, then **Run Implementation**, then
   **Generate Bitstream**. Check that WNS and WHS in the timing summary are 0 or more.
3. **Open Hardware Manager > Open Target > Auto Connect**, then **Program
   Device** with `build/nexys_a7_project/nexys_a7.runs/impl_1/nexys_top_axi.bit`.

## Run

```sh
python3 python/run_image.py --port /dev/ttyUSB1 --ping
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0 --check
```

The FT2232 on the board gives two serial ports (`/dev/ttyUSB0` and
`/dev/ttyUSB1` on Linux, two COM ports on Windows); one is JTAG, one is the
UART. If `--ping` gets no answer on one, try the other. CPU_RESET resets
the design.

## With the DDR2 as external memory

`nexys_top_ddr` is the same accelerator with the board's 128 MiB DDR2 and an
AXI DMA (block design `memory_system`: MIG, AXI DMA, SmartConnect; a Clocking
Wizard makes the clocks). The weights go to DDR once per power-up; after that a
run sends only the image (about 45 KB instead of 1.1 MB over the UART) and the
DMA copies each phase's weights from DDR into the engine. `nexys_top_axi` above
is unchanged and still works as before.

| Port | Pin | Nexys A7 signal |
|---|---|---|
| led[4] | R18 | LD4: the MIG has calibrated the DDR2 |
| ddr2_* | bank 34 | the DDR2, pins from the MIG configuration `boards/nexys_a7/mig_ddr2.prj` |

The other pins are those of the table at the top. The MIG configuration is
Digilent's, from the Nexys A7-100T board files, with the system clock taken
from the clocking wizard; the source and its sha256 are in the file.

### Build

In the GUI: **Tools > Run Tcl Script...** with `scripts/create_nexys_ddr_project.tcl`
(it creates `build/nexys_a7_ddr_project/nexys_a7_ddr.xpr` with top `nexys_top_ddr`,
the clocking wizard and the block design), then **Generate Bitstream**. In batch:

```sh
vivado -mode batch -source scripts/create_nexys_ddr_project.tcl
vivado -mode batch -source scripts/build_nexys_ddr.tcl
```

`build_nexys_ddr.tcl` writes the reports to `build/nexys_a7_ddr_reports/` and
prints WNS and WHS; both must be 0 or more. Program the device with
`build/nexys_a7_ddr_project/nexys_a7_ddr.runs/impl_1/nexys_top_ddr.bit`.

### Run

```sh
python3 python/run_image.py --port /dev/ttyUSB1 --ping              # ddr 1 = DDR2 calibrated
python3 python/ddr_check.py --port /dev/ttyUSB1                     # DDR and DMA bring-up
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0 --ddr --check
python3 python/run_image.py --port /dev/ttyUSB1 --digit 3 --ddr     # weights already in DDR
python3 python/run_image.py --port /dev/ttyUSB1 --digit 3 --run --check   # one RUN per image
```

1. After programming, **LD4 must come on** within a second: the MIG has
   calibrated the DDR2. `--ping` then prints `ddr 1`. If LD4 stays off, press
   CPU_RESET once; if it still stays off, the bitstream is not
   `nexys_top_ddr` or the MIG did not calibrate.
2. `ddr_check.py` writes 64 KiB to DDR and reads it back, then moves data into
   the engine's buffers with the DMA and reads it back, then runs two small run
   lists with `RUN` (one that must work, one that must be refused); every step
   prints PASS or FAIL with the first mismatch index. It overwrites the
   engine's buffers, which a run writes again anyway.
3. The first `run_image.py --ddr` uploads the weights (about 1 MB, roughly 12 s
   over the UART) and prints `DDR: uploaded ...`; later runs on the powered
   board print `the weights are already in DDR` and take well under a second
   of transfer. `--ddr-reload` uploads them again. `--check` compares every
   stage with the software model, as without DDR.
4. `--run` lets the board run the whole program by itself: the image goes to
   DDR, then one `RUN` command, and the board answers when the scores are
   ready. `--check` then compares the last phase's stages (the final
   LayerNorm and both classifiers) with the software model; the earlier
   stages are overwritten by then, so use `--ddr --check` to see them all.
