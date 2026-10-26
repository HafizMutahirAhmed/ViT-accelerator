# Nexys A7-100T quick start

From a fresh clone to a digit recognised on a Digilent Nexys A7-100T
(xc7a100tcsg324-1). The design is `nexys_top_ddr`: the accelerator, the board's 128 MiB DDR2 behind an AXI DMA (block design
`memory_system`: MIG, AXI DMA, SmartConnect; a Clocking Wizard makes the
clocks), and the USB-UART to the host. A 28 x 28 digit is resized to 112 x 112
on the host and sent over the UART; the board runs the first N DeiT-Tiny encoder
layers on it with the weights from DDR, and the class scores come back.

| Port | Pin | Nexys A7 signal |
|---|---|---|
| CLK100MHZ | E3 | 100 MHz oscillator |
| ck_rst | C12 | CPU_RESET button, active low |
| uart_txd_in | C4 | UART_TXD_IN, host to FPGA |
| uart_rxd_out | D4 | UART_RXD_OUT, FPGA to host |
| led[0..4] | H17, K15, J13, N14, R18 | LD0 busy, LD1 done, LD2 error, LD3 byte lost, LD4 DDR2 calibrated |
| ddr2_* | bank 34 | the DDR2, pins from the MIG configuration `boards/nexys_a7/mig_ddr2.prj` |

The pin lines are Digilent's Nexys-A7-100T-Master.xdc (digilent-xdc commit
7583b4e), in `boards/nexys_a7/nexys_a7_ddr.xdc`; the MIG configuration is
Digilent's, from the Nexys A7-100T board files, with the system clock taken
from the clocking wizard. Sources and checksums are in the files.

## Prerequisites

- Vivado 2022.2 with the cable drivers; on Linux run
  `<Vivado>/data/xicom/cable_drivers/lin64/install_script/install_drivers/install_drivers` once, as root.
- Python 3 with `numpy`, `Pillow` and `pyserial`, and on Linux membership of the
  `dialout` group for the serial port.
- A model: `python/models/nexys_L<N>/`, made by `make_model.py`, which needs
  `torch`, `timm` and `torchvision` in `.venv/`:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision timm
.venv/bin/python python/make_model.py --board nexys --layers 2
```

## Choosing the number of layers

```sh
.venv/bin/python python/make_model.py --board nexys --layers 6      # any of 1 .. 12
```

It writes `python/models/nexys_L6/`. The scales of all 12 layers are in
`python/calibration/input_112_scales.json`; the first time a layer count is made,
its digit classifier is trained (a few minutes to an hour for 12 layers) and
kept in `python/calibration/input_112_L<N>_probe.npz`. `--report` also prints the
accuracy on 2,000 test digits. The bitstream holds the tables of all 12 layers,
so **the same bitstream runs every layer count**; `run_image.py --model-dir`
picks the model.

**Any number from 1 to 12 works**, on the same bitstream. What differs is only
how long `make_model.py` takes the first time:

| Layers | Classifier in `python/calibration/` | `make_model.py` takes | Tested |
|---|---|---|---|
| 2, 6, 12 | yes, in the repo | seconds | in simulation (`RUN` through the RTL) and the software model |
| 1, 3, 4, 5, 7, 8, 9, 10, 11 | made the first time | 15 to 60 minutes once, seconds after | the same mechanism as 2, 6 and 12 |

Then run it with `--model-dir python/models/nexys_L<N>`, for example
`nexys_L8`. A model of up to 8 layers (at most 256 descriptors) loads its
descriptors once; 9 to 12 layers load them phase by phase, as the 12-layer
model does.

## Build

In the GUI: **Tools > Run Tcl Script...** with `scripts/create_nexys_ddr_project.tcl`
(it creates `build/nexys_a7_ddr_project/nexys_a7_ddr.xpr` with top `nexys_top_ddr`,
the clocking wizard and the block design), then **Generate Bitstream**. In batch:

```sh
vivado -mode batch -source scripts/create_nexys_ddr_project.tcl
vivado -mode batch -source scripts/build_nexys_ddr.tcl
```

`build_nexys_ddr.tcl` writes the reports to `build/nexys_a7_ddr_reports/` and
prints WNS and WHS; both must be 0 or more. Program the device (**Open Hardware
Manager > Open Target > Auto Connect > Program Device**) with
`build/nexys_top_ddr.bit`, where the batch build copies it (a GUI build leaves it
in `build/nexys_a7_ddr_project/nexys_a7_ddr.runs/impl_1/`). The project and the
reports can be deleted afterwards; the scripts make them again.

## Run

```sh
python3 python/run_image.py --port /dev/ttyUSB1 --ping                 # ddr 1 = DDR2 calibrated
python3 python/ddr_check.py --port /dev/ttyUSB1                        # DDR, DMA and RUN bring-up
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0 --check      # the 2-layer model
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0 --model-dir python/models/nexys_L6
python3 python/run_image.py --port /dev/ttyUSB1 --png my_digit.png     # any picture of a digit
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0 --step --check   # every stage checked
```

1. After programming, **LD4 must come on** within a second: the MIG has
   calibrated the DDR2. `--ping` then prints `ddr 1`. If LD4 stays off, press
   CPU_RESET once.
2. `ddr_check.py` writes 64 KiB to DDR and reads it back, moves data into the
   engine's buffers with the DMA and reads it back, then runs two small run
   lists with `RUN` (one that must work, one that must be refused); every step
   prints PASS or FAIL with the first mismatch index.
3. `run_image.py` uploads the model's weights to DDR the first time (about
   1 MB for 2 layers, roughly 12 s; 6 MB and over a minute for 12 layers) and
   prints `DDR: uploaded ...`; later runs of the same model on the powered board
   print `the weights are already in DDR`. `--reload` uploads them again. Per
   image the image goes to DDR and one `RUN` command makes the board run the
   whole program by itself; about 10 ms of engine time for 2 layers, 60 ms for 12.
4. `--check` compares the board's stages with the software model fed the same
   image: by default the last phase's (the final LayerNorm and both
   classifiers), with `--step` every stage. `--step` runs the phases one by
   one from the host instead of one `RUN`, so they can be read back between.
5. `--digit N` is the N-th picture of the MNIST test set (its label is
   printed); `--png` takes any picture of one digit and prepares it the way
   MNIST digits were made (`--save-input` shows the result).

## Running again without uploading the weights

Use the same command, without `--reload` (and without `--check`, which is
slower):

```sh
python3 python/run_image.py --port /dev/ttyUSB1 --digit 500 --model-dir python/models/nexys_L6
```

Before each run the script reads a 64-byte signature from DDR. When the DDR
already holds that model's weights it prints `DDR: the weights are already in
DDR ..., not uploaded again`, sends only the image and gives the answer in
about a second. It uploads the weights again only when it has to:

- `--reload` is given;
- the board was powered off or programmed again (the DDR is empty then);
- another model ran in between: the DDR holds one model at a time, so going
  from `nexys_L6` to `nexys_L12` and back uploads the 6-layer weights again.

The FT2232 on the board gives two serial ports (`/dev/ttyUSB0` and
`/dev/ttyUSB1` on Linux, two COM ports on Windows); one is JTAG, one is the
UART. If `--ping` gets no answer on one, try the other.
