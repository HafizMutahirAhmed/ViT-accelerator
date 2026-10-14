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
