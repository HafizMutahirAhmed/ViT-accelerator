# Nexys A7-100T quick start

The same design as on the Arty (`arty_top_axi`, MNIST-112, docs/ARTY_QUICKSTART.md)
on a Digilent Nexys A7-100T (xc7a100tcsg324-1). Only the constraints differ:
`boards/nexys_a7/nexys_a7.xdc` (lines from Digilent's Nexys-A7-100T-Master.xdc,
digilent-xdc commit 7583b4e) and `boards/nexys_a7/arty_config.xdc`.

| Port | Pin | Nexys A7 signal |
|---|---|---|
| CLK100MHZ | E3 | 100 MHz oscillator |
| ck_rst | C12 | CPU_RESET button, active low |
| uart_txd_in | C4 | UART_TXD_IN, host to FPGA |
| uart_rxd_out | D4 | UART_RXD_OUT, FPGA to host |
| led[0..3] | H17, K15, J13, N14 | LD0 busy, LD1 done, LD2 error, LD3 byte lost |

Prerequisites and `scripts/gen_vectors.sh` are as in steps 1 and 2 of
docs/ARTY_QUICKSTART.md. The host script needs `python/vectors/mnist112/`.

## Build in the Vivado GUI

1. **Tools > Run Tcl Script...** and pick `scripts/create_nexys_project.tcl`.
   It creates `build/nexys_a7_project/nexys_a7.xpr` with top `arty_top_axi`.
2. In the Flow Navigator: **Run Synthesis**, then **Run Implementation**, then
   **Generate Bitstream**. Check that WNS and WHS in the timing summary are 0 or more.
3. **Open Hardware Manager > Open Target > Auto Connect**, then **Program
   Device** with `build/nexys_a7_project/nexys_a7.runs/impl_1/arty_top_axi.bit`.

## Run

```sh
python3 python/run_image.py --port /dev/ttyUSB1 --ping
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0 --check
```

The FT2232 on the board gives two serial ports (`/dev/ttyUSB0` and
`/dev/ttyUSB1` on Linux, two COM ports on Windows); one is JTAG, one is the
UART. If `--ping` gets no answer on one, try the other. CPU_RESET resets
the design.
