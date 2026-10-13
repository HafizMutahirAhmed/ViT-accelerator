# Arty A7-100T quick start

From a fresh clone to a digit recognised on the board. The design is the
MNIST-112 configuration: a 28 x 28 digit is resized to 112 x 112 on the host,
sent over the USB-UART with the weights, run through two DeiT-Tiny encoder
layers on the FPGA, and the class scores come back.

> **Check the pins before you program the board.** The pin assignments in
> `boards/arty_a7/arty_top.xdc` are nine lines copied unchanged from Digilent's
> `Arty-A7-100-Master.xdc`. They were **not verified against the Arty A7
> reference manual**. Compare them with the manual (clock E3, USB-UART A9 and
> D10, LEDs H5, J5, T9, T10, reset button C2, and that the reset button is
> active low) before you load the bitstream. `boards/arty_a7/arty_config.xdc`
> adds the configuration bank voltage (CFGBVS VCCO, CONFIG_VOLTAGE 3.3), which
> is not in Digilent's file.

## 1. Prerequisites

| What | Notes |
|---|---|
| Board | Digilent Arty A7-100T (XC7A100T-1CSG324C). The A7-35T is too small. |
| Vivado | 2022.2 was used; the free edition covers this device. `source <Vivado>/settings64.sh` in the shell you work in. |
| Cable drivers | Vivado's installer offers them; on Linux run `<Vivado>/data/xicom/cable_drivers/lin64/install_script/install_drivers/install_drivers` once, as root. |
| Python 3 | with `numpy`, `Pillow` and `pyserial` for the host script: `pip install numpy Pillow pyserial`. |
| Generators | `torch`, `timm`, `torchvision` in `.venv/`, only to generate the program files (step 2). |
| Serial port access | On Linux, be in the `dialout` group: `sudo usermod -aG dialout $USER`, then log out and in. |

**Finding the port.** The Arty's USB chip shows up as two serial ports; the
UART is normally the second one.

- Linux: plug the board in and run `ls /dev/ttyUSB*`; typically `/dev/ttyUSB1`.
  `dmesg | tail` shows the names as they appear.
- Windows: Device Manager, "Ports (COM & LPT)"; use the higher of the two new
  COM numbers, for example `COM5`.

## 2. Generate the program files

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision timm
scripts/gen_vectors.sh
```

This downloads the images, MNIST and the pretrained weights the first time and
writes `python/vectors/mnist112/`, which the host script reads. It takes a few
minutes the first time and about a minute after that.

## 3. Create the Vivado project

Either in a shell:

```sh
vivado -mode batch -source scripts/create_arty_project.tcl
```

or in the Vivado GUI: **Tools > Run Tcl Script...** and pick
`scripts/create_arty_project.tcl`.

The project is `build/arty_a7_project/arty_a7.xpr`. It refers to the sources in
the repository; nothing is copied. The top level is `arty_top_axi`.

## 4. Build

**In the GUI.** Open `build/arty_a7_project/arty_a7.xpr`, then in the Flow
Navigator:

1. **Run Synthesis**, then **Run Implementation**, then **Generate Bitstream**
   (or click Generate Bitstream straight away and let it run the others).
   Allow about half an hour.
2. **Timing:** Open Implemented Design, then **Reports > Timing > Report Timing
   Summary**. The Design Timing Summary must show WNS and WHS of 0 or more and
   "All user specified timing constraints are met". The same numbers are in
   the Project Summary and in
   `build/arty_a7_project/arty_a7.runs/impl_1/arty_top_axi_timing_summary_routed.rpt`.
3. **Utilisation:** **Reports > Report Utilization**, or the Project Summary,
   or `arty_top_axi_utilization_placed.rpt` in the same folder.
4. The bitstream is
   `build/arty_a7_project/arty_a7.runs/impl_1/arty_top_axi.bit`.

**In a shell**, all of it in one go:

```sh
vivado -mode batch -source scripts/build_arty_bitstream.tcl
```

It runs the same project runs as the GUI buttons, prints WNS, TNS, WHS and the
utilisation, writes them to `build/arty_top_axi_summary.txt` and copies the
bitstream to `build/arty_top_axi.bit`.

Expected, give or take a little: WNS around +0.8 ns at 100 MHz, about 9,400
LUTs, 8,700 flip-flops, 69 BRAM36 and 50 BRAM18 (94 of 135 block RAM tiles)
and 161 of 240 DSPs.

## 5. Program the board

1. Connect the Arty's USB port to the PC and switch the board on.
2. In Vivado: **Open Hardware Manager > Open Target > Auto Connect**.
3. **Program Device**, choose the `.bit` file from step 4, **Program**.
4. The DONE LED lights. LD4 to LD7 are off.

The bitstream is lost at power-off; programming the flash is not covered here.

## 6. Check the link, then run a digit

```sh
python3 python/run_image.py --port /dev/ttyUSB1 --ping
```

Expected:

```
PING OK: /dev/ttyUSB1 at 921600 baud answered
  busy  0   the engine is idle
  done  0   no finished run
  error 0   no protocol error
  busy cycles of the last run: 0
```

Then a digit (MNIST test digit 0 is a 7):

```sh
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0
```

Expected, after about 12 seconds (almost all of it is sending 1.1 MB of weights
at 921,600 baud; the computation takes 10 ms):

```
  phase embed      1 ops done
  ...
  phase final      3 ops done
summed scores of the patch rows: [-57, -17, 51, -18, -65, 13, -59, 245, -123, 46]
class-row scores:                [-32, -5, -2, 2, 11, 0, -5, 21, -3, 18]
digit: 7   (class-row classifier: 7)   label: 7
engine cycles 1044736; ...
```

The scores are integers and must be exactly these for digit 0. `--digit N`
takes any of the 10,000 MNIST test digits, `--png FILE` any image. Add
`--check` to read every stage back from the board and compare it with the
software model. `--model` in place of `--port` runs the same thing without a
board. The baud rate is `--baud`; it must match the `BAUD_RATE` parameter of
`arty_top_axi` (921,600 by default).

The answer "digit" comes from the classifier on the patch rows, which is right
for about 85% of the test digits. The class-row classifier is printed for
reference and is right for about 30%.

## LEDs

| LED | Meaning |
|---|---|
| LD4 | the engine is running a phase |
| LD5 | a phase has finished since the last start |
| LD6 | error: a packet had a wrong checksum, an unknown command, or came while the engine was running. Stays on until the next run (`run_image.py` clears it at the start). |
| LD7 | a received byte was lost. Stays on until reset. |

The RESET button resets the design; the bitstream stays loaded.

## Troubleshooting

| Symptom | Likely cause and what to do |
|---|---|
| `cannot open /dev/ttyUSB1` | Wrong port name, the board is not plugged in, or no permission. `ls /dev/ttyUSB*`; join `dialout`. Close any terminal program that holds the port. |
| `PING FAIL: no valid answer` | The other of the two ports is the UART: try `/dev/ttyUSB0`. Or the bitstream is not loaded (DONE LED off), or the baud rate differs from `BAUD_RATE`. |
| Ping works at the start and fails later | The board was reprogrammed or reset while the script ran. Run it again. |
| `checksum error` or LD6 on | Bytes are being corrupted: try another cable or port, or a lower baud rate (rebuild with a lower `BAUD_RATE` and pass the same `--baud`). LD7 on means bytes were lost. |
| The script stops with "refused with status ... busy" | A previous run was interrupted while the engine was working. Press RESET and run again. |
| `--check` reports a stage FAIL | The board computed something else than the model. Note the first stage that fails; everything after it follows from it. Check that timing was met and the bitstream is the one built from this checkout. |
| Scores differ from the ones above | The program files and the bitstream are from different checkouts or a recalibration: regenerate with `scripts/gen_vectors.sh` and rebuild. |
| Timing not met (WNS below 0) | Do not use the bitstream. Run the build again; if it stays negative, the sources were changed. `build_arty_bitstream.tcl` says "TIMING IS NOT MET". |
| "not fitting", placement fails for block RAM | The part is not the A7-100T (the project must say xc7a100tcsg324-1), or the buffer depths in `arty_top_axi.sv` were changed. The design needs 94 of the 135 block RAM tiles. |
| Synthesis warns that a `.hex` file could not be opened | The ROM images in `rtl/rom/mnist112/` are missing: the tables would be zero. Restore them with `git checkout rtl/rom`. |
