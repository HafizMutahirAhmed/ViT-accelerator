# DeiT-Tiny FPGA accelerator

An INT8 hardware accelerator, in SystemVerilog, for the encoder layers of the
pretrained DeiT-Tiny vision transformer (ViT-Ti/16: width 192, 3 heads of 64,
MLP 768). It runs the first N of the 12 layers, N from 1 to 12, with the real
weights, and its output matches a Python integer model **bit for bit**. It runs
on a **Digilent Nexys A7-100T**, with the weights in the board's DDR2 and the
host PC on the USB-UART.

The demo task is MNIST digit recognition. The accelerator itself knows nothing
about MNIST: a new task needs a new classifier and new calibration data, not
new hardware.

This page is the one document to read. It explains how the design works, how to
run it, how to change it, and what its limits are. The files under `docs/` hold
the full details and are linked where they matter.

---

## 1. How it works

### The idea in one picture

```
 PC (Python)                               FPGA (Nexys A7)
 ───────────                               ───────────────
 make_model.py  ──► model files            ┌────────────────────────────────────────────┐
 run_image.py   ──► weights once ── UART ──►│ axi_host_bridge ──► DDR2 (weights, run list)│
                ──► image + "RUN" ─────────►│        │                    │ DMA           │
                ◄── 10 class scores ◄───────│        ▼                    ▼               │
                                            │   vit_axi_wrapper ──► matmul_engine         │
                                            │                      (8x8 systolic array,   │
                                            │                       GELU, LayerNorm,      │
                                            │                       softmax, residual add)│
                                            └────────────────────────────────────────────┘
```

1. **On the PC, `make_model.py`** takes the pretrained DeiT-Tiny from `timm`,
   quantizes it to INT8 and compiles the forward pass into a **descriptor
   program**: a list of ops such as "matmul buffer 3 by buffer 1 into buffer 5,
   requantize with scale s and shift k" or "softmax of buffer 4". It also writes
   the expected result of every stage (the golden tensors).
2. **The host software `run_image.py`** uploads the weights and the program to
   the board's DDR once. For each image it sends the image and a single `RUN`
   command.
3. **On the FPGA**, the bridge works through the run list in DDR. A DMA copies
   each phase's weights into the engine, and the engine then runs that phase's
   descriptors. At the end the 10 class scores go back to the PC.

The engine holds no weights or model structure of its own. Everything
model-specific is data: the weights, the descriptor program, the scales, and
the GELU and exp tables. That is why most changes are made in Python.

### The engine (`rtl/core`)

- **8 x 8 output-stationary systolic array**, one DSP per element, computing
  INT8 x INT8 into INT32 accumulators.
- **Drain with bias and requantizer:** `out = clip((acc * scale) >>> shift)`,
  with an 8-bit scale and a 5-bit shift for each op, rounding to nearest.
- **Four row units in integer arithmetic:** the residual add, GELU (a
  256-entry table), LayerNorm (a reciprocal-square-root table) and softmax (an
  exp table and one division per row).
- **A pool of 8 block-RAM buffers.** Each op names its input and output
  buffers, so nothing is ever copied.
- **A sequencer** that runs the descriptor table (256 entries).

One encoder layer is 26 ops: LN1; Q, K and V for each head; Q·Kᵀ; softmax;
P·V; projection; add; LN2; fc1; GELU; fc2; add. Details are in
[docs/ENGINE.md](docs/ENGINE.md).

### Number formats

| What | Format |
|---|---|
| Activations and weights, including the residual stream | INT8, one real-valued scale per tensor |
| Accumulators | INT32 |
| Biases | INT32 in the accumulator's scale |
| Requantizer | 8-bit multiplier and 5-bit right shift per op, chosen by Python |
| Scales | Largest magnitude seen on 128 calibration images ÷ 127, stored in `python/calibration/<board>_scales.json` |

### The software flow

**Once per model, on the PC: `make_model.py --board nexys --layers N`** (needs torch)

1. Loads the pretrained DeiT-Tiny from `timm` at the board's image size. For
   112 x 112 the position embedding is resampled to 50 tokens.
2. Reads the scale of every tensor from `python/calibration/<board>_scales.json`.
   With `--recalibrate` it recomputes them: 128 images go through the FP32
   model, and each scale is the largest magnitude ÷ 127.
3. Loads the classifier (`<board>_L<N>_probe.npz`), or trains one the first
   time on the integer model's features of 6,000 training digits.
4. **`golden_model.py`** quantizes the weights to INT8 and runs the test image
   through the integer model. Every op uses the arithmetic of `int8_model.py`,
   the bit-exact model of the hardware. It records every stage, and for every
   matmul the requantizer's scale and shift.
5. **`layer_program.py`** places every tensor in one of the 8 buffers, turns
   each op into a descriptor and splits the program into phases (each phase
   has its weights loaded before it runs).
6. Writes `python/models/<board>_L<N>/`: the descriptor program, the loads of
   every phase, the golden tensors of every stage and `host_config.json`. It
   also writes the GELU, exp and 1/√x tables to `rtl/rom/<board>/`. If those
   tables changed, the bitstream has to be rebuilt.

**Per image, on the PC: `run_image.py`** (no torch needed)

1. `preprocess.py`: the 28 x 28 digit is resized to 112 x 112, normalised,
   cut into 16 x 16 patches (im2col) and quantized with the input scale.
2. `host.py` lays out the weights, descriptors and run list as one DDR image.
   It reads a 64-byte signature from DDR and uploads everything
   (`WRITE_DDR`) only if the board doesn't already hold this model.
3. The image goes to DDR, followed by one `RUN` command. The bridge then works
   through the run list on its own: `LOAD` entries (a DMA copies a phase's
   weights into the engine) and `START` entries (run a phase's descriptors).
4. The host reads the classifier output back, sums the 49 patch rows and takes
   the largest of the 10 scores as the digit. With `--check` it also compares
   the board's tensors with `board_model.py` given the same image. With
   `--step` the host runs the phases one at a time, so every stage can be
   read back.

**For the tests: `scripts/gen_vectors.sh`** runs `gen_vectors.py` (synthetic
vectors for the block-level tests) and `make_model.py` for the configurations
the testbenches use. It then runs `run_image.py --model --record`, which saves
every byte the host would send and receive, so `tb_axi_replay_*` can replay
them to the RTL.

### The hardware modules

```
nexys_top_ddr                       boards/nexys_a7: board top level
├── system_clock_generator          Clocking Wizard IP: 100 MHz system clock, 200 MHz for the MIG
├── uart_rx, uart_tx                the byte transport to the PC
├── memory_system_wrapper           block design: MIG (DDR2), AXI DMA, SmartConnect
└── accelerator_system              rtl/interface: board-independent accelerator
    ├── axi_host_bridge             host protocol, WRITE_DDR / DMA_LOAD / RUN
    ├── (stream multiplexer)        bridge or DMA into the engine's stream input
    └── vit_axi_wrapper → vit_axi_core    AXI4-Lite registers + AXI4-Stream around the engine
        └── matmul_engine           rtl/core: the engine
            ├── descriptor memory, sequencer, bias store
            ├── operand_buffer x 8  the buffer pool
            ├── matmul_stream_controller → systolic_array (skew, 8 x 8 pe)
            ├── drain → requantize  bias add and INT32 → INT8
            └── row_op_reader → residual_add | gelu_lut | layernorm | softmax → row_op_writer
```

| Module | File | What it does |
|---|---|---|
| `vit_config_pkg` | `rtl/core/vit_config_pkg.sv` | Every parameter, the opcodes and the descriptor format |
| `matmul_engine` | `rtl/core/matmul_engine.sv` | Top of the engine: descriptor memory, sequencer, buffer pool, bias store; runs one op after the other |
| `operand_buffer` | `rtl/core/operand_buffer.sv` | One pool buffer: 8 block-RAM banks, one per array lane |
| `matmul_stream_controller` | `rtl/core/matmul_stream_controller.sv` | Walks a matmul tile by tile and streams both operands into the array |
| `systolic_array`, `pe`, `skew` | `rtl/core/` | 8 x 8 output-stationary array; each `pe` is one DSP multiply-accumulate; `skew` delays lane i by i cycles |
| `drain`, `requantize` | `rtl/core/` | Read a finished tile out, add the bias, requantize to INT8, write to the output buffer |
| `row_op_reader`, `row_op_writer` | `rtl/core/` | Stream a tensor row by row through a non-linear unit and write the result back |
| `residual_add` | `rtl/core/residual_add.sv` | Adds two INT8 tensors with different scales |
| `gelu_lut` | `rtl/core/gelu_lut.sv` | GELU as a 256-entry table per layer |
| `layernorm` | `rtl/core/layernorm.sv` | Mean and variance over a row, 1/√x by table, gain and offset |
| `softmax` | `rtl/core/softmax.sv` | Row maximum, exp by table, sum, one division per row; output 0..127 |
| `vit_axi_core`, `vit_axi_wrapper` | `rtl/interface/` | The engine as an AXI block (register map in [docs/AXI_REGISTER_MAP.md](docs/AXI_REGISTER_MAP.md)); the wrapper is plain Verilog so Vivado block designs can use it |
| `axi_host_bridge` | `rtl/interface/axi_host_bridge.sv` | Turns host packets into AXI transactions; DDR writes, DMA loads and the `RUN` sequencer |
| `accelerator_system` | `rtl/interface/accelerator_system.sv` | Bridge + wrapper + stream multiplexer: what a board instantiates |
| `nexys_top_ddr`, `uart_rx`, `uart_tx` | `boards/nexys_a7/` | Nexys A7 top level: clocks, reset, UART, DDR2 block design, LEDs |

Every module's header comment describes its ports, timing and fixed-point
formats in full.

---

## 2. Running it

### What you need

- Ubuntu with **Vivado 2022.2** (xsim for simulation, the toolchain for the
  bitstream). Run `source <Vivado>/settings64.sh` first.
- Python 3 with `numpy`, `Pillow` and `pyserial`. For making models you also
  need a virtual environment with torch:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision timm
```

### Without a board: simulation and the software model

```sh
.venv/bin/python python/make_model.py --board nexys --layers 2      # makes python/models/nexys_L2/
python3 python/run_image.py --model --digit 0                       # the software model of the board
scripts/gen_vectors.sh                                              # all test vectors
scripts/sim.sh encoder_program_nexys                                # one testbench: PASS or FAIL
scripts/regress.sh                                                  # every testbench, about an hour
```

The first run downloads the DeiT-Tiny weights and MNIST. Every run is
deterministic: two runs produce byte-identical files.

### On the Nexys A7

```sh
vivado -mode batch -source scripts/create_nexys_ddr_project.tcl    # project in build/
vivado -mode batch -source scripts/build_nexys_ddr.tcl             # build/nexys_top_ddr.bit; WNS, WHS must be >= 0
```

The first script creates the Vivado project in `build/nexys_a7_ddr_project/`.
It adds the sources from `rtl/` and `boards/nexys_a7/`, generates the
Clocking Wizard, and builds the block design `memory_system` (MIG with
`boards/nexys_a7/mig_ddr2.prj`, AXI DMA, SmartConnect) with
`scripts/create_memory_system_bd.tcl`. The second runs synthesis and
implementation, writes the reports to `build/nexys_a7_ddr_reports/` and
copies the bitstream to `build/nexys_top_ddr.bit`. In the GUI you can instead use **Tools > Run Tcl Script** with the first
script, then **Generate Bitstream**.

Program the board in the Hardware Manager. LD4 lights up once the DDR2 has
calibrated. Then:

```sh
python3 python/run_image.py --port /dev/ttyUSB1 --ping             # "ddr 1"
python3 python/ddr_check.py --port /dev/ttyUSB1                    # DDR, DMA and RUN bring-up, PASS/FAIL
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0 --check  # 2 layers; compares every result with the model
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0 --model-dir python/models/nexys_L6
```

The weights are uploaded on the first run of a model (about 12 s for 2 layers,
over a minute for 12). After that each image takes about a second over the
UART; the engine itself needs about 10 ms for 2 layers and 60 ms for 12.
**Changing the number of layers needs no new bitstream**, only another
`make_model.py --layers N`. Pins, LEDs and troubleshooting are in
[docs/NEXYS_QUICKSTART.md](docs/NEXYS_QUICKSTART.md).

---

## 3. Where things are

| Folder | Contents |
|---|---|
| `rtl/core/` | The engine and the non-linear units. All parameters are in `vit_config_pkg.sv`. Board-independent. |
| `rtl/interface/` | `axi_host_bridge` (host protocol, DMA and `RUN`), `vit_axi_wrapper` (AXI4-Lite and AXI4-Stream around the engine), `accelerator_system` (the two together). |
| `rtl/rom/<board>/` | GELU, exp and 1/√x tables for all 12 layers, loaded with `$readmemh`. Generated by `make_model.py` and committed, because synthesis needs them. |
| `boards/nexys_a7/` | Top level `nexys_top_ddr.sv`, UART, MIG (DDR2) configuration and constraints. |
| `boards/opal_kelly/` | The 224 x 224 configuration for an XEM7310. Simulated only, not built. |
| `tb/` | One self-checking testbench per module, `tb_<name>.sv`. Each prints PASS or FAIL with the first mismatch. |
| `python/` | Golden model, model generator, host software (table below). |
| `python/calibration/` | Committed: the tensor scales and the MNIST classifiers. |
| `scripts/` | `sim.sh`, `regress.sh`, `gen_vectors.sh`, Vivado Tcl scripts. |
| `docs/` | Reference: SPEC, ENGINE, HOST_PROTOCOL, AXI_REGISTER_MAP, NEXYS_QUICKSTART. |

Generated, not tracked: `python/models/`, `python/vectors/`, `sim/`, `build/`.

| Python file | Role |
|---|---|
| `make_model.py` | **Entry point.** Board and number of layers in; scales, classifier, ROM tables, descriptor program and golden tensors out. |
| `golden_model.py` | The DeiT forward pass in the engine's integer arithmetic. **Quantization decisions live here.** |
| `int8_model.py` | Bit-exact model of each hardware op. Reads its constants from `vit_config_pkg.sv`. |
| `layer_program.py` | Places the pass on the 8 buffers and splits it into phases. |
| `host.py` | Byte protocol, serial port, DDR layout, run list. |
| `run_image.py`, `ddr_check.py` | Command-line tools for the board. |
| `board_model.py` | Software model of the whole board, byte for byte, with cycle counts. |
| `preprocess.py` | Image → normalised tensor → im2col matrix. |

---

## 4. How to change things

There are two kinds of change:

- **Data changes:** weights, scales, tables, the program. Rerun
  `make_model.py`. If it reports that the ROM tables changed, rebuild the
  bitstream.
- **Hardware changes:** edit the SystemVerilog, update the matching Python
  model (`int8_model.py` / `golden_model.py`) so the two stay bit-exact, then
  regenerate the vectors and run the regression.

| I want to change… | Edit | Then |
|---|---|---|
| **Number of layers** (1–12) | nothing; `make_model.py --layers N` | Run with `--model-dir python/models/nexys_L<N>`. Same bitstream. |
| **The task** (other classes, other data) | `Data` and `train_probe()` in `make_model.py`: the dataset, the classifier and the calibration images | `make_model.py --recalibrate --retrain`, rebuild the bitstream (the tables follow the scales) |
| **Quantization** (calibration method, SmoothQuant, percentile) | `calibrate()` in `make_model.py`; where scales are applied: `Model` and `Forward` in `golden_model.py` | `--recalibrate`, `gen_vectors.sh`, `regress.sh`, rebuild |
| **Image size / token count** | `BOARDS` in `make_model.py` (size, tokens, buffer depths) and the `BUFFER_DEPTH_*` parameters in the board top | `make_model.py` checks that it fits; rebuild. At most 197 tokens (`ROWS_TOTAL_MAX`). |
| **Another ViT from timm** | `MODEL_NAME` in `golden_model.py` | Works without hardware changes if it fits the limits below. Otherwise raise the limits in `vit_config_pkg.sv`. |
| **Array size** (e.g. 16 x 16) | `SYS_ROW`, `SYS_COL` in `vit_config_pkg.sv` and `golden_model.py`; must be square | Regression; check timing and DSP count |
| **Precision of GELU, LayerNorm or softmax** | Table sizes and widths in `vit_config_pkg.sv`, the unit in `rtl/core`, and `LAYERNORM` / `SOFTMAX` in `golden_model.py` with the matching code in `int8_model.py` | Vectors, regression, rebuild |
| **Host commands** | `axi_host_bridge.sv`, `host.py`, `board_model.py`, [docs/HOST_PROTOCOL.md](docs/HOST_PROTOCOL.md) | `tb_axi_replay_*`, `tb_nexys_top_ddr` |
| **Clock or baud rate** | Parameters of `nexys_top_ddr.sv`, the clocking wizard in `scripts/create_nexys_ddr_project.tcl`, `--baud` on the host | Rebuild |
| **Another board** | A new `boards/<board>/` top that instantiates `accelerator_system` with a byte transport (UART, USB, PCIe…), DDR and clocks; a `BOARDS` entry | `tb_nexys_top_ddr` is the template |

**Engine limits** (`vit_config_pkg.sv`): up to 197 rows (tokens), an inner
dimension up to 768, up to 1000 output columns, 256 descriptors (longer
programs load in parts automatically), tables for 12 GELUs and 36 softmax
heads. A larger model, such as DeiT-Small with 384 width and an MLP of 1536,
needs these limits raised and more block RAM.

### Making an RTL change, step by step

The Python model and the RTL must always compute the same bits. A change to
the arithmetic is therefore made in both.

1. **Edit the SystemVerilog.** New parameters go in `vit_config_pkg.sv`, and
   the module header should state the fixed-point formats.
2. **If the arithmetic or the descriptor format changed, update the Python
   model:** the op in `int8_model.py` and, if needed, how `golden_model.py`
   uses it. `int8_model.py` reads the parameters from `vit_config_pkg.sv`
   itself.
3. **Regenerate the vectors:** `scripts/gen_vectors.sh`.
4. **Run the module's testbench:** `scripts/sim.sh <module>` (for example
   `scripts/sim.sh layernorm`). It prints PASS, or FAIL with the first
   mismatching index; the log is in `sim/sim_<module>.log`. Debug with
   `$display` in the testbench output.
5. **Run the whole regression:** `scripts/regress.sh` (about an hour). Every
   line must say PASS.
6. **Check on the board:** rebuild the bitstream, check WNS and WHS ≥ 0 in the
   build output, and run `run_image.py --digit 0 --step --check`, which
   compares every stage the board computes with the model.

For a **new module**: put it in `rtl/core/` or `rtl/interface/`, where
`sim.sh` compiles every file automatically. Write `tb/tb_<module>.sv`, which
reads golden vectors from `python/` and prints PASS or FAIL with the first
mismatch, and add a line to the `runs` list in `scripts/regress.sh`. The Vivado
project also picks up every file in `rtl/core/` and `rtl/interface/`; only
new files under `boards/` have to be added to the list in
`scripts/create_nexys_ddr_project.tcl`.

### Porting to another board (for example the Opal Kelly XEM7310)

`accelerator_system` is board-independent. A board provides only a byte
transport, clocks and reset, and DDR with an AXI DMA. Building the XEM7310
version means writing a top level with a FrontPanel USB adapter instead of the
UART, the XEM7310's DDR3 MIG settings, its constraints and build scripts, and a
FrontPanel transport class in `host.py`. The engine, the bridge and the rest of
the host software stay unchanged. That board's larger chip (XC7A200T) is what
makes the full 224 x 224 input fit.

The step-by-step guide, including what is already tested, the buffer depths to
use, and what to watch, is in
**[boards/opal_kelly/README.md](boards/opal_kelly/README.md)**.

**Rules that keep it correct:**

- Every RTL change needs a testbench run. Every module's testbench prints PASS
  or FAIL with the first mismatching index.
- Never edit golden vectors or a testbench to make a test pass. A mismatch
  means the RTL and the Python model disagree, so fix the one that is wrong.
- `rtl/core` was frozen during development, with its testbenches as the
  contract. Changing it is fine, but run the whole regression afterwards.

---

## 5. Status and results

| | Status |
|---|---|
| RTL against the golden model | Bit-exact: every testbench passes (`regress.sh`, 25 runs), including 6 and 12 layers through `RUN` |
| Nexys A7 board | 2 layers on the board: all stages match the model (`--check`) and the digits are recognised. The current bitstream (tables for all 12 layers) builds at 100 MHz with timing met (WNS +0.386 ns) and uses 112 of 135 block RAM tiles. |
| Opal Kelly (224 x 224) | Simulated only |
| MNIST accuracy (2,000 test digits) | 90.4 % with 6 layers, 81.8 % with 12 |

### How accurate the INT8 model is compared with FP32 DeiT

Bit-exactness proves that the hardware does exactly what the integer model
says. **How close the integer model is to the real FP32 DeiT-Tiny is a separate
question**, decided by the quantization recipe and not by the RTL.

Cosine similarity between the INT8 model and FP32, measured on 20 images
(1.0 = identical):

| After layer | 1 | 2 | 4 | 6 | 9 | 12 | final LayerNorm |
|---|---|---|---|---|---|---|---|
| Mean cosine | 0.992 | 0.987 | 0.976 | 0.964 | 0.964 | 0.956 | 0.928 |

- **Up to about 6 layers the INT8 model follows FP32 closely.** With 12 layers
  the error adds up, consistent with the MNIST accuracy falling from 90 % to
  82 %.
- The largest errors are at the LayerNorm outputs and the V projections of the
  last layers. These tensors have a few outlier channels, and one scale per
  tensor (set by the outlier) leaves too few levels for the other channels.
- **Not yet measured:** top-1 accuracy on ImageNet, INT8 against FP32.
- **Known remedies, all in Python with no RTL change:** percentile instead of
  max calibration; SmoothQuant (move the outliers into the weights by folding a
  per-channel factor into the LayerNorm gain and the next weights); more
  calibration images from the target task. Per-channel weight scales would
  need a scale per output column in the requantizer.

### Known limitations

- MNIST inputs are not natural images, so the numbers above may differ for the
  real use case. Calibrate on images from the target task.
- No Vivado project is tracked; `create_nexys_ddr_project.tcl` makes one.
- `rtl/core/rom` is a symbolic link to `rtl/rom/opal_kelly` (the engine's
  default tables). It does not survive a Windows checkout without symlink
  support.
- Some comments in `rtl/core` still name `golden/gen_deit_forward.py`, which
  is now `python/golden_model.py`.

---

## 6. Reference documents

| Document | Read it when |
|---|---|
| [docs/SPEC.md](docs/SPEC.md) | You want the design decisions and the DDR/DMA architecture |
| [docs/ENGINE.md](docs/ENGINE.md) | You change the engine: descriptor format, buffer layouts, cycle counts, resources |
| [docs/HOST_PROTOCOL.md](docs/HOST_PROTOCOL.md) | You change the PC ↔ board protocol |
| [docs/AXI_REGISTER_MAP.md](docs/AXI_REGISTER_MAP.md) | You connect the engine to another AXI system (a CPU or another DMA) |
| [docs/NEXYS_QUICKSTART.md](docs/NEXYS_QUICKSTART.md) | You set up the board: pins, LEDs, serial ports, troubleshooting |
