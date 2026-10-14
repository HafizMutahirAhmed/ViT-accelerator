# DeiT-Tiny FPGA accelerator

A SystemVerilog accelerator for the DeiT-Tiny vision transformer (ViT-Ti/16,
224x224) on the Opal Kelly XEM7310-A200 board (Artix-7 XC7A200T). It runs 2 of
the 12 encoder layers with the real pretrained weights, 197 tokens, embedding
width 192, INT8 data and INT32 accumulation. Matrix products run on an
output-stationary systolic array, and every module is checked bit for bit
against a Python golden model.

What exists so far is the compute engine: the systolic array, operand buffers,
drain, requantizer, the residual add, GELU, LayerNorm and softmax, and an op
sequencer that runs them from a descriptor table. The whole 2-layer forward
pass (patch embedding, both encoder layers, final LayerNorm, classifier) runs
as one 57-entry program and is bit-exact in simulation at every stage
(`tb_encoder_program`). The DDR3 and host interfaces and the top level are not
built yet.

- [docs/SPEC.md](docs/SPEC.md): project spec
- [docs/ENGINE.md](docs/ENGINE.md): the matmul engine
- [docs/HOST_PROTOCOL.md](docs/HOST_PROTOCOL.md): the byte protocol between host and board
- [docs/AXI_REGISTER_MAP.md](docs/AXI_REGISTER_MAP.md): the AXI wrapper
- [docs/NEXYS_QUICKSTART.md](docs/NEXYS_QUICKSTART.md): from a clone to a digit on the Nexys A7

## What is frozen and what is meant to change

| Part | Status |
|---|---|
| `rtl/core/` | **Frozen.** The engine and the non-linear units. Application-agnostic: nothing in it knows about MNIST, a board or a transport. Change it only on purpose, with the full regression and a timing run. |
| `rtl/interface/` | Meant to change: how a host reaches the engine (the AXI wrapper and bridge). |
| `boards/<board>/` | Meant to change: one directory per board, with its top level, transport, constraints and build script. Nothing board-specific is anywhere else. |
| `python/` application scripts | Meant to change: the model configurations, generators and host software (see the table below). The integer reference models in it (`int8_ops.py`, `nonlinear_ops.py`) define the arithmetic of `rtl/core` and change only with it. |
| `rtl/rom/<configuration>/` | Generated: table images of one model configuration, written by the generators and kept in the repo because synthesis needs them. |

## Folder layout

| Folder | Contents |
|---|---|
| `rtl/core/` | Engine and non-linear units, one module per file. Every parameter lives in `vit_config_pkg.sv`. |
| `rtl/interface/` | Host protocol on a byte stream, AXI wrapper and bridge. |
| `rtl/rom/deit224/`, `rtl/rom/mnist112/` | GELU, exp and reciprocal-square-root table images of the two configurations. |
| `tb/` | Every testbench, `tb_<name>.sv`. Each reads golden vectors and prints PASS or FAIL. |
| `boards/nexys_a7/` | Digilent Nexys A7-100T: the top level `nexys_top_axi`, UART, constraints. |
| `boards/opal_kelly/` | Opal Kelly XEM7310-A200: a stub for now. |
| `python/` | Reference models, generators and host software. Vectors and images land in `python/vectors/`, `python/images*/` and `python/mnist/`, none of them tracked. |
| `scripts/` | `sim.sh` (one testbench), `regress.sh` (all of them), `gen_vectors.sh`, and the Vivado Tcl scripts: the Nexys A7 project, the engine alone out of context. |
| `docs/` | Spec, engine description, host protocol, AXI register map. |

Simulation output goes to `sim/` and Vivado projects and bitstreams to `build/`; neither is tracked.

### Python scripts

| Script | What it is | MNIST-specific |
|---|---|---|
| `int8_ops.py`, `nonlinear_ops.py` | Bit-exact integer models of the engine's arithmetic | no |
| `pkg_config.py` | Reads `vit_config_pkg.sv`; packs and unpacks descriptors | no |
| `cycle_model.py` | Cycle counts of every op, checked against simulation logs | no |
| `gen_vectors.py` | Synthetic vectors for the core testbenches | no |
| `gen_deit_vectors.py`, `gen_deit_forward.py` | The ViT forward pass in integer arithmetic; generator of the 224 x 224 DeiT vectors | no |
| `layer_program.py` | Lays a forward pass out as a descriptor program on the buffer pool | no |
| `protocol.py`, `transports.py` | Host side of the byte protocol; serial transport | no |
| `engine_model.py` | Software model of the protocol and the engine | no (its two precision constants match the tables in `rtl/rom`) |
| `mnist_config.py`, `gen_mnist_forward.py` | The 112 x 112 MNIST configuration and its vectors | **yes** |
| `preprocess.py` | Digit to 112 x 112 im2col matrix | **yes** |
| `train_probe.py`, `probe_mnist112.npz` | The MNIST linear probes | **yes** |
| `run_image.py` | Sends one digit to a board or the model and prints the answer | **yes** |

### Testbenches by configuration

| Testbenches | Vectors |
|---|---|
| `tb_systolic_array`, `tb_matmul_stream_controller`, `tb_drain`, `tb_requantize`, `tb_matmul_engine`, `tb_matmul_engine_chain` | synthetic, `python/vectors/*.hex` |
| `tb_matmul_engine_deit224`, `tb_encoder_program` (its defaults), `tb_residual_add`, `tb_gelu_lut`, `tb_layernorm`, `tb_softmax`, `tb_matmul_engine_nonlinear` | 224 x 224 DeiT, `python/vectors/deit/` and `deit_forward/` |
| `tb_encoder_program_mnist112`, `tb_axi_replay_mnist112`, `tb_nexys_top_mnist112` | 112 x 112 MNIST, `python/vectors/mnist112/` |

### Known limitations

- **No Vivado project is tracked.** `scripts/create_nexys_project.tcl` creates
  one in `build/` from the sources here.
- **The engine's default ROM images are the DeiT-224 tables.**
  `matmul_engine` defaults to `rom/gelu_rom.hex`, `rom/exp_rom.hex` and
  `rom/rsqrt_rom.hex`, named relative to its own directory. `rtl/core/rom` is
  a symbolic link to `rtl/rom/deit224` so that those defaults still resolve.
  The GELU and exp tables depend on a model's scales, so the frozen core is
  coupled to the DeiT-224 configuration through its defaults; another
  configuration passes its own files (`../rom/mnist112/...`). The link does
  not survive a checkout on Windows without symlink support.
- **Comments inside `rtl/core` still say `golden/`** where they name the
  reference models, which are now in `python/`. The files are frozen and were
  moved without edits.

## Running the tests

The tests need Vivado xsim on the `PATH` (developed with Vivado 2022.2) and the
vectors, none of which are kept in the repo (see "Generating the vectors"). From the
repository root:

```sh
source <Vivado install>/settings64.sh
scripts/gen_vectors.sh
scripts/regress.sh
```

`regress.sh` is the one entry point: it runs every testbench configuration, 4
at a time, and prints one PASS or FAIL line for each. It exits non-zero if any
run fails. Pass a number to change how many run in parallel, for example
`scripts/regress.sh 2`. Logs are in `sim/<snapshot>.log`.

To run one testbench:

```sh
scripts/sim.sh systolic_array SYS_ROW=4 SYS_COL=4
```

## Running on a board

The board is a Digilent Nexys A7-100T, with a reduced configuration that
fits it: a 112 x 112 input (50 tokens), MNIST digits, weights sent per group
of matmuls over the USB-UART. **[docs/NEXYS_QUICKSTART.md](docs/NEXYS_QUICKSTART.md)
goes through it step by step.** In short:

```sh
scripts/gen_vectors.sh
vivado -mode batch -source scripts/create_nexys_project.tcl    # build/nexys_a7_project/nexys_a7.xpr
# synthesis, implementation and bitstream in the Vivado GUI
python3 python/run_image.py --port /dev/ttyUSB1 --ping
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0
python3 python/run_image.py --model --digit 0 --golden         # no board: the software model
```

The project script also runs from the Vivado GUI (Tools > Run Tcl Script). The
pin assignments come from Digilent's Nexys-A7-100T-Master.xdc and were not
checked against the Nexys A7 reference manual.

## Generating the vectors

`scripts/gen_vectors.sh` does all of it:

| Step | Writes | In the repo |
|---|---|---|
| `scripts/fetch_images.sh` | `python/images/` (17 images), `python/images_calibration/` (128), `python/images_heldout/` (25) | no, only the list `python/image_lists.txt` |
| `python3 python/gen_vectors.py` | synthetic vectors, `python/vectors/*.hex` | no |
| `.venv/bin/python python/gen_deit_forward.py` | `python/vectors/deit/`, `python/vectors/deit_forward/`, `rtl/rom/deit224/` | vectors no, ROM images yes |
| `.venv/bin/python python/gen_mnist_forward.py` | `python/vectors/mnist112/`, `rtl/rom/mnist112/` | vectors no, ROM images yes |
| `python3 python/run_image.py --model ... --record` | the host byte stream for `tb_axi_replay_mnist112` | no |

The generators that run the real pretrained DeiT-Tiny need `torch`, `timm` and
`torchvision` in a local virtual environment, `.venv/`, which is not tracked:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision timm
```

The first run downloads the DeiT-Tiny weights and MNIST (to `python/mnist/`).

The generators are deterministic: one thread, fixed seeds, and the tensor
scales read from `python/scales_deit224.json` and `python/scales_mnist112.json`,
which are in the repo. Two runs give byte-identical vectors and the ROM images
that are committed. `gen_vectors.sh` writes the vectors only; run a generator
without `--dump-only` for its accuracy report against FP32 (calibration and
held-out images separately), and with `--recalibrate` to work the scales out
again. A recalibration can change the ROM images, and then the bitstream has
to be rebuilt.
