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

## What is frozen and what is meant to change

| Part | Status |
|---|---|
| `rtl/core/` | **Frozen.** The engine and the non-linear units. Application-agnostic: nothing in it knows about MNIST, a board or a transport. Change it only on purpose, with the full regression and a timing run. |
| `rtl/interface/` | Meant to change: how a host reaches the engine (`host_link`, `vit_accelerator`, the AXI wrapper and bridge). |
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
| `boards/arty_a7/` | Digilent Arty A7-100T: top level, UART, constraints, build script. |
| `boards/opal_kelly/` | Opal Kelly XEM7310-A200: a stub for now. |
| `python/` | Reference models, generators and host software. Vectors and images land in `python/vectors/`, `python/images*/` and `python/mnist/`, none of them tracked. |
| `scripts/` | `sim.sh` (one testbench), `regress.sh` (all of them), `gen_vectors.sh`, Vivado Tcl scripts. |
| `docs/` | Spec, engine description, host protocol, AXI register map. |

Simulation and build output goes to `sim/`, which is not tracked.

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
| `tb_encoder_program_mnist112`, `tb_host_link_mnist112`, `tb_host_replay_mnist112`, `tb_arty_top_mnist112` | 112 x 112 MNIST, `python/vectors/mnist112/` |

### Known limitations

- **There is no Vivado project file.** Simulation goes through
  `scripts/sim.sh` and builds through the Tcl scripts, in non-project mode.
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

The first board is a Digilent Arty A7-100T, with a reduced configuration that
fits it: a 112 x 112 input (50 tokens), MNIST digits, weights sent per group
of matmuls over the USB-UART. See [docs/HOST_PROTOCOL.md](docs/HOST_PROTOCOL.md).

```sh
vivado -mode batch -source boards/arty_a7/build.tcl      # sim/arty_a7_build/arty_top.bit
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0 --check
python3 python/run_image.py --model --digit 0 --golden   # no board: the software model
```

`run_image.py` needs the program files of `scripts/gen_vectors.sh`
(`python/vectors/mnist112/`), numpy, Pillow and pyserial.

## Generating the vectors

`scripts/gen_vectors.sh` does all of it:

| Step | Writes | In the repo |
|---|---|---|
| `scripts/fetch_images.sh` | `python/images/` (17 images), `python/images_calibration/` (128), `python/images_heldout/` (25) | no, only the list `python/image_lists.txt` |
| `python3 python/gen_vectors.py` | synthetic vectors, `python/vectors/*.hex` | no |
| `.venv/bin/python python/gen_deit_forward.py` | `python/vectors/deit/`, `python/vectors/deit_forward/`, `rtl/rom/deit224/` | vectors no, ROM images yes |
| `.venv/bin/python python/gen_mnist_forward.py` | `python/vectors/mnist112/`, `rtl/rom/mnist112/` | vectors no, ROM images yes |
| `python3 python/run_image.py --model ... --record` | the host byte stream for `tb_host_replay_mnist112` | no |

The generators that run the real pretrained DeiT-Tiny need `torch`, `timm` and
`torchvision` in a local virtual environment, `.venv/`, which is not tracked:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision timm
```

The first run downloads the DeiT-Tiny weights and MNIST (to `python/mnist/`).
`gen_deit_forward.py` takes its scales from the calibration images and prints
the error against FP32 for the calibration images and for the held-out images
separately.
