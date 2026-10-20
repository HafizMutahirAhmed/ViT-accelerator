# DeiT-Tiny FPGA accelerator

A SystemVerilog accelerator for the DeiT-Tiny vision transformer (ViT-Ti/16):
the first N of its 12 encoder layers, N from 1 to 12, with the real pretrained
weights, embedding width 192, 3 heads, INT8 data and INT32 accumulation. Matrix
products run on an 8 x 8 output-stationary systolic array; GELU, softmax,
LayerNorm and the residual adds have their own units. Every module is checked
bit for bit against a Python golden model.

It runs on the **Digilent Nexys A7-100T**: MNIST digits at 112 x 112 (50
tokens), the weights in the board's DDR2, the host on the USB-UART. The 224 x 224
configuration (197 tokens, the model's own input size, also MNIST digits) is for
the **Opal Kelly XEM7310-A200**, which is not built yet; it is simulated.

- [docs/NEXYS_QUICKSTART.md](docs/NEXYS_QUICKSTART.md): from a clone to a digit on the Nexys A7
- [docs/SPEC.md](docs/SPEC.md): project spec
- [docs/ENGINE.md](docs/ENGINE.md): the matmul engine
- [docs/HOST_PROTOCOL.md](docs/HOST_PROTOCOL.md): the byte protocol between host and board
- [docs/AXI_REGISTER_MAP.md](docs/AXI_REGISTER_MAP.md): the AXI wrapper

## Choosing the model: one command

```sh
.venv/bin/python python/make_model.py --board nexys --layers 6
```

`--board` is `nexys` (112 x 112) or `opal_kelly` (224 x 224), `--layers` 1 to
12. The script makes everything the board needs in `python/models/<board>_L<N>/`:
the descriptor program, the weights of every phase, the golden tensors. Then:

```sh
python3 python/run_image.py --port /dev/ttyUSB1 --digit 0 --model-dir python/models/nexys_L6
```

The weights go to DDR on the first run. The Nexys bitstream holds the tables of
all 12 layers, so changing the number of layers needs no new bitstream.

## Folder layout

| Folder | Contents |
|---|---|
| `rtl/core/` | **Frozen.** The engine and the non-linear units, one module per file; every parameter in `vit_config_pkg.sv`. Nothing in it knows about a board, a transport or MNIST. |
| `rtl/interface/` | Host protocol on a byte stream (`axi_host_bridge`, with `RUN`), the AXI wrapper, `accelerator_system`. |
| `rtl/rom/nexys/`, `rtl/rom/opal_kelly/` | The GELU, exp and reciprocal-square-root tables of each board, for all 12 layers. Made by `make_model.py`, kept in the repo because synthesis needs them. |
| `boards/nexys_a7/` | The Nexys A7 top level `nexys_top_ddr`, UART, the DDR2 (MIG) configuration, constraints. |
| `boards/opal_kelly/` | The XEM7310: a stub for now. |
| `tb/` | Every testbench, `tb_<name>.sv`; each reads golden vectors and prints PASS or FAIL. |
| `python/` | The golden model, `make_model.py`, the host software (below). |
| `python/calibration/` | Kept in the repo: the tensor scales of each board (all 12 layers) and the digit classifiers. |
| `scripts/` | `sim.sh` (one testbench), `regress.sh` (all of them), `gen_vectors.sh`, and the Vivado Tcl scripts of the Nexys A7 project. |
| `docs/` | Spec, engine, host protocol, AXI register map, the Nexys quick start. |

Generated, not tracked: `python/models/`, `python/vectors/`, the images and
MNIST under `python/`, simulation output in `sim/`, Vivado projects and
bitstreams in `build/`.

### Python

| File | What it is |
|---|---|
| `make_model.py` | **The one entry point for a model**: board and number of layers in, everything a board and the testbenches need out |
| `run_image.py` | Sends one digit or picture to the board (or the software model) and prints the answer |
| `ddr_check.py` | Board bring-up: the DDR, the DMA and `RUN` |
| `golden_model.py` | The DeiT-Tiny pass in the engine's integer arithmetic, layer count as a parameter |
| `layer_program.py` | Lays a pass out as a descriptor program on the buffer pool, in phases |
| `int8_model.py` | The engine's arithmetic, bit for bit (integer ops, GELU, LayerNorm, softmax, the residual add); reads `vit_config_pkg.sv` and packs descriptors |
| `host.py` | The host side of the board: the byte protocol, the serial port, where everything lives in DDR, the run list |
| `board_model.py` | Software model of the board, byte for byte, and the cycle count of every op |
| `preprocess.py` | A picture to the im2col matrix the engine takes |
| `gen_vectors.py` | Synthetic vectors for the core testbenches |
| `deterministic.py` | One thread and fixed seeds, so every run makes the same files; imported before numpy and torch |

### Testbenches

| Testbenches | Vectors |
|---|---|
| `tb_systolic_array`, `tb_matmul_stream_controller`, `tb_drain`, `tb_requantize`, `tb_matmul_engine`, `tb_matmul_engine_chain` | synthetic, `python/vectors/*.hex` |
| `tb_matmul_engine_opal_kelly`, `tb_encoder_program`, `tb_residual_add`, `tb_gelu_lut`, `tb_layernorm`, `tb_softmax`, `tb_matmul_engine_nonlinear` | 224 x 224, `python/models/opal_kelly_L2/` |
| `tb_encoder_program_nexys`, `tb_axi_replay_step`, `tb_axi_replay_run`, `tb_nexys_top_ddr`, `tb_dma_ip` | 112 x 112, `python/models/nexys_L2/` |
| `tb_axi_replay_run_nexys_L6`, `tb_axi_replay_run_nexys_L12` | 6 and 12 layers through `RUN` |

### Known limitations

- **No Vivado project is tracked.** `scripts/create_nexys_ddr_project.tcl`
  creates one in `build/` from the sources here.
- **The engine's default ROM images are the Opal Kelly tables.** `matmul_engine`
  defaults to `rom/gelu_rom.hex`, `rom/exp_rom.hex` and `rom/rsqrt_rom.hex`,
  named relative to its own directory; `rtl/core/rom` is a symbolic link to
  `rtl/rom/opal_kelly`. The Nexys top passes its own files (`../rom/nexys/...`).
  The link does not survive a checkout on Windows without symlink support.
- **The widest matmul output the engine allows, 1000 columns (`COLS_TOTAL_MAX`),
  is no longer exercised** by the program testbenches: both boards now use
  10-digit classifiers instead of DeiT-Tiny's 1000-class ImageNet head.
- **Comments inside `rtl/core` still name `golden/gen_deit_forward.py`**, which
  is now `python/golden_model.py`. The files are frozen.

## Running the tests

The tests need Vivado xsim on the `PATH` (developed with Vivado 2022.2) and the
vectors, none of which are kept in the repo. From the repository root:

```sh
source <Vivado install>/settings64.sh
scripts/gen_vectors.sh
scripts/regress.sh
```

`regress.sh` runs every testbench configuration, 4 at a time, and prints one
PASS or FAIL line for each; it exits non-zero if any run fails. Logs are in
`sim/<snapshot>.log`. One testbench: `scripts/sim.sh systolic_array SYS_ROW=4 SYS_COL=4`.
The bridge against Xilinx's AXI DMA: `scripts/sim_dma_ip.sh`.

`gen_vectors.sh` needs `torch`, `timm` and `torchvision` in a local virtual
environment, `.venv/`, which is not tracked:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision timm
```

The first run downloads the DeiT-Tiny weights and MNIST (to `python/mnist/`).
Everything is
deterministic: the scales and the classifiers come from `python/calibration/`,
so two runs give byte-identical vectors and the ROM images that are committed.
`make_model.py --recalibrate` works the scales out again; if that changes the
tables, the script says so and the bitstream has to be rebuilt.
