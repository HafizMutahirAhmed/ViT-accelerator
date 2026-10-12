# DeiT-Tiny FPGA accelerator

A SystemVerilog accelerator for the DeiT-Tiny vision transformer (ViT-Ti/16,
224x224) on the Opal Kelly XEM7310-A200 board (Artix-7 XC7A200T). It runs 2 of
the 12 encoder layers with the real pretrained weights, 197 tokens, embedding
width 192, INT8 data and INT32 accumulation. Matrix products run on an
output-stationary systolic array, and every module is checked bit for bit
against a Python golden model.

What exists so far is the matmul engine: the systolic array, operand buffers,
drain, requantizer and an op sequencer. It is bit-exact in simulation on all 39
matmul ops of the two layers, the patch embedding and the classifier. The
non-linear ops, the DDR3 and host interfaces and the top level are not built
yet.

- [docs/SPEC.md](docs/SPEC.md): project spec
- [docs/ENGINE.md](docs/ENGINE.md): the matmul engine
- [docs/SCHEDULE.md](docs/SCHEDULE.md): schedule

## Folder layout

| Folder | Contents |
|---|---|
| `rtl/` | Synthesizable SystemVerilog, one module per file. Every parameter lives in `vit_config_pkg.sv`. |
| `tb/` | Self-checking testbenches, `tb_<module>.sv`. Each reads golden `.hex` vectors and prints PASS or FAIL. |
| `golden/` | Python golden model and vector generators. Vectors are in `golden/vectors/`, the DeiT ones in `golden/vectors/deit/`. |
| `scripts/` | `sim.sh` (one testbench), `regress.sh` (all of them) and Vivado Tcl scripts. |
| `docs/` | Spec, engine description and schedule. |
| `legacy/` | Files from the earlier weight-stationary design, not used. |

`ViT-implementation.xpr` is the Vivado project. Simulation output goes to
`sim/`, which is not tracked.

## Running the tests

The tests need Vivado xsim on the `PATH` (developed with Vivado 2022.2):

```sh
source <Vivado install>/settings64.sh
scripts/regress.sh
```

This runs 14 testbench configurations, 4 at a time, and prints one PASS or FAIL
line for each. It exits non-zero if any run fails. Pass a number to change how
many run in parallel, for example `scripts/regress.sh 2`. Logs are in
`sim/<snapshot>.log`.

To run one testbench:

```sh
scripts/sim.sh systolic_array SYS_ROW=4 SYS_COL=4
```

## Regenerating the vectors

The vectors are committed, so this is only needed after changing the golden
model or the package.

The synthetic vectors need Python 3 and numpy:

```sh
python3 golden/gen_vectors.py
```

The DeiT vectors come from the real pretrained model and need `torch` and
`timm`. The project uses a local virtual environment, `.venv/`, which is not
tracked:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision timm
.venv/bin/python golden/gen_deit_vectors.py
```

The first run downloads the DeiT-Tiny weights. The images it uses are in
`golden/images/`.

The chained 2-layer forward pass with the non-linear ops (residual add, GELU,
LayerNorm, softmax; integer models in `golden/nonlinear_ops.py`) is a separate
script. It prints the accuracy report and writes `golden/vectors/deit_forward/`:

```sh
.venv/bin/python golden/gen_deit_forward.py
```
