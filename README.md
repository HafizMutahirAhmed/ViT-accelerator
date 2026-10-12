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

The tests need Vivado xsim on the `PATH` (developed with Vivado 2022.2) and the
DeiT vectors, which are not kept in the repo (see the next section):

```sh
source <Vivado install>/settings64.sh
scripts/gen_vectors.sh
scripts/regress.sh
```

`regress.sh` runs 20 testbench configurations, 4 at a time, and prints one PASS
or FAIL line for each. It exits non-zero if any run fails. Pass a number to
change how many run in parallel, for example `scripts/regress.sh 2`. Logs are
in `sim/<snapshot>.log`.

To run one testbench:

```sh
scripts/sim.sh systolic_array SYS_ROW=4 SYS_COL=4
```

## Generating the vectors

`scripts/gen_vectors.sh` does all of it:

| Step | Writes | In the repo |
|---|---|---|
| `scripts/fetch_images.sh` | `golden/images_calibration/` (128 images), `golden/images_heldout/` (25 images) | no, only the list `golden/image_lists.txt` |
| `python3 golden/gen_vectors.py` | synthetic vectors, `golden/vectors/*.hex` | yes |
| `.venv/bin/python golden/gen_deit_forward.py` | `golden/vectors/deit/` and `golden/vectors/deit_forward/` | no |

The last step runs the real pretrained DeiT-Tiny and needs `torch` and `timm`
in a local virtual environment, `.venv/`, which is not tracked:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/pip install torch torchvision timm
```

The first run downloads the DeiT-Tiny weights. `gen_deit_forward.py` is the
golden model of the whole 2-layer pass in integer arithmetic (the matmuls plus
the residual add, GELU, LayerNorm and softmax of `golden/nonlinear_ops.py`). It
takes its scales from the calibration images and prints the error against FP32
for the calibration images and for the held-out images separately. The vectors
are those of one test image, the first one in `golden/images/`:

- `golden/vectors/deit/`: the 39 matmuls of the pass, for `tb_matmul_engine_deit`
- `golden/vectors/deit_forward/`: every stage of the pass, the tables and the
  constants of the non-linear ops
