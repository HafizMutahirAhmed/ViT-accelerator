# DeiT-Tiny FPGA accelerator

Two encoder layers of the pretrained DeiT-Tiny, INT8, bit-exact against an integer golden model.
Spec: @docs/SPEC.md. Tell me if a request conflicts with it.

## Layout
- rtl/core/       the engine and the non-linear units. FROZEN: do not change it unless I ask.
- rtl/interface/  host protocol, AXI wrapper and bridge.
- rtl/rom/        table images per configuration (deit224, mnist112); generated, committed.
- tb/             self-checking testbenches, tb_<name>.sv.
- boards/         one directory per board: top level, transport, constraints.
- python/         golden models, generators, host software.
- scripts/        sim.sh, regress.sh, gen_vectors.sh, Vivado Tcl scripts.
- docs/           SPEC, ENGINE, HOST_PROTOCOL, AXI_REGISTER_MAP, NEXYS_QUICKSTART.
- sim/, build/    simulation and build output, not tracked.

## Commands (Ubuntu; Vivado 2022.2 settings64.sh is sourced before claude starts)
- Vectors (none are tracked): `scripts/gen_vectors.sh` (needs .venv with torch, timm, torchvision)
- One testbench: `scripts/sim.sh <name> [PARAM=value ...]`, log in sim/sim_<name>.log
- Regression: `scripts/regress.sh [jobs]`; about an hour, the DeiT-224 runs dominate
- Nexys A7 project: `vivado -mode batch -source scripts/create_nexys_project.tcl`, or in the GUI
  Tools > Run Tcl Script; then synthesis, implementation and bitstream in the GUI
- Engine alone on the XC7A200T: `vivado -mode batch -source scripts/ooc_matmul_engine.tcl`
- Host: `python3 python/run_image.py --port <port> --ping`, then `--digit 0`; `--model` needs no board
- Only `python3` is on the PATH; the generators that need torch run with `.venv/bin/python`

## Conventions
- SystemVerilog for Vivado; parameterize widths; every parameter in rtl/core/vit_config_pkg.sv.
- Descriptive names, no abbreviations (DATA_WIDTH, not DW).
- Every module gets a self-checking testbench on golden vectors that prints PASS or FAIL with the
  first mismatch index.
- State fixed-point formats in every module header.
- Tables are ROMs loaded with $readmemh from rtl/rom/.
- Do not instantiate a vendor primitive without asking me.

## Working rules
- Run the testbench after every RTL change; fix failures before moving on.
- Never edit golden vectors to make a test pass, and do not edit a testbench to make a test pass.
- Do not commit generated files: vectors, images, sim/, build/, Vivado output.
- The scale files python/scales_*.json and the ROM images are committed on purpose; change them
  only with --recalibrate and say so.
- Commit per step. Do not push; I push.
- Do not dump full logs into the conversation; grep for errors, mismatches and timing lines.
- Debug through $display and PASS/FAIL output, not waveforms.
