# DeiT-Tiny FPGA accelerator

Two encoder layers of the pretrained DeiT-Tiny, INT8, bit-exact against an integer golden model.
Spec: @docs/SPEC.md. Tell me if a request conflicts with it.

## Layout
- rtl/core/       the engine and the non-linear units. FROZEN: do not change it unless I ask.
- rtl/interface/  host protocol, AXI wrapper and bridge.
- rtl/rom/        table images per input size (input_112: nexys, input_224: opal_kelly), all 12 layers; generated, committed.
- tb/             self-checking testbenches, tb_<name>.sv.
- boards/         one directory per board: top level, transport, constraints.
- python/         golden models, generators, host software.
- scripts/        sim.sh, regress.sh, gen_vectors.sh, Vivado Tcl scripts.
- docs/           VIVADO_GUI, SPEC, ENGINE, HOST_PROTOCOL, AXI_REGISTER_MAP, NEXYS_QUICKSTART.
- vivado/         the Vivado project vit.xpr: the Nexys build and one simulation set per testbench;
                  vit.xpr and vit.srcs/ are committed, Vivado's output in it is not.
- sim/, build/    simulation and build output, not tracked.

## Commands (Ubuntu; Vivado 2022.2 settings64.sh is sourced before claude starts)
- Vectors (none are tracked): `.venv/bin/python python/make_test_vectors.py` (or `scripts/gen_vectors.sh`)
- A model: `.venv/bin/python python/make_model.py --board nexys|opal_kelly --layers N` (1..12),
  writes python/models/<board>_L<N>/
- One testbench: `scripts/sim.sh <name> [PARAM=value ...]`, log in sim/sim_<name>.log
- Regression: `scripts/regress.sh [jobs]`; about an hour, the opal_kelly (224) runs dominate
- Nexys A7 (nexys_top_ddr): `vivado -mode batch -source scripts/create_nexys_ddr_project.tcl`,
  then `vivado -mode batch -source scripts/build_nexys_ddr.tcl` (reports in build/nexys_a7_ddr_reports)
- The bridge against the real AXI DMA IP: `scripts/sim_dma_ip.sh`, log in sim/sim_dma_ip.log
- The GUI project: `vivado -mode batch -source scripts/create_gui_project.tcl` rewrites vivado/ (after
  adding or removing a source, a testbench or an IP setting); a new testbench needs an entry in its
  `runs` list and in scripts/regress.sh
- Host: `python3 python/run_image.py --port <port> --ping`, `python3 python/ddr_check.py --port <port>`,
  then `run_image.py --port <port> --digit 0 [--model-dir python/models/nexys_L<N>] [--step] [--check]`;
  `--model` instead of `--port` needs no board
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
- I work only in the Vivado GUI (vivado/vit.xpr) and with Python in a terminal. The scripts in scripts/
  are your tools for checking; the README and docs/ describe only the GUI and the Python commands.
- Run the testbench after every RTL change; fix failures before moving on.
- Never edit golden vectors to make a test pass, and do not edit a testbench to make a test pass.
- Do not commit generated files: vectors, images, sim/, build/, Vivado output.
- The scale files python/calibration/*_scales.json, the probes python/calibration/*_probe.npz and
  the ROM images are committed on purpose; change them
  only with --recalibrate and say so.
- Commit per step. Do not push; I push.
- Do not dump full logs into the conversation; grep for errors, mismatches and timing lines.
- Debug through $display and PASS/FAIL output, not waveforms.
