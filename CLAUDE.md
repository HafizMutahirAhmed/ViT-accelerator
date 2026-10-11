# DeiT-Tiny FPGA accelerator (Opal Kelly XEM7310-A200, XC7A200T)

Deadline: Nov 2, 2026. Project restarted from scratch Oct 11. Full spec, architecture, schedule and open decisions: @docs/SPEC.md (source of truth; tell me if a request conflicts with it).

## Key facts
- 2 of 12 DeiT-Tiny layers, real pretrained weights, 197 tokens, embed 192, INT8 everywhere (INT32 accumulate).
- Output-stationary systolic array, fed from DDR3 (MIG/AXI4) via DMA + BRAM ping-pong tile buffers. Never feed the SA directly from DDR.
- Single main clock domain (MIG ui_clk); only CDC is FrontPanel okClk <-> ui_clk.
- Requantizer: out = clip((acc*scale)>>>shift) to int8; DATA_WIDTH=8, ACC_WIDTH=32, SHIFT_WIDTH=$clog2(ACC_WIDTH), SCALE_WIDTH=8.
- Standard fixed-point softmax, LUT GELU, fixed-point LayerNorm.
- Host comms: Opal Kelly FrontPanel. AXI4 only for DDR3.

## Layout
This folder (ViT-implementation) is an existing Vivado project. Add these next to the .xpr as needed:
- rtl/      synthesizable SystemVerilog, one module per file (new RTL goes here)
- tb/       self-checking testbenches (tb_<module>.sv)
- golden/   Python golden model; dumps per-stage INT8 tensors as .hex into golden/vectors/
- scripts/  sim.sh (xsim runner), build.tcl, helper scripts
- sim/      simulation outputs (gitignored)
- docs/     SPEC.md

Vivado-generated directories (*.srcs, *.runs, *.cache, *.gen, *.sim, *.hw, *.ip_user_files, *.jou, *.log) and the .xpr are managed by Vivado: never edit them by hand. Generated IP (e.g. MIG) stays where Vivado put it. Before changing anything in a new session, inventory what already exists and tell me.

## Conventions
- Synthesizable Verilog-2005/SV for Vivado; no `initial` in synth code; parameterize widths.
- Every module gets a self-checking testbench reading golden .hex vectors; print PASS/FAIL + first mismatch index.
- State fixed-point formats (Q-format, scale, shift) explicitly for every module; flag anything that risks bit-exactness vs the golden model.

## Commands (Ubuntu; Vivado settings64.sh is sourced before claude starts)
- Primary sim (Vivado xsim, headless): `xvlog -sv rtl/<mod>.sv tb/tb_<mod>.sv && xelab tb_<mod> -s sim_<mod> && xsim sim_<mod> -runall` (wrap in scripts/sim.sh <mod>; run from sim/)
- Fast lint (optional): `verilator --lint-only -Wall -sv rtl/<mod>.sv`
- Vivado batch (synth/impl/reports): `vivado -mode batch -source scripts/build.tcl`
- Golden vectors: `python golden/gen_vectors.py`
- Waveforms: I view them myself (GTKWave / Vivado). Debug via $display, assertions and PASS/FAIL output, not by reading waveforms.
- MIG/DDR3 sims need Vivado xsim (Xilinx primitives); Verilator is only for pure-RTL lint/speed.

## Working rules
- Run the testbench after every RTL change; fix failures before moving on.
- Don't dump full waveforms/logs into the conversation; grep for errors, mismatches, timing summary lines.
- Never edit golden vectors to make a test pass.
