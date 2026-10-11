#!/usr/bin/env bash
# Run a testbench in Vivado xsim, headless.
#
# Usage: scripts/sim.sh <module> [PARAM=value ...]
#   e.g. scripts/sim.sh systolic_array SYS_ROW=3 SYS_COL=5
#
# Compiles vit_config_pkg, every rtl/*.sv and tb/tb_<module>.sv, then runs
# tb_<module> from sim/. PARAM=value pairs override testbench parameters.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <module> [PARAM=value ...]" >&2
    exit 2
fi

mod=$1
shift
root=$(cd "$(dirname "$0")/.." && pwd)
pkg=$root/ViT-implementation.srcs/sources_1/new/vit_config_pkg.sv

snapshot=sim_$mod
generics=()
for g in "$@"; do
    generics+=(-generic_top "$g")
    snapshot+=_${g//=/}
done

mkdir -p "$root/sim"
cd "$root/sim"

xvlog -sv "$pkg" "$root"/rtl/*.sv "$root/tb/tb_$mod.sv"
xelab "tb_$mod" -s "$snapshot" "${generics[@]}"
xsim "$snapshot" -runall
