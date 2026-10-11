#!/usr/bin/env bash
# Run a testbench in Vivado xsim, headless.
#
# Usage: scripts/sim.sh <module> [PARAM=value ...]
#   e.g. scripts/sim.sh systolic_array SYS_ROW=3 SYS_COL=5
#
# Compiles rtl/vit_config_pkg.sv first, then every other rtl/*.sv and
# tb/tb_<module>.sv, then runs tb_<module> from sim/. PARAM=value pairs
# override testbench parameters.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <module> [PARAM=value ...]" >&2
    exit 2
fi

mod=$1
shift
root=$(cd "$(dirname "$0")/.." && pwd)
pkg=$root/rtl/vit_config_pkg.sv
rtl=()
for f in "$root"/rtl/*.sv; do
    [ "$f" = "$pkg" ] || rtl+=("$f")
done

snapshot=sim_$mod
generics=()
for g in "$@"; do
    generics+=(-generic_top "$g")
    snapshot+=_${g//=/}
done

mkdir -p "$root/sim"
cd "$root/sim"

xvlog -sv "$pkg" "${rtl[@]}" "$root/tb/tb_$mod.sv"
xelab "tb_$mod" -s "$snapshot" "${generics[@]}"
xsim "$snapshot" -runall
