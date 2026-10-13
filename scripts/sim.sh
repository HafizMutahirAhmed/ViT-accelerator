#!/usr/bin/env bash
# Run a testbench in Vivado xsim, headless.
#
# Usage: scripts/sim.sh <module> [PARAM=value ...]
#   e.g. scripts/sim.sh systolic_array SYS_ROW=3 SYS_COL=5
#
# Compiles rtl/core/vit_config_pkg.sv first, then every other file of rtl/core
# and rtl/interface and tb/tb_<module>.sv, then runs tb_<module>. PARAM=value
# pairs override testbench parameters. A testbench file may ask for another
# file to be compiled with it by a line "// sim.sh: also compile <path>", the
# path relative to tb/.
#
# Every run works in its own directory, sim/<snapshot>/, so several runs can
# go in parallel. Testbenches name their vectors from there
# ("../../python/vectors/..."). ROM images are named the way synthesis finds
# them, relative to rtl/core: the engine's defaults as "rom/...", which is
# rtl/core/rom, a link to rtl/rom/deit224, and another configuration's as
# "../rom/<configuration>/...". sim/<snapshot>/rom and sim/rom are links that
# make the same names work here.
#
# The output is also saved to sim/<snapshot>.log. The exit code is non-zero
# when a tool fails or the log contains FAIL, a $fatal ("Fatal:") or ERROR.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <module> [PARAM=value ...]" >&2
    exit 2
fi

mod=$1
shift
root=$(cd "$(dirname "$0")/.." && pwd)
pkg=$root/rtl/core/vit_config_pkg.sv
rtl=()
for f in "$root"/rtl/core/*.sv "$root"/rtl/interface/*.sv "$root"/rtl/interface/*.v; do
    [ -f "$f" ] || continue
    [ "$f" = "$pkg" ] || rtl+=("$f")
done

# a testbench that wraps another one names it in a line "// sim.sh: also compile <file>"
extra=()
while read -r also; do
    extra+=("$root/tb/$also")
done < <(sed -n 's|^// sim.sh: also compile ||p' "$root/tb/tb_$mod.sv")

snapshot=sim_$mod
generics=()
for g in "$@"; do
    generics+=(-generic_top "$g")
    snapshot+=_${g//=/}
done

mkdir -p "$root/sim/$snapshot"
ln -sfn ../../rtl/rom/deit224 "$root/sim/$snapshot/rom"
ln -sfn ../rtl/rom "$root/sim/rom"
cd "$root/sim/$snapshot"

log=../$snapshot.log
status=0
{
    xvlog -sv "$pkg" "${rtl[@]}" ${extra[@]+"${extra[@]}"} "$root/tb/tb_$mod.sv" &&
    xelab "tb_$mod" -s "$snapshot" ${generics[@]+"${generics[@]}"} &&
    xsim "$snapshot" -runall
} 2>&1 | tee "$log" || status=$?

if grep -qE 'FAIL|Fatal|\$fatal|ERROR' "$log"; then
    echo "sim.sh: FAIL, \$fatal or ERROR found in sim/$snapshot.log" >&2
    status=1
elif [ "$status" -ne 0 ]; then
    echo "sim.sh: a tool exited with status $status, see sim/$snapshot.log" >&2
fi
exit "$status"
