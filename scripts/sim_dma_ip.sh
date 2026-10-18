#!/usr/bin/env bash
# Run tb_dma_ip: accelerator_system against Xilinx's simulation model of the
# AXI DMA, configured as in the block design memory_system. Headless xsim.
#
# Usage: scripts/sim_dma_ip.sh
#
# The first run generates the DMA as a stand-alone IP and exports its
# simulation sources (scripts/export_dma_ip_sim.tcl, into build/dma_ip_sim);
# later runs reuse them. The DMA's own libraries (axi_dma_v7_1_28 and the
# ones it uses) are the precompiled ones of the Vivado installation.
# Vectors: python/ddr_check.py --model --bytes 4096 --record (scripts/gen_vectors.sh).
# The log is sim/sim_dma_ip.log; the exit code is non-zero when a tool fails or
# the log contains FAIL, a $fatal ("Fatal:") or ERROR.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
wrapper=$root/build/dma_ip_sim/project/dma_ip_sim.gen/sources_1/ip/dma_ip/sim/dma_ip.vhd
export_dir=$root/build/dma_ip_sim/export/dma_ip/xsim
if [ ! -f "$wrapper" ]; then
    vivado -mode batch -nojournal -nolog -source "$root/scripts/export_dma_ip_sim.tcl" > /dev/null
fi
# the libraries the exported elaboration names
libraries=$(sed -n 's/^ *xelab .*--relax --mt auto \(.*\) --snapshot.*/\1/p' "$export_dir/dma_ip.sh")

pkg=$root/rtl/core/vit_config_pkg.sv
rtl=()
for f in "$root"/rtl/core/*.sv "$root"/rtl/interface/*.sv "$root"/rtl/interface/*.v; do
    [ "$f" = "$pkg" ] || rtl+=("$f")
done

mkdir -p "$root/sim/sim_dma_ip"
ln -sfn ../../rtl/rom/deit224 "$root/sim/sim_dma_ip/rom"
ln -sfn ../rtl/rom "$root/sim/rom"
cd "$root/sim/sim_dma_ip"

log=../sim_dma_ip.log
status=0
{
    xvhdl --relax -work xil_defaultlib "$wrapper" &&
    xvlog --relax -work xil_defaultlib "$export_dir/glbl.v" &&
    xvlog -sv -work xil_defaultlib "$pkg" "${rtl[@]}" "$root/tb/axi_rule_checker.sv" "$root/tb/tb_dma_ip.sv" &&
    xelab --relax $libraries -L unisims_ver -L secureip xil_defaultlib.tb_dma_ip xil_defaultlib.glbl \
        -s sim_dma_ip &&
    xsim sim_dma_ip -runall
} 2>&1 | tee "$log" || status=$?

if grep -qE 'FAIL|Fatal|\$fatal|ERROR' "$log"; then
    echo "sim_dma_ip.sh: FAIL, \$fatal or ERROR found in sim/sim_dma_ip.log" >&2
    status=1
elif [ "$status" -ne 0 ]; then
    echo "sim_dma_ip.sh: a tool exited with status $status, see sim/sim_dma_ip.log" >&2
fi
exit "$status"
