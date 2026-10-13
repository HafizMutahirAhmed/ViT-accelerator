#!/usr/bin/env bash
# Run every testbench and print one PASS/FAIL line per run.
#
# Usage: scripts/regress.sh [parallel jobs, default 4]
#
# Exits non-zero if any run fails. Logs are in sim/<snapshot>.log.
set -uo pipefail

jobs=${1:-4}
root=$(cd "$(dirname "$0")/.." && pwd)

runs=(
    "matmul_engine_deit224"
    "matmul_engine_chain SYS_ROW=4 SYS_COL=4"
    "matmul_engine_chain SYS_ROW=8 SYS_COL=8"
    "matmul_engine SYS_ROW=4 SYS_COL=4"
    "matmul_engine SYS_ROW=8 SYS_COL=8"
    "drain SYS_ROW=4 SYS_COL=4"
    "drain SYS_ROW=8 SYS_COL=8"
    "drain SYS_ROW=3 SYS_COL=5"
    "systolic_array SYS_ROW=2 SYS_COL=2"
    "systolic_array SYS_ROW=4 SYS_COL=4"
    "systolic_array SYS_ROW=3 SYS_COL=5"
    "matmul_stream_controller SYS_ROW=4 SYS_COL=4"
    "matmul_stream_controller SYS_ROW=8 SYS_COL=8"
    "requantize"
    "residual_add"
    "gelu_lut"
    "layernorm"
    "softmax"
    "matmul_engine_nonlinear"
    "encoder_program"
    "encoder_program_mnist112"
    "host_link_mnist112"
    "host_replay_mnist112"
    "arty_top_mnist112"
    "axi_replay_mnist112"
)

snapshot_of() {
    local words=($1) name
    name=sim_${words[0]}
    for g in "${words[@]:1}"; do
        name+=_${g//=/}
    done
    echo "$name"
}

mkdir -p "$root/sim"
for run in "${runs[@]}"; do
    snapshot=$(snapshot_of "$run")
    rm -f "$root/sim/$snapshot.status" "$root/sim/$snapshot.log"
done
printf '%s\n' "${runs[@]}" |
    xargs -P "$jobs" -I{} sh -c "\"$root/scripts/sim.sh\" {} > /dev/null 2>&1; echo \$? > \"$root/sim/\$(echo 'sim_{}' | sed 's/ /_/g; s/=//g').status\""

failed=0
for run in "${runs[@]}"; do
    snapshot=$(snapshot_of "$run")
    status=$(cat "$root/sim/$snapshot.status" 2>/dev/null || echo missing)
    summary=$(grep -E '^(PASS|FAIL)' "$root/sim/$snapshot.log" 2>/dev/null | tail -1)
    if [ "$status" = 0 ] && [ "${summary:0:4}" = PASS ]; then
        echo "ok    tb_$run: $summary"
    else
        failed=$((failed + 1))
        echo "FAILED tb_$run (exit $status): ${summary:-no PASS/FAIL line, see sim/$snapshot.log}"
    fi
done

echo "regression: $((${#runs[@]} - failed)) of ${#runs[@]} runs pass"
[ "$failed" -eq 0 ]
