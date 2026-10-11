"""Cycle model of matmul_engine, checked against a tb_matmul_engine_deit log.

Usage: python3 golden/cycle_model.py [sim/sim_matmul_engine_deit.log]

For one op on an S_ROW x S_COL array:
  cycles   = tiles * (K + S_ROW + S_COL - 2 + DSP_LATENCY) + overhead
  tiles    = ceil(M / S_ROW) * ceil(N / S_COL)
  overhead = 4                         descriptor fetch, load, apply, launch
           + BUFFER_READ_LATENCY + 2   last read address to tile_done: buffer
                                       read, array input register, last accumulate
           + steps                     drain read steps: S_ROW in row-select,
                                       S_COL in column-select
           + 2 + REQUANTIZE_LATENCY    drain pipeline
           + POOL_WRITE_LATENCY        pool write register
The latencies come from rtl/vit_config_pkg.sv. The ops are the real DeiT-Tiny
ops listed in golden/vectors/deit/deit_ops.txt.
"""

import os
import re
import sys

from pkg_config import PARAMS, unpack_descriptor

VEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors", "deit")


def op_cycles(rows, inner, cols, column_select, sys_row, sys_col):
    """Returns (cycles, tiles, flush cycles, ideal cycles) for one op."""
    tiles = -(-rows // sys_row) * -(-cols // sys_col)
    flush = sys_row + sys_col - 2 + PARAMS["DSP_LATENCY"]
    steps = sys_col if column_select else sys_row
    overhead = (4 + PARAMS["BUFFER_READ_LATENCY"] + 2 + steps + 2 + PARAMS["REQUANTIZE_LATENCY"]
                + PARAMS["POOL_WRITE_LATENCY"])
    return tiles * (inner + flush) + overhead, tiles, tiles * flush, rows * inner * cols / (sys_row * sys_col)


def load_ops():
    ops = []
    with open(os.path.join(VEC_DIR, "deit_ops.txt")) as manifest:
        for line in manifest:
            name, group, _a, _b, _bias, _y, descriptor_file = line.split()
            with open(os.path.join(VEC_DIR, descriptor_file)) as f:
                d = unpack_descriptor([int(word, 16) for word in f])
            ops.append((name, group, d["rows_total"], d["inner_dim"], d["cols_total"],
                        d["drain_mode"] == PARAMS["DRAIN_COLUMN_SELECT"]))
    return ops


def main():
    ops = load_ops()
    simulated = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as log:
            for line in log:
                m = re.match(r"\s*(\S+)\s+PASS\s.*\scycles (\d+)\s", line)
                if m:
                    simulated[m.group(1)] = int(m.group(2))

    sys_row, sys_col = PARAMS["SYS_ROW"], PARAMS["SYS_COL"]
    print(f"model at {sys_row}x{sys_col}: DSP_LATENCY {PARAMS['DSP_LATENCY']}, BUFFER_READ_LATENCY "
          f"{PARAMS['BUFFER_READ_LATENCY']}, REQUANTIZE_LATENCY {PARAMS['REQUANTIZE_LATENCY']}, "
          f"POOL_WRITE_LATENCY {PARAMS['POOL_WRITE_LATENCY']}")
    print(f"{'op':10s} {'shape':>16s} {'tiles':>6s} {'model':>8s} {'sim':>8s}  match")
    mismatches = 0
    for name, _group, rows, inner, cols, column_select in ops:
        cycles, tiles, _flush, _ideal = op_cycles(rows, inner, cols, column_select, sys_row, sys_col)
        sim = simulated.get(name)
        match = "" if sim is None else ("yes" if sim == cycles else "NO")
        mismatches += match == "NO"
        print(f"{name:10s} {f'{rows}x{inner}@{inner}x{cols}':>16s} {tiles:6d} {cycles:8d} "
              f"{'' if sim is None else sim:>8}  {match}")
    if simulated:
        print(f"model matches the simulation for {len(simulated) - mismatches} of {len(simulated)} ops")

    print("\none encoder layer (18 ops) on square arrays:")
    print(f"{'array':>6s} {'cycles':>9s} {'ideal':>9s} {'utilization':>11s} {'flush':>8s} {'flush share':>11s}")
    for size in (8, 12, 16):
        total = flush_total = ideal_total = 0
        for _name, group, rows, inner, cols, column_select in ops:
            if group == "L0":
                cycles, _tiles, flush, ideal = op_cycles(rows, inner, cols, column_select, size, size)
                total, flush_total, ideal_total = total + cycles, flush_total + flush, ideal_total + ideal
        print(f"{f'{size}x{size}':>6s} {total:9d} {ideal_total:9.0f} {100 * ideal_total / total:10.1f}% "
              f"{flush_total:8d} {100 * flush_total / total:10.1f}%")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
