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

The non-linear ops (residual add, GELU, LayerNorm, softmax) are not built. Their
cycle counts here are the plan: one row unit per bank of the row-bank layout,
so SYS_ROW rows are processed together and each pass over a row tile takes one
cycle per row element. The pipeline depths in PLAN are estimates until there
is RTL to measure.
  add         cycles = 4 + tiles * N + ROW_OP_READ_LATENCY + RESIDUAL_ADD_LATENCY + 5
                       built; the 5 are the result register, the writer register, the
                       pool write register and two cycles to end the op. Measured in
                       tb_matmul_engine_nonlinear: 4,820 for 197 x 192.
  gelu        cycles = 4 + tiles * N + tail                    one pass
  layernorm   cycles = 4 + tiles * (2 * N + gap) + tail        statistics pass, output pass
  softmax     cycles = 4 + tiles * (3 * N + gap) + tail        max pass, sum pass, output pass
  tiles = ceil(rows / SYS_ROW), N = row length
  tail  = BUFFER_READ_LATENCY + the op's output pipeline + POOL_WRITE_LATENCY
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


# planned pipeline depths of the non-linear units, in cycles
PLAN = {
    "gelu_pipeline": 2,         # table read, table output register
    "layernorm_statistics": 7,  # square and accumulate 2, variance 3, reciprocal square root table 2
    "layernorm_pipeline": 7,    # centre 1, times reciprocal 2, shift 1, times gain 2, offset/shift/clip 1
    "softmax_max": 1,           # compare after the last score
    "softmax_sum": 4,           # distance 1, exp table 2, accumulate 1
    "softmax_divide": 20,       # one quotient bit per cycle
    "softmax_pipeline": 6,      # distance 1, exp table 2, multiply 2, round/shift/clip 1
}


def nonlinear_cycles(kind, rows, length, lanes):
    """Planned cycles of one non-linear op on rows x length, `lanes` rows at a time."""
    tiles = -(-rows // lanes)
    read, write = PARAMS["BUFFER_READ_LATENCY"], PARAMS["POOL_WRITE_LATENCY"]
    if kind == "add":
        return (4 + tiles * length + PARAMS["ROW_OP_READ_LATENCY"] + PARAMS["RESIDUAL_ADD_LATENCY"]
                + 2 + write + 2)
    if kind == "gelu":
        per_tile, pipeline = length, PLAN["gelu_pipeline"]
    elif kind == "layernorm":
        per_tile = 2 * length + read + PLAN["layernorm_statistics"]
        pipeline = PLAN["layernorm_pipeline"]
    elif kind == "softmax":
        per_tile = (3 * length + read + PLAN["softmax_max"] + read + PLAN["softmax_sum"]
                    + PLAN["softmax_divide"])
        pipeline = PLAN["softmax_pipeline"]
    else:
        raise ValueError(kind)
    return 4 + tiles * per_tile + read + pipeline + write


# the non-linear ops of one encoder layer: (name, kind, count, rows, row length)
LAYER_NONLINEAR = [
    ("LayerNorm 1", "layernorm", 1, 197, 192),
    ("softmax, per head", "softmax", 3, 197, 197),
    ("residual add 1", "add", 1, 197, 192),
    ("LayerNorm 2", "layernorm", 1, 197, 192),
    ("GELU", "gelu", 1, 197, 768),
    ("residual add 2", "add", 1, 197, 192),
]
# once per image, outside the layers
ONCE_NONLINEAR = [
    ("position add", "add", 1, 197, 192),
    ("final LayerNorm, class row", "layernorm", 1, 1, 192),
]


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

    print(f"\nnon-linear ops, planned (not built), {sys_row} rows at a time:")
    print(f"{'op':28s} {'shape':>9s} {'count':>5s} {'cycles each':>11s} {'total':>8s}")
    matmul_layer = sum(op_cycles(rows, inner, cols, column_select, sys_row, sys_col)[0]
                       for _name, group, rows, inner, cols, column_select in ops if group == "L0")
    layer_total = 0
    for name, kind, count, rows, length in LAYER_NONLINEAR:
        cycles = nonlinear_cycles(kind, rows, length, sys_row)
        layer_total += count * cycles
        print(f"{name:28s} {f'{rows}x{length}':>9s} {count:5d} {cycles:11d} {count * cycles:8d}")
    print(f"{'one encoder layer':28s} {'':>9s} {'':>5s} {'':>11s} {layer_total:8d}")
    print(f"matmul cycles of one layer: {matmul_layer}; the non-linear ops add "
          f"{100 * layer_total / matmul_layer:.2f}% ({matmul_layer + layer_total} per layer in total)")
    for name, kind, count, rows, length in ONCE_NONLINEAR:
        print(f"{name:28s} {f'{rows}x{length}':>9s} {count:5d} {nonlinear_cycles(kind, rows, length, sys_row):11d}")
    serial = sum(count * nonlinear_cycles(kind, rows, length, 1) for _n, kind, count, rows, length in LAYER_NONLINEAR)
    print(f"with one row unit instead of {sys_row}: {serial} cycles per layer, "
          f"{100 * serial / matmul_layer:.1f}% of the matmul cycles")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
