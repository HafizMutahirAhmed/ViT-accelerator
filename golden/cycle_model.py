"""Cycle model of matmul_engine, checked against a tb_matmul_engine_deit log.

Usage: python3 golden/cycle_model.py [sim/sim_matmul_engine_deit.log]
       python3 golden/cycle_model.py --program [sim/sim_encoder_program.log]
       --program models the whole descriptor program of golden/layer_program.py,
       entry by entry, and checks it against a tb_encoder_program log.

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

The non-linear ops (residual add, GELU, LayerNorm, softmax) are row ops of the
engine: SYS_ROW rows are processed together and each pass over a row tile
takes one cycle per row element. The formulas below match
tb_matmul_engine_nonlinear.
  add         cycles = 4 + tiles * N + ROW_OP_READ_LATENCY + RESIDUAL_ADD_LATENCY + 5
                       built; the 5 are the result register, the writer register, the
                       pool write register and two cycles to end the op. Measured in
                       tb_matmul_engine_nonlinear: 4,820 for 197 x 192.
  gelu        the same with GELU_LATENCY; measured 19,218 for 197 x 768.
  layernorm   cycles = 4 + tiles * (2 * N + ROW_OP_READ_LATENCY + 12)
                       + ROW_OP_READ_LATENCY + LAYERNORM_LATENCY + 5
                       built; statistics pass, 12 cycles from its last data to the
                       reciprocal, output pass. Measured 10,074 for 197 x 192 and 426
                       for the class row.
  softmax     cycles = 4 + tiles * (3 * N + 2 * ROW_OP_READ_LATENCY + 39)
                       + ROW_OP_READ_LATENCY + SOFTMAX_LATENCY + 5
                       built; max pass, sum pass, a 29-step division, output pass.
                       Measured 16,072 for 197 x 197.
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


def nonlinear_cycles(kind, rows, length, lanes):
    """Cycles of one non-linear op on rows x length, `lanes` rows at a time."""
    tiles = -(-rows // lanes)
    write = PARAMS["POOL_WRITE_LATENCY"]
    if kind in ("add", "gelu"):
        latency = PARAMS["RESIDUAL_ADD_LATENCY"] if kind == "add" else PARAMS["GELU_LATENCY"]
        return 4 + tiles * length + PARAMS["ROW_OP_READ_LATENCY"] + latency + 2 + write + 2
    elif kind == "layernorm":
        return (4 + tiles * (2 * length + PARAMS["ROW_OP_READ_LATENCY"] + 12)
                + PARAMS["ROW_OP_READ_LATENCY"] + PARAMS["LAYERNORM_LATENCY"] + 2 + write + 2)
    elif kind == "softmax":
        return (4 + tiles * (3 * length + 2 * PARAMS["ROW_OP_READ_LATENCY"] + 39)
                + PARAMS["ROW_OP_READ_LATENCY"] + PARAMS["SOFTMAX_LATENCY"] + 2 + write + 2)
    raise ValueError(kind)


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


ROW_KINDS = {1: "add", 2: "gelu", 3: "layernorm", 4: "softmax"}


def program_main(log_name):
    """The descriptor program, entry by entry, against a tb_encoder_program log."""
    forward_dir = os.path.join(os.path.dirname(VEC_DIR), "deit_forward")
    with open(os.path.join(forward_dir, "program_desc.hex")) as f:
        words = [int(line, 16) for line in f]
    count = PARAMS["DESCRIPTOR_WORDS"]
    descriptors = [unpack_descriptor(words[i:i + count]) for i in range(0, len(words), count)]
    with open(os.path.join(forward_dir, "program_ops.txt")) as f:
        names = [(line.split()[1], line.split()[2]) for line in f]
    simulated = {}
    if log_name:
        with open(log_name) as log:
            for line in log:
                m = re.match(r"\s*(\d+)\s+\S+\s+\S+\s+PASS\s.*\scycles (\d+)", line)
                if m:
                    simulated[int(m.group(1))] = int(m.group(2))
    sys_row, sys_col = PARAMS["SYS_ROW"], PARAMS["SYS_COL"]
    phases, mismatches = {}, 0
    for entry, (d, (phase, name)) in enumerate(zip(descriptors, names)):
        if d["opcode"] == PARAMS["OPCODE_MATMUL"]:
            cycles = op_cycles(d["rows_total"], d["inner_dim"], d["cols_total"],
                               d["drain_mode"] == PARAMS["DRAIN_COLUMN_SELECT"], sys_row, sys_col)[0]
            kind = "matmul"
        else:
            kind = ROW_KINDS[d["opcode"]]
            cycles = nonlinear_cycles(kind, d["rows_total"], d["cols_total"], sys_row)
        total = phases.setdefault(phase, {"model": 0, "sim": 0, "matmul": 0, "row": 0, "ops": 0})
        total["model"] += cycles
        total["matmul" if kind == "matmul" else "row"] += cycles
        total["ops"] += 1
        sim = simulated.get(entry)
        if sim is not None:
            total["sim"] += sim
            if sim != cycles:
                mismatches += 1
                print(f"MISMATCH entry {entry} {name}: model {cycles}, simulation {sim}")
    print(f"{'phase':6s} {'ops':>4s} {'matmul':>9s} {'row ops':>8s} {'model':>9s} {'sim':>9s}")
    for phase, total in phases.items():
        print(f"{phase:6s} {total['ops']:4d} {total['matmul']:9d} {total['row']:8d} {total['model']:9d} "
              f"{total['sim'] if simulated else '':>9}")
    print(f"{'all':6s} {len(descriptors):4d} {'':9s} {'':8s} {sum(t['model'] for t in phases.values()):9d} "
          f"{sum(t['sim'] for t in phases.values()) if simulated else '':>9}")
    if simulated:
        print(f"model matches the simulation for {len(simulated) - mismatches} of {len(simulated)} program entries "
              f"({len(descriptors)} in the program)")
    return 1 if mismatches or (simulated and len(simulated) != len(descriptors)) else 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--program":
        return program_main(sys.argv[2] if len(sys.argv) > 2 else None)
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

    print(f"\nnon-linear ops, {sys_row} rows at a time:")
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
