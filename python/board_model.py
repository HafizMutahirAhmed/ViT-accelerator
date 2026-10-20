"""Software model of the board, byte for byte, and the engine's cycle model.

run_image.py --model and --check talk to ModelDevice. Run it to check the cycle model against a
simulation log: python3 python/board_model.py [sim/sim_matmul_engine_opal_kelly.log], or
python3 python/board_model.py --program [sim/sim_encoder_program.log [model name]].

--- cycle model: Cycle model of matmul_engine, checked against a tb_matmul_engine_opal_kelly log.

Usage: python3 python/board_model.py [sim/sim_matmul_engine_opal_kelly.log]
       python3 python/board_model.py --program [sim/sim_encoder_program.log [model name, e.g. nexys_L2]]
       --program models the whole descriptor program of python/layer_program.py,
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
The latencies come from rtl/core/vit_config_pkg.sv. The ops are the real DeiT-Tiny
ops listed in python/models/opal_kelly_L2/matmuls/deit_ops.txt.

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

--- engine model: Software model of the board: the host protocol and the engine, bit for bit.

ModelDevice takes the bytes the host sends and produces the bytes the board
would answer. Behind the protocol is a model of matmul_engine that runs
descriptors on the buffer pool with the integer arithmetic of the golden model
(python/int8_model.py, python/int8_model.py) and the ROM images the bitstream
contains. It needs numpy only. It is what `run_image.py --model` talks to, and
what `--check` compares the board with.

The model is not cycle accurate: START runs the phase at once and STATUS then
reports idle, with the cycle count of python/board_model.py.

ModelDevice(memory=True) is a board with external memory: the DDR (a byte
array) and the DMA behind WRITE_DDR, READ_DDR and DMA_LOAD, with the bridge's
status bits, and RUN, which works through a run list in that DDR. Without it
those commands get the memory status, as on a board whose memory is not
calibrated or absent.
"""

import os
import re
import sys

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HOST_DIR)
sys.path.insert(0, HOST_DIR)

import host  # noqa: E402
from int8_model import (PARAMS, LayerNormConfig, SoftmaxConfig, gelu, layernorm, read_hex,  # noqa: E402
                        requantize, residual_add, softmax, unpack_descriptor, wrap_signed)


# ============================================================================
# cycle model
# ============================================================================

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
VEC_DIR = os.path.join(MODELS_DIR, "opal_kelly_L2", "matmuls")


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


def program_main(log_name, directory="opal_kelly_L2"):
    """The descriptor program, entry by entry, against a tb_encoder_program log."""
    forward_dir = os.path.join(MODELS_DIR, directory)
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
        return program_main(sys.argv[2] if len(sys.argv) > 2 else None, *sys.argv[3:4])
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


# ============================================================================
# engine model
# ============================================================================

LANES = 8
LAYERNORM = LayerNormConfig("lut256/r12/n8/g16", "lut", index_bits=8, rsqrt_bits=12, norm_frac=8, gain_bits=16)
SOFTMAX = SoftmaxConfig("exp12/divide", exp_bits=12)


class EngineModel:
    def __init__(self, depths, rom_dir):
        self.buffers = [np.zeros((LANES, depth), dtype=np.int64) for depth in depths]
        self.bias = np.zeros((LANES, PARAMS["BIAS_DEPTH"]), dtype=np.int64)
        self.gain = np.zeros(PARAMS["LAYERNORM_CONSTANT_DEPTH"], dtype=np.int64)
        self.offset = np.zeros(PARAMS["LAYERNORM_CONSTANT_DEPTH"], dtype=np.int64)
        self.descriptors = [[0] * PARAMS["DESCRIPTOR_WORDS"] for _ in range(PARAMS["DESCRIPTOR_COUNT"])]
        self.gelu_rom = read_hex(os.path.join(rom_dir, "gelu_rom.hex"), 8)
        self.exp_rom = np.array([int(line, 16) for line in open(os.path.join(rom_dir, "exp_rom.hex"))])
        self.rsqrt_rom = np.array([int(line, 16) for line in open(os.path.join(rom_dir, "rsqrt_rom.hex"))])

    def rows(self, buffer_id, base, rows, cols):
        """A rows x cols tensor from row banks."""
        row = np.arange(rows)[:, None]
        return self.buffers[buffer_id][row % LANES, base + (row // LANES) * cols + np.arange(cols)[None, :]]

    def put_rows(self, buffer_id, base, stride, tensor):
        row = np.arange(tensor.shape[0])[:, None]
        self.buffers[buffer_id][row % LANES, base + (row // LANES) * stride + np.arange(tensor.shape[1])[None, :]] \
            = tensor

    def run(self, entry):
        """Runs descriptors from `entry` to the one with last_op; returns the busy cycles."""
        cycles = 0
        while True:
            d = unpack_descriptor(self.descriptors[entry])
            rows, cols = d["rows_total"], d["cols_total"]
            a_id, b_id, out_id = d["activation_buffer_id"], d["weight_buffer_id"], d["output_buffer_id"]
            if d["opcode"] == PARAMS["OPCODE_MATMUL"]:
                inner = d["inner_dim"]
                a = self.rows(a_id, d["activation_base"], rows, inner)
                col = np.arange(cols)[None, :]
                b = self.buffers[b_id][col % LANES, d["weight_base"] + (col // LANES) * inner
                                       + np.arange(inner)[:, None]]
                acc = a @ b
                if d["bias_enable"]:
                    acc = acc + self.bias[np.arange(cols) % LANES, d["bias_base"] + np.arange(cols) // LANES]
                y = requantize(wrap_signed(acc, 32), d["scale"], d["shift"])
                column_select = d["drain_mode"] == PARAMS["DRAIN_COLUMN_SELECT"]
                if column_select:
                    self.put_rows(out_id, d["output_base"], d["output_stride"], y)
                else:
                    self.buffers[out_id][col % LANES, d["output_base"] + (col // LANES) * d["output_stride"]
                                         + np.arange(rows)[:, None]] = y
                cycles += op_cycles(rows, inner, cols, column_select, LANES, LANES)[0]
            elif d["opcode"] in (1, 2, 3, 4):
                x = self.rows(a_id, d["activation_base"], rows, cols)
                kind = {1: "add", 2: "gelu", 3: "layernorm", 4: "softmax"}[d["opcode"]]
                if kind == "add":
                    y = residual_add(x, self.rows(b_id, d["weight_base"], rows, cols), d["scale"],
                                     int(wrap_signed(d["bias_base"] & 0xFF, 8)), d["shift"])
                elif kind == "gelu":
                    y = gelu(x, self.gelu_rom[256 * d["bias_base"]:256 * d["bias_base"] + 256])
                elif kind == "layernorm":
                    span = slice(d["bias_base"], d["bias_base"] + cols)
                    y = layernorm(x, self.gain[span], self.offset[span], d["shift"], d["inner_dim"], LAYERNORM,
                                  self.rsqrt_rom)
                else:
                    y = softmax(x, self.exp_rom[256 * d["bias_base"]:256 * d["bias_base"] + 256], SOFTMAX)
                self.put_rows(out_id, d["output_base"], d["output_stride"], y)
                cycles += nonlinear_cycles(kind, rows, cols, LANES)
            if d["last_op"]:
                return cycles
            entry = (entry + 1) % len(self.descriptors)     # the sequencer's index wraps the same way


class ModelDevice:
    """A transport: write() takes command packets, read() returns the responses."""

    def __init__(self, depths, rom_dir, memory=False):
        self.engine = EngineModel(depths, rom_dir)
        self.pending = b""
        self.done = self.error = False
        self.cycles = 0
        self.memory = memory
        self.ddr = bytearray(host.DDR_BYTES) if memory else None

    def dma_load(self, target, address, payload):
        """DMA_LOAD: the DDR bytes go to the memory of the target's kind as stream items."""
        kind, buffer_id = target >> 4 & 3, target & 7
        offset, length = int.from_bytes(payload[0:4], "little"), int.from_bytes(payload[4:8], "little")
        data, size, engine = self.ddr[offset:offset + length], host.DDR_ITEM_BYTES[kind], self.engine
        for item in range(length // size):
            chunk = bytes(data[item * size:(item + 1) * size])
            if kind == host.KIND_POOL:
                engine.buffers[buffer_id][:, address + item] = np.frombuffer(chunk, dtype=np.int8)
            elif kind == host.KIND_BIAS:
                engine.bias[:, address + item] = np.frombuffer(chunk, dtype="<i4")
            elif kind == host.KIND_LAYERNORM:
                engine.gain[address + item] = int.from_bytes(chunk[0:2], "little", signed=True)
                engine.offset[address + item] = int.from_bytes(chunk[2:6], "little", signed=True)
            else:
                engine.descriptors[address + item] = [int.from_bytes(chunk[4 * w:4 * w + 4], "little")
                                                      for w in range(5)]

    def run_list(self, item, entries):
        """RUN: the entries of the run list at DDR item `item`. Returns (total busy cycles, status)."""
        total = 0
        for index in range(entries):
            entry = bytes(self.ddr[8 * item + host.ENTRY_BYTES * index:][:host.ENTRY_BYTES])
            word0, word1 = int.from_bytes(entry[0:4], "little"), int.from_bytes(entry[4:8], "little")
            if word0 & 0xFF == host.ENTRY_LOAD:
                self.dma_load(word0 >> 8 & 0xFF, word1 & 0xFFFFFF, entry[8:16])
            elif word0 & 0xFF == host.ENTRY_START:
                self.done = False
                self.cycles = self.engine.run(word1 & 0xFF)
                self.done = True
                total = (total + self.cycles) & 0xFFFFFFFF
            else:
                return total, host.STATUS_RUN
        return total, 0

    def write(self, data):
        while data:
            assert data[0] == host.START_BYTE
            code, target = data[1], data[2]
            address, count = int.from_bytes(data[3:6], "little"), int.from_bytes(data[6:8], "little")
            size = host.ITEM_BYTES.get(code, 0) * count
            payload, checksum = data[8:8 + size], int.from_bytes(data[8 + size:10 + size], "little")
            status = 0 if host.crc16(data[1:8 + size]) == checksum else host.STATUS_CHECKSUM
            out = b""
            engine = self.engine
            memory_command = code in (host.WRITE_DDR, host.READ_DDR) or (code == host.DMA_LOAD
                                                                                and count == 1) \
                or (code == host.RUN and count != 0)
            if memory_command and not self.memory:
                status |= host.STATUS_MEMORY
            if code == host.WRITE_DDR:           # lands whatever the checksum says, as the engine writes
                if self.memory:
                    self.ddr[8 * address:8 * address + size] = payload
            elif code == host.DMA_LOAD:
                if count != 1:
                    status |= host.STATUS_COMMAND
                elif status == 0:
                    self.dma_load(target, address, payload)
            elif code == host.RUN:
                if count == 0:
                    status |= host.STATUS_COMMAND
                elif status == 0:
                    total, status = self.run_list(address, count)
                    out = total.to_bytes(4, "little")
            elif status == 0 and code == host.READ_DDR:
                out = bytes(self.ddr[8 * address:8 * (address + count)])
            elif code in host.ITEM_BYTES:        # the writes land whatever the checksum says
                for item in range(count):
                    chunk = payload[item * (size // count):(item + 1) * (size // count)]
                    if code == host.WRITE_BUFFER:
                        engine.buffers[target][:, address + item] = np.frombuffer(chunk, dtype=np.int8)
                    elif code == host.WRITE_BIAS:
                        engine.bias[:, address + item] = np.frombuffer(chunk, dtype="<i4")
                    elif code == host.WRITE_LAYERNORM:
                        engine.gain[address + item] = int.from_bytes(chunk[0:2], "little", signed=True)
                        engine.offset[address + item] = int.from_bytes(chunk[2:6], "little", signed=True)
                    else:
                        engine.descriptors[address + item] = [int.from_bytes(chunk[4 * w:4 * w + 4], "little")
                                                              for w in range(5)]
            elif status == 0 and code == host.START:
                self.done = False
                self.cycles = engine.run(address)
                self.done = True
            elif status == 0 and code == host.READ_BUFFER:
                out = engine.buffers[target][:, address:address + count].T.astype(np.int8).tobytes()
            elif status == 0 and code == host.STATUS:
                out = bytes([(host.FLAG_DONE if self.done else 0) | (host.FLAG_ERROR if self.error else 0)
                             | (host.FLAG_MEMORY if self.memory else 0)])
                out += self.cycles.to_bytes(4, "little")
            elif status == 0 and code == host.CLEAR:
                self.done = self.error = False
            elif status == 0:
                status = host.STATUS_COMMAND
            if status:
                self.error, out = True, b""
            body = bytes([status]) + out
            self.pending += bytes([host.RESPONSE_BYTE]) + body + host.crc16(body).to_bytes(2, "little")
            data = data[10 + size:]

    def read(self, count):
        data, self.pending = self.pending[:count], self.pending[count:]
        return data


if __name__ == "__main__":
    sys.exit(main())
