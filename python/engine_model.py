"""Software model of the board: the host protocol and the engine, bit for bit.

ModelDevice takes the bytes the host sends and produces the bytes the board
would answer. Behind the protocol is a model of matmul_engine that runs
descriptors on the buffer pool with the integer arithmetic of the golden model
(python/int8_ops.py, python/nonlinear_ops.py) and the ROM images the bitstream
contains. It needs numpy only. It is what `run_image.py --model` talks to, and
what `--check` compares the board with.

The model is not cycle accurate: START runs the phase at once and STATUS then
reports idle, with the cycle count of python/cycle_model.py.

ModelDevice(memory=True) is a board with external memory: the DDR (a byte
array) and the DMA behind WRITE_DDR, READ_DDR and DMA_LOAD, with the bridge's
status bits, and RUN, which works through a run list in that DDR. Without it
those commands get the memory status, as on a board whose memory is not
calibrated or absent.
"""

import os
import sys

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HOST_DIR)
sys.path.insert(0, HOST_DIR)
import protocol  # noqa: E402
from cycle_model import nonlinear_cycles, op_cycles  # noqa: E402
from int8_ops import read_hex, requantize, wrap_signed  # noqa: E402
from nonlinear_ops import LayerNormConfig, SoftmaxConfig, gelu, layernorm, residual_add, softmax  # noqa: E402
from pkg_config import PARAMS, unpack_descriptor  # noqa: E402

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
            entry += 1


class ModelDevice:
    """A transport: write() takes command packets, read() returns the responses."""

    def __init__(self, depths, rom_dir, memory=False):
        self.engine = EngineModel(depths, rom_dir)
        self.pending = b""
        self.done = self.error = False
        self.cycles = 0
        self.memory = memory
        self.ddr = bytearray(protocol.DDR_BYTES) if memory else None

    def dma_load(self, target, address, payload):
        """DMA_LOAD: the DDR bytes go to the memory of the target's kind as stream items."""
        kind, buffer_id = target >> 4 & 3, target & 7
        offset, length = int.from_bytes(payload[0:4], "little"), int.from_bytes(payload[4:8], "little")
        data, size, engine = self.ddr[offset:offset + length], protocol.DDR_ITEM_BYTES[kind], self.engine
        for item in range(length // size):
            chunk = bytes(data[item * size:(item + 1) * size])
            if kind == protocol.KIND_POOL:
                engine.buffers[buffer_id][:, address + item] = np.frombuffer(chunk, dtype=np.int8)
            elif kind == protocol.KIND_BIAS:
                engine.bias[:, address + item] = np.frombuffer(chunk, dtype="<i4")
            elif kind == protocol.KIND_LAYERNORM:
                engine.gain[address + item] = int.from_bytes(chunk[0:2], "little", signed=True)
                engine.offset[address + item] = int.from_bytes(chunk[2:6], "little", signed=True)
            else:
                engine.descriptors[address + item] = [int.from_bytes(chunk[4 * w:4 * w + 4], "little")
                                                      for w in range(5)]

    def run_list(self, item, entries):
        """RUN: the entries of the run list at DDR item `item`. Returns (total busy cycles, status)."""
        total = 0
        for index in range(entries):
            entry = bytes(self.ddr[8 * item + protocol.ENTRY_BYTES * index:][:protocol.ENTRY_BYTES])
            word0, word1 = int.from_bytes(entry[0:4], "little"), int.from_bytes(entry[4:8], "little")
            if word0 & 0xFF == protocol.ENTRY_LOAD:
                self.dma_load(word0 >> 8 & 0xFF, word1 & 0xFFFFFF, entry[8:16])
            elif word0 & 0xFF == protocol.ENTRY_START:
                self.done = False
                self.cycles = self.engine.run(word1 & 0xFF)
                self.done = True
                total = (total + self.cycles) & 0xFFFFFFFF
            else:
                return total, protocol.STATUS_RUN
        return total, 0

    def write(self, data):
        while data:
            assert data[0] == protocol.START_BYTE
            code, target = data[1], data[2]
            address, count = int.from_bytes(data[3:6], "little"), int.from_bytes(data[6:8], "little")
            size = protocol.ITEM_BYTES.get(code, 0) * count
            payload, checksum = data[8:8 + size], int.from_bytes(data[8 + size:10 + size], "little")
            status = 0 if protocol.crc16(data[1:8 + size]) == checksum else protocol.STATUS_CHECKSUM
            out = b""
            engine = self.engine
            memory_command = code in (protocol.WRITE_DDR, protocol.READ_DDR) or (code == protocol.DMA_LOAD
                                                                                and count == 1) \
                or (code == protocol.RUN and count != 0)
            if memory_command and not self.memory:
                status |= protocol.STATUS_MEMORY
            if code == protocol.WRITE_DDR:           # lands whatever the checksum says, as the engine writes
                if self.memory:
                    self.ddr[8 * address:8 * address + size] = payload
            elif code == protocol.DMA_LOAD:
                if count != 1:
                    status |= protocol.STATUS_COMMAND
                elif status == 0:
                    self.dma_load(target, address, payload)
            elif code == protocol.RUN:
                if count == 0:
                    status |= protocol.STATUS_COMMAND
                elif status == 0:
                    total, status = self.run_list(address, count)
                    out = total.to_bytes(4, "little")
            elif status == 0 and code == protocol.READ_DDR:
                out = bytes(self.ddr[8 * address:8 * (address + count)])
            elif code in protocol.ITEM_BYTES:        # the writes land whatever the checksum says
                for item in range(count):
                    chunk = payload[item * (size // count):(item + 1) * (size // count)]
                    if code == protocol.WRITE_BUFFER:
                        engine.buffers[target][:, address + item] = np.frombuffer(chunk, dtype=np.int8)
                    elif code == protocol.WRITE_BIAS:
                        engine.bias[:, address + item] = np.frombuffer(chunk, dtype="<i4")
                    elif code == protocol.WRITE_LAYERNORM:
                        engine.gain[address + item] = int.from_bytes(chunk[0:2], "little", signed=True)
                        engine.offset[address + item] = int.from_bytes(chunk[2:6], "little", signed=True)
                    else:
                        engine.descriptors[address + item] = [int.from_bytes(chunk[4 * w:4 * w + 4], "little")
                                                              for w in range(5)]
            elif status == 0 and code == protocol.START:
                self.done = False
                self.cycles = engine.run(address)
                self.done = True
            elif status == 0 and code == protocol.READ_BUFFER:
                out = engine.buffers[target][:, address:address + count].T.astype(np.int8).tobytes()
            elif status == 0 and code == protocol.STATUS:
                out = bytes([(protocol.FLAG_DONE if self.done else 0) | (protocol.FLAG_ERROR if self.error else 0)
                             | (protocol.FLAG_MEMORY if self.memory else 0)])
                out += self.cycles.to_bytes(4, "little")
            elif status == 0 and code == protocol.CLEAR:
                self.done = self.error = False
            elif status == 0:
                status = protocol.STATUS_COMMAND
            if status:
                self.error, out = True, b""
            body = bytes([status]) + out
            self.pending += bytes([protocol.RESPONSE_BYTE]) + body + protocol.crc16(body).to_bytes(2, "little")
            data = data[10 + size:]

    def read(self, count):
        data, self.pending = self.pending[:count], self.pending[count:]
        return data
