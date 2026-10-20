"""The host side of the board: the byte protocol, the serial port, the DDR image and the run list.

run_image.py and ddr_check.py talk to the board through it; board_model.py answers like the board.

--- protocol: The host side of docs/HOST_PROTOCOL.md, on any transport.

A transport is an object with write(bytes) and read(count) -> bytes (exactly
count bytes, or fewer on a timeout). host.py has one for a serial port
and one for the software model of the board; an Opal Kelly one only has to
provide the same two methods.

--- transports: Transports for HostLink: anything with write(bytes) and read(count).

--- ddr image: A program in external memory: where each load lives in DDR, the upload, the DMA_LOADs, the run list.

The loads of program_loads.txt (weights, biases, LayerNorm constants) and the
descriptor table are put in DDR once, as the stream items the AXI wrapper takes
(docs/AXI_REGISTER_MAP.md): a pool address is 8 bytes, a bias address 32, a
LayerNorm pair 8 (gain, offset, two zero bytes), a descriptor 24 (20 bytes, four
zero bytes). Each load starts on a 64-byte boundary from IMAGE_OFFSET on, so the
DMA needs no unaligned transfers. The input image changes per run: the
step-by-step flow writes it with WRITE_BUFFER, the RUN flow with WRITE_DDR to
INPUT_OFFSET.

The run list (after the loads, also part of the image) is what RUN works
through: LOAD the descriptors, then per phase a LOAD per load of
program_loads.txt (the input image from INPUT_OFFSET) and a START. A program
longer than the descriptor memory gets its descriptors phase by phase instead
(run_image.Program.descriptor_blocks), each before its phase's START.

A 64-byte signature at SIGNATURE_OFFSET (a magic word, the SHA-256 of the
image and its length) says which image the DDR holds, so a second run on the
same powered board reads 64 bytes instead of uploading the image again.
"""

import hashlib


# ============================================================================
# protocol
# ============================================================================

START_BYTE, RESPONSE_BYTE = 0xA5, 0x5A
WRITE_BUFFER, WRITE_BIAS, WRITE_LAYERNORM, WRITE_DESCRIPTOR = 0x01, 0x02, 0x03, 0x04
START, READ_BUFFER, STATUS, CLEAR = 0x05, 0x06, 0x07, 0x08
WRITE_DDR, READ_DDR, DMA_LOAD, RUN = 0x09, 0x0A, 0x0B, 0x0C
ITEM_BYTES = {WRITE_BUFFER: 8, WRITE_BIAS: 32, WRITE_LAYERNORM: 6, WRITE_DESCRIPTOR: 20, WRITE_DDR: 8, DMA_LOAD: 8}
STATUS_CHECKSUM, STATUS_COMMAND, STATUS_BUSY, STATUS_MEMORY, STATUS_RUN = 1, 2, 4, 8, 16
FLAG_BUSY, FLAG_DONE, FLAG_ERROR, FLAG_MEMORY = 1, 2, 4, 8
MAX_ITEMS = 4096          # items per packet; the protocol allows 65,535
# DMA_LOAD targets: the STREAM_TARGET kinds of docs/AXI_REGISTER_MAP.md, and the bytes one item takes in DDR
KIND_POOL, KIND_BIAS, KIND_LAYERNORM, KIND_DESCRIPTOR = 0, 1, 2, 3
DDR_ITEM_BYTES = {KIND_POOL: 8, KIND_BIAS: 32, KIND_LAYERNORM: 8, KIND_DESCRIPTOR: 24}
DDR_BYTES = 128 << 20     # the Nexys A7's DDR2
DESCRIPTOR_SLOTS = 256    # entries of the descriptor memory (DESCRIPTOR_COUNT in rtl/core/vit_config_pkg.sv)
DESCRIPTOR_BYTES = 20     # one descriptor as WRITE_DESCRIPTOR sends it
# run list entries (docs/HOST_PROTOCOL.md): 16 bytes, four 32-bit words
ENTRY_LOAD, ENTRY_START = 1, 2
ENTRY_BYTES = 16


def load_entry(kind, buffer_id, address, offset, length):
    """A LOAD entry: the DMA copies `length` bytes from DDR byte `offset` into the memory of `kind`."""
    words = (ENTRY_LOAD | (kind << 4 | buffer_id) << 8, address, offset, length)
    return b"".join(word.to_bytes(4, "little") for word in words)


def start_entry(first):
    """A START entry: runs the descriptors from entry `first` and waits for done."""
    return b"".join(word.to_bytes(4, "little") for word in (ENTRY_START, first, 0, 0))


def crc16(data, crc=0xFFFF):
    """CRC-16/CCITT-FALSE."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


assert crc16(b"123456789") == 0x29B1


class ProtocolError(Exception):
    def __init__(self, message, status=0):
        super().__init__(message)
        self.status = status      # the response's status bits, when the board refused the command


class HostLink:
    def __init__(self, transport, retries=3):
        self.transport, self.retries = transport, retries
        self.bytes_sent = self.bytes_received = 0
        self.record = None        # a list collects (packet, response) of every exchange

    def command(self, code, target=0, address=0, count=0, payload=b"", response_bytes=0):
        """One command and its response; returns the response payload. Repeats after a checksum error."""
        body = bytes([code, target]) + address.to_bytes(3, "little") + count.to_bytes(2, "little") + payload
        packet = bytes([START_BYTE]) + body + crc16(body).to_bytes(2, "little")
        for _attempt in range(self.retries):
            self.transport.write(packet)
            self.bytes_sent += len(packet)
            head = self.transport.read(2)
            if len(head) < 2 or head[0] != RESPONSE_BYTE:
                raise ProtocolError(f"no response to command {code:#04x} (got {head.hex() or 'nothing'})")
            status = head[1]
            rest = self.transport.read((response_bytes if status == 0 else 0) + 2)
            self.bytes_received += 2 + len(rest)
            if self.record is not None:
                self.record.append((packet, head + rest))
            if len(rest) < 2 or crc16(head[1:] + rest[:-2]) != int.from_bytes(rest[-2:], "little"):
                raise ProtocolError(f"response to command {code:#04x} is cut short or has a wrong checksum")
            if status == 0:
                return rest[:-2]
            if not status & STATUS_CHECKSUM:
                reason = ("busy" if status & STATUS_BUSY else
                          "memory not ready or DMA error" if status & STATUS_MEMORY else
                          "a run list step failed" if status & STATUS_RUN else "unknown command")
                raise ProtocolError(f"command {code:#04x} refused with status {status} ({reason})", status)
            self.clear()
        raise ProtocolError(f"command {code:#04x}: checksum error {self.retries} times")

    def write(self, code, target, address, data):
        """data: all the items back to back; split into packets of at most MAX_ITEMS items."""
        size = ITEM_BYTES[code]
        assert len(data) % size == 0
        for first in range(0, len(data) // size, MAX_ITEMS):
            chunk = data[first * size:(first + MAX_ITEMS) * size]
            self.command(code, target, address + first, len(chunk) // size, chunk)

    def read_buffer(self, buffer_id, address, count):
        """count addresses of a pool buffer: bytes, 8 per address, bank 0 first."""
        data = b""
        for first in range(0, count, MAX_ITEMS):
            items = min(MAX_ITEMS, count - first)
            data += self.command(READ_BUFFER, buffer_id, address + first, items, response_bytes=8 * items)
        return data

    def write_ddr(self, offset, data):
        """data to the DDR from byte `offset` on; both multiples of 8 bytes."""
        assert offset % 8 == 0 and len(data) % 8 == 0
        self.write(WRITE_DDR, 0, offset // 8, data)

    def read_ddr(self, offset, length):
        """length bytes of the DDR from byte `offset` on; both multiples of 8."""
        assert offset % 8 == 0 and length % 8 == 0
        data = b""
        for first in range(0, length // 8, MAX_ITEMS):
            items = min(MAX_ITEMS, length // 8 - first)
            data += self.command(READ_DDR, 0, offset // 8 + first, items, response_bytes=8 * items)
        return data

    def dma_load(self, kind, buffer_id, address, offset, length):
        """The DMA copies `length` bytes from DDR byte `offset` into the memory of `kind` (a pool buffer:
        buffer_id) from `address` on, as the stream items of docs/AXI_REGISTER_MAP.md."""
        assert offset % 8 == 0 and length % DDR_ITEM_BYTES[kind] == 0 and length > 0
        self.command(DMA_LOAD, kind << 4 | buffer_id, address, 1,
                     offset.to_bytes(4, "little") + length.to_bytes(4, "little"))

    def run(self, offset, entries):
        """RUN: the bridge runs `entries` entries of the run list at DDR byte `offset`; returns the busy cycles."""
        assert offset % 8 == 0 and 0 < entries < 65536
        return int.from_bytes(self.command(RUN, 0, offset // 8, entries, response_bytes=4), "little")

    def start(self, entry):
        self.command(START, address=entry)

    def clear(self):
        self.command(CLEAR)

    def status(self):
        """(flags, busy cycles of the last run)."""
        payload = self.command(STATUS, response_bytes=5)
        return payload[0], int.from_bytes(payload[1:5], "little")

    def wait_idle(self, poll=None):
        """Polls STATUS until the engine is idle; returns the busy cycles of the run."""
        while True:
            flags, cycles = self.status()
            if not flags & FLAG_BUSY:
                if flags & FLAG_ERROR:
                    raise ProtocolError("the board's error flag is set")
                return cycles
            if poll:
                poll()


# ============================================================================
# transports
# ============================================================================

class SerialTransport:
    """A serial port (the Nexys A7's USB-UART bridge): 8N1, no flow control."""

    def __init__(self, port, baud=921600, timeout=5.0):
        import serial   # pyserial; only needed for a real board
        self.port = serial.Serial(port, baudrate=baud, bytesize=8, parity="N", stopbits=1, timeout=timeout,
                                  xonxoff=False, rtscts=False, dsrdtr=False)
        self.port.reset_input_buffer()

    def write(self, data):
        self.port.write(data)
        self.port.flush()

    def read(self, count):
        return self.port.read(count)


# ============================================================================
# ddr image
# ============================================================================

IMAGE_OFFSET = 1 << 20                  # first byte of the loads in DDR
SIGNATURE_OFFSET = IMAGE_OFFSET - 64
INPUT_OFFSET = 0                        # the input image of the RUN flow, below the signature
DESCRIPTOR_FILE = "program_desc.hex"
ALIGN = 64
MAGIC = b"VIT-DDR1"
PER_IMAGE_LOADS = ("embed_in_image.hex",)
KINDS = {"buffer": KIND_POOL, "bias": KIND_BIAS, "layernorm": KIND_LAYERNORM}


class DdrImage:
    def __init__(self, program):
        """program: run_image.Program. placements: load file -> (kind, DDR offset, bytes)."""
        self.placements = {}
        image = bytearray()

        def place(file_name, kind, data):
            nonlocal image
            image += bytes(-len(image) % ALIGN)
            self.placements[file_name] = (kind, IMAGE_OFFSET + len(image), len(data))
            image += data

        self.input_bytes = 0
        for _phase, kind, _target, _base, words, file_name in program.loads:
            if file_name in PER_IMAGE_LOADS:
                self.input_bytes = 8 * words
            if file_name in PER_IMAGE_LOADS or file_name in self.placements:
                continue
            data = program.load_bytes(kind, file_name)
            if kind == "layernorm":     # 6-byte host items, 8-byte stream beats
                data = b"".join(data[i:i + 6] + b"\0\0" for i in range(0, len(data), 6))
            place(file_name, KINDS[kind], data)
        descriptors = program.descriptor_bytes
        place(DESCRIPTOR_FILE, KIND_DESCRIPTOR,
              b"".join(descriptors[i:i + 20] + bytes(4) for i in range(0, len(descriptors), 20)))

        # the run list
        descriptor_kind, descriptor_offset, _length = self.placements[DESCRIPTOR_FILE]
        blocks = program.descriptor_blocks()

        def descriptors(phase):
            return [load_entry(descriptor_kind, 0, first % DESCRIPTOR_SLOTS,
                                        descriptor_offset + 24 * first, 24 * count)
                    for block_phase, first, count in blocks if block_phase == phase]

        entries = descriptors(None)
        for phase, first, _count in program.phases:
            entries += descriptors(phase)
            for load_phase, load_kind, target, base, _words, file_name in program.loads:
                if load_phase != phase:
                    continue
                if file_name in PER_IMAGE_LOADS:
                    kind, offset, length = KINDS[load_kind], INPUT_OFFSET, self.input_bytes
                else:
                    kind, offset, length = self.placements[file_name]
                entries.append(load_entry(kind, target, base, offset, length))
            entries.append(start_entry(first % DESCRIPTOR_SLOTS))
        self.entries = entries
        image += bytes(-len(image) % ALIGN)
        self.list_offset = IMAGE_OFFSET + len(image)
        image += b"".join(entries)
        image += bytes(-len(image) % ALIGN)
        self.image = bytes(image)
        assert INPUT_OFFSET + self.input_bytes <= SIGNATURE_OFFSET
        self.signature = (MAGIC + hashlib.sha256(self.image).digest() + len(self.image).to_bytes(8, "little")
                          + bytes(16))
        assert len(self.signature) == 64 and IMAGE_OFFSET + len(self.image) <= DDR_BYTES

    def upload(self, link, reload=False):
        """Writes the image unless the DDR already holds it; True if it was written."""
        if not reload and link.read_ddr(SIGNATURE_OFFSET, len(self.signature)) == self.signature:
            return False
        link.write_ddr(IMAGE_OFFSET, self.image)
        link.write_ddr(SIGNATURE_OFFSET, self.signature)
        return True

    def load(self, link, file_name, target, base):
        """The DMA_LOAD of one load of program_loads.txt."""
        kind, offset, length = self.placements[file_name]
        link.dma_load(kind, target, base, offset, length)

    def run(self, link, input_data):
        """The RUN flow: the input image to INPUT_OFFSET, then one RUN of the whole list. Returns the busy cycles."""
        assert len(input_data) == self.input_bytes
        link.write_ddr(INPUT_OFFSET, input_data)
        return link.run(self.list_offset, len(self.entries))

    def run_totals(self):
        """What one RUN of the whole list moves, for tb_axi_replay_run_mnist112."""
        return run_totals(self.entries)


def run_totals(entries):
    """What RUN moves for these entries: (LOAD entries, DMA beats, pool items, bias items, LayerNorm items,
    descriptor words, START entries), the numbers the replay testbenches check."""
    loads = beats = pool = bias = layernorm = descriptor_words = starts = 0
    for entry in entries:
        word0, length = int.from_bytes(entry[0:4], "little"), int.from_bytes(entry[12:16], "little")
        if word0 & 0xFF == ENTRY_START:
            starts += 1
            continue
        kind = word0 >> 12 & 3
        loads, beats = loads + 1, beats + length // 8
        if kind == KIND_POOL:
            pool += length // 8
        elif kind == KIND_BIAS:
            bias += length // 32
        elif kind == KIND_LAYERNORM:
            layernorm += length // 8
        else:
            descriptor_words += 5 * (length // 24)
    return loads, beats, pool, bias, layernorm, descriptor_words, starts
