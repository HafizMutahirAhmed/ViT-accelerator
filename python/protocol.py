"""The host side of docs/HOST_PROTOCOL.md, on any transport.

A transport is an object with write(bytes) and read(count) -> bytes (exactly
count bytes, or fewer on a timeout). transports.py has one for a serial port
and one for the software model of the board; an Opal Kelly one only has to
provide the same two methods.
"""

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
