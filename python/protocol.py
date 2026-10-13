"""The host side of docs/HOST_PROTOCOL.md, on any transport.

A transport is an object with write(bytes) and read(count) -> bytes (exactly
count bytes, or fewer on a timeout). transports.py has one for a serial port
and one for the software model of the board; an Opal Kelly one only has to
provide the same two methods.
"""

START_BYTE, RESPONSE_BYTE = 0xA5, 0x5A
WRITE_BUFFER, WRITE_BIAS, WRITE_LAYERNORM, WRITE_DESCRIPTOR = 0x01, 0x02, 0x03, 0x04
START, READ_BUFFER, STATUS, CLEAR = 0x05, 0x06, 0x07, 0x08
ITEM_BYTES = {WRITE_BUFFER: 8, WRITE_BIAS: 32, WRITE_LAYERNORM: 6, WRITE_DESCRIPTOR: 20}
STATUS_CHECKSUM, STATUS_COMMAND, STATUS_BUSY = 1, 2, 4
FLAG_BUSY, FLAG_DONE, FLAG_ERROR = 1, 2, 4
MAX_ITEMS = 4096          # items per packet; the protocol allows 65,535


def crc16(data, crc=0xFFFF):
    """CRC-16/CCITT-FALSE."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


assert crc16(b"123456789") == 0x29B1


class ProtocolError(Exception):
    pass


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
                raise ProtocolError(f"command {code:#04x} refused with status {status} "
                                    f"({'busy' if status & STATUS_BUSY else 'unknown command'})")
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
