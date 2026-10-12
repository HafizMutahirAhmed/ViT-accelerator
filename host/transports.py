"""Transports for protocol.HostLink: anything with write(bytes) and read(count)."""


class SerialTransport:
    """A serial port (the Arty A7's USB-UART bridge): 8N1, no flow control."""

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
