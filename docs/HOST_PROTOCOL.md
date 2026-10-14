# Host protocol

`axi_host_bridge` (`rtl/interface/axi_host_bridge.sv`) runs this protocol on a byte
stream and drives `vit_axi_wrapper` over AXI. The byte stream is all a board has
to provide: a UART on the Nexys A7 (`boards/nexys_a7/`), FrontPanel pipes on the
XEM7310 later. The engine, the bridge and the host software (`python/`) are the
same on every board.

The host sends a command packet and waits for the response before it sends
the next one. There is no other flow control.

## Command packet (host to FPGA)

| Byte | Field | Meaning |
|---|---|---|
| 0 | start | `0xA5` |
| 1 | command | see below |
| 2 | target | pool buffer id for the buffer commands, otherwise 0 |
| 3..5 | address | 24 bits, least significant byte first: first address, descriptor entry or start entry |
| 6..7 | count | 16 bits, least significant byte first: number of items |
| 8.. | payload | count items for the write commands, nothing otherwise |
| last 2 | checksum | CRC-16 of bytes 1 .. end of payload, least significant byte first |

Bytes outside a packet are ignored until a `0xA5` arrives.

## Response packet (FPGA to host)

| Byte | Field | Meaning |
|---|---|---|
| 0 | start | `0x5A` |
| 1 | status | see below; 0 means the command was done |
| 2.. | payload | read data or status data, only when status is 0 |
| last 2 | checksum | CRC-16 of bytes 1 .. end of payload, least significant byte first |

Status bits:

| Bit | Name | Meaning |
|---|---|---|
| 0 | checksum | the packet's checksum was wrong |
| 1 | command | unknown command |
| 2 | busy | the engine was running; the command was not done |

Any non-zero status also sets the sticky error flag (the error LED), which
stays set until `CLEAR`.

## Commands

| Code | Name | Item (bytes) | Does | Response payload |
|---|---|---|---|---|
| `0x01` | `WRITE_BUFFER` | 8: one byte per bank, bank 0 first | writes pool buffer `target` at `address`, `address + 1`, ... | none |
| `0x02` | `WRITE_BIAS` | 32: one INT32 per bank, bank 0 first, each least significant byte first | writes the bias store from `address` on | none |
| `0x03` | `WRITE_LAYERNORM` | 6: gain (INT16), then offset (INT32), each least significant byte first | writes the LayerNorm constants from `address` on | none |
| `0x04` | `WRITE_DESCRIPTOR` | 20: five 32-bit words, word 0 first, each least significant byte first | writes descriptor entries `address`, `address + 1`, ... | none |
| `0x05` | `START` | none | pulses `start` with `start_index` = `address` | none |
| `0x06` | `READ_BUFFER` | none | reads `count` addresses of pool buffer `target` from `address` on | `count` x 8 bytes, bank 0 first |
| `0x07` | `STATUS` | none | nothing | 1 byte flags, then 4 bytes: busy cycles of the last run, least significant byte first |
| `0x08` | `CLEAR` | none | clears the sticky error flag and the done flag | none |

`STATUS` flags: bit 0 = engine busy, bit 1 = done seen since the last `START`
or `CLEAR`, bit 2 = sticky error flag.

Rules:

- **A write takes effect while its payload arrives**, one item per engine
  write, before the checksum is known. If the response reports a checksum
  error, send the same packet again: writes of the same data to the same
  addresses are harmless.
- **`START` is only done when its checksum is right** and the engine is idle.
- **While the engine runs**, only `STATUS` and `CLEAR` are done. Any other
  command has its payload consumed and discarded and is answered with the
  busy status. The host polls `STATUS` until the busy flag is 0.
- **`count`** is at most 65,535 items; a larger region takes several packets.
- **A descriptor entry** must be written whole: `WRITE_DESCRIPTOR` items are
  20 bytes.

## Checksum

CRC-16/CCITT-FALSE: polynomial `0x1021`, initial value `0xFFFF`, most
significant bit first, no reflection, no final XOR. The CRC of the ASCII
string `123456789` is `0x29B1`.

## Running a program

The descriptor program, the images and their addresses are the files
`python/layer_program.py` writes (`program_desc.hex`, `program_phases.txt`,
`program_loads.txt`, `*_image.hex`). For one image:

1. `WRITE_DESCRIPTOR` the whole table (once per power-up).
2. For each phase in `program_phases.txt`:
   1. send the phase's loads from `program_loads.txt`: `WRITE_BUFFER`,
      `WRITE_BIAS` or `WRITE_LAYERNORM`;
   2. `START` with the phase's first entry;
   3. poll `STATUS` until busy is 0.
3. `READ_BUFFER` the classifier scores.

Between two phases every buffer can be read with `READ_BUFFER`, which is how
`python/run_image.py --check` compares the stages on the board with the golden
tensors.

## UART (Nexys A7)

8 data bits, no parity, 1 stop bit, least significant bit first, 921,600 baud
by default (`BAUD_RATE` parameter of `boards/nexys_a7/nexys_top_axi.sv`). No
hardware flow control: the command and response discipline above is enough,
because the FPGA handles a byte in far less than a byte time.
