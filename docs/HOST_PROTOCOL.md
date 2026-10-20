# Host protocol

`axi_host_bridge` (`rtl/interface/axi_host_bridge.sv`) runs this protocol on a byte
stream and drives `vit_axi_wrapper` over AXI, and on a board with external
memory also the AXI DMA and the DDR, through a second AXI4-Lite master
(`rtl/interface/accelerator_system.sv`). The byte stream is all a board has
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
| 3 | memory | a memory command (`WRITE_DDR`, `READ_DDR`, `DMA_LOAD`, `RUN`) on a board whose external memory is not calibrated or absent, an error response from the memory system, or a DMA error or timeout |
| 4 | run | a `RUN` stopped: an entry of an unknown kind, the engine's error bit, an error response from the wrapper, or a phase that did not finish within `PHASE_TIMEOUT_CYCLES` (2^28 clocks, 2.7 s at 100 MHz) |

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
| `0x09` | `WRITE_DDR` | 8 | writes the DDR from byte `8 * address` on (`address` counts 8-byte items, so 24 bits reach 128 MiB) | none |
| `0x0A` | `READ_DDR` | none | reads `count` items of 8 bytes of the DDR from byte `8 * address` on | `count` x 8 bytes |
| `0x0C` | `RUN` | none; `count` = number of entries, at least 1 | runs `count` entries of the run list at DDR byte `8 * address` (below) | 4 bytes: busy cycles of all its phases, least significant byte first |
| `0x0B` | `DMA_LOAD` | 8: DDR byte offset (32 bits), then length in bytes (32 bits), each least significant byte first; `count` must be 1 | the DMA copies `length` bytes from the DDR offset into the memory `target` names, from `address` on, as the stream items of [AXI_REGISTER_MAP.md](AXI_REGISTER_MAP.md); `target` bits 2:0 are the pool buffer id, bits 5:4 the kind (0 pool buffer, 1 bias store, 2 LayerNorm constants, 3 descriptors) | none |

`STATUS` flags: bit 0 = engine busy, bit 1 = done seen since the last `START`
or `CLEAR`, bit 2 = sticky error flag, bit 3 = external memory ready (the MIG
has calibrated the DDR).

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
- **The memory commands** need the external memory (STATUS flag bit 3) and
  get the memory status without it; a `WRITE_DDR` payload is then consumed
  and dropped. `WRITE_DDR` and `READ_DDR` are done while the engine runs too;
  `DMA_LOAD` needs the engine idle, like the other loads. A `WRITE_DDR` lands
  while its payload arrives, like the other writes.
- **`DMA_LOAD`** is done only when its checksum is right, and answered when
  the transfer is over: the DMA is idle and the stream's last beat has reached
  the engine. The DDR offset must be a multiple of 8 and the length a whole
  number of items in DDR: 8 bytes per pool address, 32 per bias address, 8 per
  LayerNorm pair (gain in bytes 0..1, offset in 2..5, two zero bytes), 24 per
  descriptor (20 bytes, four zero bytes).
- **A descriptor entry** must be written whole: `WRITE_DESCRIPTOR` items are
  20 bytes.
- **`RUN`** needs the external memory and the engine idle, and is done only
  when its checksum is right. It is answered when the list is done or a step
  failed; a failed step stops the run (the steps before it have been done)
  and answers with the memory status (an error on the memory system, a DMA
  error or timeout) or the run status.

## Run list

A list of 16-byte entries in DDR, four 32-bit words each, least significant
byte first. `RUN` fetches them one after the other and runs each:

| Word | `LOAD` (kind 1) | `START` (kind 2) |
|---|---|---|
| 0 | bits 7:0 kind; bits 15:8 the target byte of `DMA_LOAD` (2:0 pool buffer id, 5:4 kind of memory) | bits 7:0 kind |
| 1 | first address in the target memory | first descriptor entry |
| 2 | DDR byte offset | 0 |
| 3 | length in bytes | 0 |

- **`LOAD`** does what `DMA_LOAD` does with those fields, with the same rules
  for the offset and the length.
- **`START`** starts the engine at the entry, waits for done and adds the
  run's busy cycles to the total that `RUN` answers with.
- Any other kind stops the run with the run status.

The bridge holds no list: `count` bounds the run, so there is no end entry.

## Checksum

CRC-16/CCITT-FALSE: polynomial `0x1021`, initial value `0xFFFF`, most
significant bit first, no reflection, no final XOR. The CRC of the ASCII
string `123456789` is `0x29B1`.

## Running a program

The descriptor program, the images and their addresses are the files
`python/layer_program.py` writes (`program_desc.hex`, `program_phases.txt`,
`program_loads.txt`, `*_image.hex`). For one image:

1. `WRITE_DESCRIPTOR` the whole table, when it fits the 256 entries of the
   descriptor memory.
2. For each phase in `program_phases.txt`:
   1. for a longer program, `WRITE_DESCRIPTOR` the phase's entries: entry e
      goes to slot e mod 256;
   2. send the phase's loads from `program_loads.txt`: `WRITE_BUFFER`,
      `WRITE_BIAS` or `WRITE_LAYERNORM`, or `DMA_LOAD` from DDR;
   3. `START` with the slot of the phase's first entry;
   4. poll `STATUS` until busy is 0.
3. `READ_BUFFER` the classifier scores.

Between two phases every buffer can be read with `READ_BUFFER`, which is how
`python/run_image.py --step --check` compares the stages on the board with the
golden tensors.

What `python/run_image.py` does on the Nexys A7 (`nexys_top_ddr`): the
weights, biases, LayerNorm constants, the descriptor table and the run list go
into DDR once per power-up and model, as `python/host.py` lays them out
(from byte 1 MiB on, each load on a 64-byte boundary, with a 64-byte signature
just below 1 MiB that names the image). A run reads the signature with
`READ_DDR` and uploads the image with `WRITE_DDR` only if the DDR does not
already hold it. Then, per image, it writes the image to DDR byte 0 with
`WRITE_DDR`, sends one `RUN` (the list loads the descriptors, all at once or
per phase, then per phase every load and a `START`) and reads the scores with
`READ_BUFFER`. The stages before the last phase are overwritten by then;
`--step` runs the steps above from the host instead, with a `DMA_LOAD` per
load, and can read every stage.

## UART (Nexys A7)

8 data bits, no parity, 1 stop bit, least significant bit first, 921,600 baud
by default (`BAUD_RATE` parameter of `boards/nexys_a7/nexys_top_ddr.sv`). No
hardware flow control: the command and response discipline above is enough,
because the FPGA handles a byte in far less than a byte time.
