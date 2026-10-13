# AXI register map

`vit_axi_wrapper` (`rtl/interface/vit_axi_wrapper.v`, logic in
`vit_axi_core.sv`) puts the engine behind one AXI4-Lite slave and one
AXI4-Stream pair. One clock, `aclk`; `aresetn` is synchronous, active low.

## Interfaces

| Interface | Kind | Purpose |
|---|---|---|
| `s_axi` | AXI4-Lite slave, 32-bit data, 12-bit address | control, status, stream target, write windows |
| `s_axis` | AXI4-Stream slave, 64-bit | load a memory from `STREAM_ADDRESS` on |
| `m_axis` | AXI4-Stream master, 64-bit | read a range of a pool buffer out |
| `irq` | level, active high | `done` and `IRQ_ENABLE` |

AXI4-Lite: a write is taken when AWVALID and WVALID are both high; WSTRB is
ignored, every register is written whole. Reads always answer OKAY; an
address that is not listed reads 0.

## Registers

| Offset | Name | Access | Bits |
|---|---|---|---|
| 0x000 | CONTROL | W | 0 start (runs the table at START_INDEX); 1 soft reset; 2 clear done and error. Self-clearing. |
| 0x004 | STATUS | R | 0 busy; 1 done (sticky); 2 error (sticky); 3 a load item is partly received; 4 a store is active |
| 0x008 | START_INDEX | RW | 7:0 first descriptor entry |
| 0x00C | CYCLES | R | clock cycles the engine was busy in the last run |
| 0x010 | CONFIG | R | 7:0 SYS_ROW; 15:8 SYS_COL; 23:16 pool buffers; 31:24 banks per buffer |
| 0x014 | DESCRIPTORS | R | descriptor entries the memory holds (256) |
| 0x018 | IRQ_ENABLE | RW | 0 irq follows done |
| 0x01C | VERSION | R | 0x56495401 ("VIT", version 1) |
| 0x020 | STREAM_TARGET | RW | 2:0 pool buffer id; 9:8 kind: 0 pool, 1 bias store, 2 LayerNorm constants, 3 descriptors |
| 0x024 | STREAM_ADDRESS | RW | next address of the load or store; advances by one per item |
| 0x028 | STORE_COUNT | RW | write N: send N addresses of pool buffer STREAM_TARGET[2:0] on `m_axis`; read: addresses not yet issued |
| 0x100–0x110 | DESC_WORD0–4 | W | the five 32-bit words of one descriptor, word 0 least significant |
| 0x114 | DESC_INDEX | W | writes the five words to this descriptor entry |
| 0x200–0x21C | BIAS_WORD0–7 | W | one INT32 per bank |
| 0x220 | BIAS_ADDRESS | W | writes the eight words to this bias address |
| 0x240 | LN_GAIN | W | 15:0 gain |
| 0x244 | LN_OFFSET | W | offset |
| 0x248 | LN_ADDRESS | W | writes gain and offset to this LayerNorm address |

- **SLVERR.** While the engine runs (and while a start is pending or a store
  is active), a write to a window (0x100–0x2FF), a start and a write to
  STORE_COUNT are not done; they answer SLVERR and set the error flag.
  Everything else answers OKAY.
- **done** is set when the op with `last_op` finishes and cleared by a start
  or by CONTROL bit 2. **error** is also set by a TLAST that ends a load packet
  inside an item.
- **Start** is given to the engine one cycle after the last write has left,
  as the engine requires; a start right after a load is safe.

## Streams

`s_axis`, one item per address, address in STREAM_ADDRESS:

| STREAM_TARGET kind | Beats per item | Layout |
|---|---|---|
| pool buffer | 1 | one byte per bank, bank 0 in TDATA[7:0] |
| bias store | 4 | eight INT32, bank 0 in beat 0 [31:0] |
| LayerNorm constants | 1 | gain in [15:0], offset in [47:16] |
| descriptors | 3 | 160 bits, word 0 in beat 0 [31:0]; the upper 32 bits of beat 2 are ignored |

- TKEEP is ignored. TLAST is not needed; if it comes inside an item, that
  item is dropped and the error flag is set.
- TREADY is low while the engine runs, while a store is active, while a start
  is pending, in a cycle with a register write, and for the 5 cycles a
  descriptor takes to write. Otherwise items are taken back to back, one beat
  per cycle. A load armed while the engine runs simply waits.

`m_axis`: one beat per pool address, bank 0 in TDATA[7:0], TKEEP all ones,
TLAST on the last of STORE_COUNT beats. It tolerates any back-pressure.

An AXI DMA can drive the pair: MM2S into `s_axis` after STREAM_TARGET and
STREAM_ADDRESS are set, S2MM from `m_axis` after STORE_COUNT is written.

## Soft reset

CONTROL bit 1 holds the engine's reset for 8 clocks.

| Restored | Kept |
|---|---|
| the engine's sequencer: idle | START_INDEX, IRQ_ENABLE, STREAM_TARGET, STREAM_ADDRESS |
| a load item partly received: dropped | the descriptor memory |
| a store in progress: stopped, its FIFO emptied | every pool buffer, the bias store, the LayerNorm constants |
| done, error and CYCLES: cleared | the write windows |

A store stopped by a soft reset drops `m_axis` TVALID without a handshake, so
reset the receiver with it. A table stopped half way leaves its buffers as
they were at that moment.

## Typical sequence

1. Write every descriptor: DESC_WORD0–4 then DESC_INDEX, or STREAM_TARGET kind
   3 and a stream.
2. For each phase: set STREAM_TARGET and STREAM_ADDRESS and stream the
   weights, the biases and the LayerNorm constants; write START_INDEX; write
   CONTROL = 1; wait for `irq` or poll STATUS.
3. Set STREAM_TARGET and STREAM_ADDRESS, write STORE_COUNT and take the
   result from `m_axis`.

`axi_host_bridge` does exactly this from the byte protocol of
[HOST_PROTOCOL.md](HOST_PROTOCOL.md).
