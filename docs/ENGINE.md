# matmul_engine

The matmul engine runs a table of matrix multiplies on a systolic array. It is
the compute core of the DeiT-Tiny accelerator: every linear layer and both
attention products of the model are one engine op each.

One op computes, in INT8 with INT32 accumulation,

    output = clip(((activation @ weight + bias) * scale) >>> shift)

for an M x K activation and a K x N weight. Operands and results live in a pool
of block RAM buffers. An op names its two input buffers and its output buffer
by id, so the output of one op becomes the input of a later op without any
copy. The frozen version is tagged `engine-v1.1`.

## Block diagram

```
                     host port (write, read, start)
                              |
        +---------------------+----------------------+
        |                     |                      |
 descriptor memory        buffer pool            bias store
 (256 ops x 160 bit)   (8 operand buffers)     (INT32, per column)
        |                 |   |      ^                |
   fetch register         |   |      |                |
        |          A role |   | B role                |
   register bank          v   v      |                |
   (current op) ---->  id mux + lane mask             |
        |                   |                         |
        |            array input register             |
        v                   v                         |
 stream controller --> systolic array                 |
 (addresses, lane       SYS_ROW x SYS_COL PEs,        |
  enables, first_k,     one DSP48E1 each, with        |
  tile_done)            input skew                    |
        |                   | all accumulators        |
        |                   v                         v
        +-------------->  drain: select row/column, add bias,
                          requantize (one per lane)
                                    |
                           pool write register
                                    |
                               buffer pool
```

| Module | File | Role |
|---|---|---|
| `matmul_engine` | `rtl/core/matmul_engine.sv` | Top level: sequencer, descriptor memory, pool, bias store, masking |
| `matmul_stream_controller` | `rtl/core/matmul_stream_controller.sv` | Walks the tiles, issues read addresses, lane enables, `first_k`, `tile_done` |
| `systolic_array`, `pe`, `skew` | `rtl/core/systolic_array.sv`, `rtl/core/pe.sv`, `rtl/core/skew.sv` | Output-stationary array; each PE keeps one accumulator |
| `drain`, `requantize` | `rtl/core/drain.sv`, `rtl/core/requantize.sv` | Reads a finished tile out, adds bias, requantizes, writes INT8 |
| `operand_buffer` | `rtl/core/operand_buffer.sv` | One buffer: a block RAM bank per array lane |
| `vit_config_pkg` | `rtl/core/vit_config_pkg.sv` | Every parameter and the descriptor struct |

### How one op runs

1. The sequencer reads the op's descriptor from the descriptor memory into the
   register bank, which holds every field stable for the whole op.
2. The matmul is cut into output tiles of SYS_ROW x SYS_COL results. Tiles are
   visited row tile outer, column tile inner.
3. For each tile the controller reads k = 0 .. K-1 from both input buffers.
   Each cycle delivers one activation per array row and one weight per array
   column. Lanes past the matrix edge are fed zero.
4. The array is output-stationary: PE(row, col) accumulates
   activation[row][k] * weight[k][col] over all k. The `first_k` flag travels
   with the first slice and restarts the accumulator, so there is no clear and
   no reset.
5. After the last k the controller feeds zeros for the pipeline latency, then
   pulses `tile_done`. The drain reads the accumulators one row or one column
   per cycle, adds the bias, requantizes and writes the INT8 results.
6. The array is not stalled during the drain. The next tile starts at once;
   its first slice reaches each accumulator only after the drain has read it
   (the timing contract is written at the top of `rtl/core/drain.sv`).

## Arithmetic

| Quantity | Format |
|---|---|
| Activation, weight, result | signed INT8 |
| Accumulator | signed INT32, wraps modulo 2^32 |
| Bias | signed INT32, same scale as the accumulator, one per output column |
| Bias add | 32-bit, wraps, does not saturate |
| `scale` | signed 8-bit, so a positive scale is 1..127 |
| `shift` | 5-bit, 0..31, arithmetic (floor, no rounding offset) |
| Product `acc * scale` | kept at the full 40 bits |
| Clip | to -128..127 |

`python/int8_model.py` is the bit-exact reference for all of this.

## Descriptor format

One op is one `descriptor_t`, a packed struct in `vit_config_pkg`. This table
is generated from the package by `python3 python/int8_model.py`; regenerate it
after changing the struct.

| Field | Bits | Width | Meaning |
|---|---|---|---|
| `opcode` | 152:149 | 4 | Operation. 0 = MATMUL, 1 = ADD, 2 = GELU, 3 = LAYERNORM, 4 = SOFTMAX; any other value is skipped. |
| `rows_total` | 148:141 | 8 | M, rows of the activation operand and of the result. |
| `inner_dim` | 140:131 | 10 | K, the dimension summed over. At least 1. |
| `cols_total` | 130:121 | 10 | N, columns of the weight operand and of the result. |
| `row_tiles` | 120:113 | 8 | ceil(M / SYS_ROW), worked out by the host. |
| `col_tiles` | 112:103 | 10 | ceil(N / SYS_COL), worked out by the host. |
| `activation_buffer_id` | 102:100 | 3 | Pool buffer read as the activation operand (row banks). |
| `weight_buffer_id` | 99:97 | 3 | Pool buffer read as the weight operand (column banks). |
| `output_buffer_id` | 96:94 | 3 | Pool buffer the result is written to. |
| `activation_base` | 93:77 | 17 | Address of the activation operand's first word in each bank. |
| `weight_base` | 76:60 | 17 | Address of the weight operand's first word in each bank. |
| `output_base` | 59:43 | 17 | Address of the result's first word in each bank. |
| `output_stride` | 42:26 | 17 | Words per stored row of the result. N for column-select and M for row-select, or more to place the result inside a wider tensor. |
| `bias_base` | 25:16 | 10 | Address in the bias store of the bias words for output columns 0 .. SYS_COL-1. |
| `scale` | 15:8 | 8, signed | Requantizer multiplier, signed. |
| `shift` | 7:3 | 5 | Requantizer right shift, arithmetic. |
| `drain_mode` | 2 | 1 | 0 = row-select (result in column banks), 1 = column-select (result in row banks). |
| `bias_enable` | 1 | 1 | 1 = add the bias before requantizing. |
| `last_op` | 0 | 1 | 1 = stop after this op and pulse done. |

The struct is 153 bits. In the descriptor memory it sits in the low bits of a
160-bit word, which the host writes as five 32-bit words; word 0 holds bits
31:0. The field widths follow the package maximums through `$clog2`, so they
change if those maximums change.

`python/int8_model.py` parses the package, so `pack_descriptor()` in Python and
the RTL always agree. `python/gen_vectors.py` and `python/golden_model.py`
emit tables with it.

Rules for a descriptor:

- The three buffer ids must be different. A buffer has one read port and one
  write port. Simulation raises an error if two ids match.
- `row_tiles` and `col_tiles` must be the rounded-up tile counts for the
  array size the bitstream was built with.
- A table is a run of consecutive entries; the last one has `last_op` set.

## Buffer layouts

Every pool buffer has one bank per array lane (max(SYS_ROW, SYS_COL) banks) and
all banks are addressed together. A tensor is stored in one of two layouts.

**Row banks** hold a tensor as an activation operand. Bank `r` holds rows
r, r + SYS_ROW, r + 2*SYS_ROW, ... and each row is contiguous:

    element [row][j]  is in bank  row % SYS_ROW
                      at address  base + (row / SYS_ROW) * stride + j

**Column banks** hold a tensor as a weight operand. Bank `c` holds columns
c, c + SYS_COL, ... and each column is contiguous:

    element [i][col]  is in bank  col % SYS_COL
                      at address  base + (col / SYS_COL) * stride + i

| | Read as activation (A) | Read as weight (B) | Written by column-select drain | Written by row-select drain |
|---|---|---|---|---|
| Layout | row banks | column banks | row banks | column banks |
| Stride | K of the reading op | K of the reading op | `output_stride` (N, or wider) | `output_stride` (M, or wider) |
| Base | `activation_base` | `weight_base` | `output_base` | `output_base` |

So the drain mode of an op is chosen by how its result is used next:

- **Column-select** when the result is the activation operand of a later op.
- **Row-select** when the result is the weight operand of a later op.
- **Column-select also gives a free transpose.** A tensor T in row banks is,
  bit for bit, T-transposed in column banks. That is how K is used as K^T in
  Q @ K^T. This only works because SYS_ROW equals SYS_COL.

For one attention head: Q and K use column-select, V uses row-select, the
scores S = Q @ K^T use column-select, and O = P @ V uses column-select with
`output_stride` 192 and `output_base` head * 64, which places the three heads
side by side in one 197 x 192 tensor.

The bias store has one bank per array column: bias[col] is in bank
col % SYS_COL at address `bias_base` + col / SYS_COL. A slice of a stored
weight that starts at a whole column tile is selected with `weight_base` and
`bias_base` alone.

## Row ops

The non-linear ops run through the same descriptor table as the matmuls. A
row op works on a rows_total x cols_total tensor in row banks, SYS_ROW rows
per cycle: `row_op_reader` issues the read addresses, the op's datapath
processes the data and `row_op_writer` writes the result in row banks. They
reach the pool through registers of their own (a read address register per
buffer, read data registers, a result register), so no matmul path changed.

| Op | Opcode | Module | Passes per row tile | Fields besides the common ones |
|---|---|---|---|---|
| Residual add | 1 | `residual_add` | 1 | `weight_buffer_id`, `weight_base` = second operand; `scale` = multiplier of the first, `bias_base[7:0]` = multiplier of the second, `shift` |
| GELU | 2 | `gelu_lut` | 1 | `bias_base` = table (0, 1: encoder layer) |
| LayerNorm | 3 | `layernorm` | 2 | `bias_base` = address of the row's first gain/offset pair, `shift` = output shift, `inner_dim` = epsilon |
| Softmax | 4 | `softmax` | 3 | `bias_base` = exp table (layer * 3 + head) |

The common fields are `rows_total`, `row_tiles`, `cols_total` (the row
length), `activation_buffer_id` and `activation_base` (the operand),
`output_buffer_id`, `output_base`, `output_stride` and `last_op`.

- **Arithmetic** is in the header of each module and, bit for bit, in
  `python/int8_model.py`.
- **In place.** A row op may write to the buffer it reads; a word is written
  only after it was read. The two operands of an add must be in different
  buffers. The checks are per opcode.
- **Tables.** The GELU, exp and reciprocal-square-root tables are ROMs loaded
  with `$readmemh` from `rtl/rom/input_<size>/`, written by `python/make_model.py`:
  one GELU table per layer and one exp table per head, for all 12 layers
  (`GELU_TABLE_COUNT` 12, `SOFTMAX_TABLE_COUNT` 36). One folder per input
  size (112 or 224), because the tables are built from the calibrated scales.
- **LayerNorm constants.** Gain (INT16) and offset (INT32) per column are
  written by the host: present `host_address`, `host_layernorm_write_gain`,
  `host_layernorm_write_offset` and `host_layernorm_write_enable` for one
  cycle while `busy` is low. The memory holds 1,024 pairs.
- **Row length.** LayerNorm needs a row of at most 255 elements; LayerNorm
  and softmax need a row longer than 6 elements.

Cycles for one op, with tiles = ceil(rows / SYS_ROW) and N the row length
(`python/board_model.py`, equal to the simulation):

| Op | Cycles | 197 rows |
|---|---|---|
| Residual add | 4 + tiles * N + 16 | 4,820 (N = 192) |
| GELU | 4 + tiles * N + 14 | 19,218 (N = 768) |
| LayerNorm | 4 + tiles * (2N + 18) + 20 | 10,074 (N = 192) |
| Softmax | 4 + tiles * (3N + 51) + 18 | 16,072 (N = 197) |

One encoder layer has 2 LayerNorms, 3 softmaxes, 1 GELU and 2 adds: 97,222
cycles, 5.6% on top of its 1,747,050 matmul cycles.

## Host port

All host access happens while `busy` is low. While `busy` is high, host writes
are ignored and read data is not meaningful.

| Signal | Direction | Meaning |
|---|---|---|
| `start` | in | Pulse for one cycle to run the table at `start_index` |
| `start_index` | in | First descriptor entry of the table |
| `busy` | out | High from the cycle after `start` until the table has finished |
| `done` | out | One-cycle pulse when the op with `last_op` has finished |
| `host_descriptor_index` | in | Descriptor entry to write |
| `host_descriptor_write_enable` | in | One bit per 32-bit word of the entry |
| `host_descriptor_write_data` | in | The 32-bit word |
| `host_buffer_id` | in | Pool buffer to write or read |
| `host_address` | in | Word address, shared by the pool and the bias store |
| `host_buffer_write_enable` | in | One bit per bank |
| `host_buffer_write_data` | in | One byte per bank |
| `host_buffer_read_data` | out | One byte per bank |
| `host_bias_write_enable` | in | One bit per column bank |
| `host_bias_write_data` | in | One INT32 per column bank |
| `host_bias_read_data` | out | One INT32 per column bank |

- **Write.** Present the id, address, data and enables for one cycle. The
  write lands one cycle later, because every host write is registered once
  before the block RAM. A new write can be presented every cycle. Wait one
  extra cycle after the last write before pulsing `start`.
- **Read.** Present `host_buffer_id` and `host_address` and hold the id. The
  data is valid BUFFER_READ_LATENCY = 4 cycles later. A new address can be
  presented every cycle; the data comes out in order, 4 cycles behind.
- **Run.** Pulse `start` with `start_index`. Wait for `done`. Results can be
  read from the cycle after `done`.

## Latencies and the cycle formula

| Name (in `vit_config_pkg`) | Value | What it is |
|---|---|---|
| `DSP_LATENCY` | 2 | Operand and product registers inside a PE, before its accumulator |
| `BUFFER_READ_LATENCY` | 4 | Read address register, block RAM read, block RAM output register, register after the cascade multiplexer |
| array input register | 1 | After the buffer id multiplexer and the lane mask |
| `REQUANTIZE_LATENCY` | 4 | Operand, product, second product and shifted-value registers |
| drain pipeline | 2 + `REQUANTIZE_LATENCY` | Select register and biased-accumulator register, then the requantizer |
| `POOL_WRITE_LATENCY` | 1 | Register before the block RAM write port |

For one op on a SYS_ROW x SYS_COL array:

    cycles   = tiles * (K + SYS_ROW + SYS_COL - 2 + DSP_LATENCY) + overhead
    tiles    = ceil(M / SYS_ROW) * ceil(N / SYS_COL)
    overhead = 4                          sequencer: fetch, load, apply, launch
             + BUFFER_READ_LATENCY + 2    last read address to tile_done
             + steps                      SYS_ROW in row-select, SYS_COL in column-select
             + 2 + REQUANTIZE_LATENCY     drain pipeline
             + POOL_WRITE_LATENCY

At 8x8 the overhead is 25 cycles. `python/board_model.py` implements this
formula and checks it against a simulation log; it matches all 39 DeiT-Tiny
ops exactly.

| Op at 8x8 | Shape | Cycles |
|---|---|---|
| Q, K or V, one head | 197x192 @ 192x64 | 41,625 |
| S = Q @ K^T, one head | 197x64 @ 64x197 | 50,025 |
| O = P @ V, one head | 197x197 @ 197x64 | 42,625 |
| Output projection | 197x192 @ 192x192 | 124,825 |
| MLP fc1 | 197x192 @ 192x768 | 499,225 |
| MLP fc2 | 197x768 @ 768x192 | 470,425 |
| One encoder layer (18 ops) | | 1,747,050 |

A layer at 8x8 uses 91.3% of the array's multiply capacity; 7.2% of the
cycles are the zero-fed flush between tiles.

## The forward pass as one descriptor program

`python/layer_program.py` lays the whole N-layer pass out on the buffer pool
and writes the descriptor table, the host load images and the golden tensor of
every stage. `tb_encoder_program` runs it. Nothing is copied between ops.

### Phases

The descriptor table is written once. A phase is a run of descriptors ending
with `last_op`; the host loads the phase's weights, biases and LayerNorm
constants, pulses `start` with the phase's first entry and waits for `done`.
The residual stream stays in the engine from phase to phase.

| Phase | Entries | Ops | Cycles (simulation = model) |
|---|---|---|---|
| embed | 0..1 | patch embedding as a 197x768 @ 768x192 matmul on the im2col input with a zero row 0, then one residual add of a constant 197x192 tensor (class token and position embedding) | 475,245 |
| L0 | 2..27 | encoder layer 0, 26 ops | 1,844,272 |
| L1 | 28..53 | encoder layer 1, 26 ops | 1,844,272 |
| final | 54..56 | final LayerNorm on every row, the two 10-digit classifiers (every row, the class row) | 20,940 |
| all | 57 of 256 | | 4,184,729 |

A layer is, in this order: LN1; Q, K, V of each head (9 matmuls); S = Q @ K^T
of each head; softmax of each head; O = P @ V of each head; output projection;
residual add; LN2; fc1; GELU; fc2; residual add. Its 18 matmuls take 1,747,050
cycles and its 8 row ops 97,222.

### Buffer pool plan

| Id | Name | Holds | Words per bank | BRAM36 as built | BRAM36 if packed by hand |
|---|---|---|---|---|---|
| 0 | X | residual stream 197x192; adds are in place | 4,800 | 16 | 16 |
| 1 | W | weights of the running phase | 55,296 | 128 | 112 |
| 2 | BIG | im2col input; attention output O; fc1 output, GELU in place | 19,200 | 64 | 40 |
| 3 | N | LayerNorm output, then projection output, then fc2 output | 4,800 | 16 | 16 |
| 4 | S | scores of the three heads, softmax in place; the addend; the logits | 14,775 | 32 | 32 |
| 5 | QV | Q of the three heads (row banks) and V (column banks) | 9,528 | 32 | 24 |
| 6 | K | K of the three heads, read as K^T | 4,800 | 16 | 16 |
| 7 | | not used | 16 | 8 BRAM18 | 0 |

Every buffer is used to its last word. Vivado rounds each bank up to a power
of two, which is why the pool takes 304 BRAM36 where 256 would do. Keeping
all three heads' Q, K, V and S at once (the order above) costs 19,200 words
per bank more than computing one head at a time would.

The bias store uses 315 of its 1,024 words per bank in a layer: 216 for the
linear layers and 99 for the constant biases that round S and O.

### What the host writes

| Phase | Weight buffer | Bias store | LayerNorm constants | Other |
|---|---|---|---|---|
| embed | 18,432 | 24 | | im2col input 19,200 (BIG), addend 4,800 (S) |
| L0, L1 | 55,296 | 315 | 384 (two LayerNorms) | |
| final | 24,384 | 127 | 192 | |

These are addresses; each write carries one byte per bank (one INT32 per bank
for the bias store, one gain/offset pair for the LayerNorm constants). Inside
a layer the weights are back to back: qkv at `weight_base` 0, proj at 13,824,
fc1 at 18,432, fc2 at 36,864, with bias bases 0, 72, 96 and 192. The five
LayerNorms keep their constants at 0, 192, 384, 576 and 768.

Load sequence for a phase:

1. Wait for `busy` low.
2. Write the images, one address per cycle.
3. Wait one cycle, pulse `start` with the phase's first entry, wait for `done`.

### What tb_encoder_program checks

It writes everything through the host port only, as above. Each time the
engine moves to the next descriptor it compares the finished op's whole
output tensor with the golden tensor of that stage: 57 stages, in whichever
buffer and layout they are. The host port cannot read while `busy` is high,
so these comparisons read the pool memories directly. After each phase the
phase's last output is also read back through the host port.

## Limits

| Limit | Value | Set by |
|---|---|---|
| M, rows | 197 | `ROWS_TOTAL_MAX` |
| K, inner dimension | 1 to 768 | `INNER_DIM_MAX` |
| N, columns | 1000 | `COLS_TOTAL_MAX` |
| Words per bank a descriptor can address | 131,072 | `BANK_DEPTH_MAX` (17-bit addresses) |
| Words per bank of each pool buffer | per buffer | `BUFFER_DEPTH` parameter of `matmul_engine`, default `BANK_DEPTH` = 4096 |
| Pool buffers | 8 | `BUFFER_COUNT` |
| Bias words per column bank | 1,024 | `BIAS_DEPTH` |
| Descriptor entries | 256 | `DESCRIPTOR_COUNT` |
| Array | square | see below |

- **Square array.** `SYS_ROW` must equal `SYS_COL` in `vit_config_pkg`;
  `drain` stops elaboration otherwise. A buffer written in one layout is read
  back as either operand, and the K-transpose trick needs equal bank counts.
  Testbenches may still override the size per instance, including non-square.
- **Array size.** `SYS_ROW` and `SYS_COL` are parameters of `matmul_engine`.
  The descriptor's `row_tiles` and `col_tiles` must match them.
- **Package-only parameters.** The pool size, bias depth, descriptor count and
  the dimension maximums fix the descriptor field widths, so they cannot be
  overridden per instance.
- **Smallest op.** K + SYS_ROW + SYS_COL - 2 + DSP_LATENCY must be at least
  BUFFER_READ_LATENCY + 1, so that the bias of a tile is read in time. That
  holds for every array of 2x2 or larger.
- **Row ops.** The residual add, GELU, LayerNorm and softmax are in the engine
  on the branch `nonlinear-rtl`; see "Row ops" below. The tag `engine-v1.1`
  has only MATMUL.
- **No reset on the datapath.** Only control registers are reset.

## Verification and timing

`scripts/regress.sh` runs every testbench and prints one line per run.

| Testbench | What it checks |
|---|---|
| `tb_encoder_program` | The whole forward pass as one descriptor program, loaded through the host port, every stage compared (see above) |
| `tb_matmul_engine_nonlinear`, `tb_residual_add`, `tb_gelu_lut`, `tb_layernorm`, `tb_softmax` | The row ops, through the descriptor table and on their own |
| `tb_matmul_engine_opal_kelly` | All 39 matmuls of the real pretrained DeiT-Tiny (patch embedding, layers 0 and 1, classifier), each on its real captured input, at 8x8 |
| `tb_matmul_engine_chain` | A five-op attention-head table where each op reads what earlier ops wrote |
| `tb_matmul_engine` | Single-op tables in both drain modes, with bases, padded strides and bias off |
| `tb_drain`, `tb_matmul_stream_controller`, `tb_systolic_array`, `tb_requantize` | The blocks on their own |

Every comparison is bit-exact against `python/vectors/` and `python/models/`. The engine
testbenches also check that no word outside the result is written.

Out-of-context place-and-route on the XC7A200T-1 (`xc7a200tfbg484-1`) with a
6.667 ns clock, measured with a script that has since been removed (with 2 GELU
and 8 exp tables):

| | 8x8, `block-v1` | 8x8, `engine-v1.1` | 16x16, matmul only |
|---|---|---|---|
| Contents | matmul and the four row ops, pool plan above | matmul only, 5 buffers | matmul only |
| WNS | +0.058 ns | +0.165 ns | -0.017 ns (1 endpoint) |
| Estimated Fmax | 151.3 MHz | 153.8 MHz | 149.6 MHz |
| DSP48E1 | 161 | 81 (64 PEs, 16 requantizers, 1 drain) | 289 |
| BRAM36 / BRAM18 | 317 / 18 | 248 / 29 | 256 / 53 |
| LUT / FF | 10,314 / 9,123 | 5,250 / 5,295 | 12,390 / 12,675 |

Step by step, each at 8x8 with the five-buffer depths of `engine-v1.1`:

| Tag | Adds | WNS | LUT | FF | DSP | BRAM36 / BRAM18 |
|---|---|---|---|---|---|---|
| `nl-add` | residual add, row op reader and writer | +0.117 ns | 6,780 | 6,290 | 97 | 248 / 29 |
| `nl-gelu` | GELU | +0.084 ns | 6,849 | 6,405 | 97 | 248 / 33 |
| `nl-layernorm` | LayerNorm | +0.119 ns | 9,531 | 7,925 | 153 | 249 / 34 |
| `nl-softmax` | softmax | +0.107 ns | 10,200 | 8,921 | 161 | 253 / 34 |

- **Block RAM is the tight resource.** The `block-v1` build uses 326 of the
  365 block RAM tiles (89%): 304 BRAM36 for the pool, 8 for the bias store,
  5 for the row ops' tables and constants, and the rest as BRAM18. Vivado
  rounds each pool bank up to a power of two; packed by hand the pool would
  need 256.
- **Every PE maps to exactly one DSP48E1.** The 80 DSPs of the row ops are 16
  in the add, 56 in LayerNorm and 8 in softmax.
- **The worst paths** are the pool address routing and the drain's
  requantizer, as before the row ops.
- The 16x16 column is the scaling check of `engine-v1.1`, built with a
  64-entry descriptor memory.
