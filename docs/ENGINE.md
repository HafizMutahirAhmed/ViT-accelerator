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
| `matmul_engine` | `rtl/matmul_engine.sv` | Top level: sequencer, descriptor memory, pool, bias store, masking |
| `matmul_stream_controller` | `rtl/matmul_stream_controller.sv` | Walks the tiles, issues read addresses, lane enables, `first_k`, `tile_done` |
| `systolic_array`, `pe`, `skew` | `rtl/systolic_array.sv`, `rtl/pe.sv`, `rtl/skew.sv` | Output-stationary array; each PE keeps one accumulator |
| `drain`, `requantize` | `rtl/drain.sv`, `rtl/requantize.sv` | Reads a finished tile out, adds bias, requantizes, writes INT8 |
| `operand_buffer` | `rtl/operand_buffer.sv` | One buffer: a block RAM bank per array lane |
| `vit_config_pkg` | `rtl/vit_config_pkg.sv` | Every parameter and the descriptor struct |

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
   (the timing contract is written at the top of `rtl/drain.sv`).

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

`golden/int8_ops.py` is the bit-exact reference for all of this.

## Descriptor format

One op is one `descriptor_t`, a packed struct in `vit_config_pkg`. This table
is generated from the package by `python3 golden/pkg_config.py`; regenerate it
after changing the struct.

| Field | Bits | Width | Meaning |
|---|---|---|---|
| `opcode` | 152:149 | 4 | Operation. 0 = MATMUL; any other value is skipped. |
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

`golden/pkg_config.py` parses the package, so `pack_descriptor()` in Python and
the RTL always agree. `golden/gen_vectors.py` and `golden/gen_deit_vectors.py`
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

At 8x8 the overhead is 25 cycles. `golden/cycle_model.py` implements this
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

## Weight reload between phases

The weight buffer holds one phase at a time and the host reloads it between
phases. Activations stay in their own buffers across a reload.

| Phase | Weight words per bank (8x8) | Bias words per bank |
|---|---|---|
| Patch embedding (768x192) | 18,432 | 24 |
| Encoder layer 0 | 55,296 | 216 |
| Encoder layer 1 | 55,296 | 216 |
| Classifier (192x10) | 384 | 2 |

Inside a layer the weights are stored back to back: qkv at `weight_base` 0,
proj at 13,824, fc1 at 18,432, fc2 at 36,864. The bias bases are 0, 72, 96
and 192.

Reload sequence:

1. Wait for `busy` low.
2. Write the phase's weights into the weight buffer: for each column tile and
   each k, one write of one byte per bank at `weight_base` + tile * K + k.
3. Write the phase's biases into the bias store: one write of one INT32 per
   bank for each column tile.
4. Pulse `start` with the phase's `start_index` and wait for `done`.

The descriptor tables of all phases fit in the descriptor memory together, so
they are written once at start-up and only the weights and biases are
reloaded.

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
- **Only MATMUL.** LayerNorm, softmax, GELU and the residual add are not in
  the engine.
- **No reset on the datapath.** Only control registers are reset.

## Verification and timing

`scripts/regress.sh` runs every testbench and prints one line per run.

| Testbench | What it checks |
|---|---|
| `tb_matmul_engine_deit` | All 39 matmuls of the real pretrained DeiT-Tiny (patch embedding, layers 0 and 1, classifier), each on its real captured input, at 8x8 |
| `tb_matmul_engine_chain` | A five-op attention-head table where each op reads what earlier ops wrote |
| `tb_matmul_engine` | Single-op tables in both drain modes, with bases, padded strides and bias off |
| `tb_drain`, `tb_matmul_stream_controller`, `tb_systolic_array`, `tb_requantize` | The blocks on their own |

Every comparison is bit-exact against `golden/vectors/`. The engine
testbenches also check that no word outside the result is written.

Out-of-context place-and-route on the XC7A200T-1 (`xc7a200tfbg484-1`) with a
6.667 ns clock and the real buffer depths, from
`scripts/ooc_matmul_engine.tcl`:

| | 8x8 | 16x16 |
|---|---|---|
| WNS | +0.165 ns | -0.017 ns (1 endpoint) |
| Estimated Fmax | 153.8 MHz | 149.6 MHz |
| DSP48E1 | 81 (64 PEs, 16 requantizers, 1 drain) | 289 (256 PEs, 32 requantizers, 1 drain) |
| BRAM36 / BRAM18 | 248 / 29 | 256 / 53 |
| LUT / FF | 5,250 / 5,295 | 12,390 / 12,675 |

Every PE maps to exactly one DSP48E1 at both sizes. The buffer depths are the
same tensors at each size: one layer of weights, one 197x768 activation buffer
and three 197x197 buffers. The 8x8 column is the build with the 256-entry
descriptor memory, tagged `engine-v1.1`; commit `eeddbd6`, with 64 entries, had
WNS +0.191 ns.
The 16x16 column was built with 64 entries.

The block RAM count is higher than the tensors need because Vivado rounds a
deep bank up to a power of two. Hand-packing the 8x8 weight bank would take 14
BRAM36 per bank instead of 16.
