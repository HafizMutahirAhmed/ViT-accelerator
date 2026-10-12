`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/09/2026 05:49:23 AM
// Design Name: 
// Module Name: vit_config_pkg
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


package vit_config_pkg;

    parameter  DATA_WIDTH  = 8;
    parameter  ACC_WIDTH   = 32;
    parameter  SHIFT_WIDTH = $clog2(ACC_WIDTH);
    parameter  SCALE_WIDTH = 8;

    parameter  SYS_ROW = 8;
    parameter  SYS_COL = 8;

    // register stages inside a pe before its accumulator: operand, product
    parameter  DSP_LATENCY = 2;

    // largest matmul the stream controller is sized for: (M x K) @ (K x N)
    parameter  ROWS_TOTAL_MAX = 197;
    parameter  INNER_DIM_MAX  = 768;
    parameter  COLS_TOTAL_MAX = 1000;

    // words per operand buffer bank
    parameter  BANK_DEPTH = 4096;

    // cycles from a buffer read address to its read data: read address
    // register, block RAM read, block RAM output register, register after the
    // cascade multiplexer. In matmul_engine the address register is a separate
    // one per pool buffer, in front of the operand_buffer.
    parameter  BUFFER_READ_LATENCY = 4;

    // cycles from the requantize inputs to its result: operand, product,
    // product hold and shifted-value registers
    parameter  REQUANTIZE_LATENCY = 4;

    // register stages matmul_engine puts between a pool write (from the drain
    // or the host) and the block RAM write port
    parameter  POOL_WRITE_LATENCY = 1;

    // drain_mode values
    parameter  DRAIN_ROW_SELECT    = 1'b0;  // one accumulator row per cycle, to column banks
    parameter  DRAIN_COLUMN_SELECT = 1'b1;  // one accumulator column per cycle, to row banks

    // matmul_engine buffer pool, bias store and descriptor table
    parameter  BUFFER_COUNT     = 8;
    parameter  BANK_DEPTH_MAX   = 131072; // deepest bank a descriptor base can address
    parameter  BIAS_DEPTH       = 1024;   // bias words per column bank
    parameter  DESCRIPTOR_COUNT = 256;

    parameter  OPCODE_WIDTH       = 4;
    parameter  ROWS_WIDTH         = $clog2(ROWS_TOTAL_MAX + 1);
    parameter  INNER_WIDTH        = $clog2(INNER_DIM_MAX + 1);
    parameter  COLS_WIDTH         = $clog2(COLS_TOTAL_MAX + 1);
    parameter  BUFFER_ID_WIDTH    = $clog2(BUFFER_COUNT);
    parameter  ADDRESS_WIDTH      = $clog2(BANK_DEPTH_MAX);
    parameter  BIAS_ADDRESS_WIDTH = $clog2(BIAS_DEPTH);

    parameter  OPCODE_MATMUL = 4'd0;
    // row ops: the tensor is walked row tile by row tile, SYS_ROW rows at a time.
    // They use rows_total, row_tiles, cols_total (the row length), the buffer
    // ids and bases, output_stride and last_op like a matmul, and
    //   OPCODE_ADD  activation and weight name the two operands, both in row
    //               banks; scale = multiplier of the first, bias_base[7:0] =
    //               multiplier of the second, shift = their common shift
    //   OPCODE_GELU one operand (activation); bias_base = which table
    //   OPCODE_LAYERNORM  one operand; bias_base = address of the row's first
    //               gain/offset pair in the constants memory, shift = output
    //               shift, inner_dim = epsilon in variance units
    parameter  OPCODE_ADD       = 4'd1;
    parameter  OPCODE_GELU      = 4'd2;
    parameter  OPCODE_LAYERNORM = 4'd3;

    // row ops: passes over a row tile are counted in this many bits
    parameter  ROW_OP_PASS_WIDTH = 2;

    // cycles from a row_op_reader address to the read data at a row op
    // datapath in matmul_engine: address register per buffer, the buffer read,
    // read data register
    parameter  ROW_OP_READ_LATENCY = BUFFER_READ_LATENCY + 2;

    // cycles through residual_add: operand, product, sum, shifted, output
    parameter  RESIDUAL_ADD_LATENCY = 5;

    // gelu_lut: one table per GELU of the model; cycles through it: address
    // register, ROM read, output register
    parameter  GELU_TABLE_COUNT = 2;
    parameter  GELU_LATENCY     = 3;

    // layernorm: gain (signed) and offset (signed) per column, for every
    // LayerNorm of the model, in one host-written memory
    parameter  LAYERNORM_GAIN_WIDTH     = 16;
    parameter  LAYERNORM_OFFSET_WIDTH   = 32;
    parameter  LAYERNORM_CONSTANT_DEPTH = 1024;
    // reciprocal square root table: 2**INDEX_BITS entries of RSQRT_BITS + 1
    // bits; the normalised value has NORM_FRAC fraction bits
    parameter  LAYERNORM_RSQRT_INDEX_BITS = 8;
    parameter  LAYERNORM_RSQRT_BITS       = 12;
    parameter  LAYERNORM_NORM_FRAC        = 8;
    // cycles from a pass 1 input of layernorm to its output: operand, n * x,
    // centre, times reciprocal, shift, times gain, plus offset, shift, clip
    parameter  LAYERNORM_LATENCY = 9;

    // One op of the engine. The first field is the most significant. In the
    // descriptor memory the struct sits in the low DESCRIPTOR_WIDTH bits of a
    // DESCRIPTOR_MEMORY_WIDTH word, which is split into 32-bit words with
    // word 0 the least significant. golden/pkg_config.py reads this struct.
    typedef struct packed {
        logic [OPCODE_WIDTH-1:0]       opcode;
        logic [ROWS_WIDTH-1:0]         rows_total;            // M
        logic [INNER_WIDTH-1:0]        inner_dim;             // K
        logic [COLS_WIDTH-1:0]         cols_total;            // N
        logic [ROWS_WIDTH-1:0]         row_tiles;             // ceil(M / SYS_ROW)
        logic [COLS_WIDTH-1:0]         col_tiles;             // ceil(N / SYS_COL)
        logic [BUFFER_ID_WIDTH-1:0]    activation_buffer_id;
        logic [BUFFER_ID_WIDTH-1:0]    weight_buffer_id;
        logic [BUFFER_ID_WIDTH-1:0]    output_buffer_id;
        logic [ADDRESS_WIDTH-1:0]      activation_base;
        logic [ADDRESS_WIDTH-1:0]      weight_base;
        logic [ADDRESS_WIDTH-1:0]      output_base;
        logic [ADDRESS_WIDTH-1:0]      output_stride;         // words per stored row of an output bank
        logic [BIAS_ADDRESS_WIDTH-1:0] bias_base;
        logic signed [SCALE_WIDTH-1:0] scale;
        logic [SHIFT_WIDTH-1:0]        shift;
        logic                          drain_mode;
        logic                          bias_enable;           // flag: add the bias at drain
        logic                          last_op;               // flag: stop after this op
    } descriptor_t;

    parameter  DESCRIPTOR_WIDTH        = $bits(descriptor_t);
    parameter  DESCRIPTOR_WORDS        = (DESCRIPTOR_WIDTH + 31) / 32;
    parameter  DESCRIPTOR_MEMORY_WIDTH = 32 * DESCRIPTOR_WORDS;

endpackage
