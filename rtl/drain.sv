`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: drain
// Description: Reads a finished tile out of the systolic array, adds the bias,
// requantizes and writes the INT8 results to an output operand_buffer.
//
// drain_mode picks the read order and the output layout:
//   DRAIN_ROW_SELECT     one accumulator row per cycle, SYS_ROW cycles. Lane c
//                        is tile column c and goes to column bank c at address
//                        tile_col * rows_total + global_row, the layout of a
//                        weight-side operand whose inner dimension is M.
//   DRAIN_COLUMN_SELECT  one accumulator column per cycle, SYS_COL cycles. Lane
//                        r is tile row r and goes to row bank r at address
//                        tile_row * cols_total + global_col, the layout of an
//                        activation-side operand whose inner dimension is N.
// All lanes share one write address. There are max(SYS_ROW, SYS_COL) lanes.
// Rows and columns past the matrix edge are never written.
//
// Fixed point: accumulator_in and bias_in are signed ACC_WIDTH integers at the
// same scale. Their sum wraps modulo 2**ACC_WIDTH. Each lane then goes through
// requantize: clip((sum * scale) >>> shift_value) to a signed DATA_WIDTH
// integer. bias_in[c] is the bias of tile column c.
//
// Timing: the read starts in the tile_done cycle, step s in cycle tile_done + s,
// and the array is not stalled. This relies on the next tile's first slice
// reaching the array no earlier than the tile_done cycle: pe(row, col) is then
// overwritten row + col cycles later, always after the drain has read it.
// tile_row, tile_col and bias_in are sampled when tile_done is high. A write
// appears two cycles after its read step. drain_mode, rows_total, cols_total,
// scale and shift_value must hold until drain_busy is low after the last tile.
//////////////////////////////////////////////////////////////////////////////////

module drain #(
    parameter  int SYS_ROW        = vit_config_pkg::SYS_ROW,
    parameter  int SYS_COL        = vit_config_pkg::SYS_COL,
    parameter  int DATA_WIDTH     = vit_config_pkg::DATA_WIDTH,
    parameter  int ACC_WIDTH      = vit_config_pkg::ACC_WIDTH,
    parameter  int SCALE_WIDTH    = vit_config_pkg::SCALE_WIDTH,
    parameter  int SHIFT_WIDTH    = vit_config_pkg::SHIFT_WIDTH,
    parameter  int ROWS_TOTAL_MAX = vit_config_pkg::ROWS_TOTAL_MAX,
    parameter  int COLS_TOTAL_MAX = vit_config_pkg::COLS_TOTAL_MAX,
    parameter  int BANK_DEPTH     = vit_config_pkg::BANK_DEPTH,
    localparam int LANES          = (SYS_ROW > SYS_COL) ? SYS_ROW : SYS_COL,
    localparam int ROWS_WIDTH     = $clog2(ROWS_TOTAL_MAX + 1),
    localparam int COLS_WIDTH     = $clog2(COLS_TOTAL_MAX + 1),
    localparam int ADDRESS_WIDTH  = $clog2(BANK_DEPTH)
) (
    input  logic                                                  CLK,
    input  logic                                                  reset,
    input  logic                                                  drain_mode,
    input  logic                                                  tile_done,
    input  logic [ROWS_WIDTH-1:0]                                 tile_row,
    input  logic [COLS_WIDTH-1:0]                                 tile_col,
    input  logic [ROWS_WIDTH-1:0]                                 rows_total,
    input  logic [COLS_WIDTH-1:0]                                 cols_total,
    input  logic signed [SYS_ROW-1:0][SYS_COL-1:0][ACC_WIDTH-1:0] accumulator_in,
    input  logic signed [SYS_COL-1:0][ACC_WIDTH-1:0]              bias_in,
    input  logic signed [SCALE_WIDTH-1:0]                         scale,
    input  logic [SHIFT_WIDTH-1:0]                                shift_value,
    output logic [LANES-1:0]                                      write_enable,
    output logic [ADDRESS_WIDTH-1:0]                              write_address,
    output logic signed [LANES-1:0][DATA_WIDTH-1:0]               write_data,
    output logic                                                  drain_busy
);

    import vit_config_pkg::DRAIN_COLUMN_SELECT;

    localparam int STEP_WIDTH      = (LANES > 1) ? $clog2(LANES) : 1;
    localparam int ROW_INDEX_WIDTH = $clog2(ROWS_TOTAL_MAX + SYS_ROW + 1);
    localparam int COL_INDEX_WIDTH = $clog2(COLS_TOTAL_MAX + SYS_COL + 1);

    // The output buffer is reused as an operand of a later matmul, where the
    // row banks and the column banks are the same memories.
    generate
        if (vit_config_pkg::SYS_ROW != vit_config_pkg::SYS_COL) begin: square_array_check
            $error("vit_config_pkg: SYS_ROW (%0d) must equal SYS_COL (%0d)",
                   vit_config_pkg::SYS_ROW, vit_config_pkg::SYS_COL);
        end
    endgenerate

    // read step: step 0 is the tile_done cycle itself
    logic                  draining;
    logic [STEP_WIDTH-1:0] step_count;
    logic                  step_valid;
    logic [STEP_WIDTH-1:0] step_index;
    logic [STEP_WIDTH-1:0] step_last;

    assign step_valid = tile_done || draining;
    assign step_index = tile_done ? '0 : step_count;
    assign step_last  = (drain_mode == DRAIN_COLUMN_SELECT) ? STEP_WIDTH'(SYS_COL - 1) : STEP_WIDTH'(SYS_ROW - 1);

    always_ff @(posedge CLK) begin
        if (reset) begin
            draining <= 1'b0;
        end
        else if (step_valid) begin
            draining   <= step_index != step_last;
            step_count <= step_index + 1'b1;
        end
    end

    // tile context, sampled at tile_done and used from the next cycle on
    logic [ROW_INDEX_WIDTH-1:0]             row_base;       // global row of tile row 0
    logic [COL_INDEX_WIDTH-1:0]             col_base;       // global column of tile column 0
    logic [ADDRESS_WIDTH-1:0]               start_address;  // write address of step 0
    logic [SYS_COL-1:0][ACC_WIDTH-1:0]      bias_latched;

    logic [ADDRESS_WIDTH-1:0] bank_row_index;  // which stored row of the bank
    logic [ADDRESS_WIDTH-1:0] bank_row_length;
    logic [ADDRESS_WIDTH-1:0] first_step_offset;

    always_comb begin
        if (drain_mode == DRAIN_COLUMN_SELECT) begin
            bank_row_index    = ADDRESS_WIDTH'(tile_row);
            bank_row_length   = ADDRESS_WIDTH'(cols_total);
            first_step_offset = ADDRESS_WIDTH'(tile_col) * ADDRESS_WIDTH'(SYS_COL);
        end
        else begin
            bank_row_index    = ADDRESS_WIDTH'(tile_col);
            bank_row_length   = ADDRESS_WIDTH'(rows_total);
            first_step_offset = ADDRESS_WIDTH'(tile_row) * ADDRESS_WIDTH'(SYS_ROW);
        end
    end

    always_ff @(posedge CLK) begin
        if (tile_done) begin
            row_base      <= ROW_INDEX_WIDTH'(tile_row) * ROW_INDEX_WIDTH'(SYS_ROW);
            col_base      <= COL_INDEX_WIDTH'(tile_col) * COL_INDEX_WIDTH'(SYS_COL);
            start_address <= bank_row_index * bank_row_length + first_step_offset;
            bias_latched  <= bias_in;
        end
    end

    // stage 0: select one accumulator row or column, straight from the array
    logic [LANES-1:0][ACC_WIDTH-1:0] selected_accumulator;
    logic [LANES-1:0][ACC_WIDTH-1:0] stage_accumulator;
    logic                            stage_valid;
    logic [STEP_WIDTH-1:0]           stage_step;

    always_comb begin
        for (int lane = 0; lane < LANES; lane++) begin
            selected_accumulator[lane] = '0;
            if (drain_mode == DRAIN_COLUMN_SELECT) begin
                if (lane < SYS_ROW) begin
                    selected_accumulator[lane] = accumulator_in[lane][step_index];
                end
            end
            else if (lane < SYS_COL) begin
                selected_accumulator[lane] = accumulator_in[step_index][lane];
            end
        end
    end

    always_ff @(posedge CLK) begin
        if (reset) begin
            stage_valid <= 1'b0;
        end
        else begin
            stage_valid <= step_valid;
        end
        stage_step        <= step_index;
        stage_accumulator <= selected_accumulator;
    end

    // stage 1: bias add, edge masking and write address
    logic [LANES-1:0]                row_in_range;
    logic [LANES-1:0]                col_in_range;
    logic [LANES-1:0][ACC_WIDTH-1:0] lane_bias;
    logic [LANES-1:0][ACC_WIDTH-1:0] biased_accumulator;
    logic [LANES-1:0]                lane_write;
    logic                            step_in_range;

    always_comb begin
        row_in_range = '0;
        col_in_range = '0;
        for (int row = 0; row < SYS_ROW; row++) begin
            row_in_range[row] = row_base + ROW_INDEX_WIDTH'(row) < ROW_INDEX_WIDTH'(rows_total);
        end
        for (int col = 0; col < SYS_COL; col++) begin
            col_in_range[col] = col_base + COL_INDEX_WIDTH'(col) < COL_INDEX_WIDTH'(cols_total);
        end

        for (int lane = 0; lane < LANES; lane++) begin
            lane_bias[lane] = '0;
            if (drain_mode == DRAIN_COLUMN_SELECT) begin
                lane_bias[lane] = bias_latched[stage_step];
            end
            else if (lane < SYS_COL) begin
                lane_bias[lane] = bias_latched[lane];
            end
            biased_accumulator[lane] = stage_accumulator[lane] + lane_bias[lane];
        end

        if (drain_mode == DRAIN_COLUMN_SELECT) begin
            step_in_range = col_in_range[stage_step];
            lane_write    = (stage_valid && step_in_range) ? row_in_range : '0;
        end
        else begin
            step_in_range = row_in_range[stage_step];
            lane_write    = (stage_valid && step_in_range) ? col_in_range : '0;
        end
    end

    // stage 2: requantize; valid_out is the registered lane write enable
    always_ff @(posedge CLK) begin
        write_address <= start_address + ADDRESS_WIDTH'(stage_step);
    end

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane++) begin: lane_requantize
            requantize #(
                .DATA_WIDTH  (DATA_WIDTH),
                .ACC_WIDTH   (ACC_WIDTH),
                .SCALE_WIDTH (SCALE_WIDTH),
                .SHIFT_WIDTH (SHIFT_WIDTH)
            ) requantize_inst (
                .CLK                (CLK),
                .valid_in           (lane_write[lane]),
                .unquantized_result (biased_accumulator[lane]),
                .scale              (scale),
                .shift_value        (shift_value),
                .valid_out          (write_enable[lane]),
                .quantized_result   (write_data[lane])
            );
        end
    endgenerate

    assign drain_busy = step_valid || stage_valid || (|write_enable);

endmodule
