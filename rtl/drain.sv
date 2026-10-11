`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: drain
// Description: Reads a finished tile out of the systolic array, adds the bias,
// requantizes and writes the INT8 results to an output operand_buffer.
//
// drain_mode picks the read order and the output layout:
//   DRAIN_ROW_SELECT     one accumulator row per cycle, SYS_ROW cycles. Lane c
//                        is tile column c and goes to column bank c at address
//                        output_base + tile_col * output_stride + global_row.
//                        With output_stride = M this is the layout of a
//                        weight-side operand whose inner dimension is M.
//   DRAIN_COLUMN_SELECT  one accumulator column per cycle, SYS_COL cycles. Lane
//                        r is tile row r and goes to row bank r at address
//                        output_base + tile_row * output_stride + global_col.
//                        With output_stride = N this is the layout of an
//                        activation-side operand whose inner dimension is N.
// All lanes share one write address. There are max(SYS_ROW, SYS_COL) lanes.
// Rows and columns past the matrix edge are never written. A larger
// output_stride with an output_base offset places the result inside a wider
// stored tensor.
//
// Fixed point: accumulator_in and bias_in are signed ACC_WIDTH integers at the
// same scale. Their sum wraps modulo 2**ACC_WIDTH. Each lane then goes through
// requantize: clip((sum * scale) >>> shift_value) to a signed DATA_WIDTH
// integer. bias_in[c] is the bias of tile column c. With bias_enable low the
// bias add is skipped and the accumulator goes to requantize unchanged.
//
// Timing contract. The array is not stalled and there is no snapshot of the
// accumulators: the drain reads each one before the next tile overwrites it.
//   - T is the cycle in which tile_done is high. Every accumulator of the tile
//     is final in cycle T.
//   - Read step s happens in cycle T + s and is registered on the clock edge
//     that ends that cycle. Row-select reads pe(s, col) for every col,
//     column-select reads pe(row, s) for every row.
//   - F is the cycle in which the next tile's first slice (first_k_in high) is
//     at the array inputs. With D = vit_config_pkg::DSP_LATENCY register
//     stages inside the pe, pe(row, col) takes its new value on the clock edge
//     that ends cycle F + row + col + D, and holds the old one until then.
//   - So the caller must guarantee F + D >= T. The margin is F - T + D + col
//     cycles in row-select and F - T + D + row cycles in column-select.
//   - matmul_stream_controller flushes for D cycles more than the array skew
//     needs and gives F = T exactly, so the smallest margin is D = 2 cycles,
//     at pe(0, 0): it is read in cycle T and overwritten at the end of T + 2.
//   - Consequences: at most D register stages may be added between the array
//     and this module's read, or to tile_done, and the next tile must not
//     start earlier than the controller starts it.
//   - After the last tile nothing overwrites the accumulators as long as the
//     array is fed zeros with first_k_in low.
// tile_row, tile_col and bias_in are sampled when tile_done is high and must
// stay in use for the whole drain, so the next tile_done must come at least
// SYS_ROW cycles later in row-select and SYS_COL cycles later in column-select.
// The controller's tile period, inner_dim + SYS_ROW + SYS_COL - 2, always
// meets that. A write appears 2 + vit_config_pkg::REQUANTIZE_LATENCY = 6
// cycles after its read step: select register, biased-accumulator register,
// then the requantizer. WRITE_LATENCY is the number of register stages the
// user puts between the write outputs and the memory; drain_busy stays high
// until the last write has passed them. The write address is worked out over two cycles
// (tile index times output_stride, then the adds). drain_mode,
// bias_enable, rows_total, cols_total, output_base, output_stride, scale and
// shift_value must hold until drain_busy is low after the last tile.
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
    parameter  int WRITE_LATENCY  = 0,
    localparam int LANES          = (SYS_ROW > SYS_COL) ? SYS_ROW : SYS_COL,
    localparam int ROWS_WIDTH     = $clog2(ROWS_TOTAL_MAX + 1),
    localparam int COLS_WIDTH     = $clog2(COLS_TOTAL_MAX + 1),
    localparam int ADDRESS_WIDTH  = $clog2(BANK_DEPTH)
) (
    input  logic                                                  CLK,
    input  logic                                                  reset,
    input  logic                                                  drain_mode,
    input  logic                                                  bias_enable,
    input  logic                                                  tile_done,
    input  logic [ROWS_WIDTH-1:0]                                 tile_row,
    input  logic [COLS_WIDTH-1:0]                                 tile_col,
    input  logic [ROWS_WIDTH-1:0]                                 rows_total,
    input  logic [COLS_WIDTH-1:0]                                 cols_total,
    input  logic [ADDRESS_WIDTH-1:0]                              output_base,
    input  logic [ADDRESS_WIDTH-1:0]                              output_stride,
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
    logic [SYS_COL-1:0][ACC_WIDTH-1:0]      bias_latched;

    logic [ADDRESS_WIDTH-1:0] bank_row_index;  // which stored row of the bank
    logic [ADDRESS_WIDTH-1:0] first_step_offset;

    always_comb begin
        if (drain_mode == DRAIN_COLUMN_SELECT) begin
            bank_row_index    = ADDRESS_WIDTH'(tile_row);
            first_step_offset = ADDRESS_WIDTH'(tile_col) * ADDRESS_WIDTH'(SYS_COL);
        end
        else begin
            bank_row_index    = ADDRESS_WIDTH'(tile_col);
            first_step_offset = ADDRESS_WIDTH'(tile_row) * ADDRESS_WIDTH'(SYS_ROW);
        end
    end

    always_ff @(posedge CLK) begin
        if (tile_done) begin
            row_base     <= ROW_INDEX_WIDTH'(tile_row) * ROW_INDEX_WIDTH'(SYS_ROW);
            col_base     <= COL_INDEX_WIDTH'(tile_col) * COL_INDEX_WIDTH'(SYS_COL);
            bias_latched <= bias_in;
        end
    end

    // write address of step 0, in two cycles: the product and the offset one
    // cycle after tile_done, their sum one cycle later. The first use is the
    // cycle after that.
    logic                     tile_done_delayed;
    logic [ADDRESS_WIDTH-1:0] address_product;
    logic [ADDRESS_WIDTH-1:0] address_offset;
    logic [ADDRESS_WIDTH-1:0] start_address;

    always_ff @(posedge CLK) begin
        if (reset) begin
            tile_done_delayed <= 1'b0;
        end
        else begin
            tile_done_delayed <= tile_done;
        end
        if (tile_done) begin
            address_product <= bank_row_index * output_stride;
            address_offset  <= output_base + first_step_offset;
        end
        if (tile_done_delayed) begin
            start_address <= address_product + address_offset;
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
            if (bias_enable) begin
                if (drain_mode == DRAIN_COLUMN_SELECT) begin
                    lane_bias[lane] = bias_latched[stage_step];
                end
                else if (lane < SYS_COL) begin
                    lane_bias[lane] = bias_latched[lane];
                end
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

    // stage 2: register the biased accumulator and the lane write enables
    logic [LANES-1:0][ACC_WIDTH-1:0] biased_accumulator_reg;
    logic [LANES-1:0]                lane_write_reg;
    logic [STEP_WIDTH-1:0]           biased_step;

    always_ff @(posedge CLK) begin
        if (reset) begin
            lane_write_reg <= '0;
        end
        else begin
            lane_write_reg <= lane_write;
        end
        biased_accumulator_reg <= biased_accumulator;
        biased_step            <= stage_step;
    end

    // write address: start_address + step, then delayed to line up with the
    // requantizer output
    localparam int REQUANTIZE_LATENCY = vit_config_pkg::REQUANTIZE_LATENCY;

    logic [REQUANTIZE_LATENCY-1:0][ADDRESS_WIDTH-1:0] address_pipeline;

    always_ff @(posedge CLK) begin
        address_pipeline[0] <= start_address + ADDRESS_WIDTH'(biased_step);
        for (int stage = 1; stage < REQUANTIZE_LATENCY; stage++) begin
            address_pipeline[stage] <= address_pipeline[stage-1];
        end
    end

    assign write_address = address_pipeline[REQUANTIZE_LATENCY-1];

    // scale and shift, registered once per group of lanes so that no single
    // register drives every requantizer
    localparam int LANE_GROUP_SIZE = 4;
    localparam int LANE_GROUPS     = (LANES + LANE_GROUP_SIZE - 1) / LANE_GROUP_SIZE;

    (* dont_touch = "true" *) logic signed [LANE_GROUPS-1:0][SCALE_WIDTH-1:0] scale_group;
    (* dont_touch = "true" *) logic [LANE_GROUPS-1:0][SHIFT_WIDTH-1:0]        shift_group;

    always_ff @(posedge CLK) begin
        for (int group = 0; group < LANE_GROUPS; group++) begin
            scale_group[group] <= scale;
            shift_group[group] <= shift_value;
        end
    end

    // requantize; valid_out is the lane write enable
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
                .valid_in           (lane_write_reg[lane]),
                .unquantized_result (biased_accumulator_reg[lane]),
                .scale              (scale_group[lane / LANE_GROUP_SIZE]),
                .shift_value        (shift_group[lane / LANE_GROUP_SIZE]),
                .valid_out          (write_enable[lane]),
                .quantized_result   (write_data[lane])
            );
        end
    endgenerate

    // busy from the first read step until the last step has left the pipeline
    // and the write stages after it, whether or not that step writes anything
    localparam int DRAIN_LATENCY = 2 + REQUANTIZE_LATENCY + WRITE_LATENCY;

    logic [DRAIN_LATENCY-1:0] activity;

    always_ff @(posedge CLK) begin
        if (reset) begin
            activity <= '0;
        end
        else begin
            activity <= {activity[DRAIN_LATENCY-2:0], step_valid};
        end
    end

    assign drain_busy = step_valid || (|activity);

endmodule
