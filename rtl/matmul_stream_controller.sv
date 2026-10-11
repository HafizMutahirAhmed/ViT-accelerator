`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: matmul_stream_controller
// Description: Streams a tiled (M x K) @ (K x N) matmul from the operand
// buffers into the systolic array.
//
// Tiles are visited row tile outer, column tile inner. For every tile the
// controller reads k = 0 .. inner_dim-1 from both buffers, then idles for
// PIPELINE_LATENCY cycles so the array is fed zeros, then pulses tile_done.
//   activation_read_address = tile_row * inner_dim + k
//   weight_read_address     = tile_col * inner_dim + k
//
// Outputs come in two timing groups:
//   - the read addresses, in the cycle the buffers are addressed;
//   - activation_lane_enable, weight_lane_enable and first_k_out, one cycle
//     later, lined up with the registered buffer read data.
// The array input for a lane is its read data when the lane enable is high
// and zero otherwise. A lane enable is low outside the k stream and for lanes
// past the matrix edge (global row >= rows_total, global col >= cols_total),
// which is what makes ragged edge tiles work.
//
// tile_done is high for one cycle, in the first cycle where every accumulator
// of the tile is final. tile_row and tile_col give that tile's coordinates in
// the same cycle. The next tile's first slice reaches the array in that cycle
// too, so the accumulators must be captured on the clock edge that ends it.
// all_tiles_done pulses together with the last tile_done.
//
// start is ignored while a matmul is running. rows_total, inner_dim,
// cols_total, row_tiles and col_tiles must hold from start until
// all_tiles_done, with inner_dim >= 1, row_tiles = ceil(rows_total / SYS_ROW)
// and col_tiles = ceil(cols_total / SYS_COL). Control data only, no arithmetic
// on operands.
//////////////////////////////////////////////////////////////////////////////////

module matmul_stream_controller #(
    parameter  int SYS_ROW        = vit_config_pkg::SYS_ROW,
    parameter  int SYS_COL        = vit_config_pkg::SYS_COL,
    parameter  int ROWS_TOTAL_MAX = vit_config_pkg::ROWS_TOTAL_MAX,
    parameter  int INNER_DIM_MAX  = vit_config_pkg::INNER_DIM_MAX,
    parameter  int COLS_TOTAL_MAX = vit_config_pkg::COLS_TOTAL_MAX,
    parameter  int BANK_DEPTH     = vit_config_pkg::BANK_DEPTH,
    localparam int ROWS_WIDTH     = $clog2(ROWS_TOTAL_MAX + 1),
    localparam int INNER_WIDTH    = $clog2(INNER_DIM_MAX + 1),
    localparam int COLS_WIDTH     = $clog2(COLS_TOTAL_MAX + 1),
    localparam int ADDRESS_WIDTH  = $clog2(BANK_DEPTH)
) (
    input  logic                     CLK,
    input  logic                     reset,
    input  logic                     start,
    input  logic [ROWS_WIDTH-1:0]    rows_total,
    input  logic [INNER_WIDTH-1:0]   inner_dim,
    input  logic [COLS_WIDTH-1:0]    cols_total,
    input  logic [ROWS_WIDTH-1:0]    row_tiles,
    input  logic [COLS_WIDTH-1:0]    col_tiles,
    output logic [ADDRESS_WIDTH-1:0] activation_read_address,
    output logic [ADDRESS_WIDTH-1:0] weight_read_address,
    output logic [SYS_ROW-1:0]       activation_lane_enable,
    output logic [SYS_COL-1:0]       weight_lane_enable,
    output logic                     first_k_out,
    output logic [ROWS_WIDTH-1:0]    tile_row,
    output logic [COLS_WIDTH-1:0]    tile_col,
    output logic                     tile_done,
    output logic                     all_tiles_done
);

    // must match systolic_array.PIPELINE_LATENCY
    localparam int PIPELINE_LATENCY = SYS_ROW + SYS_COL - 2;

    localparam int CYCLE_WIDTH     = $clog2(INNER_DIM_MAX + PIPELINE_LATENCY + 1);
    localparam int ROW_INDEX_WIDTH = $clog2(ROWS_TOTAL_MAX + SYS_ROW + 1);
    localparam int COL_INDEX_WIDTH = $clog2(COLS_TOTAL_MAX + SYS_COL + 1);

    // address stage
    logic                       busy;
    logic [CYCLE_WIDTH-1:0]     cycle_count;       // position inside the tile
    logic [ROWS_WIDTH-1:0]      tile_row_count;
    logic [COLS_WIDTH-1:0]      tile_col_count;
    logic [ROW_INDEX_WIDTH-1:0] row_base;          // global row of lane 0
    logic [COL_INDEX_WIDTH-1:0] col_base;          // global column of lane 0
    logic [ADDRESS_WIDTH-1:0]   activation_tile_base;

    logic streaming;
    logic end_of_tile;
    logic last_col_tile;
    logic last_row_tile;

    assign streaming     = cycle_count < CYCLE_WIDTH'(inner_dim);
    assign end_of_tile   = cycle_count == CYCLE_WIDTH'(inner_dim) + CYCLE_WIDTH'(PIPELINE_LATENCY) - 1'b1;
    assign last_col_tile = tile_col_count == col_tiles - 1'b1;
    assign last_row_tile = tile_row_count == row_tiles - 1'b1;

    always_ff @(posedge CLK) begin
        if (reset) begin
            busy <= 1'b0;
        end
        else if (!busy) begin
            if (start) begin
                busy                    <= 1'b1;
                cycle_count             <= '0;
                tile_row_count          <= '0;
                tile_col_count          <= '0;
                row_base                <= '0;
                col_base                <= '0;
                activation_tile_base    <= '0;
                activation_read_address <= '0;
                weight_read_address     <= '0;
            end
        end
        else begin
            cycle_count <= cycle_count + 1'b1;
            if (streaming) begin
                activation_read_address <= activation_read_address + 1'b1;
                weight_read_address     <= weight_read_address + 1'b1;
            end

            // The weight address already sits on the next column tile after the
            // last k. The activation address goes back to the start of the row
            // tile, or on to the next one after the last column tile.
            if (end_of_tile) begin
                cycle_count <= '0;
                if (!last_col_tile) begin
                    tile_col_count          <= tile_col_count + 1'b1;
                    col_base                <= col_base + COL_INDEX_WIDTH'(SYS_COL);
                    activation_read_address <= activation_tile_base;
                end
                else begin
                    tile_col_count          <= '0;
                    col_base                <= '0;
                    weight_read_address     <= '0;
                    tile_row_count          <= tile_row_count + 1'b1;
                    row_base                <= row_base + ROW_INDEX_WIDTH'(SYS_ROW);
                    activation_tile_base    <= activation_tile_base + ADDRESS_WIDTH'(inner_dim);
                    activation_read_address <= activation_tile_base + ADDRESS_WIDTH'(inner_dim);
                    if (last_row_tile) begin
                        busy <= 1'b0;
                    end
                end
            end
        end
    end

    // data stage: one cycle behind the addresses, lined up with buffer read data
    logic [SYS_ROW-1:0] activation_lane_in_range;
    logic [SYS_COL-1:0] weight_lane_in_range;

    always_comb begin
        for (int lane = 0; lane < SYS_ROW; lane++) begin
            activation_lane_in_range[lane] = row_base + ROW_INDEX_WIDTH'(lane) < ROW_INDEX_WIDTH'(rows_total);
        end
        for (int lane = 0; lane < SYS_COL; lane++) begin
            weight_lane_in_range[lane] = col_base + COL_INDEX_WIDTH'(lane) < COL_INDEX_WIDTH'(cols_total);
        end
    end

    logic                  end_of_tile_delayed;
    logic                  last_tile_delayed;
    logic [ROWS_WIDTH-1:0] tile_row_delayed;
    logic [COLS_WIDTH-1:0] tile_col_delayed;

    always_ff @(posedge CLK) begin
        if (reset) begin
            activation_lane_enable <= '0;
            weight_lane_enable     <= '0;
            first_k_out            <= 1'b0;
            end_of_tile_delayed    <= 1'b0;
            last_tile_delayed      <= 1'b0;
            tile_done              <= 1'b0;
            all_tiles_done         <= 1'b0;
        end
        else begin
            activation_lane_enable <= (busy && streaming) ? activation_lane_in_range : '0;
            weight_lane_enable     <= (busy && streaming) ? weight_lane_in_range : '0;
            first_k_out            <= busy && cycle_count == '0;

            // the accumulators are final two cycles after the last flush address cycle
            end_of_tile_delayed <= busy && end_of_tile;
            last_tile_delayed   <= busy && end_of_tile && last_col_tile && last_row_tile;
            tile_done           <= end_of_tile_delayed;
            all_tiles_done      <= last_tile_delayed;
        end
        tile_row_delayed <= tile_row_count;
        tile_col_delayed <= tile_col_count;
        tile_row         <= tile_row_delayed;
        tile_col         <= tile_col_delayed;
    end

endmodule
