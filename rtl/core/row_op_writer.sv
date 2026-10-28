`timescale 1ns / 1ps

module row_op_writer #(
    parameter  int LANES          = vit_config_pkg::SYS_ROW,
    parameter  int DATA_WIDTH     = vit_config_pkg::DATA_WIDTH,
    parameter  int ROWS_TOTAL_MAX = vit_config_pkg::ROWS_TOTAL_MAX,
    parameter  int ROW_LENGTH_MAX = vit_config_pkg::COLS_TOTAL_MAX,
    parameter  int BANK_DEPTH     = vit_config_pkg::BANK_DEPTH_MAX,
    parameter  int DONE_DELAY     = vit_config_pkg::POOL_WRITE_LATENCY + 1,
    localparam int ROWS_WIDTH     = $clog2(ROWS_TOTAL_MAX + 1),
    localparam int LENGTH_WIDTH   = $clog2(ROW_LENGTH_MAX + 1),
    localparam int ADDRESS_WIDTH  = $clog2(BANK_DEPTH)
) (
    input  logic                                    CLK,
    input  logic                                    reset,
    input  logic                                    start,
    input  logic [ROWS_WIDTH-1:0]                   rows_total,
    input  logic [LENGTH_WIDTH-1:0]                 row_length,
    input  logic [ROWS_WIDTH-1:0]                   row_tiles,
    input  logic [ADDRESS_WIDTH-1:0]                output_base,
    input  logic [ADDRESS_WIDTH-1:0]                output_stride,
    input  logic                                    result_valid,
    input  logic signed [LANES-1:0][DATA_WIDTH-1:0] result_data,
    output logic [LANES-1:0]                        write_enable,
    output logic [ADDRESS_WIDTH-1:0]                write_address,
    output logic signed [LANES-1:0][DATA_WIDTH-1:0] write_data,
    output logic                                    done
);

    localparam int REMAINING_WIDTH = $clog2(ROWS_TOTAL_MAX + LANES + 1);

    logic [ADDRESS_WIDTH-1:0]   position;         // address of the next result
    logic [ADDRESS_WIDTH-1:0]   next_tile_start;
    logic [LENGTH_WIDTH-1:0]    column;
    logic [ROWS_WIDTH-1:0]      tile;
    logic [REMAINING_WIDTH-1:0] rows_remaining;   // rows from the current tile on
    logic [LANES-1:0]           lane_valid;
    logic [DONE_DELAY:0]        done_pipeline;

    logic last_column;
    logic last_tile;

    assign last_column = column == row_length - 1'b1;
    assign last_tile   = tile == row_tiles - 1'b1;

    always_comb begin
        for (int lane = 0; lane < LANES; lane++) begin
            lane_valid[lane] = REMAINING_WIDTH'(lane) < rows_remaining;
        end
    end

    always_ff @(posedge CLK) begin
        if (start) begin
            position        <= output_base;
            next_tile_start <= output_base + output_stride;
            column          <= '0;
            tile            <= '0;
            rows_remaining  <= REMAINING_WIDTH'(rows_total);
        end
        else if (result_valid) begin
            if (!last_column) begin
                position <= position + 1'b1;
                column   <= column + 1'b1;
            end
            else begin
                position        <= next_tile_start;
                next_tile_start <= next_tile_start + output_stride;
                column          <= '0;
                tile            <= tile + 1'b1;
                rows_remaining  <= rows_remaining - REMAINING_WIDTH'(LANES);
            end
        end
    end

    always_ff @(posedge CLK) begin
        if (reset) begin
            write_enable  <= '0;
            done_pipeline <= '0;
        end
        else begin
            write_enable  <= result_valid ? lane_valid : '0;
            done_pipeline <= {done_pipeline[DONE_DELAY-1:0], result_valid && last_column && last_tile};
        end
        write_address <= position;
        write_data    <= result_data;
    end

    assign done = done_pipeline[DONE_DELAY];

endmodule
