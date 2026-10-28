`timescale 1ns / 1ps

module systolic_array #(
    parameter int SYS_ROW    = vit_config_pkg::SYS_ROW,
    parameter int SYS_COL    = vit_config_pkg::SYS_COL,
    parameter int DATA_WIDTH = vit_config_pkg::DATA_WIDTH,
    parameter int ACC_WIDTH  = vit_config_pkg::ACC_WIDTH
) (
    input  logic                                                  CLK,
    input  logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0]             activation_in,
    input  logic signed [SYS_COL-1:0][DATA_WIDTH-1:0]             weight_in,
    input  logic                                                  first_k_in,
    output logic signed [SYS_ROW-1:0][SYS_COL-1:0][ACC_WIDTH-1:0] accumulator_out
);

    // cycles from the last k slice at the inputs to the last accumulator update
    localparam int PIPELINE_LATENCY = SYS_ROW + SYS_COL - 2 + vit_config_pkg::DSP_LATENCY;

    logic [SYS_ROW-1:0][DATA_WIDTH-1:0] activation_skewed;
    logic [SYS_COL-1:0][DATA_WIDTH-1:0] weight_skewed;

    logic [SYS_ROW-1:0][SYS_COL:0][DATA_WIDTH-1:0] activation_wire;
    logic [SYS_ROW:0][SYS_COL-1:0][DATA_WIDTH-1:0] weight_wire;
    logic [SYS_ROW-1:0][SYS_COL:0]                 first_k_wire;

    skew #(
        .LANE_COUNT (SYS_ROW),
        .DATA_WIDTH (DATA_WIDTH)
    ) activation_skew (
        .CLK      (CLK),
        .data_in  (activation_in),
        .data_out (activation_skewed)
    );

    skew #(
        .LANE_COUNT (SYS_COL),
        .DATA_WIDTH (DATA_WIDTH)
    ) weight_skew (
        .CLK      (CLK),
        .data_in  (weight_in),
        .data_out (weight_skewed)
    );

    genvar row, col;
    generate
        for (row = 0; row < SYS_ROW; row++) begin: row_input_wire
            assign activation_wire[row][0] = activation_skewed[row];
            if (row == 0) begin: first_k_from_port
                assign first_k_wire[row][0] = first_k_in;
            end
            else begin: first_k_from_above
                assign first_k_wire[row][0] = first_k_wire[row-1][1];
            end
        end
        for (col = 0; col < SYS_COL; col++) begin: col_input_wire
            assign weight_wire[0][col] = weight_skewed[col];
        end
        for (row = 0; row < SYS_ROW; row++) begin: row_sys_array
            for (col = 0; col < SYS_COL; col++) begin: col_sys_array
                pe #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .ACC_WIDTH  (ACC_WIDTH)
                ) pe_inst (
                    .CLK            (CLK),
                    .activation_in  (activation_wire[row][col]),
                    .weight_in      (weight_wire[row][col]),
                    .first_k_in     (first_k_wire[row][col]),
                    .activation_out (activation_wire[row][col+1]),
                    .weight_out     (weight_wire[row+1][col]),
                    .first_k_out    (first_k_wire[row][col+1]),
                    .accumulator    (accumulator_out[row][col])
                );
            end
        end
    endgenerate

endmodule
