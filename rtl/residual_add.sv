`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: residual_add
// Description: Adds two INT8 streams that have different scales and
// requantizes the sum to INT8, LANES values per cycle.
//
//   sum_out = clip((a_in * multiplier_a + b_in * multiplier_b + round) >>> shift_value)
//   round   = 2**(shift_value - 1), or 0 when shift_value is 0
//
// a_in, b_in and sum_out are signed DATA_WIDTH (INT8). The multipliers are
// signed MULTIPLIER_WIDTH (8 bits, so a positive one is 1..127) and share one
// shift, which is arithmetic; round makes it round to nearest, ties up. The
// products are DATA_WIDTH + MULTIPLIER_WIDTH bits and the sum is kept wide
// enough for the largest round, so nothing wraps. clip is to -128..127.
// Bit-exact reference: residual_add() in golden/nonlinear_ops.py.
//
// valid_out and sum_out follow valid_in and the operands by LATENCY cycles:
// operand register, product register, sum register, shifted register, output
// register. multiplier_a, multiplier_b and shift_value must be stable while
// data is in the pipeline. Only valid is reset.
//////////////////////////////////////////////////////////////////////////////////

module residual_add #(
    parameter  int LANES            = vit_config_pkg::SYS_ROW,
    parameter  int DATA_WIDTH       = vit_config_pkg::DATA_WIDTH,
    parameter  int MULTIPLIER_WIDTH = vit_config_pkg::SCALE_WIDTH,
    parameter  int SHIFT_WIDTH      = vit_config_pkg::SHIFT_WIDTH,
    localparam int LATENCY          = vit_config_pkg::RESIDUAL_ADD_LATENCY
) (
    input  logic                                    CLK,
    input  logic                                    reset,
    input  logic                                    valid_in,
    input  logic signed [LANES-1:0][DATA_WIDTH-1:0] a_in,
    input  logic signed [LANES-1:0][DATA_WIDTH-1:0] b_in,
    input  logic signed [MULTIPLIER_WIDTH-1:0]      multiplier_a,
    input  logic signed [MULTIPLIER_WIDTH-1:0]      multiplier_b,
    input  logic [SHIFT_WIDTH-1:0]                  shift_value,
    output logic                                    valid_out,
    output logic signed [LANES-1:0][DATA_WIDTH-1:0] sum_out
);

    localparam int PRODUCT_WIDTH = DATA_WIDTH + MULTIPLIER_WIDTH;
    localparam int ROUND_WIDTH   = (1 << SHIFT_WIDTH) - 1;   // 2**(largest shift - 1) fits
    // two products and the round, with a sign bit
    localparam int SUM_WIDTH     = ((PRODUCT_WIDTH + 1 > ROUND_WIDTH) ? PRODUCT_WIDTH + 1 : ROUND_WIDTH) + 2;

    localparam logic signed [SUM_WIDTH-1:0] DATA_MAX = (SUM_WIDTH'(1) << (DATA_WIDTH - 1)) - 1;
    localparam logic signed [SUM_WIDTH-1:0] DATA_MIN = -(SUM_WIDTH'(1) << (DATA_WIDTH - 1));

    logic [LATENCY-1:0]           valid_pipeline;
    logic signed [SUM_WIDTH-1:0]  round_value;

    always_ff @(posedge CLK) begin
        if (reset) begin
            valid_pipeline <= '0;
        end
        else begin
            valid_pipeline <= LATENCY'({valid_pipeline, valid_in});
        end
        round_value <= (shift_value == '0) ? '0 : SUM_WIDTH'(1) << (shift_value - 1'b1);
    end

    assign valid_out = valid_pipeline[LATENCY-1];

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane++) begin: lane_add
            logic signed [DATA_WIDTH-1:0]                     a_operand;
            logic signed [DATA_WIDTH-1:0]                     b_operand;
            logic signed [MULTIPLIER_WIDTH-1:0]               a_multiplier;
            logic signed [MULTIPLIER_WIDTH-1:0]               b_multiplier;
            (* use_dsp = "yes" *) logic signed [PRODUCT_WIDTH-1:0] a_product;
            (* use_dsp = "yes" *) logic signed [PRODUCT_WIDTH-1:0] b_product;
            logic signed [SUM_WIDTH-1:0]                      sum;
            logic signed [SUM_WIDTH-1:0]                      shifted;

            always_ff @(posedge CLK) begin
                a_operand    <= a_in[lane];
                b_operand    <= b_in[lane];
                a_multiplier <= multiplier_a;
                b_multiplier <= multiplier_b;
                a_product    <= a_operand * a_multiplier;
                b_product    <= b_operand * b_multiplier;
                sum          <= SUM_WIDTH'(a_product) + SUM_WIDTH'(b_product) + round_value;
                shifted      <= sum >>> shift_value;
                if (shifted > DATA_MAX) begin
                    sum_out[lane] <= DATA_MAX[DATA_WIDTH-1:0];
                end
                else if (shifted < DATA_MIN) begin
                    sum_out[lane] <= DATA_MIN[DATA_WIDTH-1:0];
                end
                else begin
                    sum_out[lane] <= shifted[DATA_WIDTH-1:0];
                end
            end
        end
    endgenerate

endmodule
