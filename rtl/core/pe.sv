`timescale 1ns / 1ps

(* use_dsp = "yes" *)
module pe #(
    parameter int DATA_WIDTH = vit_config_pkg::DATA_WIDTH,
    parameter int ACC_WIDTH  = vit_config_pkg::ACC_WIDTH
) (
    input  logic                         CLK,
    input  logic signed [DATA_WIDTH-1:0] activation_in,
    input  logic signed [DATA_WIDTH-1:0] weight_in,
    input  logic                         first_k_in,
    output logic signed [DATA_WIDTH-1:0] activation_out,
    output logic signed [DATA_WIDTH-1:0] weight_out,
    output logic                         first_k_out,
    output logic signed [ACC_WIDTH-1:0]  accumulator
);

    logic signed [DATA_WIDTH-1:0]   activation_operand;  // DSP A register
    logic signed [DATA_WIDTH-1:0]   weight_operand;      // DSP B register
    logic signed [2*DATA_WIDTH-1:0] product;             // DSP M register
    logic                           first_k_operand;
    logic                           first_k_product;
    logic signed [ACC_WIDTH-1:0]    accumulator_feedback;

    // first_k restarts the sum: the product is added to zero
    assign accumulator_feedback = first_k_product ? '0 : accumulator;

    always_ff @(posedge CLK) begin
        activation_out <= activation_in;
        weight_out     <= weight_in;
        first_k_out    <= first_k_in;

        activation_operand <= activation_in;
        weight_operand     <= weight_in;
        first_k_operand    <= first_k_in;

        product         <= activation_operand * weight_operand;
        first_k_product <= first_k_operand;

        accumulator <= accumulator_feedback + ACC_WIDTH'(product);  // sign extension
    end

endmodule
