`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: pe
// Description: Output-stationary processing element, one DSP48E1.
//
// The accumulator stays in the PE. The activation passes left to right, the
// weight passes top to bottom and the first_k flag travels with them, each
// through one register.
//
// Fixed point: activation and weight are signed DATA_WIDTH integers, the
// accumulator is a signed ACC_WIDTH integer that wraps modulo 2**ACC_WIDTH.
// The product is exact in 2*DATA_WIDTH bits, so ACC_WIDTH >= 2*DATA_WIDTH.
//
// The multiply-accumulate is pipelined to fit the DSP48E1 registers: operand
// registers (A, B), product register (M) and the accumulator (P). The operands
// seen in one cycle reach the accumulator vit_config_pkg::DSP_LATENCY + 1 = 3
// clock edges later. first_k is delayed alongside and selects, in the DSP's Z
// multiplexer, whether the product is added to the accumulator or to zero.
//
// No register has a reset. first_k_in loads the accumulator with the first
// product of a new output tile, which replaces a clear, and zero operands
// leave it unchanged.
//////////////////////////////////////////////////////////////////////////////////

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
