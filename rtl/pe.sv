`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: pe
// Description: Output-stationary processing element.
//
// The accumulator stays in the PE. The activation passes left to right, the
// weight passes top to bottom and the first_k flag travels with them, each
// through one register.
//
// Fixed point: activation and weight are signed DATA_WIDTH integers, the
// accumulator is a signed ACC_WIDTH integer that wraps modulo 2**ACC_WIDTH.
// The product is exact in 2*DATA_WIDTH bits, so ACC_WIDTH >= 2*DATA_WIDTH.
//
// No register has a reset. first_k_in loads the accumulator with the first
// product of a new output tile, which replaces a clear, and zero operands
// leave it unchanged.
//////////////////////////////////////////////////////////////////////////////////

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

    logic signed [2*DATA_WIDTH-1:0] product;
    logic signed [ACC_WIDTH-1:0]    product_extended;

    assign product          = activation_in * weight_in;
    assign product_extended = ACC_WIDTH'(product);  // sign extension

    always_ff @(posedge CLK) begin
        activation_out <= activation_in;
        weight_out     <= weight_in;
        first_k_out    <= first_k_in;
        if (first_k_in) begin
            accumulator <= product_extended;
        end
        else begin
            accumulator <= accumulator + product_extended;
        end
    end

endmodule
