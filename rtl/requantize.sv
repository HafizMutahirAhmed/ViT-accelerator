`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/09/2026 05:53:44 AM
// Design Name: 
// Module Name: requantize
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
// Fixed point: out = clip((unquantized_result * scale) >>> shift_value).
// unquantized_result is a signed ACC_WIDTH integer and scale a signed
// SCALE_WIDTH integer; the product keeps all ACC_WIDTH + SCALE_WIDTH bits. The
// shift is arithmetic (floor, no rounding offset) and the result saturates to
// a signed DATA_WIDTH integer. The multiply is registered, so the result and
// valid_out come one cycle after the inputs and valid_in.

module requantize #(
    parameter int DATA_WIDTH  = vit_config_pkg::DATA_WIDTH,
    parameter int ACC_WIDTH   = vit_config_pkg::ACC_WIDTH,
    parameter int SCALE_WIDTH = vit_config_pkg::SCALE_WIDTH,
    parameter int SHIFT_WIDTH = vit_config_pkg::SHIFT_WIDTH
) (
    input  logic                                 CLK,
    input  logic                                 valid_in,
    input  logic signed [ACC_WIDTH-1:0]          unquantized_result,
    input  logic signed [SCALE_WIDTH-1:0]        scale,
    input  logic [SHIFT_WIDTH-1:0]               shift_value,
    output logic                                 valid_out,
    output logic signed [DATA_WIDTH-1:0]         quantized_result
);

    localparam signed [(ACC_WIDTH + SCALE_WIDTH)-1:0] MAX_VAL =  (1 << (DATA_WIDTH-1)) - 1;  
    localparam signed [(ACC_WIDTH + SCALE_WIDTH)-1:0] MIN_VAL = -(1 << (DATA_WIDTH-1));      

    logic signed [(ACC_WIDTH + SCALE_WIDTH)-1:0] mult_reg;  
    logic [SHIFT_WIDTH-1:0]                      shift_value_reg;

    always_ff @(posedge CLK) begin
        mult_reg        <= unquantized_result * scale;
        shift_value_reg <= shift_value; 
        valid_out       <= valid_in;
    end
    
    logic signed [(ACC_WIDTH + SCALE_WIDTH)-1:0] shifted_val;
    assign shifted_val = mult_reg >>> shift_value_reg;
    
    always_comb begin
        if (shifted_val > MAX_VAL) begin
            quantized_result = DATA_WIDTH'(MAX_VAL);
        end
        else if (shifted_val < MIN_VAL) begin
            quantized_result = DATA_WIDTH'(MIN_VAL);
        end
        else begin
            quantized_result = DATA_WIDTH'(shifted_val);
        end
    end

endmodule
