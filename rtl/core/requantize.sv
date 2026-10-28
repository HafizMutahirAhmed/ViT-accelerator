`timescale 1ns / 1ps

(* use_dsp = "yes" *)
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

    localparam int LATENCY = vit_config_pkg::REQUANTIZE_LATENCY;

    localparam signed [(ACC_WIDTH + SCALE_WIDTH)-1:0] MAX_VAL =  (1 << (DATA_WIDTH-1)) - 1;  
    localparam signed [(ACC_WIDTH + SCALE_WIDTH)-1:0] MIN_VAL = -(1 << (DATA_WIDTH-1));      

    logic signed [ACC_WIDTH-1:0]                 unquantized_operand;
    logic signed [SCALE_WIDTH-1:0]               scale_operand;
    logic signed [(ACC_WIDTH + SCALE_WIDTH)-1:0] product;
    logic signed [(ACC_WIDTH + SCALE_WIDTH)-1:0] mult_reg;  
    logic signed [(ACC_WIDTH + SCALE_WIDTH)-1:0] shifted_val;
    logic [SHIFT_WIDTH-1:0]                      shift_value_reg;
    logic [LATENCY-1:0]                          valid_pipeline;

    always_ff @(posedge CLK) begin
        unquantized_operand <= unquantized_result;
        scale_operand       <= scale;
        product             <= unquantized_operand * scale_operand;
        mult_reg            <= product;
        shift_value_reg     <= shift_value; 
        shifted_val         <= mult_reg >>> shift_value_reg;
        valid_pipeline      <= {valid_pipeline[LATENCY-2:0], valid_in};
    end

    assign valid_out = valid_pipeline[LATENCY-1];
    
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
