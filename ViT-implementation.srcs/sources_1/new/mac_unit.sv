`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/28/2026 11:00:17 AM
// Design Name: 
// Module Name: mac_unit
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


module mac_unit #(
  parameter int DATA_WIDTH = 8  // Q8 data width
) (
  input  logic        clk,
  input  logic        rst,

  // Control
  input  logic        clr,      // Clear accumulator to 0
  input  logic        en,       // Enable MAC for this cycle

  // Operands
  input  logic signed [DATA_WIDTH-1:0] a,
  input  logic signed [DATA_WIDTH-1:0] b,

  // Output (latched when ready)
  output logic signed [DATA_WIDTH-1:0] result,
  output logic        result_vld
);
    logic [15:0] mul_result, mul_result_pipeline;
    assign mul_result = a * b;
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            mul_result_pipeline = mul_result;
        end
        
    end
 

endmodule
