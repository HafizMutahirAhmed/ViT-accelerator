`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/09/2026 05:45:37 AM
// Design Name: 
// Module Name: pe
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
import vit_config_pkg::*; 

module pe (	
	input logic CLK, reset, load_weight,
	input logic signed [DATA_WIDTH-1:0] x_in, 
	input logic signed [ACC_WIDTH-1:0] y_in,
	input logic signed [DATA_WIDTH-1:0] w_in,
	output logic signed [DATA_WIDTH-1:0] x_out,
	output logic signed [ACC_WIDTH-1:0] y_out 
);
	logic signed [DATA_WIDTH-1:0] w;
		
	always_ff @(posedge CLK or posedge reset) begin
		if (reset) begin
			w <= 0;
			x_out <= 0;
			y_out <= 0;
		end
		else begin 
			if (load_weight) begin
				w <= w_in;
			end
			y_out <= w * x_in + y_in;
			x_out <= x_in;
		end
	end
endmodule
		
