`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/09/2026 05:51:36 AM
// Design Name: 
// Module Name: systolic_array
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

module systolic_array (
	input logic CLK, reset, load_weight,
	input logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0] x_tensor_in,
	input logic signed [SYS_ROW-1:0][SYS_COL-1:0][DATA_WIDTH-1:0] w_tensor_in,
	output logic signed [SYS_COL-1:0][ACC_WIDTH-1:0] y_tensor_out
);

	logic signed [SYS_ROW-1:0][SYS_COL:0][DATA_WIDTH-1:0] x_in_wire;
	
	logic signed [SYS_ROW:0][SYS_COL-1:0][ACC_WIDTH-1:0] y_in_wire;

	genvar i,j;
	generate
		for (i = 0; i < SYS_ROW; i++) begin: x_input_wire
			assign x_in_wire[i][0] = x_tensor_in[i];
		end
		for (j = 0; j < SYS_COL; j++) begin: y_input_wire
			assign y_in_wire[0][j] = 0; //no sum input
		end
		for (i = 0; i < SYS_ROW; i++) begin: row_sys_array
			for (j = 0; j < SYS_COL; j++) begin: col_sys_array
				pe pe_inst(
					.CLK			(CLK), 
					.reset			(reset), 
					.load_weight	(load_weight),
					.x_in			(x_in_wire[i][j]),
					.y_in			(y_in_wire[i][j]),
					.w_in			(w_tensor_in[i][j]), 
					.x_out			(x_in_wire[i][j+1]), 
					.y_out			(y_in_wire[i+1][j])
				);
			end
		end
		for (j = 0; j < SYS_COL; j++) begin: y_output_wire
			assign y_tensor_out[j] = y_in_wire[SYS_ROW][j];
		end
	endgenerate 
	
endmodule 