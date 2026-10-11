`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/09/2026 05:49:23 AM
// Design Name: 
// Module Name: vit_config_pkg
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


package vit_config_pkg;

    parameter  DATA_WIDTH  = 8;
    parameter  ACC_WIDTH   = 32;
    parameter  SHIFT_WIDTH = $clog2(ACC_WIDTH);
    parameter  SCALE_WIDTH = 8;

    parameter  SYS_ROW = 8;
    parameter  SYS_COL = 8;

    // largest matmul the stream controller is sized for: (M x K) @ (K x N)
    parameter  ROWS_TOTAL_MAX = 197;
    parameter  INNER_DIM_MAX  = 768;
    parameter  COLS_TOTAL_MAX = 768;

    // words per operand buffer bank
    parameter  BANK_DEPTH = 4096;

endpackage
