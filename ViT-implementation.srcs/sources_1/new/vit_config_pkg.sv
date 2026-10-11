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

endpackage