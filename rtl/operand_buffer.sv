`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: operand_buffer
// Description: Banked operand store that feeds one side of the systolic array.
//
// There are SYS_LANES banks, one per array lane. Bank `lane` holds operand
// rows lane, lane + SYS_LANES, lane + 2*SYS_LANES, ... and each row is stored
// contiguous along K, so row (tile_index*SYS_LANES + lane), element k is at
// address tile_index*K + k of bank `lane`. For the weight buffer a "row" is a
// column of B.
//
// Each bank has its own write port. All banks share one read address and the
// read data is registered: read_data is valid one cycle after read_address.
// A read of an address written in the same cycle returns the old word.
// Data is DATA_WIDTH bits, stored unchanged. There is no reset.
//////////////////////////////////////////////////////////////////////////////////

module operand_buffer #(
    parameter  int SYS_LANES     = vit_config_pkg::SYS_ROW,
    parameter  int BANK_DEPTH    = vit_config_pkg::BANK_DEPTH,
    parameter  int DATA_WIDTH    = vit_config_pkg::DATA_WIDTH,
    localparam int ADDRESS_WIDTH = $clog2(BANK_DEPTH)
) (
    input  logic                                    CLK,
    input  logic [SYS_LANES-1:0]                    write_enable,
    input  logic [SYS_LANES-1:0][ADDRESS_WIDTH-1:0] write_address,
    input  logic [SYS_LANES-1:0][DATA_WIDTH-1:0]    write_data,
    input  logic [ADDRESS_WIDTH-1:0]                read_address,
    output logic [SYS_LANES-1:0][DATA_WIDTH-1:0]    read_data
);

    genvar lane;
    generate
        for (lane = 0; lane < SYS_LANES; lane++) begin: bank
            (* ram_style = "block" *) logic [DATA_WIDTH-1:0] memory[BANK_DEPTH];

            always_ff @(posedge CLK) begin
                if (write_enable[lane]) begin
                    memory[write_address[lane]] <= write_data[lane];
                end
                read_data[lane] <= memory[read_address];
            end
        end
    endgenerate

endmodule
