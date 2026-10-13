`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: gelu_lut
// Description: GELU by table lookup, LANES values per cycle.
//
//   y_out = table[table_select][x_in]
//
// x_in and y_out are signed DATA_WIDTH (INT8). A table has one entry per input
// value, addressed by the input's two's complement bits:
//   entry[x] = clip(round(gelu(x * input scale) / output scale))
// so the table carries both scales and the result is exact for an INT8 input.
// There are TABLE_COUNT tables, one per GELU of the model (encoder layer 0,
// encoder layer 1), back to back in one ROM: address = table_select * 256 + x.
// Bit-exact reference: gelu() and gelu_table() in golden/nonlinear_ops.py.
//
// The ROM is loaded from ROM_FILE with $readmemh, one entry per line; the
// file is written by golden/gen_deit_forward.py. Lanes are served in pairs
// by the two read ports of one copy of the ROM.
//
// valid_out and y_out follow valid_in and x_in by LATENCY cycles: address
// register, ROM read, output register. table_select must be stable while data
// is in the pipeline. Only valid is reset.
//////////////////////////////////////////////////////////////////////////////////

module gelu_lut #(
    parameter  int    LANES       = vit_config_pkg::SYS_ROW,
    parameter  int    DATA_WIDTH  = vit_config_pkg::DATA_WIDTH,
    parameter  int    TABLE_COUNT = vit_config_pkg::GELU_TABLE_COUNT,
    parameter  string ROM_FILE    = "rom/gelu_rom.hex",
    localparam int    LATENCY      = vit_config_pkg::GELU_LATENCY,
    localparam int    SELECT_WIDTH = (TABLE_COUNT > 1) ? $clog2(TABLE_COUNT) : 1
) (
    input  logic                                    CLK,
    input  logic                                    reset,
    input  logic                                    valid_in,
    input  logic signed [LANES-1:0][DATA_WIDTH-1:0] x_in,
    input  logic [SELECT_WIDTH-1:0]                 table_select,
    output logic                                    valid_out,
    output logic signed [LANES-1:0][DATA_WIDTH-1:0] y_out
);

    localparam int ROM_DEPTH     = TABLE_COUNT << DATA_WIDTH;
    localparam int ROM_ADDRESS   = $clog2(ROM_DEPTH);
    localparam int PAIRS         = (LANES + 1) / 2;

    logic [LATENCY-1:0] valid_pipeline;

    always_ff @(posedge CLK) begin
        if (reset) begin
            valid_pipeline <= '0;
        end
        else begin
            valid_pipeline <= LATENCY'({valid_pipeline, valid_in});
        end
    end

    assign valid_out = valid_pipeline[LATENCY-1];

    genvar pair;
    generate
        for (pair = 0; pair < PAIRS; pair++) begin: lane_pair
            (* rom_style = "block" *) logic [DATA_WIDTH-1:0] rom[ROM_DEPTH];

            initial begin
                $readmemh(ROM_FILE, rom);
            end

            logic [1:0][ROM_ADDRESS-1:0] address;
            logic [1:0][DATA_WIDTH-1:0]  read_data;
            logic [1:0][DATA_WIDTH-1:0]  result;

            always_ff @(posedge CLK) begin
                for (int port = 0; port < 2; port++) begin
                    if (2*pair + port < LANES) begin
                        address[port] <= ROM_ADDRESS'({table_select, x_in[2*pair + port]});
                    end
                    else begin
                        address[port] <= '0;
                    end
                end
                read_data[0] <= rom[address[0]];
                read_data[1] <= rom[address[1]];
                result       <= read_data;
            end

            assign y_out[2*pair] = result[0];
            if (2*pair + 1 < LANES) begin: second_lane
                assign y_out[2*pair + 1] = result[1];
            end
        end
    endgenerate

endmodule
