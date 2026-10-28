`timescale 1ns / 1ps

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
