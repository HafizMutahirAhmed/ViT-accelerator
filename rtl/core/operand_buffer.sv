`timescale 1ns / 1ps

module operand_buffer #(
    parameter  int SYS_LANES     = vit_config_pkg::SYS_ROW,
    parameter  int BANK_DEPTH    = vit_config_pkg::BANK_DEPTH,
    parameter  int DATA_WIDTH    = vit_config_pkg::DATA_WIDTH,
    parameter  int READ_LATENCY  = vit_config_pkg::BUFFER_READ_LATENCY,
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
            logic [READ_LATENCY-1:0][DATA_WIDTH-1:0]         read_pipeline;

            always_ff @(posedge CLK) begin
                if (write_enable[lane]) begin
                    memory[write_address[lane]] <= write_data[lane];
                end
                read_pipeline[0] <= memory[read_address];
                for (int stage = 1; stage < READ_LATENCY; stage++) begin
                    read_pipeline[stage] <= read_pipeline[stage-1];
                end
            end

            assign read_data[lane] = read_pipeline[READ_LATENCY-1];
        end
    endgenerate

endmodule
