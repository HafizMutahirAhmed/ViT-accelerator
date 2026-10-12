`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: vit_accelerator
// Description: The board-independent accelerator: matmul_engine with host_link
// in front of its host port. A board wraps this with a clock, a reset and a
// transport that delivers and takes bytes (a UART, FrontPanel pipes).
//
// The byte stream carries the host protocol of docs/HOST_PROTOCOL.md.
// rx_data is taken when rx_valid and rx_ready are high, tx_data is sent when
// tx_valid and tx_ready are high. busy is the engine's busy; done_flag and
// error_flag are host_link's (sticky, see there), meant for status LEDs.
//
// BUFFER_DEPTH and the ROM images select the model configuration; they are
// passed to matmul_engine unchanged. reset is synchronous.
//////////////////////////////////////////////////////////////////////////////////

module vit_accelerator #(
    parameter  int SYS_ROW    = vit_config_pkg::SYS_ROW,
    parameter  int SYS_COL    = vit_config_pkg::SYS_COL,
    parameter  bit [vit_config_pkg::BUFFER_COUNT-1:0][31:0] BUFFER_DEPTH
                              = {vit_config_pkg::BUFFER_COUNT{32'(vit_config_pkg::BANK_DEPTH)}},
    parameter  string GELU_ROM_FILE  = "rom/gelu_rom.hex",
    parameter  string EXP_ROM_FILE   = "rom/exp_rom.hex",
    parameter  string RSQRT_ROM_FILE = "rom/rsqrt_rom.hex"
) (
    input  logic       CLK,
    input  logic       reset,
    input  logic [7:0] rx_data,
    input  logic       rx_valid,
    output logic       rx_ready,
    output logic [7:0] tx_data,
    output logic       tx_valid,
    input  logic       tx_ready,
    output logic       busy,
    output logic       done_flag,
    output logic       error_flag
);

    localparam int DATA_WIDTH             = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH              = vit_config_pkg::ACC_WIDTH;
    localparam int LANES                  = (SYS_ROW > SYS_COL) ? SYS_ROW : SYS_COL;
    localparam int BUFFER_ID_WIDTH        = vit_config_pkg::BUFFER_ID_WIDTH;
    localparam int ADDRESS_WIDTH          = vit_config_pkg::ADDRESS_WIDTH;
    localparam int DESCRIPTOR_WORDS       = vit_config_pkg::DESCRIPTOR_WORDS;
    localparam int DESCRIPTOR_INDEX_WIDTH = $clog2(vit_config_pkg::DESCRIPTOR_COUNT);
    localparam int GAIN_WIDTH             = vit_config_pkg::LAYERNORM_GAIN_WIDTH;
    localparam int OFFSET_WIDTH           = vit_config_pkg::LAYERNORM_OFFSET_WIDTH;

    logic                              start;
    logic [DESCRIPTOR_INDEX_WIDTH-1:0] start_index;
    logic                              done;
    logic [DESCRIPTOR_INDEX_WIDTH-1:0] host_descriptor_index;
    logic [DESCRIPTOR_WORDS-1:0]       host_descriptor_write_enable;
    logic [31:0]                       host_descriptor_write_data;
    logic [BUFFER_ID_WIDTH-1:0]        host_buffer_id;
    logic [ADDRESS_WIDTH-1:0]          host_address;
    logic [LANES-1:0]                  host_buffer_write_enable;
    logic [LANES-1:0][DATA_WIDTH-1:0]  host_buffer_write_data;
    logic [LANES-1:0][DATA_WIDTH-1:0]  host_buffer_read_data;
    logic [LANES-1:0]                  host_bias_write_enable;
    logic [LANES-1:0][ACC_WIDTH-1:0]   host_bias_write_data;
    logic                              host_layernorm_write_enable;
    logic signed [GAIN_WIDTH-1:0]      host_layernorm_write_gain;
    logic signed [OFFSET_WIDTH-1:0]    host_layernorm_write_offset;

    host_link #(
        .LANES (LANES)
    ) link (
        .CLK                          (CLK),
        .reset                        (reset),
        .rx_data                      (rx_data),
        .rx_valid                     (rx_valid),
        .rx_ready                     (rx_ready),
        .tx_data                      (tx_data),
        .tx_valid                     (tx_valid),
        .tx_ready                     (tx_ready),
        .start                        (start),
        .start_index                  (start_index),
        .busy                         (busy),
        .done                         (done),
        .host_descriptor_index        (host_descriptor_index),
        .host_descriptor_write_enable (host_descriptor_write_enable),
        .host_descriptor_write_data   (host_descriptor_write_data),
        .host_buffer_id               (host_buffer_id),
        .host_address                 (host_address),
        .host_buffer_write_enable     (host_buffer_write_enable),
        .host_buffer_write_data       (host_buffer_write_data),
        .host_buffer_read_data        (host_buffer_read_data),
        .host_bias_write_enable       (host_bias_write_enable),
        .host_bias_write_data         (host_bias_write_data),
        .host_layernorm_write_enable  (host_layernorm_write_enable),
        .host_layernorm_write_gain    (host_layernorm_write_gain),
        .host_layernorm_write_offset  (host_layernorm_write_offset),
        .done_flag                    (done_flag),
        .error_flag                   (error_flag)
    );

    matmul_engine #(
        .SYS_ROW        (SYS_ROW),
        .SYS_COL        (SYS_COL),
        .BUFFER_DEPTH   (BUFFER_DEPTH),
        .GELU_ROM_FILE  (GELU_ROM_FILE),
        .EXP_ROM_FILE   (EXP_ROM_FILE),
        .RSQRT_ROM_FILE (RSQRT_ROM_FILE)
    ) engine (
        .CLK                          (CLK),
        .reset                        (reset),
        .start                        (start),
        .start_index                  (start_index),
        .busy                         (busy),
        .done                         (done),
        .host_descriptor_index        (host_descriptor_index),
        .host_descriptor_write_enable (host_descriptor_write_enable),
        .host_descriptor_write_data   (host_descriptor_write_data),
        .host_buffer_id               (host_buffer_id),
        .host_address                 (host_address),
        .host_buffer_write_enable     (host_buffer_write_enable),
        .host_buffer_write_data       (host_buffer_write_data),
        .host_buffer_read_data        (host_buffer_read_data),
        .host_bias_write_enable       (host_bias_write_enable[SYS_COL-1:0]),
        .host_bias_write_data         (host_bias_write_data[SYS_COL-1:0]),
        .host_bias_read_data          (),
        .host_layernorm_write_enable  (host_layernorm_write_enable),
        .host_layernorm_write_gain    (host_layernorm_write_gain),
        .host_layernorm_write_offset  (host_layernorm_write_offset)
    );

endmodule
