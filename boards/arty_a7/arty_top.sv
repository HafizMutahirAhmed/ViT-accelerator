`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: arty_top
// Description: Digilent Arty A7-100T top level of the accelerator.
//
// Everything specific to this board is here and in the files next to it:
// the clock, the reset, the UART that carries the host protocol's byte
// stream, and the LEDs. The accelerator itself (vit_accelerator: host_link
// and matmul_engine) is the board-independent one.
//
// Clock: the 100 MHz oscillator drives the design directly, no MMCM.
// Reset: synchronous. It is asserted for the first 15 clocks after
// configuration (a counter that powers up at zero) and while the RESET
// button, ck_rst, is pressed (low), after two synchronising flip-flops.
// UART: 8N1 at BAUD_RATE, one bit = round(CLOCK_HZ / BAUD_RATE) clocks.
//   uart_txd_in   data from the USB-UART bridge to the FPGA
//   uart_rxd_out  data from the FPGA to the bridge
// LEDs: led[0] engine busy, led[1] done seen, led[2] protocol error (sticky,
// cleared by the CLEAR command), led[3] a received byte was lost (sticky,
// cleared by reset).
//
// Model configuration: the 112 x 112 MNIST one. The pool depths are the plan
// python/layer_program.py prints for 50 tokens and a 16,384-word weight
// buffer (program_plan.txt), buffer 0 in the low word; the ROM images are
// rtl/rom/mnist112.
//////////////////////////////////////////////////////////////////////////////////

module arty_top #(
    parameter int CLOCK_HZ  = 100_000_000,
    parameter int BAUD_RATE = 921_600
) (
    input  logic       CLK100MHZ,
    input  logic       ck_rst,
    input  logic       uart_txd_in,
    output logic       uart_rxd_out,
    output logic [3:0] led
);

    localparam int CLOCKS_PER_BIT = (CLOCK_HZ + BAUD_RATE / 2) / BAUD_RATE;

    logic CLK;

    assign CLK = CLK100MHZ;

    // ------------------------------------------------------------------
    // reset
    // ------------------------------------------------------------------
    (* ASYNC_REG = "TRUE" *) logic [1:0] button_sync;
    logic [3:0]                          power_on_count;
    logic                                reset;

    always_ff @(posedge CLK) begin
        button_sync <= {button_sync[0], ck_rst};
        if (!button_sync[1]) begin
            power_on_count <= '0;
        end
        else if (power_on_count != '1) begin
            power_on_count <= power_on_count + 1'b1;
        end
        reset <= power_on_count != '1;
    end

    // ------------------------------------------------------------------
    // UART
    // ------------------------------------------------------------------
    logic [7:0] rx_data;
    logic       rx_valid;
    logic       rx_ready;
    logic       rx_overrun;
    logic [7:0] tx_data;
    logic       tx_valid;
    logic       tx_ready;

    uart_rx #(
        .CLOCKS_PER_BIT (CLOCKS_PER_BIT)
    ) receiver (
        .CLK        (CLK),
        .reset      (reset),
        .rxd        (uart_txd_in),
        .data       (rx_data),
        .data_valid (rx_valid),
        .data_ready (rx_ready),
        .overrun    (rx_overrun)
    );

    uart_tx #(
        .CLOCKS_PER_BIT (CLOCKS_PER_BIT)
    ) transmitter (
        .CLK        (CLK),
        .reset      (reset),
        .data       (tx_data),
        .data_valid (tx_valid),
        .data_ready (tx_ready),
        .txd        (uart_rxd_out)
    );

    // ------------------------------------------------------------------
    // the accelerator
    // ------------------------------------------------------------------
    logic busy;
    logic done_flag;
    logic error_flag;
    logic overrun_flag;

    vit_accelerator #(
        .SYS_ROW        (8),
        .SYS_COL        (8),
        .BUFFER_DEPTH   ({32'd16, 32'd1344, 32'd2544, 32'd1050, 32'd1344, 32'd5376, 32'd16384, 32'd1344}),
        .GELU_ROM_FILE  ("../rom/mnist112/gelu_rom.hex"),
        .EXP_ROM_FILE   ("../rom/mnist112/exp_rom.hex"),
        .RSQRT_ROM_FILE ("../rom/mnist112/rsqrt_rom.hex")
    ) accelerator (
        .CLK        (CLK),
        .reset      (reset),
        .rx_data    (rx_data),
        .rx_valid   (rx_valid),
        .rx_ready   (rx_ready),
        .tx_data    (tx_data),
        .tx_valid   (tx_valid),
        .tx_ready   (tx_ready),
        .busy       (busy),
        .done_flag  (done_flag),
        .error_flag (error_flag)
    );

    always_ff @(posedge CLK) begin
        if (reset) begin
            overrun_flag <= 1'b0;
        end
        else if (rx_overrun) begin
            overrun_flag <= 1'b1;
        end
        led <= {overrun_flag, error_flag, done_flag, busy};
    end

endmodule
