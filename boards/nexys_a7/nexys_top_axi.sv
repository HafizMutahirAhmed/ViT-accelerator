`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: nexys_top_axi
// Description: Digilent Nexys A7-100T top level of the accelerator, through
// the AXI wrapper. The byte stream of the UART goes to axi_host_bridge,
// which drives vit_axi_wrapper over AXI4-Lite and the AXI4-Stream pair.
// The host protocol is docs/HOST_PROTOCOL.md. Pins: boards/nexys_a7/nexys_a7.xdc.
//
// Clock: the 100 MHz oscillator drives the design directly, no MMCM.
// Reset: synchronous. It is asserted for the first 15 clocks after
// configuration (a counter that powers up at zero) and while the CPU_RESET
// button, ck_rst, is pressed (low), after two synchronising flip-flops.
// UART: 8N1 at BAUD_RATE, one bit = round(CLOCK_HZ / BAUD_RATE) clocks.
//   uart_txd_in   data from the USB-UART bridge to the FPGA
//   uart_rxd_out  data from the FPGA to the bridge
// LEDs: led[0] engine busy, led[1] done seen, led[2] error (a protocol error
// in the bridge or an error flagged by the wrapper), led[3] a received byte
// was lost (sticky, cleared by reset).
//
// Model configuration: the 112 x 112 MNIST one. The pool depths are the plan
// python/layer_program.py prints for 50 tokens and a 16,384-word weight
// buffer (program_plan.txt); the ROM images are rtl/rom/mnist112.
//////////////////////////////////////////////////////////////////////////////////

module nexys_top_axi #(
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
    // the accelerator: bridge and AXI wrapper
    // ------------------------------------------------------------------
    logic busy;
    logic done_flag;
    logic error_flag;
    logic wrapper_error;
    logic bridge_error;
    logic overrun_flag;
    logic aresetn;

    logic [11:0] axi_awaddr, axi_araddr;
    logic        axi_awvalid, axi_awready, axi_wvalid, axi_wready, axi_bvalid, axi_bready;
    logic        axi_arvalid, axi_arready, axi_rvalid, axi_rready;
    logic [31:0] axi_wdata, axi_rdata;
    logic [3:0]  axi_wstrb;
    logic [1:0]  axi_bresp, axi_rresp;
    logic [63:0] load_tdata, store_tdata;
    logic [7:0]  load_tkeep, store_tkeep;
    logic        load_tlast, load_tvalid, load_tready, store_tlast, store_tvalid, store_tready;

    assign aresetn    = !reset;
    assign error_flag = wrapper_error || bridge_error;

    axi_host_bridge bridge (
        .aclk          (CLK),
        .aresetn       (aresetn),
        .rx_data       (rx_data),
        .rx_valid      (rx_valid),
        .rx_ready      (rx_ready),
        .tx_data       (tx_data),
        .tx_valid      (tx_valid),
        .tx_ready      (tx_ready),
        .m_axi_awaddr  (axi_awaddr),
        .m_axi_awvalid (axi_awvalid),
        .m_axi_awready (axi_awready),
        .m_axi_wdata   (axi_wdata),
        .m_axi_wstrb   (axi_wstrb),
        .m_axi_wvalid  (axi_wvalid),
        .m_axi_wready  (axi_wready),
        .m_axi_bresp   (axi_bresp),
        .m_axi_bvalid  (axi_bvalid),
        .m_axi_bready  (axi_bready),
        .m_axi_araddr  (axi_araddr),
        .m_axi_arvalid (axi_arvalid),
        .m_axi_arready (axi_arready),
        .m_axi_rdata   (axi_rdata),
        .m_axi_rresp   (axi_rresp),
        .m_axi_rvalid  (axi_rvalid),
        .m_axi_rready  (axi_rready),
        .m_axis_tdata  (load_tdata),
        .m_axis_tkeep  (load_tkeep),
        .m_axis_tlast  (load_tlast),
        .m_axis_tvalid (load_tvalid),
        .m_axis_tready (load_tready),
        .s_axis_tdata  (store_tdata),
        .s_axis_tkeep  (store_tkeep),
        .s_axis_tlast  (store_tlast),
        .s_axis_tvalid (store_tvalid),
        .s_axis_tready (store_tready),
        .error_flag    (bridge_error)
    );

    vit_axi_wrapper #(
        .SYS_ROW        (8),
        .SYS_COL        (8),
        .BUFFER_DEPTH_0 (1344),
        .BUFFER_DEPTH_1 (16384),
        .BUFFER_DEPTH_2 (5376),
        .BUFFER_DEPTH_3 (1344),
        .BUFFER_DEPTH_4 (1050),
        .BUFFER_DEPTH_5 (2544),
        .BUFFER_DEPTH_6 (1344),
        .BUFFER_DEPTH_7 (16),
        .GELU_ROM_FILE  ("../rom/mnist112/gelu_rom.hex"),
        .EXP_ROM_FILE   ("../rom/mnist112/exp_rom.hex"),
        .RSQRT_ROM_FILE ("../rom/mnist112/rsqrt_rom.hex")
    ) accelerator (
        .aclk          (CLK),
        .aresetn       (aresetn),
        .s_axi_awaddr  (axi_awaddr),
        .s_axi_awvalid (axi_awvalid),
        .s_axi_awready (axi_awready),
        .s_axi_wdata   (axi_wdata),
        .s_axi_wstrb   (axi_wstrb),
        .s_axi_wvalid  (axi_wvalid),
        .s_axi_wready  (axi_wready),
        .s_axi_bresp   (axi_bresp),
        .s_axi_bvalid  (axi_bvalid),
        .s_axi_bready  (axi_bready),
        .s_axi_araddr  (axi_araddr),
        .s_axi_arvalid (axi_arvalid),
        .s_axi_arready (axi_arready),
        .s_axi_rdata   (axi_rdata),
        .s_axi_rresp   (axi_rresp),
        .s_axi_rvalid  (axi_rvalid),
        .s_axi_rready  (axi_rready),
        .s_axis_tdata  (load_tdata),
        .s_axis_tkeep  (load_tkeep),
        .s_axis_tlast  (load_tlast),
        .s_axis_tvalid (load_tvalid),
        .s_axis_tready (load_tready),
        .m_axis_tdata  (store_tdata),
        .m_axis_tkeep  (store_tkeep),
        .m_axis_tlast  (store_tlast),
        .m_axis_tvalid (store_tvalid),
        .m_axis_tready (store_tready),
        .irq           (),
        .busy          (busy),
        .done_flag     (done_flag),
        .error_flag    (wrapper_error)
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
