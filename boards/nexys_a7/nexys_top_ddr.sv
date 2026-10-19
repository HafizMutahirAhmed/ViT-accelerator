`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: nexys_top_ddr
// Description: Digilent Nexys A7-100T top level of the accelerator with the
// board's DDR2 as external memory. The UART's byte stream goes to
// accelerator_system (bridge, AXI wrapper, engine); the block design
// memory_system (scripts/create_memory_system_bd.tcl: MIG, AXI DMA,
// SmartConnect) holds the weights in DDR2 and streams them into the engine
// on DMA_LOAD. The host protocol is docs/HOST_PROTOCOL.md; the host script is
// python/run_image.py --ddr. Pins: boards/nexys_a7/nexys_a7_ddr.xdc, the DDR2
// pins are in the MIG's configuration (boards/nexys_a7/mig_ddr2.prj).
//
// Clocks: system_clock_generator (Clocking Wizard) makes 100 MHz (CLK: the
// engine, bridge, UART, DMA, the interconnect's host side and the MIG's
// system clock) and 200 MHz (the MIG's reference clock) from the 100 MHz
// oscillator. The MIG runs the DDR2 at 325 MHz and its AXI port on ui_clk
// (81.25 MHz); the interconnect crosses between CLK and ui_clk.
// Resets, all synchronous:
//   reset           CLK domain: held while the clocking wizard is not locked,
//                   while CPU_RESET (ck_rst, low) is pressed and for
//                   RESET_CYCLES clocks after; the AXI DMA needs at least 16
//   memory_resetn   the MIG's sys_rst: low until the clocks are locked and
//                   while CPU_RESET is pressed
//   memory_aresetn  the MIG's AXI reset: ui_clk_sync_rst, inverted and
//                   registered in the ui_clk domain
// memory_ready is the MIG's init_calib_complete through two flip-flops in the
// CLK domain; the bridge refuses the memory commands until it is high.
// UART: 8N1 at BAUD_RATE, one bit = round(CLOCK_HZ / BAUD_RATE) clocks.
// LEDs: led[0] engine busy, led[1] done seen, led[2] error, led[3] a received
// byte was lost (sticky, cleared by reset), led[4] DDR2 calibrated.
//
// Model configuration: the board "nexys" of python/make_model.py, 112 x 112
// (50 tokens), any number of layers from 1 to 12: the pool depths below are
// that board's (BOARDS in make_model.py), the ROM images in rtl/rom/nexys hold
// the GELU and exp tables of all 12 layers. The layers and the weights come
// from DDR at run time, so another layer count needs no new bitstream.
//////////////////////////////////////////////////////////////////////////////////

module nexys_top_ddr #(
    parameter int CLOCK_HZ     = 100_000_000,
    parameter int BAUD_RATE    = 921_600,
    parameter int RESET_CYCLES = 255
) (
    input  logic        CLK100MHZ,
    input  logic        ck_rst,
    input  logic        uart_txd_in,
    output logic        uart_rxd_out,
    output logic [4:0]  led,

    output logic [12:0] ddr2_addr,
    output logic [2:0]  ddr2_ba,
    output logic        ddr2_cas_n,
    output logic [0:0]  ddr2_ck_n,
    output logic [0:0]  ddr2_ck_p,
    output logic [0:0]  ddr2_cke,
    output logic [0:0]  ddr2_cs_n,
    output logic [1:0]  ddr2_dm,
    inout  wire  [15:0] ddr2_dq,
    inout  wire  [1:0]  ddr2_dqs_n,
    inout  wire  [1:0]  ddr2_dqs_p,
    output logic [0:0]  ddr2_odt,
    output logic        ddr2_ras_n,
    output logic        ddr2_we_n
);

    localparam int CLOCKS_PER_BIT = (CLOCK_HZ + BAUD_RATE / 2) / BAUD_RATE;
    localparam int RESET_WIDTH    = $clog2(RESET_CYCLES + 1);

    // ------------------------------------------------------------------
    // clocks
    // ------------------------------------------------------------------
    logic CLK;
    logic reference_clock;
    logic clock_locked;
    logic ui_clk;
    logic ui_clk_sync_rst;

    system_clock_generator clocks (
        .clk_in1  (CLK100MHZ),
        .clk_out1 (CLK),
        .clk_out2 (reference_clock),
        .locked   (clock_locked)
    );

    // ------------------------------------------------------------------
    // resets
    // ------------------------------------------------------------------
    (* ASYNC_REG = "TRUE" *) logic [1:0] button_sync;
    (* ASYNC_REG = "TRUE" *) logic [1:0] locked_sync;
    (* ASYNC_REG = "TRUE" *) logic [1:0] calibrated_sync;
    logic [RESET_WIDTH-1:0]              reset_count;
    logic                                reset;
    logic                                memory_resetn;
    logic                                memory_aresetn;
    logic                                memory_ready;
    logic                                init_calib_complete;

    always_ff @(posedge CLK) begin
        button_sync     <= {button_sync[0], ck_rst};
        locked_sync     <= {locked_sync[0], clock_locked};
        calibrated_sync <= {calibrated_sync[0], init_calib_complete};
        if (!button_sync[1] || !locked_sync[1]) begin
            reset_count <= '0;
        end
        else if (reset_count != RESET_WIDTH'(RESET_CYCLES)) begin
            reset_count <= reset_count + 1'b1;
        end
        reset         <= reset_count != RESET_WIDTH'(RESET_CYCLES);
        memory_resetn <= button_sync[1] && locked_sync[1];
    end

    assign memory_ready = calibrated_sync[1];

    always_ff @(posedge ui_clk) begin
        memory_aresetn <= !ui_clk_sync_rst;
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
    // the accelerator and the memory system
    // ------------------------------------------------------------------
    logic        busy;
    logic        done_flag;
    logic        error_flag;
    logic        overrun_flag;

    logic [31:0] system_awaddr, system_araddr, system_wdata, system_rdata;
    logic        system_awvalid, system_awready, system_wvalid, system_wready, system_bvalid, system_bready;
    logic        system_arvalid, system_arready, system_rvalid, system_rready;
    logic [3:0]  system_wstrb;
    logic [1:0]  system_bresp, system_rresp;
    logic [63:0] dma_tdata;
    logic [7:0]  dma_tkeep;
    logic        dma_tlast, dma_tvalid, dma_tready;

    accelerator_system #(
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
        .GELU_ROM_FILE  ("../rom/nexys/gelu_rom.hex"),
        .EXP_ROM_FILE   ("../rom/nexys/exp_rom.hex"),
        .RSQRT_ROM_FILE ("../rom/nexys/rsqrt_rom.hex")
    ) accelerator (
        .aclk                 (CLK),
        .aresetn              (!reset),
        .rx_data              (rx_data),
        .rx_valid             (rx_valid),
        .rx_ready             (rx_ready),
        .tx_data              (tx_data),
        .tx_valid             (tx_valid),
        .tx_ready             (tx_ready),
        .m_system_axi_awaddr  (system_awaddr),
        .m_system_axi_awvalid (system_awvalid),
        .m_system_axi_awready (system_awready),
        .m_system_axi_wdata   (system_wdata),
        .m_system_axi_wstrb   (system_wstrb),
        .m_system_axi_wvalid  (system_wvalid),
        .m_system_axi_wready  (system_wready),
        .m_system_axi_bresp   (system_bresp),
        .m_system_axi_bvalid  (system_bvalid),
        .m_system_axi_bready  (system_bready),
        .m_system_axi_araddr  (system_araddr),
        .m_system_axi_arvalid (system_arvalid),
        .m_system_axi_arready (system_arready),
        .m_system_axi_rdata   (system_rdata),
        .m_system_axi_rresp   (system_rresp),
        .m_system_axi_rvalid  (system_rvalid),
        .m_system_axi_rready  (system_rready),
        .s_axis_dma_tdata     (dma_tdata),
        .s_axis_dma_tkeep     (dma_tkeep),
        .s_axis_dma_tlast     (dma_tlast),
        .s_axis_dma_tvalid    (dma_tvalid),
        .s_axis_dma_tready    (dma_tready),
        .memory_ready         (memory_ready),
        .busy                 (busy),
        .done_flag            (done_flag),
        .error_flag           (error_flag)
    );

    memory_system_wrapper memory (
        .DDR2_addr           (ddr2_addr),
        .DDR2_ba             (ddr2_ba),
        .DDR2_cas_n          (ddr2_cas_n),
        .DDR2_ck_n           (ddr2_ck_n),
        .DDR2_ck_p           (ddr2_ck_p),
        .DDR2_cke            (ddr2_cke),
        .DDR2_cs_n           (ddr2_cs_n),
        .DDR2_dm             (ddr2_dm),
        .DDR2_dq             (ddr2_dq),
        .DDR2_dqs_n          (ddr2_dqs_n),
        .DDR2_dqs_p          (ddr2_dqs_p),
        .DDR2_odt            (ddr2_odt),
        .DDR2_ras_n          (ddr2_ras_n),
        .DDR2_we_n           (ddr2_we_n),
        .M_AXIS_DMA_tdata    (dma_tdata),
        .M_AXIS_DMA_tkeep    (dma_tkeep),
        .M_AXIS_DMA_tlast    (dma_tlast),
        .M_AXIS_DMA_tready   (dma_tready),
        .M_AXIS_DMA_tvalid   (dma_tvalid),
        .S_AXI_HOST_araddr   (system_araddr),
        .S_AXI_HOST_arprot   (3'b000),
        .S_AXI_HOST_arready  (system_arready),
        .S_AXI_HOST_arvalid  (system_arvalid),
        .S_AXI_HOST_awaddr   (system_awaddr),
        .S_AXI_HOST_awprot   (3'b000),
        .S_AXI_HOST_awready  (system_awready),
        .S_AXI_HOST_awvalid  (system_awvalid),
        .S_AXI_HOST_bready   (system_bready),
        .S_AXI_HOST_bresp    (system_bresp),
        .S_AXI_HOST_bvalid   (system_bvalid),
        .S_AXI_HOST_rdata    (system_rdata),
        .S_AXI_HOST_rready   (system_rready),
        .S_AXI_HOST_rresp    (system_rresp),
        .S_AXI_HOST_rvalid   (system_rvalid),
        .S_AXI_HOST_wdata    (system_wdata),
        .S_AXI_HOST_wready   (system_wready),
        .S_AXI_HOST_wstrb    (system_wstrb),
        .S_AXI_HOST_wvalid   (system_wvalid),
        .clk_100             (CLK),
        .clk_200             (reference_clock),
        .init_calib_complete (init_calib_complete),
        .memory_aresetn      (memory_aresetn),
        .memory_resetn       (memory_resetn),
        .system_aresetn      (!reset),
        .ui_clk              (ui_clk),
        .ui_clk_sync_rst     (ui_clk_sync_rst)
    );

    always_ff @(posedge CLK) begin
        if (reset) begin
            overrun_flag <= 1'b0;
        end
        else if (rx_overrun) begin
            overrun_flag <= 1'b1;
        end
        led <= {memory_ready, overrun_flag, error_flag, done_flag, busy};
    end

endmodule
