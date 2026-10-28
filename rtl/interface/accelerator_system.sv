`timescale 1ns / 1ps

module accelerator_system #(
    parameter int          SYS_ROW          = 8,
    parameter int          SYS_COL          = 8,
    parameter int          BUFFER_DEPTH_0   = 4096,
    parameter int          BUFFER_DEPTH_1   = 4096,
    parameter int          BUFFER_DEPTH_2   = 4096,
    parameter int          BUFFER_DEPTH_3   = 4096,
    parameter int          BUFFER_DEPTH_4   = 4096,
    parameter int          BUFFER_DEPTH_5   = 4096,
    parameter int          BUFFER_DEPTH_6   = 4096,
    parameter int          BUFFER_DEPTH_7   = 4096,
    parameter              GELU_ROM_FILE    = "rom/gelu_rom.hex",
    parameter              EXP_ROM_FILE     = "rom/exp_rom.hex",
    parameter              RSQRT_ROM_FILE   = "rom/rsqrt_rom.hex",
    parameter logic [31:0] DMA_BASE_ADDRESS = 32'h4040_0000,
    parameter logic [31:0] DDR_BASE_ADDRESS = 32'h8000_0000
) (
    input  logic        aclk,
    input  logic        aresetn,

    input  logic [7:0]  rx_data,
    input  logic        rx_valid,
    output logic        rx_ready,
    output logic [7:0]  tx_data,
    output logic        tx_valid,
    input  logic        tx_ready,

    output logic [31:0] m_system_axi_awaddr,
    output logic        m_system_axi_awvalid,
    input  logic        m_system_axi_awready,
    output logic [31:0] m_system_axi_wdata,
    output logic [3:0]  m_system_axi_wstrb,
    output logic        m_system_axi_wvalid,
    input  logic        m_system_axi_wready,
    input  logic [1:0]  m_system_axi_bresp,
    input  logic        m_system_axi_bvalid,
    output logic        m_system_axi_bready,
    output logic [31:0] m_system_axi_araddr,
    output logic        m_system_axi_arvalid,
    input  logic        m_system_axi_arready,
    input  logic [31:0] m_system_axi_rdata,
    input  logic [1:0]  m_system_axi_rresp,
    input  logic        m_system_axi_rvalid,
    output logic        m_system_axi_rready,

    input  logic [63:0] s_axis_dma_tdata,
    input  logic [7:0]  s_axis_dma_tkeep,
    input  logic        s_axis_dma_tlast,
    input  logic        s_axis_dma_tvalid,
    output logic        s_axis_dma_tready,

    input  logic        memory_ready,

    output logic        busy,
    output logic        done_flag,
    output logic        error_flag
);

    logic [11:0] axi_awaddr, axi_araddr;
    logic        axi_awvalid, axi_awready, axi_wvalid, axi_wready, axi_bvalid, axi_bready;
    logic        axi_arvalid, axi_arready, axi_rvalid, axi_rready;
    logic [31:0] axi_wdata, axi_rdata;
    logic [3:0]  axi_wstrb;
    logic [1:0]  axi_bresp, axi_rresp;

    logic [63:0] bridge_load_tdata;
    logic [7:0]  bridge_load_tkeep;
    logic        bridge_load_tlast, bridge_load_tvalid, bridge_load_tready;
    logic [63:0] load_tdata, store_tdata;
    logic [7:0]  load_tkeep, store_tkeep;
    logic        load_tlast, load_tvalid, load_tready, store_tlast, store_tvalid, store_tready;

    logic dma_stream_select;
    logic dma_stream_last;
    logic bridge_error;
    logic wrapper_error;

    assign error_flag = bridge_error || wrapper_error;

    // the wrapper's load stream: the bridge's, or the DMA's during DMA_LOAD
    assign load_tdata         = dma_stream_select ? s_axis_dma_tdata  : bridge_load_tdata;
    assign load_tkeep         = dma_stream_select ? s_axis_dma_tkeep  : bridge_load_tkeep;
    assign load_tlast         = dma_stream_select ? s_axis_dma_tlast  : bridge_load_tlast;
    assign load_tvalid        = dma_stream_select ? s_axis_dma_tvalid : bridge_load_tvalid;
    assign bridge_load_tready = !dma_stream_select && load_tready;
    assign s_axis_dma_tready  = dma_stream_select && load_tready;
    assign dma_stream_last    = s_axis_dma_tvalid && s_axis_dma_tready && s_axis_dma_tlast;

    axi_host_bridge #(
        .DMA_BASE_ADDRESS (DMA_BASE_ADDRESS),
        .DDR_BASE_ADDRESS (DDR_BASE_ADDRESS)
    ) bridge (
        .aclk                 (aclk),
        .aresetn              (aresetn),
        .rx_data              (rx_data),
        .rx_valid             (rx_valid),
        .rx_ready             (rx_ready),
        .tx_data              (tx_data),
        .tx_valid             (tx_valid),
        .tx_ready             (tx_ready),
        .m_axi_awaddr         (axi_awaddr),
        .m_axi_awvalid        (axi_awvalid),
        .m_axi_awready        (axi_awready),
        .m_axi_wdata          (axi_wdata),
        .m_axi_wstrb          (axi_wstrb),
        .m_axi_wvalid         (axi_wvalid),
        .m_axi_wready         (axi_wready),
        .m_axi_bresp          (axi_bresp),
        .m_axi_bvalid         (axi_bvalid),
        .m_axi_bready         (axi_bready),
        .m_axi_araddr         (axi_araddr),
        .m_axi_arvalid        (axi_arvalid),
        .m_axi_arready        (axi_arready),
        .m_axi_rdata          (axi_rdata),
        .m_axi_rresp          (axi_rresp),
        .m_axi_rvalid         (axi_rvalid),
        .m_axi_rready         (axi_rready),
        .m_axis_tdata         (bridge_load_tdata),
        .m_axis_tkeep         (bridge_load_tkeep),
        .m_axis_tlast         (bridge_load_tlast),
        .m_axis_tvalid        (bridge_load_tvalid),
        .m_axis_tready        (bridge_load_tready),
        .s_axis_tdata         (store_tdata),
        .s_axis_tkeep         (store_tkeep),
        .s_axis_tlast         (store_tlast),
        .s_axis_tvalid        (store_tvalid),
        .s_axis_tready        (store_tready),
        .m_system_axi_awaddr  (m_system_axi_awaddr),
        .m_system_axi_awvalid (m_system_axi_awvalid),
        .m_system_axi_awready (m_system_axi_awready),
        .m_system_axi_wdata   (m_system_axi_wdata),
        .m_system_axi_wstrb   (m_system_axi_wstrb),
        .m_system_axi_wvalid  (m_system_axi_wvalid),
        .m_system_axi_wready  (m_system_axi_wready),
        .m_system_axi_bresp   (m_system_axi_bresp),
        .m_system_axi_bvalid  (m_system_axi_bvalid),
        .m_system_axi_bready  (m_system_axi_bready),
        .m_system_axi_araddr  (m_system_axi_araddr),
        .m_system_axi_arvalid (m_system_axi_arvalid),
        .m_system_axi_arready (m_system_axi_arready),
        .m_system_axi_rdata   (m_system_axi_rdata),
        .m_system_axi_rresp   (m_system_axi_rresp),
        .m_system_axi_rvalid  (m_system_axi_rvalid),
        .m_system_axi_rready  (m_system_axi_rready),
        .memory_ready         (memory_ready),
        .dma_stream_select    (dma_stream_select),
        .dma_stream_last      (dma_stream_last),
        .error_flag           (bridge_error)
    );

    vit_axi_wrapper #(
        .SYS_ROW        (SYS_ROW),
        .SYS_COL        (SYS_COL),
        .BUFFER_DEPTH_0 (BUFFER_DEPTH_0),
        .BUFFER_DEPTH_1 (BUFFER_DEPTH_1),
        .BUFFER_DEPTH_2 (BUFFER_DEPTH_2),
        .BUFFER_DEPTH_3 (BUFFER_DEPTH_3),
        .BUFFER_DEPTH_4 (BUFFER_DEPTH_4),
        .BUFFER_DEPTH_5 (BUFFER_DEPTH_5),
        .BUFFER_DEPTH_6 (BUFFER_DEPTH_6),
        .BUFFER_DEPTH_7 (BUFFER_DEPTH_7),
        .GELU_ROM_FILE  (GELU_ROM_FILE),
        .EXP_ROM_FILE   (EXP_ROM_FILE),
        .RSQRT_ROM_FILE (RSQRT_ROM_FILE)
    ) accelerator (
        .aclk          (aclk),
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

endmodule
