//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2022.2 (lin64) Build 3671981 Fri Oct 14 04:59:54 MDT 2022
//Date        : Mon Oct 26 21:57:38 2026
//Host        : mutahir-HP-ProBook-450-G7 running 64-bit Ubuntu 24.04.4 LTS
//Command     : generate_target memory_system_wrapper.bd
//Design      : memory_system_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module memory_system_wrapper
   (DDR2_addr,
    DDR2_ba,
    DDR2_cas_n,
    DDR2_ck_n,
    DDR2_ck_p,
    DDR2_cke,
    DDR2_cs_n,
    DDR2_dm,
    DDR2_dq,
    DDR2_dqs_n,
    DDR2_dqs_p,
    DDR2_odt,
    DDR2_ras_n,
    DDR2_we_n,
    M_AXIS_DMA_tdata,
    M_AXIS_DMA_tkeep,
    M_AXIS_DMA_tlast,
    M_AXIS_DMA_tready,
    M_AXIS_DMA_tvalid,
    S_AXI_HOST_araddr,
    S_AXI_HOST_arprot,
    S_AXI_HOST_arready,
    S_AXI_HOST_arvalid,
    S_AXI_HOST_awaddr,
    S_AXI_HOST_awprot,
    S_AXI_HOST_awready,
    S_AXI_HOST_awvalid,
    S_AXI_HOST_bready,
    S_AXI_HOST_bresp,
    S_AXI_HOST_bvalid,
    S_AXI_HOST_rdata,
    S_AXI_HOST_rready,
    S_AXI_HOST_rresp,
    S_AXI_HOST_rvalid,
    S_AXI_HOST_wdata,
    S_AXI_HOST_wready,
    S_AXI_HOST_wstrb,
    S_AXI_HOST_wvalid,
    clk_100,
    clk_200,
    init_calib_complete,
    memory_aresetn,
    memory_resetn,
    system_aresetn,
    ui_clk,
    ui_clk_sync_rst);
  output [12:0]DDR2_addr;
  output [2:0]DDR2_ba;
  output DDR2_cas_n;
  output [0:0]DDR2_ck_n;
  output [0:0]DDR2_ck_p;
  output [0:0]DDR2_cke;
  output [0:0]DDR2_cs_n;
  output [1:0]DDR2_dm;
  inout [15:0]DDR2_dq;
  inout [1:0]DDR2_dqs_n;
  inout [1:0]DDR2_dqs_p;
  output [0:0]DDR2_odt;
  output DDR2_ras_n;
  output DDR2_we_n;
  output [63:0]M_AXIS_DMA_tdata;
  output [7:0]M_AXIS_DMA_tkeep;
  output M_AXIS_DMA_tlast;
  input M_AXIS_DMA_tready;
  output M_AXIS_DMA_tvalid;
  input [31:0]S_AXI_HOST_araddr;
  input [2:0]S_AXI_HOST_arprot;
  output S_AXI_HOST_arready;
  input S_AXI_HOST_arvalid;
  input [31:0]S_AXI_HOST_awaddr;
  input [2:0]S_AXI_HOST_awprot;
  output S_AXI_HOST_awready;
  input S_AXI_HOST_awvalid;
  input S_AXI_HOST_bready;
  output [1:0]S_AXI_HOST_bresp;
  output S_AXI_HOST_bvalid;
  output [31:0]S_AXI_HOST_rdata;
  input S_AXI_HOST_rready;
  output [1:0]S_AXI_HOST_rresp;
  output S_AXI_HOST_rvalid;
  input [31:0]S_AXI_HOST_wdata;
  output S_AXI_HOST_wready;
  input [3:0]S_AXI_HOST_wstrb;
  input S_AXI_HOST_wvalid;
  input clk_100;
  input clk_200;
  output init_calib_complete;
  input memory_aresetn;
  input memory_resetn;
  input system_aresetn;
  output ui_clk;
  output ui_clk_sync_rst;

  wire [12:0]DDR2_addr;
  wire [2:0]DDR2_ba;
  wire DDR2_cas_n;
  wire [0:0]DDR2_ck_n;
  wire [0:0]DDR2_ck_p;
  wire [0:0]DDR2_cke;
  wire [0:0]DDR2_cs_n;
  wire [1:0]DDR2_dm;
  wire [15:0]DDR2_dq;
  wire [1:0]DDR2_dqs_n;
  wire [1:0]DDR2_dqs_p;
  wire [0:0]DDR2_odt;
  wire DDR2_ras_n;
  wire DDR2_we_n;
  wire [63:0]M_AXIS_DMA_tdata;
  wire [7:0]M_AXIS_DMA_tkeep;
  wire M_AXIS_DMA_tlast;
  wire M_AXIS_DMA_tready;
  wire M_AXIS_DMA_tvalid;
  wire [31:0]S_AXI_HOST_araddr;
  wire [2:0]S_AXI_HOST_arprot;
  wire S_AXI_HOST_arready;
  wire S_AXI_HOST_arvalid;
  wire [31:0]S_AXI_HOST_awaddr;
  wire [2:0]S_AXI_HOST_awprot;
  wire S_AXI_HOST_awready;
  wire S_AXI_HOST_awvalid;
  wire S_AXI_HOST_bready;
  wire [1:0]S_AXI_HOST_bresp;
  wire S_AXI_HOST_bvalid;
  wire [31:0]S_AXI_HOST_rdata;
  wire S_AXI_HOST_rready;
  wire [1:0]S_AXI_HOST_rresp;
  wire S_AXI_HOST_rvalid;
  wire [31:0]S_AXI_HOST_wdata;
  wire S_AXI_HOST_wready;
  wire [3:0]S_AXI_HOST_wstrb;
  wire S_AXI_HOST_wvalid;
  wire clk_100;
  wire clk_200;
  wire init_calib_complete;
  wire memory_aresetn;
  wire memory_resetn;
  wire system_aresetn;
  wire ui_clk;
  wire ui_clk_sync_rst;

  memory_system memory_system_i
       (.DDR2_addr(DDR2_addr),
        .DDR2_ba(DDR2_ba),
        .DDR2_cas_n(DDR2_cas_n),
        .DDR2_ck_n(DDR2_ck_n),
        .DDR2_ck_p(DDR2_ck_p),
        .DDR2_cke(DDR2_cke),
        .DDR2_cs_n(DDR2_cs_n),
        .DDR2_dm(DDR2_dm),
        .DDR2_dq(DDR2_dq),
        .DDR2_dqs_n(DDR2_dqs_n),
        .DDR2_dqs_p(DDR2_dqs_p),
        .DDR2_odt(DDR2_odt),
        .DDR2_ras_n(DDR2_ras_n),
        .DDR2_we_n(DDR2_we_n),
        .M_AXIS_DMA_tdata(M_AXIS_DMA_tdata),
        .M_AXIS_DMA_tkeep(M_AXIS_DMA_tkeep),
        .M_AXIS_DMA_tlast(M_AXIS_DMA_tlast),
        .M_AXIS_DMA_tready(M_AXIS_DMA_tready),
        .M_AXIS_DMA_tvalid(M_AXIS_DMA_tvalid),
        .S_AXI_HOST_araddr(S_AXI_HOST_araddr),
        .S_AXI_HOST_arprot(S_AXI_HOST_arprot),
        .S_AXI_HOST_arready(S_AXI_HOST_arready),
        .S_AXI_HOST_arvalid(S_AXI_HOST_arvalid),
        .S_AXI_HOST_awaddr(S_AXI_HOST_awaddr),
        .S_AXI_HOST_awprot(S_AXI_HOST_awprot),
        .S_AXI_HOST_awready(S_AXI_HOST_awready),
        .S_AXI_HOST_awvalid(S_AXI_HOST_awvalid),
        .S_AXI_HOST_bready(S_AXI_HOST_bready),
        .S_AXI_HOST_bresp(S_AXI_HOST_bresp),
        .S_AXI_HOST_bvalid(S_AXI_HOST_bvalid),
        .S_AXI_HOST_rdata(S_AXI_HOST_rdata),
        .S_AXI_HOST_rready(S_AXI_HOST_rready),
        .S_AXI_HOST_rresp(S_AXI_HOST_rresp),
        .S_AXI_HOST_rvalid(S_AXI_HOST_rvalid),
        .S_AXI_HOST_wdata(S_AXI_HOST_wdata),
        .S_AXI_HOST_wready(S_AXI_HOST_wready),
        .S_AXI_HOST_wstrb(S_AXI_HOST_wstrb),
        .S_AXI_HOST_wvalid(S_AXI_HOST_wvalid),
        .clk_100(clk_100),
        .clk_200(clk_200),
        .init_calib_complete(init_calib_complete),
        .memory_aresetn(memory_aresetn),
        .memory_resetn(memory_resetn),
        .system_aresetn(system_aresetn),
        .ui_clk(ui_clk),
        .ui_clk_sync_rst(ui_clk_sync_rst));
endmodule
