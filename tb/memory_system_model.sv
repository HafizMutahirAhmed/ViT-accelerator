`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: memory_system_wrapper (simulation model)
// Description: Behavioural model of the block design memory_system
// (scripts/create_memory_system_bd.tcl) with the generated wrapper's name and
// ports, for the testbenches of accelerator_system and nexys_top_ddr. It is
// what the bridge sees of the MIG, the AXI DMA and the SmartConnect:
//
//   S_AXI_HOST  AXI4-Lite slave on clk_100. 0x4040_0000: the DMA's MM2S
//               registers DMACR (0x00, bit 0 run), DMASR (0x04: bit 0 halted,
//               bit 1 idle, bit 4 internal error, bit 12 IOC), SA (0x18),
//               LENGTH (0x28, a write starts a transfer); other offsets read
//               0. 0x8000_0000 + 128 MiB: the DDR, 32-bit words, unwritten
//               words read 0. Anything else answers DECERR.
//   M_AXIS_DMA  the transfer: LENGTH / 8 beats from DDR byte SA, TLAST on the
//               last. A length that is not a multiple of 8, an SA that is not
//               8-byte aligned or outside the DDR, or LENGTH written while not
//               running sets the internal error bit and moves nothing.
//
// It makes the bridge's life hard on purpose: AWREADY, WREADY, ARREADY and
// the response VALIDs come after pseudo-random delays, the stream has
// pseudo-random gaps, DMASR's idle bit only falls IDLE_LAG cycles after the
// LENGTH write and only rises IDLE_LAG cycles after the last beat.
// init_calib_complete rises CALIBRATION_CYCLES clocks of clk_100 after
// memory_resetn and system_aresetn are high. ui_clk is a free-running 81.25 MHz
// clock; ui_clk_sync_rst follows memory_resetn. The DDR2 pins are idle.
// beats_sent and transfers count what the DMA moved.
//////////////////////////////////////////////////////////////////////////////////

module memory_system_wrapper #(
    parameter int CALIBRATION_CYCLES = 200,
    parameter int IDLE_LAG           = 6
) (
    output logic [12:0] DDR2_addr,
    output logic [2:0]  DDR2_ba,
    output logic        DDR2_cas_n,
    output logic [0:0]  DDR2_ck_n,
    output logic [0:0]  DDR2_ck_p,
    output logic [0:0]  DDR2_cke,
    output logic [0:0]  DDR2_cs_n,
    output logic [1:0]  DDR2_dm,
    inout  wire  [15:0] DDR2_dq,
    inout  wire  [1:0]  DDR2_dqs_n,
    inout  wire  [1:0]  DDR2_dqs_p,
    output logic [0:0]  DDR2_odt,
    output logic        DDR2_ras_n,
    output logic        DDR2_we_n,
    output logic [63:0] M_AXIS_DMA_tdata,
    output logic [7:0]  M_AXIS_DMA_tkeep,
    output logic        M_AXIS_DMA_tlast,
    input  logic        M_AXIS_DMA_tready,
    output logic        M_AXIS_DMA_tvalid,
    input  logic [31:0] S_AXI_HOST_araddr,
    input  logic [2:0]  S_AXI_HOST_arprot,
    output logic        S_AXI_HOST_arready,
    input  logic        S_AXI_HOST_arvalid,
    input  logic [31:0] S_AXI_HOST_awaddr,
    input  logic [2:0]  S_AXI_HOST_awprot,
    output logic        S_AXI_HOST_awready,
    input  logic        S_AXI_HOST_awvalid,
    input  logic        S_AXI_HOST_bready,
    output logic [1:0]  S_AXI_HOST_bresp,
    output logic        S_AXI_HOST_bvalid,
    output logic [31:0] S_AXI_HOST_rdata,
    input  logic        S_AXI_HOST_rready,
    output logic [1:0]  S_AXI_HOST_rresp,
    output logic        S_AXI_HOST_rvalid,
    input  logic [31:0] S_AXI_HOST_wdata,
    output logic        S_AXI_HOST_wready,
    input  logic [3:0]  S_AXI_HOST_wstrb,
    input  logic        S_AXI_HOST_wvalid,
    input  logic        clk_100,
    input  logic        clk_200,
    output logic        init_calib_complete,
    input  logic        memory_aresetn,
    input  logic        memory_resetn,
    input  logic        system_aresetn,
    output logic        ui_clk,
    output logic        ui_clk_sync_rst
);

    localparam logic [31:0] DMA_BASE = 32'h4040_0000;
    localparam logic [31:0] DDR_BASE = 32'h8000_0000;
    localparam longint      DDR_SIZE = 64'd134217728;

    assign DDR2_addr  = '0;
    assign DDR2_ba    = '0;
    assign DDR2_cas_n = 1'b1;
    assign DDR2_ck_n  = 1'b1;
    assign DDR2_ck_p  = 1'b0;
    assign DDR2_cke   = 1'b0;
    assign DDR2_cs_n  = 1'b1;
    assign DDR2_dm    = '0;
    assign DDR2_odt   = 1'b0;
    assign DDR2_ras_n = 1'b1;
    assign DDR2_we_n  = 1'b1;

    initial ui_clk = 1'b0;
    always #6.154 ui_clk = ~ui_clk;
    assign ui_clk_sync_rst = !memory_resetn;

    logic reset;
    assign reset = !system_aresetn;

    logic [31:0] ddr_words[longint];
    logic [15:0] lfsr = 16'hBEEF;
    logic        coin;           // pseudo-random, one draw per cycle

    always @(posedge clk_100) begin
        lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
    end
    assign coin = lfsr[2] ^ lfsr[7];

    // ------------------------------------------------------------------
    // calibration
    // ------------------------------------------------------------------
    int calibration_count = 0;

    always @(posedge clk_100) begin
        if (!memory_resetn || reset) begin
            calibration_count   <= 0;
            init_calib_complete <= 1'b0;
        end
        else if (calibration_count < CALIBRATION_CYCLES) begin
            calibration_count <= calibration_count + 1;
        end
        else begin
            init_calib_complete <= 1'b1;
        end
    end

    // ------------------------------------------------------------------
    // the DMA's registers and state
    // ------------------------------------------------------------------
    logic        running;
    logic        idle;
    logic        internal_error;
    logic        ioc;
    logic [31:0] source_address;
    logic        transfer_active;
    longint      next_byte;
    longint      beats_left;
    int          idle_count;
    logic        start_pending;
    longint      beats_sent = 0;
    int          transfers  = 0;

    function automatic logic [31:0] dmasr();
        return {19'b0, ioc, 7'b0, internal_error, 2'b0, idle, !running};
    endfunction

    function automatic logic [31:0] ddr_read(input longint byte_address);
        longint word;
        word = byte_address >> 2;
        return ddr_words.exists(word) ? ddr_words[word] : 32'h0;
    endfunction

    // ------------------------------------------------------------------
    // AXI4-Lite slave: one write and one read at a time, random delays
    // ------------------------------------------------------------------
    logic        write_address_taken, write_data_taken;
    logic [31:0] write_address, write_data;
    int          write_delay, read_delay;
    logic        read_address_taken;
    logic [31:0] read_address;

    assign S_AXI_HOST_awready = !reset && !write_address_taken && coin;
    assign S_AXI_HOST_wready  = !reset && !write_data_taken && lfsr[5];
    assign S_AXI_HOST_arready = !reset && !read_address_taken && !S_AXI_HOST_rvalid && lfsr[9];

    always @(posedge clk_100) begin
        if (reset) begin
            write_address_taken <= 1'b0;
            write_data_taken    <= 1'b0;
            read_address_taken  <= 1'b0;
            S_AXI_HOST_bvalid   <= 1'b0;
            S_AXI_HOST_rvalid   <= 1'b0;
            running             <= 1'b0;
            idle                <= 1'b1;
            internal_error      <= 1'b0;
            ioc                 <= 1'b0;
            transfer_active     <= 1'b0;
            start_pending       <= 1'b0;
            M_AXIS_DMA_tvalid   <= 1'b0;
        end
        else begin
            // ---- writes
            if (S_AXI_HOST_awvalid && S_AXI_HOST_awready) begin
                write_address_taken <= 1'b1;
                write_address       <= S_AXI_HOST_awaddr;
                write_delay         <= {lfsr[3:0]};
            end
            if (S_AXI_HOST_wvalid && S_AXI_HOST_wready) begin
                write_data_taken <= 1'b1;
                write_data       <= S_AXI_HOST_wdata;
            end
            if (write_address_taken && write_data_taken && !S_AXI_HOST_bvalid) begin
                if (write_delay > 0) begin
                    write_delay <= write_delay - 1;
                end
                else begin
                    S_AXI_HOST_bvalid   <= 1'b1;
                    S_AXI_HOST_bresp    <= 2'b00;
                    write_address_taken <= 1'b0;
                    write_data_taken    <= 1'b0;
                    if (write_address >= DDR_BASE && longint'(write_address - DDR_BASE) < DDR_SIZE) begin
                        ddr_words[longint'(write_address - DDR_BASE) >> 2] = write_data;
                    end
                    else if (write_address == DMA_BASE + 32'h00) begin
                        running <= write_data[0];
                    end
                    else if (write_address == DMA_BASE + 32'h18) begin
                        source_address <= write_data;
                    end
                    else if (write_address == DMA_BASE + 32'h28) begin
                        if (!running || write_data[2:0] != 3'b000 || source_address[2:0] != 3'b000
                            || source_address < DDR_BASE
                            || longint'(source_address - DDR_BASE) + longint'(write_data) > DDR_SIZE) begin
                            internal_error <= 1'b1;
                        end
                        else if (write_data != 0) begin
                            start_pending <= 1'b1;
                            next_byte     <= longint'(source_address - DDR_BASE);
                            beats_left    <= longint'(write_data) / 8;
                            idle_count    <= IDLE_LAG;
                            transfers     <= transfers + 1;
                        end
                    end
                    else if ((write_address & 32'hFFFF_0000) != DMA_BASE) begin
                        S_AXI_HOST_bresp <= 2'b11;
                    end
                end
            end
            if (S_AXI_HOST_bvalid && S_AXI_HOST_bready) begin
                S_AXI_HOST_bvalid <= 1'b0;
            end

            // ---- reads
            if (S_AXI_HOST_arvalid && S_AXI_HOST_arready) begin
                read_address_taken <= 1'b1;
                read_address       <= S_AXI_HOST_araddr;
                read_delay         <= {lfsr[4:1]};
            end
            if (read_address_taken) begin
                if (read_delay > 0) begin
                    read_delay <= read_delay - 1;
                end
                else begin
                    read_address_taken <= 1'b0;
                    S_AXI_HOST_rvalid  <= 1'b1;
                    S_AXI_HOST_rresp   <= 2'b00;
                    S_AXI_HOST_rdata   <= 32'h0;
                    if (read_address >= DDR_BASE && longint'(read_address - DDR_BASE) < DDR_SIZE) begin
                        S_AXI_HOST_rdata <= ddr_read(longint'(read_address - DDR_BASE));
                    end
                    else if (read_address == DMA_BASE + 32'h00) begin
                        S_AXI_HOST_rdata <= {31'b0, running};
                    end
                    else if (read_address == DMA_BASE + 32'h04) begin
                        S_AXI_HOST_rdata <= dmasr();
                    end
                    else if (read_address == DMA_BASE + 32'h18) begin
                        S_AXI_HOST_rdata <= source_address;
                    end
                    else if ((read_address & 32'hFFFF_0000) != DMA_BASE) begin
                        S_AXI_HOST_rresp <= 2'b11;
                    end
                end
            end
            if (S_AXI_HOST_rvalid && S_AXI_HOST_rready) begin
                S_AXI_HOST_rvalid <= 1'b0;
            end

            // ---- the transfer: idle falls late, beats with gaps, idle rises late
            if (start_pending) begin
                if (idle_count > 0) begin
                    idle_count <= idle_count - 1;
                end
                else begin
                    start_pending   <= 1'b0;
                    idle            <= 1'b0;
                    transfer_active <= 1'b1;
                    idle_count      <= IDLE_LAG;
                end
            end
            if (transfer_active) begin
                if (M_AXIS_DMA_tvalid && M_AXIS_DMA_tready) begin
                    M_AXIS_DMA_tvalid <= 1'b0;
                    beats_sent        <= beats_sent + 1;
                end
                if (beats_left > 0 && (!M_AXIS_DMA_tvalid || M_AXIS_DMA_tready) && lfsr[11]) begin
                    M_AXIS_DMA_tvalid <= 1'b1;
                    M_AXIS_DMA_tdata  <= {ddr_read(next_byte + 4), ddr_read(next_byte)};
                    M_AXIS_DMA_tkeep  <= '1;
                    M_AXIS_DMA_tlast  <= beats_left == 1;
                    next_byte         <= next_byte + 8;
                    beats_left        <= beats_left - 1;
                end
                else if (beats_left == 0 && !(M_AXIS_DMA_tvalid && !M_AXIS_DMA_tready)) begin
                    if (idle_count > 0) begin
                        idle_count <= idle_count - 1;
                    end
                    else begin
                        transfer_active <= 1'b0;
                        idle            <= 1'b1;
                        ioc             <= 1'b1;
                    end
                end
            end
        end
    end

endmodule
