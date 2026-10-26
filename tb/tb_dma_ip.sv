`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench of accelerator_system with the real AXI DMA: Xilinx's
// simulation model of axi_dma 7.1, configured as in the block design
// memory_system (scripts/export_dma_ip_sim.tcl), in place of the behavioural
// memory_system_model.sv. It checks the bridge's DMA_LOAD sequence against the
// DMA itself: MM2S_DMACR, MM2S_SA, MM2S_LENGTH, the DMASR polling, the stream
// with TLAST, through the stream multiplexer into the AXI wrapper.
//
// The testbench is the DDR: an AXI4-Lite slave for the bridge's accesses at
// 0x8000_0000 (the DMA's registers at 0x4040_0000 go to the DMA), and an AXI4
// read slave for the DMA's M_AXI_MM2S, with outstanding bursts, pseudo-random
// gaps in RVALID, and a check of every burst: INCR, 8-byte beats, inside the
// DDR, not across a 4 KiB boundary. Both slaves share one word store.
//
// Stimulus: python/ddr_check.py --model --bytes 4096 --record, the byte stream
// of the bring-up check against the software model of the board (DDR write and
// read-back, DMA_LOADs into pool buffers with READ_BUFFER of each, DMA_LOADs
// into the bias store, LayerNorm constants and descriptor memory, then two
// RUNs: a run list of three LOAD entries, and one whose second entry has an
// unknown kind, which the bridge must refuse after doing the first). Every
// response byte must equal the model's, and the engine must see one write per
// item of every DMA_LOAD and of every LOAD entry the RUNs carry out
// (host_stream_ddrcheck_summary.txt, as host.run_totals counts them).
//
// Run with scripts/sim_dma_ip.sh, not sim.sh: it needs the DMA's libraries.
// Vectors: python/models/nexys_L2/host_stream_ddrcheck_*.
//////////////////////////////////////////////////////////////////////////////////

module tb_dma_ip #(
    parameter string VECTOR_DIR = "../../python/models/nexys_L2/"
);

    localparam int          TX_BYTES_MAX     = 262144;
    localparam int          RX_BYTES_MAX     = 262144;
    localparam int          RESPONSE_TIMEOUT = 2000000;
    localparam logic [31:0] DDR_BASE         = 32'h8000_0000;
    localparam longint      DDR_SIZE         = 64'd134217728;

    logic       CLK = 1'b0;
    logic       reset;
    logic [7:0] rx_data;
    logic       rx_valid;
    logic       rx_ready;
    logic [7:0] tx_data;
    logic       tx_valid;
    logic       tx_ready;
    logic       busy;
    logic       done_flag;
    logic       error_flag;

    always #5 CLK = ~CLK;

    // the bridge's system master
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
        .GELU_ROM_FILE  ("../rom/input_112/gelu_rom.hex"),
        .EXP_ROM_FILE   ("../rom/input_112/exp_rom.hex"),
        .RSQRT_ROM_FILE ("../rom/input_112/rsqrt_rom.hex")
    ) dut (
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
        .memory_ready         (!reset),
        .busy                 (busy),
        .done_flag            (done_flag),
        .error_flag           (error_flag)
    );

    // ------------------------------------------------------------------
    // the address decoder of the bridge's accesses: DMA registers or DDR
    // (the bridge does one access at a time and holds its address registers)
    // ------------------------------------------------------------------
    logic write_to_dma, read_from_dma;
    assign write_to_dma  = system_awaddr[31:16] == 16'h4040;
    assign read_from_dma = system_araddr[31:16] == 16'h4040;

    logic        dma_awready, dma_wready, dma_bvalid, dma_arready, dma_rvalid;
    logic [1:0]  dma_bresp, dma_rresp;
    logic [31:0] dma_rdata;
    logic        ddr_awready, ddr_wready, ddr_bvalid, ddr_arready, ddr_rvalid;
    logic [31:0] ddr_rdata;

    assign system_awready = write_to_dma ? dma_awready : ddr_awready;
    assign system_wready  = write_to_dma ? dma_wready  : ddr_wready;
    assign system_bvalid  = write_to_dma ? dma_bvalid  : ddr_bvalid;
    assign system_bresp   = write_to_dma ? dma_bresp   : 2'b00;
    assign system_arready = read_from_dma ? dma_arready : ddr_arready;
    assign system_rvalid  = read_from_dma ? dma_rvalid  : ddr_rvalid;
    assign system_rdata   = read_from_dma ? dma_rdata   : ddr_rdata;
    assign system_rresp   = read_from_dma ? dma_rresp   : 2'b00;

    // ------------------------------------------------------------------
    // the real DMA
    // ------------------------------------------------------------------
    logic [31:0] mm2s_araddr;
    logic [7:0]  mm2s_arlen;
    logic [2:0]  mm2s_arsize;
    logic [1:0]  mm2s_arburst;
    logic        mm2s_arvalid, mm2s_arready;
    logic [63:0] mm2s_rdata;
    logic        mm2s_rlast, mm2s_rvalid, mm2s_rready;
    logic        dma_interrupt;

    dma_ip dma (
        .s_axi_lite_aclk        (CLK),
        .m_axi_mm2s_aclk        (CLK),
        .axi_resetn             (!reset),
        .s_axi_lite_awvalid     (system_awvalid && write_to_dma),
        .s_axi_lite_awready     (dma_awready),
        .s_axi_lite_awaddr      (system_awaddr[9:0]),
        .s_axi_lite_wvalid      (system_wvalid && write_to_dma),
        .s_axi_lite_wready      (dma_wready),
        .s_axi_lite_wdata       (system_wdata),
        .s_axi_lite_bresp       (dma_bresp),
        .s_axi_lite_bvalid      (dma_bvalid),
        .s_axi_lite_bready      (system_bready && write_to_dma),
        .s_axi_lite_arvalid     (system_arvalid && read_from_dma),
        .s_axi_lite_arready     (dma_arready),
        .s_axi_lite_araddr      (system_araddr[9:0]),
        .s_axi_lite_rvalid      (dma_rvalid),
        .s_axi_lite_rready      (system_rready && read_from_dma),
        .s_axi_lite_rdata       (dma_rdata),
        .s_axi_lite_rresp       (dma_rresp),
        .m_axi_mm2s_araddr      (mm2s_araddr),
        .m_axi_mm2s_arlen       (mm2s_arlen),
        .m_axi_mm2s_arsize      (mm2s_arsize),
        .m_axi_mm2s_arburst     (mm2s_arburst),
        .m_axi_mm2s_arprot      (),
        .m_axi_mm2s_arcache     (),
        .m_axi_mm2s_arvalid     (mm2s_arvalid),
        .m_axi_mm2s_arready     (mm2s_arready),
        .m_axi_mm2s_rdata       (mm2s_rdata),
        .m_axi_mm2s_rresp       (2'b00),
        .m_axi_mm2s_rlast       (mm2s_rlast),
        .m_axi_mm2s_rvalid      (mm2s_rvalid),
        .m_axi_mm2s_rready      (mm2s_rready),
        .mm2s_prmry_reset_out_n (),
        .m_axis_mm2s_tdata      (dma_tdata),
        .m_axis_mm2s_tkeep      (dma_tkeep),
        .m_axis_mm2s_tvalid     (dma_tvalid),
        .m_axis_mm2s_tready     (dma_tready),
        .m_axis_mm2s_tlast      (dma_tlast),
        .mm2s_introut           (dma_interrupt),
        .axi_dma_tstvec         ()
    );

    // ------------------------------------------------------------------
    // the DDR: a word store, an AXI4-Lite slave for the bridge, an AXI4 read
    // slave for the DMA
    // ------------------------------------------------------------------
    logic [31:0] ddr_words[longint];
    logic [15:0] lfsr = 16'h1D2C;

    always @(posedge CLK) begin
        lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
    end

    function automatic logic [31:0] ddr_word(input longint byte_offset);
        return ddr_words.exists(byte_offset >> 2) ? ddr_words[byte_offset >> 2] : 32'h0;
    endfunction

    logic        lite_write_address_taken, lite_write_data_taken, lite_read_taken;
    logic [31:0] lite_write_address, lite_write_data, lite_read_address;

    assign ddr_awready = !reset && !write_to_dma ? !lite_write_address_taken && lfsr[1] : 1'b0;
    assign ddr_wready  = !reset && !write_to_dma ? !lite_write_data_taken && lfsr[4] : 1'b0;
    assign ddr_arready = !reset && !read_from_dma ? !lite_read_taken && !ddr_rvalid && lfsr[6] : 1'b0;

    int bad_lite_addresses = 0;

    always @(posedge CLK) begin
        if (reset) begin
            lite_write_address_taken <= 1'b0;
            lite_write_data_taken    <= 1'b0;
            lite_read_taken          <= 1'b0;
            ddr_bvalid               <= 1'b0;
            ddr_rvalid               <= 1'b0;
        end
        else begin
            if (system_awvalid && ddr_awready) begin
                lite_write_address_taken <= 1'b1;
                lite_write_address       <= system_awaddr;
            end
            if (system_wvalid && ddr_wready) begin
                lite_write_data_taken <= 1'b1;
                lite_write_data       <= system_wdata;
            end
            if (lite_write_address_taken && lite_write_data_taken && !ddr_bvalid && lfsr[8]) begin
                if (lite_write_address < DDR_BASE) begin
                    bad_lite_addresses++;
                end
                else begin
                    ddr_words[longint'(lite_write_address - DDR_BASE) >> 2] = lite_write_data;
                end
                lite_write_address_taken <= 1'b0;
                lite_write_data_taken    <= 1'b0;
                ddr_bvalid               <= 1'b1;
            end
            if (ddr_bvalid && system_bready && !write_to_dma) begin
                ddr_bvalid <= 1'b0;
            end
            if (system_arvalid && ddr_arready) begin
                lite_read_taken   <= 1'b1;
                lite_read_address <= system_araddr;
            end
            if (lite_read_taken && lfsr[10]) begin
                lite_read_taken <= 1'b0;
                ddr_rvalid      <= 1'b1;
                ddr_rdata       <= ddr_word(longint'(lite_read_address - DDR_BASE));
            end
            if (ddr_rvalid && system_rready && !read_from_dma) begin
                ddr_rvalid <= 1'b0;
            end
        end
    end

    // the DMA's read bursts: a queue of accepted addresses, served in order
    longint burst_address[$];
    int     burst_beats[$];
    int     burst_violations = 0;
    longint bursts           = 0;
    longint mm2s_beats       = 0;
    longint beat_in_burst    = 0;

    assign mm2s_arready = !reset && burst_address.size() < 4 && lfsr[3];

    always @(posedge CLK) begin
        if (reset) begin
            mm2s_rvalid <= 1'b0;
        end
        else begin
            if (mm2s_arvalid && mm2s_arready) begin
                automatic longint first = longint'(mm2s_araddr) - longint'(DDR_BASE);
                automatic longint last  = first + 8 * (longint'(mm2s_arlen) + 1) - 1;
                if (mm2s_arburst != 2'b01 || mm2s_arsize != 3'd3 || mm2s_araddr < DDR_BASE || last >= DDR_SIZE
                    || (first >> 12) != (last >> 12) || mm2s_araddr[2:0] != 3'b000) begin
                    burst_violations++;
                    $error("DMA read burst breaks a rule: address %h, length %0d, size %0d, burst %0d",
                           mm2s_araddr, mm2s_arlen + 1, mm2s_arsize, mm2s_arburst);
                end
                burst_address.push_back(first);
                burst_beats.push_back(int'(mm2s_arlen) + 1);
                bursts++;
            end
            if (mm2s_rvalid && mm2s_rready) begin
                mm2s_rvalid <= 1'b0;
                mm2s_beats++;
                if (mm2s_rlast) begin
                    void'(burst_address.pop_front());
                    void'(burst_beats.pop_front());
                    beat_in_burst = 0;
                end
                else begin
                    beat_in_burst++;
                end
            end
            if (burst_address.size() > 0 && (!mm2s_rvalid || mm2s_rready) && lfsr[12]
                && !(mm2s_rvalid && mm2s_rready && mm2s_rlast && burst_address.size() == 1)) begin
                automatic longint offset = burst_address[0] + 8 * beat_in_burst;
                mm2s_rvalid <= 1'b1;
                mm2s_rdata  <= {ddr_word(offset + 4), ddr_word(offset)};
                mm2s_rlast  <= beat_in_burst == burst_beats[0] - 1;
            end
        end
    end

    // ------------------------------------------------------------------
    // checks on the way, as in tb_axi_replay_step
    // ------------------------------------------------------------------
    axi_rule_checker #(.PAYLOAD_WIDTH(73), .NAME("DMA stream")) check_dma (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(dma_tvalid), .ready(dma_tready),
        .payload({dma_tlast, dma_tkeep, dma_tdata}));
    axi_rule_checker #(.PAYLOAD_WIDTH(32), .NAME("system AW")) check_aw (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(system_awvalid), .ready(system_awready),
        .payload(system_awaddr));
    axi_rule_checker #(.PAYLOAD_WIDTH(32), .NAME("system AR")) check_ar (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(system_arvalid), .ready(system_arready),
        .payload(system_araddr));
    axi_rule_checker #(.PAYLOAD_WIDTH(32), .NAME("DMA read address")) check_mm2s_ar (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(mm2s_arvalid), .ready(mm2s_arready),
        .payload(mm2s_araddr));

    logic [15:0] ready_lfsr = 16'hACE1;

    always @(negedge CLK) begin
        ready_lfsr <= {ready_lfsr[14:0], ready_lfsr[15] ^ ready_lfsr[13] ^ ready_lfsr[12] ^ ready_lfsr[10]};
        tx_ready   <= ready_lfsr[0] | ready_lfsr[3];
    end

    longint pool_writes = 0, bias_writes = 0, layernorm_writes = 0, descriptor_writes = 0;
    longint error_responses = 0;

    always @(posedge CLK) begin
        if (!reset) begin
            pool_writes      += dut.accelerator.core.host_buffer_write_enable[0] === 1'b1;
            bias_writes      += dut.accelerator.core.host_bias_write_enable[0] === 1'b1;
            layernorm_writes += dut.accelerator.core.host_layernorm_write_enable === 1'b1;
            for (int word = 0; word < 5; word++) begin
                descriptor_writes += dut.accelerator.core.host_descriptor_write_enable[word] === 1'b1;
            end
            error_responses += (system_bvalid && system_bready && system_bresp != 2'b00) === 1'b1;
            error_responses += (system_rvalid && system_rready && system_rresp != 2'b00) === 1'b1;
        end
    end

    logic [7:0] host_bytes[TX_BYTES_MAX];
    logic [7:0] model_bytes[RX_BYTES_MAX];
    int received = 0, mismatches = 0, first_mismatch = -1;

    always @(posedge CLK) begin
        if (tx_valid && tx_ready) begin
            if (tx_data !== model_bytes[received]) begin
                if (first_mismatch < 0) begin
                    first_mismatch = received;
                    $display("  first mismatch at response byte %0d: got %h, the model sent %h",
                             received, tx_data, model_bytes[received]);
                end
                mismatches++;
            end
            received++;
        end
    end

    function automatic string vector_file(input string file_name);
        string name;
        int    fd;
        name = {VECTOR_DIR, file_name};
        fd = $fopen(name, "r");
        if (fd == 0) begin
            $fatal(1, "FAIL: cannot open vector file %s (run python/ddr_check.py --model --bytes 4096 --record)", name);
        end
        $fclose(fd);
        return name;
    endfunction

    initial begin
        int packets, fields, tx_count, rx_count, is_start;
        int sent = 0, expected = 0, packet_count = 0, dma_loads = 0;
        int waited;
        longint expected_pool = 0, expected_bias = 0, expected_layernorm = 0, expected_descriptor = 0;
        longint expected_dma_beats = 0;
        int violations;
        int summary, run_packets = 0;
        longint run_loads, run_beats, run_pool, run_bias, run_layernorm, run_descriptor_words, run_starts;

        $display("tb_dma_ip: the bring-up byte stream through the bridge and the real AXI DMA");
        $readmemh(vector_file("host_stream_ddrcheck_tx.hex"), host_bytes);
        $readmemh(vector_file("host_stream_ddrcheck_rx.hex"), model_bytes);
        summary = $fopen(vector_file("host_stream_ddrcheck_summary.txt"), "r");
        if ($fscanf(summary, "%d %d %d %d %d %d %d", run_loads, run_beats, run_pool, run_bias, run_layernorm,
                    run_descriptor_words, run_starts) != 7) begin
            $fatal(1, "FAIL: host_stream_ddrcheck_summary.txt does not hold seven numbers");
        end
        $fclose(summary);
        reset    = 1'b1;
        rx_valid = 1'b0;
        rx_data  = '0;
        repeat (20) @(negedge CLK);     // the DMA needs 16 clocks of reset
        reset = 1'b0;
        repeat (20) @(negedge CLK);

        packets = $fopen(vector_file("host_stream_ddrcheck_packets.txt"), "r");
        while (!$feof(packets)) begin
            fields = $fscanf(packets, "%d %d %d\n", tx_count, rx_count, is_start);
            if (fields != 3) begin
                continue;
            end
            begin
                automatic int     count  = {host_bytes[sent + 7], host_bytes[sent + 6]};
                automatic int     kind   = (host_bytes[sent + 2] >> 4) & 3;
                automatic longint length = {host_bytes[sent + 15], host_bytes[sent + 14],
                                            host_bytes[sent + 13], host_bytes[sent + 12]};
                case (host_bytes[sent + 1])
                    8'h01: expected_pool       += count;
                    8'h02: expected_bias       += count;
                    8'h03: expected_layernorm  += count;
                    8'h04: expected_descriptor += 5*count;
                    8'h0B: begin
                        dma_loads++;
                        expected_dma_beats += length / 8;
                        case (kind)
                            0: expected_pool       += length / 8;
                            1: expected_bias       += length / 32;
                            2: expected_layernorm  += length / 8;
                            default: expected_descriptor += 5 * (length / 24);
                        endcase
                    end
                    8'h0C: run_packets++;
                    default: begin
                    end
                endcase
            end
            for (int i = 0; i < tx_count; i++) begin
                rx_data  = host_bytes[sent + i];
                rx_valid = 1'b1;
                while (!rx_ready) begin
                    @(negedge CLK);
                end
                @(negedge CLK);
            end
            rx_valid  = 1'b0;
            sent     += tx_count;
            expected += rx_count;
            waited    = 0;
            while (received < expected && waited < RESPONSE_TIMEOUT) begin
                @(negedge CLK);
                waited++;
            end
            if (received < expected) begin
                $fatal(1, "FAIL: packet %0d got %0d of %0d response bytes", packet_count,
                       received - (expected - rx_count), rx_count);
            end
            packet_count++;
        end
        $fclose(packets);
        repeat (20) @(negedge CLK);
        // what the RUNs' LOAD entries moved
        expected_dma_beats  += run_beats;
        expected_pool       += run_pool;
        expected_bias       += run_bias;
        expected_layernorm  += run_layernorm;
        expected_descriptor += run_descriptor_words;

        violations = check_dma.violations + check_aw.violations + check_ar.violations + check_mm2s_ar.violations
                     + burst_violations;
        $display("  DMA: %0d DMA_LOAD packets, %0d RUN packets with %0d LOAD entries, %0d read bursts, %0d beats read, %0d of %0d stream beats; %0d bad bursts",
                 dma_loads, run_packets, run_loads, bursts, mm2s_beats, check_dma.beats, expected_dma_beats,
                 burst_violations);
        $display("  engine writes: pool %0d of %0d, bias %0d of %0d, LayerNorm %0d of %0d, descriptor words %0d of %0d",
                 pool_writes, expected_pool, bias_writes, expected_bias, layernorm_writes, expected_layernorm,
                 descriptor_writes, expected_descriptor);
        if (mismatches == 0 && received == expected && packet_count > 0 && dma_loads > 0 && run_packets > 0
            && !error_flag
            && violations == 0 && error_responses == 0 && bad_lite_addresses == 0
            && check_dma.beats == expected_dma_beats && mm2s_beats == expected_dma_beats
            && pool_writes == expected_pool && bias_writes == expected_bias
            && layernorm_writes == expected_layernorm && descriptor_writes == expected_descriptor) begin
            $display("PASS: %0d packets (%0d DMA loads and %0d run list LOADs through the AXI DMA IP), %0d of %0d response bytes equal to the model's, no rule violation, no beat lost",
                     packet_count, dma_loads, run_loads, received, expected);
        end
        else begin
            $display("FAIL: %0d of %0d response bytes differ, first mismatch at byte %0d, %0d received, %0d rule violations, %0d error responses, error flag %b",
                     mismatches, expected, first_mismatch, received, violations, error_responses, error_flag);
        end
        $finish;
    end

endmodule
