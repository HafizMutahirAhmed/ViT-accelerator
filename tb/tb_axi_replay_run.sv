`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench of the RUN command: accelerator_system with
// external memory (the bridge, the stream multiplexer, the AXI wrapper and the
// engine), with memory_system_model.sv standing in for the MIG, the AXI DMA
// and the SmartConnect, against the byte stream of the real host script.
//
// python/run_image.py --model --digit 0 --golden --clear --record ran
// the whole 112 x 112 MNIST program against the software model of a board with
// DDR: it uploaded the weights, biases, LayerNorm constants, descriptors and
// the run list to DDR with WRITE_DDR (after a READ_DDR of the signature),
// cleared the pool buffers, wrote the image to DDR and sent one RUN; the
// bridge then fetched every entry of the list from DDR and ran it: a DMA load
// per LOAD, a start and a wait for done per START. The RUN response carries
// the total busy cycles, which must equal the model's (python/board_model.py),
// like every other response byte. Then the scores are read back.
//
// No beat may be lost or doubled: the engine must see exactly one write per
// item the packets carry plus what the run list loads
// (host_stream_run_summary.txt: LOAD entries, DMA beats, pool, bias and
// LayerNorm items, descriptor words, START entries, over all RUNs), the DMA must
// have sent exactly those beats in that many transfers, the engine must be
// started once per START entry, and the store stream must deliver exactly the
// READ_BUFFER addresses. An
// axi_rule_checker watches the five AXI4-Lite channels of each master and the
// three streams. tx_ready toggles pseudo-randomly; the memory model adds its
// own random delays and stream gaps.
//
// Vectors: python/models/nexys_L2/host_stream_run_tx.hex, host_stream_run_rx.hex,
// host_stream_run_packets.txt and host_stream_run_summary.txt. 8x8 only.
//
// also compile: axi_rule_checker.sv
// also compile: memory_system_model.sv
//////////////////////////////////////////////////////////////////////////////////

`include "vit_paths.svh"

module tb_axi_replay_run #(
    parameter string VECTOR_DIR = {`VIT_ROOT, "python/models/nexys_L2/"}
);

    localparam int TX_BYTES_MAX     = 16777216;     // a 12-layer model uploads about 6.3 MB
    localparam int RX_BYTES_MAX     = 1048576;
    localparam int RESPONSE_TIMEOUT = 100000000;    // clocks; a RUN answers after the whole program

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
    logic       memory_ready;
    logic       init_calib_complete;

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
        .GELU_ROM_FILE  (`VIT_ROM_112("gelu_rom.hex")),
        .EXP_ROM_FILE   (`VIT_ROM_112("exp_rom.hex")),
        .RSQRT_ROM_FILE (`VIT_ROM_112("rsqrt_rom.hex"))
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
        .memory_ready         (memory_ready),
        .busy                 (busy),
        .done_flag            (done_flag),
        .error_flag           (error_flag)
    );

    memory_system_wrapper memory (
        .DDR2_addr           (),
        .DDR2_ba             (),
        .DDR2_cas_n          (),
        .DDR2_ck_n           (),
        .DDR2_ck_p           (),
        .DDR2_cke            (),
        .DDR2_cs_n           (),
        .DDR2_dm             (),
        .DDR2_dq             (),
        .DDR2_dqs_n          (),
        .DDR2_dqs_p          (),
        .DDR2_odt            (),
        .DDR2_ras_n          (),
        .DDR2_we_n           (),
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
        .clk_200             (1'b0),
        .init_calib_complete (init_calib_complete),
        .memory_aresetn      (1'b1),
        .memory_resetn       (!reset),
        .system_aresetn      (!reset),
        .ui_clk              (),
        .ui_clk_sync_rst     ()
    );

    assign memory_ready = init_calib_complete;

    // the AXI rules on every channel of the system master, and on the streams
    axi_rule_checker #(.PAYLOAD_WIDTH(32), .NAME("system AW")) check_aw (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(system_awvalid), .ready(system_awready),
        .payload(system_awaddr));
    axi_rule_checker #(.PAYLOAD_WIDTH(36), .NAME("system W")) check_w (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(system_wvalid), .ready(system_wready),
        .payload({system_wstrb, system_wdata}));
    axi_rule_checker #(.PAYLOAD_WIDTH(2), .NAME("system B")) check_b (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(system_bvalid), .ready(system_bready),
        .payload(system_bresp));
    axi_rule_checker #(.PAYLOAD_WIDTH(32), .NAME("system AR")) check_ar (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(system_arvalid), .ready(system_arready),
        .payload(system_araddr));
    axi_rule_checker #(.PAYLOAD_WIDTH(34), .NAME("system R")) check_r (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(system_rvalid), .ready(system_rready),
        .payload({system_rresp, system_rdata}));
    axi_rule_checker #(.PAYLOAD_WIDTH(73), .NAME("DMA stream")) check_dma (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(dma_tvalid), .ready(dma_tready),
        .payload({dma_tlast, dma_tkeep, dma_tdata}));
    axi_rule_checker #(.PAYLOAD_WIDTH(73), .NAME("load stream")) check_load (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(dut.load_tvalid), .ready(dut.load_tready),
        .payload({dut.load_tlast, dut.load_tkeep, dut.load_tdata}));
    axi_rule_checker #(.PAYLOAD_WIDTH(73), .NAME("store stream")) check_store (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(dut.store_tvalid), .ready(dut.store_tready),
        .payload({dut.store_tlast, dut.store_tkeep, dut.store_tdata}));
    // and on the wrapper's AXI4-Lite port, through the bridge's other master
    axi_rule_checker #(.PAYLOAD_WIDTH(12), .NAME("wrapper AW")) check_wrapper_aw (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(dut.axi_awvalid), .ready(dut.axi_awready),
        .payload(dut.axi_awaddr));
    axi_rule_checker #(.PAYLOAD_WIDTH(12), .NAME("wrapper AR")) check_wrapper_ar (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(dut.axi_arvalid), .ready(dut.axi_arready),
        .payload(dut.axi_araddr));

    // back-pressure on the response bytes, and through the bridge on the store stream
    logic [15:0] ready_lfsr = 16'hACE1;

    always @(negedge CLK) begin
        ready_lfsr <= {ready_lfsr[14:0], ready_lfsr[15] ^ ready_lfsr[13] ^ ready_lfsr[12] ^ ready_lfsr[10]};
        tx_ready   <= ready_lfsr[0] | ready_lfsr[3];
    end

    // what reaches the engine and what leaves it
    longint pool_writes       = 0;
    longint bias_writes       = 0;
    longint layernorm_writes  = 0;
    longint descriptor_writes = 0;
    longint store_lasts       = 0;
    longint slave_errors      = 0;
    longint wrong_stream      = 0;     // a beat of the other source while one is selected
    longint engine_starts     = 0;

    always @(posedge CLK) begin
        if (!reset) begin
            pool_writes      += dut.accelerator.core.host_buffer_write_enable[0] === 1'b1;
            bias_writes      += dut.accelerator.core.host_bias_write_enable[0] === 1'b1;
            layernorm_writes += dut.accelerator.core.host_layernorm_write_enable === 1'b1;
            for (int word = 0; word < 5; word++) begin
                descriptor_writes += dut.accelerator.core.host_descriptor_write_enable[word] === 1'b1;
            end
            store_lasts  += (dut.store_tvalid && dut.store_tready && dut.store_tlast) === 1'b1;
            slave_errors += (system_bvalid && system_bready && system_bresp != 2'b00) === 1'b1;
            slave_errors += (system_rvalid && system_rready && system_rresp != 2'b00) === 1'b1;
            wrong_stream += (dut.dma_stream_select ? dut.bridge_load_tvalid : dma_tvalid) === 1'b1;
            engine_starts += dut.accelerator.core.start === 1'b1;
        end
    end

    always #5 CLK = ~CLK;

    logic [7:0] host_bytes[TX_BYTES_MAX];
    logic [7:0] model_bytes[RX_BYTES_MAX];

    int received       = 0;
    int mismatches     = 0;
    int first_mismatch = -1;

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
            $fatal(1, "FAIL: cannot open vector file %s (run python/run_image.py --model --digit 0 --golden --clear --record)",
                   name);
        end
        $fclose(fd);
        return name;
    endfunction

    initial begin
        int packets;
        int fields;
        int tx_count, rx_count, is_start;
        int sent         = 0;
        int expected     = 0;
        int packet_count = 0;
        int starts       = 0;
        int dma_loads    = 0;
        int ddr_writes   = 0;
        int run_packets  = 0;
        int summary;
        longint run_loads, run_beats, run_pool, run_bias, run_layernorm, run_descriptor_words, run_starts;
        int waited;
        longint expected_pool = 0, expected_bias = 0, expected_layernorm = 0, expected_descriptor = 0;
        longint expected_store = 0, expected_reads = 0, expected_dma_beats = 0;
        int     violations;

        $display("tb_axi_replay_run: the host script's RUN byte stream through bridge, DMA model, AXI wrapper and engine");
        $readmemh(vector_file("host_stream_run_tx.hex"), host_bytes);
        $readmemh(vector_file("host_stream_run_rx.hex"), model_bytes);
        summary = $fopen(vector_file("host_stream_run_summary.txt"), "r");
        if ($fscanf(summary, "%d %d %d %d %d %d %d", run_loads, run_beats, run_pool, run_bias, run_layernorm,
                    run_descriptor_words, run_starts) != 7) begin
            $fatal(1, "FAIL: host_stream_run_summary.txt does not hold seven numbers");
        end
        $fclose(summary);
        reset    = 1'b1;
        rx_valid = 1'b0;
        rx_data  = '0;
        repeat (3) @(negedge CLK);
        reset = 1'b0;
        while (!memory_ready) begin
            @(negedge CLK);
        end

        packets = $fopen(vector_file("host_stream_run_packets.txt"), "r");
        while (!$feof(packets)) begin
            fields = $fscanf(packets, "%d %d %d\n", tx_count, rx_count, is_start);
            if (fields != 3) begin
                continue;
            end
            // what this packet asks for: command in byte 1, target in 2, count in 6 and 7
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
                    8'h06: begin
                        expected_store += count;
                        expected_reads += count != 0;
                    end
                    8'h09: ddr_writes++;
                    8'h0C: run_packets++;
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
            if (is_start != 0) begin
                starts++;
                @(negedge CLK);
                while (busy) begin
                    @(negedge CLK);
                end
                @(negedge CLK);
            end
            packet_count++;
        end
        $fclose(packets);
        repeat (20) @(negedge CLK);
        expected_dma_beats  += run_beats;
        expected_pool       += run_pool;
        expected_bias       += run_bias;
        expected_layernorm  += run_layernorm;
        expected_descriptor += run_descriptor_words;

        violations = check_aw.violations + check_w.violations + check_b.violations + check_ar.violations
                     + check_r.violations + check_dma.violations + check_load.violations
                     + check_store.violations + check_wrapper_aw.violations + check_wrapper_ar.violations;
        $display("  system AXI4-Lite: %0d writes, %0d reads, %0d error responses; %0d WRITE_DDR, %0d DMA_LOAD and %0d RUN packets",
                 check_aw.beats, check_ar.beats, slave_errors, ddr_writes, dma_loads, run_packets);
        $display("  run list: %0d LOAD and %0d START entries run; engine started %0d times",
                 run_loads, run_starts, engine_starts);
        $display("  DMA: %0d transfers, %0d beats of %0d; load stream %0d beats; store stream %0d beats, %0d TLAST",
                 memory.transfers, check_dma.beats, expected_dma_beats, check_load.beats, check_store.beats,
                 store_lasts);
        $display("  engine writes: pool %0d of %0d, bias %0d of %0d, LayerNorm %0d of %0d, descriptor words %0d of %0d; stored %0d of %0d addresses",
                 pool_writes, expected_pool, bias_writes, expected_bias, layernorm_writes, expected_layernorm,
                 descriptor_writes, expected_descriptor, check_store.beats, expected_store);
        if (mismatches == 0 && received == expected && packet_count > 0 && !error_flag
            && violations == 0 && slave_errors == 0 && wrong_stream == 0 && run_packets > 0
            && memory.transfers == dma_loads + run_loads && check_dma.beats == expected_dma_beats
            && engine_starts == starts + run_starts
            && pool_writes == expected_pool && bias_writes == expected_bias
            && layernorm_writes == expected_layernorm && descriptor_writes == expected_descriptor
            && check_store.beats == expected_store && store_lasts == expected_reads) begin
            $display("PASS: %0d packets (%0d RUN: %0d phases started, %0d DMA loads by the run list), %0d bytes sent, %0d of %0d response bytes equal to the model's, no AXI rule violation, no beat lost",
                     packet_count, run_packets, engine_starts, memory.transfers, sent, received, expected);
        end
        else begin
            $display("FAIL: %0d of %0d response bytes differ, first mismatch at byte %0d, %0d received, %0d AXI rule violations, %0d error responses, %0d beats on the unselected stream, error flag %b, %0d of %0d DMA transfers, %0d of %0d engine starts",
                     mismatches, expected, first_mismatch, received, violations, slave_errors, wrong_stream,
                     error_flag, memory.transfers, dma_loads + run_loads, engine_starts, starts + run_starts);
        end
        $finish;
    end

endmodule
