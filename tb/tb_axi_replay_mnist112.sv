`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench of the AXI path: axi_host_bridge, vit_axi_wrapper
// and the engine, against the byte stream of the real host script.
//
// python/run_image.py --model --digit 0 --golden --clear --record ran
// the whole 112 x 112 MNIST program against the software model of the board
// and wrote down every byte it sent and every byte the model answered. The
// same packets go to the bridge here, and every response byte must equal the
// model's, so the AXI path is byte for byte the protocol the host script
// expects. After a START packet the
// testbench waits until the engine is idle before it sends the next packet.
//
// AXI rules. An axi_rule_checker on each of the five AXI4-Lite channels and
// on both streams checks that VALID and the payload hold until READY.
// tx_ready is toggled pseudo-randomly, so the store stream is back-pressured
// through the bridge. No beat may be lost: the engine must see exactly one
// write per item the packets carry (pool, bias, LayerNorm, descriptor words),
// and the store stream must deliver exactly the addresses the READ_BUFFER
// packets ask for, with one TLAST per packet.
//
// Vectors: python/vectors/mnist112/host_stream_tx.hex, host_stream_rx.hex
// (one byte per line) and host_stream_packets.txt (bytes sent, bytes
// received, 1 for a START, per packet). 8x8 only.
//
// sim.sh: also compile axi_rule_checker.sv
//////////////////////////////////////////////////////////////////////////////////

module tb_axi_replay_mnist112 #(
    parameter string VECTOR_DIR = "../../python/vectors/mnist112/"
);

    localparam int TX_BYTES_MAX     = 2097152;
    localparam int RX_BYTES_MAX     = 1048576;
    localparam int RESPONSE_TIMEOUT = 4000000;

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
    logic       bridge_error;
    logic       irq;

    logic [11:0] axi_awaddr, axi_araddr;
    logic        axi_awvalid, axi_awready, axi_wvalid, axi_wready, axi_bvalid, axi_bready;
    logic        axi_arvalid, axi_arready, axi_rvalid, axi_rready;
    logic [31:0] axi_wdata, axi_rdata;
    logic [3:0]  axi_wstrb;
    logic [1:0]  axi_bresp, axi_rresp;
    logic [63:0] load_tdata, store_tdata;
    logic [7:0]  load_tkeep, store_tkeep;
    logic        load_tlast, load_tvalid, load_tready, store_tlast, store_tvalid, store_tready;

    axi_host_bridge bridge (
        .aclk          (CLK),
        .aresetn       (!reset),
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

    // the 112 x 112 MNIST plan: words per bank of buffers 0 .. 7
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
    ) dut (
        .aclk          (CLK),
        .aresetn       (!reset),
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
        .irq           (irq),
        .busy          (busy),
        .done_flag     (done_flag),
        .error_flag    (error_flag)
    );

    // the AXI rules on every channel
    axi_rule_checker #(.PAYLOAD_WIDTH(12), .NAME("AW")) check_aw (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(axi_awvalid), .ready(axi_awready), .payload(axi_awaddr));
    axi_rule_checker #(.PAYLOAD_WIDTH(36), .NAME("W")) check_w (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(axi_wvalid), .ready(axi_wready),
        .payload({axi_wstrb, axi_wdata}));
    axi_rule_checker #(.PAYLOAD_WIDTH(2), .NAME("B")) check_b (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(axi_bvalid), .ready(axi_bready), .payload(axi_bresp));
    axi_rule_checker #(.PAYLOAD_WIDTH(12), .NAME("AR")) check_ar (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(axi_arvalid), .ready(axi_arready), .payload(axi_araddr));
    axi_rule_checker #(.PAYLOAD_WIDTH(34), .NAME("R")) check_r (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(axi_rvalid), .ready(axi_rready),
        .payload({axi_rresp, axi_rdata}));
    axi_rule_checker #(.PAYLOAD_WIDTH(73), .NAME("load stream")) check_load (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(load_tvalid), .ready(load_tready),
        .payload({load_tlast, load_tkeep, load_tdata}));
    axi_rule_checker #(.PAYLOAD_WIDTH(73), .NAME("store stream")) check_store (
        .CLK(CLK), .reset(reset), .flush(1'b0), .valid(store_tvalid), .ready(store_tready),
        .payload({store_tlast, store_tkeep, store_tdata}));

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

    always @(posedge CLK) begin
        if (!reset) begin
            pool_writes      += dut.core.host_buffer_write_enable[0] === 1'b1;
            bias_writes      += dut.core.host_bias_write_enable[0] === 1'b1;
            layernorm_writes += dut.core.host_layernorm_write_enable === 1'b1;
            for (int word = 0; word < 5; word++) begin
                descriptor_writes += dut.core.host_descriptor_write_enable[word] === 1'b1;
            end
            store_lasts      += (store_tvalid && store_tready && store_tlast) === 1'b1;
            slave_errors     += (axi_bvalid && axi_bready && axi_bresp != 2'b00) === 1'b1;
        end
    end

    always #5 CLK = ~CLK;

    logic [7:0] host_bytes[TX_BYTES_MAX];
    logic [7:0] model_bytes[RX_BYTES_MAX];

    int received       = 0;     // bytes the DUT has sent so far
    int mismatches     = 0;
    int first_mismatch = -1;

    // every byte the DUT sends, against the model's
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
        int sent      = 0;
        int expected  = 0;
        int packet_count = 0;
        int starts    = 0;
        int waited;
        longint expected_pool = 0, expected_bias = 0, expected_layernorm = 0, expected_descriptor = 0;
        longint expected_store = 0, expected_reads = 0;
        int     violations;

        $display("tb_axi_replay_mnist112: the host script's byte stream through bridge, AXI wrapper and engine");
        $readmemh(vector_file("host_stream_tx.hex"), host_bytes);
        $readmemh(vector_file("host_stream_rx.hex"), model_bytes);
        reset    = 1'b1;
        rx_valid = 1'b0;
        rx_data  = '0;
        repeat (3) @(negedge CLK);
        reset = 1'b0;

        packets = $fopen(vector_file("host_stream_packets.txt"), "r");
        while (!$feof(packets)) begin
            fields = $fscanf(packets, "%d %d %d\n", tx_count, rx_count, is_start);
            if (fields != 3) begin
                continue;
            end
            // what this packet asks for: command in byte 1, count in bytes 6 and 7
            begin
                automatic int count = {host_bytes[sent + 7], host_bytes[sent + 6]};
                case (host_bytes[sent + 1])
                    8'h01: expected_pool       += count;
                    8'h02: expected_bias       += count;
                    8'h03: expected_layernorm  += count;
                    8'h04: expected_descriptor += 5*count;
                    8'h06: begin
                        expected_store += count;
                        expected_reads += count != 0;
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
                // the model finishes a phase at once; here the engine has to run
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

        violations = check_aw.violations + check_w.violations + check_b.violations + check_ar.violations
                     + check_r.violations + check_load.violations + check_store.violations;
        $display("  AXI4-Lite: %0d writes, %0d reads, %0d SLVERR; load stream %0d beats; store stream %0d beats, %0d TLAST",
                 check_aw.beats, check_ar.beats, slave_errors, check_load.beats, check_store.beats, store_lasts);
        $display("  engine writes: pool %0d of %0d, bias %0d of %0d, LayerNorm %0d of %0d, descriptor words %0d of %0d; stored %0d of %0d addresses",
                 pool_writes, expected_pool, bias_writes, expected_bias, layernorm_writes, expected_layernorm,
                 descriptor_writes, expected_descriptor, check_store.beats, expected_store);
        if (mismatches == 0 && received == expected && packet_count > 0 && !bridge_error && !error_flag
            && violations == 0 && slave_errors == 0
            && pool_writes == expected_pool && bias_writes == expected_bias
            && layernorm_writes == expected_layernorm && descriptor_writes == expected_descriptor
            && check_store.beats == expected_store && store_lasts == expected_reads) begin
            $display("PASS: %0d packets (%0d phases started), %0d bytes sent, %0d of %0d response bytes equal to the model's, no AXI rule violation, no beat lost",
                     packet_count, starts, sent, received, expected);
        end
        else begin
            $display("FAIL: %0d of %0d response bytes differ, first mismatch at byte %0d, %0d received, %0d AXI rule violations, error flags %b %b",
                     mismatches, expected, first_mismatch, received, violations, bridge_error, error_flag);
        end
        $finish;
    end

endmodule
