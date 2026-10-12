`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench that plays the byte stream of the real host script
// to the RTL (vit_accelerator: host_link and matmul_engine).
//
// host/run_image.py --model --digit 0 --golden --clear --record ran the whole
// 112 x 112 MNIST program against the software model of the board and wrote
// down every byte it sent and every byte the model answered. Here the same
// packets go to the RTL, one byte per cycle, and every response must equal
// the model's byte for byte: the status, the stage tensors and class scores
// read back with READ_BUFFER, the busy cycle counts in STATUS, the checksums.
// After a START packet the testbench waits until the engine is idle before
// it sends the next packet, as the model is idle at once.
//
// So this checks the host software's packets against the hardware's protocol
// implementation, and the software model against the hardware.
//
// Vectors: golden/vectors/mnist112/host_stream_tx.hex, host_stream_rx.hex
// (one byte per line) and host_stream_packets.txt (bytes sent, bytes
// received, 1 for a START, per packet). 8x8 only.
//////////////////////////////////////////////////////////////////////////////////

module tb_host_replay #(
    parameter string VECTOR_DIR = "../golden/vectors/mnist112/"
);

    localparam int BUFFER_COUNT = vit_config_pkg::BUFFER_COUNT;
    // the 112 x 112 MNIST plan: words per bank of buffers 7 .. 0
    localparam bit [BUFFER_COUNT-1:0][31:0] BUFFER_DEPTH
        = {32'd16, 32'd1344, 32'd2544, 32'd1050, 32'd1344, 32'd5376, 32'd16384, 32'd1344};

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
    logic       busy;
    logic       done_flag;
    logic       error_flag;

    vit_accelerator #(
        .SYS_ROW        (8),
        .SYS_COL        (8),
        .BUFFER_DEPTH   (BUFFER_DEPTH),
        .GELU_ROM_FILE  ("rom/mnist112/gelu_rom.hex"),
        .EXP_ROM_FILE   ("rom/mnist112/exp_rom.hex"),
        .RSQRT_ROM_FILE ("rom/mnist112/rsqrt_rom.hex")
    ) dut (
        .CLK        (CLK),
        .reset      (reset),
        .rx_data    (rx_data),
        .rx_valid   (rx_valid),
        .rx_ready   (rx_ready),
        .tx_data    (tx_data),
        .tx_valid   (tx_valid),
        .tx_ready   (1'b1),
        .busy       (busy),
        .done_flag  (done_flag),
        .error_flag (error_flag)
    );

    always #5 CLK = ~CLK;

    logic [7:0] host_bytes[TX_BYTES_MAX];
    logic [7:0] model_bytes[RX_BYTES_MAX];

    int received       = 0;     // bytes the DUT has sent so far
    int mismatches     = 0;
    int first_mismatch = -1;

    // every byte the DUT sends, against the model's
    always @(posedge CLK) begin
        if (tx_valid) begin
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
            $fatal(1, "FAIL: cannot open vector file %s (run host/run_image.py --model --digit 0 --golden --clear --record)",
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

        $display("tb_host_replay: the host script's byte stream against the RTL");
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

        if (mismatches == 0 && received == expected && packet_count > 0 && !error_flag) begin
            $display("PASS: %0d packets (%0d phases started), %0d bytes sent, %0d of %0d response bytes equal to the model's",
                     packet_count, starts, sent, received, expected);
        end
        else begin
            $display("FAIL: %0d of %0d response bytes differ, first mismatch at byte %0d, %0d received, error flag %b",
                     mismatches, expected, first_mismatch, received, error_flag);
        end
        $finish;
    end

endmodule
