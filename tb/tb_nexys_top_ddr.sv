`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for nexys_top_ddr: the accelerator with external
// memory through the Nexys A7's pins, with a UART model on the serial lines.
// The two IPs of the top are replaced by models with their names and ports:
// system_clock_generator_model.sv (the Clocking Wizard) and
// memory_system_model.sv (the block design: MIG, AXI DMA, SmartConnect).
// tb_dma_ip checks the bridge against the real AXI DMA; this testbench checks
// the top: its resets, the calibration flag's way to the bridge and LED 4, the
// UART and the wiring of accelerator_system to the memory system.
//
// The UART model sends and receives 8N1 frames at the design's bit time
// (BAUD_RATE is set so that one bit is 8 clocks, to keep the run short). The
// testbench presses CPU_RESET (ck_rst low), checks that LED 4 is off while the
// memory is not calibrated and comes on after, then plays the byte stream of
// python/ddr_check.py --model --bytes 4096 --record over the UART: STATUS, a
// DDR write and read-back, DMA_LOADs into pool buffers read back with
// READ_BUFFER, DMA_LOADs into the bias store, the LayerNorm constants and the
// descriptor memory. Every response byte must equal the model's; no received
// byte may be lost (LED 3) and the error LED (LED 2) must stay off.
//
// Vectors: python/vectors/mnist112/host_stream_ddrcheck_tx.hex, _rx.hex,
// _packets.txt (scripts/gen_vectors.sh).
//
// sim.sh: also compile ../boards/nexys_a7/uart_rx.sv
// sim.sh: also compile ../boards/nexys_a7/uart_tx.sv
// sim.sh: also compile ../boards/nexys_a7/nexys_top_ddr.sv
// sim.sh: also compile system_clock_generator_model.sv
// sim.sh: also compile memory_system_model.sv
//////////////////////////////////////////////////////////////////////////////////

module tb_nexys_top_ddr #(
    parameter string VECTOR_DIR = "../../python/vectors/mnist112/"
);

    localparam int CLOCKS_PER_BIT   = 8;
    localparam int CLOCK_HZ         = 100_000_000;
    localparam int BIT_TIME         = 10 * CLOCKS_PER_BIT;      // ns
    localparam int TX_BYTES_MAX     = 262144;
    localparam int RX_BYTES_MAX     = 262144;
    localparam int RESPONSE_TIMEOUT = 2000000;                  // ns, on top of the response's own UART time

    logic       CLK = 1'b0;
    logic       ck_rst = 1'b0;
    logic       uart_txd_in = 1'b1;
    logic       uart_rxd_out;
    logic [4:0] led;

    nexys_top_ddr #(
        .CLOCK_HZ  (CLOCK_HZ),
        .BAUD_RATE (CLOCK_HZ / CLOCKS_PER_BIT)
    ) dut (
        .CLK100MHZ    (CLK),
        .ck_rst       (ck_rst),
        .uart_txd_in  (uart_txd_in),
        .uart_rxd_out (uart_rxd_out),
        .led          (led),
        .ddr2_addr    (),
        .ddr2_ba      (),
        .ddr2_cas_n   (),
        .ddr2_ck_n    (),
        .ddr2_ck_p    (),
        .ddr2_cke     (),
        .ddr2_cs_n    (),
        .ddr2_dm      (),
        .ddr2_dq      (),
        .ddr2_dqs_n   (),
        .ddr2_dqs_p   (),
        .ddr2_odt     (),
        .ddr2_ras_n   (),
        .ddr2_we_n    ()
    );

    always #5 CLK = ~CLK;

    logic [7:0] host_bytes[TX_BYTES_MAX];
    logic [7:0] model_bytes[RX_BYTES_MAX];

    int checks_passed = 0;
    int checks_failed = 0;

    task automatic check(input bit ok, input string what);
        if (ok) begin
            checks_passed++;
        end
        else begin
            checks_failed++;
            $display("  FAIL: %s", what);
        end
    endtask

    // UART model, receiving: every byte the board sends, against the model's
    int received       = 0;
    int mismatches     = 0;
    int first_mismatch = -1;

    initial begin
        logic [7:0] value;
        forever begin
            @(negedge uart_rxd_out);
            #(BIT_TIME / 2);
            if (uart_rxd_out === 1'b0) begin
                for (int i = 0; i < 8; i++) begin
                    #(BIT_TIME);
                    value[i] = uart_rxd_out;
                end
                #(BIT_TIME);
                if (uart_rxd_out !== 1'b1) begin
                    check(1'b0, "UART frame from the board without a stop bit");
                end
                if (value !== model_bytes[received]) begin
                    if (first_mismatch < 0) begin
                        first_mismatch = received;
                        $display("  first mismatch at response byte %0d: got %h, the model sent %h",
                                 received, value, model_bytes[received]);
                    end
                    mismatches++;
                end
                received++;
            end
        end
    end

    // UART model, sending: start bit, 8 data bits least significant first, stop bit
    task automatic send_byte(input logic [7:0] value);
        uart_txd_in = 1'b0;
        #(BIT_TIME);
        for (int i = 0; i < 8; i++) begin
            uart_txd_in = value[i];
            #(BIT_TIME);
        end
        uart_txd_in = 1'b1;
        #(BIT_TIME);
    endtask

    function automatic string vector_file(input string file_name);
        string name;
        int    fd;
        name = {VECTOR_DIR, file_name};
        fd = $fopen(name, "r");
        if (fd == 0) begin
            $fatal(1, "FAIL: cannot open vector file %s (run python/ddr_check.py --model --bytes 4096 --record)",
                   name);
        end
        $fclose(fd);
        return name;
    endfunction

    initial begin
        int packets, fields, tx_count, rx_count, is_start;
        int sent = 0, expected = 0, packet_count = 0;
        int waited;

        $display("tb_nexys_top_ddr: nexys_top_ddr through its pins, with the memory system modelled");
        $readmemh(vector_file("host_stream_ddrcheck_tx.hex"), host_bytes);
        $readmemh(vector_file("host_stream_ddrcheck_rx.hex"), model_bytes);

        // CPU_RESET pressed, then released
        repeat (100) @(negedge CLK);
        check(led[4] === 1'b0, "LED 4 (DDR calibrated) is off while CPU_RESET is pressed");
        ck_rst = 1'b1;
        repeat (100) @(negedge CLK);
        check(led[4] === 1'b0, "LED 4 is off before the memory has calibrated");
        waited = 0;
        while (led[4] !== 1'b1 && waited < 100000) begin
            @(negedge CLK);
            waited++;
        end
        check(led[4] === 1'b1, "LED 4 comes on once the memory has calibrated");
        check(led[2:0] === 3'b000, "busy, done and error LEDs are off after reset");

        packets = $fopen(vector_file("host_stream_ddrcheck_packets.txt"), "r");
        while (!$feof(packets)) begin
            fields = $fscanf(packets, "%d %d %d\n", tx_count, rx_count, is_start);
            if (fields != 3) begin
                continue;
            end
            for (int i = 0; i < tx_count; i++) begin
                send_byte(host_bytes[sent + i]);
            end
            sent     += tx_count;
            expected += rx_count;
            waited    = 0;
            // a response takes 10 bit times per byte on the line, plus the board's own time
            while (received < expected && waited < RESPONSE_TIMEOUT + 10 * BIT_TIME * rx_count) begin
                #(BIT_TIME);
                waited += BIT_TIME;
            end
            check(received >= expected, $sformatf("packet %0d is answered (%0d of %0d response bytes)",
                                                 packet_count, received - (expected - rx_count), rx_count));
            packet_count++;
        end
        $fclose(packets);
        repeat (200) @(negedge CLK);

        check(received == expected && mismatches == 0,
              $sformatf("%0d of %0d response bytes equal to the model's, first mismatch at byte %0d",
                        expected - mismatches, expected, first_mismatch));
        check(led[3] === 1'b0, "no received byte was lost (LED 3 off)");
        check(led[2] === 1'b0, "the error LED is off");
        check(led[4] === 1'b1, "LED 4 stays on");
        check(dut.memory.transfers > 0, "the DMA moved data");

        $display("  %0d packets, %0d bytes sent over the UART, %0d response bytes, %0d DMA transfers",
                 packet_count, sent, received, dut.memory.transfers);
        if (checks_failed == 0 && packet_count > 0) begin
            $display("PASS: %0d of %0d checks, %0d of %0d response bytes equal to the model's through the UART pins",
                     checks_passed, checks_passed, received, expected);
        end
        else begin
            $display("FAIL: %0d of %0d checks failed, first mismatch at response byte %0d",
                     checks_failed, checks_passed + checks_failed, first_mismatch);
        end
        $finish;
    end

endmodule
