`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for arty_top: the accelerator through the Arty's
// pins, with a UART model on the serial lines.
//
// The UART model sends and receives 8N1 frames at the bit time of the design
// (BAUD_RATE is set so that one bit is 8 clocks, to keep the run short; the
// design's own UART modules are the ones under test). On top of it the
// testbench plays the host of docs/HOST_PROTOCOL.md for the last phase of the
// 112 x 112 MNIST program: it clears the buffers that phase uses, writes the
// descriptor table, writes the golden layer-2 output into the residual
// stream buffer, loads the phase's weights, biases and LayerNorm constants,
// starts it, polls STATUS and reads back the final LayerNorm output and both
// classifiers' scores, which must equal the golden tensors. It also checks
// that a packet with a wrong checksum is answered with the checksum status
// and lights the error LED until CLEAR, and that the busy and done LEDs
// follow the run.
//
// The whole program through the byte stream is tb_host_link_mnist112; over the UART
// model it would take about 86 million clocks.
//
// Vectors: python/gen_mnist_forward.py, in python/vectors/mnist112. Reset is
// the board's RESET button, ck_rst, pressed (low) at the start.
//
// sim.sh: also compile ../boards/arty_a7/uart_rx.sv
// sim.sh: also compile ../boards/arty_a7/uart_tx.sv
// sim.sh: also compile ../boards/arty_a7/arty_top.sv
//////////////////////////////////////////////////////////////////////////////////

module tb_arty_top_mnist112 #(
    parameter string VECTOR_DIR = "../../python/vectors/mnist112/"
);

    localparam int CLOCKS_PER_BIT = 8;
    localparam int CLOCK_HZ       = 100_000_000;
    localparam int BIT_TIME       = 10 * CLOCKS_PER_BIT;      // ns

    localparam int LANES            = 8;
    localparam int DATA_WIDTH       = vit_config_pkg::DATA_WIDTH;
    localparam int BUFFER_COUNT     = vit_config_pkg::BUFFER_COUNT;
    localparam int DESCRIPTOR_WORDS = vit_config_pkg::DESCRIPTOR_WORDS;
    localparam int DESCRIPTOR_COUNT = vit_config_pkg::DESCRIPTOR_COUNT;

    localparam int IMAGE_WORDS_MAX  = 65536;
    localparam int VECTOR_WORDS_MAX = 262144;
    localparam int PHASES_MAX       = 64;
    localparam int NAME_BITS        = 8 * 40;
    localparam int RESPONSE_TIMEOUT = 4000000;

    localparam logic [7:0] COMMAND_WRITE_BUFFER     = 8'h01;
    localparam logic [7:0] COMMAND_WRITE_BIAS       = 8'h02;
    localparam logic [7:0] COMMAND_WRITE_LAYERNORM  = 8'h03;
    localparam logic [7:0] COMMAND_WRITE_DESCRIPTOR = 8'h04;
    localparam logic [7:0] COMMAND_START            = 8'h05;
    localparam logic [7:0] COMMAND_READ_BUFFER      = 8'h06;
    localparam logic [7:0] COMMAND_STATUS           = 8'h07;
    localparam logic [7:0] COMMAND_CLEAR            = 8'h08;

    localparam int STATUS_CHECKSUM = 1;
    localparam int STATUS_COMMAND  = 2;
    localparam int STATUS_BUSY     = 4;

    logic       CLK = 1'b0;
    logic       ck_rst;
    logic       uart_txd_in = 1'b1;
    logic       uart_rxd_out;
    logic [3:0] led;

    arty_top #(
        .CLOCK_HZ  (CLOCK_HZ),
        .BAUD_RATE (CLOCK_HZ / CLOCKS_PER_BIT)
    ) dut (
        .CLK100MHZ    (CLK),
        .ck_rst       (ck_rst),
        .uart_txd_in  (uart_txd_in),
        .uart_rxd_out (uart_rxd_out),
        .led          (led)
    );

    always #5 CLK = ~CLK;

    logic [31:0]                 descriptor_mem[DESCRIPTOR_WORDS*DESCRIPTOR_COUNT];
    logic [LANES*DATA_WIDTH-1:0] image_mem[IMAGE_WORDS_MAX];
    logic [LANES*32-1:0]         bias_image_mem[1024];
    logic [47:0]                 layernorm_image_mem[1024];
    logic [DATA_WIDTH-1:0]       expected_mem[VECTOR_WORDS_MAX];

    // names are kept as character vectors: xelab 2022.2 crashes on arrays of string
    logic [NAME_BITS-1:0] op_phase[DESCRIPTOR_COUNT];
    logic [NAME_BITS-1:0] op_name[DESCRIPTOR_COUNT];
    logic [NAME_BITS-1:0] op_kind[DESCRIPTOR_COUNT];
    logic [NAME_BITS-1:0] op_file[DESCRIPTOR_COUNT];
    int    op_buffer[DESCRIPTOR_COUNT];
    bit    op_row_banks[DESCRIPTOR_COUNT];
    int    op_base[DESCRIPTOR_COUNT];
    int    op_stride[DESCRIPTOR_COUNT];
    int    op_rows[DESCRIPTOR_COUNT];
    int    op_cols[DESCRIPTOR_COUNT];
    int    op_count = 0;

    logic [NAME_BITS-1:0] phase_name[PHASES_MAX];
    int    phase_first[PHASES_MAX];
    int    phase_ops[PHASES_MAX];
    int    phase_count = 0;

    byte unsigned payload[];          // payload of the packet being sent
    byte unsigned received[];         // payload of the last response

    int checks_passed   = 0;
    int checks_failed   = 0;
    int entries_ok      = 0;
    int entries_wrong   = 0;
    longint bytes_sent  = 0;

    // UART model, receiving: every byte the board sends
    logic [7:0] response[$];

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
                    $display("  FAIL: UART frame from the board without a stop bit");
                    checks_failed++;
                end
                response.push_back(value);
            end
        end
    end

    function automatic string text(input logic [NAME_BITS-1:0] characters);
        return $sformatf("%0s", characters);
    endfunction

    function automatic string vector_file(input string file_name);
        string name;
        int    fd;
        name = {VECTOR_DIR, file_name};
        fd = $fopen(name, "r");
        if (fd == 0) begin
            $fatal(1, "FAIL: cannot open golden vector file %s", name);
        end
        $fclose(fd);
        return name;
    endfunction

    // CRC-16/CCITT-FALSE, one byte
    function automatic logic [15:0] crc_next(input logic [15:0] crc, input logic [7:0] data);
        logic [15:0] value;
        value = crc ^ {data, 8'h00};
        for (int i = 0; i < 8; i++) begin
            value = value[15] ? (value << 1) ^ 16'h1021 : value << 1;
        end
        return value;
    endfunction

    task automatic check(input bit ok, input string what);
        if (ok) begin
            checks_passed++;
        end
        else begin
            checks_failed++;
            $display("  FAIL: %s", what);
        end
    endtask

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
        bytes_sent++;
    endtask

    // one command packet; payload[] holds payload_bytes bytes. corrupt flips the checksum.
    task automatic send_packet(input logic [7:0] command, input int target, input int address, input int count,
                               input int payload_bytes, input bit corrupt = 1'b0);
        logic [15:0] crc = 16'hFFFF;
        logic [7:0]  header[7];
        header[0] = command;
        header[1] = 8'(target);
        header[2] = address[7:0];
        header[3] = address[15:8];
        header[4] = address[23:16];
        header[5] = count[7:0];
        header[6] = count[15:8];
        send_byte(8'hA5);
        for (int i = 0; i < 7; i++) begin
            crc = crc_next(crc, header[i]);
            send_byte(header[i]);
        end
        for (int i = 0; i < payload_bytes; i++) begin
            crc = crc_next(crc, payload[i]);
            send_byte(payload[i]);
        end
        if (corrupt) begin
            crc = ~crc;
        end
        send_byte(crc[7:0]);
        send_byte(crc[15:8]);
    endtask

    // waits for a whole response with payload_bytes of payload; checks its framing and status
    task automatic get_response(input int payload_bytes, input int expected_status, input string what);
        int          waited = 0;
        int          total;
        logic [15:0] crc    = 16'hFFFF;
        logic [7:0]  status;

        // an error response has no payload
        while (response.size() < 2 && waited < RESPONSE_TIMEOUT) begin
            @(negedge CLK);
            waited++;
        end
        if (response.size() < 2) begin
            $fatal(1, "FAIL: no response to %s", what);
        end
        status = response[1];
        total  = 2 + ((status == 8'd0) ? payload_bytes : 0) + 2;
        while (response.size() < total && waited < RESPONSE_TIMEOUT) begin
            @(negedge CLK);
            waited++;
        end
        if (response.size() < total) begin
            $fatal(1, "FAIL: response to %s stopped after %0d of %0d bytes", what, response.size(), total);
        end
        for (int i = 1; i < total - 2; i++) begin
            crc = crc_next(crc, response[i]);
        end
        check(response[0] == 8'h5A, {what, ": response does not start with 0x5A"});
        check(int'(status) == expected_status, $sformatf("%s: status %0d, expected %0d", what, status, expected_status));
        check({response[total-1], response[total-2]} == crc, {what, ": response checksum is wrong"});
        received = new[total - 4];
        for (int i = 0; i < total - 4; i++) begin
            received[i] = response[2 + i];
        end
        repeat (total) begin
            void'(response.pop_front());
        end
        // nothing more may follow
        repeat (12 * CLOCKS_PER_BIT) @(negedge CLK);
        check(response.size() == 0, {what, ": extra bytes after the response"});
    endtask

    task automatic command_without_payload(input logic [7:0] command, input int target, input int address,
                                           input int count, input int response_bytes, input string what);
        send_packet(command, target, address, count, 0);
        get_response(response_bytes, 0, what);
    endtask

    // STATUS: flags in received[0], busy cycles in received[1..4]
    task automatic get_status(output logic [7:0] flags, output int cycles);
        command_without_payload(COMMAND_STATUS, 0, 0, 0, 5, "STATUS");
        flags  = received[0];
        cycles = {received[4], received[3], received[2], received[1]};
    endtask

    // everything the host writes before a phase
    task automatic host_load(input string phase);
        int    loads;
        int    fields;
        int    target, base, words;
        string load_phase, kind, file_name;

        loads = $fopen(vector_file("program_loads.txt"), "r");
        while (!$feof(loads)) begin
            fields = $fscanf(loads, "%s %s %d %d %d %s\n", load_phase, kind, target, base, words, file_name);
            if (fields != 6 || load_phase != phase) begin
                continue;
            end
            if (kind == "buffer") begin
                $readmemh(vector_file(file_name), image_mem, 0, words - 1);
                payload = new[8*words];
                for (int word = 0; word < words; word++) begin
                    for (int b = 0; b < 8; b++) begin
                        payload[8*word + b] = image_mem[word][8*b +: 8];
                    end
                end
                send_packet(COMMAND_WRITE_BUFFER, target, base, words, 8*words);
            end
            else if (kind == "bias") begin
                $readmemh(vector_file(file_name), bias_image_mem, 0, words - 1);
                payload = new[32*words];
                for (int word = 0; word < words; word++) begin
                    for (int b = 0; b < 32; b++) begin
                        payload[32*word + b] = bias_image_mem[word][8*b +: 8];
                    end
                end
                send_packet(COMMAND_WRITE_BIAS, 0, base, words, 32*words);
            end
            else begin
                $readmemh(vector_file(file_name), layernorm_image_mem, 0, words - 1);
                payload = new[6*words];
                for (int word = 0; word < words; word++) begin
                    for (int b = 0; b < 6; b++) begin
                        payload[6*word + b] = layernorm_image_mem[word][8*b +: 8];
                    end
                end
                send_packet(COMMAND_WRITE_LAYERNORM, 0, base, words, 6*words);
            end
            get_response(0, 0, {"write ", file_name});
        end
        $fclose(loads);
    endtask

    // READ_BUFFER of one op's output region, compared with its golden tensor
    task automatic read_and_check(input int entry);
        int rows           = op_rows[entry];
        int cols           = op_cols[entry];
        int outer          = op_row_banks[entry] ? rows : cols;     // tensor dimension spread over the banks
        int inner          = op_row_banks[entry] ? cols : rows;     // tensor dimension stored along the address
        int words          = ((outer + LANES - 1) / LANES - 1) * op_stride[entry] + inner;
        int mismatches     = 0;
        int first_mismatch = -1;

        $readmemh(vector_file(text(op_file[entry])), expected_mem, 0, rows*cols - 1);
        command_without_payload(COMMAND_READ_BUFFER, op_buffer[entry], op_base[entry], words, 8*words,
                                {"READ_BUFFER ", text(op_name[entry])});
        for (int row = 0; row < rows; row++) begin
            for (int col = 0; col < cols; col++) begin
                automatic int bank    = op_row_banks[entry] ? row % LANES : col % LANES;
                automatic int address = op_row_banks[entry] ? (row / LANES)*op_stride[entry] + col
                                                            : (col / LANES)*op_stride[entry] + row;
                if (received[8*address + bank] !== expected_mem[row*cols + col]) begin
                    if (first_mismatch < 0) begin
                        first_mismatch = row*cols + col;
                    end
                    mismatches++;
                end
            end
        end
        entries_ok    += rows*cols - mismatches;
        entries_wrong += mismatches;
        check(mismatches == 0, $sformatf("stage %s: %0d of %0d entries wrong, first mismatch index %0d",
                                         text(op_name[entry]), mismatches, rows*cols, first_mismatch));
        if (mismatches == 0) begin
            $display("  %3d %-9s %-13s PASS  %-9s %0dx%0d read from buffer %0d, %0d of %0d entries",
                     entry, text(op_phase[entry]), text(op_name[entry]), text(op_kind[entry]), rows, cols,
                     op_buffer[entry], rows*cols, rows*cols);
        end
    endtask

    // writes a golden tensor to the row banks of a buffer
    task automatic write_tensor(input int buffer_id, input string file_name, input int rows, input int cols);
        int words = ((rows + LANES - 1) / LANES) * cols;
        $readmemh(vector_file(file_name), expected_mem, 0, rows*cols - 1);
        payload = new[8*words];
        for (int word = 0; word < words; word++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                automatic int row = (word / cols)*LANES + lane;
                payload[8*word + lane] = (row < rows) ? expected_mem[row*cols + word % cols] : 8'h00;
            end
        end
        send_packet(COMMAND_WRITE_BUFFER, buffer_id, 0, words, 8*words);
        get_response(0, 0, {"write ", file_name});
    endtask

    task automatic clear_buffer(input int buffer_id, input int words);
        payload = new[8*words];
        for (int i = 0; i < 8*words; i++) begin
            payload[i] = 8'h00;
        end
        send_packet(COMMAND_WRITE_BUFFER, buffer_id, 0, words, 8*words);
        get_response(0, 0, "clearing a buffer");
    endtask

    initial begin
        int file;
        int fields;
        int entry;
        int phase;
        int polls = 0;
        logic [7:0] flags;
        int cycles;
        string layout;

        $display("tb_arty_top_mnist112: UART model at %0d clocks per bit", CLOCKS_PER_BIT);
        ck_rst = 1'b0;
        repeat (5) @(negedge CLK);
        ck_rst = 1'b1;
        repeat (40) @(negedge CLK);
        check(led === 4'b0000, $sformatf("LEDs %b after reset", led));

        file = $fopen(vector_file("program_ops.txt"), "r");
        while (!$feof(file)) begin
            fields = $fscanf(file, "%d", entry);
            if (fields != 1) begin
                break;
            end
            fields = $fscanf(file, "%s %s %s %d %s %d %d %d %d %s\n",
                             op_phase[entry], op_name[entry], op_kind[entry], op_buffer[entry], layout,
                             op_base[entry], op_stride[entry], op_rows[entry], op_cols[entry], op_file[entry]);
            if (fields != 10 || entry != op_count) begin
                $fatal(1, "FAIL: program_ops.txt line %0d is malformed", op_count);
            end
            op_row_banks[entry] = layout == "row";
            op_count++;
        end
        $fclose(file);
        file = $fopen(vector_file("program_phases.txt"), "r");
        while (!$feof(file)) begin
            fields = $fscanf(file, "%s %d %d\n", phase_name[phase_count], phase_first[phase_count],
                             phase_ops[phase_count]);
            if (fields == 3) begin
                phase_count++;
            end
        end
        $fclose(file);
        phase = phase_count - 1;
        if (text(phase_name[phase]) != "final") begin
            $fatal(1, "FAIL: the last phase is %s, not final", text(phase_name[phase]));
        end

        // --- protocol errors and the error LED --------------------------------------
        get_status(flags, cycles);
        check(flags == 8'd0, $sformatf("flags %b after reset", flags));
        payload = new[8];
        for (int b = 0; b < 8; b++) begin
            payload[b] = 8'h11 * 8'(b);
        end
        send_packet(COMMAND_WRITE_BUFFER, 7, 0, 1, 8, 1'b1);
        get_response(0, STATUS_CHECKSUM, "packet with a wrong checksum");
        check(led[2] === 1'b1, "error LED off after a wrong checksum");
        command_without_payload(COMMAND_CLEAR, 0, 0, 0, 0, "CLEAR");
        repeat (4) @(negedge CLK);
        check(led[2] === 1'b0, "error LED on after CLEAR");
        send_packet(COMMAND_WRITE_BUFFER, 7, 0, 1, 8);
        get_response(0, 0, "WRITE_BUFFER");
        command_without_payload(COMMAND_READ_BUFFER, 7, 0, 1, 8, "READ_BUFFER");
        for (int b = 0; b < 8; b++) begin
            check(received[b] == payload[b], $sformatf("read back byte %0d is %h", b, received[b]));
        end

        // --- the last phase: final LayerNorm and the classifiers ----------------------
        // buffers 3 (N) and 4 (S) take its results; cleared so the read-back is defined
        clear_buffer(3, 1344);
        clear_buffer(4, 1050);
        $readmemh(vector_file("program_desc.hex"), descriptor_mem, 0, DESCRIPTOR_WORDS*op_count - 1);
        payload = new[4*DESCRIPTOR_WORDS*op_count];
        for (int word = 0; word < DESCRIPTOR_WORDS*op_count; word++) begin
            for (int b = 0; b < 4; b++) begin
                payload[4*word + b] = descriptor_mem[word][8*b +: 8];
            end
        end
        send_packet(COMMAND_WRITE_DESCRIPTOR, 0, 0, op_count, 4*DESCRIPTOR_WORDS*op_count);
        get_response(0, 0, "WRITE_DESCRIPTOR");
        // the residual stream after layer 2, as the phases before would have left it
        write_tensor(0, "L1_x_out.hex", op_rows[phase_first[phase]], op_cols[phase_first[phase]]);
        host_load("final");
        command_without_payload(COMMAND_START, 0, phase_first[phase], 0, 0, "START");
        repeat (4) @(negedge CLK);
        check(led[0] === 1'b1, "busy LED off right after START");
        do begin
            repeat (2000) @(negedge CLK);
            get_status(flags, cycles);
            polls++;
        end while (flags[0] && polls < 2000);
        check(!flags[0] && flags[1] && !flags[2], $sformatf("flags %b after the run", flags));
        check(led === 4'b0010, $sformatf("LEDs %b after the run", led));
        $display("  phase final %0d ops  cycles %0d", phase_ops[phase], cycles);
        for (int op = phase_first[phase]; op < phase_first[phase] + phase_ops[phase]; op++) begin
            read_and_check(op);
        end

        $display("  bytes sent over the UART model %0d", bytes_sent);
        if (checks_failed == 0 && checks_passed > 0) begin
            $display("PASS: %0d of %0d checks, %0d of %0d entries correct through the UART pins",
                     checks_passed, checks_passed, entries_ok, entries_ok + entries_wrong);
        end
        else begin
            $display("FAIL: %0d of %0d checks failed, %0d of %0d entries wrong",
                     checks_failed, checks_passed + checks_failed, entries_wrong, entries_ok + entries_wrong);
        end
        $finish;
    end

endmodule
