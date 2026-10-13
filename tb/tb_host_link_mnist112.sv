`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for host_link, with the engine behind it
// (vit_accelerator), driven only through the byte stream.
//
// It plays the host of docs/HOST_PROTOCOL.md for the whole 112 x 112 MNIST
// program: WRITE_DESCRIPTOR for the table, then for each phase the phase's
// WRITE_BUFFER, WRITE_BIAS and WRITE_LAYERNORM packets, START, STATUS polled
// until the engine is idle, and READ_BUFFER of every op output that is still
// whole when the phase ends (program_survivors.txt), compared with the golden
// tensor of that stage. The last phase's outputs are the class scores.
//
// Before the program it writes zeros to every buffer, so that the bank words
// no op writes (lanes past a tensor's last row) are defined when read back.
//
// Protocol checks: every response must start with 0x5A, carry the expected
// status and a correct CRC. A packet with a wrong checksum, an unknown
// command and a write sent while the engine runs must be answered with the
// checksum, command and busy status and set the sticky error flag, CLEAR
// must clear it, and none of them may disturb the run.
//
// Bytes are offered one per cycle whenever rx_ready allows; tx_ready is
// toggled pseudo-randomly, so the response path sees back-pressure.
//
// Vectors: golden/gen_mnist_forward.py, in golden/vectors/mnist112 (the same
// files tb_encoder_program_mnist112 uses). 8x8 only.
//////////////////////////////////////////////////////////////////////////////////

module tb_host_link #(
    parameter string VECTOR_DIR     = "../golden/vectors/mnist112/",
    parameter int    DEPTH_X        = 1344,
    parameter int    DEPTH_W        = 16384,
    parameter int    DEPTH_BIG      = 5376,
    parameter int    DEPTH_N        = 1344,
    parameter int    DEPTH_S        = 1050,
    parameter int    DEPTH_QV       = 2544,
    parameter int    DEPTH_K        = 1344,
    parameter int    DEPTH_UNUSED   = 16,
    parameter string GELU_ROM_FILE  = "rom/mnist112/gelu_rom.hex",
    parameter string EXP_ROM_FILE   = "rom/mnist112/exp_rom.hex",
    parameter string RSQRT_ROM_FILE = "rom/mnist112/rsqrt_rom.hex"
);

    localparam int LANES            = 8;
    localparam int DATA_WIDTH       = vit_config_pkg::DATA_WIDTH;
    localparam int BUFFER_COUNT     = vit_config_pkg::BUFFER_COUNT;
    localparam int DESCRIPTOR_WORDS = vit_config_pkg::DESCRIPTOR_WORDS;
    localparam int DESCRIPTOR_COUNT = vit_config_pkg::DESCRIPTOR_COUNT;

    localparam bit [BUFFER_COUNT-1:0][31:0] BUFFER_DEPTH
        = {32'(DEPTH_UNUSED), 32'(DEPTH_K), 32'(DEPTH_QV), 32'(DEPTH_S), 32'(DEPTH_N), 32'(DEPTH_BIG),
           32'(DEPTH_W), 32'(DEPTH_X)};

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

    vit_accelerator #(
        .SYS_ROW        (LANES),
        .SYS_COL        (LANES),
        .BUFFER_DEPTH   (BUFFER_DEPTH),
        .GELU_ROM_FILE  (GELU_ROM_FILE),
        .EXP_ROM_FILE   (EXP_ROM_FILE),
        .RSQRT_ROM_FILE (RSQRT_ROM_FILE)
    ) dut (
        .CLK        (CLK),
        .reset      (reset),
        .rx_data    (rx_data),
        .rx_valid   (rx_valid),
        .rx_ready   (rx_ready),
        .tx_data    (tx_data),
        .tx_valid   (tx_valid),
        .tx_ready   (tx_ready),
        .busy       (busy),
        .done_flag  (done_flag),
        .error_flag (error_flag)
    );

    always #5 CLK = ~CLK;

    // back-pressure on the response path
    logic [15:0] ready_lfsr = 16'hACE1;

    always @(negedge CLK) begin
        ready_lfsr <= {ready_lfsr[14:0], ready_lfsr[15] ^ ready_lfsr[13] ^ ready_lfsr[12] ^ ready_lfsr[10]};
        tx_ready   <= ready_lfsr[0] | ready_lfsr[3];
    end

    // every byte the DUT sends
    logic [7:0] response[$];

    always @(posedge CLK) begin
        if (tx_valid && tx_ready) begin
            response.push_back(tx_data);
        end
    end

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
    bit    op_survives[DESCRIPTOR_COUNT];
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
    longint engine_cycles = 0;

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

    task automatic send_byte(input logic [7:0] value);
        rx_data  = value;
        rx_valid = 1'b1;
        while (!rx_ready) begin
            @(negedge CLK);
        end
        @(negedge CLK);
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
        rx_valid = 1'b0;
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
        repeat (4) @(negedge CLK);
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

    task automatic run_phase(input int phase, input bit disturb);
        logic [7:0] flags;
        int         cycles;
        int         polls = 0;

        host_load(text(phase_name[phase]));
        command_without_payload(COMMAND_START, 0, phase_first[phase], 0, 0, "START");
        if (disturb) begin
            // a write while the engine runs must be refused and must not land
            payload = new[8];
            for (int b = 0; b < 8; b++) begin
                payload[b] = 8'hEE;
            end
            send_packet(COMMAND_WRITE_BUFFER, 0, 0, 1, 8);
            get_response(0, STATUS_BUSY, "write while busy");
            send_packet(COMMAND_START, 0, 0, 0, 0);
            get_response(0, STATUS_BUSY, "START while busy");
            command_without_payload(COMMAND_CLEAR, 0, 0, 0, 0, "CLEAR");
        end
        do begin
            repeat (2000) @(negedge CLK);
            get_status(flags, cycles);
            polls++;
        end while (flags[0] && polls < 2000);
        check(!flags[0] && flags[1], $sformatf("phase %s: flags %b after the run", text(phase_name[phase]), flags));
        check(!flags[2], $sformatf("phase %s: error flag set", text(phase_name[phase])));
        engine_cycles += cycles;
        $display("  phase %-9s %0d ops  cycles %0d", text(phase_name[phase]), phase_ops[phase], cycles);
        for (int entry = phase_first[phase]; entry < phase_first[phase] + phase_ops[phase]; entry++) begin
            if (op_survives[entry]) begin
                read_and_check(entry);
            end
        end
    endtask

    initial begin
        int file;
        int fields;
        int entry;
        int tokens;
        int plan_depth[BUFFER_COUNT];
        logic [7:0] flags;
        int cycles;
        string layout;

        $display("tb_host_link: byte-level host of the 112 x 112 MNIST program");
        reset    = 1'b1;
        rx_valid = 1'b0;
        rx_data  = '0;
        repeat (3) @(negedge CLK);
        reset = 1'b0;

        file = $fopen(vector_file("program_plan.txt"), "r");
        fields = $fscanf(file, "%d %d %d %d %d %d %d %d %d", tokens, plan_depth[0], plan_depth[1], plan_depth[2],
                         plan_depth[3], plan_depth[4], plan_depth[5], plan_depth[6], plan_depth[7]);
        $fclose(file);
        for (int b = 0; b < BUFFER_COUNT; b++) begin
            if (fields != 9 || plan_depth[b] != int'(BUFFER_DEPTH[b])) begin
                $fatal(1, "FAIL: buffer %0d is %0d words per bank here, the program was laid out for %0d",
                       b, BUFFER_DEPTH[b], plan_depth[b]);
            end
        end
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
            op_survives[entry]  = 1'b0;
            op_count++;
        end
        $fclose(file);
        file = $fopen(vector_file("program_survivors.txt"), "r");
        while (!$feof(file)) begin
            fields = $fscanf(file, "%d\n", entry);
            if (fields == 1) begin
                op_survives[entry] = 1'b1;
            end
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

        // --- protocol errors before anything runs ---------------------------------
        get_status(flags, cycles);
        check(flags == 8'd0, $sformatf("flags %b after reset", flags));
        payload = new[8];
        for (int b = 0; b < 8; b++) begin
            payload[b] = 8'h11 * 8'(b);
        end
        send_packet(COMMAND_WRITE_BUFFER, 7, 0, 1, 8, 1'b1);
        get_response(0, STATUS_CHECKSUM, "packet with a wrong checksum");
        get_status(flags, cycles);
        check(flags[2], "error flag not set after a wrong checksum");
        check(error_flag, "error_flag output not set after a wrong checksum");
        command_without_payload(COMMAND_CLEAR, 0, 0, 0, 0, "CLEAR");
        get_status(flags, cycles);
        check(flags == 8'd0, $sformatf("flags %b after CLEAR", flags));
        send_packet(8'h7F, 0, 0, 0, 0);
        get_response(0, STATUS_COMMAND, "unknown command");
        command_without_payload(COMMAND_CLEAR, 0, 0, 0, 0, "CLEAR");
        // bytes outside a packet are ignored
        send_byte(8'h00);
        send_byte(8'h5A);
        rx_valid = 1'b0;
        // the same write again, this time intact, and read back
        send_packet(COMMAND_WRITE_BUFFER, 7, 0, 1, 8);
        get_response(0, 0, "WRITE_BUFFER");
        command_without_payload(COMMAND_READ_BUFFER, 7, 0, 1, 8, "READ_BUFFER");
        for (int b = 0; b < 8; b++) begin
            check(received[b] == payload[b], $sformatf("read back byte %0d is %h", b, received[b]));
        end

        // --- every buffer cleared once -------------------------------------------------
        // Bank words past a tensor's last row are never written by an op. A read-back
        // region contains them, so they get a known value here; on the board they
        // would hold whatever the RAM powered up with.
        for (int b = 0; b < BUFFER_COUNT; b++) begin
            payload = new[8*int'(BUFFER_DEPTH[b])];
            for (int i = 0; i < 8*int'(BUFFER_DEPTH[b]); i++) begin
                payload[i] = 8'h00;
            end
            send_packet(COMMAND_WRITE_BUFFER, b, 0, int'(BUFFER_DEPTH[b]), 8*int'(BUFFER_DEPTH[b]));
            get_response(0, 0, "clearing a buffer");
        end

        // --- the descriptor table ---------------------------------------------------
        $readmemh(vector_file("program_desc.hex"), descriptor_mem, 0, DESCRIPTOR_WORDS*op_count - 1);
        payload = new[4*DESCRIPTOR_WORDS*op_count];
        for (int word = 0; word < DESCRIPTOR_WORDS*op_count; word++) begin
            for (int b = 0; b < 4; b++) begin
                payload[4*word + b] = descriptor_mem[word][8*b +: 8];
            end
        end
        send_packet(COMMAND_WRITE_DESCRIPTOR, 0, 0, op_count, 4*DESCRIPTOR_WORDS*op_count);
        get_response(0, 0, "WRITE_DESCRIPTOR");
        $display("  %0d descriptors written, %0d phases, %0d tokens", op_count, phase_count, tokens);

        // --- the program ------------------------------------------------------------
        for (int phase = 0; phase < phase_count; phase++) begin
            run_phase(phase, phase == 2);
        end

        $display("  all phases  engine cycles %0d  bytes sent %0d", engine_cycles, bytes_sent);
        if (checks_failed == 0 && checks_passed > 0) begin
            $display("PASS: %0d of %0d checks, %0d of %0d entries correct through the byte stream",
                     checks_passed, checks_passed, entries_ok, entries_ok + entries_wrong);
        end
        else begin
            $display("FAIL: %0d of %0d checks failed, %0d of %0d entries wrong",
                     checks_failed, checks_passed + checks_failed, entries_wrong, entries_ok + entries_wrong);
        end
        $finish;
    end

endmodule
