`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for matmul_engine on the real pretrained DeiT-Tiny.
//
// python/make_model.py --board opal_kelly --layers 2 captures the real input of every matmul of the
// patch embedding, encoder layers 0 and 1 and the classifier head, quantizes
// it to INT8 and builds each op's golden INT8 result. Softmax, LayerNorm and
// GELU are not built yet, so every op is tested on its own: the testbench
// loads the op's real operands through the host port, writes its one-op
// descriptor table, runs the engine and compares the output region bit-exact
// with the golden file. Words of the region outside the result must keep the
// pattern they were filled with.
//
// It reports, per op and per group (embed, L0, L1, head):
//   cycles       engine busy cycles for the op
//   ideal        MACs / (SYS_ROW * SYS_COL)
//   utilization  ideal / cycles
//   flush        tiles * PIPELINE_LATENCY, the zero-fed cycles between tiles
//
// Vectors, in VECTOR_DIR:
//   deit_ops.txt   one line per op: name, group, A file, B file, bias file or
//                  "-", Y file, descriptor file
//   A  INT8 A[row][k], B  INT8 B[k][col], bias  INT32 bias[col], Y  INT8,
//   descriptor  descriptor_t as 32-bit words. Tables are for an 8x8 array.
//////////////////////////////////////////////////////////////////////////////////

`include "vit_paths.svh"

module tb_matmul_engine_opal_kelly #(
    parameter int    SYS_ROW    = 8,
    parameter int    SYS_COL    = 8,
    parameter string VECTOR_DIR = {`VIT_ROOT, "python/models/opal_kelly_L2/matmuls/"}
);

    import vit_config_pkg::descriptor_t;
    import vit_config_pkg::DRAIN_COLUMN_SELECT;
    import vit_config_pkg::OPCODE_MATMUL;

    localparam int DATA_WIDTH              = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH               = vit_config_pkg::ACC_WIDTH;
    localparam int BUFFER_COUNT            = vit_config_pkg::BUFFER_COUNT;
    localparam int BUFFER_ID_WIDTH         = vit_config_pkg::BUFFER_ID_WIDTH;
    localparam int ADDRESS_WIDTH           = vit_config_pkg::ADDRESS_WIDTH;
    localparam int DESCRIPTOR_COUNT        = vit_config_pkg::DESCRIPTOR_COUNT;
    localparam int DESCRIPTOR_WORDS        = vit_config_pkg::DESCRIPTOR_WORDS;
    localparam int DESCRIPTOR_WIDTH        = vit_config_pkg::DESCRIPTOR_WIDTH;
    localparam int DESCRIPTOR_MEMORY_WIDTH = vit_config_pkg::DESCRIPTOR_MEMORY_WIDTH;
    localparam int DESCRIPTOR_INDEX_WIDTH  = $clog2(DESCRIPTOR_COUNT);

    localparam int LANES            = (SYS_ROW > SYS_COL) ? SYS_ROW : SYS_COL;
    localparam int PIPELINE_LATENCY = SYS_ROW + SYS_COL - 2 + vit_config_pkg::DSP_LATENCY;

    // must match BANK_DEPTH in python/golden_model.py
    localparam int BUFFER_DEPTH = vit_config_pkg::BANK_DEPTH_MAX;

    localparam int GROUP_COUNT      = 4;  // embed, L0, L1, head
    localparam int VECTOR_WORDS_MAX = 262144;
    localparam int TIMEOUT_CYCLES   = 4000000;
    localparam int REGION_SLACK     = 16;

    logic CLK = 1'b0;
    logic reset;

    logic                              start;
    logic [DESCRIPTOR_INDEX_WIDTH-1:0] start_index;
    logic                              busy;
    logic                              done;

    logic [DESCRIPTOR_INDEX_WIDTH-1:0] host_descriptor_index;
    logic [DESCRIPTOR_WORDS-1:0]       host_descriptor_write_enable;
    logic [31:0]                       host_descriptor_write_data;

    logic [BUFFER_ID_WIDTH-1:0]        host_buffer_id;
    logic [ADDRESS_WIDTH-1:0]          host_address;
    logic [LANES-1:0]                  host_buffer_write_enable;
    logic [LANES-1:0][DATA_WIDTH-1:0]  host_buffer_write_data;
    logic [LANES-1:0][DATA_WIDTH-1:0]  host_buffer_read_data;

    logic [SYS_COL-1:0]                host_bias_write_enable;
    logic [SYS_COL-1:0][ACC_WIDTH-1:0] host_bias_write_data;
    logic [SYS_COL-1:0][ACC_WIDTH-1:0] host_bias_read_data;

    matmul_engine #(
        .SYS_ROW        (SYS_ROW),
        .SYS_COL        (SYS_COL),
        .BUFFER_DEPTH   ({BUFFER_COUNT{32'(BUFFER_DEPTH)}}),
        .GELU_ROM_FILE  (`VIT_ROM_224("gelu_rom.hex")),
        .EXP_ROM_FILE   (`VIT_ROM_224("exp_rom.hex")),
        .RSQRT_ROM_FILE (`VIT_ROM_224("rsqrt_rom.hex"))
    ) dut (
        .CLK                          (CLK),
        .reset                        (reset),
        .start                        (start),
        .start_index                  (start_index),
        .busy                         (busy),
        .done                         (done),
        .host_descriptor_index        (host_descriptor_index),
        .host_descriptor_write_enable (host_descriptor_write_enable),
        .host_descriptor_write_data   (host_descriptor_write_data),
        .host_buffer_id               (host_buffer_id),
        .host_address                 (host_address),
        .host_buffer_write_enable     (host_buffer_write_enable),
        .host_buffer_write_data       (host_buffer_write_data),
        .host_buffer_read_data        (host_buffer_read_data),
        .host_bias_write_enable       (host_bias_write_enable),
        .host_bias_write_data         (host_bias_write_data),
        .host_bias_read_data          (host_bias_read_data)
    );

    always #5 CLK = ~CLK;

    logic [31:0]           descriptor_mem[DESCRIPTOR_WORDS];
    logic [DATA_WIDTH-1:0] activation_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] weight_mem[VECTOR_WORDS_MAX];
    logic [ACC_WIDTH-1:0]  bias_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] expected_mem[VECTOR_WORDS_MAX];

    int busy_cycles;

    int  group_ops[GROUP_COUNT];
    int  group_passed[GROUP_COUNT];
    longint group_cycles[GROUP_COUNT];
    longint group_macs[GROUP_COUNT];
    longint group_flush[GROUP_COUNT];

    int ops_passed    = 0;
    int ops_failed    = 0;
    int entries_ok    = 0;
    int entries_wrong = 0;

    always @(posedge CLK) begin
        if (busy) begin
            busy_cycles++;
        end
    end

    function automatic int group_index(input string group);
        if (group == "embed") return 0;
        if (group == "L0")    return 1;
        if (group == "L1")    return 2;
        if (group == "head")  return 3;
        $fatal(1, "FAIL: unknown op group %s", group);
        return 0;
    endfunction

    function automatic string group_name(input int index);
        case (index)
            0:       return "embed";
            1:       return "L0";
            2:       return "L1";
            default: return "head";
        endcase
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

    function automatic logic [DATA_WIDTH-1:0] fill_pattern(input int address, input int lane);
        return DATA_WIDTH'(8'h5A ^ (address * 7) ^ (lane * 37));
    endfunction

    task automatic run_op(input string op_name, input string group, input string a_file, input string b_file,
                          input string bias_file, input string y_file, input string descriptor_file);
        descriptor_t                        descriptor;
        logic [DESCRIPTOR_MEMORY_WIDTH-1:0] descriptor_word;
        int     rows, inner, cols, row_tiles, col_tiles;
        int     bank_rows, stride, output_base, region_end;
        int     mismatches     = 0;
        int     first_mismatch = -1;
        int     entries_seen   = 0;
        int     overwritten    = 0;
        int     cycles         = 0;
        int     op_cycles;
        int     flush_cycles;
        longint macs;
        real    ideal;
        bit     finished       = 1'b0;
        bit     op_ok;
        int     g;

        g = group_index(group);
        $readmemh(vector_file(descriptor_file), descriptor_mem);
        for (int word = 0; word < DESCRIPTOR_WORDS; word++) begin
            descriptor_word[32*word +: 32] = descriptor_mem[word];
        end
        descriptor  = descriptor_t'(descriptor_word[DESCRIPTOR_WIDTH-1:0]);
        rows        = int'(descriptor.rows_total);
        inner       = int'(descriptor.inner_dim);
        cols        = int'(descriptor.cols_total);
        row_tiles   = int'(descriptor.row_tiles);
        col_tiles   = int'(descriptor.col_tiles);
        stride      = int'(descriptor.output_stride);
        output_base = int'(descriptor.output_base);

        if (descriptor.opcode != OPCODE_MATMUL || !descriptor.last_op
            || row_tiles != (rows + SYS_ROW - 1) / SYS_ROW || col_tiles != (cols + SYS_COL - 1) / SYS_COL
            || descriptor.bias_enable != (bias_file != "-")) begin
            $fatal(1, "FAIL: descriptor table %s does not fit this testbench", descriptor_file);
        end
        if (rows*inner > VECTOR_WORDS_MAX || inner*cols > VECTOR_WORDS_MAX || rows*cols > VECTOR_WORDS_MAX) begin
            $fatal(1, "FAIL: op %s is larger than VECTOR_WORDS_MAX", op_name);
        end
        $readmemh(vector_file(a_file), activation_mem, 0, rows*inner - 1);
        $readmemh(vector_file(b_file), weight_mem, 0, inner*cols - 1);
        $readmemh(vector_file(y_file), expected_mem, 0, rows*cols - 1);
        if (descriptor.bias_enable) begin
            $readmemh(vector_file(bias_file), bias_mem, 0, cols - 1);
        end

        bank_rows  = (descriptor.drain_mode == DRAIN_COLUMN_SELECT) ? row_tiles : col_tiles;
        region_end = output_base + bank_rows*stride + REGION_SLACK;
        if (region_end > BUFFER_DEPTH) begin
            region_end = BUFFER_DEPTH;
        end

        // descriptor
        @(negedge CLK);
        start_index           = '0;
        host_descriptor_index = '0;
        for (int word = 0; word < DESCRIPTOR_WORDS; word++) begin
            host_descriptor_write_enable       = '0;
            host_descriptor_write_enable[word] = 1'b1;
            host_descriptor_write_data         = descriptor_mem[word];
            @(negedge CLK);
        end
        host_descriptor_write_enable = '0;

        // A[row][k] goes to row bank row % SYS_ROW at activation_base + (row / SYS_ROW)*K + k
        host_buffer_id = descriptor.activation_buffer_id;
        for (int tile = 0; tile < row_tiles; tile++) begin
            for (int k = 0; k < inner; k++) begin
                host_address = int'(descriptor.activation_base) + tile*inner + k;
                for (int lane = 0; lane < LANES; lane++) begin
                    automatic int row = tile*SYS_ROW + lane;
                    host_buffer_write_enable[lane] = lane < SYS_ROW && row < rows;
                    host_buffer_write_data[lane]   = (lane < SYS_ROW && row < rows) ? activation_mem[row*inner + k] : '0;
                end
                @(negedge CLK);
            end
        end

        // B[k][col] goes to column bank col % SYS_COL at weight_base + (col / SYS_COL)*K + k
        host_buffer_id = descriptor.weight_buffer_id;
        for (int tile = 0; tile < col_tiles; tile++) begin
            for (int k = 0; k < inner; k++) begin
                host_address = int'(descriptor.weight_base) + tile*inner + k;
                for (int lane = 0; lane < LANES; lane++) begin
                    automatic int col = tile*SYS_COL + lane;
                    host_buffer_write_enable[lane] = lane < SYS_COL && col < cols;
                    host_buffer_write_data[lane]   = (lane < SYS_COL && col < cols) ? weight_mem[k*cols + col] : '0;
                end
                @(negedge CLK);
            end
        end

        // output region: known pattern in every bank
        host_buffer_id = descriptor.output_buffer_id;
        for (int address = 0; address < region_end; address++) begin
            host_address = address;
            for (int lane = 0; lane < LANES; lane++) begin
                host_buffer_write_enable[lane] = 1'b1;
                host_buffer_write_data[lane]   = fill_pattern(address, lane);
            end
            @(negedge CLK);
        end
        host_buffer_write_enable = '0;

        // bias[col] goes to column bank col % SYS_COL at bias_base + col / SYS_COL
        if (descriptor.bias_enable) begin
            for (int tile = 0; tile < col_tiles; tile++) begin
                host_address = int'(descriptor.bias_base) + tile;
                for (int lane = 0; lane < SYS_COL; lane++) begin
                    automatic int col = tile*SYS_COL + lane;
                    host_bias_write_enable[lane] = col < cols;
                    host_bias_write_data[lane]   = (col < cols) ? bias_mem[col] : '0;
                end
                @(negedge CLK);
            end
            host_bias_write_enable = '0;
        end

        // run
        busy_cycles = 0;
        start       = 1'b1;
        @(negedge CLK);
        start = 1'b0;
        while (!finished && cycles < TIMEOUT_CYCLES) begin
            @(posedge CLK);
            if (done) begin
                finished = 1'b1;
            end
            cycles++;
        end
        @(negedge CLK);
        op_cycles = busy_cycles;

        // read the whole output region back
        host_buffer_id = descriptor.output_buffer_id;
        for (int address = 0; address < region_end; address++) begin
            host_address = address;
            repeat (vit_config_pkg::BUFFER_READ_LATENCY) @(negedge CLK);
            for (int lane = 0; lane < LANES; lane++) begin
                automatic int relative = address - output_base;
                automatic int bank_row = (relative >= 0) ? relative / stride : -1;
                automatic int position = (relative >= 0) ? relative % stride : -1;
                automatic int row;
                automatic int col;
                automatic bit valid;
                if (descriptor.drain_mode == DRAIN_COLUMN_SELECT) begin
                    // row bank `lane`: stored row bank_row, element position - (output_base % stride)
                    row   = bank_row*SYS_ROW + lane;
                    col   = position;
                    valid = relative >= 0 && lane < SYS_ROW && row < rows && col < cols;
                end
                else begin
                    // column bank `lane`: stored column bank_row, element position
                    col   = bank_row*SYS_COL + lane;
                    row   = position;
                    valid = relative >= 0 && lane < SYS_COL && col < cols && row < rows;
                end
                if (valid) begin
                    entries_seen++;
                    if (host_buffer_read_data[lane] !== expected_mem[row*cols + col]) begin
                        if (first_mismatch < 0) begin
                            first_mismatch = row*cols + col;
                            $display("  %s first mismatch at index %0d (row %0d, col %0d): got %h, expected %h",
                                     op_name, first_mismatch, row, col,
                                     host_buffer_read_data[lane], expected_mem[row*cols + col]);
                        end
                        mismatches++;
                    end
                end
                else if (host_buffer_read_data[lane] !== fill_pattern(address, lane)) begin
                    if (overwritten == 0) begin
                        $display("  %s wrote outside the result at address %0d, bank %0d", op_name, address, lane);
                    end
                    overwritten++;
                end
            end
        end

        if (!finished) begin
            $display("  %s timed out after %0d cycles", op_name, cycles);
        end
        if (entries_seen != rows*cols) begin
            $display("  %s checked %0d entries, expected %0d", op_name, entries_seen, rows*cols);
        end

        macs         = longint'(rows) * inner * cols;
        ideal        = real'(macs) / (SYS_ROW * SYS_COL);
        flush_cycles = row_tiles * col_tiles * PIPELINE_LATENCY;
        op_ok        = finished && mismatches == 0 && overwritten == 0 && entries_seen == rows*cols;

        group_ops[g]++;
        group_cycles[g] += op_cycles;
        group_macs[g]   += macs;
        group_flush[g]  += flush_cycles;
        entries_ok      += entries_seen - mismatches;
        entries_wrong   += mismatches;
        if (op_ok) begin
            ops_passed++;
            group_passed[g]++;
            $display("  %-9s PASS  %0dx%0d @ %0dx%0d  %0d of %0d entries  cycles %0d  ideal %0.0f  utilization %0.1f%%  flush %0d",
                     op_name, rows, inner, inner, cols, rows*cols, rows*cols,
                     op_cycles, ideal, 100.0 * ideal / op_cycles, flush_cycles);
        end
        else begin
            ops_failed++;
            $display("  %-9s FAIL  %0dx%0d @ %0dx%0d  %0d of %0d entries wrong, first mismatch index %0d, %0d words written outside the result",
                     op_name, rows, inner, inner, cols, mismatches, rows*cols, first_mismatch, overwritten);
        end
    endtask

    initial begin
        int     manifest;
        int     fields;
        string  op_name, group, a_file, b_file, bias_file, y_file, descriptor_file;
        longint total_cycles = 0;
        longint total_macs   = 0;
        longint total_flush  = 0;

        $display("tb_matmul_engine_opal_kelly: SYS_ROW=%0d SYS_COL=%0d", SYS_ROW, SYS_COL);
        reset                        = 1'b1;
        start                        = 1'b0;
        start_index                  = '0;
        host_descriptor_index        = '0;
        host_descriptor_write_enable = '0;
        host_descriptor_write_data   = '0;
        host_buffer_id               = '0;
        host_address                 = '0;
        host_buffer_write_enable     = '0;
        host_buffer_write_data       = '0;
        host_bias_write_enable       = '0;
        host_bias_write_data         = '0;
        for (int g = 0; g < GROUP_COUNT; g++) begin
            group_ops[g]    = 0;
            group_passed[g] = 0;
            group_cycles[g] = 0;
            group_macs[g]   = 0;
            group_flush[g]  = 0;
        end
        repeat (3) @(negedge CLK);
        reset = 1'b0;

        manifest = $fopen(vector_file("deit_ops.txt"), "r");
        while (!$feof(manifest)) begin
            fields = $fscanf(manifest, "%s %s %s %s %s %s %s\n",
                             op_name, group, a_file, b_file, bias_file, y_file, descriptor_file);
            if (fields == 7) begin
                run_op(op_name, group, a_file, b_file, bias_file, y_file, descriptor_file);
            end
        end
        $fclose(manifest);

        for (int g = 0; g < GROUP_COUNT; g++) begin
            $display("  group %-5s %0d of %0d ops pass  cycles %0d  ideal %0.0f  utilization %0.1f%%  flush %0d (%0.1f%% of cycles)",
                     group_name(g), group_passed[g], group_ops[g], group_cycles[g],
                     real'(group_macs[g]) / (SYS_ROW * SYS_COL),
                     100.0 * real'(group_macs[g]) / (SYS_ROW * SYS_COL) / group_cycles[g],
                     group_flush[g], 100.0 * group_flush[g] / group_cycles[g]);
            total_cycles += group_cycles[g];
            total_macs   += group_macs[g];
            total_flush  += group_flush[g];
        end
        $display("  all ops     cycles %0d  ideal %0.0f  utilization %0.1f%%  flush %0d (%0.1f%% of cycles)",
                 total_cycles, real'(total_macs) / (SYS_ROW * SYS_COL),
                 100.0 * real'(total_macs) / (SYS_ROW * SYS_COL) / total_cycles,
                 total_flush, 100.0 * total_flush / total_cycles);

        if (ops_failed == 0 && ops_passed > 0) begin
            $display("PASS: %0dx%0d array, %0d of %0d ops, %0d of %0d entries correct",
                     SYS_ROW, SYS_COL, ops_passed, ops_passed, entries_ok, entries_ok + entries_wrong);
        end
        else begin
            $display("FAIL: %0dx%0d array, %0d of %0d ops failed, %0d of %0d entries wrong",
                     SYS_ROW, SYS_COL, ops_failed, ops_passed + ops_failed, entries_wrong, entries_ok + entries_wrong);
        end
        $finish;
    end

endmodule
