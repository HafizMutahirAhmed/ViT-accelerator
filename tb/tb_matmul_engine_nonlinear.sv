`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for the row ops of matmul_engine (the non-linear
// ops: residual add, GELU, LayerNorm, softmax), run through the descriptor table like a matmul.
//
// For every op in the manifest whose kind this testbench knows, it writes the
// one-op descriptor table, fills the output region with a known pattern,
// loads the operands into the row banks of their pool buffers through the
// host port, runs the engine, reads the whole output region back and compares
// it with the golden INT8 tensor. Words outside the result must still hold
// the pattern, or, where an operand was loaded there, the operand. For a
// LayerNorm it first writes the gain and offset of every column.
//
// Vectors come from golden/gen_deit_forward.py, in golden/vectors/deit_forward:
//   nonlinear_ops.txt  one line per op: name, kind, a file, b file or "-",
//                      y file, descriptor file, two more files or "-"
//   a, b, y            INT8 rows x cols, row-major
//   gain, offset       LayerNorm only: INT16 and INT32 per column
// Tensor element [row][col] is in row bank row % SYS_ROW at
// base + (row / SYS_ROW) * cols + col (the output uses output_stride).
// The array size must be the one the descriptors were made for (8x8).
//////////////////////////////////////////////////////////////////////////////////

module tb_matmul_engine_nonlinear #(
    parameter int    SYS_ROW    = vit_config_pkg::SYS_ROW,
    parameter int    SYS_COL    = vit_config_pkg::SYS_COL,
    parameter string VECTOR_DIR = "../golden/vectors/deit_forward/"
);

    import vit_config_pkg::descriptor_t;
    import vit_config_pkg::OPCODE_ADD;
    import vit_config_pkg::OPCODE_GELU;
    import vit_config_pkg::OPCODE_LAYERNORM;
    import vit_config_pkg::OPCODE_SOFTMAX;

    localparam int DATA_WIDTH              = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH               = vit_config_pkg::ACC_WIDTH;
    localparam int LANES                   = (SYS_ROW > SYS_COL) ? SYS_ROW : SYS_COL;
    localparam int BUFFER_COUNT            = vit_config_pkg::BUFFER_COUNT;
    localparam int BUFFER_ID_WIDTH         = vit_config_pkg::BUFFER_ID_WIDTH;
    localparam int ADDRESS_WIDTH           = vit_config_pkg::ADDRESS_WIDTH;
    localparam int DESCRIPTOR_WORDS        = vit_config_pkg::DESCRIPTOR_WORDS;
    localparam int DESCRIPTOR_WIDTH        = vit_config_pkg::DESCRIPTOR_WIDTH;
    localparam int DESCRIPTOR_MEMORY_WIDTH = vit_config_pkg::DESCRIPTOR_MEMORY_WIDTH;
    localparam int DESCRIPTOR_INDEX_WIDTH  = $clog2(vit_config_pkg::DESCRIPTOR_COUNT);

    // 197 x 768 in row banks is 25 * 768 words per bank
    localparam int BUFFER_DEPTH     = 32768;
    localparam int VECTOR_WORDS_MAX = 262144;
    localparam int REGION_SLACK     = 64;
    localparam int TIMEOUT_CYCLES   = 2000000;

    logic                              CLK = 1'b0;
    logic                              reset;
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
    logic                                                     host_layernorm_write_enable = 1'b0;
    logic signed [vit_config_pkg::LAYERNORM_GAIN_WIDTH-1:0]   host_layernorm_write_gain   = '0;
    logic signed [vit_config_pkg::LAYERNORM_OFFSET_WIDTH-1:0] host_layernorm_write_offset = '0;

    matmul_engine #(
        .SYS_ROW      (SYS_ROW),
        .SYS_COL      (SYS_COL),
        .BUFFER_DEPTH ({BUFFER_COUNT{32'(BUFFER_DEPTH)}})
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
        .host_bias_read_data          (host_bias_read_data),
        .host_layernorm_write_enable  (host_layernorm_write_enable),
        .host_layernorm_write_gain    (host_layernorm_write_gain),
        .host_layernorm_write_offset  (host_layernorm_write_offset)
    );

    always #5 CLK = ~CLK;

    logic [DATA_WIDTH-1:0] a_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] b_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] expected_mem[VECTOR_WORDS_MAX];
    logic [31:0]           descriptor_mem[DESCRIPTOR_WORDS];
    logic [vit_config_pkg::LAYERNORM_GAIN_WIDTH-1:0]   gain_mem[1024];
    logic [vit_config_pkg::LAYERNORM_OFFSET_WIDTH-1:0] offset_mem[1024];

    int busy_cycles;
    int ops_passed    = 0;
    int ops_failed    = 0;
    int entries_ok    = 0;
    int entries_wrong = 0;

    always @(posedge CLK) begin
        if (busy) begin
            busy_cycles++;
        end
    end

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

    // writes a rows x cols tensor to the row banks of a buffer
    task automatic load_tensor(input int buffer_id, input int base, input int rows, input int cols,
                               input bit second);
        host_buffer_id = BUFFER_ID_WIDTH'(buffer_id);
        for (int tile = 0; tile < (rows + SYS_ROW - 1) / SYS_ROW; tile++) begin
            for (int column = 0; column < cols; column++) begin
                host_address = ADDRESS_WIDTH'(base + tile*cols + column);
                for (int lane = 0; lane < LANES; lane++) begin
                    automatic int row = tile*SYS_ROW + lane;
                    host_buffer_write_enable[lane] = lane < SYS_ROW && row < rows;
                    host_buffer_write_data[lane]   = !(lane < SYS_ROW && row < rows) ? '0
                                                     : second ? b_mem[row*cols + column] : a_mem[row*cols + column];
                end
                @(negedge CLK);
            end
        end
        host_buffer_write_enable = '0;
    endtask

    task automatic run_op(input string op_name, input string kind, input string a_file, input string b_file,
                          input string y_file, input string descriptor_file,
                          input string gain_file, input string offset_file);
        descriptor_t                        descriptor;
        logic [DESCRIPTOR_MEMORY_WIDTH-1:0] descriptor_word;
        int rows, cols, row_tiles, stride, output_base, region_end;
        int a_buffer, b_buffer, output_buffer, a_base, b_base;
        int mismatches     = 0;
        int first_mismatch = -1;
        int entries_seen   = 0;
        int overwritten    = 0;
        int cycles         = 0;
        bit finished       = 1'b0;
        bit two_operands;

        $readmemh(vector_file(descriptor_file), descriptor_mem);
        for (int word = 0; word < DESCRIPTOR_WORDS; word++) begin
            descriptor_word[32*word +: 32] = descriptor_mem[word];
        end
        descriptor    = descriptor_t'(descriptor_word[DESCRIPTOR_WIDTH-1:0]);
        rows          = int'(descriptor.rows_total);
        cols          = int'(descriptor.cols_total);
        row_tiles     = int'(descriptor.row_tiles);
        stride        = int'(descriptor.output_stride);
        output_base   = int'(descriptor.output_base);
        a_buffer      = int'(descriptor.activation_buffer_id);
        b_buffer      = int'(descriptor.weight_buffer_id);
        output_buffer = int'(descriptor.output_buffer_id);
        a_base        = int'(descriptor.activation_base);
        b_base        = int'(descriptor.weight_base);
        two_operands  = kind == "add";

        if (!descriptor.last_op || row_tiles != (rows + SYS_ROW - 1) / SYS_ROW
            || (kind == "add" && descriptor.opcode != OPCODE_ADD)
            || (kind == "gelu" && descriptor.opcode != OPCODE_GELU)
            || (kind == "layernorm" && descriptor.opcode != OPCODE_LAYERNORM)
            || (kind == "softmax" && descriptor.opcode != OPCODE_SOFTMAX)) begin
            $fatal(1, "FAIL: descriptor table %s does not fit this testbench", descriptor_file);
        end
        if (rows*cols > VECTOR_WORDS_MAX || output_base + row_tiles*stride > BUFFER_DEPTH) begin
            $fatal(1, "FAIL: op %s is too large for this testbench", op_name);
        end
        $readmemh(vector_file(a_file), a_mem, 0, rows*cols - 1);
        if (two_operands) begin
            $readmemh(vector_file(b_file), b_mem, 0, rows*cols - 1);
        end
        $readmemh(vector_file(y_file), expected_mem, 0, rows*cols - 1);

        region_end = output_base + row_tiles*stride + REGION_SLACK;
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

        // output region: known pattern in every bank, then the operands on top
        host_buffer_id = BUFFER_ID_WIDTH'(output_buffer);
        for (int address = 0; address < region_end; address++) begin
            host_address = ADDRESS_WIDTH'(address);
            for (int lane = 0; lane < LANES; lane++) begin
                host_buffer_write_enable[lane] = 1'b1;
                host_buffer_write_data[lane]   = fill_pattern(address, lane);
            end
            @(negedge CLK);
        end
        host_buffer_write_enable = '0;
        load_tensor(a_buffer, a_base, rows, cols, 1'b0);
        if (two_operands) begin
            load_tensor(b_buffer, b_base, rows, cols, 1'b1);
        end
        // LayerNorm: gain and offset of column c at bias_base + c
        if (kind == "layernorm") begin
            $readmemh(vector_file(gain_file), gain_mem, 0, cols - 1);
            $readmemh(vector_file(offset_file), offset_mem, 0, cols - 1);
            for (int column = 0; column < cols; column++) begin
                host_address                = ADDRESS_WIDTH'(int'(descriptor.bias_base) + column);
                host_layernorm_write_enable = 1'b1;
                host_layernorm_write_gain   = gain_mem[column];
                host_layernorm_write_offset = offset_mem[column];
                @(negedge CLK);
            end
            host_layernorm_write_enable = 1'b0;
        end
        @(negedge CLK);

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

        // read the whole output region back
        host_buffer_id = BUFFER_ID_WIDTH'(output_buffer);
        for (int address = 0; address < region_end; address++) begin
            host_address = ADDRESS_WIDTH'(address);
            repeat (vit_config_pkg::BUFFER_READ_LATENCY) @(negedge CLK);
            for (int lane = 0; lane < LANES; lane++) begin
                automatic int relative = address - output_base;
                automatic int tile     = (relative >= 0) ? relative / stride : -1;
                automatic int column   = (relative >= 0) ? relative % stride : -1;
                automatic int row      = tile*SYS_ROW + lane;
                automatic bit valid    = relative >= 0 && lane < SYS_ROW && row < rows && column < cols;
                automatic logic [DATA_WIDTH-1:0] untouched = fill_pattern(address, lane);
                // an operand loaded into the output buffer replaces the pattern
                if (a_buffer == output_buffer && address >= a_base && address < a_base + row_tiles*cols
                    && lane < SYS_ROW && ((address - a_base) / cols)*SYS_ROW + lane < rows) begin
                    untouched = a_mem[(((address - a_base) / cols)*SYS_ROW + lane)*cols + (address - a_base) % cols];
                end
                if (two_operands && b_buffer == output_buffer && address >= b_base
                    && address < b_base + row_tiles*cols
                    && lane < SYS_ROW && ((address - b_base) / cols)*SYS_ROW + lane < rows) begin
                    untouched = b_mem[(((address - b_base) / cols)*SYS_ROW + lane)*cols + (address - b_base) % cols];
                end
                if (valid) begin
                    entries_seen++;
                    if (host_buffer_read_data[lane] !== expected_mem[row*cols + column]) begin
                        if (first_mismatch < 0) begin
                            first_mismatch = row*cols + column;
                            $display("  %s first mismatch at index %0d (row %0d, col %0d): got %h, expected %h",
                                     op_name, first_mismatch, row, column,
                                     host_buffer_read_data[lane], expected_mem[row*cols + column]);
                        end
                        mismatches++;
                    end
                end
                else if (host_buffer_read_data[lane] !== untouched) begin
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
        entries_ok    += entries_seen - mismatches;
        entries_wrong += mismatches;
        if (finished && mismatches == 0 && overwritten == 0 && entries_seen == rows*cols) begin
            ops_passed++;
            $display("  %-24s PASS  %-9s %0dx%0d  buffers %0d, %0d -> %0d  %0d of %0d entries  cycles %0d",
                     op_name, kind, rows, cols, a_buffer, b_buffer, output_buffer, rows*cols, rows*cols, busy_cycles);
        end
        else begin
            ops_failed++;
            $display("  %-24s FAIL  %-9s %0dx%0d  %0d of %0d entries wrong (%0d checked), first mismatch index %0d, %0d words written outside the result",
                     op_name, kind, rows, cols, mismatches, rows*cols, entries_seen, first_mismatch, overwritten);
        end
    endtask

    initial begin
        int    manifest;
        int    fields;
        string op_name, kind, a_file, b_file, y_file, descriptor_file, extra_1, extra_2;

        $display("tb_matmul_engine_nonlinear: SYS_ROW=%0d SYS_COL=%0d", SYS_ROW, SYS_COL);
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
        repeat (3) @(negedge CLK);
        reset = 1'b0;

        manifest = $fopen(vector_file("nonlinear_ops.txt"), "r");
        while (!$feof(manifest)) begin
            fields = $fscanf(manifest, "%s %s %s %s %s %s %s %s\n",
                             op_name, kind, a_file, b_file, y_file, descriptor_file, extra_1, extra_2);
            if (fields == 8 && (kind == "add" || kind == "gelu" || kind == "layernorm" || kind == "softmax")) begin
                run_op(op_name, kind, a_file, b_file, y_file, descriptor_file, extra_1, extra_2);
            end
        end
        $fclose(manifest);

        if (ops_failed == 0 && ops_passed > 0) begin
            $display("PASS: %0d of %0d row ops, %0d of %0d entries correct",
                     ops_passed, ops_passed, entries_ok, entries_ok + entries_wrong);
        end
        else begin
            $display("FAIL: %0d of %0d row ops failed, %0d of %0d entries wrong",
                     ops_failed, ops_passed + ops_failed, entries_wrong, entries_ok + entries_wrong);
        end
        $finish;
    end

endmodule
