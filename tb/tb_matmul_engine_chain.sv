`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for matmul_engine running a chain of ops: one
// attention head with DeiT-Tiny dimensions and the softmax left out.
//
//   op 0  Q = rq(X @ Wq + bq)   197x192 @ 192x64   column-select
//   op 1  K = rq(X @ Wk + bk)   197x192 @ 192x64   column-select
//   op 2  V = rq(X @ Wv + bv)   197x192 @ 192x64   row-select
//   op 3  S = rq(Q @ K^T)       197x64  @ 64x197   column-select, no bias
//   op 4  O = rq(S @ V)         197x197 @ 197x64   column-select, no bias,
//                                                  into a 197x192 tensor
// Each op reads the buffers earlier ops wrote. Wq, Wk and Wv are column
// slices of one stored 192x576 weight, picked with weight_base and bias_base.
//
// The testbench writes the five-op table, loads X, the weight and the bias
// through the host port, fills every output region with a known pattern,
// starts the engine once and then checks every stage against its own golden
// file. Words of an output region outside the result must keep the pattern.
// It also reports the cycles each op took.
//
// Vectors come from golden/gen_vectors.py:
//   attn_cfg.hex    32-bit words: tokens, embed, head dimension, weight columns
//   attn_X.hex      INT8   X[token][k]
//   attn_W.hex      INT8   W[k][col]
//   attn_bias.hex   INT32  bias[col]
//   attn_Q.hex, attn_K.hex, attn_V.hex, attn_S.hex, attn_O.hex   INT8
//   attn_desc_r<SYS_ROW>c<SYS_COL>.hex   descriptor_t x 5, 32-bit words
// Tables exist for 4x4 and 8x8 arrays.
//////////////////////////////////////////////////////////////////////////////////

module tb_matmul_engine_chain #(
    parameter int    SYS_ROW    = 4,
    parameter int    SYS_COL    = 4,
    parameter string VECTOR_DIR = "../golden/vectors/"
);

    import vit_config_pkg::descriptor_t;
    import vit_config_pkg::DRAIN_COLUMN_SELECT;

    localparam int DATA_WIDTH              = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH               = vit_config_pkg::ACC_WIDTH;
    localparam int BUFFER_ID_WIDTH         = vit_config_pkg::BUFFER_ID_WIDTH;
    localparam int ADDRESS_WIDTH           = vit_config_pkg::ADDRESS_WIDTH;
    localparam int DESCRIPTOR_COUNT        = vit_config_pkg::DESCRIPTOR_COUNT;
    localparam int DESCRIPTOR_WORDS        = vit_config_pkg::DESCRIPTOR_WORDS;
    localparam int DESCRIPTOR_WIDTH        = vit_config_pkg::DESCRIPTOR_WIDTH;
    localparam int DESCRIPTOR_MEMORY_WIDTH = vit_config_pkg::DESCRIPTOR_MEMORY_WIDTH;
    localparam int DESCRIPTOR_INDEX_WIDTH  = $clog2(DESCRIPTOR_COUNT);

    localparam int LANES = (SYS_ROW > SYS_COL) ? SYS_ROW : SYS_COL;

    // words per bank: buffer 1 holds the whole 192x576 weight, the rest hold
    // one activation tensor each
    localparam int WEIGHT_BUFFER_ID     = 1;
    localparam int WEIGHT_BUFFER_DEPTH  = 32768;
    localparam int BUFFER_DEPTH         = 16384;

    localparam int OP_COUNT         = 5;
    localparam int FIRST_DESCRIPTOR = 7;  // the table does not have to start at entry 0
    localparam int VECTOR_WORDS_MAX = 131072;
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
        .SYS_ROW      (SYS_ROW),
        .SYS_COL      (SYS_COL),
        .BUFFER_DEPTH ({{6{32'(BUFFER_DEPTH)}}, 32'(WEIGHT_BUFFER_DEPTH), 32'(BUFFER_DEPTH)})
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

    logic [31:0]           cfg_mem[4];
    logic [31:0]           descriptor_mem[OP_COUNT*DESCRIPTOR_WORDS];
    logic [DATA_WIDTH-1:0] activation_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] weight_mem[VECTOR_WORDS_MAX];
    logic [ACC_WIDTH-1:0]  bias_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] expected_mem[VECTOR_WORDS_MAX];

    descriptor_t descriptors[OP_COUNT];

    int op_cycles[OP_COUNT];
    int total_cycles;

    int stages_passed = 0;
    int stages_failed = 0;
    int entries_ok    = 0;
    int entries_wrong = 0;

    // cycles the engine spends on each descriptor, from fetch to end of drain
    always @(posedge CLK) begin
        if (busy) begin
            total_cycles++;
            if (int'(dut.descriptor_index) >= FIRST_DESCRIPTOR && int'(dut.descriptor_index) < FIRST_DESCRIPTOR + OP_COUNT) begin
                op_cycles[int'(dut.descriptor_index) - FIRST_DESCRIPTOR]++;
            end
        end
    end

    function automatic string vector_file(input string suffix);
        string name;
        int    fd;
        name = $sformatf("%sattn_%s.hex", VECTOR_DIR, suffix);
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

    function automatic int buffer_depth(input int buffer_id);
        return (buffer_id == WEIGHT_BUFFER_ID) ? WEIGHT_BUFFER_DEPTH : BUFFER_DEPTH;
    endfunction

    // one past the last word an op's output can touch, plus some slack
    function automatic int region_end(input descriptor_t descriptor);
        int bank_rows;
        int last;
        bank_rows = (descriptor.drain_mode == DRAIN_COLUMN_SELECT) ? int'(descriptor.row_tiles) : int'(descriptor.col_tiles);
        last      = int'(descriptor.output_base) + bank_rows*int'(descriptor.output_stride) + REGION_SLACK;
        return (last > buffer_depth(int'(descriptor.output_buffer_id))) ? buffer_depth(int'(descriptor.output_buffer_id)) : last;
    endfunction

    task automatic fill_region(input descriptor_t descriptor);
        host_buffer_id = descriptor.output_buffer_id;
        for (int address = 0; address < region_end(descriptor); address++) begin
            host_address = address;
            for (int lane = 0; lane < LANES; lane++) begin
                host_buffer_write_enable[lane] = 1'b1;
                host_buffer_write_data[lane]   = fill_pattern(address, lane);
            end
            @(negedge CLK);
        end
        host_buffer_write_enable = '0;
    endtask

    // read an op's output region back and compare it with its golden file
    task automatic check_stage(input int op_index, input string stage_name);
        descriptor_t descriptor;
        int rows, cols, stride, output_base;
        int mismatches     = 0;
        int first_mismatch = -1;
        int entries_seen   = 0;
        int overwritten    = 0;

        descriptor  = descriptors[op_index];
        rows        = int'(descriptor.rows_total);
        cols        = int'(descriptor.cols_total);
        stride      = int'(descriptor.output_stride);
        output_base = int'(descriptor.output_base);
        $readmemh(vector_file(stage_name), expected_mem, 0, rows*cols - 1);

        host_buffer_id = descriptor.output_buffer_id;
        for (int address = 0; address < region_end(descriptor); address++) begin
            host_address = address;
            @(negedge CLK);
            for (int lane = 0; lane < LANES; lane++) begin
                automatic int relative = address - output_base;
                automatic int bank_row = (relative >= 0) ? relative / stride : -1;
                automatic int position = (relative >= 0) ? relative % stride : -1;
                automatic int row;
                automatic int col;
                automatic bit valid;
                if (descriptor.drain_mode == DRAIN_COLUMN_SELECT) begin
                    // row bank `lane`: stored row bank_row, element position
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
                            $display("  op %0d %s first mismatch at index %0d (row %0d, col %0d): got %h, expected %h",
                                     op_index, stage_name, first_mismatch, row, col,
                                     host_buffer_read_data[lane], expected_mem[row*cols + col]);
                        end
                        mismatches++;
                    end
                end
                else if (host_buffer_read_data[lane] !== fill_pattern(address, lane)) begin
                    if (overwritten == 0) begin
                        $display("  op %0d %s wrote outside the result at address %0d, bank %0d", op_index, stage_name, address, lane);
                    end
                    overwritten++;
                end
            end
        end

        if (entries_seen != rows*cols) begin
            $display("  op %0d %s checked %0d entries, expected %0d", op_index, stage_name, entries_seen, rows*cols);
        end
        entries_ok    += entries_seen - mismatches;
        entries_wrong += mismatches;
        if (mismatches == 0 && overwritten == 0 && entries_seen == rows*cols) begin
            stages_passed++;
            $display("  op %0d %s PASS: %0dx%0d @ %0dx%0d, %0d of %0d entries correct, 0 words written outside the result, %0d cycles",
                     op_index, stage_name, rows, int'(descriptor.inner_dim), int'(descriptor.inner_dim), cols,
                     rows*cols, rows*cols, op_cycles[op_index]);
        end
        else begin
            stages_failed++;
            $display("  op %0d %s FAIL: %0d of %0d entries wrong, first mismatch index %0d, %0d words written outside the result, %0d cycles",
                     op_index, stage_name, mismatches, rows*cols, first_mismatch, overwritten, op_cycles[op_index]);
        end
    endtask

    initial begin
        int tokens, embed, weight_cols;
        int row_tiles, weight_col_tiles;
        int cycles;
        bit finished;
        logic [DESCRIPTOR_MEMORY_WIDTH-1:0] descriptor_word;

        $display("tb_matmul_engine_chain: SYS_ROW=%0d SYS_COL=%0d", SYS_ROW, SYS_COL);
        reset                        = 1'b1;
        start                        = 1'b0;
        start_index                  = DESCRIPTOR_INDEX_WIDTH'(FIRST_DESCRIPTOR);
        host_descriptor_index        = '0;
        host_descriptor_write_enable = '0;
        host_descriptor_write_data   = '0;
        host_buffer_id               = '0;
        host_address                 = '0;
        host_buffer_write_enable     = '0;
        host_buffer_write_data       = '0;
        host_bias_write_enable       = '0;
        host_bias_write_data         = '0;
        total_cycles                 = 0;
        for (int i = 0; i < OP_COUNT; i++) begin
            op_cycles[i] = 0;
        end

        $readmemh(vector_file("cfg"), cfg_mem);
        tokens      = cfg_mem[0];
        embed       = cfg_mem[1];
        weight_cols = cfg_mem[3];
        $readmemh(vector_file($sformatf("desc_r%0dc%0d", SYS_ROW, SYS_COL)), descriptor_mem);
        $readmemh(vector_file("X"), activation_mem, 0, tokens*embed - 1);
        $readmemh(vector_file("W"), weight_mem, 0, embed*weight_cols - 1);
        $readmemh(vector_file("bias"), bias_mem, 0, weight_cols - 1);
        for (int i = 0; i < OP_COUNT; i++) begin
            for (int word = 0; word < DESCRIPTOR_WORDS; word++) begin
                descriptor_word[32*word +: 32] = descriptor_mem[i*DESCRIPTOR_WORDS + word];
            end
            descriptors[i] = descriptor_t'(descriptor_word[DESCRIPTOR_WIDTH-1:0]);
        end
        if (int'(descriptors[0].rows_total) != tokens || int'(descriptors[0].inner_dim) != embed
            || int'(descriptors[0].weight_buffer_id) != WEIGHT_BUFFER_ID || !descriptors[OP_COUNT-1].last_op) begin
            $fatal(1, "FAIL: descriptor table attn_desc_r%0dc%0d does not match the case", SYS_ROW, SYS_COL);
        end
        row_tiles        = (tokens + SYS_ROW - 1) / SYS_ROW;
        weight_col_tiles = (weight_cols + SYS_COL - 1) / SYS_COL;

        repeat (3) @(negedge CLK);
        reset = 1'b0;

        // descriptor table
        @(negedge CLK);
        for (int i = 0; i < OP_COUNT; i++) begin
            host_descriptor_index = DESCRIPTOR_INDEX_WIDTH'(FIRST_DESCRIPTOR + i);
            for (int word = 0; word < DESCRIPTOR_WORDS; word++) begin
                host_descriptor_write_enable       = '0;
                host_descriptor_write_enable[word] = 1'b1;
                host_descriptor_write_data         = descriptor_mem[i*DESCRIPTOR_WORDS + word];
                @(negedge CLK);
            end
        end
        host_descriptor_write_enable = '0;

        // X[token][k] goes to row bank token % SYS_ROW at (token / SYS_ROW)*embed + k
        host_buffer_id = descriptors[0].activation_buffer_id;
        for (int tile = 0; tile < row_tiles; tile++) begin
            for (int k = 0; k < embed; k++) begin
                host_address = tile*embed + k;
                for (int lane = 0; lane < LANES; lane++) begin
                    automatic int row = tile*SYS_ROW + lane;
                    host_buffer_write_enable[lane] = lane < SYS_ROW && row < tokens;
                    host_buffer_write_data[lane]   = (lane < SYS_ROW && row < tokens) ? activation_mem[row*embed + k] : '0;
                end
                @(negedge CLK);
            end
        end

        // the whole weight: W[k][col] goes to column bank col % SYS_COL at (col / SYS_COL)*embed + k
        host_buffer_id = BUFFER_ID_WIDTH'(WEIGHT_BUFFER_ID);
        for (int tile = 0; tile < weight_col_tiles; tile++) begin
            for (int k = 0; k < embed; k++) begin
                host_address = tile*embed + k;
                for (int lane = 0; lane < LANES; lane++) begin
                    automatic int col = tile*SYS_COL + lane;
                    host_buffer_write_enable[lane] = lane < SYS_COL && col < weight_cols;
                    host_buffer_write_data[lane]   = (lane < SYS_COL && col < weight_cols) ? weight_mem[k*weight_cols + col] : '0;
                end
                @(negedge CLK);
            end
        end
        host_buffer_write_enable = '0;

        // the whole bias: bias[col] goes to column bank col % SYS_COL at col / SYS_COL
        for (int tile = 0; tile < weight_col_tiles; tile++) begin
            host_address = tile;
            for (int lane = 0; lane < SYS_COL; lane++) begin
                automatic int col = tile*SYS_COL + lane;
                host_bias_write_enable[lane] = col < weight_cols;
                host_bias_write_data[lane]   = (col < weight_cols) ? bias_mem[col] : '0;
            end
            @(negedge CLK);
        end
        host_bias_write_enable = '0;

        for (int i = 0; i < OP_COUNT; i++) begin
            fill_region(descriptors[i]);
        end

        // one start runs all five ops
        start = 1'b1;
        @(negedge CLK);
        start    = 1'b0;
        finished = 1'b0;
        cycles   = 0;
        while (!finished && cycles < TIMEOUT_CYCLES) begin
            @(posedge CLK);
            if (done) begin
                finished = 1'b1;
            end
            cycles++;
        end
        @(negedge CLK);
        if (!finished) begin
            $display("  timed out after %0d cycles", cycles);
        end

        check_stage(0, "Q");
        check_stage(1, "K");
        check_stage(2, "V");
        check_stage(3, "S");
        check_stage(4, "O");

        $display("  cycles per op: Q %0d, K %0d, V %0d, S %0d, O %0d, total %0d",
                 op_cycles[0], op_cycles[1], op_cycles[2], op_cycles[3], op_cycles[4], total_cycles);
        if (finished && stages_failed == 0) begin
            $display("PASS: %0dx%0d array, %0d of %0d stages, %0d of %0d entries correct, %0d cycles",
                     SYS_ROW, SYS_COL, stages_passed, OP_COUNT, entries_ok, entries_ok + entries_wrong, total_cycles);
        end
        else begin
            $display("FAIL: %0dx%0d array, %0d of %0d stages failed, %0d of %0d entries wrong",
                     SYS_ROW, SYS_COL, stages_failed, OP_COUNT, entries_wrong, entries_ok + entries_wrong);
        end
        $finish;
    end

endmodule
