`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for matmul_engine, driven only through its ports.
//
// For each golden case and each drain mode it writes a one-op descriptor table,
// loads A, B and the bias through the host port into the buffers and at the
// bases the descriptor names, fills the output region with a known pattern,
// runs the engine and reads the output region back through the host port:
//   - every Y entry must equal the golden <case>_Y.hex;
//   - every other word of the region (lanes, rows and columns past the matrix
//     edge, and the gaps left by a padded output_stride) must still hold the
//     fill pattern, so nothing outside the matrix was written.
//
// Vectors come from golden/gen_vectors.py:
//   <case>_cfg.hex   32-bit words: M, K, N, scale, shift
//   <case>_A.hex     M*K  INT8   A[row][k]
//   <case>_B.hex     K*N  INT8   B[k][col]
//   <case>_bias.hex  N    INT32  bias[col]
//   <case>_Y.hex     M*N  INT8   requantized output
//   <case>_desc_r<SYS_ROW>c<SYS_COL>_<row|col>.hex  descriptor_t, 32-bit words
// Tables exist for 4x4 and 8x8 arrays.
//////////////////////////////////////////////////////////////////////////////////

module tb_matmul_engine #(
    parameter int    SYS_ROW    = 4,
    parameter int    SYS_COL    = 4,
    parameter string VECTOR_DIR = "../golden/vectors/"
);

    import vit_config_pkg::descriptor_t;
    import vit_config_pkg::DRAIN_COLUMN_SELECT;
    import vit_config_pkg::DRAIN_ROW_SELECT;
    import vit_config_pkg::OPCODE_MATMUL;

    localparam int DATA_WIDTH              = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH               = vit_config_pkg::ACC_WIDTH;
    localparam int BUFFER_COUNT            = vit_config_pkg::BUFFER_COUNT;
    localparam int BUFFER_ID_WIDTH         = vit_config_pkg::BUFFER_ID_WIDTH;
    localparam int ADDRESS_WIDTH           = vit_config_pkg::ADDRESS_WIDTH;
    localparam int BIAS_DEPTH              = vit_config_pkg::BIAS_DEPTH;
    localparam int DESCRIPTOR_COUNT        = vit_config_pkg::DESCRIPTOR_COUNT;
    localparam int DESCRIPTOR_WORDS        = vit_config_pkg::DESCRIPTOR_WORDS;
    localparam int DESCRIPTOR_WIDTH        = vit_config_pkg::DESCRIPTOR_WIDTH;
    localparam int DESCRIPTOR_MEMORY_WIDTH = vit_config_pkg::DESCRIPTOR_MEMORY_WIDTH;
    localparam int DESCRIPTOR_INDEX_WIDTH  = $clog2(DESCRIPTOR_COUNT);

    localparam int LANES = (SYS_ROW > SYS_COL) ? SYS_ROW : SYS_COL;

    // must match ENGINE_TB_BANK_DEPTH in golden/gen_vectors.py
    localparam int BUFFER_DEPTH = 16384;

    localparam int CASE_COUNT       = 6;  // mm_small, mm_ragged, mm_k1, mm_k2, mm_qkv, mm_nobias
    localparam int VECTOR_WORDS_MAX = 65536;
    localparam int TIMEOUT_CYCLES   = 4000000;
    localparam int REGION_SLACK     = 16; // extra words checked after the output region

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
        .host_bias_read_data          (host_bias_read_data)
    );

    always #5 CLK = ~CLK;

    logic [31:0]           cfg_mem[5];
    logic [31:0]           descriptor_mem[DESCRIPTOR_WORDS];
    logic [DATA_WIDTH-1:0] activation_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] weight_mem[VECTOR_WORDS_MAX];
    logic [ACC_WIDTH-1:0]  bias_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] expected_mem[VECTOR_WORDS_MAX];

    int runs_started  = 0;
    int runs_passed   = 0;
    int runs_failed   = 0;
    int entries_ok    = 0;
    int entries_wrong = 0;

    function automatic string vector_file(input string case_name, input string suffix);
        string name;
        int    fd;
        name = $sformatf("%s%s_%s.hex", VECTOR_DIR, case_name, suffix);
        fd = $fopen(name, "r");
        if (fd == 0) begin
            $fatal(1, "FAIL: cannot open golden vector file %s", name);
        end
        $fclose(fd);
        return name;
    endfunction

    // the word an untouched output location must still hold
    function automatic logic [DATA_WIDTH-1:0] fill_pattern(input int address, input int lane);
        return DATA_WIDTH'(8'h5A ^ (address * 7) ^ (lane * 37));
    endfunction

    task automatic run_mode(input string case_name, input logic mode);
        descriptor_t                        descriptor;
        logic [DESCRIPTOR_MEMORY_WIDTH-1:0] descriptor_word;
        string mode_name;
        string table_suffix;
        int    rows, inner, cols, row_tiles, col_tiles;
        int    bank_rows, stride, output_base, region_end;
        int    mismatches     = 0;
        int    first_mismatch = -1;
        int    entries_seen   = 0;
        int    overwritten    = 0;
        int    cycles         = 0;
        bit    finished       = 1'b0;
        bit    run_ok;

        mode_name    = (mode == DRAIN_COLUMN_SELECT) ? "column-select" : "row-select";
        table_suffix = $sformatf("desc_r%0dc%0d_%s", SYS_ROW, SYS_COL, (mode == DRAIN_COLUMN_SELECT) ? "col" : "row");

        $readmemh(vector_file(case_name, "cfg"), cfg_mem);
        $readmemh(vector_file(case_name, table_suffix), descriptor_mem);
        rows  = cfg_mem[0];
        inner = cfg_mem[1];
        cols  = cfg_mem[2];
        if (rows*inner > VECTOR_WORDS_MAX || inner*cols > VECTOR_WORDS_MAX || rows*cols > VECTOR_WORDS_MAX) begin
            $fatal(1, "FAIL: case %s is larger than VECTOR_WORDS_MAX", case_name);
        end
        $readmemh(vector_file(case_name, "A"), activation_mem, 0, rows*inner - 1);
        $readmemh(vector_file(case_name, "B"), weight_mem, 0, inner*cols - 1);
        $readmemh(vector_file(case_name, "bias"), bias_mem, 0, cols - 1);
        $readmemh(vector_file(case_name, "Y"), expected_mem, 0, rows*cols - 1);

        for (int word = 0; word < DESCRIPTOR_WORDS; word++) begin
            descriptor_word[32*word +: 32] = descriptor_mem[word];
        end
        descriptor = descriptor_t'(descriptor_word[DESCRIPTOR_WIDTH-1:0]);
        row_tiles   = int'(descriptor.row_tiles);
        col_tiles   = int'(descriptor.col_tiles);
        stride      = int'(descriptor.output_stride);
        output_base = int'(descriptor.output_base);

        // the table must describe this case, this array and this drain mode
        if (descriptor.opcode != OPCODE_MATMUL || int'(descriptor.rows_total) != rows
            || int'(descriptor.inner_dim) != inner || int'(descriptor.cols_total) != cols
            || row_tiles != (rows + SYS_ROW - 1) / SYS_ROW || col_tiles != (cols + SYS_COL - 1) / SYS_COL
            || descriptor.drain_mode != mode || !descriptor.last_op
            || descriptor.scale != cfg_mem[3][vit_config_pkg::SCALE_WIDTH-1:0]
            || descriptor.shift != cfg_mem[4][vit_config_pkg::SHIFT_WIDTH-1:0]) begin
            $fatal(1, "FAIL: descriptor table %s_%s does not match the case", case_name, table_suffix);
        end

        bank_rows  = (mode == DRAIN_COLUMN_SELECT) ? row_tiles : col_tiles;
        region_end = output_base + bank_rows*stride + REGION_SLACK;
        if (region_end > BUFFER_DEPTH) begin
            region_end = BUFFER_DEPTH;
        end

        // descriptor table: one entry, at a different index on every run
        @(negedge CLK);
        start_index           = DESCRIPTOR_INDEX_WIDTH'(runs_started);
        host_descriptor_index = DESCRIPTOR_INDEX_WIDTH'(runs_started);
        runs_started++;
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

        // run
        start = 1'b1;
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

        // read the whole output region back, one address per cycle
        host_buffer_id = descriptor.output_buffer_id;
        for (int address = 0; address < region_end; address++) begin
            host_address = address;
            @(negedge CLK);
            for (int lane = 0; lane < LANES; lane++) begin
                automatic int relative = address - output_base;
                automatic int bank_row = (relative >= 0) ? relative / stride : -1;
                automatic int position = (relative >= 0) ? relative % stride : -1;
                automatic int row;
                automatic int col;
                automatic bit valid;
                if (mode == DRAIN_COLUMN_SELECT) begin
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
                            $display("  %s %s first mismatch at index %0d (row %0d, col %0d): got %h, expected %h",
                                     case_name, mode_name, first_mismatch, row, col,
                                     host_buffer_read_data[lane], expected_mem[row*cols + col]);
                        end
                        mismatches++;
                    end
                end
                else if (host_buffer_read_data[lane] !== fill_pattern(address, lane)) begin
                    if (overwritten == 0) begin
                        $display("  %s %s wrote outside the matrix at address %0d, bank %0d", case_name, mode_name, address, lane);
                    end
                    overwritten++;
                end
            end
        end

        if (!finished) begin
            $display("  %s %s timed out after %0d cycles", case_name, mode_name, cycles);
        end
        if (entries_seen != rows*cols) begin
            $display("  %s %s checked %0d Y entries, expected %0d", case_name, mode_name, entries_seen, rows*cols);
        end

        run_ok = finished && mismatches == 0 && overwritten == 0 && entries_seen == rows*cols;
        entries_ok    += entries_seen - mismatches;
        entries_wrong += mismatches;
        if (run_ok) begin
            runs_passed++;
            $display("  %s %s PASS: %0dx%0d @ %0dx%0d, %0d of %0d Y entries correct, 0 words written outside the matrix",
                     case_name, mode_name, rows, inner, inner, cols, rows*cols, rows*cols);
        end
        else begin
            runs_failed++;
            $display("  %s %s FAIL: %0d of %0d Y entries wrong, first mismatch index %0d, %0d words written outside the matrix",
                     case_name, mode_name, mismatches, rows*cols, first_mismatch, overwritten);
        end
    endtask

    task automatic run_case(input string case_name);
        run_mode(case_name, DRAIN_ROW_SELECT);
        run_mode(case_name, DRAIN_COLUMN_SELECT);
    endtask

    initial begin
        $display("tb_matmul_engine: SYS_ROW=%0d SYS_COL=%0d", SYS_ROW, SYS_COL);
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

        run_case("mm_small");
        run_case("mm_ragged");
        run_case("mm_k1");
        run_case("mm_k2");
        run_case("mm_qkv");
        // bias_enable = 0: the stored bias must be ignored
        run_case("mm_nobias");

        if (runs_failed == 0) begin
            $display("PASS: %0dx%0d array, %0d of %0d runs (%0d cases x 2 drain modes), %0d of %0d Y entries correct",
                     SYS_ROW, SYS_COL, runs_passed, 2*CASE_COUNT, CASE_COUNT,
                     entries_ok, entries_ok + entries_wrong);
        end
        else begin
            $display("FAIL: %0dx%0d array, %0d of %0d runs failed, %0d of %0d Y entries wrong",
                     SYS_ROW, SYS_COL, runs_failed, 2*CASE_COUNT,
                     entries_wrong, entries_ok + entries_wrong);
        end
        $finish;
    end

endmodule
