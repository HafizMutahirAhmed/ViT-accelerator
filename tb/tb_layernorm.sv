`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for layernorm.
//
// Runs every "layernorm" op of the golden forward pass: two per encoder
// layer on 197 rows and the final one on the class row. For each op it writes
// the gain and offset of every column into the constants memory, then streams
// the input the way the engine does it: LANES rows per cycle and, for each
// row tile, the statistics pass, a wait for pass_ready, and the output pass.
// valid_in drops now and then. Every valid output is compared with the golden
// INT8 tensor. The ROM is the one the design uses, rtl/rom/deit224/rsqrt_rom.hex.
//
// Vectors come from python/gen_deit_forward.py, in python/vectors/deit_forward:
//   nonlinear_ops.txt  one line per op: name, kind, x file, "-", y file,
//                      descriptor file, gain file, offset file
//   x, y               INT8 rows x cols, row-major
//   gain, offset       INT16 and INT32 per column
//   descriptor         rows_total, cols_total, bias_base = constants base,
//                      shift = output shift, inner_dim = epsilon
//////////////////////////////////////////////////////////////////////////////////

module tb_layernorm #(
    parameter int    LANES      = vit_config_pkg::SYS_ROW,
    parameter string VECTOR_DIR = "../../python/vectors/deit_forward/"
);

    import vit_config_pkg::descriptor_t;

    localparam int DATA_WIDTH       = vit_config_pkg::DATA_WIDTH;
    localparam int DESCRIPTOR_WORDS = vit_config_pkg::DESCRIPTOR_WORDS;
    localparam int DESCRIPTOR_WIDTH = vit_config_pkg::DESCRIPTOR_WIDTH;
    localparam int VECTOR_WORDS_MAX = 262144;
    localparam int GAIN_WIDTH       = vit_config_pkg::LAYERNORM_GAIN_WIDTH;
    localparam int OFFSET_WIDTH     = vit_config_pkg::LAYERNORM_OFFSET_WIDTH;

    logic                                    CLK = 1'b0;
    logic                                    reset;
    logic                                    valid_in;
    logic signed [LANES-1:0][DATA_WIDTH-1:0] x_in;
    logic                                    pass_in;
    logic                                    first_in;
    logic                                    last_in;
    logic                                    pass_ready;
    logic                                    constants_write_enable;
    logic [9:0]                              constants_write_address;
    logic signed [GAIN_WIDTH-1:0]            constants_write_gain;
    logic signed [OFFSET_WIDTH-1:0]          constants_write_offset;
    logic [vit_config_pkg::COLS_WIDTH-1:0]   row_length;
    logic [9:0]                              constants_base;
    logic [vit_config_pkg::SHIFT_WIDTH-1:0]  out_shift;
    logic [vit_config_pkg::INNER_WIDTH-1:0]  epsilon;
    logic                                    valid_out;
    logic signed [LANES-1:0][DATA_WIDTH-1:0] y_out;

    logic [DATA_WIDTH-1:0] x_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] expected_mem[VECTOR_WORDS_MAX];
    logic [31:0]           descriptor_mem[DESCRIPTOR_WORDS];
    logic [GAIN_WIDTH-1:0]   gain_mem[1024];
    logic [OFFSET_WIDTH-1:0] offset_mem[1024];

    int rows;
    int cols;
    int outputs_seen;
    int mismatches;
    int first_mismatch;

    int ops_passed    = 0;
    int ops_failed    = 0;
    int entries_ok    = 0;
    int entries_wrong = 0;

    layernorm #(
        .LANES (LANES)
    ) dut (
        .CLK                     (CLK),
        .reset                   (reset),
        .constants_write_enable  (constants_write_enable),
        .constants_write_address (constants_write_address),
        .constants_write_gain    (constants_write_gain),
        .constants_write_offset  (constants_write_offset),
        .row_length              (row_length),
        .constants_base          (constants_base),
        .out_shift               (out_shift),
        .epsilon                 (epsilon),
        .valid_in                (valid_in),
        .pass_in                 (pass_in),
        .first_in                (first_in),
        .last_in                 (last_in),
        .x_in                    (x_in),
        .pass_ready              (pass_ready),
        .valid_out               (valid_out),
        .y_out                   (y_out)
    );

    always #5 CLK = ~CLK;

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

    // every valid output is the next column of the current row tile
    always @(posedge CLK) begin
        if (valid_out) begin
            automatic int tile   = outputs_seen / cols;
            automatic int column = outputs_seen % cols;
            for (int lane = 0; lane < LANES; lane++) begin
                automatic int row = tile*LANES + lane;
                if (row < rows) begin
                    if (y_out[lane] !== expected_mem[row*cols + column]) begin
                        if (first_mismatch < 0) begin
                            first_mismatch = row*cols + column;
                            $display("  first mismatch at index %0d (row %0d, col %0d): got %h, expected %h",
                                     first_mismatch, row, column, y_out[lane], expected_mem[row*cols + column]);
                        end
                        mismatches++;
                    end
                end
            end
            outputs_seen++;
        end
    end

    task automatic run_op(input string op_name, input string x_file, input string y_file,
                          input string descriptor_file, input string gain_file, input string offset_file);
        descriptor_t                    descriptor;
        logic [32*DESCRIPTOR_WORDS-1:0] descriptor_word;
        int                             row_tiles;
        int                             waited = 0;

        $readmemh(vector_file(descriptor_file), descriptor_mem);
        for (int word = 0; word < DESCRIPTOR_WORDS; word++) begin
            descriptor_word[32*word +: 32] = descriptor_mem[word];
        end
        descriptor = descriptor_t'(descriptor_word[DESCRIPTOR_WIDTH-1:0]);
        rows       = descriptor.rows_total;
        cols       = descriptor.cols_total;
        row_tiles  = (rows + LANES - 1) / LANES;
        if (rows*cols > VECTOR_WORDS_MAX) begin
            $fatal(1, "FAIL: op %s is larger than VECTOR_WORDS_MAX", op_name);
        end
        $readmemh(vector_file(x_file), x_mem, 0, rows*cols - 1);
        $readmemh(vector_file(y_file), expected_mem, 0, rows*cols - 1);

        outputs_seen   = 0;
        mismatches     = 0;
        first_mismatch = -1;
        row_length     = descriptor.cols_total;
        constants_base = descriptor.bias_base;
        out_shift      = descriptor.shift;
        epsilon        = descriptor.inner_dim;

        $readmemh(vector_file(gain_file), gain_mem, 0, cols - 1);
        $readmemh(vector_file(offset_file), offset_mem, 0, cols - 1);
        for (int column = 0; column < cols; column++) begin
            constants_write_enable  = 1'b1;
            constants_write_address = 10'(int'(constants_base) + column);
            constants_write_gain    = gain_mem[column];
            constants_write_offset  = offset_mem[column];
            @(negedge CLK);
        end
        constants_write_enable = 1'b0;
        @(negedge CLK);

        for (int tile = 0; tile < row_tiles; tile++) begin
            for (int pass = 0; pass < 2; pass++) begin
                if (pass == 1) begin
                    // like row_op_reader: the output pass starts when the statistics are ready
                    while (!pass_ready) begin
                        @(negedge CLK);
                        waited++;
                        if (waited > 1000) begin
                            $fatal(1, "FAIL: %s pass_ready never came", op_name);
                        end
                    end
                    waited = 0;
                end
                for (int column = 0; column < cols; column++) begin
                    // a gap in the stream every 37th word
                    if ((tile*cols + column) % 37 == 36) begin
                        valid_in = 1'b0;
                        x_in     = 'x;
                        @(negedge CLK);
                    end
                    valid_in = 1'b1;
                    pass_in  = pass[0];
                    first_in = column == 0;
                    last_in  = column == cols - 1;
                    for (int lane = 0; lane < LANES; lane++) begin
                        automatic int row = tile*LANES + lane;
                        x_in[lane] = (row < rows) ? x_mem[row*cols + column] : '0;
                    end
                    @(negedge CLK);
                end
                valid_in = 1'b0;
                first_in = 1'b0;
                last_in  = 1'b0;
                // the reader sees the last data of a pass before it looks at pass_ready
                repeat (2) @(negedge CLK);
            end
        end
        repeat (dut.LATENCY + 2) @(negedge CLK);

        entries_ok    += rows*cols - mismatches;
        entries_wrong += mismatches;
        if (mismatches == 0 && outputs_seen == row_tiles*cols) begin
            ops_passed++;
            $display("  %-16s PASS  %0dx%0d  constants at %0d  shift %0d  epsilon %0d  %0d of %0d entries",
                     op_name, rows, cols, constants_base, out_shift, epsilon, rows*cols, rows*cols);
        end
        else begin
            ops_failed++;
            $display("  %-16s FAIL  %0dx%0d  %0d of %0d entries wrong, first mismatch index %0d, %0d of %0d outputs seen",
                     op_name, rows, cols, mismatches, rows*cols, first_mismatch, outputs_seen, row_tiles*cols);
        end
    endtask

    initial begin
        int    manifest;
        int    fields;
        string op_name, kind, x_file, b_file, y_file, descriptor_file, extra_1, extra_2;

        $display("tb_layernorm: LANES=%0d LATENCY=%0d", LANES, dut.LATENCY);
        reset                  = 1'b1;
        valid_in               = 1'b0;
        pass_in                = 1'b0;
        first_in               = 1'b0;
        last_in                = 1'b0;
        constants_write_enable = 1'b0;
        x_in                   = '0;
        repeat (3) @(negedge CLK);
        reset = 1'b0;

        manifest = $fopen(vector_file("nonlinear_ops.txt"), "r");
        while (!$feof(manifest)) begin
            fields = $fscanf(manifest, "%s %s %s %s %s %s %s %s\n",
                             op_name, kind, x_file, b_file, y_file, descriptor_file, extra_1, extra_2);
            if (fields == 8 && kind == "layernorm") begin
                run_op(op_name, x_file, y_file, descriptor_file, extra_1, extra_2);
            end
        end
        $fclose(manifest);

        if (ops_failed == 0 && ops_passed > 0) begin
            $display("PASS: %0d of %0d layernorm ops, %0d of %0d entries correct",
                     ops_passed, ops_passed, entries_ok, entries_ok + entries_wrong);
        end
        else begin
            $display("FAIL: %0d of %0d layernorm ops failed, %0d of %0d entries wrong",
                     ops_failed, ops_passed + ops_failed, entries_wrong, entries_ok + entries_wrong);
        end
        $finish;
    end

endmodule
