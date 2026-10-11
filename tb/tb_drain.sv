`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for drain, together with the operand buffers, the
// stream controller, the systolic array and the requantizers.
//
// For each golden case and each drain mode it loads A, B and the bias, runs
// every tile and compares the INT8 output buffer with the golden <case>_Y.hex.
// Two checks run on every case:
//   - a monitor on the output buffer write port counts the writes per C entry
//     and flags any write to a lane, row or column outside the M x N matrix;
//     every entry must be written exactly once;
//   - the output buffer is read back through its read port and compared.
//
// Vectors come from golden/gen_vectors.py:
//   <case>_cfg.hex   32-bit words: M, K, N, scale, shift
//   <case>_A.hex     M*K  INT8   A[row][k]
//   <case>_B.hex     K*N  INT8   B[k][col]
//   <case>_bias.hex  N    INT32  bias[col]
//   <case>_Y.hex     M*N  INT8   clip(((A @ B + bias) * scale) >>> shift)
//
// The bias sits in an operand_buffer with ACC_WIDTH words: column bank c holds
// bias[c], bias[c + SYS_COL], ... and is addressed by the tile column.
//////////////////////////////////////////////////////////////////////////////////

module tb_drain #(
    parameter int    SYS_ROW    = 4,
    parameter int    SYS_COL    = 4,
    parameter string VECTOR_DIR = "../golden/vectors/"
);

    import vit_config_pkg::DRAIN_ROW_SELECT;
    import vit_config_pkg::DRAIN_COLUMN_SELECT;

    localparam int DATA_WIDTH  = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH   = vit_config_pkg::ACC_WIDTH;
    localparam int SCALE_WIDTH = vit_config_pkg::SCALE_WIDTH;
    localparam int SHIFT_WIDTH = vit_config_pkg::SHIFT_WIDTH;

    localparam int LANES = (SYS_ROW > SYS_COL) ? SYS_ROW : SYS_COL;

    // deep enough for mm_qkv on a 3x5 array
    localparam int BUFFER_DEPTH      = 16384;
    localparam int BIAS_BUFFER_DEPTH = 1024;

    localparam int ROWS_WIDTH         = $clog2(vit_config_pkg::ROWS_TOTAL_MAX + 1);
    localparam int INNER_WIDTH        = $clog2(vit_config_pkg::INNER_DIM_MAX + 1);
    localparam int COLS_WIDTH         = $clog2(vit_config_pkg::COLS_TOTAL_MAX + 1);
    localparam int ADDRESS_WIDTH      = $clog2(BUFFER_DEPTH);
    localparam int BIAS_ADDRESS_WIDTH = $clog2(BIAS_BUFFER_DEPTH);

    localparam int CASE_COUNT       = 3;  // mm_small, mm_ragged, mm_qkv
    localparam int VECTOR_WORDS_MAX = 65536;
    localparam int TIMEOUT_CYCLES   = 4000000;

    logic CLK = 1'b0;
    logic reset;
    logic start;
    logic drain_mode;

    logic [ROWS_WIDTH-1:0]         rows_total;
    logic [INNER_WIDTH-1:0]        inner_dim;
    logic [COLS_WIDTH-1:0]         cols_total;
    logic [ROWS_WIDTH-1:0]         row_tiles;
    logic [COLS_WIDTH-1:0]         col_tiles;
    logic signed [SCALE_WIDTH-1:0] scale;
    logic [SHIFT_WIDTH-1:0]        shift_value;

    logic [SYS_ROW-1:0]                    activation_write_enable;
    logic [SYS_ROW-1:0][ADDRESS_WIDTH-1:0] activation_write_address;
    logic [SYS_ROW-1:0][DATA_WIDTH-1:0]    activation_write_data;
    logic [ADDRESS_WIDTH-1:0]              activation_read_address;
    logic [SYS_ROW-1:0][DATA_WIDTH-1:0]    activation_read_data;
    logic [SYS_ROW-1:0]                    activation_lane_enable;

    logic [SYS_COL-1:0]                    weight_write_enable;
    logic [SYS_COL-1:0][ADDRESS_WIDTH-1:0] weight_write_address;
    logic [SYS_COL-1:0][DATA_WIDTH-1:0]    weight_write_data;
    logic [ADDRESS_WIDTH-1:0]              weight_read_address;
    logic [SYS_COL-1:0][DATA_WIDTH-1:0]    weight_read_data;
    logic [SYS_COL-1:0]                    weight_lane_enable;

    logic [SYS_COL-1:0]                         bias_write_enable;
    logic [SYS_COL-1:0][BIAS_ADDRESS_WIDTH-1:0] bias_write_address;
    logic [SYS_COL-1:0][ACC_WIDTH-1:0]          bias_write_data;
    logic [SYS_COL-1:0][ACC_WIDTH-1:0]          bias_read_data;

    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0]             activation_in;
    logic signed [SYS_COL-1:0][DATA_WIDTH-1:0]             weight_in;
    logic                                                  first_k;
    logic signed [SYS_ROW-1:0][SYS_COL-1:0][ACC_WIDTH-1:0] accumulator_out;

    logic [ROWS_WIDTH-1:0] tile_row;
    logic [COLS_WIDTH-1:0] tile_col;
    logic                  tile_done;
    logic                  all_tiles_done;

    logic [LANES-1:0]                        output_write_enable;
    logic [ADDRESS_WIDTH-1:0]                output_write_address;
    logic signed [LANES-1:0][DATA_WIDTH-1:0] output_write_data;
    logic [LANES-1:0][ADDRESS_WIDTH-1:0]     output_write_address_lanes;
    logic [ADDRESS_WIDTH-1:0]                output_read_address;
    logic [LANES-1:0][DATA_WIDTH-1:0]        output_read_data;
    logic                                    drain_busy;

    operand_buffer #(
        .SYS_LANES  (SYS_ROW),
        .BANK_DEPTH (BUFFER_DEPTH)
    ) activation_buffer (
        .CLK           (CLK),
        .write_enable  (activation_write_enable),
        .write_address (activation_write_address),
        .write_data    (activation_write_data),
        .read_address  (activation_read_address),
        .read_data     (activation_read_data)
    );

    operand_buffer #(
        .SYS_LANES  (SYS_COL),
        .BANK_DEPTH (BUFFER_DEPTH)
    ) weight_buffer (
        .CLK           (CLK),
        .write_enable  (weight_write_enable),
        .write_address (weight_write_address),
        .write_data    (weight_write_data),
        .read_address  (weight_read_address),
        .read_data     (weight_read_data)
    );

    operand_buffer #(
        .SYS_LANES  (SYS_COL),
        .BANK_DEPTH (BIAS_BUFFER_DEPTH),
        .DATA_WIDTH (ACC_WIDTH)
    ) bias_buffer (
        .CLK           (CLK),
        .write_enable  (bias_write_enable),
        .write_address (bias_write_address),
        .write_data    (bias_write_data),
        .read_address  (BIAS_ADDRESS_WIDTH'(tile_col)),
        .read_data     (bias_read_data)
    );

    matmul_stream_controller #(
        .SYS_ROW    (SYS_ROW),
        .SYS_COL    (SYS_COL),
        .BANK_DEPTH (BUFFER_DEPTH)
    ) controller (
        .CLK                     (CLK),
        .reset                   (reset),
        .start                   (start),
        .rows_total              (rows_total),
        .inner_dim               (inner_dim),
        .cols_total              (cols_total),
        .row_tiles               (row_tiles),
        .col_tiles               (col_tiles),
        .activation_read_address (activation_read_address),
        .weight_read_address     (weight_read_address),
        .activation_lane_enable  (activation_lane_enable),
        .weight_lane_enable      (weight_lane_enable),
        .first_k_out             (first_k),
        .tile_row                (tile_row),
        .tile_col                (tile_col),
        .tile_done               (tile_done),
        .all_tiles_done          (all_tiles_done)
    );

    // the lane enables select buffer data or zero
    always_comb begin
        for (int lane = 0; lane < SYS_ROW; lane++) begin
            activation_in[lane] = activation_lane_enable[lane] ? activation_read_data[lane] : '0;
        end
        for (int lane = 0; lane < SYS_COL; lane++) begin
            weight_in[lane] = weight_lane_enable[lane] ? weight_read_data[lane] : '0;
        end
    end

    systolic_array #(
        .SYS_ROW (SYS_ROW),
        .SYS_COL (SYS_COL)
    ) array (
        .CLK             (CLK),
        .activation_in   (activation_in),
        .weight_in       (weight_in),
        .first_k_in      (first_k),
        .accumulator_out (accumulator_out)
    );

    drain #(
        .SYS_ROW    (SYS_ROW),
        .SYS_COL    (SYS_COL),
        .BANK_DEPTH (BUFFER_DEPTH)
    ) dut (
        .CLK            (CLK),
        .reset          (reset),
        .drain_mode     (drain_mode),
        .tile_done      (tile_done),
        .tile_row       (tile_row),
        .tile_col       (tile_col),
        .rows_total     (rows_total),
        .cols_total     (cols_total),
        .accumulator_in (accumulator_out),
        .bias_in        (bias_read_data),
        .scale          (scale),
        .shift_value    (shift_value),
        .write_enable   (output_write_enable),
        .write_address  (output_write_address),
        .write_data     (output_write_data),
        .drain_busy     (drain_busy)
    );

    // one shared write address for every bank
    always_comb begin
        for (int lane = 0; lane < LANES; lane++) begin
            output_write_address_lanes[lane] = output_write_address;
        end
    end

    operand_buffer #(
        .SYS_LANES  (LANES),
        .BANK_DEPTH (BUFFER_DEPTH)
    ) output_buffer (
        .CLK           (CLK),
        .write_enable  (output_write_enable),
        .write_address (output_write_address_lanes),
        .write_data    (output_write_data),
        .read_address  (output_read_address),
        .read_data     (output_read_data)
    );

    always #5 CLK = ~CLK;

    logic [31:0]           cfg_mem[5];
    logic [DATA_WIDTH-1:0] activation_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] weight_mem[VECTOR_WORDS_MAX];
    logic [ACC_WIDTH-1:0]  bias_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] expected_mem[VECTOR_WORDS_MAX];
    int                    write_count[VECTOR_WORDS_MAX];

    int rows;
    int inner;
    int cols;
    int tiles_seen;
    int illegal_writes;
    bit finished;

    int runs_passed   = 0;
    int runs_failed   = 0;
    int entries_ok    = 0;
    int entries_wrong = 0;

    // write port monitor: which C entry does each lane write land on?
    always @(posedge CLK) begin
        for (int lane = 0; lane < LANES; lane++) begin
            if (output_write_enable[lane] === 1'b1) begin
                automatic int address = int'(output_write_address);
                automatic int row;
                automatic int col;
                automatic bit legal;
                if (drain_mode == DRAIN_COLUMN_SELECT) begin
                    // row bank `lane`, stored row address / N, element address % N
                    row   = (address / cols)*SYS_ROW + lane;
                    col   = address % cols;
                    legal = lane < SYS_ROW && row < rows;
                end
                else begin
                    // column bank `lane`, stored column address / M, element address % M
                    col   = (address / rows)*SYS_COL + lane;
                    row   = address % rows;
                    legal = lane < SYS_COL && col < cols;
                end
                if (legal) begin
                    write_count[row*cols + col]++;
                end
                else begin
                    illegal_writes++;
                end
            end
            else if (output_write_enable[lane] !== 1'b0 && !reset) begin
                illegal_writes++;
            end
        end
        if (tile_done) begin
            tiles_seen++;
        end
        if (all_tiles_done) begin
            finished = 1'b1;
        end
    end

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

    task automatic load_case(input string case_name);
        $readmemh(vector_file(case_name, "cfg"), cfg_mem);
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

        @(negedge CLK);
        rows_total  = rows;
        inner_dim   = inner;
        cols_total  = cols;
        row_tiles   = (rows + SYS_ROW - 1) / SYS_ROW;
        col_tiles   = (cols + SYS_COL - 1) / SYS_COL;
        scale       = cfg_mem[3][SCALE_WIDTH-1:0];
        shift_value = cfg_mem[4][SHIFT_WIDTH-1:0];
        if (int'(row_tiles)*inner > BUFFER_DEPTH || int'(col_tiles)*inner > BUFFER_DEPTH
            || int'(col_tiles)*rows > BUFFER_DEPTH || int'(row_tiles)*cols > BUFFER_DEPTH
            || int'(col_tiles) > BIAS_BUFFER_DEPTH) begin
            $fatal(1, "FAIL: case %s does not fit the testbench buffers", case_name);
        end

        // A[row][k] goes to bank row % SYS_ROW at address (row / SYS_ROW)*K + k
        for (int tile = 0; tile < int'(row_tiles); tile++) begin
            for (int k = 0; k < inner; k++) begin
                for (int lane = 0; lane < SYS_ROW; lane++) begin
                    automatic int row = tile*SYS_ROW + lane;
                    activation_write_enable[lane]  = row < rows;
                    activation_write_address[lane] = tile*inner + k;
                    activation_write_data[lane]    = (row < rows) ? activation_mem[row*inner + k] : '0;
                end
                @(negedge CLK);
            end
        end
        activation_write_enable = '0;

        // B[k][col] goes to bank col % SYS_COL at address (col / SYS_COL)*K + k,
        // bias[col] to bank col % SYS_COL at address col / SYS_COL
        for (int tile = 0; tile < int'(col_tiles); tile++) begin
            for (int k = 0; k < inner; k++) begin
                for (int lane = 0; lane < SYS_COL; lane++) begin
                    automatic int col = tile*SYS_COL + lane;
                    weight_write_enable[lane]  = col < cols;
                    weight_write_address[lane] = tile*inner + k;
                    weight_write_data[lane]    = (col < cols) ? weight_mem[k*cols + col] : '0;
                    bias_write_enable[lane]    = col < cols && k == 0;
                    bias_write_address[lane]   = tile;
                    bias_write_data[lane]      = (col < cols) ? bias_mem[col] : '0;
                end
                @(negedge CLK);
            end
        end
        weight_write_enable = '0;
        bias_write_enable   = '0;
    endtask

    task automatic run_mode(input string case_name, input logic mode);
        int    tiles_expected;
        int    mismatches     = 0;
        int    first_mismatch = -1;
        int    not_once       = 0;
        int    cycles         = 0;
        int    idle_cycles    = 0;
        string mode_name;
        bit    run_ok;

        mode_name      = (mode == DRAIN_COLUMN_SELECT) ? "column-select" : "row-select";
        tiles_expected = int'(row_tiles) * int'(col_tiles);

        @(negedge CLK);
        drain_mode = mode;
        for (int i = 0; i < rows*cols; i++) begin
            write_count[i] = 0;
        end
        tiles_seen     = 0;
        illegal_writes = 0;
        finished       = 1'b0;

        start = 1'b1;
        @(negedge CLK);
        start = 1'b0;
        // wait for the last tile, then for the drain to go idle
        while (idle_cycles < 4 && cycles < TIMEOUT_CYCLES) begin
            @(negedge CLK);
            cycles++;
            idle_cycles = (finished && !drain_busy) ? idle_cycles + 1 : 0;
        end

        for (int i = 0; i < rows*cols; i++) begin
            if (write_count[i] != 1) begin
                not_once++;
            end
        end

        // read the output buffer back in its bank layout
        if (mode == DRAIN_COLUMN_SELECT) begin
            for (int tile = 0; tile < int'(row_tiles); tile++) begin
                for (int col = 0; col < cols; col++) begin
                    output_read_address = tile*cols + col;
                    @(negedge CLK);
                    @(negedge CLK);
                    for (int lane = 0; lane < SYS_ROW; lane++) begin
                        automatic int row = tile*SYS_ROW + lane;
                        if (row < rows && output_read_data[lane] !== expected_mem[row*cols + col]) begin
                            if (first_mismatch < 0) begin
                                first_mismatch = row*cols + col;
                                $display("  %s %s first mismatch at index %0d (row %0d, col %0d): got %h, expected %h",
                                         case_name, mode_name, first_mismatch, row, col,
                                         output_read_data[lane], expected_mem[row*cols + col]);
                            end
                            mismatches++;
                        end
                    end
                end
            end
        end
        else begin
            for (int tile = 0; tile < int'(col_tiles); tile++) begin
                for (int row = 0; row < rows; row++) begin
                    output_read_address = tile*rows + row;
                    @(negedge CLK);
                    @(negedge CLK);
                    for (int lane = 0; lane < SYS_COL; lane++) begin
                        automatic int col = tile*SYS_COL + lane;
                        if (col < cols && output_read_data[lane] !== expected_mem[row*cols + col]) begin
                            if (first_mismatch < 0) begin
                                first_mismatch = row*cols + col;
                                $display("  %s %s first mismatch at index %0d (row %0d, col %0d): got %h, expected %h",
                                         case_name, mode_name, first_mismatch, row, col,
                                         output_read_data[lane], expected_mem[row*cols + col]);
                            end
                            mismatches++;
                        end
                    end
                end
            end
        end

        if (!finished) begin
            $display("  %s %s timed out after %0d cycles", case_name, mode_name, cycles);
        end
        if (tiles_seen != tiles_expected) begin
            $display("  %s %s saw %0d tile_done pulses, expected %0d", case_name, mode_name, tiles_seen, tiles_expected);
        end
        if (illegal_writes != 0) begin
            $display("  %s %s made %0d writes outside the matrix", case_name, mode_name, illegal_writes);
        end
        if (not_once != 0) begin
            $display("  %s %s left %0d entries not written exactly once", case_name, mode_name, not_once);
        end

        run_ok = finished && mismatches == 0 && tiles_seen == tiles_expected
                 && illegal_writes == 0 && not_once == 0;
        entries_ok    += rows*cols - mismatches;
        entries_wrong += mismatches;
        if (run_ok) begin
            runs_passed++;
            $display("  %s %s PASS: %0dx%0d @ %0dx%0d, %0d tiles, %0d of %0d Y entries correct, 0 writes outside the matrix",
                     case_name, mode_name, rows, inner, inner, cols, tiles_seen, rows*cols, rows*cols);
        end
        else begin
            runs_failed++;
            $display("  %s %s FAIL: %0d of %0d Y entries wrong, first mismatch index %0d",
                     case_name, mode_name, mismatches, rows*cols, first_mismatch);
        end
    endtask

    task automatic run_case(input string case_name);
        load_case(case_name);
        run_mode(case_name, DRAIN_ROW_SELECT);
        run_mode(case_name, DRAIN_COLUMN_SELECT);
    endtask

    initial begin
        $display("tb_drain: SYS_ROW=%0d SYS_COL=%0d", SYS_ROW, SYS_COL);
        reset                   = 1'b1;
        start                   = 1'b0;
        drain_mode              = DRAIN_ROW_SELECT;
        activation_write_enable = '0;
        weight_write_enable     = '0;
        bias_write_enable       = '0;
        output_read_address     = '0;
        repeat (3) @(negedge CLK);
        reset = 1'b0;

        run_case("mm_small");
        run_case("mm_ragged");
        run_case("mm_qkv");

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
