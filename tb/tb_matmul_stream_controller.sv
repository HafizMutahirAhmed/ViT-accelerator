`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for matmul_stream_controller with two operand_buffer
// instances and the systolic_array.
//
// For each golden case it loads A into the activation buffer and B into the
// weight buffer through the bank write ports, starts the controller, captures
// accumulator_out at every tile_done and compares the assembled C with the
// INT32 golden <case>_C_acc.hex. It also checks the tile count, the tile
// coordinates and that accumulators of lanes past the matrix edge are zero.
//
// Vectors come from python/gen_vectors.py:
//   <case>_cfg.hex    32-bit words: M, K, N, scale, shift
//   <case>_A.hex      M*K  INT8   A[row][k]
//   <case>_B.hex      K*N  INT8   B[k][col]
//   <case>_C_acc.hex  M*N  INT32  A @ B, no bias
//
// The cases run back to back without clearing the buffers, so the second case
// sees stale words from the first one in the banks.
//////////////////////////////////////////////////////////////////////////////////

`include "vit_paths.svh"

module tb_matmul_stream_controller #(
    parameter int    SYS_ROW    = 4,
    parameter int    SYS_COL    = 4,
    parameter string VECTOR_DIR = {`VIT_ROOT, "python/vectors/"}
);

    localparam int DATA_WIDTH = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH  = vit_config_pkg::ACC_WIDTH;

    localparam int ROWS_WIDTH    = $clog2(vit_config_pkg::ROWS_TOTAL_MAX + 1);
    localparam int INNER_WIDTH   = $clog2(vit_config_pkg::INNER_DIM_MAX + 1);
    localparam int COLS_WIDTH    = $clog2(vit_config_pkg::COLS_TOTAL_MAX + 1);
    localparam int ADDRESS_WIDTH = $clog2(vit_config_pkg::BANK_DEPTH);

    localparam int CASE_COUNT       = 2;  // mm_small, mm_ragged
    localparam int VECTOR_WORDS_MAX = 4096;
    localparam int TIMEOUT_CYCLES   = 100000;

    logic CLK = 1'b0;
    logic reset;
    logic start;

    logic [ROWS_WIDTH-1:0]  rows_total;
    logic [INNER_WIDTH-1:0] inner_dim;
    logic [COLS_WIDTH-1:0]  cols_total;
    logic [ROWS_WIDTH-1:0]  row_tiles;
    logic [COLS_WIDTH-1:0]  col_tiles;

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

    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0]             activation_in;
    logic signed [SYS_COL-1:0][DATA_WIDTH-1:0]             weight_in;
    logic                                                  first_k;
    logic signed [SYS_ROW-1:0][SYS_COL-1:0][ACC_WIDTH-1:0] accumulator_out;

    logic [ROWS_WIDTH-1:0] tile_row;
    logic [COLS_WIDTH-1:0] tile_col;
    logic                  tile_done;
    logic                  all_tiles_done;

    operand_buffer #(
        .SYS_LANES (SYS_ROW)
    ) activation_buffer (
        .CLK           (CLK),
        .write_enable  (activation_write_enable),
        .write_address (activation_write_address),
        .write_data    (activation_write_data),
        .read_address  (activation_read_address),
        .read_data     (activation_read_data)
    );

    operand_buffer #(
        .SYS_LANES (SYS_COL)
    ) weight_buffer (
        .CLK           (CLK),
        .write_enable  (weight_write_enable),
        .write_address (weight_write_address),
        .write_data    (weight_write_data),
        .read_address  (weight_read_address),
        .read_data     (weight_read_data)
    );

    matmul_stream_controller #(
        .SYS_ROW (SYS_ROW),
        .SYS_COL (SYS_COL)
    ) controller (
        .CLK                     (CLK),
        .reset                   (reset),
        .start                   (start),
        .rows_total              (rows_total),
        .inner_dim               (inner_dim),
        .cols_total              (cols_total),
        .row_tiles               (row_tiles),
        .col_tiles               (col_tiles),
        .activation_base         ('0),
        .weight_base             ('0),
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

    always #5 CLK = ~CLK;

    logic [31:0]           cfg_mem[5];
    logic [DATA_WIDTH-1:0] activation_mem[VECTOR_WORDS_MAX];
    logic [DATA_WIDTH-1:0] weight_mem[VECTOR_WORDS_MAX];
    logic [ACC_WIDTH-1:0]  expected_mem[VECTOR_WORDS_MAX];
    logic [ACC_WIDTH-1:0]  assembled_mem[VECTOR_WORDS_MAX];

    int tiles_seen;
    int coordinate_errors;
    int masked_checked;
    int masked_nonzero;
    bit finished;

    int cases_passed    = 0;
    int cases_failed    = 0;
    int entries_ok      = 0;
    int entries_wrong   = 0;
    int tiles_total     = 0;

    // capture every accumulator on the clock edge that ends the tile_done cycle
    always @(posedge CLK) begin
        if (tile_done) begin
            if (int'(tile_row) != tiles_seen / int'(col_tiles) || int'(tile_col) != tiles_seen % int'(col_tiles)) begin
                coordinate_errors++;
            end
            for (int row = 0; row < SYS_ROW; row++) begin
                for (int col = 0; col < SYS_COL; col++) begin
                    automatic int global_row = int'(tile_row)*SYS_ROW + row;
                    automatic int global_col = int'(tile_col)*SYS_COL + col;
                    if (global_row < int'(rows_total) && global_col < int'(cols_total)) begin
                        assembled_mem[global_row*int'(cols_total) + global_col] = accumulator_out[row][col];
                    end
                    else begin
                        masked_checked++;
                        if (accumulator_out[row][col] !== '0) begin
                            masked_nonzero++;
                        end
                    end
                end
            end
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

    task automatic run_case(input string case_name);
        int rows, inner, cols, tiles_expected;
        int mismatches     = 0;
        int first_mismatch = -1;
        int cycles         = 0;
        bit case_ok;

        $readmemh(vector_file(case_name, "cfg"), cfg_mem);
        rows  = cfg_mem[0];
        inner = cfg_mem[1];
        cols  = cfg_mem[2];
        if (rows*inner > VECTOR_WORDS_MAX || inner*cols > VECTOR_WORDS_MAX || rows*cols > VECTOR_WORDS_MAX) begin
            $fatal(1, "FAIL: case %s is larger than VECTOR_WORDS_MAX", case_name);
        end
        $readmemh(vector_file(case_name, "A"), activation_mem, 0, rows*inner - 1);
        $readmemh(vector_file(case_name, "B"), weight_mem, 0, inner*cols - 1);
        $readmemh(vector_file(case_name, "C_acc"), expected_mem, 0, rows*cols - 1);

        @(negedge CLK);
        rows_total = rows;
        inner_dim  = inner;
        cols_total = cols;
        row_tiles  = (rows + SYS_ROW - 1) / SYS_ROW;
        col_tiles  = (cols + SYS_COL - 1) / SYS_COL;
        tiles_expected = int'(row_tiles) * int'(col_tiles);

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

        // B[k][col] goes to bank col % SYS_COL at address (col / SYS_COL)*K + k
        for (int tile = 0; tile < int'(col_tiles); tile++) begin
            for (int k = 0; k < inner; k++) begin
                for (int lane = 0; lane < SYS_COL; lane++) begin
                    automatic int col = tile*SYS_COL + lane;
                    weight_write_enable[lane]  = col < cols;
                    weight_write_address[lane] = tile*inner + k;
                    weight_write_data[lane]    = (col < cols) ? weight_mem[k*cols + col] : '0;
                end
                @(negedge CLK);
            end
        end
        weight_write_enable = '0;

        for (int i = 0; i < rows*cols; i++) begin
            assembled_mem[i] = 'x;
        end
        tiles_seen        = 0;
        coordinate_errors = 0;
        masked_checked    = 0;
        masked_nonzero    = 0;
        finished          = 1'b0;

        start = 1'b1;
        @(negedge CLK);
        start = 1'b0;
        while (!finished && cycles < TIMEOUT_CYCLES) begin
            @(negedge CLK);
            cycles++;
        end

        for (int i = 0; i < rows*cols; i++) begin
            if (assembled_mem[i] !== expected_mem[i]) begin
                if (first_mismatch < 0) begin
                    first_mismatch = i;
                    $display("  %s first mismatch at index %0d (row %0d, col %0d): got %h, expected %h",
                             case_name, i, i / cols, i % cols, assembled_mem[i], expected_mem[i]);
                end
                mismatches++;
            end
        end

        if (!finished) begin
            $display("  %s timed out after %0d cycles", case_name, cycles);
        end
        if (tiles_seen != tiles_expected) begin
            $display("  %s saw %0d tile_done pulses, expected %0d", case_name, tiles_seen, tiles_expected);
        end
        if (coordinate_errors != 0) begin
            $display("  %s had %0d wrong tile coordinates", case_name, coordinate_errors);
        end
        if (masked_nonzero != 0) begin
            $display("  %s had %0d of %0d out-of-range accumulators nonzero", case_name, masked_nonzero, masked_checked);
        end

        case_ok = finished && mismatches == 0 && tiles_seen == tiles_expected
                  && coordinate_errors == 0 && masked_nonzero == 0;
        entries_ok    += rows*cols - mismatches;
        entries_wrong += mismatches;
        tiles_total   += tiles_seen;
        if (case_ok) begin
            cases_passed++;
            $display("  %s PASS: %0dx%0d @ %0dx%0d, %0d tiles, %0d of %0d C entries correct, %0d out-of-range accumulators zero",
                     case_name, rows, inner, inner, cols, tiles_seen, rows*cols, rows*cols, masked_checked);
        end
        else begin
            cases_failed++;
            $display("  %s FAIL: %0d of %0d C entries wrong, first mismatch index %0d",
                     case_name, mismatches, rows*cols, first_mismatch);
        end
    endtask

    initial begin
        $display("tb_matmul_stream_controller: SYS_ROW=%0d SYS_COL=%0d", SYS_ROW, SYS_COL);
        reset                   = 1'b1;
        start                   = 1'b0;
        activation_write_enable = '0;
        weight_write_enable     = '0;
        repeat (3) @(negedge CLK);
        reset = 1'b0;

        run_case("mm_small");
        run_case("mm_ragged");

        if (cases_failed == 0) begin
            $display("PASS: %0dx%0d array, %0d of %0d cases, %0d tiles, %0d of %0d C entries correct",
                     SYS_ROW, SYS_COL, cases_passed, CASE_COUNT, tiles_total,
                     entries_ok, entries_ok + entries_wrong);
        end
        else begin
            $display("FAIL: %0dx%0d array, %0d of %0d cases failed, %0d of %0d C entries wrong",
                     SYS_ROW, SYS_COL, cases_failed, CASE_COUNT,
                     entries_wrong, entries_ok + entries_wrong);
        end
        $finish;
    end

endmodule
