`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for systolic_array (with pe and skew inside).
//
// For each K in K_CASES it feeds one SYS_ROW x SYS_COL output tile: K slices
// with first_k_in high on slice 0, then zeros for PIPELINE_LATENCY cycles, and
// compares every accumulator with the INT32 golden A @ B.
//
// Vectors come from python/gen_vectors.py:
//   tile_r<SYS_ROW>c<SYS_COL>_k<K>_A.hex      SYS_ROW*K  INT8   A[row][k]
//   tile_r<SYS_ROW>c<SYS_COL>_k<K>_B.hex      K*SYS_COL  INT8   B[k][col]
//   tile_r<SYS_ROW>c<SYS_COL>_k<K>_C_acc.hex  SYS_ROW*SYS_COL INT32
//
// The DUT is never reset and the cases run back to back, so the first case
// starts from unknown register contents and the others from a stale tile.
// Vectors exist for 2x2, 4x4, 3x5, 12x12 and 16x16 arrays.
//////////////////////////////////////////////////////////////////////////////////

module tb_systolic_array #(
    parameter int    SYS_ROW    = 4,
    parameter int    SYS_COL    = 4,
    parameter string VECTOR_DIR = "../../python/vectors/"
);

    localparam int DATA_WIDTH = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH  = vit_config_pkg::ACC_WIDTH;

    localparam int K_CASE_COUNT          = 4;
    localparam int K_CASES[K_CASE_COUNT] = '{1, 3, 8, 197};
    localparam int K_MAX                 = 197;

    logic                                                  CLK = 1'b0;
    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0]             activation_in;
    logic signed [SYS_COL-1:0][DATA_WIDTH-1:0]             weight_in;
    logic                                                  first_k_in;
    logic signed [SYS_ROW-1:0][SYS_COL-1:0][ACC_WIDTH-1:0] accumulator_out;

    logic [DATA_WIDTH-1:0] activation_mem[SYS_ROW*K_MAX];
    logic [DATA_WIDTH-1:0] weight_mem[K_MAX*SYS_COL];
    logic [ACC_WIDTH-1:0]  expected_mem[SYS_ROW*SYS_COL];

    int cases_passed       = 0;
    int cases_failed       = 0;
    int accumulators_ok    = 0;
    int accumulators_wrong = 0;

    systolic_array #(
        .SYS_ROW    (SYS_ROW),
        .SYS_COL    (SYS_COL),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) dut (
        .CLK             (CLK),
        .activation_in   (activation_in),
        .weight_in       (weight_in),
        .first_k_in      (first_k_in),
        .accumulator_out (accumulator_out)
    );

    always #5 CLK = ~CLK;

    function automatic string vector_file(input int k_dim, input string suffix);
        string name;
        int    fd;
        name = $sformatf("%stile_r%0dc%0d_k%0d_%s.hex", VECTOR_DIR, SYS_ROW, SYS_COL, k_dim, suffix);
        fd = $fopen(name, "r");
        if (fd == 0) begin
            $fatal(1, "FAIL: cannot open golden vector file %s", name);
        end
        $fclose(fd);
        return name;
    endfunction

    task automatic run_case(input int k_dim);
        int mismatches     = 0;
        int first_mismatch = -1;

        $readmemh(vector_file(k_dim, "A"), activation_mem, 0, SYS_ROW*k_dim - 1);
        $readmemh(vector_file(k_dim, "B"), weight_mem, 0, k_dim*SYS_COL - 1);
        $readmemh(vector_file(k_dim, "C_acc"), expected_mem, 0, SYS_ROW*SYS_COL - 1);

        // k_dim slices, then PIPELINE_LATENCY cycles of zeros
        for (int k = 0; k < k_dim + dut.PIPELINE_LATENCY; k++) begin
            @(negedge CLK);
            first_k_in = (k == 0);
            for (int row = 0; row < SYS_ROW; row++) begin
                activation_in[row] = (k < k_dim) ? activation_mem[row*k_dim + k] : '0;
            end
            for (int col = 0; col < SYS_COL; col++) begin
                weight_in[col] = (k < k_dim) ? weight_mem[k*SYS_COL + col] : '0;
            end
        end
        @(negedge CLK);

        for (int row = 0; row < SYS_ROW; row++) begin
            for (int col = 0; col < SYS_COL; col++) begin
                if (accumulator_out[row][col] !== expected_mem[row*SYS_COL + col]) begin
                    if (first_mismatch < 0) begin
                        first_mismatch = row*SYS_COL + col;
                        $display("  K=%0d first mismatch at index %0d (row %0d, col %0d): got %h, expected %h",
                                 k_dim, first_mismatch, row, col,
                                 accumulator_out[row][col], expected_mem[row*SYS_COL + col]);
                    end
                    mismatches++;
                end
            end
        end

        accumulators_ok    += SYS_ROW*SYS_COL - mismatches;
        accumulators_wrong += mismatches;
        if (mismatches == 0) begin
            cases_passed++;
            $display("  K=%0d PASS (%0d accumulators)", k_dim, SYS_ROW*SYS_COL);
        end
        else begin
            cases_failed++;
            $display("  K=%0d FAIL (%0d of %0d accumulators wrong, first mismatch index %0d)",
                     k_dim, mismatches, SYS_ROW*SYS_COL, first_mismatch);
        end
    endtask

    initial begin
        $display("tb_systolic_array: SYS_ROW=%0d SYS_COL=%0d PIPELINE_LATENCY=%0d",
                 SYS_ROW, SYS_COL, dut.PIPELINE_LATENCY);
        for (int i = 0; i < K_CASE_COUNT; i++) begin
            run_case(K_CASES[i]);
        end
        if (cases_failed == 0) begin
            $display("PASS: %0dx%0d array, %0d of %0d K cases, %0d of %0d accumulators correct",
                     SYS_ROW, SYS_COL, cases_passed, K_CASE_COUNT,
                     accumulators_ok, accumulators_ok + accumulators_wrong);
        end
        else begin
            $display("FAIL: %0dx%0d array, %0d of %0d K cases failed, %0d of %0d accumulators wrong",
                     SYS_ROW, SYS_COL, cases_failed, K_CASE_COUNT,
                     accumulators_wrong, accumulators_ok + accumulators_wrong);
        end
        $finish;
    end

endmodule
