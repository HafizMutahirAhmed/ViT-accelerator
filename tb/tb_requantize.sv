`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for requantize.
//
// Applies every (acc, scale, shift) vector, holds it for the requantizer's
// LATENCY cycles and compares the output with the golden INT8 result.
// valid_in is high on every other vector and valid_out must follow it by
// LATENCY cycles.
//
// Vectors come from golden/gen_vectors.py:
//   requant_cfg.hex    32-bit word: number of vectors
//   requant_acc.hex    INT32
//   requant_scale.hex  INT8 (signed)
//   requant_shift.hex  8-bit word holding the shift
//   requant_out.hex    INT8
//////////////////////////////////////////////////////////////////////////////////

module tb_requantize #(
    parameter string VECTOR_DIR = "../golden/vectors/"
);

    localparam int DATA_WIDTH  = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH   = vit_config_pkg::ACC_WIDTH;
    localparam int SCALE_WIDTH = vit_config_pkg::SCALE_WIDTH;
    localparam int SHIFT_WIDTH = vit_config_pkg::SHIFT_WIDTH;

    localparam int VECTORS_MAX = 8192;

    logic                          CLK = 1'b0;
    logic                          valid_in;
    logic signed [ACC_WIDTH-1:0]   unquantized_result;
    logic signed [SCALE_WIDTH-1:0] scale;
    logic [SHIFT_WIDTH-1:0]        shift_value;
    logic                          valid_out;
    logic signed [DATA_WIDTH-1:0]  quantized_result;

    logic [31:0]            cfg_mem[1];
    logic [ACC_WIDTH-1:0]   acc_mem[VECTORS_MAX];
    logic [SCALE_WIDTH-1:0] scale_mem[VECTORS_MAX];
    logic [7:0]             shift_mem[VECTORS_MAX];
    logic [DATA_WIDTH-1:0]  out_mem[VECTORS_MAX];

    int vector_count;
    int mismatches       = 0;
    int first_mismatch   = -1;
    int valid_mismatches = 0;

    requantize dut (
        .CLK                (CLK),
        .valid_in           (valid_in),
        .unquantized_result (unquantized_result),
        .scale              (scale),
        .shift_value        (shift_value),
        .valid_out          (valid_out),
        .quantized_result   (quantized_result)
    );

    always #5 CLK = ~CLK;

    function automatic string vector_file(input string suffix);
        string name;
        int    fd;
        name = $sformatf("%srequant_%s.hex", VECTOR_DIR, suffix);
        fd = $fopen(name, "r");
        if (fd == 0) begin
            $fatal(1, "FAIL: cannot open golden vector file %s", name);
        end
        $fclose(fd);
        return name;
    endfunction

    initial begin
        $readmemh(vector_file("cfg"), cfg_mem);
        vector_count = cfg_mem[0];
        if (vector_count > VECTORS_MAX) begin
            $fatal(1, "FAIL: %0d vectors is more than VECTORS_MAX", vector_count);
        end
        $readmemh(vector_file("acc"), acc_mem, 0, vector_count - 1);
        $readmemh(vector_file("scale"), scale_mem, 0, vector_count - 1);
        $readmemh(vector_file("shift"), shift_mem, 0, vector_count - 1);
        $readmemh(vector_file("out"), out_mem, 0, vector_count - 1);

        for (int i = 0; i < vector_count; i++) begin
            @(negedge CLK);
            valid_in           = i[0];
            unquantized_result = acc_mem[i];
            scale              = scale_mem[i];
            shift_value        = shift_mem[i][SHIFT_WIDTH-1:0];
            repeat (dut.LATENCY) @(negedge CLK);
            if (quantized_result !== out_mem[i]) begin
                if (first_mismatch < 0) begin
                    first_mismatch = i;
                    $display("  first mismatch at index %0d: acc %h scale %h shift %0d, got %h, expected %h",
                             i, acc_mem[i], scale_mem[i], shift_mem[i], quantized_result, out_mem[i]);
                end
                mismatches++;
            end
            if (valid_out !== i[0]) begin
                valid_mismatches++;
            end
        end

        if (mismatches == 0 && valid_mismatches == 0) begin
            $display("PASS: %0d of %0d requantize vectors correct, valid_out correct", vector_count, vector_count);
        end
        else begin
            $display("FAIL: %0d of %0d vectors wrong, first mismatch index %0d, %0d valid_out errors",
                     mismatches, vector_count, first_mismatch, valid_mismatches);
        end
        $finish;
    end

endmodule
