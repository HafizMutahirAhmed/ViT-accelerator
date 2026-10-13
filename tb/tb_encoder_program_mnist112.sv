`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// tb_encoder_program for the 112 x 112 MNIST configuration (50 tokens): the
// same testbench with that configuration's vectors, pool depths and ROM
// images. The depths are the plan python/layer_program.py prints for 50
// tokens and a 16,384-word weight buffer; the testbench checks them against
// program_plan.txt.
//
// Vectors: python/gen_mnist_forward.py, in python/vectors/mnist112.
//
// sim.sh: also compile tb_encoder_program.sv
//////////////////////////////////////////////////////////////////////////////////

module tb_encoder_program_mnist112;

    tb_encoder_program #(
        .VECTOR_DIR     ("../../python/vectors/mnist112/"),
        .DEPTH_X        (1344),
        .DEPTH_W        (16384),
        .DEPTH_BIG      (5376),
        .DEPTH_N        (1344),
        .DEPTH_S        (1050),
        .DEPTH_QV       (2544),
        .DEPTH_K        (1344),
        .DEPTH_UNUSED   (16),
        .GELU_ROM_FILE  ("../rom/mnist112/gelu_rom.hex"),
        .EXP_ROM_FILE   ("../rom/mnist112/exp_rom.hex"),
        .RSQRT_ROM_FILE ("../rom/mnist112/rsqrt_rom.hex")
    ) program_tb ();

endmodule
