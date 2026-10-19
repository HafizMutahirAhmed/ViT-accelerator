`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// tb_encoder_program for the 112 x 112 MNIST configuration (50 tokens): the
// same testbench with that configuration's vectors, pool depths and ROM
// images. The depths are the plan python/layer_program.py prints for 50
// tokens and a 16,384-word weight buffer; the testbench checks them against
// program_plan.txt.
//
// Vectors: python/make_model.py --board nexys --layers 2, in python/models/nexys_L2.
//
// sim.sh: also compile tb_encoder_program.sv
//////////////////////////////////////////////////////////////////////////////////

module tb_encoder_program_nexys;

    tb_encoder_program #(
        .VECTOR_DIR     ("../../python/models/nexys_L2/"),
        .DEPTH_X        (1344),
        .DEPTH_W        (16384),
        .DEPTH_BIG      (5376),
        .DEPTH_N        (1344),
        .DEPTH_S        (1050),
        .DEPTH_QV       (2544),
        .DEPTH_K        (1344),
        .DEPTH_UNUSED   (16),
        .GELU_ROM_FILE  ("../rom/nexys/gelu_rom.hex"),
        .EXP_ROM_FILE   ("../rom/nexys/exp_rom.hex"),
        .RSQRT_ROM_FILE ("../rom/nexys/rsqrt_rom.hex")
    ) program_tb ();

endmodule
