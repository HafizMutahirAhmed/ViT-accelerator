`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// tb_axi_replay_run with the 6-layer Nexys model: the same testbench and the same
// hardware (the board's buffer depths and ROM images), the byte stream of
// python/run_image.py --model --digit 0 --golden --clear --record
// --model-dir python/models/nexys_L6, made after
// python/make_model.py --board nexys --layers 6.
//
// also compile: tb_axi_replay_run.sv
// also compile: axi_rule_checker.sv
// also compile: memory_system_model.sv
//////////////////////////////////////////////////////////////////////////////////

`include "vit_paths.svh"

module tb_axi_replay_run_nexys_L6;

    tb_axi_replay_run #(
        .VECTOR_DIR ({`VIT_ROOT, "python/models/nexys_L6/"})
    ) replay ();

endmodule
