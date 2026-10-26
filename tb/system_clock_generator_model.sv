`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: system_clock_generator (simulation model)
// Description: Behavioural model of the Clocking Wizard IP of the same name
// in vivado/vit.xpr, for tb_nexys_top_ddr:
// clk_out1 is clk_in1 (100 MHz), clk_out2 a 200 MHz clock, and locked rises
// LOCK_CYCLES cycles of clk_in1 after the start.
//////////////////////////////////////////////////////////////////////////////////

module system_clock_generator #(
    parameter int LOCK_CYCLES = 40
) (
    input  logic clk_in1,
    output logic clk_out1,
    output logic clk_out2,
    output logic locked
);

    int cycles = 0;

    assign clk_out1 = clk_in1;

    initial clk_out2 = 1'b0;
    always #2.5 clk_out2 = ~clk_out2;

    always @(posedge clk_in1) begin
        if (cycles < LOCK_CYCLES) begin
            cycles <= cycles + 1;
        end
    end
    assign locked = cycles >= LOCK_CYCLES;

endmodule
