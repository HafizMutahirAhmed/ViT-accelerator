`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: skew
// Description: Input skew for the systolic array.
//
// Lane i is delayed by i clock cycles: lane 0 is a wire, lane i has i
// registers. Data is passed through unchanged, DATA_WIDTH bits per lane.
// The delay registers have no reset.
//////////////////////////////////////////////////////////////////////////////////

module skew #(
    parameter int LANE_COUNT = vit_config_pkg::SYS_ROW,
    parameter int DATA_WIDTH = vit_config_pkg::DATA_WIDTH
) (
    input  logic                                  CLK,
    input  logic [LANE_COUNT-1:0][DATA_WIDTH-1:0] data_in,
    output logic [LANE_COUNT-1:0][DATA_WIDTH-1:0] data_out
);

    genvar lane;
    generate
        for (lane = 0; lane < LANE_COUNT; lane++) begin: lane_delay
            if (lane == 0) begin: no_delay
                assign data_out[lane] = data_in[lane];
            end
            else begin: delay_line
                logic [lane-1:0][DATA_WIDTH-1:0] delay_reg;

                always_ff @(posedge CLK) begin
                    delay_reg[0] <= data_in[lane];
                    for (int stage = 1; stage < lane; stage++) begin
                        delay_reg[stage] <= delay_reg[stage-1];
                    end
                end

                assign data_out[lane] = delay_reg[lane-1];
            end
        end
    endgenerate

endmodule
