`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: axi_rule_checker
// Description: Simulation checker of one VALID/READY channel (an AXI4-Lite
// channel or an AXI4-Stream). The AXI rule: once VALID is high it stays high,
// and the payload stays the same, until the cycle READY is high with it.
//
// A violation is reported with $error (an ERROR line in the log) and counted
// in `violations`. `beats` counts the transfers. Checking is off while reset
// is high, and can be relaxed for one cycle after `flush` (a soft reset is
// allowed to drop a pending transfer).
//////////////////////////////////////////////////////////////////////////////////

module axi_rule_checker #(
    parameter int    PAYLOAD_WIDTH = 32,
    parameter string NAME          = "channel"
) (
    input logic                     CLK,
    input logic                     reset,
    input logic                     flush,
    input logic                     valid,
    input logic                     ready,
    input logic [PAYLOAD_WIDTH-1:0] payload
);

    logic                     pending;        // valid was high without ready in the last cycle
    logic [PAYLOAD_WIDTH-1:0] pending_payload;
    int                       violations = 0;
    longint                   beats      = 0;

    always @(posedge CLK) begin
        if (reset || flush) begin
            pending <= 1'b0;
        end
        else begin
            if (pending) begin
                if (valid !== 1'b1) begin
                    violations++;
                    $error("%s: VALID dropped before READY", NAME);
                end
                else if (payload !== pending_payload) begin
                    violations++;
                    $error("%s: payload changed while VALID waited for READY", NAME);
                end
            end
            if (valid === 1'b1 && ready === 1'b1) begin
                beats++;
            end
            pending         <= valid === 1'b1 && ready !== 1'b1;
            pending_payload <= payload;
        end
    end

endmodule
