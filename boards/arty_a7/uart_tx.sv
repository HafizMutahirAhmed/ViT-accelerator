`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: uart_tx
// Description: UART transmitter: 8 data bits, no parity, 1 stop bit, least
// significant bit first. One bit lasts CLOCKS_PER_BIT clocks.
//
// A byte is taken in a cycle where data_valid and data_ready are high.
// data_ready is high while the line is idle; it goes low for the start bit,
// the eight data bits and the stop bit. txd is registered and idles high.
//////////////////////////////////////////////////////////////////////////////////

module uart_tx #(
    parameter int CLOCKS_PER_BIT = 109
) (
    input  logic       CLK,
    input  logic       reset,
    input  logic [7:0] data,
    input  logic       data_valid,
    output logic       data_ready,
    output logic       txd
);

    localparam int COUNT_WIDTH = $clog2(CLOCKS_PER_BIT);

    logic [COUNT_WIDTH-1:0] count;
    logic [3:0]             bits_left;   // start, 8 data, stop
    logic [8:0]             shift;       // data above the bit on the line

    assign data_ready = bits_left == '0;

    always_ff @(posedge CLK) begin
        if (reset) begin
            bits_left <= '0;
            txd       <= 1'b1;
        end
        else if (bits_left == '0) begin
            txd <= 1'b1;
            if (data_valid) begin
                txd       <= 1'b0;                     // start bit
                shift     <= {1'b1, data};             // data, then the stop bit
                bits_left <= 4'd10;
                count     <= COUNT_WIDTH'(CLOCKS_PER_BIT - 1);
            end
        end
        else if (count == '0) begin
            count     <= COUNT_WIDTH'(CLOCKS_PER_BIT - 1);
            bits_left <= bits_left - 1'b1;
            if (bits_left != 4'd1) begin
                txd   <= shift[0];
                shift <= {1'b1, shift[8:1]};
            end
        end
        else begin
            count <= count - 1'b1;
        end
    end

endmodule
