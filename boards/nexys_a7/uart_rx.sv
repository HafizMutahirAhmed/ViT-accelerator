`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: uart_rx
// Description: UART receiver: 8 data bits, no parity, 1 stop bit, least
// significant bit first. One bit lasts CLOCKS_PER_BIT clocks (the clock
// frequency divided by the baud rate, rounded; at least 4).
//
// rxd is asynchronous and goes through two flip-flops first. A start bit is
// accepted if the line is still low half a bit after the falling edge; every
// data bit and the stop bit are then sampled in the middle of the bit.
// data_valid goes high with the byte when the stop bit was high and stays
// high until data_ready is high in the same cycle. overrun pulses if another
// byte completes while one is still waiting, in which case the new byte is
// dropped. A byte with a low stop bit is dropped silently.
//////////////////////////////////////////////////////////////////////////////////

module uart_rx #(
    parameter int CLOCKS_PER_BIT = 109
) (
    input  logic       CLK,
    input  logic       reset,
    input  logic       rxd,
    output logic [7:0] data,
    output logic       data_valid,
    input  logic       data_ready,
    output logic       overrun
);

    localparam int COUNT_WIDTH = $clog2(CLOCKS_PER_BIT);

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_START,   // waiting for the middle of the start bit
        STATE_DATA,
        STATE_STOP
    } state_t;

    (* ASYNC_REG = "TRUE" *) logic [1:0] rxd_sync;
    state_t                 state;
    logic [COUNT_WIDTH-1:0] count;
    logic [2:0]             bit_index;
    logic [7:0]             shift;
    logic                   rxd_in;

    assign rxd_in = rxd_sync[1];

    always_ff @(posedge CLK) begin
        rxd_sync <= {rxd_sync[0], rxd};
    end

    always_ff @(posedge CLK) begin
        if (reset) begin
            state      <= STATE_IDLE;
            data_valid <= 1'b0;
            overrun    <= 1'b0;
        end
        else begin
            overrun <= 1'b0;
            if (data_valid && data_ready) begin
                data_valid <= 1'b0;
            end
            case (state)
                STATE_IDLE: begin
                    count <= COUNT_WIDTH'(CLOCKS_PER_BIT / 2 - 1);
                    if (!rxd_in) begin
                        state <= STATE_START;
                    end
                end
                STATE_START: begin
                    if (count == '0) begin
                        count     <= COUNT_WIDTH'(CLOCKS_PER_BIT - 1);
                        bit_index <= '0;
                        state     <= rxd_in ? STATE_IDLE : STATE_DATA;
                    end
                    else begin
                        count <= count - 1'b1;
                    end
                end
                STATE_DATA: begin
                    if (count == '0) begin
                        count     <= COUNT_WIDTH'(CLOCKS_PER_BIT - 1);
                        shift     <= {rxd_in, shift[7:1]};
                        bit_index <= bit_index + 1'b1;
                        if (bit_index == 3'd7) begin
                            state <= STATE_STOP;
                        end
                    end
                    else begin
                        count <= count - 1'b1;
                    end
                end
                default: begin   // STATE_STOP
                    if (count == '0) begin
                        state <= STATE_IDLE;
                        if (rxd_in) begin
                            if (data_valid && !data_ready) begin
                                overrun <= 1'b1;
                            end
                            else begin
                                data       <= shift;
                                data_valid <= 1'b1;
                            end
                        end
                    end
                    else begin
                        count <= count - 1'b1;
                    end
                end
            endcase
        end
    end

endmodule
