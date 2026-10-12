`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: host_link
// Description: Runs the host protocol of docs/HOST_PROTOCOL.md on a byte
// stream and drives the host port of matmul_engine. Board independent: the
// transport (a UART, FrontPanel pipes) only has to deliver and take bytes.
//
// Byte stream. rx_data is taken in a cycle where rx_valid and rx_ready are
// high; tx_data is sent in a cycle where tx_valid and tx_ready are high.
// rx_ready is low only while a received item is being written to the engine
// (1 cycle, 5 for a descriptor) and while a response is being produced.
//
// Command packet: 0xA5, command, target, address (3 bytes), count (2 bytes),
// payload, CRC-16 (2 bytes); multi-byte fields least significant byte first.
// Response packet: 0x5A, status, payload, CRC-16. See the document for the
// commands. A write command's items are written to the engine as they arrive,
// one engine write per item: the pool (8 bytes, one per bank), the bias store
// (32 bytes, one INT32 per bank), the LayerNorm constants (6 bytes: gain,
// offset) or the descriptor memory (20 bytes, five 32-bit words, written one
// word per cycle). While the engine is busy only STATUS and CLEAR are done;
// other commands are consumed and answered with the busy status.
//
// Every engine port output is registered. A read presents the address and
// takes the data READ_WAIT cycles later. error_flag is sticky: it is set by
// any response with a non-zero status and cleared by the CLEAR command.
// done_flag is set by the engine's done and cleared by START and CLEAR.
// No arithmetic on the data: bytes are moved, never changed.
//////////////////////////////////////////////////////////////////////////////////

module host_link #(
    parameter  int LANES      = vit_config_pkg::SYS_ROW,
    parameter  int DATA_WIDTH = vit_config_pkg::DATA_WIDTH,
    parameter  int ACC_WIDTH  = vit_config_pkg::ACC_WIDTH,
    localparam int BUFFER_ID_WIDTH        = vit_config_pkg::BUFFER_ID_WIDTH,
    localparam int ADDRESS_WIDTH          = vit_config_pkg::ADDRESS_WIDTH,
    localparam int DESCRIPTOR_WORDS       = vit_config_pkg::DESCRIPTOR_WORDS,
    localparam int DESCRIPTOR_INDEX_WIDTH = $clog2(vit_config_pkg::DESCRIPTOR_COUNT),
    localparam int GAIN_WIDTH             = vit_config_pkg::LAYERNORM_GAIN_WIDTH,
    localparam int OFFSET_WIDTH           = vit_config_pkg::LAYERNORM_OFFSET_WIDTH
) (
    input  logic                                   CLK,
    input  logic                                   reset,

    input  logic [7:0]                             rx_data,
    input  logic                                   rx_valid,
    output logic                                   rx_ready,
    output logic [7:0]                             tx_data,
    output logic                                   tx_valid,
    input  logic                                   tx_ready,

    output logic                                   start,
    output logic [DESCRIPTOR_INDEX_WIDTH-1:0]      start_index,
    input  logic                                   busy,
    input  logic                                   done,
    output logic [DESCRIPTOR_INDEX_WIDTH-1:0]      host_descriptor_index,
    output logic [DESCRIPTOR_WORDS-1:0]            host_descriptor_write_enable,
    output logic [31:0]                            host_descriptor_write_data,
    output logic [BUFFER_ID_WIDTH-1:0]             host_buffer_id,
    output logic [ADDRESS_WIDTH-1:0]               host_address,
    output logic [LANES-1:0]                       host_buffer_write_enable,
    output logic [LANES-1:0][DATA_WIDTH-1:0]       host_buffer_write_data,
    input  logic [LANES-1:0][DATA_WIDTH-1:0]       host_buffer_read_data,
    output logic [LANES-1:0]                       host_bias_write_enable,
    output logic [LANES-1:0][ACC_WIDTH-1:0]        host_bias_write_data,
    output logic                                   host_layernorm_write_enable,
    output logic signed [GAIN_WIDTH-1:0]           host_layernorm_write_gain,
    output logic signed [OFFSET_WIDTH-1:0]         host_layernorm_write_offset,

    output logic                                   done_flag,
    output logic                                   error_flag
);

    localparam logic [7:0] COMMAND_START_BYTE  = 8'hA5;
    localparam logic [7:0] RESPONSE_START_BYTE = 8'h5A;

    localparam logic [7:0] COMMAND_WRITE_BUFFER     = 8'h01;
    localparam logic [7:0] COMMAND_WRITE_BIAS       = 8'h02;
    localparam logic [7:0] COMMAND_WRITE_LAYERNORM  = 8'h03;
    localparam logic [7:0] COMMAND_WRITE_DESCRIPTOR = 8'h04;
    localparam logic [7:0] COMMAND_START            = 8'h05;
    localparam logic [7:0] COMMAND_READ_BUFFER      = 8'h06;
    localparam logic [7:0] COMMAND_STATUS           = 8'h07;
    localparam logic [7:0] COMMAND_CLEAR            = 8'h08;

    localparam int HEADER_BYTES     = 7;                          // command .. count
    localparam int BUFFER_ITEM      = LANES * DATA_WIDTH / 8;
    localparam int BIAS_ITEM        = LANES * ACC_WIDTH / 8;
    localparam int LAYERNORM_ITEM   = (GAIN_WIDTH + OFFSET_WIDTH) / 8;
    localparam int DESCRIPTOR_ITEM  = 4 * DESCRIPTOR_WORDS;
    localparam int ITEM_BYTES_MAX   = BIAS_ITEM;
    localparam int ITEM_INDEX_WIDTH = $clog2(ITEM_BYTES_MAX);
    localparam int STATUS_BYTES     = 5;                          // flags, 4 bytes of cycles
    // from the address at this module's outputs to the read data
    localparam int READ_WAIT        = vit_config_pkg::BUFFER_READ_LATENCY + 1;

    // CRC-16/CCITT-FALSE, one byte
    function automatic logic [15:0] crc_next(input logic [15:0] crc, input logic [7:0] data);
        logic [15:0] value;
        value = crc ^ {data, 8'h00};
        for (int bit_index = 0; bit_index < 8; bit_index++) begin
            value = value[15] ? (value << 1) ^ 16'h1021 : value << 1;
        end
        return value;
    endfunction

    typedef enum logic [3:0] {
        STATE_IDLE,        // waiting for the start byte
        STATE_HEADER,
        STATE_PAYLOAD,
        STATE_WRITE,       // one item to the engine
        STATE_CHECKSUM,
        STATE_EXECUTE,
        STATE_READ_WAIT,   // read address presented, waiting for the data
        STATE_SEND_START,
        STATE_SEND_STATUS,
        STATE_SEND_PAYLOAD,
        STATE_SEND_CHECKSUM
    } state_t;

    state_t                       state;
    logic [7:0]                   command;
    logic [BUFFER_ID_WIDTH-1:0]   target;
    logic [23:0]                  address;
    logic [15:0]                  items_left;
    logic [2:0]                   header_index;
    logic [ITEM_INDEX_WIDTH-1:0]  item_index;
    logic [ITEM_INDEX_WIDTH:0]    item_bytes;
    logic [8*ITEM_BYTES_MAX-1:0]  item;
    logic [15:0]                  receive_crc;
    logic [15:0]                  received_checksum;
    logic                         checksum_index;
    logic [2:0]                   word_index;        // descriptor word being written
    logic                         is_write;
    logic                         is_known;
    logic                         rejected;          // the engine is busy and the command needs it idle
    logic [2:0]                   status;
    logic [15:0]                  send_crc;
    logic [ITEM_INDEX_WIDTH-1:0]  send_index;
    logic [8*BUFFER_ITEM-1:0]     read_item;
    logic [3:0]                   wait_count;
    logic [31:0]                  busy_cycles;

    assign rx_ready = state == STATE_IDLE || state == STATE_HEADER || state == STATE_PAYLOAD
                      || state == STATE_CHECKSUM;

    always_comb begin
        case (command)
            COMMAND_WRITE_BUFFER:     item_bytes = (ITEM_INDEX_WIDTH+1)'(BUFFER_ITEM);
            COMMAND_WRITE_BIAS:       item_bytes = (ITEM_INDEX_WIDTH+1)'(BIAS_ITEM);
            COMMAND_WRITE_LAYERNORM:  item_bytes = (ITEM_INDEX_WIDTH+1)'(LAYERNORM_ITEM);
            default:                  item_bytes = (ITEM_INDEX_WIDTH+1)'(DESCRIPTOR_ITEM);
        endcase
    end

    assign is_write = command == COMMAND_WRITE_BUFFER || command == COMMAND_WRITE_BIAS
                      || command == COMMAND_WRITE_LAYERNORM || command == COMMAND_WRITE_DESCRIPTOR;
    assign is_known = is_write || command == COMMAND_START || command == COMMAND_READ_BUFFER
                      || command == COMMAND_STATUS || command == COMMAND_CLEAR;

    // what the response payload byte is
    logic [7:0] status_bytes[STATUS_BYTES];

    always_comb begin
        status_bytes[0] = {5'b0, error_flag, done_flag, busy};
        for (int i = 0; i < 4; i++) begin
            status_bytes[i+1] = busy_cycles[8*i +: 8];
        end
    end

    always_ff @(posedge CLK) begin
        if (reset) begin
            state                        <= STATE_IDLE;
            start                        <= 1'b0;
            host_descriptor_write_enable <= '0;
            host_buffer_write_enable     <= '0;
            host_bias_write_enable       <= '0;
            host_layernorm_write_enable  <= 1'b0;
            tx_valid                     <= 1'b0;
            done_flag                    <= 1'b0;
            error_flag                   <= 1'b0;
            busy_cycles                  <= '0;
        end
        else begin
            start                        <= 1'b0;
            host_descriptor_write_enable <= '0;
            host_buffer_write_enable     <= '0;
            host_bias_write_enable       <= '0;
            host_layernorm_write_enable  <= 1'b0;

            if (done) begin
                done_flag <= 1'b1;
            end
            if (busy) begin
                busy_cycles <= busy_cycles + 1'b1;
            end

            case (state)
                STATE_IDLE: begin
                    header_index <= '0;
                    receive_crc  <= 16'hFFFF;
                    if (rx_valid && rx_data == COMMAND_START_BYTE) begin
                        state <= STATE_HEADER;
                    end
                end

                STATE_HEADER: begin
                    if (rx_valid) begin
                        receive_crc <= crc_next(receive_crc, rx_data);
                        case (header_index)
                            3'd0: command           <= rx_data;
                            3'd1: target            <= rx_data[BUFFER_ID_WIDTH-1:0];
                            3'd2: address[7:0]      <= rx_data;
                            3'd3: address[15:8]     <= rx_data;
                            3'd4: address[23:16]    <= rx_data;
                            3'd5: items_left[7:0]   <= rx_data;
                            default: items_left[15:8] <= rx_data;
                        endcase
                        header_index <= header_index + 1'b1;
                        if (header_index == 3'(HEADER_BYTES - 1)) begin
                            // command is complete here; the count's high byte is rx_data
                            rejected       <= busy && command != COMMAND_STATUS && command != COMMAND_CLEAR;
                            item_index     <= '0;
                            checksum_index <= 1'b0;
                            state          <= (is_write && {rx_data, items_left[7:0]} != '0) ? STATE_PAYLOAD
                                                                                             : STATE_CHECKSUM;
                        end
                    end
                end

                STATE_PAYLOAD: begin
                    if (rx_valid) begin
                        receive_crc           <= crc_next(receive_crc, rx_data);
                        item[8*item_index +: 8] <= rx_data;
                        if ((ITEM_INDEX_WIDTH+1)'(item_index) == item_bytes - 1'b1) begin
                            item_index <= '0;
                            word_index <= '0;
                            state      <= STATE_WRITE;
                        end
                        else begin
                            item_index <= item_index + 1'b1;
                        end
                    end
                end

                STATE_WRITE: begin
                    host_buffer_id        <= target;
                    host_address          <= address[ADDRESS_WIDTH-1:0];
                    host_descriptor_index <= address[DESCRIPTOR_INDEX_WIDTH-1:0];
                    host_buffer_write_data      <= item[8*BUFFER_ITEM-1:0];
                    host_bias_write_data        <= item[8*BIAS_ITEM-1:0];
                    host_layernorm_write_gain   <= item[GAIN_WIDTH-1:0];
                    host_layernorm_write_offset <= item[GAIN_WIDTH +: OFFSET_WIDTH];
                    host_descriptor_write_data  <= item[32*word_index +: 32];
                    if (!rejected) begin
                        case (command)
                            COMMAND_WRITE_BUFFER:    host_buffer_write_enable    <= '1;
                            COMMAND_WRITE_BIAS:      host_bias_write_enable      <= '1;
                            COMMAND_WRITE_LAYERNORM: host_layernorm_write_enable <= 1'b1;
                            default:                 host_descriptor_write_enable[word_index] <= 1'b1;
                        endcase
                    end
                    if (command == COMMAND_WRITE_DESCRIPTOR && word_index != 3'(DESCRIPTOR_WORDS - 1)) begin
                        word_index <= word_index + 1'b1;
                    end
                    else begin
                        address    <= address + 1'b1;
                        items_left <= items_left - 1'b1;
                        state      <= (items_left == 16'd1) ? STATE_CHECKSUM : STATE_PAYLOAD;
                    end
                end

                STATE_CHECKSUM: begin
                    if (rx_valid) begin
                        checksum_index <= 1'b1;
                        if (!checksum_index) begin
                            received_checksum[7:0] <= rx_data;
                        end
                        else begin
                            received_checksum[15:8] <= rx_data;
                            state                   <= STATE_EXECUTE;
                        end
                    end
                end

                STATE_EXECUTE: begin
                    status   <= {rejected && is_known, !is_known, received_checksum != receive_crc};
                    send_crc <= 16'hFFFF;
                    state    <= STATE_SEND_START;
                    if (received_checksum == receive_crc && is_known && !rejected) begin
                        if (command == COMMAND_START) begin
                            start       <= 1'b1;
                            start_index <= address[DESCRIPTOR_INDEX_WIDTH-1:0];
                            done_flag   <= 1'b0;
                            busy_cycles <= '0;
                        end
                        if (command == COMMAND_CLEAR) begin
                            done_flag  <= 1'b0;
                            error_flag <= 1'b0;
                        end
                    end
                end

                STATE_SEND_START: begin
                    tx_valid <= 1'b1;
                    tx_data  <= RESPONSE_START_BYTE;
                    if (tx_valid && tx_ready) begin
                        tx_data  <= {5'b0, status};
                        send_crc <= crc_next(send_crc, {5'b0, status});
                        state    <= STATE_SEND_STATUS;
                        if (status != '0) begin
                            error_flag <= 1'b1;
                        end
                    end
                end

                STATE_SEND_STATUS: begin
                    if (tx_ready) begin
                        send_index     <= '0;
                        checksum_index <= 1'b0;
                        if (status == '0 && command == COMMAND_READ_BUFFER && items_left != '0) begin
                            tx_valid       <= 1'b0;
                            host_buffer_id <= target;
                            host_address   <= address[ADDRESS_WIDTH-1:0];
                            wait_count     <= 4'(READ_WAIT);
                            state          <= STATE_READ_WAIT;
                        end
                        else if (status == '0 && command == COMMAND_STATUS) begin
                            tx_data  <= status_bytes[0];
                            send_crc <= crc_next(send_crc, status_bytes[0]);
                            state    <= STATE_SEND_PAYLOAD;
                        end
                        else begin
                            tx_data <= send_crc[7:0];
                            state   <= STATE_SEND_CHECKSUM;
                        end
                    end
                end

                STATE_READ_WAIT: begin
                    wait_count <= wait_count - 1'b1;
                    if (wait_count == '0) begin
                        read_item  <= host_buffer_read_data;
                        tx_valid   <= 1'b1;
                        tx_data    <= host_buffer_read_data[0];
                        send_crc   <= crc_next(send_crc, host_buffer_read_data[0]);
                        send_index <= '0;
                        address    <= address + 1'b1;
                        items_left <= items_left - 1'b1;
                        state      <= STATE_SEND_PAYLOAD;
                    end
                end

                STATE_SEND_PAYLOAD: begin
                    if (tx_ready) begin
                        send_index <= send_index + 1'b1;
                        if (command == COMMAND_STATUS) begin
                            if (send_index == ITEM_INDEX_WIDTH'(STATUS_BYTES - 1)) begin
                                tx_data <= send_crc[7:0];
                                state   <= STATE_SEND_CHECKSUM;
                            end
                            else begin
                                tx_data  <= status_bytes[send_index + 1'b1];
                                send_crc <= crc_next(send_crc, status_bytes[send_index + 1'b1]);
                            end
                        end
                        else if (send_index != ITEM_INDEX_WIDTH'(BUFFER_ITEM - 1)) begin
                            tx_data  <= read_item[8*(send_index + 1'b1) +: 8];
                            send_crc <= crc_next(send_crc, read_item[8*(send_index + 1'b1) +: 8]);
                        end
                        else if (items_left != '0) begin
                            tx_valid     <= 1'b0;
                            host_address <= address[ADDRESS_WIDTH-1:0];
                            wait_count   <= 4'(READ_WAIT);
                            state        <= STATE_READ_WAIT;
                        end
                        else begin
                            tx_data <= send_crc[7:0];
                            state   <= STATE_SEND_CHECKSUM;
                        end
                    end
                end

                STATE_SEND_CHECKSUM: begin
                    if (tx_ready) begin
                        checksum_index <= 1'b1;
                        if (!checksum_index) begin
                            tx_data <= send_crc[15:8];
                        end
                        else begin
                            tx_valid <= 1'b0;
                            state    <= STATE_IDLE;
                        end
                    end
                end

                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule
