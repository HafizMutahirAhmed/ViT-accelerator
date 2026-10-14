`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: axi_host_bridge
// Description: Runs the host protocol of docs/HOST_PROTOCOL.md on a byte
// stream and turns it into AXI transactions for vit_axi_wrapper. Board
// independent: a UART or a FrontPanel adapter only delivers and takes bytes.
// It answers every packet with the bytes the host model in python/ answers.
//
// Byte stream. rx_data is taken when rx_valid and rx_ready are high, tx_data
// is sent when tx_valid and tx_ready are high. rx_ready is low while a
// register access or a stream beat is in progress and while a response is
// being produced.
//
// What a command becomes (registers of docs/AXI_REGISTER_MAP.md):
//   every command    read STATUS first: while the engine is busy only STATUS
//                    and CLEAR are done, anything else is consumed and
//                    answered with the busy status
//   WRITE_BUFFER, WRITE_BIAS, WRITE_LAYERNORM, WRITE_DESCRIPTOR
//                    write STREAM_TARGET and STREAM_ADDRESS, then the payload
//                    as 64-bit beats on m_axis, first byte in TDATA[7:0]: one
//                    beat per pool address or LayerNorm pair, four per bias
//                    address, three per descriptor; TLAST on the packet's
//                    last beat
//   START            write START_INDEX, then CONTROL bit 0
//   CLEAR            write CONTROL bit 2; clears error_flag here
//   STATUS           read STATUS and CYCLES
//   READ_BUFFER      write STREAM_TARGET, STREAM_ADDRESS and STORE_COUNT,
//                    then take count beats from s_axis, 8 bytes each
//
// The AXI4-Lite master does one access at a time: AWVALID and WVALID
// together, each held until taken, then the response. error_flag is sticky:
// set by any response with a non-zero status, cleared by CLEAR.
// No arithmetic on the data: bytes are moved, never changed.
//////////////////////////////////////////////////////////////////////////////////

module axi_host_bridge #(
    parameter int LITE_ADDRESS_WIDTH = 12
) (
    input  logic                          aclk,
    input  logic                          aresetn,

    input  logic [7:0]                    rx_data,
    input  logic                          rx_valid,
    output logic                          rx_ready,
    output logic [7:0]                    tx_data,
    output logic                          tx_valid,
    input  logic                          tx_ready,

    output logic [LITE_ADDRESS_WIDTH-1:0] m_axi_awaddr,
    output logic                          m_axi_awvalid,
    input  logic                          m_axi_awready,
    output logic [31:0]                   m_axi_wdata,
    output logic [3:0]                    m_axi_wstrb,
    output logic                          m_axi_wvalid,
    input  logic                          m_axi_wready,
    input  logic [1:0]                    m_axi_bresp,
    input  logic                          m_axi_bvalid,
    output logic                          m_axi_bready,
    output logic [LITE_ADDRESS_WIDTH-1:0] m_axi_araddr,
    output logic                          m_axi_arvalid,
    input  logic                          m_axi_arready,
    input  logic [31:0]                   m_axi_rdata,
    input  logic [1:0]                    m_axi_rresp,
    input  logic                          m_axi_rvalid,
    output logic                          m_axi_rready,

    output logic [63:0]                   m_axis_tdata,
    output logic [7:0]                    m_axis_tkeep,
    output logic                          m_axis_tlast,
    output logic                          m_axis_tvalid,
    input  logic                          m_axis_tready,

    input  logic [63:0]                   s_axis_tdata,
    input  logic [7:0]                    s_axis_tkeep,
    input  logic                          s_axis_tlast,
    input  logic                          s_axis_tvalid,
    output logic                          s_axis_tready,

    output logic                          error_flag
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

    // registers of vit_axi_core
    localparam logic [11:0] REG_CONTROL        = 12'h000;
    localparam logic [11:0] REG_STATUS         = 12'h004;
    localparam logic [11:0] REG_START_INDEX    = 12'h008;
    localparam logic [11:0] REG_CYCLES         = 12'h00C;
    localparam logic [11:0] REG_STREAM_TARGET  = 12'h020;
    localparam logic [11:0] REG_STREAM_ADDRESS = 12'h024;
    localparam logic [11:0] REG_STORE_COUNT    = 12'h028;

    localparam int HEADER_BYTES = 7;     // command .. count
    localparam int STATUS_BYTES = 5;     // flags, 4 bytes of cycles

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
        STATE_IDLE,          // waiting for the start byte
        STATE_HEADER,
        STATE_BEFORE,        // register accesses before the payload
        STATE_PAYLOAD,
        STATE_BEAT,          // a beat on m_axis, until taken
        STATE_CHECKSUM,
        STATE_EXECUTE,
        STATE_AFTER,         // register accesses after the checksum
        STATE_SEND_START,
        STATE_SEND_STATUS,
        STATE_READ_BEAT,     // waiting for a beat on s_axis
        STATE_SEND_PAYLOAD,
        STATE_SEND_CHECKSUM
    } state_t;

    logic reset;
    assign reset = !aresetn;

    state_t      state;
    logic [7:0]  command;
    logic [2:0]  target;
    logic [23:0] address;
    logic [15:0] items_left;
    logic [2:0]  header_index;
    logic [5:0]  item_index;         // byte of the item
    logic [5:0]  item_bytes;
    logic [2:0]  beat_bytes;         // byte of the beat
    logic [15:0] receive_crc;
    logic [15:0] received_checksum;
    logic        checksum_index;
    logic        is_write;
    logic        is_known;
    logic        rejected;           // the engine is busy and the command needs it idle
    logic [2:0]  status;
    logic [15:0] send_crc;
    logic [2:0]  send_index;
    logic [63:0] read_item;
    logic [1:0]  step;               // register access of the running sequence
    logic        engine_busy;
    logic        engine_done;
    logic [31:0] engine_cycles;
    logic [1:0]  kind;

    always_comb begin
        case (command)
            COMMAND_WRITE_BUFFER:     item_bytes = 6'd8;
            COMMAND_WRITE_BIAS:       item_bytes = 6'd32;
            COMMAND_WRITE_LAYERNORM:  item_bytes = 6'd6;
            default:                  item_bytes = 6'd20;
        endcase
        case (command)
            COMMAND_WRITE_BIAS:       kind = 2'd1;
            COMMAND_WRITE_LAYERNORM:  kind = 2'd2;
            COMMAND_WRITE_DESCRIPTOR: kind = 2'd3;
            default:                  kind = 2'd0;
        endcase
    end

    assign is_write = command == COMMAND_WRITE_BUFFER || command == COMMAND_WRITE_BIAS
                      || command == COMMAND_WRITE_LAYERNORM || command == COMMAND_WRITE_DESCRIPTOR;
    assign is_known = is_write || command == COMMAND_START || command == COMMAND_READ_BUFFER
                      || command == COMMAND_STATUS || command == COMMAND_CLEAR;

    assign rx_ready      = state == STATE_IDLE || state == STATE_HEADER || state == STATE_PAYLOAD
                           || state == STATE_CHECKSUM;
    assign s_axis_tready = state == STATE_READ_BEAT;
    assign m_axis_tkeep  = '1;
    assign m_axi_wstrb   = '1;

    // ------------------------------------------------------------------
    // AXI4-Lite master: one access at a time
    // ------------------------------------------------------------------
    logic        lite_request;       // one cycle: start the access below
    logic        lite_write;
    logic [11:0] lite_address;
    logic [31:0] lite_data;
    logic        lite_active;
    logic        lite_done;          // one cycle: the response arrived
    logic [31:0] lite_read_data;

    assign m_axi_bready = lite_active;
    assign m_axi_rready = lite_active;

    always_ff @(posedge aclk) begin
        if (reset) begin
            m_axi_awvalid <= 1'b0;
            m_axi_wvalid  <= 1'b0;
            m_axi_arvalid <= 1'b0;
            lite_active   <= 1'b0;
            lite_done     <= 1'b0;
        end
        else begin
            lite_done <= 1'b0;
            if (m_axi_awvalid && m_axi_awready) begin
                m_axi_awvalid <= 1'b0;
            end
            if (m_axi_wvalid && m_axi_wready) begin
                m_axi_wvalid <= 1'b0;
            end
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_arvalid <= 1'b0;
            end
            if (lite_request) begin
                lite_active   <= 1'b1;
                m_axi_awvalid <= lite_write;
                m_axi_wvalid  <= lite_write;
                m_axi_arvalid <= !lite_write;
                m_axi_awaddr  <= LITE_ADDRESS_WIDTH'(lite_address);
                m_axi_araddr  <= LITE_ADDRESS_WIDTH'(lite_address);
                m_axi_wdata   <= lite_data;
            end
            if (lite_active && (m_axi_bvalid || m_axi_rvalid)) begin
                lite_active    <= 1'b0;
                lite_done      <= 1'b1;
                lite_read_data <= m_axi_rdata;
            end
        end
    end

    // the register accesses of a sequence, by step
    logic        sequence_write;
    logic [11:0] sequence_address;
    logic [31:0] sequence_data;
    logic        sequence_last;      // this step is the last of the sequence

    always_comb begin
        sequence_write   = 1'b1;
        sequence_address = REG_STATUS;
        sequence_data    = '0;
        sequence_last    = 1'b1;
        if (state == STATE_BEFORE) begin
            // STATUS, then for a write that will be done: the target and the address
            case (step)
                2'd0: begin
                    sequence_write = 1'b0;
                    sequence_last  = 1'b0;      // decided when the status is in
                end
                2'd1: begin
                    sequence_address = REG_STREAM_TARGET;
                    sequence_data    = {22'b0, kind, 5'b0, target};
                    sequence_last    = 1'b0;
                end
                default: begin
                    sequence_address = REG_STREAM_ADDRESS;
                    sequence_data    = {8'b0, address};
                end
            endcase
        end
        else begin
            case (command)
                COMMAND_START: begin
                    sequence_address = (step == 2'd0) ? REG_START_INDEX : REG_CONTROL;
                    sequence_data    = (step == 2'd0) ? {8'b0, address} : 32'd1;
                    sequence_last    = step == 2'd1;
                end
                COMMAND_CLEAR: begin
                    sequence_address = REG_CONTROL;
                    sequence_data    = 32'd4;
                end
                COMMAND_STATUS: begin
                    sequence_write   = 1'b0;
                    sequence_address = (step == 2'd0) ? REG_STATUS : REG_CYCLES;
                    sequence_last    = step == 2'd1;
                end
                default: begin   // READ_BUFFER
                    case (step)
                        2'd0: begin
                            sequence_address = REG_STREAM_TARGET;
                            sequence_data    = {29'b0, target};
                            sequence_last    = 1'b0;
                        end
                        2'd1: begin
                            sequence_address = REG_STREAM_ADDRESS;
                            sequence_data    = {8'b0, address};
                            sequence_last    = 1'b0;
                        end
                        default: begin
                            sequence_address = REG_STORE_COUNT;
                            sequence_data    = {16'b0, items_left};
                        end
                    endcase
                end
            endcase
        end
    end

    logic [7:0] status_bytes[STATUS_BYTES];

    always_comb begin
        status_bytes[0] = {5'b0, error_flag, engine_done, engine_busy};
        for (int i = 0; i < 4; i++) begin
            status_bytes[i+1] = engine_cycles[8*i +: 8];
        end
    end

    // the last byte of the item, of the beat, of the packet
    logic item_ends;
    logic beat_ends;

    assign item_ends = item_index == item_bytes - 1'b1;
    assign beat_ends = beat_bytes == 3'd7 || item_ends;

    always_ff @(posedge aclk) begin
        if (reset) begin
            state         <= STATE_IDLE;
            tx_valid      <= 1'b0;
            m_axis_tvalid <= 1'b0;
            lite_request  <= 1'b0;
            error_flag    <= 1'b0;
        end
        else begin
            lite_request <= 1'b0;

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
                            3'd1: target            <= rx_data[2:0];
                            3'd2: address[7:0]      <= rx_data;
                            3'd3: address[15:8]     <= rx_data;
                            3'd4: address[23:16]    <= rx_data;
                            3'd5: items_left[7:0]   <= rx_data;
                            default: items_left[15:8] <= rx_data;
                        endcase
                        header_index <= header_index + 1'b1;
                        if (header_index == 3'(HEADER_BYTES - 1)) begin
                            item_index     <= '0;
                            beat_bytes     <= '0;
                            checksum_index <= 1'b0;
                            step           <= '0;
                            lite_request   <= 1'b1;
                            lite_write     <= 1'b0;
                            lite_address   <= REG_STATUS;
                            state          <= STATE_BEFORE;
                        end
                    end
                end

                STATE_BEFORE: begin
                    if (lite_done) begin
                        if (step == 2'd0) begin
                            engine_busy <= lite_read_data[0];
                            rejected    <= lite_read_data[0] && command != COMMAND_STATUS
                                           && command != COMMAND_CLEAR;
                            if (!is_write || items_left == '0) begin
                                state <= STATE_CHECKSUM;
                            end
                            else if (lite_read_data[0]) begin
                                state <= STATE_PAYLOAD;        // consumed, not passed on
                            end
                            else begin
                                step         <= 2'd1;
                                lite_request <= 1'b1;
                                lite_write   <= 1'b1;
                                lite_address <= REG_STREAM_TARGET;
                                lite_data    <= {22'b0, kind, 5'b0, target};
                            end
                        end
                        else if (step == 2'd1) begin
                            step         <= 2'd2;
                            lite_request <= 1'b1;
                            lite_write   <= 1'b1;
                            lite_address <= REG_STREAM_ADDRESS;
                            lite_data    <= {8'b0, address};
                        end
                        else begin
                            state <= STATE_PAYLOAD;
                        end
                    end
                end

                STATE_PAYLOAD: begin
                    if (rx_valid) begin
                        receive_crc <= crc_next(receive_crc, rx_data);
                        if (beat_bytes == '0) begin
                            m_axis_tdata <= {56'b0, rx_data};
                        end
                        else begin
                            m_axis_tdata[8*beat_bytes +: 8] <= rx_data;
                        end
                        beat_bytes <= beat_bytes + 1'b1;
                        item_index <= item_ends ? '0 : item_index + 1'b1;
                        if (item_ends) begin
                            items_left <= items_left - 1'b1;
                        end
                        if (beat_ends) begin
                            beat_bytes   <= '0;
                            m_axis_tlast <= item_ends && items_left == 16'd1;
                            if (rejected) begin
                                state <= (item_ends && items_left == 16'd1) ? STATE_CHECKSUM : STATE_PAYLOAD;
                            end
                            else begin
                                m_axis_tvalid <= 1'b1;
                                state         <= STATE_BEAT;
                            end
                        end
                    end
                end

                STATE_BEAT: begin
                    if (m_axis_tready) begin
                        m_axis_tvalid <= 1'b0;
                        state         <= m_axis_tlast ? STATE_CHECKSUM : STATE_PAYLOAD;
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
                    step     <= '0;
                    if (received_checksum == receive_crc && is_known && !rejected && !is_write
                        && !(command == COMMAND_READ_BUFFER && items_left == '0)) begin
                        lite_request <= 1'b1;
                        lite_write   <= sequence_write;
                        lite_address <= sequence_address;
                        lite_data    <= sequence_data;
                        state        <= STATE_AFTER;
                        if (command == COMMAND_CLEAR) begin
                            error_flag <= 1'b0;
                        end
                    end
                    else begin
                        state <= STATE_SEND_START;
                    end
                end

                STATE_AFTER: begin
                    if (lite_done) begin
                        if (command == COMMAND_STATUS) begin
                            if (step == 2'd0) begin
                                engine_busy <= lite_read_data[0];
                                engine_done <= lite_read_data[1];
                            end
                            else begin
                                engine_cycles <= lite_read_data;
                            end
                        end
                        if (sequence_last) begin
                            state <= STATE_SEND_START;
                        end
                        else begin
                            step <= step + 1'b1;
                        end
                    end
                    else if (!lite_active && !lite_request && step != '0 && !lite_done) begin
                        // the next access of the sequence
                        lite_request <= 1'b1;
                        lite_write   <= sequence_write;
                        lite_address <= sequence_address;
                        lite_data    <= sequence_data;
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
                            tx_valid <= 1'b0;
                            state    <= STATE_READ_BEAT;
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

                STATE_READ_BEAT: begin
                    if (s_axis_tvalid) begin
                        read_item  <= s_axis_tdata;
                        tx_valid   <= 1'b1;
                        tx_data    <= s_axis_tdata[7:0];
                        send_crc   <= crc_next(send_crc, s_axis_tdata[7:0]);
                        send_index <= '0;
                        items_left <= items_left - 1'b1;
                        state      <= STATE_SEND_PAYLOAD;
                    end
                end

                STATE_SEND_PAYLOAD: begin
                    if (tx_ready) begin
                        send_index <= send_index + 1'b1;
                        if (command == COMMAND_STATUS) begin
                            if (send_index == 3'(STATUS_BYTES - 1)) begin
                                tx_data <= send_crc[7:0];
                                state   <= STATE_SEND_CHECKSUM;
                            end
                            else begin
                                tx_data  <= status_bytes[send_index + 1'b1];
                                send_crc <= crc_next(send_crc, status_bytes[send_index + 1'b1]);
                            end
                        end
                        else if (send_index != 3'd7) begin
                            tx_data  <= read_item[8*(send_index + 1'b1) +: 8];
                            send_crc <= crc_next(send_crc, read_item[8*(send_index + 1'b1) +: 8]);
                        end
                        else if (items_left != '0) begin
                            tx_valid <= 1'b0;
                            state    <= STATE_READ_BEAT;
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
