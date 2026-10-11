`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: matmul_engine
// Description: Runs a table of matmul ops on the systolic array.
//
// Contents: the descriptor memory and register bank, the sequencer, a pool of
// BUFFER_COUNT operand buffers, the bias store, matmul_stream_controller,
// systolic_array and drain (with its requantizers).
//
// One op is a descriptor_t (see vit_config_pkg). For an OPCODE_MATMUL op the
// engine computes, in INT8 with INT32 accumulation,
//   output = clip(((activation @ weight + bias) * scale) >>> shift)
// reading the activation operand from the row banks of buffer
// activation_buffer_id, the weight operand from the column banks of buffer
// weight_buffer_id and writing the result to buffer output_buffer_id in the
// layout picked by drain_mode. An op's output buffer is used by a later op
// simply by naming its id: nothing is copied. The three ids of an op must
// differ, because a buffer has one read port and one write port. Other
// opcodes are skipped.
//
// Pool buffers have max(SYS_ROW, SYS_COL) banks; the activation side uses the
// first SYS_ROW of them and the weight side the first SYS_COL. BUFFER_DEPTH
// gives the words per bank of each buffer, 32 bits per buffer, buffer 0 in the
// low bits. Array inputs are the buffer read data where the controller's lane
// enable is high and zero elsewhere.
//
// Sequencer: start loads descriptor start_index into the register bank, runs
// the op, waits for the last tile and for the drain to finish, then goes on
// to the next descriptor, or stops and pulses done when last_op is set. The
// register bank holds every field stable for the whole op.
//
// Host port. While busy is low the host can write and read the pool
// (host_buffer_id, host_address, one write enable per bank), the bias store
// (host_address, one write enable per column bank) and the descriptor memory
// (host_descriptor_index, one write enable per 32-bit word, word 0 least
// significant). Read data is valid one cycle after the address. While busy is
// high host writes are ignored and the read data is not meaningful.
//
// The descriptor field widths, BUFFER_COUNT, BIAS_DEPTH and DESCRIPTOR_COUNT
// come from vit_config_pkg and cannot be overridden per instance.
//////////////////////////////////////////////////////////////////////////////////

module matmul_engine #(
    parameter  int SYS_ROW    = vit_config_pkg::SYS_ROW,
    parameter  int SYS_COL    = vit_config_pkg::SYS_COL,
    parameter  int DATA_WIDTH = vit_config_pkg::DATA_WIDTH,
    parameter  int ACC_WIDTH  = vit_config_pkg::ACC_WIDTH,
    parameter  bit [vit_config_pkg::BUFFER_COUNT-1:0][31:0] BUFFER_DEPTH
                              = {vit_config_pkg::BUFFER_COUNT{32'(vit_config_pkg::BANK_DEPTH)}},
    localparam int LANES                  = (SYS_ROW > SYS_COL) ? SYS_ROW : SYS_COL,
    localparam int BUFFER_ID_WIDTH        = vit_config_pkg::BUFFER_ID_WIDTH,
    localparam int ADDRESS_WIDTH          = vit_config_pkg::ADDRESS_WIDTH,
    localparam int DESCRIPTOR_WORDS       = vit_config_pkg::DESCRIPTOR_WORDS,
    localparam int DESCRIPTOR_INDEX_WIDTH = $clog2(vit_config_pkg::DESCRIPTOR_COUNT)
) (
    input  logic                                   CLK,
    input  logic                                   reset,

    input  logic                                   start,
    input  logic [DESCRIPTOR_INDEX_WIDTH-1:0]      start_index,
    output logic                                   busy,
    output logic                                   done,

    input  logic [DESCRIPTOR_INDEX_WIDTH-1:0]      host_descriptor_index,
    input  logic [DESCRIPTOR_WORDS-1:0]            host_descriptor_write_enable,
    input  logic [31:0]                            host_descriptor_write_data,

    input  logic [BUFFER_ID_WIDTH-1:0]             host_buffer_id,
    input  logic [ADDRESS_WIDTH-1:0]               host_address,
    input  logic [LANES-1:0]                       host_buffer_write_enable,
    input  logic [LANES-1:0][DATA_WIDTH-1:0]       host_buffer_write_data,
    output logic [LANES-1:0][DATA_WIDTH-1:0]       host_buffer_read_data,

    input  logic [SYS_COL-1:0]                     host_bias_write_enable,
    input  logic [SYS_COL-1:0][ACC_WIDTH-1:0]      host_bias_write_data,
    output logic [SYS_COL-1:0][ACC_WIDTH-1:0]      host_bias_read_data
);

    import vit_config_pkg::descriptor_t;
    import vit_config_pkg::OPCODE_MATMUL;

    localparam int BUFFER_COUNT            = vit_config_pkg::BUFFER_COUNT;
    localparam int BANK_DEPTH_MAX          = vit_config_pkg::BANK_DEPTH_MAX;
    localparam int BIAS_DEPTH              = vit_config_pkg::BIAS_DEPTH;
    localparam int BIAS_ADDRESS_WIDTH      = vit_config_pkg::BIAS_ADDRESS_WIDTH;
    localparam int DESCRIPTOR_COUNT        = vit_config_pkg::DESCRIPTOR_COUNT;
    localparam int DESCRIPTOR_WIDTH        = vit_config_pkg::DESCRIPTOR_WIDTH;
    localparam int DESCRIPTOR_MEMORY_WIDTH = vit_config_pkg::DESCRIPTOR_MEMORY_WIDTH;
    localparam int ROWS_WIDTH              = vit_config_pkg::ROWS_WIDTH;
    localparam int COLS_WIDTH              = vit_config_pkg::COLS_WIDTH;

    // ------------------------------------------------------------------
    // descriptor memory: one 32-bit wide RAM per descriptor word
    // ------------------------------------------------------------------
    logic [DESCRIPTOR_INDEX_WIDTH-1:0]  descriptor_index;
    logic [DESCRIPTOR_MEMORY_WIDTH-1:0] descriptor_read_data;

    genvar word;
    generate
        for (word = 0; word < DESCRIPTOR_WORDS; word++) begin: descriptor_word
            logic [31:0] memory[DESCRIPTOR_COUNT];

            always_ff @(posedge CLK) begin
                if (!busy && host_descriptor_write_enable[word]) begin
                    memory[host_descriptor_index] <= host_descriptor_write_data;
                end
                descriptor_read_data[32*word +: 32] <= memory[descriptor_index];
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // sequencer and descriptor register bank
    // ------------------------------------------------------------------
    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_FETCH,   // descriptor memory read
        STATE_LOAD,    // register bank load
        STATE_LAUNCH,  // start the stream controller
        STATE_RUN,     // until the last tile is done
        STATE_DRAIN    // until the drain has written everything
    } state_t;

    state_t      state;
    descriptor_t descriptor;

    logic controller_start;
    logic all_tiles_done;
    logic drain_busy;

    assign busy             = state != STATE_IDLE;
    assign controller_start = state == STATE_LAUNCH && descriptor.opcode == OPCODE_MATMUL;

    always_ff @(posedge CLK) begin
        if (reset) begin
            state <= STATE_IDLE;
            done  <= 1'b0;
        end
        else begin
            done <= 1'b0;
            case (state)
                STATE_IDLE: begin
                    if (start) begin
                        descriptor_index <= start_index;
                        state            <= STATE_FETCH;
                    end
                end
                STATE_FETCH: begin
                    state <= STATE_LOAD;
                end
                STATE_LOAD: begin
                    descriptor <= descriptor_t'(descriptor_read_data[DESCRIPTOR_WIDTH-1:0]);
                    state      <= STATE_LAUNCH;
                end
                STATE_LAUNCH: begin
                    state <= (descriptor.opcode == OPCODE_MATMUL) ? STATE_RUN : STATE_DRAIN;
                end
                STATE_RUN: begin
                    if (all_tiles_done) begin
                        state <= STATE_DRAIN;
                    end
                end
                STATE_DRAIN: begin
                    if (!drain_busy) begin
                        if (descriptor.last_op) begin
                            state <= STATE_IDLE;
                            done  <= 1'b1;
                        end
                        else begin
                            descriptor_index <= descriptor_index + 1'b1;
                            state            <= STATE_FETCH;
                        end
                    end
                end
                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

    // simulation check: a buffer cannot have two roles in one op
    always_ff @(posedge CLK) begin
        if (controller_start) begin
            assert (descriptor.activation_buffer_id != descriptor.weight_buffer_id
                    && descriptor.activation_buffer_id != descriptor.output_buffer_id
                    && descriptor.weight_buffer_id != descriptor.output_buffer_id)
            else $error("matmul_engine: descriptor %0d uses a buffer id twice (activation %0d, weight %0d, output %0d)",
                        descriptor_index, descriptor.activation_buffer_id,
                        descriptor.weight_buffer_id, descriptor.output_buffer_id);
        end
    end

    // ------------------------------------------------------------------
    // stream controller
    // ------------------------------------------------------------------
    logic [ADDRESS_WIDTH-1:0] activation_read_address;
    logic [ADDRESS_WIDTH-1:0] weight_read_address;
    logic [SYS_ROW-1:0]       activation_lane_enable;
    logic [SYS_COL-1:0]       weight_lane_enable;
    logic                     first_k;
    logic [ROWS_WIDTH-1:0]    tile_row;
    logic [COLS_WIDTH-1:0]    tile_col;
    logic                     tile_done;

    matmul_stream_controller #(
        .SYS_ROW    (SYS_ROW),
        .SYS_COL    (SYS_COL),
        .BANK_DEPTH (BANK_DEPTH_MAX)
    ) controller (
        .CLK                     (CLK),
        .reset                   (reset),
        .start                   (controller_start),
        .rows_total              (descriptor.rows_total),
        .inner_dim               (descriptor.inner_dim),
        .cols_total              (descriptor.cols_total),
        .row_tiles               (descriptor.row_tiles),
        .col_tiles               (descriptor.col_tiles),
        .activation_base         (descriptor.activation_base),
        .weight_base             (descriptor.weight_base),
        .activation_read_address (activation_read_address),
        .weight_read_address     (weight_read_address),
        .activation_lane_enable  (activation_lane_enable),
        .weight_lane_enable      (weight_lane_enable),
        .first_k_out             (first_k),
        .tile_row                (tile_row),
        .tile_col                (tile_col),
        .tile_done               (tile_done),
        .all_tiles_done          (all_tiles_done)
    );

    // ------------------------------------------------------------------
    // buffer pool
    // ------------------------------------------------------------------
    logic [LANES-1:0]                                   drain_write_enable;
    logic [ADDRESS_WIDTH-1:0]                           drain_write_address;
    logic signed [LANES-1:0][DATA_WIDTH-1:0]            drain_write_data;
    logic [BUFFER_COUNT-1:0][LANES-1:0][DATA_WIDTH-1:0] pool_read_data;

    genvar buffer;
    generate
        for (buffer = 0; buffer < BUFFER_COUNT; buffer++) begin: pool
            localparam int DEPTH               = BUFFER_DEPTH[buffer];
            localparam int DEPTH_ADDRESS_WIDTH = $clog2(DEPTH);

            logic [LANES-1:0]                 write_enable;
            logic [DEPTH_ADDRESS_WIDTH-1:0]   write_address;
            logic [LANES-1:0][DATA_WIDTH-1:0] write_data;
            logic [DEPTH_ADDRESS_WIDTH-1:0]   read_address;

            always_comb begin
                if (busy) begin
                    // writes come from the drain, gated by the output buffer id
                    write_enable  = (descriptor.output_buffer_id == BUFFER_ID_WIDTH'(buffer)) ? drain_write_enable : '0;
                    write_address = drain_write_address[DEPTH_ADDRESS_WIDTH-1:0];
                    write_data    = drain_write_data;
                    read_address  = (descriptor.activation_buffer_id == BUFFER_ID_WIDTH'(buffer))
                                    ? activation_read_address[DEPTH_ADDRESS_WIDTH-1:0]
                                    : weight_read_address[DEPTH_ADDRESS_WIDTH-1:0];
                end
                else begin
                    write_enable  = (host_buffer_id == BUFFER_ID_WIDTH'(buffer)) ? host_buffer_write_enable : '0;
                    write_address = host_address[DEPTH_ADDRESS_WIDTH-1:0];
                    write_data    = host_buffer_write_data;
                    read_address  = host_address[DEPTH_ADDRESS_WIDTH-1:0];
                end
            end

            operand_buffer #(
                .SYS_LANES  (LANES),
                .BANK_DEPTH (DEPTH),
                .DATA_WIDTH (DATA_WIDTH)
            ) buffer_inst (
                .CLK           (CLK),
                .write_enable  (write_enable),
                .write_address ({LANES{write_address}}),
                .write_data    (write_data),
                .read_address  (read_address),
                .read_data     (pool_read_data[buffer])
            );
        end
    endgenerate

    assign host_buffer_read_data = pool_read_data[host_buffer_id];

    // ------------------------------------------------------------------
    // bias store: column bank c holds bias[c], bias[c + SYS_COL], ...
    // ------------------------------------------------------------------
    logic [BIAS_ADDRESS_WIDTH-1:0]      bias_address;
    logic [SYS_COL-1:0][ACC_WIDTH-1:0]  bias_read_data;

    assign bias_address = busy ? descriptor.bias_base + BIAS_ADDRESS_WIDTH'(tile_col)
                               : host_address[BIAS_ADDRESS_WIDTH-1:0];

    operand_buffer #(
        .SYS_LANES  (SYS_COL),
        .BANK_DEPTH (BIAS_DEPTH),
        .DATA_WIDTH (ACC_WIDTH)
    ) bias_store (
        .CLK           (CLK),
        .write_enable  (busy ? '0 : host_bias_write_enable),
        .write_address ({SYS_COL{bias_address}}),
        .write_data    (host_bias_write_data),
        .read_address  (bias_address),
        .read_data     (bias_read_data)
    );

    assign host_bias_read_data = bias_read_data;

    // ------------------------------------------------------------------
    // lane masking and the array
    // ------------------------------------------------------------------
    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0]             activation_in;
    logic signed [SYS_COL-1:0][DATA_WIDTH-1:0]             weight_in;
    logic signed [SYS_ROW-1:0][SYS_COL-1:0][ACC_WIDTH-1:0] accumulator_out;

    always_comb begin
        for (int i = 0; i < SYS_ROW; i++) begin
            activation_in[i] = activation_lane_enable[i] ? pool_read_data[descriptor.activation_buffer_id][i] : '0;
        end
        for (int i = 0; i < SYS_COL; i++) begin
            weight_in[i] = weight_lane_enable[i] ? pool_read_data[descriptor.weight_buffer_id][i] : '0;
        end
    end

    systolic_array #(
        .SYS_ROW    (SYS_ROW),
        .SYS_COL    (SYS_COL),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) array (
        .CLK             (CLK),
        .activation_in   (activation_in),
        .weight_in       (weight_in),
        .first_k_in      (first_k),
        .accumulator_out (accumulator_out)
    );

    // ------------------------------------------------------------------
    // drain
    // ------------------------------------------------------------------
    drain #(
        .SYS_ROW    (SYS_ROW),
        .SYS_COL    (SYS_COL),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .BANK_DEPTH (BANK_DEPTH_MAX)
    ) drain_inst (
        .CLK            (CLK),
        .reset          (reset),
        .drain_mode     (descriptor.drain_mode),
        .bias_enable    (descriptor.bias_enable),
        .tile_done      (tile_done),
        .tile_row       (tile_row),
        .tile_col       (tile_col),
        .rows_total     (descriptor.rows_total),
        .cols_total     (descriptor.cols_total),
        .output_base    (descriptor.output_base),
        .output_stride  (descriptor.output_stride),
        .accumulator_in (accumulator_out),
        .bias_in        (bias_read_data),
        .scale          (descriptor.scale),
        .shift_value    (descriptor.shift),
        .write_enable   (drain_write_enable),
        .write_address  (drain_write_address),
        .write_data     (drain_write_data),
        .drain_busy     (drain_busy)
    );

endmodule
