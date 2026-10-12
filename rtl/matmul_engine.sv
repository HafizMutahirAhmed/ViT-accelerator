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
// differ, because a buffer has one read port and one write port.
//
// Row ops (OPCODE_ADD, OPCODE_GELU; see vit_config_pkg) work on tensors in row banks, SYS_ROW
// rows at a time. row_op_reader issues the read addresses, the op's datapath
// processes the read data and row_op_writer writes the result. A row op may
// write to the buffer it reads (in place): a word is written only after it
// was read. The two operands of an add must be in different buffers. Row ops
// reach the pool through registers of their own: a read address register per
// buffer, read data registers and a result register.
//
// Any other opcode is skipped.
//
// Pool buffers have max(SYS_ROW, SYS_COL) banks; the activation side uses the
// first SYS_ROW of them and the weight side the first SYS_COL. BUFFER_DEPTH
// gives the words per bank of each buffer, 32 bits per buffer, buffer 0 in the
// low bits. Array inputs are the buffer read data where the controller's lane
// enable is high and zero elsewhere, registered once together with first_k.
//
// Sequencer: start reads descriptor start_index from the memory, moves it
// through a fetch register into the register bank (so every field the
// datapath uses comes from a plain register, not a memory output), runs
// the op, waits for the last tile and for the drain to finish, then goes on
// to the next descriptor, or stops and pulses done when last_op is set. The
// register bank holds every field stable for the whole op.
//
// Host port. While busy is low the host can write and read the pool
// (host_buffer_id, host_address, one write enable per bank), the bias store
// (host_address, one write enable per column bank) and the descriptor memory
// (host_descriptor_index, one write enable per 32-bit word, word 0 least
// significant). Every host write is registered once after the busy gate, so
// it lands one cycle (vit_config_pkg::POOL_WRITE_LATENCY for the pool) after
// it is presented. Read data is valid vit_config_pkg::BUFFER_READ_LATENCY cycles
// after the address. While busy is high host writes are ignored and the read
// data is not meaningful.
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
    import vit_config_pkg::OPCODE_ADD;
    import vit_config_pkg::OPCODE_GELU;

    localparam int BUFFER_COUNT            = vit_config_pkg::BUFFER_COUNT;
    localparam int BANK_DEPTH_MAX          = vit_config_pkg::BANK_DEPTH_MAX;
    localparam int BIAS_DEPTH              = vit_config_pkg::BIAS_DEPTH;
    localparam int BIAS_ADDRESS_WIDTH      = vit_config_pkg::BIAS_ADDRESS_WIDTH;
    localparam int BUFFER_READ_LATENCY     = vit_config_pkg::BUFFER_READ_LATENCY;
    localparam int POOL_WRITE_LATENCY      = vit_config_pkg::POOL_WRITE_LATENCY;
    // the read address register in front of every buffer is one of the
    // BUFFER_READ_LATENCY stages
    localparam int MEMORY_READ_LATENCY     = BUFFER_READ_LATENCY - 1;
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

    // host write, registered after the busy gate
    logic [DESCRIPTOR_WORDS-1:0]       descriptor_write_enable_reg;
    logic [DESCRIPTOR_INDEX_WIDTH-1:0] descriptor_write_index_reg;
    logic [31:0]                       descriptor_write_data_reg;

    always_ff @(posedge CLK) begin
        if (reset) begin
            descriptor_write_enable_reg <= '0;
        end
        else begin
            descriptor_write_enable_reg <= busy ? '0 : host_descriptor_write_enable;
        end
        descriptor_write_index_reg <= host_descriptor_index;
        descriptor_write_data_reg  <= host_descriptor_write_data;
    end

    genvar word;
    generate
        for (word = 0; word < DESCRIPTOR_WORDS; word++) begin: descriptor_word
            logic [31:0] memory[DESCRIPTOR_COUNT];

            always_ff @(posedge CLK) begin
                if (descriptor_write_enable_reg[word]) begin
                    memory[descriptor_write_index_reg] <= descriptor_write_data_reg;
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
        STATE_LOAD,    // memory output into the fetch register
        STATE_APPLY,   // fetch register into the register bank
        STATE_LAUNCH,  // start the stream controller or the row op
        STATE_RUN,     // matmul: until the last tile is done
        STATE_DRAIN,   // until the drain has written everything
        STATE_ROW_OP   // row op: until its last word is written
    } state_t;

    state_t      state;
    descriptor_t descriptor_fetched;
    descriptor_t descriptor;

    logic controller_start;
    logic all_tiles_done;
    logic drain_busy;

    logic is_row_op;
    logic row_op_start;
    logic row_op_done;
    logic row_op_active;   // registered: high from the cycle after a row op starts until it is done

    assign busy             = state != STATE_IDLE;
    assign controller_start = state == STATE_LAUNCH && descriptor.opcode == OPCODE_MATMUL;
    assign is_row_op        = descriptor.opcode == OPCODE_ADD || descriptor.opcode == OPCODE_GELU;
    assign row_op_start     = state == STATE_LAUNCH && is_row_op;

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
                    descriptor_fetched <= descriptor_t'(descriptor_read_data[DESCRIPTOR_WIDTH-1:0]);
                    state              <= STATE_APPLY;
                end
                STATE_APPLY: begin
                    descriptor <= descriptor_fetched;
                    state      <= STATE_LAUNCH;
                end
                STATE_LAUNCH: begin
                    state <= (descriptor.opcode == OPCODE_MATMUL) ? STATE_RUN
                             : is_row_op ? STATE_ROW_OP : STATE_DRAIN;
                end
                STATE_ROW_OP: begin
                    if (row_op_done) begin
                        state <= STATE_DRAIN;
                    end
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

    always_ff @(posedge CLK) begin
        if (reset) begin
            row_op_active <= 1'b0;
        end
        else if (row_op_start) begin
            row_op_active <= 1'b1;
        end
        else if (row_op_done) begin
            row_op_active <= 1'b0;
        end
    end

    // simulation checks, per opcode: a buffer has one read port and one write port
    always_ff @(posedge CLK) begin
        if (controller_start) begin
            // matmul: no buffer can have two roles
            assert (descriptor.activation_buffer_id != descriptor.weight_buffer_id
                    && descriptor.activation_buffer_id != descriptor.output_buffer_id
                    && descriptor.weight_buffer_id != descriptor.output_buffer_id)
            else $error("matmul_engine: descriptor %0d uses a buffer id twice (activation %0d, weight %0d, output %0d)",
                        descriptor_index, descriptor.activation_buffer_id,
                        descriptor.weight_buffer_id, descriptor.output_buffer_id);
        end
        if (row_op_start && descriptor.opcode == OPCODE_ADD) begin
            // add: two reads at once need two buffers; the output may be either of them
            assert (descriptor.activation_buffer_id != descriptor.weight_buffer_id)
            else $error("matmul_engine: add descriptor %0d reads both operands from buffer %0d",
                        descriptor_index, descriptor.activation_buffer_id);
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
        .SYS_COL             (SYS_COL),
        .BANK_DEPTH          (BANK_DEPTH_MAX),
        .READ_LATENCY        (BUFFER_READ_LATENCY),
        .ARRAY_INPUT_LATENCY (1)
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
    // row ops: reader, datapath, writer
    // ------------------------------------------------------------------
    localparam int ROW_OP_PASS_WIDTH = vit_config_pkg::ROW_OP_PASS_WIDTH;

    logic [ADDRESS_WIDTH-1:0]                  row_read_address_a;
    logic [ADDRESS_WIDTH-1:0]                  row_read_address_b;
    logic                                      row_data_valid;
    logic [ROW_OP_PASS_WIDTH-1:0]              row_data_pass;
    logic                                      row_data_first;
    logic                                      row_data_last;
    logic [SYS_ROW-1:0]                        row_data_lane_valid;
    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0] row_data_a;
    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0] row_data_b;
    logic                                      row_result_valid;
    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0] row_result_data;
    logic [SYS_ROW-1:0]                        row_write_enable;
    logic [ADDRESS_WIDTH-1:0]                  row_write_address;
    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0] row_write_data;

    row_op_reader #(
        .LANES        (SYS_ROW),
        .BANK_DEPTH   (BANK_DEPTH_MAX),
        .DATA_LATENCY (vit_config_pkg::ROW_OP_READ_LATENCY)
    ) row_reader (
        .CLK             (CLK),
        .reset           (reset),
        .start           (row_op_start),
        .rows_total      (descriptor.rows_total),
        .row_length      (descriptor.cols_total),
        .row_tiles       (descriptor.row_tiles),
        .passes          ('0),
        .base_a          (descriptor.activation_base),
        .base_b          (descriptor.weight_base),
        .pass_ready      (1'b1),
        .read_address_a  (row_read_address_a),
        .read_address_b  (row_read_address_b),
        .data_valid      (row_data_valid),
        .data_pass       (row_data_pass),
        .data_first      (row_data_first),
        .data_last       (row_data_last),
        .data_lane_valid (row_data_lane_valid)
    );

    logic                                      add_valid;
    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0] add_sum;

    residual_add #(
        .LANES      (SYS_ROW),
        .DATA_WIDTH (DATA_WIDTH)
    ) add_inst (
        .CLK          (CLK),
        .reset        (reset),
        .valid_in     (row_data_valid && descriptor.opcode == OPCODE_ADD),
        .a_in         (row_data_a),
        .b_in         (row_data_b),
        .multiplier_a (descriptor.scale),
        .multiplier_b (descriptor.bias_base[vit_config_pkg::SCALE_WIDTH-1:0]),
        .shift_value  (descriptor.shift),
        .valid_out    (add_valid),
        .sum_out      (add_sum)
    );

    localparam int GELU_SELECT_WIDTH = (vit_config_pkg::GELU_TABLE_COUNT > 1)
                                       ? $clog2(vit_config_pkg::GELU_TABLE_COUNT) : 1;

    logic                                      gelu_valid;
    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0] gelu_result;

    gelu_lut #(
        .LANES      (SYS_ROW),
        .DATA_WIDTH (DATA_WIDTH)
    ) gelu_inst (
        .CLK          (CLK),
        .reset        (reset),
        .valid_in     (row_data_valid && descriptor.opcode == OPCODE_GELU),
        .x_in         (row_data_a),
        .table_select (descriptor.bias_base[GELU_SELECT_WIDTH-1:0]),
        .valid_out    (gelu_valid),
        .y_out        (gelu_result)
    );

    // result register: the datapath of the running row op
    always_ff @(posedge CLK) begin
        if (reset) begin
            row_result_valid <= 1'b0;
        end
        else begin
            row_result_valid <= add_valid || gelu_valid;
        end
        row_result_data <= (descriptor.opcode == OPCODE_GELU) ? gelu_result : add_sum;
    end

    row_op_writer #(
        .LANES      (SYS_ROW),
        .DATA_WIDTH (DATA_WIDTH),
        .BANK_DEPTH (BANK_DEPTH_MAX),
        .DONE_DELAY (POOL_WRITE_LATENCY + 1)
    ) row_writer (
        .CLK           (CLK),
        .reset         (reset),
        .start         (row_op_start),
        .rows_total    (descriptor.rows_total),
        .row_length    (descriptor.cols_total),
        .row_tiles     (descriptor.row_tiles),
        .output_base   (descriptor.output_base),
        .output_stride (descriptor.output_stride),
        .result_valid  (row_result_valid),
        .result_data   (row_result_data),
        .write_enable  (row_write_enable),
        .write_address (row_write_address),
        .write_data    (row_write_data),
        .done          (row_op_done)
    );

    // ------------------------------------------------------------------
    // buffer pool
    // ------------------------------------------------------------------
    logic [LANES-1:0]                                   drain_write_enable;
    logic [ADDRESS_WIDTH-1:0]                           drain_write_address;
    logic signed [LANES-1:0][DATA_WIDTH-1:0]            drain_write_data;
    logic [BUFFER_COUNT-1:0][LANES-1:0][DATA_WIDTH-1:0] pool_read_data;

    // row op read data: registered multiplexers over the pool
    always_ff @(posedge CLK) begin
        for (int i = 0; i < SYS_ROW; i++) begin
            row_data_a[i] <= pool_read_data[descriptor.activation_buffer_id][i];
            row_data_b[i] <= pool_read_data[descriptor.weight_buffer_id][i];
        end
    end

    genvar buffer;
    generate
        for (buffer = 0; buffer < BUFFER_COUNT; buffer++) begin: pool
            localparam int DEPTH               = BUFFER_DEPTH[buffer];
            localparam int DEPTH_ADDRESS_WIDTH = $clog2(DEPTH);

            logic [LANES-1:0]                 write_enable;
            logic [DEPTH_ADDRESS_WIDTH-1:0]   write_address;
            logic [LANES-1:0][DATA_WIDTH-1:0] write_data;
            logic [DEPTH_ADDRESS_WIDTH-1:0]   read_address;

            // row op: this buffer's read address, registered. The second
            // operand's buffer gets its address, every other buffer the first's.
            logic [DEPTH_ADDRESS_WIDTH-1:0] row_read_address_reg;
            logic                           row_write_select;

            always_ff @(posedge CLK) begin
                row_read_address_reg <= (descriptor.weight_buffer_id == BUFFER_ID_WIDTH'(buffer))
                                        ? row_read_address_b[DEPTH_ADDRESS_WIDTH-1:0]
                                        : row_read_address_a[DEPTH_ADDRESS_WIDTH-1:0];
                row_write_select     <= descriptor.output_buffer_id == BUFFER_ID_WIDTH'(buffer);
            end

            always_comb begin
                if (row_op_active) begin
                    write_enable  = '0;
                    write_data    = '0;
                    write_enable[SYS_ROW-1:0] = row_write_select ? row_write_enable : '0;
                    write_data[SYS_ROW-1:0]   = row_write_data;
                    write_address = row_write_address[DEPTH_ADDRESS_WIDTH-1:0];
                    read_address  = row_read_address_reg;
                end
                else if (busy) begin
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

            // read address register, one per buffer, right before the block RAM
            logic [DEPTH_ADDRESS_WIDTH-1:0] read_address_reg;

            always_ff @(posedge CLK) begin
                read_address_reg <= read_address;
            end

            // write port registers, one set per buffer, right before the block RAM
            logic [POOL_WRITE_LATENCY-1:0][LANES-1:0]                 write_enable_pipeline;
            logic [POOL_WRITE_LATENCY-1:0][DEPTH_ADDRESS_WIDTH-1:0]   write_address_pipeline;
            logic [POOL_WRITE_LATENCY-1:0][LANES-1:0][DATA_WIDTH-1:0] write_data_pipeline;

            always_ff @(posedge CLK) begin
                if (reset) begin
                    write_enable_pipeline <= '0;
                end
                else begin
                    write_enable_pipeline[0] <= write_enable;
                    for (int stage = 1; stage < POOL_WRITE_LATENCY; stage++) begin
                        write_enable_pipeline[stage] <= write_enable_pipeline[stage-1];
                    end
                end
                write_address_pipeline[0] <= write_address;
                write_data_pipeline[0]    <= write_data;
                for (int stage = 1; stage < POOL_WRITE_LATENCY; stage++) begin
                    write_address_pipeline[stage] <= write_address_pipeline[stage-1];
                    write_data_pipeline[stage]    <= write_data_pipeline[stage-1];
                end
            end

            operand_buffer #(
                .SYS_LANES    (LANES),
                .BANK_DEPTH   (DEPTH),
                .DATA_WIDTH   (DATA_WIDTH),
                .READ_LATENCY (MEMORY_READ_LATENCY)
            ) buffer_inst (
                .CLK           (CLK),
                .write_enable  (write_enable_pipeline[POOL_WRITE_LATENCY-1]),
                .write_address ({LANES{write_address_pipeline[POOL_WRITE_LATENCY-1]}}),
                .write_data    (write_data_pipeline[POOL_WRITE_LATENCY-1]),
                .read_address  (read_address_reg),
                .read_data     (pool_read_data[buffer])
            );
        end
    endgenerate

    assign host_buffer_read_data = pool_read_data[host_buffer_id];

    // ------------------------------------------------------------------
    // bias store: column bank c holds bias[c], bias[c + SYS_COL], ...
    // ------------------------------------------------------------------
    logic [BIAS_ADDRESS_WIDTH-1:0]      bias_address;
    logic [BIAS_ADDRESS_WIDTH-1:0]      bias_read_address_reg;
    logic [SYS_COL-1:0][ACC_WIDTH-1:0]  bias_read_data;

    assign bias_address = busy ? descriptor.bias_base + BIAS_ADDRESS_WIDTH'(tile_col)
                               : host_address[BIAS_ADDRESS_WIDTH-1:0];

    always_ff @(posedge CLK) begin
        bias_read_address_reg <= bias_address;
    end

    // host write, registered after the busy gate
    logic [SYS_COL-1:0]                bias_write_enable_reg;
    logic [BIAS_ADDRESS_WIDTH-1:0]     bias_write_address_reg;
    logic [SYS_COL-1:0][ACC_WIDTH-1:0] bias_write_data_reg;

    always_ff @(posedge CLK) begin
        if (reset) begin
            bias_write_enable_reg <= '0;
        end
        else begin
            bias_write_enable_reg <= busy ? '0 : host_bias_write_enable;
        end
        bias_write_address_reg <= host_address[BIAS_ADDRESS_WIDTH-1:0];
        bias_write_data_reg    <= host_bias_write_data;
    end

    operand_buffer #(
        .SYS_LANES    (SYS_COL),
        .BANK_DEPTH   (BIAS_DEPTH),
        .DATA_WIDTH   (ACC_WIDTH),
        .READ_LATENCY (MEMORY_READ_LATENCY)
    ) bias_store (
        .CLK           (CLK),
        .write_enable  (bias_write_enable_reg),
        .write_address ({SYS_COL{bias_write_address_reg}}),
        .write_data    (bias_write_data_reg),
        .read_address  (bias_read_address_reg),
        .read_data     (bias_read_data)
    );

    assign host_bias_read_data = bias_read_data;

    // ------------------------------------------------------------------
    // lane masking and the array
    // ------------------------------------------------------------------
    logic signed [SYS_ROW-1:0][DATA_WIDTH-1:0]             activation_in;
    logic signed [SYS_COL-1:0][DATA_WIDTH-1:0]             weight_in;
    logic                                                  first_k_array;
    logic signed [SYS_ROW-1:0][SYS_COL-1:0][ACC_WIDTH-1:0] accumulator_out;

    // array input register, after the buffer id multiplexer and the lane mask
    always_ff @(posedge CLK) begin
        for (int i = 0; i < SYS_ROW; i++) begin
            activation_in[i] <= activation_lane_enable[i] ? pool_read_data[descriptor.activation_buffer_id][i] : '0;
        end
        for (int i = 0; i < SYS_COL; i++) begin
            weight_in[i] <= weight_lane_enable[i] ? pool_read_data[descriptor.weight_buffer_id][i] : '0;
        end
        first_k_array <= first_k;
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
        .first_k_in      (first_k_array),
        .accumulator_out (accumulator_out)
    );

    // ------------------------------------------------------------------
    // drain
    // ------------------------------------------------------------------
    drain #(
        .SYS_ROW    (SYS_ROW),
        .SYS_COL    (SYS_COL),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH     (ACC_WIDTH),
        .BANK_DEPTH    (BANK_DEPTH_MAX),
        .WRITE_LATENCY (POOL_WRITE_LATENCY)
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
