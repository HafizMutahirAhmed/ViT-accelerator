`timescale 1ns / 1ps

module row_op_reader #(
    parameter  int LANES          = vit_config_pkg::SYS_ROW,
    parameter  int ROWS_TOTAL_MAX = vit_config_pkg::ROWS_TOTAL_MAX,
    parameter  int ROW_LENGTH_MAX = vit_config_pkg::COLS_TOTAL_MAX,
    parameter  int BANK_DEPTH     = vit_config_pkg::BANK_DEPTH_MAX,
    parameter  int DATA_LATENCY   = vit_config_pkg::ROW_OP_READ_LATENCY,
    localparam int ROWS_WIDTH     = $clog2(ROWS_TOTAL_MAX + 1),
    localparam int LENGTH_WIDTH   = $clog2(ROW_LENGTH_MAX + 1),
    localparam int ADDRESS_WIDTH  = $clog2(BANK_DEPTH),
    localparam int PASS_WIDTH     = vit_config_pkg::ROW_OP_PASS_WIDTH
) (
    input  logic                     CLK,
    input  logic                     reset,
    input  logic                     start,
    input  logic [ROWS_WIDTH-1:0]    rows_total,
    input  logic [LENGTH_WIDTH-1:0]  row_length,
    input  logic [ROWS_WIDTH-1:0]    row_tiles,
    input  logic [PASS_WIDTH-1:0]    passes,          // passes per tile, minus 1
    input  logic [ADDRESS_WIDTH-1:0] base_a,
    input  logic [ADDRESS_WIDTH-1:0] base_b,
    input  logic                     pass_ready,
    output logic [ADDRESS_WIDTH-1:0] read_address_a,
    output logic [ADDRESS_WIDTH-1:0] read_address_b,
    output logic                     data_valid,
    output logic [PASS_WIDTH-1:0]    data_pass,
    output logic                     data_first,
    output logic                     data_last,
    output logic [LANES-1:0]         data_lane_valid
);

    localparam int REMAINING_WIDTH = $clog2(ROWS_TOTAL_MAX + LANES + 1);

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_READ,   // one address per cycle
        STATE_WAIT    // between two passes of a tile, until pass_ready
    } state_t;

    state_t                     state;
    logic [ADDRESS_WIDTH-1:0]   tile_start_a;
    logic [ADDRESS_WIDTH-1:0]   tile_start_b;
    logic [LENGTH_WIDTH-1:0]    column;
    logic [ROWS_WIDTH-1:0]      tile;
    logic [PASS_WIDTH-1:0]      pass;
    logic [REMAINING_WIDTH-1:0] rows_remaining;   // rows from the current tile on
    logic [LANES-1:0]           lane_valid;

    logic last_column;
    logic last_pass;
    logic last_tile;

    assign last_column = column == row_length - 1'b1;
    assign last_pass   = pass == passes;
    assign last_tile   = tile == row_tiles - 1'b1;

    always_ff @(posedge CLK) begin
        if (reset) begin
            state <= STATE_IDLE;
        end
        else begin
            case (state)
                STATE_IDLE: begin
                    if (start) begin
                        state <= STATE_READ;
                    end
                end
                STATE_READ: begin
                    if (last_column) begin
                        if (!last_pass) begin
                            state <= STATE_WAIT;
                        end
                        else if (last_tile) begin
                            state <= STATE_IDLE;
                        end
                    end
                end
                STATE_WAIT: begin
                    if (pass_ready) begin
                        state <= STATE_READ;
                    end
                end
                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

    always_ff @(posedge CLK) begin
        if (state == STATE_IDLE) begin
            read_address_a <= base_a;
            read_address_b <= base_b;
            tile_start_a   <= base_a;
            tile_start_b   <= base_b;
            column         <= '0;
            tile           <= '0;
            pass           <= '0;
            rows_remaining <= REMAINING_WIDTH'(rows_total);
        end
        else if (state == STATE_READ) begin
            if (!last_column) begin
                read_address_a <= read_address_a + 1'b1;
                read_address_b <= read_address_b + 1'b1;
                column         <= column + 1'b1;
            end
            else begin
                column <= '0;
                if (last_pass) begin
                    // the next tile follows this one in the bank
                    read_address_a <= read_address_a + 1'b1;
                    read_address_b <= read_address_b + 1'b1;
                    tile_start_a   <= read_address_a + 1'b1;
                    tile_start_b   <= read_address_b + 1'b1;
                    tile           <= tile + 1'b1;
                    pass           <= '0;
                    rows_remaining <= rows_remaining - REMAINING_WIDTH'(LANES);
                end
                else begin
                    read_address_a <= tile_start_a;
                    read_address_b <= tile_start_b;
                    pass           <= pass + 1'b1;
                end
            end
        end
    end

    always_comb begin
        for (int lane = 0; lane < LANES; lane++) begin
            lane_valid[lane] = REMAINING_WIDTH'(lane) < rows_remaining;
        end
    end

    // flags, delayed to line up with the read data
    logic [DATA_LATENCY-1:0]                 valid_pipeline;
    logic [DATA_LATENCY-1:0][PASS_WIDTH-1:0] pass_pipeline;
    logic [DATA_LATENCY-1:0]                 first_pipeline;
    logic [DATA_LATENCY-1:0]                 last_pipeline;
    logic [DATA_LATENCY-1:0][LANES-1:0]      lane_valid_pipeline;

    always_ff @(posedge CLK) begin
        if (reset) begin
            valid_pipeline <= '0;
        end
        else begin
            valid_pipeline[0] <= state == STATE_READ;
            for (int stage = 1; stage < DATA_LATENCY; stage++) begin
                valid_pipeline[stage] <= valid_pipeline[stage-1];
            end
        end
        pass_pipeline[0]       <= pass;
        first_pipeline[0]      <= column == '0;
        last_pipeline[0]       <= last_column;
        lane_valid_pipeline[0] <= lane_valid;
        for (int stage = 1; stage < DATA_LATENCY; stage++) begin
            pass_pipeline[stage]       <= pass_pipeline[stage-1];
            first_pipeline[stage]      <= first_pipeline[stage-1];
            last_pipeline[stage]       <= last_pipeline[stage-1];
            lane_valid_pipeline[stage] <= lane_valid_pipeline[stage-1];
        end
    end

    assign data_valid      = valid_pipeline[DATA_LATENCY-1];
    assign data_pass       = pass_pipeline[DATA_LATENCY-1];
    assign data_first      = first_pipeline[DATA_LATENCY-1] && data_valid;
    assign data_last       = last_pipeline[DATA_LATENCY-1] && data_valid;
    assign data_lane_valid = lane_valid_pipeline[DATA_LATENCY-1];

endmodule
