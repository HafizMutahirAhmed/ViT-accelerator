`timescale 1ns / 1ps

module layernorm #(
    parameter  int    LANES          = vit_config_pkg::SYS_ROW,
    parameter  int    DATA_WIDTH     = vit_config_pkg::DATA_WIDTH,
    parameter  int    ROW_LENGTH_MAX = vit_config_pkg::COLS_TOTAL_MAX,
    parameter  int    SHIFT_WIDTH    = vit_config_pkg::SHIFT_WIDTH,
    parameter  string ROM_FILE       = "rom/rsqrt_rom.hex",
    localparam int    LATENCY        = vit_config_pkg::LAYERNORM_LATENCY,
    localparam int    GAIN_WIDTH     = vit_config_pkg::LAYERNORM_GAIN_WIDTH,
    localparam int    OFFSET_WIDTH   = vit_config_pkg::LAYERNORM_OFFSET_WIDTH,
    localparam int    CONSTANT_DEPTH = vit_config_pkg::LAYERNORM_CONSTANT_DEPTH,
    localparam int    EPSILON_WIDTH  = vit_config_pkg::INNER_WIDTH,
    localparam int    LENGTH_WIDTH   = $clog2(ROW_LENGTH_MAX + 1),
    localparam int    CONSTANT_ADDRESS_WIDTH = $clog2(CONSTANT_DEPTH)
) (
    input  logic                                    CLK,
    input  logic                                    reset,

    input  logic                                    constants_write_enable,
    input  logic [CONSTANT_ADDRESS_WIDTH-1:0]       constants_write_address,
    input  logic signed [GAIN_WIDTH-1:0]            constants_write_gain,
    input  logic signed [OFFSET_WIDTH-1:0]          constants_write_offset,

    input  logic [LENGTH_WIDTH-1:0]                 row_length,
    input  logic [CONSTANT_ADDRESS_WIDTH-1:0]       constants_base,
    input  logic [SHIFT_WIDTH-1:0]                  out_shift,
    input  logic [EPSILON_WIDTH-1:0]                epsilon,

    input  logic                                    valid_in,
    input  logic                                    pass_in,
    input  logic                                    first_in,
    input  logic                                    last_in,
    input  logic signed [LANES-1:0][DATA_WIDTH-1:0] x_in,
    output logic                                    pass_ready,
    output logic                                    valid_out,
    output logic signed [LANES-1:0][DATA_WIDTH-1:0] y_out
);

    localparam int RSQRT_INDEX_BITS = vit_config_pkg::LAYERNORM_RSQRT_INDEX_BITS;   // 8
    localparam int RSQRT_BITS       = vit_config_pkg::LAYERNORM_RSQRT_BITS;         // 12
    localparam int NORM_FRAC        = vit_config_pkg::LAYERNORM_NORM_FRAC;          // 8

    localparam int TOTAL_WIDTH     = 2 * DATA_WIDTH;                  // signed sum of up to 256 INT8
    localparam int SQUARES_WIDTH   = 3 * DATA_WIDTH - 2;              // 256 * 2**14
    localparam int VARIANCE_WIDTH  = 30;
    localparam int EXPONENT_WIDTH  = $clog2(VARIANCE_WIDTH / 2);
    localparam int RSQRT_WIDTH     = RSQRT_BITS + 1;
    localparam int CENTRED_WIDTH   = 2 * DATA_WIDTH + 2;              // n * x - total
    localparam int PRODUCT_WIDTH   = CENTRED_WIDTH + RSQRT_WIDTH + 1;
    localparam int NORM_WIDTH      = NORM_FRAC + 6;                   // below 16 in magnitude, and a spare bit
    localparam int NORM_SHIFT_BASE = RSQRT_BITS - NORM_FRAC;
    localparam int NORM_SHIFT_WIDTH = $clog2(NORM_SHIFT_BASE + VARIANCE_WIDTH / 2 + 1);
    localparam int SCALED_WIDTH    = ((NORM_WIDTH + GAIN_WIDTH > OFFSET_WIDTH) ? NORM_WIDTH + GAIN_WIDTH : OFFSET_WIDTH) + 1;
    localparam int WIDE            = 2 * TOTAL_WIDTH + 2;             // n * squares, total * total and their difference

    // stages from the last pass 0 input to total and squares being final
    localparam int ACCUMULATE_STAGES = 3;
    // stages from there to the ROM output: operands, products, difference,
    // clamp, exponent, normalise, ROM address, ROM read
    localparam int STATISTICS_STAGES = 8;

    localparam logic signed [SCALED_WIDTH-1:0] DATA_MAX = (SCALED_WIDTH'(1) << (DATA_WIDTH - 1)) - 1;
    localparam logic signed [SCALED_WIDTH-1:0] DATA_MIN = -(SCALED_WIDTH'(1) << (DATA_WIDTH - 1));

    // control: flags travelling with the data
    logic [LATENCY-1:0]                                     output_valid_pipeline;
    logic [ACCUMULATE_STAGES-1:0]                           statistics_valid_pipeline;
    logic [ACCUMULATE_STAGES-1:0]                           statistics_first_pipeline;
    logic [ACCUMULATE_STAGES+STATISTICS_STAGES-1:0]         statistics_last_pipeline;
    logic                                                   statistics_done;

    assign statistics_done = statistics_last_pipeline[ACCUMULATE_STAGES+STATISTICS_STAGES-1];

    always_ff @(posedge CLK) begin
        if (reset) begin
            output_valid_pipeline     <= '0;
            statistics_valid_pipeline <= '0;
            statistics_last_pipeline  <= '0;
            pass_ready                <= 1'b0;
        end
        else begin
            output_valid_pipeline     <= LATENCY'({output_valid_pipeline, valid_in && pass_in});
            statistics_valid_pipeline <= ACCUMULATE_STAGES'({statistics_valid_pipeline, valid_in && !pass_in});
            statistics_last_pipeline  <= (ACCUMULATE_STAGES+STATISTICS_STAGES)'({statistics_last_pipeline,
                                                                                 valid_in && !pass_in && last_in});
            if (valid_in && !pass_in) begin
                pass_ready <= 1'b0;
            end
            else if (statistics_done) begin
                pass_ready <= 1'b1;
            end
        end
        statistics_first_pipeline <= ACCUMULATE_STAGES'({statistics_first_pipeline, first_in});
    end

    assign valid_out = output_valid_pipeline[LATENCY-1];

    // constants memory: gain and offset per column, one read for all lanes
    logic [GAIN_WIDTH+OFFSET_WIDTH-1:0] constants_memory[CONSTANT_DEPTH];
    logic [CONSTANT_ADDRESS_WIDTH-1:0]  constants_address;      // stage 1
    logic [GAIN_WIDTH+OFFSET_WIDTH-1:0] constants_read;         // stage 2
    logic [GAIN_WIDTH+OFFSET_WIDTH-1:0] constants_out;          // stage 3
    logic signed [1:0][GAIN_WIDTH-1:0]   gain_pipeline;         // stages 4, 5
    logic signed [2:0][OFFSET_WIDTH-1:0] offset_pipeline;       // stages 4, 5, 6

    always_ff @(posedge CLK) begin
        if (constants_write_enable) begin
            constants_memory[constants_write_address] <= {constants_write_offset, constants_write_gain};
        end
        if (valid_in && pass_in) begin
            constants_address <= first_in ? constants_base : constants_address + 1'b1;
        end
        constants_read  <= constants_memory[constants_address];
        constants_out   <= constants_read;
        gain_pipeline   <= {gain_pipeline[0], constants_out[GAIN_WIDTH-1:0]};
        offset_pipeline <= {offset_pipeline[1:0], constants_out[GAIN_WIDTH +: OFFSET_WIDTH]};
    end

    // one datapath per lane
    logic signed [LENGTH_WIDTH:0] length_signed;

    assign length_signed = $signed({1'b0, row_length});

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane++) begin: lane_norm
            // stage 1: operand
            logic signed [DATA_WIDTH-1:0]      x_operand;
            // pass 0: square (stage 2), accumulate (stage 3)
            (* use_dsp = "yes" *) logic signed [2*DATA_WIDTH-1:0] square;
            logic signed [DATA_WIDTH-1:0]      x_delayed;
            logic signed [TOTAL_WIDTH-1:0]     total;
            logic [SQUARES_WIDTH-1:0]          squares;
            // statistics
            logic signed [TOTAL_WIDTH-1:0]     total_operand;
            logic signed [SQUARES_WIDTH:0]     squares_operand;
            (* use_dsp = "yes" *) logic signed [WIDE-1:0] length_squares;
            (* use_dsp = "yes" *) logic signed [WIDE-1:0] total_squared;
            logic signed [WIDE-1:0]            variance_raw;
            logic [VARIANCE_WIDTH-1:0]         variance;
            logic [VARIANCE_WIDTH-1:0]         variance_delayed;
            logic [EXPONENT_WIDTH-1:0]         exponent;
            logic [EXPONENT_WIDTH-1:0]         exponent_delayed;
            logic [VARIANCE_WIDTH-1:0]         normalised;
            logic [RSQRT_WIDTH-1:0]            rsqrt_out;
            // held for the output pass
            logic signed [TOTAL_WIDTH-1:0]     total_held;
            logic signed [RSQRT_WIDTH:0]       reciprocal;
            logic [NORM_SHIFT_WIDTH-1:0]       norm_shift;
            // pass 1
            (* use_dsp = "yes" *) logic signed [CENTRED_WIDTH-1:0] length_x;        // stage 2
            logic signed [CENTRED_WIDTH-1:0]   centred;                             // stage 3
            (* use_dsp = "yes" *) logic signed [PRODUCT_WIDTH-1:0] product;         // stage 4
            logic signed [NORM_WIDTH-1:0]      norm;                                // stage 5
            (* use_dsp = "yes" *) logic signed [NORM_WIDTH+GAIN_WIDTH-1:0] scaled_product;   // stage 6
            logic signed [SCALED_WIDTH-1:0]    scaled;                              // stage 7
            logic signed [SCALED_WIDTH-1:0]    shifted;                             // stage 8

            logic signed [PRODUCT_WIDTH-1:0]   product_shifted;

            assign product_shifted = product >>> norm_shift;

            always_ff @(posedge CLK) begin
                x_operand <= x_in[lane];

                // pass 0
                square    <= x_operand * x_operand;
                x_delayed <= x_operand;
                if (statistics_valid_pipeline[1]) begin
                    if (statistics_first_pipeline[1]) begin
                        total   <= TOTAL_WIDTH'(x_delayed);
                        squares <= SQUARES_WIDTH'(square);
                    end
                    else begin
                        total   <= total + TOTAL_WIDTH'(x_delayed);
                        squares <= squares + SQUARES_WIDTH'(square);
                    end
                end

                // statistics: every stage runs all the time, the result is taken at statistics_done
                total_operand    <= total;
                squares_operand  <= $signed({1'b0, squares});
                length_squares   <= squares_operand * length_signed;
                total_squared    <= total_operand * total_operand;
                variance_raw     <= length_squares - total_squared + WIDE'($signed({1'b0, epsilon}));
                variance         <= (variance_raw < 1) ? VARIANCE_WIDTH'(1) : variance_raw[VARIANCE_WIDTH-1:0];
                exponent         <= '0;
                for (int e = 1; e < VARIANCE_WIDTH / 2; e++) begin
                    if ((variance >> (2*e)) != '0) begin
                        exponent <= EXPONENT_WIDTH'(e);
                    end
                end
                variance_delayed <= variance;
                normalised       <= variance_delayed << (VARIANCE_WIDTH - 2 - 2*exponent);
                exponent_delayed <= exponent;
            end

            // reciprocal square root ROM: address register, read, then the held result
            logic [RSQRT_INDEX_BITS-1:0]     rsqrt_address;
            logic [EXPONENT_WIDTH-1:0]       exponent_address;
            logic [EXPONENT_WIDTH-1:0]       exponent_read;
            (* rom_style = "distributed" *) logic [RSQRT_WIDTH-1:0] rsqrt_rom[1 << RSQRT_INDEX_BITS];

            initial begin
                $readmemh(ROM_FILE, rsqrt_rom);
            end

            always_ff @(posedge CLK) begin
                rsqrt_address    <= normalised[VARIANCE_WIDTH-1 -: RSQRT_INDEX_BITS];
                exponent_address <= exponent_delayed;
                rsqrt_out        <= rsqrt_rom[rsqrt_address];
                exponent_read    <= exponent_address;
                if (statistics_done) begin
                    reciprocal <= $signed({1'b0, rsqrt_out});
                    norm_shift <= NORM_SHIFT_WIDTH'(NORM_SHIFT_BASE) + NORM_SHIFT_WIDTH'(exponent_read);
                    total_held <= total;
                end
            end

            // pass 1
            always_ff @(posedge CLK) begin
                length_x       <= x_operand * length_signed;
                centred        <= length_x - CENTRED_WIDTH'(total_held);
                product        <= centred * reciprocal;
                norm           <= product_shifted[NORM_WIDTH-1:0];
                scaled_product <= norm * $signed(gain_pipeline[1]);
                scaled         <= SCALED_WIDTH'(scaled_product) + SCALED_WIDTH'($signed(offset_pipeline[2]));
                shifted        <= scaled >>> out_shift;
                if (shifted > DATA_MAX) begin
                    y_out[lane] <= DATA_MAX[DATA_WIDTH-1:0];
                end
                else if (shifted < DATA_MIN) begin
                    y_out[lane] <= DATA_MIN[DATA_WIDTH-1:0];
                end
                else begin
                    y_out[lane] <= shifted[DATA_WIDTH-1:0];
                end
            end
        end
    endgenerate

endmodule
