`timescale 1ns / 1ps

module softmax #(
    parameter  int    LANES        = vit_config_pkg::SYS_ROW,
    parameter  int    DATA_WIDTH   = vit_config_pkg::DATA_WIDTH,
    parameter  int    TABLE_COUNT  = vit_config_pkg::SOFTMAX_TABLE_COUNT,
    parameter  string ROM_FILE     = "rom/exp_rom.hex",
    localparam int    LATENCY      = vit_config_pkg::SOFTMAX_LATENCY,
    localparam int    PASS_WIDTH   = vit_config_pkg::ROW_OP_PASS_WIDTH,
    localparam int    SELECT_WIDTH = (TABLE_COUNT > 1) ? $clog2(TABLE_COUNT) : 1
) (
    input  logic                                    CLK,
    input  logic                                    reset,
    input  logic [SELECT_WIDTH-1:0]                 table_select,
    input  logic                                    valid_in,
    input  logic [PASS_WIDTH-1:0]                   pass_in,
    input  logic                                    first_in,
    input  logic                                    last_in,
    input  logic signed [LANES-1:0][DATA_WIDTH-1:0] score_in,
    output logic                                    pass_ready,
    output logic                                    valid_out,
    output logic [LANES-1:0][DATA_WIDTH-1:0]        p_out
);

    localparam int EXP_BITS        = vit_config_pkg::SOFTMAX_EXP_BITS;          // 12
    localparam int QUOTIENT_FRAC   = vit_config_pkg::SOFTMAX_QUOTIENT_FRAC;     // 10
    localparam int OUT_SHIFT       = EXP_BITS + QUOTIENT_FRAC;                  // 22
    localparam int TOTAL_WIDTH     = EXP_BITS + DATA_WIDTH;                     // up to 256 entries
    localparam int NUMERATOR_WIDTH = (DATA_WIDTH - 1) + OUT_SHIFT;              // 127 * 2**22
    localparam int QUOTIENT_WIDTH  = NUMERATOR_WIDTH - EXP_BITS + 1;            // total >= 2**EXP_BITS - 1
    localparam int PRODUCT_WIDTH   = EXP_BITS + QUOTIENT_WIDTH;
    localparam int ROM_DEPTH       = TABLE_COUNT << DATA_WIDTH;
    localparam int ROM_ADDRESS     = $clog2(ROM_DEPTH);
    localparam int PAIRS           = (LANES + 1) / 2;
    localparam int STEP_WIDTH      = $clog2(NUMERATOR_WIDTH + 1);

    localparam logic [NUMERATOR_WIDTH-1:0] NUMERATOR = NUMERATOR_WIDTH'((1 << (DATA_WIDTH - 1)) - 1) << OUT_SHIFT;
    localparam logic [DATA_WIDTH-1:0]      P_MAX     = DATA_WIDTH'((1 << (DATA_WIDTH - 1)) - 1);

    // stages from an input to: the peak (2), the table entry (4), the total (5)
    localparam int PEAK_STAGES  = 2;
    localparam int TOTAL_STAGES = 5;

    // control: flags travelling with the data, the divider steps, pass_ready
    logic [LATENCY-1:0]      output_valid_pipeline;
    logic [PEAK_STAGES-1:0]  peak_valid_pipeline;
    logic [PEAK_STAGES-1:0]  peak_first_pipeline;
    logic [PEAK_STAGES-1:0]  peak_last_pipeline;
    logic [TOTAL_STAGES-1:0] total_valid_pipeline;
    logic [TOTAL_STAGES-1:0] total_first_pipeline;
    logic [TOTAL_STAGES-1:0] total_last_pipeline;

    logic                    divide_start;    // total is final in this cycle
    logic                    divide_busy;
    logic                    divide_done;
    logic [STEP_WIDTH-1:0]   divide_steps;

    assign divide_start = total_last_pipeline[TOTAL_STAGES-1];

    always_ff @(posedge CLK) begin
        if (reset) begin
            output_valid_pipeline <= '0;
            peak_valid_pipeline   <= '0;
            peak_last_pipeline    <= '0;
            total_valid_pipeline  <= '0;
            total_last_pipeline   <= '0;
            divide_busy           <= 1'b0;
            divide_done           <= 1'b0;
            pass_ready            <= 1'b0;
        end
        else begin
            output_valid_pipeline <= LATENCY'({output_valid_pipeline, valid_in && pass_in == PASS_WIDTH'(2)});
            peak_valid_pipeline   <= PEAK_STAGES'({peak_valid_pipeline, valid_in && pass_in == '0});
            peak_last_pipeline    <= PEAK_STAGES'({peak_last_pipeline, valid_in && pass_in == '0 && last_in});
            total_valid_pipeline  <= TOTAL_STAGES'({total_valid_pipeline, valid_in && pass_in == PASS_WIDTH'(1)});
            total_last_pipeline   <= TOTAL_STAGES'({total_last_pipeline,
                                                    valid_in && pass_in == PASS_WIDTH'(1) && last_in});

            divide_done <= 1'b0;
            if (divide_start) begin
                divide_busy  <= 1'b1;
                divide_steps <= STEP_WIDTH'(NUMERATOR_WIDTH);
            end
            else if (divide_busy) begin
                divide_steps <= divide_steps - 1'b1;
                if (divide_steps == STEP_WIDTH'(1)) begin
                    divide_busy <= 1'b0;
                    divide_done <= 1'b1;
                end
            end

            if (valid_in && pass_in != PASS_WIDTH'(2)) begin
                pass_ready <= 1'b0;
            end
            else if (peak_last_pipeline[PEAK_STAGES-1] || divide_done) begin
                pass_ready <= 1'b1;
            end
        end
        peak_first_pipeline  <= PEAK_STAGES'({peak_first_pipeline, first_in});
        total_first_pipeline <= TOTAL_STAGES'({total_first_pipeline, first_in});
    end

    assign valid_out = output_valid_pipeline[LATENCY-1];

    // per lane: peak, distance, total, divider, output
    logic [LANES-1:0][DATA_WIDTH-1:0] distance;     // stage 2
    logic [LANES-1:0][EXP_BITS-1:0]   entry;        // stage 4

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane++) begin: lane_softmax
            logic signed [DATA_WIDTH-1:0]   score;          // stage 1
            logic signed [DATA_WIDTH-1:0]   peak;
            logic signed [DATA_WIDTH-1:0]   peak_held;
            logic signed [DATA_WIDTH:0]     difference;
            logic [TOTAL_WIDTH-1:0]         total;          // stage 5
            logic [TOTAL_WIDTH-1:0]         divisor;
            logic [TOTAL_WIDTH:0]           remainder;
            logic [NUMERATOR_WIDTH-1:0]     numerator;
            logic [QUOTIENT_WIDTH-1:0]      quotient;
            logic [QUOTIENT_WIDTH-1:0]      quotient_held;
            (* use_dsp = "yes" *) logic [PRODUCT_WIDTH-1:0] product;    // stage 5
            logic [PRODUCT_WIDTH-1:0]       rounded;        // stage 6

            logic [TOTAL_WIDTH+1:0]         trial;
            logic                           trial_fits;

            assign difference = peak_held - score;
            // restoring step: bring down the next numerator bit, subtract if it fits
            assign trial      = {remainder, numerator[NUMERATOR_WIDTH-1]};
            assign trial_fits = trial >= (TOTAL_WIDTH+2)'(divisor);

            always_ff @(posedge CLK) begin
                score <= score_in[lane];

                // pass 0
                if (peak_valid_pipeline[0]) begin
                    if (peak_first_pipeline[0] || score > peak) begin
                        peak <= score;
                    end
                end
                if (peak_last_pipeline[PEAK_STAGES-1]) begin
                    peak_held <= peak;
                end

                // passes 1 and 2: distance, then the table (below)
                distance[lane] <= difference[DATA_WIDTH-1:0];

                // pass 1
                if (total_valid_pipeline[TOTAL_STAGES-2]) begin
                    total <= total_first_pipeline[TOTAL_STAGES-2] ? TOTAL_WIDTH'(entry[lane])
                                                                  : total + TOTAL_WIDTH'(entry[lane]);
                end

                // divider
                if (divide_start) begin
                    divisor   <= total;
                    remainder <= '0;
                    numerator <= NUMERATOR;
                    quotient  <= '0;
                end
                else if (divide_busy) begin
                    remainder <= trial_fits ? (TOTAL_WIDTH+1)'(trial - (TOTAL_WIDTH+2)'(divisor))
                                            : trial[TOTAL_WIDTH:0];
                    numerator <= numerator << 1;
                    quotient  <= QUOTIENT_WIDTH'({quotient, trial_fits});
                end
                if (divide_done) begin
                    quotient_held <= quotient;
                end

                // pass 2
                product <= entry[lane] * quotient_held;
                rounded <= (product + (PRODUCT_WIDTH'(1) << (OUT_SHIFT - 1))) >> OUT_SHIFT;
                p_out[lane] <= (rounded > PRODUCT_WIDTH'(P_MAX)) ? P_MAX : rounded[DATA_WIDTH-1:0];
            end
        end
    endgenerate

    // exp ROM: one copy per pair of lanes, read (stage 3), output register (stage 4)
    genvar pair;
    generate
        for (pair = 0; pair < PAIRS; pair++) begin: lane_pair
            (* rom_style = "block" *) logic [EXP_BITS-1:0] rom[ROM_DEPTH];

            initial begin
                $readmemh(ROM_FILE, rom);
            end

            logic [1:0][ROM_ADDRESS-1:0] address;
            logic [1:0][EXP_BITS-1:0]    read_data;

            assign address[0] = ROM_ADDRESS'({table_select, distance[2*pair]});
            if (2*pair + 1 < LANES) begin: second_lane
                assign address[1] = ROM_ADDRESS'({table_select, distance[2*pair + 1]});
            end
            else begin: no_second_lane
                assign address[1] = '0;
            end

            always_ff @(posedge CLK) begin
                read_data[0]  <= rom[address[0]];
                read_data[1]  <= rom[address[1]];
                entry[2*pair] <= read_data[0];
            end
            if (2*pair + 1 < LANES) begin: second_entry
                always_ff @(posedge CLK) begin
                    entry[2*pair + 1] <= read_data[1];
                end
            end
        end
    endgenerate

endmodule
