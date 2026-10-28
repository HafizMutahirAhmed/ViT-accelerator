`timescale 1ns / 1ps

module vit_axi_core #(
    parameter  int SYS_ROW    = vit_config_pkg::SYS_ROW,
    parameter  int SYS_COL    = vit_config_pkg::SYS_COL,
    parameter  bit [vit_config_pkg::BUFFER_COUNT-1:0][31:0] BUFFER_DEPTH
                              = {vit_config_pkg::BUFFER_COUNT{32'(vit_config_pkg::BANK_DEPTH)}},
    parameter  string GELU_ROM_FILE  = "rom/gelu_rom.hex",
    parameter  string EXP_ROM_FILE   = "rom/exp_rom.hex",
    parameter  string RSQRT_ROM_FILE = "rom/rsqrt_rom.hex",
    parameter  int LITE_ADDRESS_WIDTH = 12
) (
    input  logic                          aclk,
    input  logic                          aresetn,

    input  logic [LITE_ADDRESS_WIDTH-1:0] s_axi_awaddr,
    input  logic                          s_axi_awvalid,
    output logic                          s_axi_awready,
    input  logic [31:0]                   s_axi_wdata,
    input  logic [3:0]                    s_axi_wstrb,
    input  logic                          s_axi_wvalid,
    output logic                          s_axi_wready,
    output logic [1:0]                    s_axi_bresp,
    output logic                          s_axi_bvalid,
    input  logic                          s_axi_bready,
    input  logic [LITE_ADDRESS_WIDTH-1:0] s_axi_araddr,
    input  logic                          s_axi_arvalid,
    output logic                          s_axi_arready,
    output logic [31:0]                   s_axi_rdata,
    output logic [1:0]                    s_axi_rresp,
    output logic                          s_axi_rvalid,
    input  logic                          s_axi_rready,

    input  logic [63:0]                   s_axis_tdata,
    input  logic [7:0]                    s_axis_tkeep,
    input  logic                          s_axis_tlast,
    input  logic                          s_axis_tvalid,
    output logic                          s_axis_tready,

    output logic [63:0]                   m_axis_tdata,
    output logic [7:0]                    m_axis_tkeep,
    output logic                          m_axis_tlast,
    output logic                          m_axis_tvalid,
    input  logic                          m_axis_tready,

    output logic                          irq,
    output logic                          busy,
    output logic                          done_flag,
    output logic                          error_flag
);

    localparam int DATA_WIDTH             = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH              = vit_config_pkg::ACC_WIDTH;
    localparam int LANES                  = (SYS_ROW > SYS_COL) ? SYS_ROW : SYS_COL;
    localparam int BUFFER_COUNT           = vit_config_pkg::BUFFER_COUNT;
    localparam int BUFFER_ID_WIDTH        = vit_config_pkg::BUFFER_ID_WIDTH;
    localparam int ADDRESS_WIDTH          = vit_config_pkg::ADDRESS_WIDTH;
    localparam int DESCRIPTOR_WORDS       = vit_config_pkg::DESCRIPTOR_WORDS;
    localparam int DESCRIPTOR_COUNT       = vit_config_pkg::DESCRIPTOR_COUNT;
    localparam int DESCRIPTOR_INDEX_WIDTH = $clog2(DESCRIPTOR_COUNT);
    localparam int GAIN_WIDTH             = vit_config_pkg::LAYERNORM_GAIN_WIDTH;
    localparam int OFFSET_WIDTH           = vit_config_pkg::LAYERNORM_OFFSET_WIDTH;

    localparam logic [31:0] VERSION = 32'h5649_5401;       // "VIT", version 1

    // register offsets
    localparam logic [11:0] REG_CONTROL        = 12'h000;
    localparam logic [11:0] REG_STATUS         = 12'h004;
    localparam logic [11:0] REG_START_INDEX    = 12'h008;
    localparam logic [11:0] REG_CYCLES         = 12'h00C;
    localparam logic [11:0] REG_CONFIG         = 12'h010;
    localparam logic [11:0] REG_DESCRIPTORS    = 12'h014;
    localparam logic [11:0] REG_IRQ_ENABLE     = 12'h018;
    localparam logic [11:0] REG_VERSION        = 12'h01C;
    localparam logic [11:0] REG_STREAM_TARGET  = 12'h020;
    localparam logic [11:0] REG_STREAM_ADDRESS = 12'h024;
    localparam logic [11:0] REG_STORE_COUNT    = 12'h028;
    localparam logic [11:0] REG_DESC_WORD      = 12'h100;   // 5 words
    localparam logic [11:0] REG_DESC_INDEX     = 12'h114;
    localparam logic [11:0] REG_BIAS_WORD      = 12'h200;   // 8 words
    localparam logic [11:0] REG_BIAS_ADDRESS   = 12'h220;
    localparam logic [11:0] REG_LN_GAIN        = 12'h240;
    localparam logic [11:0] REG_LN_OFFSET      = 12'h244;
    localparam logic [11:0] REG_LN_ADDRESS     = 12'h248;

    localparam logic [1:0] KIND_POOL       = 2'd0;
    localparam logic [1:0] KIND_BIAS       = 2'd1;
    localparam logic [1:0] KIND_LAYERNORM  = 2'd2;
    localparam logic [1:0] KIND_DESCRIPTOR = 2'd3;

    localparam logic [1:0] RESP_OKAY   = 2'b00;
    localparam logic [1:0] RESP_SLVERR = 2'b10;

    localparam int ITEM_WIDTH   = LANES * ACC_WIDTH;        // the widest item, a bias address
    localparam int BEATS_MAX    = ITEM_WIDTH / 64;
    localparam int RESET_CYCLES = 8;
    // from host_address at this module's outputs to the engine's read data
    localparam int READ_WAIT    = vit_config_pkg::BUFFER_READ_LATENCY + 1;
    localparam int FIFO_DEPTH   = 16;
    localparam int FIFO_WIDTH   = $clog2(FIFO_DEPTH);

    logic reset;          // of this module
    logic engine_reset;   // of the engine: reset or soft reset
    logic [$clog2(RESET_CYCLES+1)-1:0] soft_reset_count;
    logic soft_reset;

    assign reset      = !aresetn;
    assign soft_reset = soft_reset_count != '0;

    always_ff @(posedge aclk) begin
        engine_reset <= reset || soft_reset;
    end

    // engine
    logic                              start;
    logic [DESCRIPTOR_INDEX_WIDTH-1:0] start_index;
    logic                              done;
    logic [DESCRIPTOR_INDEX_WIDTH-1:0] host_descriptor_index;
    logic [DESCRIPTOR_WORDS-1:0]       host_descriptor_write_enable;
    logic [31:0]                       host_descriptor_write_data;
    logic [BUFFER_ID_WIDTH-1:0]        host_buffer_id;
    logic [ADDRESS_WIDTH-1:0]          host_address;
    logic [LANES-1:0]                  host_buffer_write_enable;
    logic [LANES-1:0][DATA_WIDTH-1:0]  host_buffer_write_data;
    logic [LANES-1:0][DATA_WIDTH-1:0]  host_buffer_read_data;
    logic [SYS_COL-1:0]                host_bias_write_enable;
    logic [SYS_COL-1:0][ACC_WIDTH-1:0] host_bias_write_data;
    logic                              host_layernorm_write_enable;
    logic signed [GAIN_WIDTH-1:0]      host_layernorm_write_gain;
    logic signed [OFFSET_WIDTH-1:0]    host_layernorm_write_offset;

    matmul_engine #(
        .SYS_ROW        (SYS_ROW),
        .SYS_COL        (SYS_COL),
        .BUFFER_DEPTH   (BUFFER_DEPTH),
        .GELU_ROM_FILE  (GELU_ROM_FILE),
        .EXP_ROM_FILE   (EXP_ROM_FILE),
        .RSQRT_ROM_FILE (RSQRT_ROM_FILE)
    ) engine (
        .CLK                          (aclk),
        .reset                        (engine_reset),
        .start                        (start),
        .start_index                  (start_index),
        .busy                         (busy),
        .done                         (done),
        .host_descriptor_index        (host_descriptor_index),
        .host_descriptor_write_enable (host_descriptor_write_enable),
        .host_descriptor_write_data   (host_descriptor_write_data),
        .host_buffer_id               (host_buffer_id),
        .host_address                 (host_address),
        .host_buffer_write_enable     (host_buffer_write_enable),
        .host_buffer_write_data       (host_buffer_write_data),
        .host_buffer_read_data        (host_buffer_read_data),
        .host_bias_write_enable       (host_bias_write_enable),
        .host_bias_write_data         (host_bias_write_data),
        .host_bias_read_data          (),
        .host_layernorm_write_enable  (host_layernorm_write_enable),
        .host_layernorm_write_gain    (host_layernorm_write_gain),
        .host_layernorm_write_offset  (host_layernorm_write_offset)
    );

    // registers
    logic [DESCRIPTOR_INDEX_WIDTH-1:0] start_index_reg;
    logic                              irq_enable;
    logic [BUFFER_ID_WIDTH-1:0]        target_buffer;
    logic [1:0]                        target_kind;
    logic [ADDRESS_WIDTH-1:0]          stream_address;
    logic [31:0]                       cycles;
    logic [31:0]                       window_descriptor[DESCRIPTOR_WORDS];
    logic [31:0]                       window_bias[LANES];
    logic [GAIN_WIDTH-1:0]             window_gain;
    logic [OFFSET_WIDTH-1:0]           window_offset;

    // item writer: one item to the engine's host port
    logic                  writer_busy;
    logic                  writer_request;        // from the stream or a window, one cycle
    logic [1:0]            writer_kind;
    logic [ITEM_WIDTH-1:0] writer_item;
    logic [ADDRESS_WIDTH-1:0] writer_address;
    logic [BUFFER_ID_WIDTH-1:0] writer_buffer;
    logic [2:0]            writer_word;           // descriptor word being written
    logic                  wrote_last_cycle;      // an engine write was issued in the last cycle

    // start: given once no write is on its way
    logic start_pending;

    // store
    logic                     store_active;
    logic [ADDRESS_WIDTH:0]   store_left;         // addresses still to issue
    logic [READ_WAIT-1:0]     read_valid_pipeline;
    logic [READ_WAIT-1:0]     read_last_pipeline;
    logic [64:0]              fifo[FIFO_DEPTH];
    logic [FIFO_WIDTH-1:0]    fifo_write_pointer;
    logic [FIFO_WIDTH-1:0]    fifo_read_pointer;
    logic [FIFO_WIDTH:0]      fifo_count;
    logic [FIFO_WIDTH:0]      in_flight;          // addresses issued, data not yet in the FIFO
    logic                     store_issue;

    // stream load: beats of the item being collected
    logic [$clog2(BEATS_MAX)-1:0] beat_index;
    logic [$clog2(BEATS_MAX)-1:0] beats_minus_one;
    logic [ITEM_WIDTH-1:0]        load_item;

    always_comb begin
        case (target_kind)
            KIND_BIAS:       beats_minus_one = ($clog2(BEATS_MAX))'(BEATS_MAX - 1);
            KIND_DESCRIPTOR: beats_minus_one = ($clog2(BEATS_MAX))'((32*DESCRIPTOR_WORDS + 63) / 64 - 1);
            default:         beats_minus_one = '0;
        endcase
    end

    logic stream_is_window;     // the request in writer_request came from a window
    logic load_fire;
    logic load_last_beat;

    // the engine must be idle and nothing else may be using its host port. A
    // register write has priority over a beat in the same cycle.
    logic port_free;

    assign port_free     = !busy && !soft_reset && !store_active && !start_pending && !writer_busy;
    assign s_axis_tready = port_free && !reset && !(s_axi_awvalid && s_axi_wvalid);

    // AXI4-Lite handshakes: address and data together, one response at a time
    logic write_fire;
    logic read_fire;
    logic is_window;
    logic write_refused;

    assign write_fire    = s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid && !writer_busy && !writer_request
                           && !reset;
    assign s_axi_awready = write_fire;
    assign s_axi_wready  = write_fire;
    assign read_fire     = s_axi_arvalid && !s_axi_rvalid && !reset;
    assign s_axi_arready = read_fire;
    assign s_axi_rresp   = RESP_OKAY;

    assign is_window     = s_axi_awaddr[11:8] == 4'h1 || s_axi_awaddr[11:8] == 4'h2;
    // what needs an idle engine: the windows, a start and a store
    assign write_refused = (busy || start_pending || store_active)
                           && (is_window
                               || (s_axi_awaddr[11:0] == REG_CONTROL && s_axi_wdata[0])
                               || s_axi_awaddr[11:0] == REG_STORE_COUNT);

    assign store_issue = store_active && store_left != '0
                         && (fifo_count + in_flight) < (FIFO_WIDTH+1)'(FIFO_DEPTH - 1);

    assign m_axis_tvalid = fifo_count != '0;
    assign m_axis_tdata  = fifo[fifo_read_pointer][63:0];
    assign m_axis_tlast  = fifo[fifo_read_pointer][64];
    assign m_axis_tkeep  = '1;
    assign irq           = done_flag && irq_enable;

    always_ff @(posedge aclk) begin
        if (reset) begin
            s_axi_bvalid     <= 1'b0;
            s_axi_rvalid     <= 1'b0;
            start_index_reg  <= '0;
            irq_enable       <= 1'b0;
            target_buffer    <= '0;
            target_kind      <= KIND_POOL;
            stream_address   <= '0;
            soft_reset_count <= '0;
        end
        else begin
            if (soft_reset_count != '0) begin
                soft_reset_count <= soft_reset_count - 1'b1;
            end
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end
            if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end

            // ---- writes ----
            if (write_fire) begin
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= write_refused ? RESP_SLVERR : RESP_OKAY;
                if (!write_refused) begin
                    case (s_axi_awaddr[11:0])
                        REG_CONTROL: begin
                            if (s_axi_wdata[1]) begin
                                soft_reset_count <= ($clog2(RESET_CYCLES+1))'(RESET_CYCLES);
                            end
                        end
                        REG_START_INDEX:    start_index_reg <= s_axi_wdata[DESCRIPTOR_INDEX_WIDTH-1:0];
                        REG_IRQ_ENABLE:     irq_enable      <= s_axi_wdata[0];
                        REG_STREAM_TARGET: begin
                            target_buffer <= s_axi_wdata[BUFFER_ID_WIDTH-1:0];
                            target_kind   <= s_axi_wdata[9:8];
                        end
                        REG_STREAM_ADDRESS: stream_address <= s_axi_wdata[ADDRESS_WIDTH-1:0];
                        REG_LN_GAIN:        window_gain    <= s_axi_wdata[GAIN_WIDTH-1:0];
                        REG_LN_OFFSET:      window_offset  <= s_axi_wdata[OFFSET_WIDTH-1:0];
                        default: begin
                            for (int word = 0; word < DESCRIPTOR_WORDS; word++) begin
                                if (s_axi_awaddr[11:0] == REG_DESC_WORD + 12'(4*word)) begin
                                    window_descriptor[word] <= s_axi_wdata;
                                end
                            end
                            for (int word = 0; word < LANES; word++) begin
                                if (s_axi_awaddr[11:0] == REG_BIAS_WORD + 12'(4*word)) begin
                                    window_bias[word] <= s_axi_wdata;
                                end
                            end
                        end
                    endcase
                end
            end

            // the stream address advances with every item loaded or read out
            if (load_fire && load_last_beat) begin
                stream_address <= stream_address + 1'b1;
            end
            if (store_issue) begin
                stream_address <= stream_address + 1'b1;
            end

            // ---- reads ----
            if (read_fire) begin
                s_axi_rvalid <= 1'b1;
                case (s_axi_araddr[11:0])
                    REG_STATUS:         s_axi_rdata <= {27'b0, store_active, beat_index != '0, error_flag, done_flag,
                                                        busy};
                    REG_START_INDEX:    s_axi_rdata <= 32'(start_index_reg);
                    REG_CYCLES:         s_axi_rdata <= cycles;
                    REG_CONFIG:         s_axi_rdata <= {8'(LANES), 8'(BUFFER_COUNT), 8'(SYS_COL), 8'(SYS_ROW)};
                    REG_DESCRIPTORS:    s_axi_rdata <= 32'(DESCRIPTOR_COUNT);
                    REG_IRQ_ENABLE:     s_axi_rdata <= {31'b0, irq_enable};
                    REG_VERSION:        s_axi_rdata <= VERSION;
                    REG_STREAM_TARGET:  s_axi_rdata <= {22'b0, target_kind, 5'b0, target_buffer};
                    REG_STREAM_ADDRESS: s_axi_rdata <= 32'(stream_address);
                    REG_STORE_COUNT:    s_axi_rdata <= 32'(store_left);
                    default:            s_axi_rdata <= '0;
                endcase
            end
        end
    end

    // requests to the item writer: a stream item or a window commit
    assign load_fire      = s_axis_tvalid && s_axis_tready;
    assign load_last_beat = beat_index == beats_minus_one;

    always_ff @(posedge aclk) begin
        if (reset || soft_reset) begin
            writer_request <= 1'b0;
            beat_index     <= '0;
        end
        else begin
            writer_request <= 1'b0;
            if (load_fire) begin
                load_item[64*beat_index +: 64] <= s_axis_tdata;
                if (load_last_beat) begin
                    beat_index       <= '0;
                    writer_request   <= 1'b1;
                    stream_is_window <= 1'b0;
                    writer_kind      <= target_kind;
                    writer_buffer    <= target_buffer;
                    writer_address   <= stream_address;
                end
                else if (s_axis_tlast) begin
                    beat_index <= '0;          // a packet ended inside an item: the item is dropped
                end
                else begin
                    beat_index <= beat_index + 1'b1;
                end
            end
            else if (write_fire && !write_refused) begin
                stream_is_window <= 1'b1;
                writer_buffer    <= '0;
                writer_address   <= s_axi_wdata[ADDRESS_WIDTH-1:0];
                case (s_axi_awaddr[11:0])
                    REG_DESC_INDEX: begin
                        writer_request <= 1'b1;
                        writer_kind    <= KIND_DESCRIPTOR;
                    end
                    REG_BIAS_ADDRESS: begin
                        writer_request <= 1'b1;
                        writer_kind    <= KIND_BIAS;
                    end
                    REG_LN_ADDRESS: begin
                        writer_request <= 1'b1;
                        writer_kind    <= KIND_LAYERNORM;
                    end
                    default: begin
                    end
                endcase
            end
        end
    end

    // the item the writer takes
    always_comb begin
        writer_item = load_item;
        if (stream_is_window) begin
            writer_item = '0;
            case (writer_kind)
                KIND_DESCRIPTOR: begin
                    for (int word = 0; word < DESCRIPTOR_WORDS; word++) begin
                        writer_item[32*word +: 32] = window_descriptor[word];
                    end
                end
                KIND_BIAS: begin
                    for (int word = 0; word < LANES; word++) begin
                        writer_item[32*word +: 32] = window_bias[word];
                    end
                end
                default: begin
                    writer_item[GAIN_WIDTH-1:0]              = window_gain;
                    writer_item[GAIN_WIDTH +: OFFSET_WIDTH]  = window_offset;
                end
            endcase
        end
    end

    // item writer, start, store address issue: the engine's host port
    logic [ITEM_WIDTH-1:0] writing_item;
    logic [1:0]            writing_kind;

    always_ff @(posedge aclk) begin
        if (reset || soft_reset) begin
            writer_busy                  <= 1'b0;
            wrote_last_cycle             <= 1'b0;
            start                        <= 1'b0;
            start_pending                <= 1'b0;
            host_descriptor_write_enable <= '0;
            host_buffer_write_enable     <= '0;
            host_bias_write_enable       <= '0;
            host_layernorm_write_enable  <= 1'b0;
            store_active                 <= 1'b0;
            store_left                   <= '0;
        end
        else begin
            start                        <= 1'b0;
            host_descriptor_write_enable <= '0;
            host_buffer_write_enable     <= '0;
            host_bias_write_enable       <= '0;
            host_layernorm_write_enable  <= 1'b0;
            wrote_last_cycle             <= 1'b0;

            // ---- writes ----
            if (writer_request) begin
                writing_item <= writer_item;
                writing_kind <= writer_kind;
                host_buffer_id        <= writer_buffer;
                host_address          <= writer_address;
                host_descriptor_index <= writer_address[DESCRIPTOR_INDEX_WIDTH-1:0];
                host_buffer_write_data      <= writer_item[LANES*DATA_WIDTH-1:0];
                host_bias_write_data        <= writer_item[SYS_COL*ACC_WIDTH-1:0];
                host_layernorm_write_gain   <= writer_item[GAIN_WIDTH-1:0];
                host_layernorm_write_offset <= writer_item[GAIN_WIDTH +: OFFSET_WIDTH];
                host_descriptor_write_data  <= writer_item[31:0];
                wrote_last_cycle            <= 1'b1;
                case (writer_kind)
                    KIND_POOL:      host_buffer_write_enable    <= '1;
                    KIND_BIAS:      host_bias_write_enable      <= '1;
                    KIND_LAYERNORM: host_layernorm_write_enable <= 1'b1;
                    default: begin
                        host_descriptor_write_enable[0] <= 1'b1;
                        writer_busy                     <= 1'b1;
                        writer_word                     <= 3'd1;
                    end
                endcase
            end
            else if (writer_busy) begin
                host_descriptor_write_data               <= writing_item[32*writer_word +: 32];
                host_descriptor_write_enable[writer_word] <= 1'b1;
                wrote_last_cycle                          <= 1'b1;
                writer_word                               <= writer_word + 1'b1;
                if (writer_word == 3'(DESCRIPTOR_WORDS - 1)) begin
                    writer_busy <= 1'b0;
                end
            end

            // ---- start: one cycle after the last write has left ----
            if (write_fire && !write_refused && s_axi_awaddr[11:0] == REG_CONTROL && s_axi_wdata[0]) begin
                start_pending <= 1'b1;
            end
            if (start_pending && !writer_request && !writer_busy && !wrote_last_cycle) begin
                start         <= 1'b1;
                start_index   <= start_index_reg;
                start_pending <= 1'b0;
            end

            // ---- store ----
            if (write_fire && !write_refused && s_axi_awaddr[11:0] == REG_STORE_COUNT
                && s_axi_wdata[ADDRESS_WIDTH:0] != '0) begin
                store_active <= 1'b1;
                store_left   <= s_axi_wdata[ADDRESS_WIDTH:0];
            end
            if (store_issue) begin
                host_buffer_id <= target_buffer;
                host_address   <= stream_address;
                store_left     <= store_left - 1'b1;
            end
            if (store_active && store_left == '0 && in_flight == '0 && fifo_count == '0) begin
                store_active <= 1'b0;
            end
        end
    end

    // read data into the FIFO, READ_WAIT cycles after its address was issued
    logic fifo_push;
    logic fifo_pop;

    assign fifo_push = read_valid_pipeline[READ_WAIT-1];
    assign fifo_pop  = m_axis_tvalid && m_axis_tready;

    always_ff @(posedge aclk) begin
        if (reset || soft_reset) begin
            read_valid_pipeline <= '0;
            fifo_write_pointer  <= '0;
            fifo_read_pointer   <= '0;
            fifo_count          <= '0;
            in_flight           <= '0;
        end
        else begin
            read_valid_pipeline <= READ_WAIT'({read_valid_pipeline, store_issue});
            in_flight           <= in_flight + (FIFO_WIDTH+1)'(store_issue) - (FIFO_WIDTH+1)'(fifo_push);
            fifo_count          <= fifo_count + (FIFO_WIDTH+1)'(fifo_push) - (FIFO_WIDTH+1)'(fifo_pop);
            if (fifo_push) begin
                fifo_write_pointer <= fifo_write_pointer + 1'b1;
            end
            if (fifo_pop) begin
                fifo_read_pointer <= fifo_read_pointer + 1'b1;
            end
        end
        read_last_pipeline <= READ_WAIT'({read_last_pipeline, store_issue && store_left == (ADDRESS_WIDTH+1)'(1)});
        if (fifo_push) begin
            fifo[fifo_write_pointer] <= {read_last_pipeline[READ_WAIT-1], host_buffer_read_data};
        end
    end

    // status flags and the cycle counter
    always_ff @(posedge aclk) begin
        if (reset || soft_reset) begin
            done_flag  <= 1'b0;
            error_flag <= 1'b0;
            cycles     <= '0;
        end
        else begin
            if (busy) begin
                cycles <= cycles + 1'b1;
            end
            if (done) begin
                done_flag <= 1'b1;
            end
            if (write_fire && write_refused) begin
                error_flag <= 1'b1;
            end
            if (load_fire && s_axis_tlast && !load_last_beat) begin
                error_flag <= 1'b1;
            end
            if (write_fire && !write_refused && s_axi_awaddr[11:0] == REG_CONTROL) begin
                if (s_axi_wdata[0]) begin
                    done_flag <= 1'b0;
                    cycles    <= '0;
                end
                if (s_axi_wdata[2]) begin
                    done_flag  <= 1'b0;
                    error_flag <= 1'b0;
                end
            end
        end
    end

endmodule
