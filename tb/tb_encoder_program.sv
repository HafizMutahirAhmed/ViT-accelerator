`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Self-checking testbench for the whole forward pass as one descriptor
// program on matmul_engine: patch embedding and class-and-position add, two
// encoder layers, the final LayerNorm on the class row and the classifiers.
//
// Host model. Everything the engine is given goes through its host port, one
// write per cycle while busy is low, the way the FrontPanel host will do it:
// the descriptor table once, then before each phase that phase's images (the
// weight buffer, the bias store, the LayerNorm constants; for the first phase
// also the im2col input and the constant addend). Then one start pulse per
// phase. Nothing is loaded into a memory behind the port, and the residual
// stream stays in the engine from one phase to the next.
//
// Checks. While a phase runs, every time the engine finishes an op the
// testbench compares that op's whole output tensor with the golden tensor of
// that stage. The host port cannot read while busy is high, so these checks
// read the pool memories directly (hierarchical reference, no simulation time
// passes). After each phase the residual stream, or the logits, is also read
// back through the host port. Every stage prints PASS or FAIL with its name
// and the first mismatching index.
//
// Vectors come from golden/gen_deit_forward.py (golden/layer_program.py), in
// golden/vectors/deit_forward: program_desc.hex, program_phases.txt,
// program_ops.txt, program_loads.txt, the *_image.hex files and one golden
// tensor per stage. The pool buffers have exactly the planned depths, which
// are parameters here together with the vector directory and the ROM images,
// so a wrapper can run another model configuration (tb_encoder_program_mnist112).
// 8x8 only.
//////////////////////////////////////////////////////////////////////////////////

module tb_encoder_program #(
    parameter string VECTOR_DIR     = "../golden/vectors/deit_forward/",
    // words per bank of pool buffers 0 .. 7: the plan in golden/layer_program.py,
    // which the generator also writes to program_plan.txt
    parameter int    DEPTH_X        = 4800,
    parameter int    DEPTH_W        = 55296,
    parameter int    DEPTH_BIG      = 19200,
    parameter int    DEPTH_N        = 4800,
    parameter int    DEPTH_S        = 14775,
    parameter int    DEPTH_QV       = 9528,
    parameter int    DEPTH_K        = 4800,
    parameter int    DEPTH_UNUSED   = 16,
    parameter string GELU_ROM_FILE  = "rom/gelu_rom.hex",
    parameter string EXP_ROM_FILE   = "rom/exp_rom.hex",
    parameter string RSQRT_ROM_FILE = "rom/rsqrt_rom.hex"
);

    localparam int SYS_ROW                = 8;
    localparam int SYS_COL                = 8;
    localparam int LANES                  = 8;
    localparam int DATA_WIDTH             = vit_config_pkg::DATA_WIDTH;
    localparam int ACC_WIDTH              = vit_config_pkg::ACC_WIDTH;
    localparam int BUFFER_COUNT           = vit_config_pkg::BUFFER_COUNT;
    localparam int BUFFER_ID_WIDTH        = vit_config_pkg::BUFFER_ID_WIDTH;
    localparam int ADDRESS_WIDTH          = vit_config_pkg::ADDRESS_WIDTH;
    localparam int DESCRIPTOR_WORDS       = vit_config_pkg::DESCRIPTOR_WORDS;
    localparam int DESCRIPTOR_COUNT       = vit_config_pkg::DESCRIPTOR_COUNT;
    localparam int DESCRIPTOR_INDEX_WIDTH = $clog2(DESCRIPTOR_COUNT);
    localparam int GAIN_WIDTH             = vit_config_pkg::LAYERNORM_GAIN_WIDTH;
    localparam int OFFSET_WIDTH           = vit_config_pkg::LAYERNORM_OFFSET_WIDTH;

    localparam bit [BUFFER_COUNT-1:0][31:0] BUFFER_DEPTH
        = {32'(DEPTH_UNUSED), 32'(DEPTH_K), 32'(DEPTH_QV), 32'(DEPTH_S), 32'(DEPTH_N), 32'(DEPTH_BIG),
           32'(DEPTH_W), 32'(DEPTH_X)};

    localparam int IMAGE_WORDS_MAX  = 65536;
    localparam int VECTOR_WORDS_MAX = 262144;
    localparam int PHASES_MAX       = 64;
    localparam int TIMEOUT_CYCLES   = 4000000;

    logic                              CLK = 1'b0;
    logic                              reset;
    logic                              start;
    logic [DESCRIPTOR_INDEX_WIDTH-1:0] start_index;
    logic                              busy;
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
    logic [SYS_COL-1:0][ACC_WIDTH-1:0] host_bias_read_data;
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
    ) dut (
        .CLK                          (CLK),
        .reset                        (reset),
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
        .host_bias_read_data          (host_bias_read_data),
        .host_layernorm_write_enable  (host_layernorm_write_enable),
        .host_layernorm_write_gain    (host_layernorm_write_gain),
        .host_layernorm_write_offset  (host_layernorm_write_offset)
    );

    always #5 CLK = ~CLK;

    logic [31:0]                    descriptor_mem[DESCRIPTOR_WORDS*DESCRIPTOR_COUNT];
    logic [LANES*DATA_WIDTH-1:0]    image_mem[IMAGE_WORDS_MAX];
    logic [SYS_COL*ACC_WIDTH-1:0]   bias_image_mem[1024];
    logic [OFFSET_WIDTH+GAIN_WIDTH-1:0] layernorm_image_mem[1024];
    logic [DATA_WIDTH-1:0]          expected_mem[VECTOR_WORDS_MAX];

    // the program: one record per descriptor entry
    // names are kept as character vectors: xelab 2022.2 crashes on arrays of string
    localparam int NAME_BITS = 8 * 40;
    logic [NAME_BITS-1:0] op_phase[DESCRIPTOR_COUNT];
    logic [NAME_BITS-1:0] op_name[DESCRIPTOR_COUNT];
    logic [NAME_BITS-1:0] op_kind[DESCRIPTOR_COUNT];
    logic [NAME_BITS-1:0] op_file[DESCRIPTOR_COUNT];
    int    op_buffer[DESCRIPTOR_COUNT];
    bit    op_row_banks[DESCRIPTOR_COUNT];
    int    op_base[DESCRIPTOR_COUNT];
    int    op_stride[DESCRIPTOR_COUNT];
    int    op_rows[DESCRIPTOR_COUNT];
    int    op_cols[DESCRIPTOR_COUNT];
    int    op_count = 0;

    logic [NAME_BITS-1:0] phase_name[PHASES_MAX];
    int    phase_first[PHASES_MAX];
    int    phase_ops[PHASES_MAX];
    int    phase_count = 0;

    int stages_passed  = 0;
    int stages_failed  = 0;
    int entries_ok     = 0;
    int entries_wrong  = 0;
    longint total_cycles = 0;

    function automatic string text(input logic [NAME_BITS-1:0] characters);
        return $sformatf("%0s", characters);
    endfunction

    function automatic string vector_file(input string file_name);
        string name;
        int    fd;
        name = {VECTOR_DIR, file_name};
        fd = $fopen(name, "r");
        if (fd == 0) begin
            $fatal(1, "FAIL: cannot open golden vector file %s", name);
        end
        $fclose(fd);
        return name;
    endfunction

    // one word of a pool memory, read directly: generate indices must be constants
    `define PEEK_BUFFER(B) \
        B: case (bank) \
            0: return dut.pool[B].buffer_inst.bank[0].memory[address]; \
            1: return dut.pool[B].buffer_inst.bank[1].memory[address]; \
            2: return dut.pool[B].buffer_inst.bank[2].memory[address]; \
            3: return dut.pool[B].buffer_inst.bank[3].memory[address]; \
            4: return dut.pool[B].buffer_inst.bank[4].memory[address]; \
            5: return dut.pool[B].buffer_inst.bank[5].memory[address]; \
            6: return dut.pool[B].buffer_inst.bank[6].memory[address]; \
            default: return dut.pool[B].buffer_inst.bank[7].memory[address]; \
        endcase

    function automatic logic [DATA_WIDTH-1:0] peek(input int buffer_id, input int bank, input int address);
        case (buffer_id)
            `PEEK_BUFFER(0)
            `PEEK_BUFFER(1)
            `PEEK_BUFFER(2)
            `PEEK_BUFFER(3)
            `PEEK_BUFFER(4)
            `PEEK_BUFFER(5)
            `PEEK_BUFFER(6)
            default: return 'x;
        endcase
    endfunction

    // where element [row][col] of a stage tensor is stored
    function automatic int element_bank(input int entry, input int row, input int col);
        return op_row_banks[entry] ? row % LANES : col % LANES;
    endfunction

    function automatic int element_address(input int entry, input int row, input int col);
        return op_row_banks[entry] ? op_base[entry] + (row / LANES)*op_stride[entry] + col
                                   : op_base[entry] + (col / LANES)*op_stride[entry] + row;
    endfunction

    // compares the output tensor of a finished op with its golden stage, in zero time
    function automatic void check_stage(input int entry, input int cycles);
        int rows           = op_rows[entry];
        int cols           = op_cols[entry];
        int mismatches     = 0;
        int first_mismatch = -1;
        logic [DATA_WIDTH-1:0] got;

        if (rows*cols > VECTOR_WORDS_MAX) begin
            $fatal(1, "FAIL: stage %0s is larger than VECTOR_WORDS_MAX", op_name[entry]);
        end
        $readmemh(vector_file(text(op_file[entry])), expected_mem, 0, rows*cols - 1);
        for (int row = 0; row < rows; row++) begin
            for (int col = 0; col < cols; col++) begin
                got = peek(op_buffer[entry], element_bank(entry, row, col), element_address(entry, row, col));
                if (got !== expected_mem[row*cols + col]) begin
                    if (first_mismatch < 0) begin
                        first_mismatch = row*cols + col;
                        $display("  stage %0s first mismatch at index %0d (row %0d, col %0d): got %h, expected %h",
                                 text(op_name[entry]), first_mismatch, row, col, got, expected_mem[row*cols + col]);
                    end
                    mismatches++;
                end
            end
        end
        entries_ok    += rows*cols - mismatches;
        entries_wrong += mismatches;
        if (mismatches == 0) begin
            stages_passed++;
            $display("  %3d %-5s %-13s PASS  %-9s %0dx%0d in buffer %0d  %0d of %0d entries  cycles %0d",
                     entry, text(op_phase[entry]), text(op_name[entry]), text(op_kind[entry]), rows, cols, op_buffer[entry],
                     rows*cols, rows*cols, cycles);
        end
        else begin
            stages_failed++;
            $display("  %3d %-5s %-13s FAIL  %-9s %0dx%0d in buffer %0d  %0d of %0d entries wrong, first mismatch index %0d",
                     entry, text(op_phase[entry]), text(op_name[entry]), text(op_kind[entry]), rows, cols, op_buffer[entry],
                     mismatches, rows*cols, first_mismatch);
        end
    endfunction

    // the same tensor read back through the host port, after the phase
    task automatic host_check(input int entry);
        int rows           = op_rows[entry];
        int cols           = op_cols[entry];
        int mismatches     = 0;
        int first_mismatch = -1;

        $readmemh(vector_file(text(op_file[entry])), expected_mem, 0, rows*cols - 1);
        host_buffer_id = BUFFER_ID_WIDTH'(op_buffer[entry]);
        for (int tile = 0; tile < (rows + LANES - 1) / LANES; tile++) begin
            for (int col = 0; col < cols; col++) begin
                host_address = ADDRESS_WIDTH'(op_base[entry] + tile*op_stride[entry] + col);
                repeat (vit_config_pkg::BUFFER_READ_LATENCY) @(negedge CLK);
                for (int lane = 0; lane < LANES; lane++) begin
                    automatic int row = tile*LANES + lane;
                    if (row < rows && host_buffer_read_data[lane] !== expected_mem[row*cols + col]) begin
                        if (first_mismatch < 0) begin
                            first_mismatch = row*cols + col;
                        end
                        mismatches++;
                    end
                end
            end
        end
        entries_ok    += rows*cols - mismatches;
        entries_wrong += mismatches;
        if (mismatches == 0) begin
            stages_passed++;
            $display("      %-5s %-13s PASS  host port read back, %0d of %0d entries",
                     text(op_phase[entry]), text(op_name[entry]), rows*cols, rows*cols);
        end
        else begin
            stages_failed++;
            $display("      %-5s %-13s FAIL  host port read back, %0d of %0d entries wrong, first mismatch index %0d",
                     text(op_phase[entry]), text(op_name[entry]), mismatches, rows*cols, first_mismatch);
        end
    endtask

    // everything the host writes before a phase, one address per cycle
    task automatic host_load(input string phase);
        int    loads;
        int    fields;
        int    target, base, words;
        int    writes = 0;
        string load_phase, kind, file_name;

        loads = $fopen(vector_file("program_loads.txt"), "r");
        while (!$feof(loads)) begin
            fields = $fscanf(loads, "%s %s %d %d %d %s\n", load_phase, kind, target, base, words, file_name);
            if (fields != 6 || load_phase != phase) begin
                continue;
            end
            writes += words;
            if (kind == "buffer") begin
                if (words > IMAGE_WORDS_MAX) begin
                    $fatal(1, "FAIL: image %s is larger than IMAGE_WORDS_MAX", file_name);
                end
                $readmemh(vector_file(file_name), image_mem, 0, words - 1);
                host_buffer_id = BUFFER_ID_WIDTH'(target);
                for (int word = 0; word < words; word++) begin
                    host_address             = ADDRESS_WIDTH'(base + word);
                    host_buffer_write_enable = '1;
                    host_buffer_write_data   = image_mem[word];
                    @(negedge CLK);
                end
                host_buffer_write_enable = '0;
            end
            else if (kind == "bias") begin
                $readmemh(vector_file(file_name), bias_image_mem, 0, words - 1);
                for (int word = 0; word < words; word++) begin
                    host_address           = ADDRESS_WIDTH'(base + word);
                    host_bias_write_enable = '1;
                    host_bias_write_data   = bias_image_mem[word];
                    @(negedge CLK);
                end
                host_bias_write_enable = '0;
            end
            else if (kind == "layernorm") begin
                $readmemh(vector_file(file_name), layernorm_image_mem, 0, words - 1);
                for (int word = 0; word < words; word++) begin
                    host_address                = ADDRESS_WIDTH'(base + word);
                    host_layernorm_write_enable = 1'b1;
                    host_layernorm_write_gain   = layernorm_image_mem[word][GAIN_WIDTH-1:0];
                    host_layernorm_write_offset = layernorm_image_mem[word][GAIN_WIDTH +: OFFSET_WIDTH];
                    @(negedge CLK);
                end
                host_layernorm_write_enable = 1'b0;
            end
            else begin
                $fatal(1, "FAIL: unknown load kind %s", kind);
            end
        end
        $fclose(loads);
        $display("  phase %0s host wrote %0d addresses", phase, writes);
        // one extra cycle after the last write before start
        @(negedge CLK);
    endtask

    task automatic run_phase(input int phase);
        int entry       = phase_first[phase];
        int last        = phase_first[phase] + phase_ops[phase] - 1;
        // a new descriptor index is seen one cycle after the op before it ended,
        // so the first op of a phase starts one cycle early here and the last
        // one ends one cycle late
        int op_cycles   = -1;
        int cycles      = 0;
        int waited      = 0;
        bit finished    = 1'b0;

        host_load(text(phase_name[phase]));
        start_index = DESCRIPTOR_INDEX_WIDTH'(entry);
        start       = 1'b1;
        @(negedge CLK);
        start = 1'b0;
        // busy is high and the first descriptor index is loaded from here on
        while (!finished && waited < TIMEOUT_CYCLES) begin
            waited++;
            @(posedge CLK);
            // busy cycles, as in the cycle model: the cycle done is seen in is not one
            cycles    += busy;
            op_cycles += busy;
            if (done) begin
                check_stage(entry, op_cycles + 1);
                finished = 1'b1;
            end
            else if (int'(dut.descriptor_index) != entry) begin
                // the engine went on to the next descriptor: the op is complete
                check_stage(entry, op_cycles);
                op_cycles = 0;
                entry     = int'(dut.descriptor_index);
            end
        end
        @(negedge CLK);
        if (!finished || entry != last) begin
            stages_failed++;
            $display("  phase %0s FAIL: %s, at descriptor %0d of %0d..%0d", phase_name[phase],
                     finished ? "done came at the wrong descriptor" : "timed out", entry, phase_first[phase], last);
        end
        total_cycles += cycles;
        $display("  phase %0s %0d ops  cycles %0d", phase_name[phase], phase_ops[phase], cycles);
        // the phase's last output, through the host port
        host_check(last);
    endtask

    initial begin
        int file;
        int fields;
        int entry;
        int tokens;
        int plan_depth[BUFFER_COUNT];
        string layout;

        $display("tb_encoder_program: 8x8 array, buffer depths per bank %0d %0d %0d %0d %0d %0d %0d %0d",
                 BUFFER_DEPTH[0], BUFFER_DEPTH[1], BUFFER_DEPTH[2], BUFFER_DEPTH[3],
                 BUFFER_DEPTH[4], BUFFER_DEPTH[5], BUFFER_DEPTH[6], BUFFER_DEPTH[7]);
        reset                        = 1'b1;
        start                        = 1'b0;
        start_index                  = '0;
        host_descriptor_index        = '0;
        host_descriptor_write_enable = '0;
        host_descriptor_write_data   = '0;
        host_buffer_id               = '0;
        host_address                 = '0;
        host_buffer_write_enable     = '0;
        host_buffer_write_data       = '0;
        host_bias_write_enable       = '0;
        host_bias_write_data         = '0;
        host_layernorm_write_enable  = 1'b0;
        host_layernorm_write_gain    = '0;
        host_layernorm_write_offset  = '0;
        repeat (3) @(negedge CLK);
        reset = 1'b0;

        // the pool must be the one the program was laid out for
        file = $fopen(vector_file("program_plan.txt"), "r");
        fields = $fscanf(file, "%d %d %d %d %d %d %d %d %d", tokens, plan_depth[0], plan_depth[1], plan_depth[2],
                         plan_depth[3], plan_depth[4], plan_depth[5], plan_depth[6], plan_depth[7]);
        $fclose(file);
        for (int b = 0; b < BUFFER_COUNT; b++) begin
            if (fields != 9 || plan_depth[b] != int'(BUFFER_DEPTH[b])) begin
                $fatal(1, "FAIL: buffer %0d is %0d words per bank here, the program was laid out for %0d",
                       b, BUFFER_DEPTH[b], plan_depth[b]);
            end
        end
        $display("  program for %0d tokens", tokens);

        file = $fopen(vector_file("program_ops.txt"), "r");
        while (!$feof(file)) begin
            fields = $fscanf(file, "%d", entry);
            if (fields != 1) begin
                break;
            end
            fields = $fscanf(file, "%s %s %s %d %s %d %d %d %d %s\n",
                             op_phase[entry], op_name[entry], op_kind[entry], op_buffer[entry], layout,
                             op_base[entry], op_stride[entry], op_rows[entry], op_cols[entry], op_file[entry]);
            if (fields != 10 || entry != op_count) begin
                $fatal(1, "FAIL: program_ops.txt line %0d is malformed", op_count);
            end
            op_row_banks[entry] = layout == "row";
            op_count++;
        end
        $fclose(file);
        file = $fopen(vector_file("program_phases.txt"), "r");
        while (!$feof(file)) begin
            fields = $fscanf(file, "%s %d %d\n", phase_name[phase_count], phase_first[phase_count],
                             phase_ops[phase_count]);
            if (fields == 3) begin
                phase_count++;
            end
        end
        $fclose(file);
        if (op_count > DESCRIPTOR_COUNT) begin
            $fatal(1, "FAIL: %0d descriptors do not fit the descriptor memory", op_count);
        end

        // the descriptor table, once
        $readmemh(vector_file("program_desc.hex"), descriptor_mem, 0, DESCRIPTOR_WORDS*op_count - 1);
        for (int index = 0; index < op_count; index++) begin
            for (int word = 0; word < DESCRIPTOR_WORDS; word++) begin
                host_descriptor_index              = DESCRIPTOR_INDEX_WIDTH'(index);
                host_descriptor_write_enable       = '0;
                host_descriptor_write_enable[word] = 1'b1;
                host_descriptor_write_data         = descriptor_mem[index*DESCRIPTOR_WORDS + word];
                @(negedge CLK);
            end
        end
        host_descriptor_write_enable = '0;
        $display("  %0d descriptors written, %0d phases", op_count, phase_count);

        for (int phase = 0; phase < phase_count; phase++) begin
            run_phase(phase);
        end

        $display("  all phases  cycles %0d", total_cycles);
        if (stages_failed == 0 && stages_passed > 0) begin
            $display("PASS: %0d of %0d stage checks, %0d of %0d entries correct, %0d descriptors",
                     stages_passed, stages_passed, entries_ok, entries_ok + entries_wrong, op_count);
        end
        else begin
            $display("FAIL: %0d of %0d stage checks failed, %0d of %0d entries wrong",
                     stages_failed, stages_passed + stages_failed, entries_wrong, entries_ok + entries_wrong);
        end
        $finish;
    end

endmodule
