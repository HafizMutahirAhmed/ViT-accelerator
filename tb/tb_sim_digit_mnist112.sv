`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// tb_axi_replay_mnist112 for any image: the same testbench, unchanged, with its
// vectors taken from the run directory, and every byte the design sends written
// to dut_response.hex there (one byte per line, in order) so that the host
// script can read the stages back. The replay testbench's own checks and its
// PASS or FAIL line are the result.
//
// python/sim_digit.py writes the host stream of one image (host_stream_tx.hex,
// host_stream_rx.hex, host_stream_packets.txt) into sim/sim_sim_digit_mnist112_SLOT<n>/
// and runs scripts/sim.sh sim_digit_mnist112 SLOT=<n>. SLOT only names the run
// directory, so that several images can run at once.
//
// sim.sh: also compile axi_rule_checker.sv
// sim.sh: also compile tb_axi_replay_mnist112.sv
//////////////////////////////////////////////////////////////////////////////////

module tb_sim_digit_mnist112 #(
    parameter int SLOT = 0
);

    tb_axi_replay_mnist112 #(
        .VECTOR_DIR ("")
    ) replay ();

    int dump;

    initial begin
        dump = $fopen("dut_response.hex", "w");
        if (dump == 0) begin
            $fatal(1, "FAIL: cannot write dut_response.hex");
        end
        $display("tb_sim_digit_mnist112: slot %0d, response bytes to dut_response.hex", SLOT);
    end

    always @(posedge replay.CLK) begin
        if (replay.tx_valid && replay.tx_ready) begin
            $fwrite(dump, "%h\n", replay.tx_data);
        end
    end

    final begin
        $fclose(dump);
    end

endmodule
