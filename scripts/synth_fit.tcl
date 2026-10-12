# Synthesis of matmul_engine in the 112 x 112 MNIST configuration, to see what
# it needs before anything is built around it.
#
# Usage: vivado -mode batch -source scripts/synth_fit.tcl [-tclargs <part>]
#
# The part defaults to the Arty A7-100T's, xc7a100tcsg324-1. The pool depths are
# the plan of golden/layer_program.py for 50 tokens and a 16,384-word weight
# buffer (program_plan.txt), the ROM images are rtl/rom/mnist112. Reports go to
# sim/synth_fit/ (gitignored): utilization.rpt, utilization_hier.rpt.

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file dirname $script_dir]
set part xc7a100tcsg324-1
if {[llength $argv] > 0} {
    set part [lindex $argv 0]
}
set out_dir [file join $repo_dir sim synth_fit]
file mkdir $out_dir

# the configuration as a wrapper, so the string parameters are plain SystemVerilog
set wrapper [file join $out_dir engine_mnist112_fit.sv]
set fh [open $wrapper w]
puts $fh {
module engine_mnist112_fit (
    input  logic        CLK,
    input  logic        reset,
    input  logic        start,
    input  logic [7:0]  start_index,
    output logic        busy,
    output logic        done,
    input  logic [7:0]  host_descriptor_index,
    input  logic [4:0]  host_descriptor_write_enable,
    input  logic [31:0] host_descriptor_write_data,
    input  logic [2:0]  host_buffer_id,
    input  logic [16:0] host_address,
    input  logic [7:0]  host_buffer_write_enable,
    input  logic [63:0] host_buffer_write_data,
    output logic [63:0] host_buffer_read_data,
    input  logic [7:0]  host_bias_write_enable,
    input  logic [255:0] host_bias_write_data,
    output logic [255:0] host_bias_read_data,
    input  logic        host_layernorm_write_enable,
    input  logic [15:0] host_layernorm_write_gain,
    input  logic [31:0] host_layernorm_write_offset
);
    matmul_engine #(
        .SYS_ROW        (8),
        .SYS_COL        (8),
        .BUFFER_DEPTH   ({32'd16, 32'd1344, 32'd2544, 32'd1050, 32'd1344, 32'd5376, 32'd16384, 32'd1344}),
        .GELU_ROM_FILE  ("rom/mnist112/gelu_rom.hex"),
        .EXP_ROM_FILE   ("rom/mnist112/exp_rom.hex"),
        .RSQRT_ROM_FILE ("rom/mnist112/rsqrt_rom.hex")
    ) engine (.*);
endmodule
}
close $fh

set rtl_dir [file join $repo_dir rtl]
set package [file join $rtl_dir vit_config_pkg.sv]
set others {}
foreach f [lsort [glob -directory $rtl_dir *.sv]] {
    if {$f ne $package} { lappend others $f }
}
read_verilog -sv [concat $package $others $wrapper]
synth_design -top engine_mnist112_fit -part $part -mode out_of_context
report_utilization -file [file join $out_dir utilization.rpt]
report_utilization -hierarchical -file [file join $out_dir utilization_hier.rpt]
