# Creates the Vivado project <root>/vivado/vit.xpr for working in the Vivado GUI only:
#   - the Nexys A7 build, as scripts/create_nexys_ddr_project.tcl makes it: the sources, the
#     ROM images of rtl/rom/input_112, the constraints, the Clocking Wizard
#     system_clock_generator and the block design memory_system (with its wrapper imported);
#     Generate Bitstream builds nexys_top_ddr;
#   - one simulation set per run of scripts/regress.sh, named after the run, plus
#     axi_replay_run_nexys_L12 (about two hours) and dma_ip for
#     tb_dma_ip with the AXI DMA IP (dma_ip, configured as scripts/export_dma_ip_sim.tcl does).
#     A simulation set holds exactly the files scripts/sim.sh compiles for that run (it does not
#     take the design sources), its generics and the define VIT_VIVADO_PROJECT, which points the
#     testbenches' VIT_ROOT (tb/vit_paths.svh) from vivado/vit.sim/<set>/behav/xsim back to
#     <root>. Run Simulation runs it to the end; the Tcl console shows PASS or FAIL.
# The output products of the IPs and the block design are not generated here: Vivado makes them
# the first time they are needed. Every file is referred to relative to the project, so <root>
# can be moved or zipped.
#
# Batch: vivado -mode batch -source scripts/create_gui_project.tcl [-tclargs <root>]
# <root> is the tree the project refers to, the repository by default. An existing
# <root>/vivado is replaced.

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file dirname $script_dir]
set root [expr {[llength $argv] > 0 ? [file normalize [lindex $argv 0]] : $repo_dir}]
set project_dir [file join $root vivado]
set top_name nexys_top_ddr

# The runs of scripts/regress.sh and the 12-layer replay: testbench and generics.
set runs {
    {matmul_engine_opal_kelly {}}
    {matmul_engine_chain {SYS_ROW=4 SYS_COL=4}}
    {matmul_engine_chain {SYS_ROW=8 SYS_COL=8}}
    {matmul_engine {SYS_ROW=4 SYS_COL=4}}
    {matmul_engine {SYS_ROW=8 SYS_COL=8}}
    {drain {SYS_ROW=4 SYS_COL=4}}
    {drain {SYS_ROW=8 SYS_COL=8}}
    {drain {SYS_ROW=3 SYS_COL=5}}
    {systolic_array {SYS_ROW=2 SYS_COL=2}}
    {systolic_array {SYS_ROW=4 SYS_COL=4}}
    {systolic_array {SYS_ROW=3 SYS_COL=5}}
    {matmul_stream_controller {SYS_ROW=4 SYS_COL=4}}
    {matmul_stream_controller {SYS_ROW=8 SYS_COL=8}}
    {requantize {}}
    {residual_add {}}
    {gelu_lut {}}
    {layernorm {}}
    {softmax {}}
    {matmul_engine_nonlinear {}}
    {encoder_program {}}
    {encoder_program_nexys {}}
    {axi_replay_step {}}
    {axi_replay_run {}}
    {axi_replay_run_nexys_L6 {}}
    {axi_replay_run_nexys_L12 {}}
    {nexys_top_ddr {}}
}

if {[llength [get_projects -quiet]] > 0} {
    close_project
}
file delete -force $project_dir
create_project vit $project_dir -part xc7a100tcsg324-1
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# The engine and the interface, package first, as sim.sh compiles them.
set package [file join $root rtl core vit_config_pkg.sv]
set rtl_sources [list $package]
foreach pattern {rtl/core/*.sv rtl/interface/*.sv rtl/interface/*.v} {
    foreach f [lsort [glob -nocomplain [file join $root {*}[split $pattern /]]]] {
        if {$f ne $package} { lappend rtl_sources $f }
    }
}

# Design sources: the Nexys A7 build.
set sources $rtl_sources
foreach name {uart_rx.sv uart_tx.sv nexys_top_ddr.sv} {
    lappend sources [file join $root boards nexys_a7 $name]
}
add_files -norecurse -fileset sources_1 $sources
set_property file_type SystemVerilog [get_files -of_objects [get_filesets sources_1] *.sv]
set roms [lsort [glob [file join $root rtl rom input_112 *.hex]]]
add_files -norecurse -fileset sources_1 $roms
set_property file_type {Memory File} [get_files -of_objects [get_filesets sources_1] *.hex]
add_files -norecurse -fileset constrs_1 [list \
    [file join $root boards nexys_a7 nexys_a7_ddr.xdc] \
    [file join $root boards nexys_a7 nexys_config.xdc] \
    [file join $root boards nexys_a7 nexys_ddr_timing.xdc]]

create_ip -name clk_wiz -vendor xilinx.com -library ip -version 6.0 -module_name system_clock_generator
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ {100.000} \
    CONFIG.PRIM_SOURCE {Single_ended_clock_capable_pin} \
    CONFIG.CLKOUT2_USED {true} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {100.000} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {200.000} \
    CONFIG.USE_RESET {false} \
    CONFIG.USE_LOCKED {true} \
] [get_ips system_clock_generator]

# create_memory_system_bd.tcl reads the MIG configuration from $repo_dir.
set repo_dir_of_scripts $repo_dir
set repo_dir $root
source [file join $script_dir create_memory_system_bd.tcl]
set repo_dir $repo_dir_of_scripts
set bd_file [get_files memory_system.bd]
make_wrapper -files $bd_file -top -import

set_property top $top_name [get_filesets sources_1]
update_compile_order -fileset sources_1

# The AXI DMA of tb_dma_ip, outside the build: synthesised only if something used it.
create_ip -name axi_dma -vendor xilinx.com -library ip -version 7.1 -module_name dma_ip
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_addr_width {32} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {64} \
    CONFIG.c_mm2s_burst_size {64} \
    CONFIG.c_include_mm2s_dre {0} \
] [get_ips dma_ip]
set_property generate_synth_checkpoint false [get_files dma_ip.xci]

# A testbench that wraps another one names it in a line "// also compile: <file>".
proc also_compiled {root tb_file} {
    set files {}
    set in [open $tb_file r]
    foreach line [split [read $in] \n] {
        if {[regexp {^// also compile: (.*)$} $line -> name]} {
            lappend files [file normalize [file join $root tb [string trim $name]]]
        }
    }
    close $in
    return $files
}

proc set_name {module generics} {
    set name $module
    if {[regexp {^SYS_ROW=(\d+) SYS_COL=(\d+)$} $generics -> rows columns]} {
        append name _${rows}x${columns}
    }
    return $name
}

proc add_simulation_set {name top files generics} {
    create_fileset -simset $name
    set fileset [get_filesets $name]
    set_property SOURCE_SET {} $fileset
    add_files -norecurse -fileset $fileset $files
    set_property file_type SystemVerilog [get_files -of_objects $fileset *.sv]
    set_property file_type {Verilog Header} [get_files -of_objects $fileset *.svh]
    set_property top $top $fileset
    set_property top_lib xil_defaultlib $fileset
    set_property verilog_define {VIT_VIVADO_PROJECT} $fileset
    set_property include_dirs [list $::root/tb] $fileset
    if {$generics ne {}} {
        set_property generic $generics $fileset
    }
    set_property xsim.simulate.runtime {all} $fileset
    set_property xsim.simulate.log_all_signals false $fileset
}

set paths_header [file join $root tb vit_paths.svh]
foreach run $runs {
    lassign $run module generics
    set tb_file [file join $root tb tb_$module.sv]
    set files [concat [list $paths_header] $rtl_sources [also_compiled $root $tb_file] [list $tb_file]]
    add_simulation_set [set_name $module $generics] tb_$module $files $generics
}

set dma_files [concat [list $paths_header] $rtl_sources \
    [list [file join $root tb axi_rule_checker.sv] [file join $root tb tb_dma_ip.sv]]]
add_simulation_set dma_ip tb_dma_ip $dma_files {}
move_files -fileset [get_filesets dma_ip] [get_files dma_ip.xci]

current_fileset -simset [get_filesets encoder_program_nexys]
delete_fileset [get_filesets sim_1]

puts "create_gui_project: [file join $project_dir vit.xpr], top $top_name,\
[llength $sources] source files, [llength $roms] ROM images, [expr {[llength $runs] + 1}] simulation sets"
close_project
