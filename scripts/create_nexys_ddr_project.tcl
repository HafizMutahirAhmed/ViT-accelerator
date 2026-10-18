# Creates a fresh Vivado project for the Nexys A7-100T with the DDR2 as external
# memory, in build/nexys_a7_ddr_project.
#
# Batch:  vivado -mode batch -source scripts/create_nexys_ddr_project.tcl
# GUI:    Tools > Run Tcl Script... > scripts/create_nexys_ddr_project.tcl
# Then synthesis, implementation and bitstream (in the GUI, or with
# scripts/build_nexys_ddr.tcl in batch).
#
# The top level is nexys_top_ddr (UART > accelerator_system > engine, and the
# block design memory_system: MIG, AXI DMA, SmartConnect). The project refers
# to the sources where they are in the repository:
#   rtl/core/*.sv, rtl/interface/*.sv and *.v
#   boards/nexys_a7/uart_rx.sv, uart_tx.sv, nexys_top_ddr.sv
#   rtl/rom/mnist112/*.hex                the ROM images, as memory files
#   boards/nexys_a7/nexys_a7_ddr.xdc      pins, from Digilent's master XDC
#   boards/nexys_a7/nexys_config.xdc      configuration bank voltage
#   boards/nexys_a7/nexys_ddr_timing.xdc  the calibration flag's synchroniser
# and generates in the project:
#   system_clock_generator   Clocking Wizard: 100 MHz and 200 MHz from CLK100MHZ
#   memory_system            block design, scripts/create_memory_system_bd.tcl,
#                            with the MIG configuration boards/nexys_a7/mig_ddr2.prj
# An existing build/nexys_a7_ddr_project is replaced. build/ is not tracked.

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file dirname $script_dir]
set project_dir [file join $repo_dir build nexys_a7_ddr_project]

set top_name nexys_top_ddr

if {[llength [get_projects -quiet]] > 0} {
    close_project
}
file mkdir [file join $repo_dir build]
create_project -force nexys_a7_ddr $project_dir -part xc7a100tcsg324-1
set_property target_language Verilog [current_project]

set package [file join $repo_dir rtl core vit_config_pkg.sv]
set sources [list $package]
foreach pattern {rtl/core/*.sv rtl/interface/*.sv rtl/interface/*.v} {
    foreach f [lsort [glob -nocomplain [file join $repo_dir {*}[split $pattern /]]]] {
        if {$f ne $package} { lappend sources $f }
    }
}
foreach name {uart_rx.sv uart_tx.sv nexys_top_ddr.sv} {
    lappend sources [file join $repo_dir boards nexys_a7 $name]
}
add_files -norecurse -fileset sources_1 $sources
set_property file_type SystemVerilog [get_files -of_objects [get_filesets sources_1] *.sv]

set roms [lsort [glob [file join $repo_dir rtl rom mnist112 *.hex]]]
add_files -norecurse -fileset sources_1 $roms
set_property file_type {Memory File} [get_files -of_objects [get_filesets sources_1] *.hex]

add_files -norecurse -fileset constrs_1 [list \
    [file join $repo_dir boards nexys_a7 nexys_a7_ddr.xdc] \
    [file join $repo_dir boards nexys_a7 nexys_config.xdc] \
    [file join $repo_dir boards nexys_a7 nexys_ddr_timing.xdc]]

# the clocking wizard: 100 MHz for the design and the MIG, 200 MHz for the MIG's reference
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
generate_target all [get_files system_clock_generator.xci]

# the memory system
source [file join $script_dir create_memory_system_bd.tcl]
set bd_file [get_files memory_system.bd]
generate_target all $bd_file
add_files -norecurse [make_wrapper -files $bd_file -top]

set_property top $top_name [get_filesets sources_1]
update_compile_order -fileset sources_1

puts "create_nexys_ddr_project: [file join $project_dir nexys_a7_ddr.xpr], top $top_name,\
[llength $sources] source files, [llength $roms] ROM images, IP system_clock_generator, block design memory_system"
