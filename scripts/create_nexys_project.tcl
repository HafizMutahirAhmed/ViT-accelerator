# Creates a fresh Vivado project for the Nexys A7-100T in build/nexys_a7_project.
#
# Batch:  vivado -mode batch -source scripts/create_nexys_project.tcl
# GUI:    Tools > Run Tcl Script... > scripts/create_nexys_project.tcl
#
# The top level is nexys_top_axi (UART > bridge > AXI wrapper > engine). The
# project refers to the sources where they are in the repository; nothing is
# copied:
#   rtl/core/*.sv, rtl/interface/*.sv and *.v, boards/nexys_a7/*.sv
#   rtl/rom/mnist112/*.hex           the ROM images, as memory files
#   boards/nexys_a7/nexys_a7.xdc     pins and clock, from Digilent's master XDC
#   boards/nexys_a7/nexys_config.xdc configuration bank voltage
# An existing build/nexys_a7_project is replaced. build/ is not tracked.
#
# The ROM images are named in the RTL relative to rtl/core
# ("../rom/mnist112/gelu_rom.hex"), which is where Vivado looks first.

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file dirname $script_dir]
set project_dir [file join $repo_dir build nexys_a7_project]

set top_name nexys_top_axi

# a project that is open (GUI) is closed first
if {[llength [get_projects -quiet]] > 0} {
    close_project
}
file mkdir [file join $repo_dir build]
create_project -force nexys_a7 $project_dir -part xc7a100tcsg324-1
set_property target_language Verilog [current_project]

# the package first, so that every other file can import it
set package [file join $repo_dir rtl core vit_config_pkg.sv]
set sources [list $package]
foreach pattern {rtl/core/*.sv rtl/interface/*.sv rtl/interface/*.v boards/nexys_a7/*.sv} {
    foreach f [lsort [glob -nocomplain [file join $repo_dir {*}[split $pattern /]]]] {
        if {$f ne $package} { lappend sources $f }
    }
}
add_files -norecurse -fileset sources_1 $sources
set_property file_type SystemVerilog [get_files -of_objects [get_filesets sources_1] *.sv]

set roms [lsort [glob [file join $repo_dir rtl rom mnist112 *.hex]]]
add_files -norecurse -fileset sources_1 $roms
set_property file_type {Memory File} [get_files -of_objects [get_filesets sources_1] *.hex]

add_files -norecurse -fileset constrs_1 [list \
    [file join $repo_dir boards nexys_a7 nexys_a7.xdc] \
    [file join $repo_dir boards nexys_a7 nexys_config.xdc]]

set_property top $top_name [get_filesets sources_1]
update_compile_order -fileset sources_1

puts "create_nexys_project: [file join $project_dir nexys_a7.xpr], top $top_name,\
[llength $sources] source files, [llength $roms] ROM images"
