# Synthesis, implementation and bitstream of the Arty A7-100T project, then the
# numbers that matter.
#
# Batch:  vivado -mode batch -source scripts/build_arty_bitstream.tcl
#         vivado -mode batch -source scripts/build_arty_bitstream.tcl -tclargs arty_top
# GUI:    Tools > Run Tcl Script... > scripts/build_arty_bitstream.tcl
#
# It uses the project of scripts/create_arty_project.tcl and creates it if
# build/arty_a7_project is not there (or was made for the other top). The runs
# are the project's own synth_1 and impl_1, the ones "Run Synthesis", "Run
# Implementation" and "Generate Bitstream" start in the GUI, with their default
# strategies, so both ways give the same result.
#
# Output in build/ (not tracked):
#   <top>.bit            the bitstream
#   <top>_summary.txt    WNS, TNS, WHS, THS and utilisation, as printed
#   arty_a7_project/     the project, with every report under arty_a7.runs/

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file dirname $script_dir]
set project_file [file join $repo_dir build arty_a7_project arty_a7.xpr]

if {![info exists arty_top_name]} {
    set arty_top_name arty_top_axi
}
if {[info exists argv] && [llength $argv] > 0} {
    set arty_top_name [lindex $argv 0]
}

if {[llength [get_projects -quiet]] == 0 && [file exists $project_file]} {
    open_project $project_file
}
if {[llength [get_projects -quiet]] == 0
    || [get_property top [get_filesets sources_1]] ne $arty_top_name} {
    source [file join $script_dir create_arty_project.tcl]
}

reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    error "build_arty_bitstream: synthesis failed, see build/arty_a7_project/arty_a7.runs/synth_1/runme.log"
}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "build_arty_bitstream: implementation failed, see build/arty_a7_project/arty_a7.runs/impl_1/runme.log"
}

set run [get_runs impl_1]
set wns [get_property STATS.WNS $run]
set tns [get_property STATS.TNS $run]
set whs [get_property STATS.WHS $run]
set ths [get_property STATS.THS $run]

open_run impl_1 -name impl_1
set report [report_utilization -return_string]
proc used {report label} {
    if {[regexp -line "^\\|\\s*[string map {/ \\/ * \\*} $label]\\S*\\s*\\|\\s*(\[0-9.\]+)\\s*\\|" $report -> value]} {
        return $value
    }
    return "?"
}
set lines [list \
    "top            $arty_top_name" \
    "part           [get_property PART [current_project]]" \
    "WNS            $wns ns" \
    "TNS            $tns ns" \
    "WHS            $whs ns" \
    "THS            $ths ns" \
    "LUT            [used $report {Slice LUTs}]" \
    "FF             [used $report {Slice Registers}]" \
    "BRAM36         [used $report {  RAMB36/FIFO}]" \
    "BRAM18         [used $report {  RAMB18}]" \
    "BRAM tiles     [used $report {Block RAM Tile}]" \
    "DSP            [used $report {DSPs}]"]

set run_dir [get_property DIRECTORY $run]
set bitstream [file join $repo_dir build $arty_top_name.bit]
file copy -force [file join $run_dir $arty_top_name.bit] $bitstream
lappend lines "bitstream      $bitstream"
if {$wns < 0 || $whs < 0} {
    lappend lines "TIMING IS NOT MET: do not use this bitstream"
}

set fh [open [file join $repo_dir build ${arty_top_name}_summary.txt] w]
foreach line $lines {
    puts $fh $line
    puts "build_arty_bitstream: $line"
}
close $fh
