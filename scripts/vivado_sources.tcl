# Point the Vivado project at rtl/.
#
# Removes every design source that is not in rtl/ (the legacy files that used
# to live under ViT-implementation.srcs/) and adds rtl/*.sv, including
# vit_config_pkg.sv. Safe to run again after adding new files to rtl/.
#
# Usage: vivado -mode batch -source scripts/vivado_sources.tcl
#        vivado -mode batch -source scripts/vivado_sources.tcl -tclargs <project.xpr>
# or, with the project open in the GUI: source scripts/vivado_sources.tcl

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file dirname $script_dir]
set rtl_dir    [file join $repo_dir rtl]

set opened_here 0
if {[catch {current_project}]} {
    set project_file [file join $repo_dir ViT-implementation.xpr]
    if {[llength $argv] > 0} {
        set project_file [file normalize [lindex $argv 0]]
    }
    open_project $project_file
    set opened_here 1
}

set rtl_files [lsort [glob -directory $rtl_dir *.sv]]
set sources   [get_filesets sources_1]

set old_files {}
foreach f [get_files -quiet -of_objects $sources] {
    if {[lsearch -exact $rtl_files [file normalize $f]] < 0} {
        lappend old_files $f
    }
}
if {[llength $old_files] > 0} {
    remove_files -fileset $sources $old_files
}

add_files -quiet -norecurse -fileset $sources $rtl_files
set_property file_type SystemVerilog [get_files -of_objects $sources *.sv]
update_compile_order -fileset $sources

puts "vivado_sources: removed [llength $old_files] old source(s), sources_1 now has:"
foreach f [lsort [get_files -of_objects $sources]] {
    puts "  $f"
}

if {$opened_here} {
    close_project
}
