# Runs simulation sets of vivado/vit.xpr one after another, as Run Simulation does in the GUI, and
# copies each set's simulation log to <out_dir>/<set>.log. Prints one line per set:
#   SETRESULT <set> <seconds> <the PASS or FAIL line, or what went wrong>
# and writes the same lines to <out_dir>/results.txt.
#
# Batch: vivado -mode batch -source scripts/run_gui_sets.tcl -tclargs <project.xpr> <out_dir> all|<set>...

lassign $argv project_file out_dir
set names [lrange $argv 2 end]
file mkdir $out_dir
open_project $project_file
if {$names eq "all"} {
    set names [lsort [get_filesets -filter {FILESET_TYPE == SimulationSrcs}]]
}
set results [open [file join $out_dir results.txt] w]
foreach set $names {
    current_fileset -simset [get_filesets $set]
    set start [clock seconds]
    if {[catch {launch_simulation -simset $set} message]} {
        set result "LAUNCH-ERROR [lindex [split $message \n] 0]"
    } else {
        close_sim -force
        set result "no PASS/FAIL line"
    }
    set seconds [expr {[clock seconds] - $start}]
    set log [file join [file dirname $project_file] vit.sim $set behav xsim simulate.log]
    if {[file exists $log]} {
        file copy -force $log [file join $out_dir $set.log]
        set in [open $log r]
        foreach line [split [read $in] \n] {
            if {[regexp {^(PASS|FAIL)} $line]} { set result $line }
        }
        close $in
    }
    puts "SETRESULT $set $seconds $result"
    puts $results "$set\t$seconds\t$result"
    flush $results
}
close $results
close_project
