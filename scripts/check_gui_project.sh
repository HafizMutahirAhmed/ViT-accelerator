#!/usr/bin/env bash
# The final check of a commit for the package: copies the tree (HEAD, or a tree-ish such as
# $(git write-tree) of the staged files) with the vectors and models to a new directory, runs
# Generate Bitstream of vivado/vit.xpr there, then every simulation set as Run Simulation does.
# Writes what scripts/make_package.sh takes to <results_dir>. Several hours.
#
# Usage: scripts/check_gui_project.sh <results_dir> [tree-ish] [work_dir]
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$1"
results=$(cd "$1" && pwd)
treeish=${2:-HEAD}
work=${3:-$(mktemp -d)}
tree=$work/ViT-implementation

rm -rf "$tree"
mkdir -p "$tree/python"
git -C "$root" archive "$treeish" | tar -x -C "$tree"
cp -r "$root/python/vectors" "$root/python/models" "$tree/python/"

cd "$work"
vivado -mode batch -nojournal -nolog -source "$root/scripts/build_gui_bitstream.tcl" \
    -tclargs "$tree/vivado/vit.xpr" "$results" | grep -E '^build_gui_bitstream|^ERROR' || true
vivado -mode batch -nojournal -nolog -source "$root/scripts/run_gui_sets.tcl" \
    -tclargs "$tree/vivado/vit.xpr" "$results" all | grep '^SETRESULT' || true
failed=$(awk -F'\t' '$3 !~ /^PASS/' "$results/results.txt" | wc -l)
echo "check_gui_project: $(wc -l < "$results/results.txt") simulation sets, $failed not PASS; $(cat "$results/build.txt" 2>/dev/null || echo 'no bitstream')"
[ "$failed" -eq 0 ] && [ -f "$results/nexys_top_ddr.bit" ]
