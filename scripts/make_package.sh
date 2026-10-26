#!/usr/bin/env bash
# Make the package for the GUI user: a zip of the committed tree without the scripts, with the
# vectors, the models, the MNIST test and training files, the bitstream, the test results and one
# PDF guide.
#
# Usage: scripts/make_package.sh <results_dir> [out.zip]
#
# <results_dir> is what the final check of a commit wrote:
#   results.txt, <set>.log           scripts/run_gui_sets.tcl on vivado/vit.xpr, every simulation set
#   nexys_top_ddr.bit                Generate Bitstream of vivado/vit.xpr
#   build.txt                        "WNS <ns> WHS <ns>" of that build
#   timing_summary.txt, utilization.txt
# The default zip is ../ViT-implementation-<commit>.zip. The tree is HEAD, so commit first.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
results=$(cd "$1" && pwd)
commit=$(git -C "$root" rev-parse --short HEAD)
out=${2:-$root/../ViT-implementation-$commit.zip}
out=$(realpath -m "$out")

if [ -n "$(git -C "$root" status --porcelain --untracked-files=no)" ]; then
    echo "make_package.sh: the working tree has uncommitted changes; the package is HEAD ($commit)" >&2
fi
for f in results.txt nexys_top_ddr.bit build.txt timing_summary.txt utilization.txt; do
    [ -f "$results/$f" ] || { echo "make_package.sh: $results/$f is missing" >&2; exit 1; }
done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
stage=$work/ViT-implementation
mkdir -p "$stage"
git -C "$root" archive HEAD | tar -x -C "$stage"
rm -rf "$stage/scripts" "$stage/CLAUDE.md" "$stage/.gitignore"

mkdir -p "$stage/python/models" "$stage/python/mnist/MNIST/raw" "$stage/bitstream" "$stage/test_results"
cp -r "$root/python/vectors" "$stage/python/"
for model in nexys_L2 nexys_L6 nexys_L12 opal_kelly_L2; do
    cp -r "$root/python/models/$model" "$stage/python/models/"
done
cp "$root"/python/mnist/MNIST/raw/*.gz "$stage/python/mnist/MNIST/raw/"
cp "$results/nexys_top_ddr.bit" "$stage/bitstream/"
cp "$results"/*.log "$results/timing_summary.txt" "$results/utilization.txt" "$stage/test_results/"
# The check ran in a temporary copy; the logs name files relative to the project folder instead.
sed -i -E -e 's#/[^ "]*/ViT-implementation/##g' -e 's#-file /[^ ]*/#-file #' "$stage"/test_results/*

python3 - "$results" "$stage/test_results/SUMMARY.md" "$commit" <<'EOF'
import datetime, os, re, sys
results, out, commit = sys.argv[1:4]
rows = [line.rstrip("\n").split("\t") for line in open(os.path.join(results, "results.txt")) if line.strip()]
passed = sum(1 for _, _, result in rows if result.startswith("PASS"))
wns, whs = re.search(r"WNS (\S+) WHS (\S+)", open(os.path.join(results, "build.txt")).read()).groups()
utilization = open(os.path.join(results, "utilization.txt")).read()

def used(name):
    m = re.search(r"^\|\s*" + re.escape(name) + r"\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", utilization, re.M)
    return f"{m.group(1)} of {m.group(2)} ({m.group(3)} %)" if m else "see utilization.txt"

def duration(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min" if seconds >= 3600 else f"{seconds // 60} min {seconds % 60:02d} s"

date = datetime.date.fromtimestamp(os.path.getmtime(os.path.join(results, "results.txt"))).isoformat()
lines = [
    "# Test results", "",
    f"Every simulation set of `vivado/vit.xpr` was run in Vivado 2022.2 on Linux, the way **Run Simulation** "
    f"runs it, on {date} (git commit `{commit}`). Each set's full simulation log is `test_results/<set>.log`; "
    "running a set in the GUI prints the same PASS line.", "",
    f"**{passed} of {len(rows)} simulation sets pass.**", "",
    "| Simulation set | Result | Time |", "|---|---|---|",
]
for name, seconds, result in rows:
    lines.append(f"| `{name}` | {result.replace('|', '/')} | {duration(seconds)} |")
lines += [
    "", "## Bitstream", "",
    f"`bitstream/nexys_top_ddr.bit` is the result of **Generate Bitstream** in `vivado/vit.xpr` at the same commit, "
    f"for the Nexys A7-100T (xc7a100tcsg324-1) at 100 MHz.", "",
    "| | |", "|---|---|",
    f"| Worst setup slack (WNS) | {wns} ns |",
    f"| Worst hold slack (WHS) | {whs} ns |",
    f"| LUTs | {used('Slice LUTs')} |",
    f"| Flip-flops | {used('Slice Registers')} |",
    f"| Block RAM tiles | {used('Block RAM Tile')} |",
    f"| DSPs | {used('DSPs')} |",
    "", "The full reports are `test_results/timing_summary.txt` and `test_results/utilization.txt`.", "",
]
open(out, "w").write("\n".join(lines))
EOF

"$root/.venv/bin/python" "$root/scripts/build_guide_pdf.py" "$stage" "$stage/ViT_accelerator_guide.pdf" > /dev/null

rm -f "$out"
(cd "$work" && zip -q -r -y "$out" ViT-implementation)
echo "make_package.sh: $out ($(du -h "$out" | cut -f1)), commit $commit"
