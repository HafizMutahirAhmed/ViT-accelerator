"""Combine the package's Markdown documents into one PDF guide: an HTML file printed by headless Chrome.

Usage: .venv/bin/python scripts/build_guide_pdf.py <package root> <out.pdf>     (needs markdown)
The chapters are read from <package root>; test_results/SUMMARY.md is the second chapter when it
exists. scripts/make_package.sh calls this.
"""
import datetime
import os
import re
import subprocess
import sys

import markdown

repo, out_pdf = sys.argv[1], sys.argv[2]

# (file, anchor) in reading order
chapters = [
    ("docs/VIVADO_GUI.md", "ch-gui"),
    ("test_results/SUMMARY.md", "ch-tests"),
    ("README.md", "ch-readme"),
    ("docs/SPEC.md", "ch-spec"),
    ("docs/ENGINE.md", "ch-engine"),
    ("docs/HOST_PROTOCOL.md", "ch-host-protocol"),
    ("docs/AXI_REGISTER_MAP.md", "ch-axi"),
    ("docs/NEXYS_QUICKSTART.md", "ch-nexys"),
    ("boards/opal_kelly/README.md", "ch-opal-kelly"),
]
anchor_of = {os.path.basename(f) if f.startswith("docs/") else f: a for f, a in chapters}
anchor_of.update({f: a for f, a in chapters})


def relink(text):
    def repl(m):
        label, target = m.group(1), m.group(2)
        path = target.split("#")[0]
        if path in anchor_of:
            return f"[{label}](#{anchor_of[path]})"
        return m.group(0)
    return re.sub(r"\[([^\]]*)\]\(([^)#][^)]*)\)", repl, text)


chapters = [(path, anchor) for path, anchor in chapters if os.path.exists(os.path.join(repo, path))]
parts = []
titles = []
for path, anchor in chapters:
    text = open(os.path.join(repo, path), encoding="utf-8").read()
    title = re.search(r"^# (.+)$", text, re.M).group(1)
    titles.append((title, anchor))
    parts.append(f'\n\n<div class="chapter" id="{anchor}"></div>\n\n' + relink(text))

body = markdown.markdown("".join(parts), extensions=["tables", "fenced_code", "toc", "sane_lists"])
toc = "\n".join(f'<li><a href="#{a}">{t}</a></li>' for t, a in titles)
today = datetime.date.today().isoformat()

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DeiT-Tiny FPGA accelerator</title>
<style>
@page {{ size: A4; margin: 18mm 16mm; }}
body {{ font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #111; }}
h1 {{ font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }}
h2 {{ font-size: 13.5pt; margin-top: 1.4em; }}
h3 {{ font-size: 11.5pt; }}
.chapter {{ page-break-before: always; }}
code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 8.8pt; background: #f2f2f2; padding: 0 2px; }}
pre {{ background: #f5f5f5; border: 1px solid #ddd; padding: 6px 8px; font-size: 8.2pt; line-height: 1.3;
      white-space: pre-wrap; page-break-inside: avoid; }}
pre code {{ background: none; padding: 0; font-size: inherit; }}
table {{ border-collapse: collapse; margin: 0.6em 0; font-size: 9pt; width: 100%; }}
th, td {{ border: 1px solid #bbb; padding: 3px 6px; vertical-align: top; text-align: left; }}
th {{ background: #eee; }}
tr {{ page-break-inside: avoid; }}
a {{ color: #1a4f9c; text-decoration: none; }}
.cover {{ text-align: center; margin-top: 70mm; }}
.cover h1 {{ border: none; font-size: 24pt; }}
.toc li {{ margin: 0.3em 0; font-size: 11pt; }}
</style></head><body>
<div class="cover"><h1>DeiT-Tiny FPGA accelerator</h1>
<p>One to twelve INT8 DeiT-Tiny encoder layers on the Nexys A7-100T,<br>bit-exact against an integer golden model</p>
<p>{today}</p></div>
<div class="chapter"></div>
<h1>Contents</h1><ol class="toc">{toc}</ol>
{body}
</body></html>"""

html_path = os.path.splitext(os.path.abspath(out_pdf))[0] + ".html"
open(html_path, "w", encoding="utf-8").write(html)
subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={out_pdf}", "file://" + os.path.abspath(html_path)],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove(html_path)
print(out_pdf)
