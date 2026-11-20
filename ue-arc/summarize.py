"""Results table: mean ± std over seeds for each results/<name>_s<N>.json group; single files (zero-shot) as-is.

python summarize.py [results]   -> markdown table
"""
import glob
import json
import os
import re
import statistics
import sys

groups = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "results", "*.json"))):
    name = re.sub(r"_s\d+$", "", os.path.basename(path)[:-len(".json")])
    with open(path) as f:
        groups.setdefault(name, []).append(100 * json.load(f)["accuracy"])

print("| run | seeds | accuracy (%) |\n|---|---|---|")
for name, accs in groups.items():
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print(f"| {name} | {len(accs)} | {statistics.mean(accs):.1f} ± {sd:.1f} |")
