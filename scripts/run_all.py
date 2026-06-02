"""
Regenerate every figure and console table for the paper in one command.

Runs, in order:
  1. queueing_analysis.py   -> figures/queueing_analysis.png
  2. constellation_analysis.py -> figures/constellation.png
  3. robustness_sweeps.py   -> figures/robustness_sweeps.png
  4. cmdp_sweep.py          -> figures/cmdp_pareto.png

Each script's console output is captured to results/<name>.txt so the paper's
tables can be diffed against committed output.

Run:  python scripts/run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")

SCRIPTS = [
    "queueing_analysis.py",
    "constellation_analysis.py",
    "robustness_sweeps.py",
    "cmdp_sweep.py",
]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for script in SCRIPTS:
        path = os.path.join(HERE, script)
        print(f"\n{'='*70}\nRunning {script}\n{'='*70}", flush=True)
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        sys.stdout.write(out)
        log = os.path.join(RESULTS, script.replace(".py", ".txt"))
        with open(log, "w") as f:
            f.write(out)
        if proc.returncode != 0:
            print(f"!! {script} exited with code {proc.returncode}", flush=True)
    print(f"\nAll results written to {RESULTS}")


if __name__ == "__main__":
    main()
