"""Run the whole study end to end:  python run_all.py
Expected run time on a 2-core machine: ~40-60 minutes (Colab is faster).
Also records the exact package versions in results/ENVIRONMENT.txt."""
import subprocess, sys, os, platform
here = os.path.dirname(os.path.abspath(__file__))
res = os.path.join(os.path.dirname(here), "results")
os.makedirs(res, exist_ok=True)

import importlib.metadata as md
with open(os.path.join(res, "ENVIRONMENT.txt"), "w") as fh:
    fh.write(f"python {platform.python_version()} on {platform.platform()}\n")
    for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib", "openpyxl"]:
        try:
            fh.write(f"{p}=={md.version(p)}\n")
        except md.PackageNotFoundError:
            fh.write(f"{p}: not installed\n")

STEPS = ["01_audit_dataco.py", "02_audit_iot.py", "03_benchmark_dataco.py",
         "04_figure1.py", "05_figure2.py", "07_rule_ceiling.py",
         "08_figures_3_4.py", "06_results_workbook.py", "09_make_summary.py"]
for s in STEPS:
    print(f"\n{'=' * 70}\nRUNNING {s}\n{'=' * 70}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(here, s)], cwd=here)
    if r.returncode != 0:
        sys.exit(f"{s} failed")
print("\nAll steps finished. Tables -> results/, figures -> figures/, summary -> RESULTS_SUMMARY.md")