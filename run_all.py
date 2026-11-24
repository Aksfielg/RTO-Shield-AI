"""
run_all.py
==========
Runs all RTO Shield pipeline scripts in order.
Logs output to outputs/run_log.txt with timestamps and exit codes.
Exits with non-zero code if any script fails.

Usage:
    python run_all.py

This script uses the same Python interpreter it is running under,
so it automatically picks up the venv when invoked as:
    venv\\Scripts\\python.exe run_all.py
"""

import os

os.environ.setdefault("PYTHONUTF8", "1")        # child scripts inherit UTF-8 mode on Windows (cp1252 consoles crash on the banners)

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):        # this process prints "—" and similar too; make that safe on a cp1252 console
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE     = OUTPUTS_DIR / "run_log.txt"

SCRIPTS = [
    "01_prepare.py",
    "02_train.py",
    "05_tune_multiseed.py",        # produces the FINAL model + calibrator
    "03_calibrate_threshold.py",   # thresholds/tiers for the FINAL model
    "04_explain.py",
    "06_robustness.py",
    "07_predict_demo.py",
    "08_sanity_checks.py",
    "09_verification.py",
    "09b_type_audit.py",           # payment-type proxy audit -> outputs/type_audit.json
    "10_finalize.py",              # writes outputs/final_metrics.json
    "11_experiments.py",           # read-only experiments -> outputs/experiments.md (never replaces the model)
    "12_ceiling_and_challengers.py",  # read-only: lookup ceiling + challengers -> outputs/ceiling_and_challengers.md
    "13_operating_points.py",      # read-only: validation-chosen thresholds + capacity -> outputs/operating_points.md
    "11b_figure_captions.py",      # docs/figure_captions.{md,json}: exact order/settings/score of every screenshot (checked against it)
    "11_render_docs.py",           # README / MODEL_CARD / report_tables from JSON
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_script(script_name: str, log_fh) -> int:
    """Run a single script, stream output to console + log file."""
    script_path = PROJECT_ROOT / "src" / script_name
    header = f"\n{'=' * 70}\n[{ts()}]  STARTING: {script_name}\n{'=' * 70}\n"
    print(header, end="")
    log_fh.write(header)
    log_fh.flush()

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",           # children run in UTF-8 mode (PYTHONUTF8=1); decode with that, not the console code page
        errors="replace",
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)
        log_fh.write(line)
        log_fh.flush()

    proc.wait()
    elapsed = time.time() - t0
    footer = (
        f"\n[{ts()}]  FINISHED: {script_name}  |  "
        f"exit_code={proc.returncode}  |  elapsed={elapsed:.1f}s\n"
    )
    print(footer, end="")
    log_fh.write(footer)
    log_fh.flush()
    return proc.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    failures = []

    with open(LOG_FILE, "w", encoding="utf-8") as log_fh:
        banner = (
            f"RTO Shield AI — Full Pipeline Run\n"
            f"Started:  {ts()}\n"
            f"Python:   {sys.executable}\n"
            f"Python version: {platform.python_version()}\n"
            f"{'=' * 70}\n"
        )
        print(banner)
        log_fh.write(banner)

        for script in SCRIPTS:
            rc = run_script(script, log_fh)
            if rc != 0:
                msg = f"[FAIL]  {script} FAILED (exit code {rc}) — stopping pipeline.\n"
                print(msg)
                log_fh.write(msg)
                failures.append(script)
                break  # stop on first failure to avoid cascading errors
            else:
                msg = f"[OK]  {script} OK\n"
                print(msg)
                log_fh.write(msg)

        total_elapsed = time.time() - start_time
        total_minutes = round(total_elapsed / 60, 1)
        summary = (
            f"\n{'=' * 70}\n"
            f"Pipeline complete.  Elapsed: {total_minutes:.1f} min\n"
            f"Total minutes: {total_minutes:.1f}\n"
            f"Python version: {platform.python_version()}\n"
            f"Failures: {failures if failures else 'None'}\n"
            f"Log saved to: {LOG_FILE}\n"
        )
        print(summary)
        log_fh.write(summary)

    if failures:
        sys.exit(1)

    record_run(total_minutes)


def record_run(total_minutes: float) -> None:
    """Store wall-clock minutes and the Python version in final_metrics.json (run.*), then re-render the documents from it."""
    fm_path = OUTPUTS_DIR / "final_metrics.json"
    if not fm_path.exists():
        return
    fm = json.loads(fm_path.read_text(encoding="utf-8"))
    ver = platform.python_version()
    fm["run"] = {"total_minutes": total_minutes, "python": ver, "python_short": ".".join(ver.split(".")[:2]),
                 "platform": platform.platform(), "cpu_threads": os.cpu_count(), "finished": ts()}
    fm_path.write_text(json.dumps(fm, indent=2), encoding="utf-8")
    print(f"[run] recorded run.total_minutes={total_minutes} run.python={ver} in {fm_path.name}; re-rendering documents")
    rc = subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "11_render_docs.py")], cwd=str(PROJECT_ROOT),
                        env={**os.environ, "PYTHONUTF8": "1"}).returncode
    if rc != 0:
        print("[run] document re-render FAILED")
        sys.exit(rc)


if __name__ == "__main__":
    main()

