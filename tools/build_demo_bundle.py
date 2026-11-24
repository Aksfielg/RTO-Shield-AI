"""
tools/build_demo_bundle.py
==========================
Build dist/demo_artifacts.zip: the files the dashboard needs, so someone can run the app WITHOUT the pipeline and without the raw data.
Attach the zip to a GitHub Release (it is deliberately not committed: dist/ and *.zip are git-ignored).

Usage:
    python tools/build_demo_bundle.py            # run after `python run_all.py`
    unzip demo_artifacts.zip -d <project folder> # the files extract to models/, outputs/, data/processed/

The bundle contains the final model, calibrator, preprocessor, thresholds.json, final_metrics.json, the benchmark demo scores and the
processed order table. It does not contain the raw DataCo CSV (CC BY 4.0, not redistributed here).
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "models/final_model.joblib",
    "models/final_calibrator.joblib",
    "models/preprocessor.joblib",
    "models/thresholds.json",
    "outputs/final_metrics.json",
    "outputs/demo_scores.csv",
    "data/processed/orders.parquet",
]
OUT = ROOT / "dist" / "demo_artifacts.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"missing {missing}: run `python run_all.py` first")
    fm = json.loads((ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))
    notes = [
        "RTO Shield AI - Model 1: demo artefacts",
        "",
        "Unzip this file in the project folder (next to app.py). It creates/overwrites:",
        *[f"  {f}  (sha256 {sha256(ROOT / f)[:16]}...)" for f in FILES],
        "",
        f"Built from a pipeline run on Python {fm['run']['python']} ({fm['run']['total_minutes']} minutes); test ROC-AUC {fm['test']['roc_auc']:.4f}.",
        "Then:  pip install -r requirements.txt   and   streamlit run app.py",
        "The raw DataCo dataset (CC BY 4.0) is NOT included; see README section 8 to download it if you want to rebuild.",
    ]
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(ROOT / f, f)
        z.writestr("DEMO_ARTIFACTS.txt", "\n".join(notes) + "\n")
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.relative_to(ROOT)} ({mb:.1f} MB, {len(FILES)} files + DEMO_ARTIFACTS.txt); sha256 {sha256(OUT)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
