"""
Repository hygiene and reproducibility guards: licence, citation, pinned dependencies, Python range, raw-data check,
Windows launchers, untracked artefacts, README facts that must come from data (run time, Python version), screenshots.
"""

import importlib.metadata as md
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

FM = json.loads((ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))
DOCS = ["README.md", "MODEL_CARD.md", "docs/VIVA_CHEATSHEET.md"]


def _load_prepare():
    spec = importlib.util.spec_from_file_location("prepare01_hygiene", ROOT / "src" / "01_prepare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ licence + citation
def test_license_file_exists_whenever_the_docs_mention_mit():
    mentions = [d for d in DOCS + ["templates/README.md.tmpl"] if re.search(r"\bMIT\b", (ROOT / d).read_text(encoding="utf-8"))]
    assert mentions, "expected the README to state the code licence"
    lic = ROOT / "LICENSE"
    assert lic.exists(), f"{mentions} mention MIT but there is no LICENSE file"
    text = lic.read_text(encoding="utf-8")
    assert text.startswith("MIT License") and "Permission is hereby granted" in text
    assert "2026" in text and all(n in text for n in ("Shitij Chanana", "Saksham Singla", "Rishabh Kumar"))


def test_readme_states_the_data_licence_and_that_the_dataset_is_not_redistributed():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "CC BY 4.0" in text and "10.17632/8gx2fvg2k6.5" in text
    assert re.search(r"not redistributed", text) and "CITATION.cff" in text


def test_citation_cff_covers_the_project_and_the_dataset():
    c = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for needle in ("cff-version: 1.2.0", "title:", "authors:", "Chanana", "Singla", "Kumar", "license: MIT", "references:", "type: dataset",
                   "doi: 10.17632/8gx2fvg2k6.5", "CC-BY-4.0", "Constante", "Silva", "Pereira", "year: 2019"):
        assert needle in c, f"CITATION.cff is missing {needle!r}"


# ------------------------------------------------------------------ dependencies and Python range
def _pins():
    lines = [ln.split("#")[0].strip() for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def test_requirements_are_pinned_and_match_the_installed_versions():
    pins = _pins()
    assert len(pins) >= 12
    for ln in pins:
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][A-Za-z0-9.\-]*", ln), f"not an exact pin: {ln!r}"
        name, ver = ln.split("==")
        try:
            assert md.version(name) == ver, f"{name} is {md.version(name)} here but requirements.txt pins {ver}"
        except md.PackageNotFoundError:
            pytest.fail(f"{name} is pinned but not installed")


def test_python_range_is_derived_from_the_pins_and_stated_consistently():
    def minor(spec):
        m = re.search(r">=\s*3\.(\d+)", spec or "")
        return int(m.group(1)) if m else 0
    need = max(minor(md.metadata(ln.split("==")[0]).get("Requires-Python")) for ln in _pins())
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = int(re.search(r'requires-python\s*=\s*">=3\.(\d+)"', pyproject).group(1))
    assert declared == need, f"pyproject says >=3.{declared} but the pinned libraries need >=3.{need}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"Python 3.{declared} or newer" in readme
    assert (ROOT / "requirements.lock").exists() and "==" in (ROOT / "requirements.lock").read_text(encoding="utf-8")


# ------------------------------------------------------------------ facts that must come from data
def test_run_time_and_python_version_come_from_the_run_and_reach_the_docs():
    run = FM["run"]
    assert run["total_minutes"] and run["python"].count(".") == 2
    log = (ROOT / "outputs" / "run_log.txt").read_text(encoding="utf-8")
    assert f"Total minutes: {run['total_minutes']:.1f}" in log and f"Python version: {run['python']}" in log
    for doc in ("README.md", "MODEL_CARD.md", "docs/VIVA_CHEATSHEET.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert f"{run['total_minutes']:.1f} minutes" in text, f"{doc} does not quote the measured run time"
        assert run["python"] in text, f"{doc} does not quote the Python version of the run"


# ------------------------------------------------------------------ raw-data check (01_prepare.py)
def test_raw_data_check_stops_with_a_clear_message(tmp_path):
    prep = _load_prepare()
    with pytest.raises(SystemExit) as e:
        prep.verify_raw_csv(tmp_path / "nope.csv")
    msg = str(e.value)
    assert "RAW DATA MISSING" in msg and "10.17632/8gx2fvg2k6.5" in msg and "latin-1" in msg and "data/raw/" in msg
    bad = tmp_path / "bad.csv"
    bad.write_text("not the dataset", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        prep.verify_raw_csv(bad)
    assert "RAW DATA MISMATCH" in str(e.value) and "Mendeley Data V5" in str(e.value)
    assert prep.verify_raw_csv(bad, expected_sha=prep.sha256_of(bad)) == prep.sha256_of(bad)
    import pandas as pd
    with pytest.raises(SystemExit) as e:
        prep.verify_raw_shape(pd.DataFrame({"a": [1, 2]}))
    assert "RAW DATA SHAPE MISMATCH" in str(e.value) and "180,519" in str(e.value)


def test_expected_raw_file_facts_live_in_config_and_match_the_metrics_file():
    import config
    assert config.RAW_CSV_SHAPE == (180519, 53) and config.RAW_CSV_DOI == "10.17632/8gx2fvg2k6.5"
    assert FM["dataset"]["sha256"] == config.RAW_CSV_SHA256
    assert (FM["dataset"]["raw_rows"], FM["dataset"]["raw_columns"]) == config.RAW_CSV_SHAPE


# ------------------------------------------------------------------ Windows robustness
def test_windows_launchers_and_utf8_defaults():
    for name in ("run_all.bat", "run_app.bat"):
        bat = (ROOT / name).read_text(encoding="utf-8")
        assert "PYTHONUTF8=1" in bat and r"venv\Scripts\activate.bat" in bat
    assert "python run_all.py" in (ROOT / "run_all.bat").read_text(encoding="utf-8")
    assert "streamlit run app.py" in (ROOT / "run_app.bat").read_text(encoding="utf-8")
    for py in ("run_all.py", "app.py"):
        assert 'os.environ.setdefault("PYTHONUTF8", "1")' in (ROOT / py).read_text(encoding="utf-8"), py
    # run_all must decode its children's UTF-8 output itself (a fresh-clone run on a cp1252 console crashed without this)
    assert 'encoding="utf-8"' in (ROOT / "run_all.py").read_text(encoding="utf-8").split("subprocess.Popen(")[1].split(")")[0]


# ------------------------------------------------------------------ git tracks no artefacts
def test_git_tracks_no_models_data_venv_or_caches():
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git unavailable")
    bad = [f for f in files if re.search(r"\.(joblib|parquet|pkl|zip)$", f) or re.match(r"data/raw/.+\.csv$", f)
           or f.startswith("venv/") or "__pycache__" in f or f.endswith(".pyc")]
    assert not bad, f"tracked files that must stay out of git: {bad}"
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pat in ("models/*.joblib", "data/processed/*.parquet", "data/raw/*.csv", "venv/", "__pycache__/", "dist/"):
        assert pat in ignore, f".gitignore lacks {pat}"


# ------------------------------------------------------------------ screenshots
def test_screenshots_exist_and_the_readme_embeds_only_existing_ones():
    shots = ROOT / "docs" / "screenshots"
    names = {"home", "score-order", "real-orders", "impact", "performance", "robustness", "limitations", "batch"}
    assert {p.stem for p in shots.glob("*.png")} >= names
    assert all((shots / f"{n}.png").stat().st_size > 20_000 for n in names)
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", (ROOT / "README.md").read_text(encoding="utf-8"))
    embedded = [l for l in links if l.startswith("docs/screenshots/")]
    assert len(embedded) >= 3
    for l in embedded:
        assert (ROOT / l).exists(), f"README embeds a missing image: {l}"
