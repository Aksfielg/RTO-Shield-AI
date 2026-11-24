"""
The published test count: final_metrics.json (tests.total / tests.by_file, collected by 10_finalize.py with
`pytest --collect-only -q`) must equal the live collection, and README, MODEL_CARD, every template and the app must quote it.
If this fails after adding or removing tests, run `python src/10_finalize.py` and `python src/11_render_docs.py`.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from test_app_pages import all_text, page_app  # noqa: E402

FM = json.loads((ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))


def _live_collection() -> tuple[int, dict]:
    p = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests/"],
                       cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"})
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-400:]
    by = {}
    for line in p.stdout.splitlines():
        if "::" in line:
            f = line.split("::")[0].replace("\\", "/").split("/")[-1]
            by[f] = by.get(f, 0) + 1
    total = int(re.search(r"(\d+) tests? collected", p.stdout).group(1))
    return total, by


def test_published_total_equals_the_live_pytest_collection():
    total, by = _live_collection()
    t = FM["tests"]
    assert t["total"] == total, f"final_metrics.json says {t['total']} tests, pytest collects {total}: rerun src/10_finalize.py"
    assert sum(r["tests"] for r in t["by_file"]) == t["total"]
    published = {r["file"]: r["tests"] for r in t["by_file"] if r["tests"]}
    assert published == by


def test_every_test_file_is_listed_with_a_complete_description():
    on_disk = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))
    listed = [r["file"] for r in FM["tests"]["by_file"]]
    assert sorted(listed) == on_disk, "tests.by_file must list EVERY test file (including test_imports.py)"
    assert "test_imports.py" in listed and next(r for r in FM["tests"]["by_file"] if r["file"] == "test_imports.py")["tests"] >= 1
    assert FM["tests"]["n_files"] == len(on_disk)
    for r in FM["tests"]["by_file"]:
        d = r["description"]
        assert len(d) > 25 and d[0].isupper() and d.endswith("."), f"{r['file']}: description is not a complete sentence: {d!r}"
        assert not d.rstrip(".").endswith(":") and "..." not in d, r["file"]


def test_readme_model_card_viva_and_all_templates_quote_the_same_total():
    total = FM["tests"]["total"]
    for doc in ("README.md", "MODEL_CARD.md", "docs/VIVA_CHEATSHEET.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert re.search(rf"\b{total}(?:\*\*)? automated tests\b", text), f"{doc} does not quote {total} automated tests"
        assert not [n for n in re.findall(r"\b(\d+)(?:\*\*)? automated tests\b", text) if int(n) != total], f"{doc} quotes another total"
    templates = sorted((ROOT / "templates").glob("*.tmpl"))            # every document source (there is no separate PDF source)
    assert templates
    for t in templates:
        assert "M.tests.total" in t.read_text(encoding="utf-8"), f"{t.name} does not take the test total from final_metrics.json"
    rt = (ROOT / "outputs" / "report_tables.md").read_text(encoding="utf-8")
    assert "{{" not in rt


def test_the_methodology_page_quotes_the_same_total_and_lists_every_file():
    at = page_app("methodology").run()
    assert not at.exception
    text = all_text(at)
    total = FM["tests"]["total"]
    assert f"{total:,} tests in {FM['tests']['n_files']} files" in text
    for r in FM["tests"]["by_file"]:
        assert r["file"] in text, f"{r['file']} missing from the Methodology table"
