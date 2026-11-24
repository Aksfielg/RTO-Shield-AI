"""
Stale-text tripwire: retired numbers and wordings must never come back into the rendered documents or the UI source.

Deliberately NOT banned: the bare word "synthetic" (it appears legitimately in "no synthetic data") and 0.771 (the legitimate
10-seed mean ROC-AUC).  Numbers are matched as whole numbers, so e.g. 0.3750x or 10.699 do not trigger.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

DOCS = ["README.md", "MODEL_CARD.md", "docs/VIVA_CHEATSHEET.md", "outputs/report_tables.md"]

# (label, regex).  Numbers use look-arounds so only the exact retired value matches.
STALE = [
    ("0.375", r"(?<![\d.])0\.375(?!\d)"),
    ("0.699", r"(?<![\d.])0\.699(?!\d)"),
    ("p33", r"\bp33\b"),
    ("p67", r"\bp67\b"),
    ("about 0.78", r"about 0\.78"),
    ("94.4%", r"(?<![\d.])94\.4%"),
    ("synthetic dispatch", r"synthetic dispatch"),
    ("synthetic rule", r"synthetic rule"),
    ("30-60 minutes", r"30\s*[-–]\s*60\s*min"),
    ("Python 3.11", r"Python 3\.11|3\.11\+"),
    ('"$-" (negative-money glitch)', r"\$-"),
]
UNRENDERED = ("unrendered {{ placeholder", r"\{\{")          # only meaningful in rendered documents (UI source uses {{ in f-strings)


def _ui_sources():
    return sorted((ROOT / "ui").rglob("*.py")) + [ROOT / "app.py"]


def _hits(text: str, patterns) -> list[str]:
    return [label for label, rx in patterns if re.search(rx, text)]


@pytest.mark.parametrize("doc", DOCS)
def test_rendered_documents_have_no_stale_strings_or_placeholders(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    bad = _hits(text, STALE + [UNRENDERED])
    assert not bad, f"{doc} contains retired text: {bad}"


@pytest.mark.parametrize("path", _ui_sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_ui_source_has_no_stale_strings(path):
    bad = _hits(path.read_text(encoding="utf-8"), STALE)
    assert not bad, f"{path.relative_to(ROOT)} contains retired text: {bad}"


def test_the_detector_itself_works_and_does_not_over_ban():
    assert _hits("flags 0.375 of orders", STALE) == ["0.375"]
    assert _hits("p67 cutoff and p33 cutoff", STALE) == ["p33", "p67"]
    assert _hits("shows $-12.50", STALE) == ['"$-" (negative-money glitch)']
    assert _hits("an unrendered {{ M.test.roc_auc }}", [UNRENDERED])
    # legitimate text must pass
    ok = "There is no synthetic data. The 10-seed mean is 0.771. AUC 0.3750x, 10.699 and 0.6988 are different numbers."
    assert _hits(ok, STALE) == []
    assert not any(lbl == "synthetic" or "0.771" in lbl for lbl, _ in STALE)
