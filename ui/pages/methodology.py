"""Methodology: pipeline, allowed vs forbidden features, leakage guards, split dates, proxy target."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import streamlit as st

from config import CATEGORICAL_FEATURES, FORBIDDEN_FEATURES, NUMERIC_FEATURES, TARGET_COL
from ui import components as C
from ui import data

TESTS_DIR = Path(__file__).resolve().parents[2] / "tests"

PIPELINE_DOT = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF2FF", color="#4F46E5", fontname="Helvetica", fontsize=11];
  edge [color="#94A3B8"];
  raw [label="Raw DataCo CSV\\n(line items)", fillcolor="#F1F5F9", color="#94A3B8"];
  prep [label="01 prepare\\norder-level table +\\nleakage-safe history"];
  train [label="02 train\\n5 candidate models"];
  tune [label="05 tune + calibrate\\nfinal LightGBM\\n+ isotonic"];
  tiers [label="03 tiers + thresholds\\nlow-density gaps"];
  explain [label="04 explain\\nSHAP"];
  robust [label="06 robustness\\nablation, backtest, PSI"];
  checks [label="08 / 09 sanity +\\nverification"];
  fin [label="10 finalize\\nfinal_metrics.json"];
  app [label="Dashboard, README,\\nMODEL_CARD", fillcolor="#D1FAE5", color="#10B981"];
  raw -> prep -> train -> tune -> tiers -> fin;
  tune -> explain -> fin; tune -> robust -> fin; tune -> checks -> fin;
  fin -> app;
}
"""


def _test_inventory() -> pd.DataFrame:
    rows = []
    for p in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        if not names:                      # script-style smoke tests collect no pytest items
            continue
        doc = [ln.strip() for ln in (ast.get_docstring(tree) or "").splitlines()]
        text = [ln for ln in doc if ln and set(ln) - set("=-") and not ln.startswith("tests/") and not ln.startswith("Run")]
        rows.append({"file": p.name, "tests": len(names), "purpose": text[0] if text else ""})
    return pd.DataFrame(rows)


def render() -> None:
    fm, thr = data.fm(), data.thr()
    C.page_header("Methodology", "How the model is built, what it may and may not see, and how leakage is prevented.")

    with st.container(border=True):
        C.card_title("1. Pipeline")
        st.graphviz_chart(PIPELINE_DOT)
        st.caption("Each step is a script in `src/`; `run_all.py` runs them in dependency order. Every number shown in this app is "
                   "computed once in step 10 and read from `outputs/final_metrics.json` and `models/thresholds.json`.")

    with st.container(border=True):
        C.card_title("2. Allowed vs forbidden features")
        rows = ([{"feature": f, "kind": "categorical", "status": "✅ allowed (pre-dispatch)"} for f in CATEGORICAL_FEATURES]
                + [{"feature": f, "kind": "numeric", "status": "✅ allowed (pre-dispatch)"} for f in NUMERIC_FEATURES]
                + [{"feature": f, "kind": "excluded", "status": "⛔ forbidden"} for f in FORBIDDEN_FEATURES])
        why = {"Days_for_shipping_real": "post-dispatch outcome", "Delivery Status": "post-dispatch outcome", "Order Status": "post-dispatch outcome",
               "shipping_date": "post-dispatch timestamp", "aux_target_cancel_fraud": "evaluation-only secondary target"}
        df = pd.DataFrame(rows)
        df["reason"] = df["feature"].map(lambda f: why.get(f, "customer PII / post-dispatch / non-feature" if f in FORBIDDEN_FEATURES else "known at order placement"))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Allowed: {len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES)} features**")
            C.styled_table(df[df["kind"] != "excluded"][["feature", "kind", "reason"]], height=320)
        with c2:
            st.markdown(f"**Forbidden: {len(FORBIDDEN_FEATURES)} columns**")
            C.styled_table(df[df["kind"] == "excluded"][["feature", "reason"]], height=320)

    with st.container(border=True):
        C.card_title("3. Leakage guards")
        ls = fm["label_shuffle"]
        st.markdown(
            "- **History rule:** customer and country history features only count earlier orders whose delivery outcome was already "
            "*known* before the current order was placed (outcome date earlier than order date); a brute-force test re-derives them from the raw data.\n"
            "- **Chronological split:** train, validation and test never overlap in time; encoders, scalers and imputers are fitted on train only.\n"
            "- **Forbidden columns:** post-dispatch outcomes, PII and the secondary target are never model inputs (enforced by tests).\n"
            f"- **Label-shuffle check:** retraining on shuffled labels gives AUC {ls['val_roc_auc']:.3f} (validation) / {ls['test_roc_auc']:.3f} (test): "
            f"{ls['status']}.\n"
            "- **One-shot test:** thresholds, tiers and calibration are chosen on validation; the test set is used only to report.")
        st.markdown("**Automated tests (pytest)**")
        C.styled_table(_test_inventory(), rename={"file": "Test File", "tests": "Tests", "purpose": "What It Guards"})

    with st.container(border=True):
        C.card_title("4. Split dates")
        names = {"train": "Train", "val": "Validation", "test": "Test (holdout)"}
        sp = pd.DataFrame([{"split": names[k], "orders": v["orders"], "from": v["date_min"], "to": v["date_max"], "prevalence": v["positive_rate"]}
                           for k, v in fm["splits"].items()])
        C.styled_table(sp, rename={"split": "Split", "from": "From", "to": "To"},
                       formats={"orders": "{:,}", "prevalence": lambda v: C.fmt_pct(v, clip=False)})

    with st.container(border=True):
        C.card_title("5. The proxy target")
        st.markdown(
            f"The target `{TARGET_COL}` says whether an order's delivery exceeded its contracted shipping window in the public DataCo "
            "dataset. It is **not** Return-To-Origin (RTO), doorstep refusal or COD failure.\n\n"
            "- The label follows near rule-based dispatch logic (for example Same Day orders placed after noon), so part of the model's "
            "skill is learning that rule.\n"
            "- Cancelled / suspected-fraud orders are all labelled on-time, which makes the TRANSFER payment type look protective.\n"
            "- Real Indian D2C RTO involves NDR attempts, PIN-code reachability and COD behaviour, none of which are in this data.\n"
            "- Treat the results as a methodology prototype; retrain on real courier and return outcomes before any operational use.")
