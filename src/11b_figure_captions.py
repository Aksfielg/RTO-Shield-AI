"""
src/11b_figure_captions.py
==========================
Step 11b - docs/figure_captions.md and docs/figure_captions.json.

For every screenshot in docs/screenshots/ this records the page, the exact order id or settings shown and the exact score(s).
The values are computed with the SAME code path the dashboard uses (predict.score_order / predict.score_batch / the Impact
economics function / final_metrics.json) and are then CROSS-CHECKED against the text that tools/capture_screenshots.py read
from the rendered page (docs/screenshots/manifest.json).  If a screenshot is stale (different order, score or number), this
step fails instead of publishing a wrong caption.  README reads docs/figure_captions.json (namespace F) in 11_render_docs.py.

Usage:  python src/11b_figure_captions.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
for p in (str(SRC_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import predict  # noqa: E402
from config import TARGET_COL  # noqa: E402
from ui import components as C  # noqa: E402
from ui import data  # noqa: E402
from ui.pages import impact, score_order  # noqa: E402

SHOTS = ROOT / "docs" / "screenshots"
OUT_MD, OUT_JSON = ROOT / "docs" / "figure_captions.md", ROOT / "docs" / "figure_captions.json"
README_FIGURES = {"home", "score-order", "performance"}          # embedded in README.md


def pct(v: float) -> str:
    return C.fmt_pct(v)


def build() -> dict:
    fm, thr, opts = data.fm(), data.thr(), data.option_lists()
    reps, (_, test_df) = data.representative_orders(), data.datasets()
    t = fm["test"]
    high = next(p for p in fm["policies"] if p["policy"].startswith("High tier"))
    figs: dict[str, dict] = {}

    # ---- Home
    figs["home"] = dict(
        page="Home", file="home.png", settings="Default page, no controls used.",
        shows=f"ROC-AUC {t['roc_auc']:.3f} (95% CI [{t['roc_auc_ci'][0]:.3f}, {t['roc_auc_ci'][1]:.3f}]); High-tier precision {pct(high['precision'])}",
        expect=[f"{t['roc_auc']:.3f}", pct(high["precision"])])

    # ---- Score an Order: the preset shown, plus the Custom-input comparisons for context
    hi = reps["High"]
    res = predict.score_order(hi)
    d = score_order.default_values({}, opts)
    base_vals = {"Shipping_Mode": d["Shipping_Mode"], "Type": d["Type"], "order_value": d["order_value"], "n_items": d["n_items"],
                 "avg_discount_rate": d["avg_discount_rate"], "Market": d["Market"], "Department_Name": d["Department_Name"],
                 "Customer_Segment": d["Customer_Segment"], "order_hour": d["order_hour"], "order_dow": d["order_dow"],
                 "order_month": d["order_month"]}

    def custom(**over):
        vals = {**base_vals, **over}
        return predict.score_order(score_order.build_order({}, vals, d, opts["mode_days"]))

    c_default = custom()
    c_sc_cash = custom(Shipping_Mode="Second Class", Type="CASH")
    figs["score-order"] = dict(
        page="Score an Order", file="score-order.png",
        settings=f"Preset 'Real high-risk order #{hi['Order Id']}' selected (all form fields loaded from that holdout row).",
        shows=(f"preset order #{hi['Order Id']}, {hi['Shipping_Mode']}, payment type {hi['Type']}, scored {pct(res['risk_score'])} ({res['tier'].upper()} tier, "
               f"{'flagged' if res['flagged'] else 'not flagged'}); for contrast, Custom input with its default values "
               f"({d['Shipping_Mode']}, {d['Type']}) scores {pct(c_default['risk_score'])} and Custom input Second Class + CASH scores {pct(c_sc_cash['risk_score'])}"),
        facts={"preset_order_id": int(hi["Order Id"]), "preset_mode": hi["Shipping_Mode"], "preset_type": hi["Type"],
               "preset_score": float(res["risk_score"]), "preset_tier": res["tier"],
               "custom_default": {"mode": d["Shipping_Mode"], "type": d["Type"], "score": float(c_default["risk_score"])},
               "custom_second_class_cash": {"score": float(c_sc_cash["risk_score"])}},
        expect=[f"#{hi['Order Id']}", pct(res["risk_score"])])

    # ---- Real Test Orders: first order of the default (unfiltered) dropdown
    oid = int(test_df["Order Id"].head(200).iloc[0])
    row = test_df[test_df["Order Id"] == oid].iloc[0].to_dict()
    r = predict.score_order(row)
    figs["real-orders"] = dict(
        page="Real Test Orders", file="real-orders.png",
        settings="Default filters (Shipping mode = All, Actual outcome = All); the first order in the dropdown is shown in detail.",
        shows=(f"order #{oid}, {row['Shipping_Mode']}, payment type {row['Type']}, scored {pct(r['risk_score'])} ({r['tier'].upper()}), "
               f"actually {'late' if int(row[TARGET_COL]) == 1 else 'on time'}"),
        expect=[f"#{oid}", pct(r["risk_score"])])

    # ---- Impact Simulator: widget defaults
    sc = data.test_scores()
    cal, y = sc["cal"], sc["y"]
    modes = test_df["Shipping_Mode"].values
    all_modes = sorted(set(modes))
    inside_modes = [m for m in all_modes if m != "First Class"]
    e = impact.economics(cal >= thr["tier_cutoffs"]["high_min"], y, np.isin(modes, inside_modes), 15.0, 2.5, 0.30, 0.0)
    figs["impact"] = dict(
        page="Impact Simulator", file="impact.png",
        settings=("Default controls: failed delivery $15.00, intervention $2.50, modes Same Day + Second Class + Standard Class (First Class excluded), "
                  "30% success inside, 0% outside, policy 'Flag High tier'."),
        shows=f"{e['n']:,} orders targeted, {e['late']:,} late orders reached, net benefit {C.fmt_money(e['net'])}",
        expect=[f"{e['n']:,}", C.fmt_money(e["net"])])

    # ---- Performance / Robustness / Limitations: default tabs
    figs["performance"] = dict(
        page="Performance", file="performance.png", settings="Default tab 'Model comparison'.",
        shows=(f"test ROC-AUC {t['roc_auc']:.3f}, PR-AUC {t['pr_auc']:.3f}, High-tier precision {pct(high['precision'])}, ECE {t['ece_calibrated']:.3f}"),
        expect=[f"{t['roc_auc']:.3f}", f"{t['pr_auc']:.3f}", f"{t['ece_calibrated']:.3f}"])
    std = next(x for x in fm["per_mode"] if x["mode"] == "Standard Class")
    figs["robustness"] = dict(
        page="Robustness", file="robustness.png", settings="Default tab 'Per shipping mode'.",
        shows=f"per-mode ROC-AUC with 95% CIs; Standard Class {std['roc_auc']:.3f}", expect=["Standard Class", f"{std['roc_auc']:.3f}"])
    figs["limitations"] = dict(
        page="Limitations", file="limitations.png", settings="Default view, no controls.",
        shows="the six limitation cards (no numbers are the subject of this figure)", expect=["Limitations"])

    # ---- Batch Scoring: first sample button
    sample = test_df.sample(50, random_state=42).reset_index(drop=True)
    scored = predict.score_batch(sample)
    ev = predict.evaluate_batch(sample[TARGET_COL].astype(int).values, scored["raw_score"].values, scored["risk_score"].values, scored["tier"].values)
    counts = {k: int((scored["tier"] == k).sum()) for k in ("Low", "Medium", "High")}
    figs["batch"] = dict(
        page="Batch Scoring", file="batch.png",
        settings="Button '50 test orders (with target)' clicked (50 holdout orders, sample seed 42).",
        shows=(f"{len(scored)} orders: {counts['Low']} Low / {counts['Medium']} Medium / {counts['High']} High; batch ROC-AUC {ev['roc_auc']:.3f} "
               "(a demo sample, not a result)"),
        expect=[f"{ev['roc_auc']:.3f}", str(counts["High"])])

    # ---- cross-check against the text read from the screenshots
    manifest_path = SHOTS / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("docs/screenshots/manifest.json missing: run `python tools/capture_screenshots.py` with the app running")
    pages = json.loads(manifest_path.read_text(encoding="utf-8"))["pages"]
    problems = []
    for key, f in figs.items():
        shot = pages.get(key)
        if shot is None or not (SHOTS / f["file"]).exists():
            problems.append(f"{key}: screenshot or manifest entry missing")
            continue
        for needle in f["expect"]:
            if needle not in shot["text"] and needle not in (shot.get("radio") or ""):
                problems.append(f"{key}: expected {needle!r} on the captured page but it is not there (stale screenshot?)")
        if key == "score-order" and f"#{hi['Order Id']}" not in (shot.get("radio") or ""):
            problems.append(f"score-order: the selected preset in the screenshot is {shot.get('radio')!r}, not #{hi['Order Id']}")
    if problems:
        raise SystemExit("FIGURE CAPTION CHECK FAILED:\n  " + "\n  ".join(problems) + "\nRe-run tools/capture_screenshots.py on the current app.")
    return figs


def write(figs: dict) -> None:
    out = {}
    for i, (key, f) in enumerate(figs.items(), start=1):
        out[key] = {**f, "number": i, "in_readme": key in README_FIGURES,
                    "caption": f"Figure {i}, {f['page']} page: {f['shows']}. Settings: {f['settings']}"}
        out[key].pop("expect", None)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Figure captions", "",
             "> Generated by `src/11b_figure_captions.py` from the dashboard's own code path and checked against the text captured from each "
             "screenshot (`docs/screenshots/manifest.json`). Do not edit by hand. **Every figure caption in a report must name which of the "
             "views below it shows** (preset order versus Custom input, which tab, which sample).", "",
             "| # | Screenshot | Page | Exact order / settings shown | Exact values shown | Used in |", "|---|---|---|---|---|---|"]
    for key, f in out.items():
        lines.append(f"| {f['number']} | `docs/screenshots/{f['file']}` | {f['page']} | {f['settings']} | {f['shows']} | "
                     f"{'README.md and reports' if f['in_readme'] else 'reports'} |")
    lines += ["", "## Score an Order: which view is which", "",
              "The page has four starting points: three real holdout presets and a Custom input. The screenshot shows the **preset**; the Custom-input "
              "scores below are what the same page shows when you choose Custom input instead:", ""]
    so = out["score-order"]["facts"]
    lines += [f"- **Preset order #{so['preset_order_id']}** ({so['preset_mode']}, {so['preset_type']}): **{pct(so['preset_score'])}**, {so['preset_tier']} tier (this is the screenshot).",
              f"- **Custom input, default values** ({so['custom_default']['mode']}, {so['custom_default']['type']}): **{pct(so['custom_default']['score'])}**.",
              f"- **Custom input, Second Class + CASH** (other fields default): **{pct(so['custom_second_class_cash']['score'])}**.", ""]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    figs = build()
    write(figs)
    print(f"  wrote {OUT_MD.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)} ({len(figs)} figures, all checked against the screenshots)")


if __name__ == "__main__":
    main()
