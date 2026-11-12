"""
src/11_render_docs.py
=====================
Step 11 - Render README.md, MODEL_CARD.md, docs/VIVA_CHEATSHEET.md and outputs/report_tables.md from
outputs/final_metrics.json and models/thresholds.json ONLY.

Templates live in templates/*.tmpl and contain placeholders:

    {{ M.test.roc_auc | f4 }}              value from final_metrics.json
    {{ T.tier_cutoffs.high_min | f4 }}     value from thresholds.json
    {{ M.per_mode[mode=Standard Class].roc_auc | f4 }}   list lookup by key
    {{ @tier_table }}                      generated markdown table block

Formats: f2 f3 f4 | pct0 pct1 pct2 | int | ci4 (list -> [lo, hi]) | pp1 | raw

No metric is typed by hand in any template.  Missing keys fail loudly.

Usage:  python src/11_render_docs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODELS_DIR, OUTPUTS_DIR  # noqa: E402

TEMPLATES = ROOT / "templates"
M = json.loads((OUTPUTS_DIR / "final_metrics.json").read_text(encoding="utf-8"))
T = json.loads((MODELS_DIR / "thresholds.json").read_text(encoding="utf-8"))

MODEL_LABELS = {
    "baseline_lr": "Baseline LR (shipping+days)", "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM (default)",
}


# ---------------------------------------------------------------------------
# value lookup + formatting
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"([^.\[\]]+)|\[([^\]=]+)=([^\]]+)\]|\[(\d+)\]")


def lookup(path: str):
    root, _, rest = path.partition(".")
    cur = {"M": M, "T": T}[root]
    for m in re.finditer(r"\[([^\]=]+)=([^\]]+)\]|\[(\d+)\]|([^.\[\]]+)", rest):
        k, v, idx, key = m.groups()
        if key is not None:
            cur = cur[key]
        elif idx is not None:
            cur = cur[int(idx)]
        else:
            cur = next(r for r in cur if str(r[k]) == v)
    return cur


def fmt(value, spec: str) -> str:
    if value is None:
        return "n/a"
    if spec in ("f1", "f2", "f3", "f4"):
        return f"{value:.{int(spec[1])}f}"
    if spec.startswith("pct"):
        return f"{value * 100:.{int(spec[3])}f}%"
    if spec == "pp1":
        return f"{value * 100:+.1f} pp"
    if spec == "int":
        return f"{int(round(value)):,}"
    if spec == "ci4":
        return f"[{value[0]:.4f}, {value[1]:.4f}]"
    if spec == "raw":
        return str(value)
    raise ValueError(f"unknown format '{spec}'")


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# generated blocks
# ---------------------------------------------------------------------------
def _comp(split: str) -> str:
    rows = [r for r in M["model_comparison"] if r["split"] == split]
    cols = [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Brier", "brier"), ("Precision", "precision"),
            ("Recall", "recall"), ("F1", "f1"), ("Accuracy", "accuracy"), ("Balanced acc.", "balanced_accuracy"),
            ("Specificity", "specificity"), ("MCC", "mcc"), ("Prec@Top10%", "precision_at_top10pct"), ("ECE", "ece")]
    return md_table(
        ["Model"] + [c for c, _ in cols],
        [[MODEL_LABELS.get(r["model"], r["model"])] + [fmt(r.get(k), "f4") for _, k in cols] for r in rows])


def b_model_comparison_test(): return _comp("test")
def b_model_comparison_val(): return _comp("val")


def b_model_comparison_short() -> str:
    by = {(r["model"], r["split"]): r for r in M["model_comparison"]}
    rows = []
    for m in MODEL_LABELS:
        v, t = by[(m, "val")], by[(m, "test")]
        rows.append([MODEL_LABELS[m], f"{v['roc_auc']:.4f}", f"{t['roc_auc']:.4f}", f"{t['pr_auc']:.4f}",
                     f"{t['f1']:.3f}", f"{t['brier']:.4f}"])
    return md_table(["Model", "Val ROC-AUC", "Test ROC-AUC", "Test PR-AUC", "Test F1", "Test Brier"], rows)


def b_final_metrics() -> str:
    t = M["test"]
    thr_txt = f"{t['threshold_max_f1']:.4f}"
    return md_table(["Metric", "TEST", "95% bootstrap CI"], [
        ["ROC-AUC (raw scores)", f"{t['roc_auc']:.4f}", fmt(t["roc_auc_ci"], "ci4")],
        ["PR-AUC (raw scores)", f"{t['pr_auc']:.4f}", fmt(t["pr_auc_ci"], "ci4")],
        ["PR-AUC lift over prevalence", f"{t['pr_auc_lift']:.3f}", "-"],
        [f"F1 @ max-F1 threshold ({thr_txt})", f"{t['f1_at_max_f1']:.4f}", "-"],
        ["Precision @ max-F1 threshold", f"{t['precision_at_max_f1']:.4f}", "-"],
        ["Recall @ max-F1 threshold", f"{t['recall_at_max_f1']:.4f}", "-"],
        ["Accuracy @ max-F1 threshold", f"{t['accuracy_at_max_f1']:.4f}", "-"],
        ["Balanced accuracy @ max-F1 threshold", f"{t['balanced_accuracy_at_max_f1']:.4f}", "-"],
        ["Specificity @ max-F1 threshold", f"{t['specificity_at_max_f1']:.4f}", "-"],
        ["MCC @ max-F1 threshold", f"{t['mcc_at_max_f1']:.4f}", "-"],
        ["Brier (calibrated)", f"{t['brier_calibrated']:.4f}", "-"],
        ["ECE (calibrated)", f"{t['ece_calibrated']:.4f}", "-"],
    ])


def b_calibration_table() -> str:
    t = M["test"]
    return md_table(["Metric (TEST, final tuned model)", "Before calibration (raw)", "After isotonic calibration"], [
        ["Brier score", f"{t['brier_raw']:.4f}", f"{t['brier_calibrated']:.4f}"],
        ["ECE", f"{t['ece_raw']:.4f}", f"{t['ece_calibrated']:.4f}"],
    ])


def b_tier_table() -> str:
    c = M["tiers"]["cutoffs"]
    rng = {"Low": f"score < {c['low_max']:.4f}", "Medium": f"{c['low_max']:.4f} <= score < {c['high_min']:.4f}",
           "High": f"score >= {c['high_min']:.4f}"}
    rows = []
    for k in ["Low", "Medium", "High"]:
        s, d = M["tiers"]["test"][k], M["tiers"]["descriptions"][k]
        rows.append([k, rng[k], f"{s['orders']:,}", f"{s['share'] * 100:.1f}%", f"{s['late_rate'] * 100:.1f}%",
                     f"{s['late_captured'] * 100:.1f}%", d["action"]])
    return md_table(["Tier", "Score range", "Test orders", "Share", "Late rate", "Late captured", "Suggested handling"], rows)


def b_tier_descriptions() -> str:
    return "\n".join(f"- **{k}** - {M['tiers']['descriptions'][k]['description']}" for k in ["Low", "Medium", "High"])


def b_policy_table() -> str:
    rows = []
    for p in M["policies"]:
        th = "-" if p["threshold"] is None else f"{p['threshold']:.4f}"
        rows.append([p["policy"], th, f"{p['flagged_share'] * 100:.2f}% ({p['flagged_orders']:,})", f"{p['precision']:.4f}",
                     f"{p['recall']:.4f}", f"{p['f1']:.4f}", f"{p['accuracy']:.4f}", f"{p['balanced_accuracy']:.4f}",
                     f"{p['specificity']:.4f}", f"{p['mcc']:.4f}"])
    return md_table(["Policy", "Threshold", "Flagged", "Precision", "Recall", "F1", "Accuracy", "Balanced acc.",
                     "Specificity", "MCC"], rows)


def b_stability_table() -> str:
    rows = [[r["policy"], f"{r['shift']:+.2f}", f"{r['threshold']:.4f}", f"{r['flagged_share_pct']:.2f}%",
             f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['accuracy']:.4f}",
             f"{r['balanced_accuracy']:.4f}", f"{r['specificity']:.4f}", f"{r['mcc']:.4f}"]
            for r in M["stability"]["rows"]]
    return md_table(["Policy", "Shift", "Threshold", "Flagged share", "Precision", "Recall", "F1", "Accuracy",
                     "Balanced acc.", "Specificity", "MCC"], rows)


def b_stability_summary() -> str:
    sw = M["stability"]["swing_pp"]
    return "; ".join(f"{k}: {v:.2f} pp" for k, v in sw.items()) + f" (limit {M['stability']['limit_pp']:.0f} pp)"


def b_per_mode() -> str:
    rows = []
    for r in M["per_mode"]:
        ci = "n/a" if r["roc_auc"] is None else f"[{r['roc_auc_ci_lo']:.4f}, {r['roc_auc_ci_hi']:.4f}]"
        rows.append([r["mode"], f"{r['orders']:,}", f"{r['share'] * 100:.1f}%", f"{r['prevalence'] * 100:.1f}%",
                     fmt(r["roc_auc"], "f4"), ci, fmt(r["pr_auc"], "f4"),
                     "n/a" if r["pr_auc_lift"] is None else f"{r['pr_auc_lift']:.2f}x", r["flag"]])
    return md_table(["Shipping mode", "n (test)", "Share", "Prevalence", "ROC-AUC", "ROC-AUC 95% CI", "PR-AUC",
                     "PR-AUC lift", "Signal"], rows)


def b_per_mode_note() -> str:
    lim = [r["mode"] for r in M["per_mode"] if r["flag"] == "limited"]
    return ("Signal = **limited** when the ROC-AUC 95% bootstrap CI lower bound is below 0.55 (which also covers any CI "
            "that includes 0.5); otherwise informative. Limited modes: " + (", ".join(lim) if lim else "none") +
            ". PR-AUC lift = PR-AUC divided by the mode's prevalence (1.0x = no better than the base rate; a mode with "
            "very high prevalence cannot show much lift).")


def b_ranking_note() -> str:
    return ("ROC-AUC, PR-AUC, lifts and curves use the RAW (uncalibrated) model scores, because isotonic calibration "
            "collapses scores into ties and distorts rankings. Calibrated scores are used only for calibration (Brier, ECE), "
            "tiers, thresholds and every threshold-based metric (precision, recall, F1, accuracy, balanced accuracy, "
            "specificity, MCC).")


def b_ablation() -> str:
    rows = []
    for r in M["ablation"]:
        sig = "-" if r["label"] == "All features" else (
            f"{r['delta_roc_vs_all']:+.4f} [{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]"
            + (" *" if r["delta_significant"] else ""))
        rows.append([r["label"], r["n_features"], f"{r['roc_auc']:.4f}", f"[{r['roc_auc_ci_lo']:.4f}, {r['roc_auc_ci_hi']:.4f}]",
                     f"{r['pr_auc']:.4f}", f"{r['brier']:.4f}", f"{r['threshold']:.3f}", f"{r['f1']:.3f}",
                     f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['accuracy']:.3f}",
                     f"{r['balanced_accuracy']:.3f}", f"{r['specificity']:.3f}", f"{r['mcc']:.3f}", sig])
    return md_table(["Feature subset", "# feat.", "ROC-AUC", "ROC-AUC 95% CI", "PR-AUC", "Brier", "Own thr.", "F1",
                     "Precision", "Recall", "Accuracy", "Bal. acc.", "Spec.", "MCC", "Delta ROC-AUC vs all [95% CI]"], rows)


def b_ablation_verdict() -> str:
    return ("Threshold metrics use each ablated model's OWN max-F1 threshold chosen on validation. Delta = "
            "ROC-AUC(all features) - ROC-AUC(ablated), paired bootstrap, * = CI excludes 0. "
            + M["ablation_summary"]["history_verdict"])


def b_backtest() -> str:
    return md_table(["Fold", "Train N", "Test N", "Test start", "Test end", "Prevalence", "ROC-AUC", "PR-AUC",
                     "PR-AUC lift", "F1", "Accuracy", "Bal. acc.", "MCC"],
                    [[r["fold"], f"{r['train_n']:,}", f"{r['test_n']:,}", r["test_start"], r["test_end"],
                      f"{r['prevalence']:.3f}", f"{r['roc_auc']:.4f}", f"{r['pr_auc']:.4f}", f"{r['pr_auc_lift']:.3f}",
                      f"{r['f1']:.4f}", f"{r['accuracy']:.4f}", f"{r['balanced_accuracy']:.4f}", f"{r['mcc']:.4f}"]
                     for r in M["backtest"]["rows"]])


def b_psi() -> str:
    return md_table(["Feature", "PSI", "Status"], [[r["feature"], f"{r['psi']:.5f}", r["status"]] for r in M["psi"]["rows"]])


def b_shap_top() -> str:
    return md_table(["Rank", "Feature", "Mean |SHAP|", "Share of total"],
                    [[i + 1, r["feature"], f"{r['mean_abs_shap']:.4f}", f"{r['share'] * 100:.1f}%"]
                     for i, r in enumerate(M["shap"]["top_features"][:5])])


def b_shap_grouped() -> str:
    return md_table(["Business factor", "Mean |SHAP|", "Share of total"],
                    [[r["factor"], f"{r['mean_abs_shap']:.4f}", f"{r['share'] * 100:.1f}%"] for r in M["shap"]["grouped"][:6]])


def b_hyperparams() -> str:
    hp = M["model"]["hyperparameters"]
    return md_table(["Parameter", "Value"], [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in sorted(hp.items())])


def b_splits() -> str:
    names = {"train": "Train", "val": "Validation", "test": "Test"}
    return md_table(["Split", "Orders", "Date range", "Positive rate"],
                    [[names[k], f"{v['orders']:,}", f"{v['date_min']} -> {v['date_max']}", f"{v['positive_rate'] * 100:.2f}%"]
                     for k, v in M["splits"].items()])


def b_label_shuffle() -> str:
    s = M["label_shuffle"]
    return md_table(["Check", "Validation ROC-AUC", "Test ROC-AUC", "Status"],
                    [["Label shuffle (expect ~0.50)", f"{s['val_roc_auc']:.4f}", f"{s['test_roc_auc']:.4f}", s["status"]]])


def b_top_decile() -> str:
    td = M["score_clusters"]["top_decile"]
    return md_table(["Shipping mode", "Top-10% count", "Top-10% share", "Test count", "Overall share", "Late rate"],
                    [[r["shipping_mode"], r["top10_count"], f"{r['top10_share_pct']:.2f}%", f"{r['overall_test_count']:,}",
                      f"{r['overall_share_pct']:.2f}%", f"{r['empirical_late_rate'] * 100:.1f}%"] for r in td["rows"]])


def b_verification_segment() -> str:
    seg = M["verification"]["segment_models"]
    return md_table(["Segment", "Final model ROC-AUC", "Segment-only model ROC-AUC", "Final PR-AUC", "Segment-only PR-AUC"],
                    [[k, f"{v['final_model_roc_auc']:.4f}", f"{v['segment_model_roc_auc']:.4f}",
                      f"{v['final_model_pr_auc']:.4f}", f"{v['segment_model_pr_auc']:.4f}"] for k, v in seg.items()])


BLOCKS = {k[2:]: v for k, v in dict(globals()).items() if k.startswith("b_") and callable(v)}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
_PLACE = re.compile(r"\{\{\s*(@?[^}|]+?)\s*(?:\|\s*(\w+)\s*)?\}\}")


def render(text: str, name: str) -> str:
    def sub(m):
        expr, spec = m.group(1).strip(), m.group(2)
        try:
            if expr.startswith("@"):
                return BLOCKS[expr[1:]]()
            return fmt(lookup(expr), spec or "raw")
        except Exception as exc:  # fail loudly: a broken placeholder must never ship
            raise RuntimeError(f"{name}: cannot render '{{{{ {expr} | {spec} }}}}': {exc!r}") from exc
    return _PLACE.sub(sub, text)


def report_tables() -> str:
    s = []
    s.append("# RTO Shield AI - Report Tables\n")
    s.append("> Generated by `src/11_render_docs.py` from `outputs/final_metrics.json` and `models/thresholds.json`. "
             f"Do not edit by hand. Metrics generated {M['meta']['generated']}.\n")
    s.append("> **Metric conventions.** " + b_ranking_note() + "\n")
    s.append("## Table 1 - Model Comparison (Validation, threshold = 0.5)\n\n" + b_model_comparison_val())
    s.append("## Table 2 - Model Comparison (Test, threshold = 0.5)\n\n" + b_model_comparison_test())
    s.append("## Table 3 - Final Model Test Metrics\n\n" + b_final_metrics() +
             f"\n\nMulti-seed ({M['multi_seed']['n_seeds']} seeds): ROC-AUC {M['multi_seed']['roc_auc_mean']:.4f} +/- "
             f"{M['multi_seed']['roc_auc_std']:.4f}, PR-AUC {M['multi_seed']['pr_auc_mean']:.4f} +/- {M['multi_seed']['pr_auc_std']:.4f}. "
             f"Delta ROC-AUC vs baseline (bootstrap 95% CI): {fmt(M['test']['delta_roc_auc_ci'], 'ci4')}.")
    s.append("## Table 4 - Calibration (final tuned model)\n\n" + b_calibration_table())
    s.append("## Table 5 - Policies on TEST\n\n" + b_policy_table())
    s.append("## Table 6 - Threshold Stability on TEST (threshold +/-0.01, +/-0.02)\n\n" + b_stability_table() +
             "\n\nFlagged-share swing: " + b_stability_summary() + ".")
    s.append("## Table 7 - Risk Tiers (cutoffs = midpoints of low-density score gaps, chosen on validation)\n\n"
             + b_tier_table() + "\n\n" + b_tier_descriptions())
    s.append("## Table 8 - Per-Shipping-Mode Performance (TEST, raw-score ranking)\n\n" + b_per_mode() +
             "\n\n" + b_per_mode_note())
    s.append("## Table 9 - Feature Ablation (TEST)\n\n" + b_ablation() + "\n\n" + b_ablation_verdict())
    s.append("## Table 10 - Rolling-Origin Backtest\n\n" + b_backtest() +
             f"\n\nMean ROC-AUC {M['backtest']['roc_auc_mean']:.4f} +/- {M['backtest']['roc_auc_std']:.4f}; "
             f"mean PR-AUC {M['backtest']['pr_auc_mean']:.4f} +/- {M['backtest']['pr_auc_std']:.4f}; "
             f"mean PR-AUC lift over prevalence {M['backtest']['pr_auc_lift_mean']:.3f} +/- {M['backtest']['pr_auc_lift_std']:.3f}.")
    s.append("## Table 11 - Population Stability Index (train vs test)\n\n" + b_psi())
    s.append("## Table 12 - Hyperparameters (tuned)\n\n" + b_hyperparams() +
             f"\n\nValidation ROC-AUC: default {M['model']['val_roc_auc_default']:.4f} vs tuned {M['model']['val_roc_auc_tuned']:.4f} "
             f"(CV ROC-AUC {M['model']['cv_roc_auc']:.4f}).")
    s.append("## Table 13 - SHAP Global Importance (500-row test sample)\n\n" + b_shap_top() + "\n\n" + b_shap_grouped())
    s.append("## Table 14 - Sanity & Verification\n\n" + b_label_shuffle() + "\n\n" + b_verification_segment())
    figs = sorted(p.name for p in OUTPUTS_DIR.glob("*.png"))
    s.append("## Figure File Index\n\n" + "\n".join(f"- `outputs/{f}`" for f in figs))
    return "\n\n".join(s) + "\n"


def main() -> None:
    (ROOT / "docs").mkdir(exist_ok=True)
    for tmpl_name, dest in [("README.md.tmpl", ROOT / "README.md"), ("MODEL_CARD.md.tmpl", ROOT / "MODEL_CARD.md"),
                            ("VIVA_CHEATSHEET.md.tmpl", ROOT / "docs" / "VIVA_CHEATSHEET.md")]:
        text = (TEMPLATES / tmpl_name).read_text(encoding="utf-8")
        dest.write_text(render(text, tmpl_name), encoding="utf-8")
        print(f"  rendered {dest.name}")
    (OUTPUTS_DIR / "report_tables.md").write_text(report_tables(), encoding="utf-8")
    print("  rendered outputs/report_tables.md")


if __name__ == "__main__":
    main()
