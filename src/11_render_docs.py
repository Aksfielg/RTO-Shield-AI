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


def _pyproject() -> dict:
    """Facts from pyproject.toml the documents quote (so the supported Python range has one source)."""
    import tomllib
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    minimum = meta["requires-python"].replace(">=", "").strip()
    return {"requires_python": meta["requires-python"], "python_min": minimum, "version": meta["version"]}


P = _pyproject()
F = json.loads((ROOT / "docs" / "figure_captions.json").read_text(encoding="utf-8"))      # written by 11b_figure_captions.py


def lookup(path: str):
    root, _, rest = path.partition(".")
    cur = {"M": M, "T": T, "P": P, "F": F}[root]
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
    if spec == "s4":
        return f"{value:+.4f}"
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
    sig = {r["model"]: r for r in M["model_comparison_significance"]["rows"]}
    rows = []
    for m in MODEL_LABELS:
        v, t = by[(m, "val")], by[(m, "test")]
        s = sig.get(m)
        rows.append([MODEL_LABELS[m], f"{v['roc_auc']:.4f}", f"{t['roc_auc']:.4f}", f"{t['pr_auc']:.4f}",
                     f"{t['f1']:.3f}", f"{t['brier']:.4f}",
                     "n/a" if s is None else f"{s['delta']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]",
                     "n/a" if s is None else ("Yes" if s["distinguishable"] else "No")])
    return md_table(["Model", "Val ROC-AUC", "Test ROC-AUC", "Test PR-AUC", "Test F1", "Test Brier",
                     "Delta vs final (95% CI)", "Distinguishable?"], rows)


def b_test_table() -> str:
    T = M["tests"]
    rows = [[f"`{r['file']}`", f"{r['tests']:,}", r["description"]] for r in T["by_file"]]
    rows.append(["**Total**", f"**{T['total']:,}**", f"{T['n_files']} files, counted with `{T['source']}`"])
    return md_table(["Test file", "Tests", "What it protects"], rows)


def b_model_significance() -> str:
    S = M["model_comparison_significance"]
    rows = [[r["label"], f"{r['roc_auc']:.4f}", f"{r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]",
             "Yes" if r["distinguishable"] else "No"] for r in S["rows"]]
    return (md_table(["Model", "Test ROC-AUC", "Delta vs final (95% CI)", "Distinguishable?"], rows)
            + f"\n\nDelta = {S['delta_definition']} ({S['n_boot']:,} resamples). " + S["summary"])


def b_rule_baseline() -> str:
    rb = M["rule_baseline"]
    eq = rb["equal_share_comparison"]
    rows = []
    for name, m in [("Plain rule", rb["rule"]), (f"Model top-{rb['model_top_k']['k']:,} by raw score (same flagged share)", rb["model_top_k"]),
                    ("Model High tier", rb["high_tier"])]:
        rows.append([name, f"{m['flagged_share'] * 100:.1f}% ({m['flagged_orders']:,})", f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                     f"{m['f1']:.4f}", f"{m['mcc']:.4f}"])
    ev = rb["train_evidence"]
    modes = ", ".join(f"{k} {v['late_rate'] * 100:.1f}%" for k, v in ev["late_rate_by_mode"].items())
    return (f"**Rule (from TRAIN-period findings only, nothing tuned on test):** {rb['definition']} Train late rates: {modes}; Same Day "
            f"before 12:00 {ev['same_day_before_noon_late_rate'] * 100:.1f}%, from 12:00 {ev['same_day_from_noon_late_rate'] * 100:.1f}%.\n\n"
            + md_table(["Policy (TEST)", "Flagged", "Precision", "Recall", "F1", "MCC"], rows)
            + f"\n\nAt the same flagged share ({eq['flagged_share'] * 100:.1f}%, {eq['k']:,} orders) the model's precision minus the rule's is "
              f"{eq['delta_precision']:+.4f} (paired-bootstrap 95% CI [{eq['delta_precision_ci'][0]:+.4f}, {eq['delta_precision_ci'][1]:+.4f}]); "
              f"recall {eq['recall_a']:.4f} vs {eq['recall_b']:.4f}. {rb['top_k_overlap_with_rule'] * 100:.1f}% of the model's top-{eq['k']:,} orders are also "
              "flagged by the rule" + (" (the two flag exactly the same orders, so the paired difference is 0 by construction). "
                                       if rb["top_k_overlap_with_rule"] == 1.0 else ". ") + rb["verdict"])


def b_per_mode_calibration() -> str:
    P = M["per_mode_calibration"]
    rows = [[r["mode"], f"{r['orders']:,}", f"{r['mean_predicted'] * 100:.1f}%", f"{r['observed'] * 100:.1f}%", f"{r['gap_pp']:+.2f} pp",
             f"{r['ece']:.4f}", "above 5 pp" if r["flagged"] else "within tolerance"] for r in P["rows"]]
    flagged = [r["mode"] for r in P["rows"] if r["flagged"]]
    return (md_table(["Shipping mode", "Test orders", "Mean predicted", "Observed late rate", "Gap (predicted - observed)",
                      "ECE (within mode)", f"Flag (gap > {P['flag_gap_pp']:.0f} pp)"], rows)
            + "\n\n" + (", ".join(flagged) + f" exceed{'s' if len(flagged) == 1 else ''} a {P['flag_gap_pp']:.0f} pp gap. " if flagged
                         else f"No mode exceeds a {P['flag_gap_pp']:.0f} pp gap. ") + "Reported only; no per-mode recalibration.")


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
    t, ef = M["test"], M["calibration_ef"]
    return md_table(["Metric (TEST, final tuned model)", "Before calibration (raw)", "After isotonic calibration"], [
        ["Brier score", f"{t['brier_raw']:.4f}", f"{t['brier_calibrated']:.4f}"],
        ["Brier skill score (1 - Brier / (p(1-p)), p = test prevalence)", f"{ef['brier_skill_score']['raw']:.4f}",
         f"{ef['brier_skill_score']['calibrated']:.4f}"],
        ["ECE, 10 equal-width bins", f"{ef['ece']['raw_equal_width']:.4f}", f"{ef['ece']['calibrated_equal_width']:.4f}"],
        ["ECE, 10 equal-frequency bins", f"{ef['ece']['raw_equal_frequency']:.4f}", f"{ef['ece']['calibrated_equal_frequency']:.4f}"],
    ]) + ("\n\nECE is a size-weighted average, so a few small bins can sit well off the diagonal. "
          + ef["binning_note"].capitalize() + ".")


def b_tier_table() -> str:
    c = M["tiers"]["cutoffs"]
    rng = {"Low": f"score < {c['low_max']:.4f}", "Medium": f"{c['low_max']:.4f} <= score < {c['high_min']:.4f}",
           "High": f"score >= {c['high_min']:.4f}"}
    rows = []
    for k in ["Low", "Medium", "High"]:
        s, d = M["tiers"]["test"][k], M["tiers"]["descriptions"][k]
        rows.append([k, rng[k], f"{s['orders']:,}", f"{s['share'] * 100:.1f}%", f"{s['late_rate'] * 100:.1f}%",
                     f"[{s['late_rate_ci_lo'] * 100:.1f}%, {s['late_rate_ci_hi'] * 100:.1f}%]",
                     f"{s['late_captured'] * 100:.1f}%", d["action"]])
    return md_table(["Tier", "Score range", "Test orders", "Share", "Late rate", "Late rate 95% CI (Wilson)", "Late captured",
                     "Suggested handling"], rows)


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
    s.append("## Table 8b - Is any candidate model distinguishable from the final model? (TEST ROC-AUC)\n\n" + b_model_significance())
    s.append("## Table 8c - Plain-rule baseline (TEST)\n\n" + b_rule_baseline())
    s.append("## Table 8d - Calibration within shipping mode (TEST)\n\n" + b_per_mode_calibration())
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


# ---------------------------------------------------------------------------
# outputs/report_numbers.md: every number a report may quote, with the JSON path it comes from
# ---------------------------------------------------------------------------
def _path_label(path: str) -> str:
    root, _, rest = path.partition(".")
    return f"`{'thresholds.json' if root == 'T' else 'final_metrics.json'}` → `{rest}`"


def _num_table(rows: list[tuple[str, str, str]]) -> str:
    """rows: (label, 'M.json.path', format) -> table with the formatted value and the JSON path of that value."""
    return md_table(["Quantity", "Value", "JSON path"], [[lab, fmt(lookup(p), f), _path_label(p)] for lab, p, f in rows])


def report_numbers() -> str:
    s: list[str] = []
    s.append("# RTO Shield AI - Report Numbers\n")
    s.append("> Generated by `src/11_render_docs.py` from `outputs/final_metrics.json` and `models/thresholds.json`. Do not edit by hand. "
             f"Metrics generated {M['meta']['generated']}. Every value is followed by the JSON path it was read from; ranking metrics use RAW "
             "scores, decision and calibration metrics use calibrated scores.\n")

    s.append("## 1. Test ROC-AUC\n\n" + _num_table([
        ("Test ROC-AUC (3 decimals)", "M.test.roc_auc", "f3"), ("Test ROC-AUC (4 decimals)", "M.test.roc_auc", "f4"),
        ("95% bootstrap CI", "M.test.roc_auc_ci", "ci4"),
        ("10-seed mean ROC-AUC", "M.multi_seed.roc_auc_mean", "f4"), ("10-seed std ROC-AUC", "M.multi_seed.roc_auc_std", "f4"),
        ("Number of seeds", "M.multi_seed.n_seeds", "int"), ("Test orders", "M.test.n", "int"),
        ("Delta ROC-AUC, final minus baseline LR (paired bootstrap 95% CI)", "M.test.delta_roc_auc_ci", "ci4")]))

    s.append("## 2. PR-AUC\n\n" + _num_table([
        ("Test PR-AUC", "M.test.pr_auc", "f4"), ("95% bootstrap CI", "M.test.pr_auc_ci", "ci4"),
        ("Prevalence (late rate)", "M.test.base_rate", "pct1"), ("PR-AUC lift over prevalence", "M.test.pr_auc_lift", "f2"),
        ("10-seed mean PR-AUC", "M.multi_seed.pr_auc_mean", "f4"), ("10-seed std PR-AUC", "M.multi_seed.pr_auc_std", "f4")]))

    s.append("## 3. Calibration: ECE, Brier and Brier skill score\n\n" + _num_table([
        ("ECE raw (10 equal-width bins)", "M.test.ece_raw", "f4"), ("ECE calibrated (10 equal-width bins)", "M.test.ece_calibrated", "f4"),
        ("ECE raw (10 equal-frequency bins)", "M.calibration_ef.ece.raw_equal_frequency", "f4"),
        ("ECE calibrated (10 equal-frequency bins)", "M.calibration_ef.ece.calibrated_equal_frequency", "f4"),
        ("Brier raw", "M.test.brier_raw", "f4"), ("Brier calibrated", "M.test.brier_calibrated", "f4"),
        ("Brier skill score raw (1 - Brier / (p(1-p)))", "M.calibration_ef.brier_skill_score.raw", "f4"),
        ("Brier skill score calibrated", "M.calibration_ef.brier_skill_score.calibrated", "f4"),
        ("Prevalence p used in the skill score", "M.calibration_ef.prevalence", "pct1")]))

    c = M["tiers"]["cutoffs"]
    rows = []
    for k in ["Low", "Medium", "High"]:
        t = M["tiers"]["test"][k]
        rows.append([k, f"{t['orders']:,}", f"{t['share'] * 100:.1f}%", f"{t['late_rate'] * 100:.1f}%",
                     f"[{t['late_rate_ci_lo'] * 100:.1f}%, {t['late_rate_ci_hi'] * 100:.1f}%]", f"{t['late_captured'] * 100:.1f}%",
                     f"`tiers.test.{k}`"])
    s.append(f"## 4. Risk tiers with Wilson 95% intervals (cutoffs {T['tier_cutoffs']['low_max']:.4f} and {T['tier_cutoffs']['high_min']:.4f})\n\n"
             + md_table(["Tier", "Test orders", "Share", "Late rate", "Late rate 95% CI (Wilson)", "Late captured", "JSON path (final_metrics.json)"], rows)
             + "\n\nCutoffs: `thresholds.json` → `tier_cutoffs.low_max`, `tier_cutoffs.high_min`.")

    rows = []
    for r in M["per_mode"]:
        ci = "n/a" if r["roc_auc"] is None else f"[{r['roc_auc_ci_lo']:.4f}, {r['roc_auc_ci_hi']:.4f}]"
        rows.append([r["mode"], f"{r['orders']:,}", f"{r['share'] * 100:.1f}%", f"{r['prevalence'] * 100:.1f}%", fmt(r["roc_auc"], "f4"), ci,
                     fmt(r["pr_auc"], "f4"), r["flag"], f"{r['calibration_error'] * 100:+.2f} pp", f"`per_mode[mode={r['mode']}]`"])
    s.append("## 5. Per shipping mode (TEST)\n\n" + md_table(
        ["Mode", "Orders", "Share", "Prevalence", "ROC-AUC", "ROC-AUC 95% CI", "PR-AUC", "Signal", "Calibration gap", "JSON path (final_metrics.json)"], rows)
        + "\n\n" + md_table(["Calibration within mode", "Mean predicted", "Observed", "Gap", "ECE", "JSON path"], [
            [r["mode"], f"{r['mean_predicted'] * 100:.1f}%", f"{r['observed'] * 100:.1f}%", f"{r['gap_pp']:+.2f} pp", f"{r['ece']:.4f}",
             f"`per_mode_calibration.rows[mode={r['mode']}]`"] for r in M["per_mode_calibration"]["rows"]]))

    rows = [[r["mode"], f"{r['orders']:,}", f"{r['flagged']:,}", f"{r['late']:,}", f"{r['caught']:,}", fmt(r["recall"], "pct1"),
             fmt(r["share_of_missed"], "pct1"), f"`per_mode_flagging[mode={r['mode']}]`"] for r in M["per_mode_flagging"]]
    s.append("## 6. Per-mode flagging (High tier, TEST)\n\n" + md_table(
        ["Mode", "Test orders", "Flagged", "Late orders", "Late caught", "Recall within mode", "Share of all missed late orders",
         "JSON path (final_metrics.json)"], rows))

    sig = {r["model"]: r for r in M["model_comparison_significance"]["rows"]}
    by = {(r["model"], r["split"]): r for r in M["model_comparison"]}
    rows = []
    for m in MODEL_LABELS:
        v, t_ = by[(m, "val")], by[(m, "test")]
        sg = sig.get(m)
        rows.append([MODEL_LABELS[m], f"{v['roc_auc']:.4f}", f"{t_['roc_auc']:.4f}",
                     "n/a" if sg is None else f"{sg['delta']:+.4f} [{sg['ci_lo']:+.4f}, {sg['ci_hi']:+.4f}]",
                     "n/a" if sg is None else ("Yes" if sg["distinguishable"] else "No"),
                     f"`model_comparison[model={m}]`, `model_comparison_significance.rows[model={m}]`"])
    s.append("## 7. Model comparison with paired deltas vs the final model\n\n" + md_table(
        ["Model", "Val ROC-AUC", "Test ROC-AUC", "Delta vs final (95% CI)", "Distinguishable?", "JSON path (final_metrics.json)"], rows)
        + f"\n\n{M['model_comparison_significance']['summary']} (`model_comparison_significance.summary`)")

    rows = []
    for a in M["ablation"]:
        d = "-" if a["label"] == "All features" else f"{a['delta_roc_vs_all']:+.4f} [{a['delta_ci_lo']:+.4f}, {a['delta_ci_hi']:+.4f}]" + (" *" if a["delta_significant"] else "")
        rows.append([a["label"], str(a["n_features"]), f"{a['roc_auc']:.4f}", f"[{a['roc_auc_ci_lo']:.4f}, {a['roc_auc_ci_hi']:.4f}]", d,
                     f"`ablation[label={a['label']}]`"])
    ta = M["type_audit"]
    ab = md_table(["Feature subset", "Features", "ROC-AUC", "ROC-AUC 95% CI", "Delta vs all features (95% CI)", "JSON path (final_metrics.json)"], rows)
    hist = M["ablation"][[a["label"] for a in M["ablation"]].index("No history features")]
    s.append("## 8. Feature ablation, history delta and the payment-type audit\n\n" + ab
             + f"\n\n`*` = paired 95% CI excludes 0. History features: delta {hist['delta_roc_vs_all']:+.4f} [{hist['delta_ci_lo']:+.4f}, {hist['delta_ci_hi']:+.4f}] "
               f"(`ablation[label=No history features]`); verdict: {M['ablation_summary']['history_verdict']} (`ablation_summary.history_verdict`)\n\n"
             + _num_table([
                 ("Payment features: ROC-AUC with / all orders", "M.type_audit.all_orders.model_a.roc_auc", "f4"),
                 ("Payment features: ROC-AUC without / all orders", "M.type_audit.all_orders.model_b.roc_auc", "f4"),
                 ("Delta (with minus without), all orders", "M.type_audit.all_orders.delta_roc_auc.point", "s4"),
                 ("  95% CI", "M.type_audit.all_orders.delta_roc_auc.ci", "ci4"),
                 ("ROC-AUC with / cleaned orders (cancelled/fraud removed)", "M.type_audit.cleaned_orders.model_a.roc_auc", "f4"),
                 ("ROC-AUC without / cleaned orders", "M.type_audit.cleaned_orders.model_b.roc_auc", "f4"),
                 ("Delta (with minus without), cleaned orders", "M.type_audit.cleaned_orders.delta_roc_auc.point", "s4"),
                 ("  95% CI", "M.type_audit.cleaned_orders.delta_roc_auc.ci", "ci4"),
                 ("Cleaned orders (n)", "M.type_audit.cleaned_orders.n", "int"),
                 ("Equal-coverage precision, model with payment features (all orders)", "M.type_audit.equal_coverage.all_orders.model_a.precision", "pct1"),
                 ("Equal-coverage precision, model without payment features (all orders)", "M.type_audit.equal_coverage.all_orders.model_b.precision", "pct1")]))

    rb = M["rule_baseline"]
    rows = [[n, f"{m['flagged_share'] * 100:.1f}% ({m['flagged_orders']:,})", f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
             f"{m['mcc']:.4f}", f"`rule_baseline.{key}`"] for n, m, key in
            [("Plain rule", rb["rule"], "rule"), (f"Model top-{rb['model_top_k']['k']:,} (same flagged share)", rb["model_top_k"], "model_top_k"),
             ("Model High tier", rb["high_tier"], "high_tier")]]
    eq = rb["equal_share_comparison"]
    s.append("## 9. Plain-rule baseline (TEST)\n\n" + md_table(["Policy", "Flagged", "Precision", "Recall", "F1", "MCC", "JSON path (final_metrics.json)"], rows)
             + f"\n\nRule: {rb['definition']} (`rule_baseline.definition`). Equal-share precision difference (model - rule): {eq['delta_precision']:+.4f} "
               f"[{eq['delta_precision_ci'][0]:+.4f}, {eq['delta_precision_ci'][1]:+.4f}] (`rule_baseline.equal_share_comparison`); overlap of the model's top-k with the "
               f"rule: {rb['top_k_overlap_with_rule'] * 100:.1f}% (`rule_baseline.top_k_overlap_with_rule`). {rb['verdict']}")

    bt = M["backtest"]
    rows = [[str(r["fold"]), f"{r['train_n']:,}", f"{r['test_n']:,}", f"{r['test_start']} to {r['test_end']}", f"{r['roc_auc']:.4f}", f"{r['pr_auc']:.4f}",
             f"`backtest.rows[fold={r['fold']}]`"] for r in bt["rows"]]
    s.append("## 10. Rolling-origin backtest\n\n" + md_table(["Fold", "Train n", "Test n", "Test window", "ROC-AUC", "PR-AUC", "JSON path (final_metrics.json)"], rows)
             + f"\n\nMean ROC-AUC {bt['roc_auc_mean']:.4f} ± {bt['roc_auc_std']:.4f} (`backtest.roc_auc_mean`, `backtest.roc_auc_std`).")

    rows = [[r["feature"], f"{r['psi']:.4f}", r["status"], f"`psi.rows[feature={r['feature']}]`"] for r in M["psi"]["rows"]]
    s.append("## 11. Population Stability Index (train vs test)\n\n" + md_table(["Feature", "PSI", "Status", "JSON path (final_metrics.json)"], rows)
             + f"\n\nMaximum PSI {M['psi']['max_psi']:.4f} (`psi.max_psi`).")

    s.append("## 12. Label-shuffle check\n\n" + _num_table([
        ("Validation ROC-AUC with shuffled labels", "M.label_shuffle.val_roc_auc", "f4"), ("Test ROC-AUC with shuffled labels", "M.label_shuffle.test_roc_auc", "f4"),
        ("Status", "M.label_shuffle.status", "raw")]))

    rows = [[nm, f"{M['splits'][k]['orders']:,}", M["splits"][k]["date_min"], M["splits"][k]["date_max"], f"{M['splits'][k]['positive_rate'] * 100:.1f}%",
             f"`splits.{k}`"] for k, nm in [("train", "Train"), ("val", "Validation"), ("test", "Test")]]
    s.append("## 13. Split sizes and dates\n\n" + md_table(["Split", "Orders", "From", "To", "Late rate", "JSON path (final_metrics.json)"], rows))

    s.append("## 14. Test suite and run time\n\n" + _num_table([
        ("Automated tests (pytest --collect-only)", "M.tests.total", "int"), ("Test files", "M.tests.n_files", "int"),
        ("Full pipeline run time (minutes)", "M.run.total_minutes", "f1"), ("Python version of that run", "M.run.python", "raw"),
        ("Platform", "M.run.platform", "raw")]))
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
    (OUTPUTS_DIR / "report_numbers.md").write_text(report_numbers(), encoding="utf-8")
    print("  rendered outputs/report_numbers.md")


if __name__ == "__main__":
    main()
