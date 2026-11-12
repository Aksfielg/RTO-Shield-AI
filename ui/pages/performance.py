"""Performance: headline KPIs, model comparison, curves, calibration, lift & gain, tiers and policies."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import charts
from ui import components as C
from ui import data

COMPARISON_COLS = ["model", "split", "roc_auc", "pr_auc", "brier", "precision", "recall", "f1", "accuracy",
                   "balanced_accuracy", "specificity", "mcc", "precision_at_top10pct", "ece"]


def render() -> None:
    fm, thr = data.fm(), data.thr()
    t = fm["test"]
    high = next(p for p in fm["policies"] if p["policy"].startswith("High tier"))
    C.page_header("Model Performance", "Empirical validation and test diagnostics across chronological holdout partitions.")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        C.kpi_card("ROC-AUC (95% CI)", f"{t['roc_auc']:.3f}", f"[{t['roc_auc_ci'][0]:.3f}, {t['roc_auc_ci'][1]:.3f}]")
    with k2:
        C.kpi_card("PR-AUC vs prevalence", f"{t['pr_auc']:.3f}", f"prevalence {C.fmt_pct(t['base_rate'], clip=False)}; lift {t['pr_auc_lift']:.2f}x")
    with k3:
        C.kpi_card("High-tier precision", C.fmt_pct(high["precision"], clip=False),
                   f"coverage: flags {C.fmt_pct(high['flagged_share'], clip=False)}, catches {C.fmt_pct(high['recall'], clip=False)} of late")
    with k4:
        C.kpi_card("ECE (final model)", f"{t['ece_calibrated']:.3f}", f"{t['ece_raw']:.3f} before isotonic calibration")

    tab_comp, tab_curves, tab_cal, tab_lift, tab_tiers = st.tabs(
        ["📊 Model comparison", "📈 Curves", "🎯 Calibration", "🚀 Lift & gain", "🏷️ Risk tiers & policies"])

    with tab_comp, st.container(border=True):
        C.card_title("Model comparison")
        comp = pd.DataFrame(fm["model_comparison"])
        comp["model"] = comp["model"].map(data.MODEL_LABELS)
        C.show(charts.model_comparison_bar(comp[comp["split"] == "test"], t["roc_auc"]), key="perf_models")
        st.markdown("**Validation and test metrics at a 0.5 threshold**")
        C.styled_table(comp[COMPARISON_COLS], rename={"model": "Model", "split": "Split"})

    with tab_curves, st.container(border=True):
        C.card_title("Diagnostic curves (ROC & precision-recall)")
        sc = data.test_scores()
        roc, pr = charts.roc_pr_curves(sc["y"], sc["raw"])
        a, b = st.columns(2)
        with a:
            C.show(roc, key="perf_roc")
        with b:
            C.show(pr, key="perf_pr")
        st.caption("Curves use the RAW (uncalibrated) model scores: isotonic calibration collapses scores into ties, which "
                   "distorts rankings. Calibrated scores are used only for calibration, tiers and decision thresholds.")

    with tab_cal, st.container(border=True):
        C.card_title("Probability calibration (final tuned model)")
        st.write(f"Isotonic regression reduces the test-set Expected Calibration Error from **{t['ece_raw']:.3f}** to "
                 f"**{t['ece_calibrated']:.3f}** (Brier {t['brier_raw']:.3f} to {t['brier_calibrated']:.3f}).")
        C.show(charts.reliability(fm["calibration"]), key="perf_cal")
        C.styled_table(pd.DataFrame({"metric": ["Brier score", "ECE"], "raw": [t["brier_raw"], t["ece_raw"]],
                                     "calibrated": [t["brier_calibrated"], t["ece_calibrated"]]}),
                       rename={"raw": "Raw Scores", "calibrated": "Isotonic-Calibrated"}, formats={"raw": "{:.4f}", "calibrated": "{:.4f}"})

    with tab_lift, st.container(border=True):
        C.card_title("Decile lift & cumulative gain")
        dec, h = pd.DataFrame(fm["deciles"]), fm["decile_highlights"]
        max_lift = 1.0 / t["base_rate"]
        a, b = st.columns(2)
        with a:
            C.show(charts.lift_bar(dec), key="perf_lift")
        with b:
            C.show(charts.gain_curve(dec, t["base_rate"]), key="perf_gain")
        at_ceiling = abs(h["top_decile_lift"] - max_lift) < 0.01
        st.info(f"Top-decile lift {h['top_decile_lift']:.2f}x "
                + (f"equals the maximum possible (1 / prevalence = {max_lift:.2f}x): every order in the top decile was late. "
                   "That is the ceiling for a decile lift, not evidence of a perfect model; the gain curve and the lower deciles are the "
                   "informative part." if at_ceiling else f"vs a maximum possible {max_lift:.2f}x (1 / prevalence)."))
        st.markdown(f"- **Top 3 deciles** capture {h['top3_gain_pct']:.1f}% of all late deliveries while covering {h['top3_volume_pct']:.0f}% of volume.")
        st.caption("Deciles rank orders by raw model score; the dashed line in each chart is the random baseline.")
        C.styled_table(dec, rename={"n": "Orders", "late_in_decile": "Late Orders"})

    with tab_tiers:
        with st.container(border=True):
            C.card_title("Operational risk tiers")
            c = fm["tiers"]["cutoffs"]
            st.caption("Cutoffs are midpoints of low-density gaps between calibrated-score clusters, chosen on validation scores "
                       "and verified on the test set. Only the High tier is flagged.")
            rng = {"Low": f"score < {c['low_max']:.4f}", "Medium": f"{c['low_max']:.4f} <= score < {c['high_min']:.4f}",
                   "High": f"score >= {c['high_min']:.4f}"}
            rows = []
            for k in ["Low", "Medium", "High"]:
                s = fm["tiers"]["test"][k]
                rows.append({"tier": k, "score_range": rng[k], "orders": s["orders"], "share": s["share"], "late_rate": s["late_rate"],
                             "late_captured": s["late_captured"], "handling": fm["tiers"]["descriptions"][k]["action"]})
            pc = lambda v: C.fmt_pct(v, clip=False)
            C.styled_table(pd.DataFrame(rows), rename={"score_range": "Score Range", "handling": "Suggested Handling"},
                           formats={"share": pc, "late_rate": pc, "late_captured": pc, "orders": "{:,}"})
            for k in ["Low", "Medium", "High"]:
                st.markdown(f"- {fm['tiers']['descriptions'][k]['description']}")
            C.show(charts.tier_late_rate_bar(fm["tiers"]["test"], t["base_rate"]), key="perf_tiers")
        with st.container(border=True):
            C.card_title("Policies on the holdout test set")
            pc = lambda v: C.fmt_pct(v, clip=False)
            C.styled_table(pd.DataFrame(fm["policies"]), rename={"flagged_orders": "Flagged Orders", "flagged_share": "Flagged Share"},
                           formats={"flagged_share": pc, "flagged_orders": "{:,}", "threshold": "{:.4f}"})
            fa = next(p for p in fm["policies"] if p["policy"] == "Flag all")
            st.caption(f"Flag-all reaches F1 {fa['f1']:.3f} only because {C.fmt_pct(t['base_rate'], clip=False)} of orders are late, "
                       f"while its MCC is {fa['mcc']:.2f} and balanced accuracy {fa['balanced_accuracy']:.2f}. F1 is shown for reference, "
                       "not as a headline: MCC and balanced accuracy separate real policies from trivial ones.")
        with st.container(border=True):
            C.card_title("Threshold stability")
            st.caption("Each threshold is shifted by +/-0.01 and +/-0.02; thresholds sit in low-density gaps so the flagged share barely moves.")
            C.styled_table(pd.DataFrame(fm["stability"]["rows"]), rename={"flagged_share_pct": "Flagged Share (%)"},
                           formats={"threshold": "{:.4f}", "flagged_share_pct": "{:.2f}", "shift": "{:+.2f}", "flagged_orders": "{:,}"})
            sw = fm["stability"]["swing_pp"]
            st.caption("Maximum change in flagged share: " + "; ".join(f"{k} {v:.2f} pp" for k, v in sw.items())
                       + f" (limit {fm['stability']['limit_pp']:.0f} pp).")
