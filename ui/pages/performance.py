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

    st.caption(C.payment_caption(fm))

    tab_comp, tab_curves, tab_cal, tab_lift, tab_tiers = st.tabs(
        ["📊 Model comparison", "📈 Curves", "🎯 Calibration", "🚀 Lift & gain", "🏷️ Risk tiers & policies"])

    with tab_comp, st.container(border=True):
        C.card_title("Model comparison")
        comp = pd.DataFrame(fm["model_comparison"])
        fm_models = comp["model"].tolist()                      # raw model keys, used to look up the significance rows
        comp["model"] = comp["model"].map(data.MODEL_LABELS)
        C.show(charts.model_comparison_bar(comp[comp["split"] == "test"], t["roc_auc"]), key="perf_models")
        st.markdown("**Validation and test metrics at a 0.5 threshold**")
        sig = {r["model"]: r for r in fm["model_comparison_significance"]["rows"]}
        comp["delta_ci"] = [(f"{sig[m]['delta']:+.4f} [{sig[m]['ci_lo']:+.4f}, {sig[m]['ci_hi']:+.4f}]" if (s == "test" and m in sig) else "–")
                            for m, s in zip(fm_models, comp["split"])]
        comp["dist"] = [(("Yes" if sig[m]["distinguishable"] else "No") if (s == "test" and m in sig) else "–")
                        for m, s in zip(fm_models, comp["split"])]
        C.styled_table(comp[COMPARISON_COLS + ["delta_ci", "dist"]],
                       rename={"model": "Model", "split": "Split", "delta_ci": "Delta vs final (95% CI)", "dist": "Distinguishable?"})
        st.caption("Delta = candidate test ROC-AUC minus the final tuned model's, paired bootstrap "
                   f"({fm['model_comparison_significance']['n_boot']:,} resamples, shared indices); distinguishable = the 95% CI excludes 0. "
                   "Validation rows have no test-set delta.")
        st.markdown(fm["model_comparison_significance"]["summary"])

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
        ef = fm["calibration_ef"]
        C.show(charts.reliability_ef(ef), key="perf_cal")
        st.caption("ECE is a size-weighted average, so a few small bins can sit well off the diagonal. Bins whose points share almost the same x "
                   "belong to one isotonic plateau (many orders receive the same calibrated probability).")
        C.styled_table(pd.DataFrame({
            "metric": ["Brier score", "Brier skill score (vs base rate)", "ECE, 10 equal-width bins", "ECE, 10 equal-frequency bins"],
            "raw": [t["brier_raw"], ef["brier_skill_score"]["raw"], ef["ece"]["raw_equal_width"], ef["ece"]["raw_equal_frequency"]],
            "calibrated": [t["brier_calibrated"], ef["brier_skill_score"]["calibrated"], ef["ece"]["calibrated_equal_width"],
                           ef["ece"]["calibrated_equal_frequency"]]}),
            rename={"raw": "Raw Scores", "calibrated": "Isotonic-Calibrated"}, formats={"raw": "{:.4f}", "calibrated": "{:.4f}"})
        st.caption(f"Brier skill score = 1 - Brier / (p (1 - p)) with p = test prevalence ({C.fmt_pct(ef['prevalence'], clip=False)}); "
                   "0 = no better than always predicting the base rate. " + ef["binning_note"].capitalize() + ".")
        st.markdown("**Calibrated-score bins (n per bin, bootstrap 95% CI of the observed rate)**")
        C.styled_table(pd.DataFrame([{"bin": b["bin"], "n": b["n"], "mean_predicted": b["mean_predicted"], "observed": b["observed"],
                                      "ci": f"[{b['ci_lo']:.1%}, {b['ci_hi']:.1%}]"} for b in ef["calibrated"]]),
                       rename={"bin": "Bin", "n": "Orders", "mean_predicted": "Mean Predicted", "observed": "Observed Late Rate", "ci": "95% CI"},
                       formats={"n": "{:,}", "mean_predicted": lambda v: C.fmt_pct(v, clip=False), "observed": lambda v: C.fmt_pct(v, clip=False)})

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
                             "wilson": f"[{s['late_rate_ci_lo']:.1%}, {s['late_rate_ci_hi']:.1%}]",
                             "late_captured": s["late_captured"], "handling": fm["tiers"]["descriptions"][k]["action"]})
            pc = lambda v: C.fmt_pct(v, clip=False)
            C.styled_table(pd.DataFrame(rows), rename={"score_range": "Score Range", "handling": "Suggested Handling",
                                                       "wilson": "Late Rate 95% CI (Wilson)"},
                           formats={"share": pc, "late_rate": pc, "late_captured": pc, "orders": "{:,}"})
            med = fm["tiers"]["test"]["Medium"]
            st.caption(f"Wilson 95% intervals: the Medium tier has only {med['orders']:,} orders, so its interval is the widest "
                       f"({med['late_rate_ci_hi'] - med['late_rate_ci_lo']:.1%} wide).")
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
            rb = fm["rule_baseline"]
            C.card_title("Plain-rule baseline")
            st.markdown(f"**Rule (written from TRAIN-period findings only):** {rb['definition']}")
            ev = rb["train_evidence"]
            lr = ev["late_rate_by_mode"]
            st.caption("Train-period evidence: late rate " + ", ".join(f"{m} {C.fmt_pct(v['late_rate'], clip=False)}" for m, v in lr.items())
                       + f"; Same Day before 12:00 {C.fmt_pct(ev['same_day_before_noon_late_rate'], clip=False)}, "
                       f"from 12:00 {C.fmt_pct(ev['same_day_from_noon_late_rate'], clip=False)}. Nothing was tuned on the test set.")
            pc = lambda v: C.fmt_pct(v, clip=False)
            lines = [("Plain rule", rb["rule"]), (f"Model top-{rb['model_top_k']['k']:,} (same flagged share)", rb["model_top_k"]),
                     ("Model High tier", rb["high_tier"])]
            C.styled_table(pd.DataFrame([{"policy": n, "flagged_share": m["flagged_share"], "flagged_orders": m["flagged_orders"],
                                          "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "mcc": m["mcc"]} for n, m in lines]),
                           rename={"policy": "Policy", "flagged_share": "Flagged Share", "flagged_orders": "Flagged Orders"},
                           formats={"flagged_share": pc, "flagged_orders": "{:,}", "precision": pc, "recall": pc, "f1": "{:.3f}", "mcc": "{:.3f}"})
            eq = rb["equal_share_comparison"]
            lo, hi = eq["delta_precision_ci"]
            st.markdown(f"At the same flagged share ({pc(eq['flagged_share'])}, {eq['k']:,} orders) the model's precision minus the rule's is "
                        f"**{eq['delta_precision']:+.3%}** (paired-bootstrap 95% CI [{lo:+.3%}, {hi:+.3%}]); "
                        f"recall {pc(eq['recall_a'])} vs {pc(eq['recall_b'])}.")
            st.caption(f"Overlap: {C.fmt_pct(rb['top_k_overlap_with_rule'], clip=False)} of the model's top-{eq['k']:,} orders are also flagged by the rule"
                       + (" (the two flag exactly the same orders, so the paired difference is 0 by construction)."
                          if rb["top_k_overlap_with_rule"] == 1.0 else "."))
            C.callout("info" if rb["matches_model"] else "note", "What the model adds" if rb["matches_model"] else "Model vs plain rule", rb["verdict"])
        with st.container(border=True):
            C.card_title("Who gets flagged: High tier by shipping mode (test set)")
            pmf = pd.DataFrame(fm["per_mode_flagging"])
            tot = {"mode": "All modes", "orders": pmf["orders"].sum(), "flagged": pmf["flagged"].sum(), "late": pmf["late"].sum(),
                   "caught": pmf["caught"].sum(), "missed": pmf["missed"].sum()}
            tot["recall"] = tot["caught"] / tot["late"] if tot["late"] else None
            tot["share_of_missed"] = pmf["share_of_missed"].sum()
            pmf = pd.concat([pmf, pd.DataFrame([tot])], ignore_index=True)
            pc = lambda v: C.fmt_pct(v, clip=False)
            C.styled_table(pmf[["mode", "orders", "flagged", "late", "caught", "recall", "share_of_missed"]],
                           rename={"mode": "Shipping Mode", "orders": "Test Orders", "flagged": "Flagged", "late": "Late Orders",
                                   "caught": "Late Orders Caught", "recall": "Recall Within Mode", "share_of_missed": "Share of All Missed Late Orders"},
                           formats={"orders": "{:,}", "flagged": "{:,}", "late": "{:,}", "caught": "{:,}", "recall": pc, "share_of_missed": pc})
            rows = fm["per_mode_flagging"]
            worst = max(rows, key=lambda r: r["share_of_missed"])
            overall = next(p for p in fm["policies"] if p["policy"].startswith("High tier"))["recall"]
            if worst["flagged"] == 0:
                lead = f"{worst['mode']} is never flagged"
            else:
                lead = f"{worst['mode']} has only {C.fmt_pct(worst['flagged'] / worst['orders'], clip=False)} of its orders flagged"
            st.markdown(f"{lead} and holds **{C.fmt_pct(worst['share_of_missed'], clip=False)}** of the missed late orders, so the "
                        f"overall recall of {C.fmt_pct(overall, clip=False)} is capped by design.")
        with st.container(border=True):
            C.card_title("Threshold stability")
            st.caption("Each threshold is shifted by +/-0.01 and +/-0.02; thresholds sit in low-density gaps so the flagged share barely moves.")
            C.styled_table(pd.DataFrame(fm["stability"]["rows"]), rename={"flagged_share_pct": "Flagged Share (%)"},
                           formats={"threshold": "{:.4f}", "flagged_share_pct": "{:.2f}", "shift": "{:+.2f}", "flagged_orders": "{:,}"})
            sw = fm["stability"]["swing_pp"]
            st.caption("Maximum change in flagged share: " + "; ".join(f"{k} {v:.2f} pp" for k, v in sw.items())
                       + f" (limit {fm['stability']['limit_pp']:.0f} pp).")
