"""Robustness: per-mode quality, ablations, backtest, drift and sanity checks."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import charts
from ui import components as C
from ui import data


def render() -> None:
    fm = data.fm()
    C.page_header("Robustness & Stress Testing", "Feature ablations, shipping-mode segment performance, temporal stability and sanity checks.")
    pc = lambda v: C.fmt_pct(v, clip=False)

    std, sec, fc, sd = (data.mode_stats(m) for m in ("Standard Class", "Second Class", "First Class", "Same Day"))
    seg = fm["verification"]["segment_models"]
    share = std["share"] + sec["share"]
    C.callout("danger", "Critical limitation: Standard and Second Class",
              f"Standard Class (**{pc(std['share'])}**) and Second Class (**{pc(sec['share'])}**) together are **{pc(share)}** of test volume, "
              f"yet the model cannot rank orders inside them: ROC-AUC **{std['roc_auc']:.3f}** (95% CI {std['roc_auc_ci_lo']:.3f} to "
              f"{std['roc_auc_ci_hi']:.3f}) and **{sec['roc_auc']:.3f}** ({sec['roc_auc_ci_lo']:.3f} to {sec['roc_auc_ci_hi']:.3f}). It ranks "
              f"Same Day ({sd['roc_auc']:.2f}) and First Class ({fc['roc_auc']:.2f}) well. Models trained on each of these modes alone reach "
              f"only {seg['Standard Class']['segment_model_roc_auc']:.3f} and {seg['Second Class']['segment_model_roc_auc']:.3f}: the missing "
              "signal is not in the checkout-time features.")

    tab_mode, tab_abl, tab_bt, tab_psi, tab_sanity = st.tabs(
        ["📦 Per shipping mode", "🧪 Feature ablation", "⏳ Rolling backtest", "🛡️ Drift (PSI)", "✅ Sanity checks"])

    with tab_mode, st.container(border=True):
        C.card_title("Ranking quality across shipping modes")
        C.show(charts.per_mode_ci(fm["per_mode"]), key="rob_mode")
        pm = pd.DataFrame(fm["per_mode"])
        pm["roc_ci"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(pm["roc_auc_ci_lo"], pm["roc_auc_ci_hi"])]
        pm["flag"] = pm["flag"].map(C.signal_chip)
        C.styled_table(pm[["mode", "orders", "share", "prevalence", "roc_auc", "roc_ci", "pr_auc", "pr_auc_lift", "flag",
                           "mean_predicted", "calibration_error"]],
                       rename={"roc_ci": "ROC-AUC 95% CI", "flag": "Signal", "mean_predicted": "Mean Predicted",
                               "calibration_error": "Calibration Error"},
                       formats={"share": pc, "prevalence": pc, "mean_predicted": pc, "orders": "{:,}",
                                "calibration_error": lambda v: f"{v * 100:+.2f} pp", "pr_auc_lift": "{:.2f}x"})
        st.caption("Ranking metrics use raw model scores. Signal = 🟠 limited when the ROC-AUC 95% bootstrap CI lower bound is below 0.55 "
                   "(which also covers any CI that includes 0.5), otherwise 🟢 informative. PR-AUC lift = PR-AUC / prevalence.")

    with tab_abl, st.container(border=True):
        C.card_title("Feature ablation study")
        st.write("ROC-AUC when a feature family is removed (each bar has its 95% bootstrap CI).")
        abl = pd.DataFrame(fm["ablation"])
        C.show(charts.ablation_bar(fm["ablation"]), key="rob_abl")
        C.callout("info", "History features", fm["ablation_summary"]["history_verdict"])
        abl["roc_ci"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(abl["roc_auc_ci_lo"], abl["roc_auc_ci_hi"])]
        abl["delta"] = ["-" if lbl == "All features" else f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (" *" if sg else "")
                        for lbl, d, lo, hi, sg in zip(abl["label"], abl["delta_roc_vs_all"], abl["delta_ci_lo"], abl["delta_ci_hi"],
                                                      abl["delta_significant"])]
        C.styled_table(abl[["label", "n_features", "roc_auc", "roc_ci", "pr_auc", "brier", "threshold", "f1", "precision", "recall",
                            "accuracy", "balanced_accuracy", "specificity", "mcc", "delta"]],
                       rename={"label": "Feature Subset", "roc_ci": "ROC-AUC 95% CI", "threshold": "Own Threshold",
                               "delta": "Delta ROC-AUC vs All"})
        st.caption("Threshold metrics use each ablated model's own validation-chosen max-F1 threshold. "
                   "Delta = ROC-AUC(all) - ROC-AUC(ablated), paired bootstrap; * = 95% CI excludes 0.")

    with tab_bt, st.container(border=True):
        C.card_title("Rolling-origin temporal backtest (5 expanding folds)")
        bt = fm["backtest"]
        C.show(charts.backtest_lines(bt["rows"]), key="rob_bt")
        btd = pd.DataFrame(bt["rows"])[["fold", "train_n", "test_n", "test_start", "test_end", "prevalence", "roc_auc", "pr_auc",
                                        "pr_auc_lift", "accuracy", "balanced_accuracy", "mcc"]]
        C.styled_table(btd, rename={"train_n": "Train N", "test_n": "Test N", "test_start": "Start Date", "test_end": "End Date"},
                       formats={"train_n": "{:,}", "test_n": "{:,}"})
        st.markdown(f"- **Mean fold ROC-AUC**: {bt['roc_auc_mean']:.3f} ± {bt['roc_auc_std']:.3f}\n"
                    f"- **Mean fold PR-AUC**: {bt['pr_auc_mean']:.3f} ± {bt['pr_auc_std']:.3f}\n"
                    f"- **Mean PR-AUC lift over prevalence**: {bt['pr_auc_lift_mean']:.2f}x ± {bt['pr_auc_lift_std']:.2f}")

    with tab_psi, st.container(border=True):
        C.card_title("Population Stability Index (drift check)")
        C.show(charts.psi_bar(fm["psi"]["rows"]), key="rob_psi")
        psi = pd.DataFrame(fm["psi"]["rows"])
        psi["feature"] = [charts.pretty_feature(f) for f in psi["feature"]]
        C.styled_table(psi, rename={"feature": "Feature", "psi": "PSI", "status": "Status"}, formats={"psi": "{:.5f}"})
        st.caption(f"PSI < 0.1 indicates no significant covariate shift between training and test distributions (max here: {fm['psi']['max_psi']:.4f}).")

    with tab_sanity:
        ls, ver = fm["label_shuffle"], fm["verification"]
        with st.container(border=True):
            C.card_title("1. Label-shuffle test")
            s1, s2, s3 = st.columns(3)
            with s1:
                C.kpi_card("Validation AUC (shuffled labels)", f"{ls['val_roc_auc']:.3f}", "expected ~0.50")
            with s2:
                C.kpi_card("Test AUC (shuffled labels)", f"{ls['test_roc_auc']:.3f}", "expected ~0.50")
            with s3:
                C.kpi_card("Result", ls["status"], "both within [0.47, 0.53]")
            st.caption("The same model is retrained on randomly shuffled training labels. Chance-level AUC proves the real model's "
                       "signal is not leakage or memorisation.")
        with st.container(border=True):
            C.card_title("2. Trivial-policy comparison (test set)")
            tp = pd.DataFrame(fm["trivial_policies"])
            C.styled_table(tp, rename={"share_flagged_pct": "Flagged Share (%)", "flagged_orders": "Flagged Orders"},
                           formats={"flagged_orders": "{:,}", "share_flagged_pct": "{:.2f}"})
            st.caption("Flag-all looks competitive on F1 only because most orders are late; the model's tiers add precision and MCC.")
        with st.container(border=True):
            C.card_title("3. Payment-type (TRANSFER) finding")
            rows = pd.DataFrame({
                "check": ["Cancel / fraud orders (all TRANSFER)", "Cancel / fraud share of TRANSFER orders", "Cancel / fraud share of other types",
                          "Late rate, CANCELED orders", "Late rate, SUSPECTED_FRAUD orders", "Late rate, all other statuses",
                          "TRANSFER late-rate gap within shipping mode", "... after removing cancel / fraud orders",
                          "Model test ROC-AUC, all orders", "Model test ROC-AUC, cancel / fraud removed"],
                "value": [f"{ver['cancel_fraud_orders']:,}", pc(ver["cancel_fraud_share_transfer"]), pc(ver["cancel_fraud_share_other_types"]),
                          pc(ver["late_rate_canceled"]), pc(ver["late_rate_suspected_fraud"]), pc(ver["late_rate_other_statuses"]),
                          C.fmt_pp(ver['transfer_gap_within_mode_all']), C.fmt_pp(ver['transfer_gap_within_mode_clean']),
                          f"{ver['test_roc_auc_all']:.4f}", f"{ver['test_roc_auc_without_cancel_fraud']:.4f}"]})
            C.styled_table(rows, rename={"check": "Check", "value": "Value"})
            C.callout("warning", "TRANSFER is a labelling artifact, not a payment effect",
                      "Every CANCELED / SUSPECTED_FRAUD order is a TRANSFER payment and is labelled on-time. Once those orders are removed "
                      "the TRANSFER difference disappears.")
