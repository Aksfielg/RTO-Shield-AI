"""
ui/charts.py
============
Every chart in the dashboard, as Plotly figures built on the shared "rto" template
(ui/theme.py).  Palette: green = low, amber = medium, red = high, indigo = neutral.
No matplotlib / PNG images are shown in the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

from reasons import _CAT_PARENT_LABELS, _NUMERIC_LABELS
from ui import theme
from ui.components import fmt_pct
from ui.theme import AMBER, GREEN, INDIGO, INDIGO_LIGHT, MUTED, RED, SLATE, SLATE_LIGHT, TIER_BG, TIER_COLORS, style


def pretty_feature(name: str) -> str:
    """'cat__Shipping_Mode_Standard Class' / 'Days_for_shipment_scheduled' -> business label."""
    for prefix in ("cat__", "num__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for col in sorted(_CAT_PARENT_LABELS, key=len, reverse=True):
        if name.startswith(col + "_"):
            return f"{_CAT_PARENT_LABELS[col]}: {name[len(col) + 1:]}"
    return _NUMERIC_LABELS.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------------
# Single-order charts
# ---------------------------------------------------------------------------
def gauge(score: float, low_cut: float, high_cut: float) -> go.Figure:
    """Risk gauge; the centre number uses fmt_pct, like the badge and the sentence."""
    fig = go.Figure(go.Indicator(
        mode="gauge", value=score * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": SLATE, "ticksuffix": "%"},
            "bar": {"color": INDIGO, "thickness": 0.3},
            "bgcolor": "white", "borderwidth": 1, "bordercolor": SLATE_LIGHT,
            "steps": [
                {"range": [0, low_cut * 100], "color": TIER_BG["Low"]},
                {"range": [low_cut * 100, high_cut * 100], "color": TIER_BG["Medium"]},
                {"range": [high_cut * 100, 100], "color": TIER_BG["High"]},
            ],
            "threshold": {"line": {"color": RED, "width": 3.5}, "thickness": 0.85, "value": high_cut * 100},
        },
    ))
    fig.add_annotation(text=f"<b>{fmt_pct(score)}</b>", xref="paper", yref="paper", x=0.5, y=0.08,
                       showarrow=False, font=dict(size=38, color=theme.INK))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=10))
    return style(fig, height=230)


def grouped_shap_bar(labels, values, val_strs, top_k: int = 6) -> go.Figure:
    order = np.argsort(np.abs(values))[::-1][:top_k][::-1]
    vals = [float(values[i]) for i in order]
    names = [f"{labels[i]} ({val_strs[i]})" for i in order]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker_color=[RED if v > 0 else GREEN for v in vals],
        hovertext=[f"<b>{n}</b><br>SHAP log-odds: {v:+.4f}<br>{'Raises' if v > 0 else 'Lowers'} risk" for n, v in zip(names, vals)],
        hoverinfo="text",
    ))
    fig.update_xaxes(title="SHAP impact (log-odds)  |  red raises risk, green lowers it", zeroline=True)
    fig.update_layout(margin=dict(l=10, r=20, t=45, b=40))
    return style(fig, height=290, title="Key risk drivers (grouped factors)")


def shap_waterfall(labels, values, val_strs, base_value: float, top_k: int = 8) -> go.Figure:
    order = np.argsort(np.abs(values))[::-1]
    top, rest = order[:top_k], order[top_k:]
    names = ["Base value (average log-odds)"] + [f"{labels[i]} ({val_strs[i]})" for i in top]
    vals = [base_value] + [float(values[i]) for i in top]
    measures = ["absolute"] + ["relative"] * len(top)
    if len(rest):
        names.append("Other factors")
        vals.append(float(sum(values[i] for i in rest)))
        measures.append("relative")
    names.append("Final log-odds")
    vals.append(0.0)
    measures.append("total")
    fig = go.Figure(go.Waterfall(
        orientation="h", y=names, x=vals, measure=measures,
        increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=GREEN)),
        totals=dict(marker=dict(color=INDIGO)), connector=dict(line=dict(color=SLATE_LIGHT, width=1)),
        text=[("0.00" if abs(v) < 0.005 else f"{v:+.2f}") if m == "relative" else f"{v:.2f}" if m == "absolute" else ""
              for v, m in zip(vals, measures)],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Log-odds")
    fig.update_layout(margin=dict(l=10, r=70, t=45, b=40), showlegend=False)
    return style(fig, height=380, title="Local SHAP attribution (grouped factors)")


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def model_comparison_bar(df_test: pd.DataFrame, final_roc: float) -> go.Figure:
    colors = [INDIGO if "LightGBM" in m else SLATE for m in df_test["model"]]
    fig = go.Figure(go.Bar(x=df_test["model"], y=df_test["roc_auc"], marker_color=colors,
                           text=[f"{v:.3f}" for v in df_test["roc_auc"]], textposition="outside"))
    fig.add_hline(y=final_roc, line_dash="dash", line_color=INDIGO,
                  annotation_text=f"Final tuned model {final_roc:.3f}", annotation_position="top left")
    fig.update_yaxes(range=[0.5, 0.85], title="Test ROC-AUC (raw scores)")
    fig.update_layout(showlegend=False)
    return style(fig, height=330, title="Test-set discrimination across candidate models")


def roc_pr_curves(y: np.ndarray, raw: np.ndarray) -> tuple[go.Figure, go.Figure]:
    """ROC and PR curves from RAW scores (calibrated scores are heavily tied)."""
    fpr, tpr, _ = roc_curve(y, raw)
    prec, rec, _ = precision_recall_curve(y, raw)
    from sklearn.metrics import average_precision_score, roc_auc_score
    auc, ap = roc_auc_score(y, raw), average_precision_score(y, raw)
    f1 = go.Figure()
    f1.add_scatter(x=fpr, y=tpr, mode="lines", name=f"Final model (AUC {auc:.3f})", line=dict(color=INDIGO, width=2.5))
    f1.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC 0.500)", line=dict(color=SLATE, dash="dash"))
    f1.update_xaxes(title="False positive rate")
    f1.update_yaxes(title="True positive rate")
    f2 = go.Figure()
    f2.add_scatter(x=rec, y=prec, mode="lines", name=f"Final model (PR-AUC {ap:.3f})", line=dict(color=INDIGO, width=2.5))
    f2.add_scatter(x=[0, 1], y=[y.mean(), y.mean()], mode="lines", name=f"Prevalence ({y.mean():.3f})",
                   line=dict(color=SLATE, dash="dash"))
    f2.update_xaxes(title="Recall")
    f2.update_yaxes(title="Precision")
    return style(f1, 340, "ROC curve (test set)"), style(f2, 340, "Precision-recall curve (test set)")


def reliability(cal: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color=SLATE, dash="dash"))
    fig.add_scatter(x=cal["raw"]["mean_predicted"], y=cal["raw"]["fraction_positive"], mode="lines+markers",
                    name="Raw scores", line=dict(color=SLATE_LIGHT, width=2.5), marker=dict(size=8, color=SLATE))
    fig.add_scatter(x=cal["calibrated"]["mean_predicted"], y=cal["calibrated"]["fraction_positive"],
                    mode="lines+markers", name="Isotonic-calibrated", line=dict(color=INDIGO, width=2.5), marker=dict(size=8))
    fig.update_xaxes(title="Mean predicted probability", range=[0, 1])
    fig.update_yaxes(title="Observed late fraction", range=[0, 1])
    return style(fig, 380, "Reliability diagram (final model, test set)")


def lift_bar(deciles: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=deciles["decile"], y=deciles["lift"], marker_color=INDIGO,
                           text=[f"{v:.2f}x" for v in deciles["lift"]], textposition="outside"))
    fig.add_hline(y=1, line_dash="dash", line_color=SLATE, annotation_text="Random = 1.0x")
    fig.update_xaxes(title="Score decile (1 = highest raw score)", dtick=1)
    fig.update_yaxes(title="Lift")
    fig.update_layout(showlegend=False)
    return style(fig, 320, "Lift by predicted-risk decile")


def gain_curve(deciles: pd.DataFrame, prevalence: float) -> go.Figure:
    xs = [0] + list(deciles["decile"] * 10)
    ys = [0] + list(deciles["cum_gain_pct"])
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines+markers", name="Model", line=dict(color=INDIGO, width=2.5))
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random", line=dict(color=SLATE, dash="dash"))
    fig.add_scatter(x=[0, prevalence * 100, 100], y=[0, 100, 100], mode="lines", name="Perfect model",
                    line=dict(color=SLATE_LIGHT, dash="dot"))
    fig.update_xaxes(title="% of orders reviewed (highest score first)", range=[0, 100])
    fig.update_yaxes(title="% of late orders captured", range=[0, 102])
    return style(fig, 320, "Cumulative gain")


def tier_late_rate_bar(tier_stats: dict, base_rate: float) -> go.Figure:
    tiers = ["Low", "Medium", "High"]
    fig = go.Figure(go.Bar(
        x=tiers, y=[tier_stats[t]["late_rate"] for t in tiers], marker_color=[TIER_COLORS[t] for t in tiers],
        text=[f"{fmt_pct(tier_stats[t]['late_rate'], clip=False)}<br>n = {tier_stats[t]['orders']:,}" for t in tiers],
        textposition="outside"))
    fig.add_hline(y=base_rate, line_dash="dash", line_color=SLATE,
                  annotation_text=f"Overall late rate {fmt_pct(base_rate, clip=False)}", annotation_position="top left")
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%", title="Share of orders that were late")
    fig.update_layout(showlegend=False)
    return style(fig, 320, "Observed late rate by risk tier (test set)")


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def importance_grouped(grouped: list[dict]) -> go.Figure:
    g = grouped[::-1]
    fig = go.Figure(go.Bar(x=[r["share"] for r in g], y=[r["factor"] for r in g], orientation="h", marker_color=INDIGO,
                           text=[f"{r['share'] * 100:.1f}%" for r in g], textposition="outside"))
    fig.update_xaxes(title="Share of total mean |SHAP|", tickformat=".0%")
    fig.update_layout(showlegend=False)
    return style(fig, 340, "Global importance by business factor")


def importance_encoded(top: list[dict]) -> go.Figure:
    t = top[::-1]
    fig = go.Figure(go.Bar(x=[r["mean_abs_shap"] for r in t], y=[pretty_feature(r["feature"]) for r in t],
                           orientation="h", marker_color=INDIGO_LIGHT, text=[f"{r['mean_abs_shap']:.3f}" for r in t],
                           textposition="outside"))
    fig.update_xaxes(title="Mean |SHAP| (log-odds)")
    fig.update_layout(showlegend=False)
    return style(fig, 380, "Top encoded features")


def beeswarm(sample: dict, top_n: int = 12) -> go.Figure:
    sv, X, names = sample["sv"], sample["X"], sample["feat_names"]
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:top_n][::-1]
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for k, j in enumerate(order):
        col = X[:, j]
        lo, hi = np.percentile(col, 2), np.percentile(col, 98)
        norm = np.clip((col - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(col)
        fig.add_scatter(
            x=sv[:, j], y=k + rng.uniform(-0.3, 0.3, len(col)), mode="markers", showlegend=False,
            marker=dict(size=5, color=norm, colorscale=[[0, INDIGO_LIGHT], [1, theme.INDIGO_DARK]], opacity=0.75,
                        showscale=(k == len(order) - 1),
                        colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["Low", "High"], len=0.5, thickness=10)
                        if k == len(order) - 1 else None),
            hovertemplate=f"<b>{pretty_feature(names[j])}</b><br>SHAP %{{x:+.3f}}<extra></extra>")
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[pretty_feature(names[j]) for j in order], gridcolor="white",
                     zeroline=False)
    fig.add_vline(x=0, line_color=SLATE_LIGHT)
    fig.update_xaxes(title="SHAP value (impact on log-odds of being late)")
    return style(fig, 460, f"SHAP beeswarm - top {top_n} features ({len(sv)}-order test sample)")


def dependence(x: np.ndarray, shap_vals: np.ndarray, xlabel: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=x, y=shap_vals, mode="markers", marker=dict(size=6, color=INDIGO, opacity=0.6)))
    fig.add_hline(y=0, line_dash="dash", line_color=SLATE)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="SHAP value (log-odds impact)")
    fig.update_layout(showlegend=False)
    return style(fig, 340, f"SHAP dependence: {xlabel}")


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def per_mode_ci(per_mode: list[dict]) -> go.Figure:
    rows = per_mode[::-1]
    fig = go.Figure()
    for r in rows:
        color = GREEN if r["flag"] == "informative" else AMBER
        fig.add_scatter(
            x=[r["roc_auc"]], y=[r["mode"]], mode="markers", showlegend=False,
            marker=dict(size=11, color=color),
            error_x=dict(type="data", symmetric=False, array=[r["roc_auc_ci_hi"] - r["roc_auc"]],
                         arrayminus=[r["roc_auc"] - r["roc_auc_ci_lo"]], color=color, thickness=2.5),
            hovertemplate=f"<b>{r['mode']}</b><br>ROC-AUC {r['roc_auc']:.3f} "
                          f"[{r['roc_auc_ci_lo']:.3f}, {r['roc_auc_ci_hi']:.3f}]<br>{r['flag']}<extra></extra>")
    fig.add_vline(x=0.5, line_dash="dash", line_color=SLATE, annotation_text="chance 0.50", annotation_position="top")
    fig.add_vline(x=0.55, line_dash="dot", line_color=AMBER, annotation_text="0.55 floor", annotation_position="bottom")
    fig.update_xaxes(title="ROC-AUC with 95% bootstrap CI (raw scores)", range=[0.45, 1.02])
    return style(fig, 300, "Ranking quality by shipping mode (green = informative, amber = limited)")


def ablation_bar(abl: list[dict]) -> go.Figure:
    labels = [r["label"] for r in abl]
    roc = np.array([r["roc_auc"] for r in abl])
    lo = np.array([r["roc_auc_ci_lo"] for r in abl])
    hi = np.array([r["roc_auc_ci_hi"] for r in abl])
    fig = go.Figure()
    colors = [INDIGO if r["label"] == "All features" else SLATE for r in abl]
    fig.add_bar(x=labels, y=roc, name="ROC-AUC (95% CI)", marker_color=colors,
                error_y=dict(type="data", symmetric=False, array=hi - roc, arrayminus=roc - lo, color=theme.INK),
                text=[f"{v:.3f}" for v in roc], textposition="outside")
    fig.add_hline(y=0.5, line_dash="dash", line_color=SLATE, annotation_text="chance 0.50")
    fig.update_yaxes(range=[0.45, 0.85], title="ROC-AUC (raw scores)")
    fig.update_layout(showlegend=False)
    return style(fig, 380, "Feature ablation: ROC-AUC with 95% bootstrap CI")


def backtest_lines(rows: list[dict]) -> go.Figure:
    x = [f"F{r['fold']}<br>{r['test_start']}" for r in rows]
    fig = go.Figure()
    fig.add_scatter(x=x, y=[r["roc_auc"] for r in rows], mode="lines+markers", name="ROC-AUC", line=dict(color=INDIGO, width=2.5))
    fig.add_scatter(x=x, y=[r["pr_auc"] for r in rows], mode="lines+markers", name="PR-AUC", line=dict(color=SLATE, width=2.5))
    fig.update_yaxes(range=[0.5, 1.0], title="Score (raw model scores)")
    return style(fig, 340, "Rolling-origin backtest")


def psi_bar(rows: list[dict]) -> go.Figure:
    rows = rows[::-1]
    fig = go.Figure(go.Bar(x=[r["psi"] for r in rows], y=[pretty_feature(r["feature"]) for r in rows], orientation="h",
                           marker_color=[GREEN if r["psi"] < 0.1 else AMBER if r["psi"] < 0.2 else RED for r in rows]))
    fig.add_vline(x=0.1, line_dash="dot", line_color=AMBER, annotation_text="0.1")
    fig.add_vline(x=0.2, line_dash="dot", line_color=RED, annotation_text="0.2")
    fig.update_xaxes(title="Population Stability Index (train vs test)", range=[0, 0.25])
    fig.update_layout(showlegend=False)
    return style(fig, 320, "Covariate drift of the top features")


# ---------------------------------------------------------------------------
# Impact / batch
# ---------------------------------------------------------------------------
def net_benefit_bar(df: pd.DataFrame, n: int) -> go.Figure:
    colors = [SLATE if s == "Do Nothing" else AMBER if s == "Contact Everyone" else INDIGO for s in df["Strategy"]]
    from ui.components import fmt_money
    fig = go.Figure(go.Bar(x=df["Strategy"], y=df["Net Benefit"], marker_color=colors,
                           text=[fmt_money(v, signed=True) for v in df["Net Benefit"]], textposition="outside"))
    fig.update_yaxes(title="Net benefit ($)", zeroline=True, zerolinecolor=SLATE)
    fig.update_layout(showlegend=False)
    return style(fig, 300, f"Net benefit by strategy (test set, N = {n:,})")


def tier_counts_bar(counts: dict, shares: dict, labels: dict) -> go.Figure:
    tiers = ["Low", "Medium", "High"]
    fig = go.Figure(go.Bar(x=[labels[t] for t in tiers], y=[counts[t] for t in tiers], marker_color=[TIER_COLORS[t] for t in tiers],
                           text=[f"{counts[t]:,} ({shares[t] * 100:.1f}%)" for t in tiers], textposition="outside"))
    fig.update_layout(showlegend=False)
    return style(fig, 280, "Batch distribution across risk tiers")


def predicted_vs_actual(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=[r["tier"] for r in rows], y=[r["mean_predicted"] for r in rows], name="Mean predicted", marker_color=INDIGO)
    fig.add_bar(x=[r["tier"] for r in rows], y=[r["actual_late_rate"] for r in rows], name="Actual late rate", marker_color=SLATE)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_layout(barmode="group")
    return style(fig, 280, "Predicted vs actual late rate by tier")


# ---------------------------------------------------------------------------
# Additional charts
# ---------------------------------------------------------------------------
def split_timeline(splits: dict) -> go.Figure:
    names = {"train": "Train", "val": "Validation", "test": "Test (holdout)"}
    colors = {"train": INDIGO, "val": INDIGO_LIGHT, "test": AMBER}
    fig = go.Figure()
    for k in ["train", "val", "test"]:
        sp = splits[k]
        start, end = pd.Timestamp(sp["date_min"]), pd.Timestamp(sp["date_max"])
        label = f"{names[k]}<br>{sp['orders']:,} orders"
        fig.add_bar(y=[label], x=[(end - start).total_seconds() * 1000], base=[start], orientation="h",
                    marker_color=colors[k], showlegend=False,
                    hovertemplate=f"<b>{names[k]}</b><br>{sp['date_min']} to {sp['date_max']}<br>{sp['orders']:,} orders<extra></extra>")
    fig.update_xaxes(type="date", title="Order date (strictly chronological split)")
    fig.update_yaxes(autorange="reversed")
    return style(fig, 220, "Train / validation / test timeline")


def sensitivity_chart(curves: dict, breakevens: dict) -> go.Figure:
    colors = {0: INDIGO, 1: AMBER, 2: SLATE}
    fig = go.Figure()
    for i, (name, (rates, net)) in enumerate(curves.items()):
        fig.add_scatter(x=[r * 100 for r in rates], y=net, mode="lines", name=name, line=dict(color=colors.get(i, SLATE), width=2.5))
        be = breakevens.get(name)
        if be is not None and 0 <= be <= rates[-1]:
            fig.add_scatter(x=[be * 100], y=[0], mode="markers", showlegend=False, marker=dict(size=10, color=colors.get(i, SLATE),
                            line=dict(color="white", width=2)), hovertemplate=f"Break-even {be * 100:.1f}%<extra>{name}</extra>")
    fig.add_hline(y=0, line_color=SLATE_LIGHT)
    fig.update_xaxes(title="Intervention success rate inside the selected shipping modes (%)")
    fig.update_yaxes(title="Net benefit ($)")
    return style(fig, 340, "Net benefit vs success rate (dots = break-even)")


def tier_by_mode(ct: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for t in ["Low", "Medium", "High"]:
        if t in ct.columns:
            fig.add_bar(x=ct.index, y=ct[t], name=t, marker_color=TIER_COLORS[t], text=ct[t], textposition="inside")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Orders")
    return style(fig, 300, "Tier counts by shipping mode")
