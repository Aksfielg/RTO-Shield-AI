"""Impact Simulator: what-if intervention economics on the real holdout scores (assumptions, not measurements)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ui import charts
from ui import components as C
from ui import data

RATES = np.round(np.arange(0.0, 0.6001, 0.01), 2)


def economics(mask, y, inside, late_cost, int_cost, r_in, r_out) -> dict:
    """Net benefit of targeting `mask`: success rate r_in inside the selected modes, r_out outside them."""
    n_t = int(mask.sum())
    late = mask & (y == 1)
    late_in, late_out = int((late & inside).sum()), int((late & ~inside).sum())
    gross = late_cost * (r_in * late_in + r_out * late_out)
    cost = n_t * int_cost
    return {"n": n_t, "late": late_in + late_out, "late_in": late_in, "late_out": late_out, "gross": gross, "cost": cost, "net": gross - cost}


def break_even(mask, y, inside, late_cost, int_cost, r_out) -> float | None:
    """Inside-mode success rate at which net benefit is zero (None if nothing inside can be rescued)."""
    late = mask & (y == 1)
    late_in, late_out = int((late & inside).sum()), int((late & ~inside).sum())
    if late_in == 0:
        return None
    return (int_cost * int(mask.sum()) - late_cost * r_out * late_out) / (late_cost * late_in)


def render() -> None:
    thr = data.thr()
    sc = data.test_scores()
    _, test_df = data.datasets()
    cal_p, y = sc["cal"], sc["y"]                    # calibrated scores drive DECISIONS
    modes = test_df["Shipping_Mode"].values
    N, n_late = len(y), int(y.sum())
    base_rate = n_late / N
    all_modes = sorted(set(modes))

    C.page_header("Impact Simulator", "Simulate economic cost-benefit and intervention ROI using real holdout test-set scores.")

    col_in, col_out = st.columns([1, 1.25], gap="large")
    with col_in, st.container(border=True):
        C.card_title("Business inputs")
        late_cost = st.number_input("Cost of a failed delivery ($)", 1.0, 200.0, 15.0, 1.0,
                                    help="Average cost when an order is delivered late: friction, refunds, return handling.")
        int_cost = st.number_input("Cost per intervention ($)", 0.10, 50.0, 2.50, 0.25,
                                   help="Direct cost of proactive SMS/call verification, address confirmation or prioritised dispatch.")
        inside_modes = st.multiselect("Shipping modes where intervention can work", all_modes,
                                      default=[m for m in all_modes if m != "First Class"])
        r_in = st.slider("Success rate inside those modes (%)", 0, 100, 30, 1) / 100.0
        r_out = st.slider("Success rate in all other modes (%)", 0, 100, 0, 1,
                          help="User-set. First Class lateness comes from the carrier SLA (1-day promise), which a call or address check cannot fix.") / 100.0
        st.caption("First Class lateness comes from the carrier SLA, so it is excluded by default (0% success outside the selected modes).")

        C.card_title("Triage policy")
        policy = st.radio("Select triage policy:", ["Flag High tier", "Flag by threshold", "Flag all"], index=0)
        high_min = thr["tier_cutoffs"]["high_min"]
        t = st.slider("Risk score threshold (threshold policy)", 0.05, 0.95, float(thr["thresholds"]["precision_80"]), 0.01,
                      help="Orders with a calibrated score at or above the threshold are targeted. Defaults to the precision >= 0.80 threshold.")

    inside = np.isin(modes, inside_modes)
    policies = {"High tier": cal_p >= high_min, f"Threshold >= {t:.2f}": cal_p >= t, "Flag all": np.ones(N, dtype=bool)}
    selected = {"Flag High tier": "High tier", "Flag by threshold": f"Threshold >= {t:.2f}", "Flag all": "Flag all"}[policy]
    e_sel = economics(policies[selected], y, inside, late_cost, int_cost, r_in, r_out)
    e_all = economics(policies["Flag all"], y, inside, late_cost, int_cost, r_in, r_out)
    precision = e_sel["late"] / e_sel["n"] if e_sel["n"] else float("nan")

    with col_out, st.container(border=True):
        C.card_title(f"Test-set economics: {selected}")
        if not inside_modes:
            st.warning("No shipping mode selected: interventions rescue nothing inside the selected modes.")
        k1, k2 = st.columns(2)
        with k1:
            C.metric("Orders targeted", f"{e_sel['n']:,}", f"{C.fmt_pct(e_sel['n'] / N, clip=False)} of {N:,} test orders")
            C.metric("Late orders reached", f"{e_sel['late']:,}", f"{C.fmt_pct(e_sel['late'] / n_late, clip=False)} of all late orders")
        with k2:
            C.metric("Targeting precision", C.fmt_pct(precision, clip=False), f"Baseline late rate: {C.fmt_pct(base_rate, clip=False)}")
            C.metric("Net benefit vs doing nothing", C.fmt_money(e_sel["net"]),
                     f"{C.fmt_money(e_sel['net'] - e_all['net'], signed=True)} vs contacting everyone")
        bars = {"Do Nothing": 0.0, "Contact Everyone": e_all["net"]}
        if selected != "Flag all":                    # avoid a duplicate bar when the selected policy IS contact-everyone
            bars[selected] = e_sel["net"]
        C.show(charts.net_benefit_bar(pd.DataFrame({"Strategy": list(bars), "Net Benefit": list(bars.values())}), N), key="impact_net")

    with st.container(border=True):
        C.card_title("Sensitivity: net benefit vs intervention success rate")
        curves, be = {}, {}
        for name, mask in policies.items():
            curves[name] = (RATES, [economics(mask, y, inside, late_cost, int_cost, r, r_out)["net"] for r in RATES])
            be[name] = break_even(mask, y, inside, late_cost, int_cost, r_out)
        C.show(charts.sensitivity_chart(curves, be), key="impact_sens")
        rows = []
        for name, mask in policies.items():
            b = be[name]
            if b is None:
                txt = "n/a (no late orders reached inside the selected modes)"
            elif b <= 0:
                txt = "always net-positive"
            elif b > 1:
                txt = "not reachable (> 100%)"
            else:
                txt = C.fmt_pct(b, clip=False)
            rows.append({"policy": name, "orders_targeted": int(mask.sum()), "break_even": txt})
        C.styled_table(pd.DataFrame(rows), rename={"policy": "Policy", "orders_targeted": "Orders Targeted", "break_even": "Break-Even Success Rate"},
                       formats={"orders_targeted": "{:,}"})
        st.caption(f"Break-even = the success rate inside the selected modes at which net benefit is $0, with {C.fmt_pct(r_out, clip=False)} "
                   "success outside them. Above it the policy pays for itself.")

    with st.container(border=True):
        C.card_title(f"Strategic policy comparison (holdout test partition: {N:,} orders)")
        pc = lambda v: C.fmt_pct(v, clip=False)
        rows = []
        for name, mask in policies.items():
            e = economics(mask, y, inside, late_cost, int_cost, r_in, r_out)
            rows.append({"policy": name, "orders_targeted": e["n"], "share": e["n"] / N, "late_reached": e["late"],
                         "recall": e["late"] / n_late, "precision": e["late"] / e["n"] if e["n"] else np.nan,
                         "intervention_cost": e["cost"], "gross_savings": e["gross"], "net_benefit": e["net"],
                         "net_vs_contact_all": e["net"] - e_all["net"]})
        C.styled_table(pd.DataFrame(rows),
                       rename={"share": "Share of Test Set", "late_reached": "Late Orders Reached", "net_vs_contact_all": "Net vs Contact All"},
                       formats={"share": pc, "recall": pc, "precision": pc, "orders_targeted": "{:,}", "late_reached": "{:,}",
                                "intervention_cost": C.fmt_money, "gross_savings": C.fmt_money, "net_benefit": C.fmt_money,
                                "net_vs_contact_all": C.fmt_money})

    C.callout("note", "Assumptions, not measured",
              "Costs and success rates are user-set assumptions, not measured values. Estimating the real intervention effect is the job "
              "of Model 2 (uplift modelling).")
