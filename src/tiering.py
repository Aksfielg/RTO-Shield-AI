"""
src/tiering.py
==============
Cluster-aware cutoff selection for RTO Shield AI - Model 1.

The calibrated scores of the final model are heavily discretised (isotonic
step function): a handful of score values hold most orders.  A cutoff placed
on or next to one of those clusters is fragile, so every cutoff in this
project is chosen as the MIDPOINT OF A LOW-DENSITY GAP between score clusters.

Definitions
-----------
cluster     : maximal run of consecutive unique scores whose neighbouring
              differences are all < LINK_GAP (single linkage).
candidate   : midpoint of a gap >= LINK_GAP between two clusters, with at most
              MAX_WINDOW_MASS of orders within +/-WINDOW of the midpoint.
Selection uses VALIDATION scores only; TEST is used only to verify.
"""

from __future__ import annotations

import numpy as np

LINK_GAP = 0.01          # scores closer than this belong to one cluster
WINDOW = 0.02            # density window half-width around a cutoff
MAX_WINDOW_MASS = 0.05   # max share of orders allowed inside the window
BIG_CLUSTER = 0.02       # clusters above this share may never be split
MIN_TIER_SHARE = 0.05    # every tier must hold at least this share of VAL
MAX_FLAG_SWING_PP = 5.0  # max change in flagged share for a +/-0.02 shift
SHIFTS = (-0.02, -0.01, 0.0, 0.01, 0.02)


class CutoffError(RuntimeError):
    """Raised when a cutoff cannot be placed safely or fails verification."""


def clusters(scores: np.ndarray, link: float = LINK_GAP) -> list[dict]:
    """Single-linkage clusters of unique score values."""
    scores = np.asarray(scores, dtype=float)
    vals, counts = np.unique(np.round(scores, 10), return_counts=True)
    out: list[dict] = []
    start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] - vals[i - 1] >= link:
            n = int(counts[start:i].sum())
            out.append({"lo": float(vals[start]), "hi": float(vals[i - 1]),
                        "n": n, "share": n / len(scores)})
            start = i
    return out


def window_mass(scores: np.ndarray, c: float, w: float = WINDOW) -> float:
    return float((np.abs(np.asarray(scores) - c) <= w).mean())


def candidate_cutoffs(val_scores: np.ndarray) -> list[float]:
    """Midpoints of low-density gaps (validation scores only)."""
    cl = clusters(val_scores)
    cands = []
    for a, b in zip(cl[:-1], cl[1:]):
        mid = (a["hi"] + b["lo"]) / 2.0
        if window_mass(val_scores, mid) <= MAX_WINDOW_MASS:
            cands.append(round(mid, 4))
    return cands


def cutoff_check(scores: np.ndarray, c: float) -> dict:
    """Describe how a cutoff sits relative to the clusters of `scores`."""
    scores = np.asarray(scores, dtype=float)
    cl = clusters(scores)
    inside = [k for k in cl if k["lo"] < c <= k["hi"]]
    big_split = [k for k in inside if k["share"] > BIG_CLUSTER]
    below = scores[scores < c]
    above = scores[scores >= c]
    return {
        "cutoff": float(c),
        "inside_cluster": bool(inside),
        "big_clusters_split": len(big_split),
        "nearest_score_below": float(below.max()) if len(below) else None,
        "nearest_score_above": float(above.min()) if len(above) else None,
        "share_within_0.01": float((np.abs(scores - c) <= 0.01).mean()),
        "share_within_0.02": window_mass(scores, c, 0.02),
        "in_gap": (not inside),
    }


def select_tier_cutoffs(val_scores: np.ndarray) -> tuple[float, float, list[float]]:
    """
    Pick (low_max, high_min) from candidate gap midpoints so that every tier
    holds >= MIN_TIER_SHARE of validation orders and the cumulative shares are
    as close as possible to the terciles (33% / 67%).
    """
    s = np.asarray(val_scores, dtype=float)
    cands = candidate_cutoffs(s)
    best, best_cost = None, np.inf
    for i, lo in enumerate(cands):
        for hi in cands[i + 1:]:
            sh_low = float((s < lo).mean())
            sh_high = float((s >= hi).mean())
            sh_mid = 1.0 - sh_low - sh_high
            if min(sh_low, sh_mid, sh_high) < MIN_TIER_SHARE:
                continue
            cost = abs(sh_low - 1 / 3) + abs((1 - sh_high) - 2 / 3)
            if cost < best_cost:
                best, best_cost = (lo, hi), cost
    if best is None:
        raise CutoffError("No pair of low-density gap midpoints gives three "
                          f"tiers of at least {MIN_TIER_SHARE:.0%} each.")
    return best[0], best[1], cands


def select_precision_threshold(val_scores, y_val, cands, target: float = 0.80) -> float:
    """Smallest candidate gap midpoint whose VALIDATION precision >= target."""
    s = np.asarray(val_scores, dtype=float)
    y = np.asarray(y_val)
    for c in sorted(cands):
        m = s >= c
        if m.sum() and y[m].mean() >= target:
            return float(c)
    raise CutoffError(f"No low-density gap midpoint reaches validation precision >= {target}.")


def policy_row(y, scores, thr: float) -> dict:
    y = np.asarray(y)
    f = np.asarray(scores) >= thr
    tp = int((f & (y == 1)).sum())
    fp = int((f & (y == 0)).sum())
    fn = int((~f & (y == 1)).sum())
    tn = int((~f & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    denom = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "flagged_share_pct": float(f.mean() * 100),
        "flagged_orders": int(f.sum()),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (rec + spec) / 2,
        "specificity": spec,
        "mcc": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
    }


def stability_table(scores, y, thr: float, name: str) -> tuple[list[dict], float]:
    """Flagged share / precision / recall at thr + shifts. Returns (rows, swing_pp)."""
    rows = []
    for d in SHIFTS:
        r = policy_row(y, scores, thr + d)
        rows.append({"policy": name, "shift": d, "threshold": round(thr + d, 4), **r})
    shares = [r["flagged_share_pct"] for r in rows]
    return rows, float(max(shares) - min(shares))
