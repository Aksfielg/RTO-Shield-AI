"""
src/reasons.py
==============
Plain-English reason codes from a SHAP explanation row.

Public API
----------
get_reason_codes(shap_row, feature_names, feature_values, k=3)
    Returns a list of k plain-English reason strings for one prediction,
    ordered by absolute SHAP magnitude (largest first).

Each reason has the form:
    "<Feature label> is <value_str> (<direction> risk)"

where direction is "raises" if the SHAP value is positive (pushes toward
late delivery) and "lowers" if negative.

Usage
-----
    from reasons import get_reason_codes
    codes = get_reason_codes(shap_values[i], feature_names, raw_feature_values[i])
    for c in codes:
        print(" •", c)
"""

from __future__ import annotations

from typing import Any
import numpy as np

# ---------------------------------------------------------------------------
# Human-readable label mapping (strip sklearn prefix / OHE suffix)
# cat__Shipping_Mode_First Class  →  "Shipping Mode = First Class"
# num__customer_prev_late_rate    →  "Customer historical late rate"
# ---------------------------------------------------------------------------

_NUMERIC_LABELS: dict[str, str] = {
    "Days_for_shipment_scheduled": "Scheduled shipping days",
    "order_value":                 "Order value ($)",
    "n_items":                     "Number of items",
    "avg_discount_rate":           "Average discount rate",
    "value_per_item":              "Value per item ($)",
    "is_cash":                     "Cash (COD) payment",
    "is_weekend":                  "Weekend order",
    "is_night_order":              "Night-time order",
    "high_discount_flag":          "High discount (>15%)",
    "n_distinct_categories":       "Distinct product categories",
    "order_month":                 "Order month",
    "order_dow":                   "Day of week",
    "order_hour":                  "Hour of day",
    "customer_prev_orders":        "Customer order history (count)",
    "customer_prev_late_rate":     "Customer historical late rate",
    "country_prev_orders":         "Country order history (count)",
    "country_prev_late_rate":      "Country historical late rate",
    "customer_prev_cash_share":    "Customer cash-payment share",
    "days_since_prev_order":       "Days since previous order",
}


def _parse_feature_name(raw: str) -> tuple[str, str | None]:
    """
    Strip sklearn ColumnTransformer prefix and return (clean_name, category_value).

    Examples
    --------
    "cat__Shipping_Mode_First Class" → ("Shipping Mode", "First Class")
    "num__customer_prev_late_rate"   → ("Customer historical late rate", None)
    "cat__Type_CASH"                 → ("Payment type", "CASH")
    """
    if raw.startswith("num__"):
        col = raw[len("num__"):]
        label = _NUMERIC_LABELS.get(col, col.replace("_", " ").title())
        return label, None

    if raw.startswith("cat__"):
        rest = raw[len("cat__"):]
        # Find the last segment after each known categorical column name
        # Pattern: <ColName>_<CategoryValue>
        # We try to split at the first underscore that follows a known column
        # (some column names themselves contain underscores).
        known_cols = [
            "Type", "Shipping_Mode", "Market", "Order_Region", "Order_Country",
            "Customer_Segment", "Department_Name", "main_item_category",
        ]
        for col in sorted(known_cols, key=len, reverse=True):
            prefix = col + "_"
            if rest.startswith(prefix):
                cat_value = rest[len(prefix):]
                # Pretty-print column name
                col_label = col.replace("_", " ").replace("  ", " ").strip()
                return col_label, cat_value

    # Fallback: strip prefix, replace underscores
    name = raw.split("__", 1)[-1].replace("_", " ").strip()
    return name.title(), None


def _format_raw_value(col: str, raw_val: Any) -> str:
    """Format an unscaled/raw feature value for display in the reason string."""
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return "N/A"
    if isinstance(raw_val, (bool, np.bool_)):
        return "Yes" if raw_val else "No"
    if col in {
        "Days_for_shipment_scheduled", "n_items", "order_hour", "order_dow",
        "order_month", "customer_prev_orders", "country_prev_orders",
        "days_since_prev_order", "n_distinct_categories"
    }:
        try:
            return str(int(round(float(raw_val))))
        except Exception:
            return str(raw_val)
    if col in {"order_value", "value_per_item"}:
        try:
            return f"${float(raw_val):.2f}"
        except Exception:
            return f"${raw_val}"
    if col in {
        "avg_discount_rate", "customer_prev_late_rate",
        "country_prev_late_rate", "customer_prev_cash_share"
    }:
        try:
            return f"{float(raw_val):.1%}"
        except Exception:
            return str(raw_val)
    if isinstance(raw_val, float):
        return f"{raw_val:.3g}"
    return str(raw_val)


def _format_value(raw_val, cat_value: str | None) -> str:
    """Format a feature value for display in the reason string."""
    if cat_value is not None:
        return cat_value
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return "N/A"
    if isinstance(raw_val, (bool, np.bool_)):
        return "Yes" if raw_val else "No"
    if isinstance(raw_val, float):
        return f"{raw_val:.3g}"
    return str(raw_val)


def get_reason_codes(
    shap_row:       "np.ndarray",
    feature_names:  list[str],
    feature_values: "np.ndarray | list",
    k: int = 3,
    raw_dict: dict | None = None,
) -> list[str]:
    """
    Return k plain-English reason strings for one prediction.

    Parameters
    ----------
    shap_row : 1-D array of SHAP values, one per feature.
    feature_names : list of feature name strings (from preprocessor).
    feature_values : 1-D array/list of transformed feature values for this row.
    k : number of top reasons to return (default 3).
    raw_dict : optional dict containing raw (unscaled) feature values.

    Returns
    -------
    list[str] of length <= k, ordered by |SHAP| descending.
    """
    shap_arr = np.asarray(shap_row, dtype=float)
    vals_arr = list(feature_values)

    # Sort by absolute SHAP magnitude
    order = np.argsort(np.abs(shap_arr))[::-1]
    top_k = order[:k]

    reasons: list[str] = []
    for idx in top_k:
        raw_name  = feature_names[idx]
        shap_val  = float(shap_arr[idx])
        feat_val  = vals_arr[idx] if idx < len(vals_arr) else None

        label, cat_value = _parse_feature_name(raw_name)

        if raw_name.startswith("num__"):
            col = raw_name[len("num__"):]
            if raw_dict is not None and col in raw_dict and raw_dict[col] is not None:
                value_str = _format_raw_value(col, raw_dict[col])
            else:
                value_str = _format_value(feat_val, None)
        elif cat_value is not None:
            # Check if this category is inactive (0 in one-hot)
            if feat_val is not None:
                try:
                    is_zero = abs(float(feat_val)) < 0.5
                except Exception:
                    is_zero = False
                value_str = f"not {cat_value}" if is_zero else cat_value
            else:
                value_str = cat_value
        else:
            value_str = _format_value(feat_val, cat_value)

        direction = "raises" if shap_val > 0 else "lowers"
        reasons.append(f"{label} is {value_str} ({direction} risk)")

    return reasons

