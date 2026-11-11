"""
src/reasons.py
==============
Plain-English reason codes from grouped SHAP explanations.

Groups/sums SHAP values of one-hot encoded columns into their parent features:
  - Shipping mode (from cat__Shipping_Mode_*)
  - Payment type (from cat__Type_*)
  - Market (from cat__Market_*)
  - Department (from cat__Department_Name_*)
  - Segment (from cat__Customer_Segment_*)
  - Country (from cat__Order_Country_*)
  - Order region (from cat__Order_Region_*)
  - Item category (from cat__main_item_category_*)

Generates human-readable reason strings formatted as:
  "<Parent feature>: <raw_value> <raises/lowers> risk"

Example:
  "Shipping mode: Second Class raises risk"
  "Scheduled shipping days: 4 lowers risk"
"""

from __future__ import annotations

from typing import Any
import numpy as np

# ---------------------------------------------------------------------------
# Human-readable label mappings
# ---------------------------------------------------------------------------

_CAT_PARENT_LABELS: dict[str, str] = {
    "Shipping_Mode":      "Shipping mode",
    "Type":               "Payment type",
    "Market":             "Market",
    "Customer_Segment":   "Segment",
    "Department_Name":    "Department",
    "Order_Country":      "Country",
    "Order_Region":       "Order region",
    "main_item_category": "Item category",
}

_NUMERIC_LABELS: dict[str, str] = {
    "Days_for_shipment_scheduled": "Scheduled shipping days",
    "order_value":                 "Order value",
    "n_items":                     "Number of items",
    "avg_discount_rate":           "Average discount rate",
    "value_per_item":              "Value per item",
    "is_cash":                     "Cash payment",
    "is_weekend":                  "Weekend order",
    "is_night_order":              "Night-time order",
    "high_discount_flag":          "High discount (>15%)",
    "n_distinct_categories":       "Distinct product categories",
    "order_month":                 "Order month",
    "order_dow":                   "Day of week",
    "order_hour":                  "Hour of day",
    "customer_prev_orders":        "Customer order history",
    "customer_prev_late_rate":     "Customer historical late rate",
    "country_prev_orders":         "Country order history",
    "country_prev_late_rate":      "Country historical late rate",
    "customer_prev_cash_share":    "Customer cash-payment share",
    "days_since_prev_order":       "Days since previous order",
}

_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _format_raw_value(col: str, raw_val: Any) -> str:
    """Format an unscaled/raw feature value for display in the reason string."""
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return "N/A"
    if isinstance(raw_val, (bool, np.bool_)):
        return "Yes" if raw_val else "No"

    if col in {"Days_for_shipment_scheduled", "n_items", "customer_prev_orders", "country_prev_orders",
                "days_since_prev_order", "n_distinct_categories"}:
        try:
            return str(int(round(float(raw_val))))
        except Exception:
            return str(raw_val)

    if col == "order_hour":
        try:
            h = int(round(float(raw_val)))
            return f"{h:02d}:00"
        except Exception:
            return str(raw_val)

    if col == "order_dow":
        try:
            d = int(round(float(raw_val)))
            if 0 <= d < 7:
                return _DOW_NAMES[d]
            return str(d)
        except Exception:
            return str(raw_val)

    if col == "order_month":
        try:
            m = int(round(float(raw_val)))
            if 1 <= m <= 12:
                return _MONTH_NAMES[m]
            return str(m)
        except Exception:
            return str(raw_val)

    if col in {"order_value", "value_per_item"}:
        try:
            return f"${float(raw_val):.2f}"
        except Exception:
            return f"${raw_val}"

    if col in {"avg_discount_rate", "customer_prev_late_rate", "country_prev_late_rate", "customer_prev_cash_share"}:
        try:
            return f"{float(raw_val):.1%}"
        except Exception:
            return str(raw_val)

    if col in {"is_cash", "is_weekend", "is_night_order", "high_discount_flag"}:
        try:
            return "Yes" if int(round(float(raw_val))) == 1 else "No"
        except Exception:
            return str(raw_val)

    if isinstance(raw_val, float):
        return f"{raw_val:.3g}"
    return str(raw_val)


def group_shap_values(
    shap_row:      "np.ndarray",
    feature_names: list[str],
    raw_dict:      dict | None = None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """
    Sum SHAP values of one-hot encoded columns into their parent features.

    Returns
    -------
    parent_labels : list[str]
        Human-readable feature labels (e.g. 'Shipping mode', 'Scheduled shipping days').
    parent_shap : np.ndarray
        Consolidated SHAP values (1-D array of summed contributions).
    formatted_vals : list[str]
        Formatted raw feature values in unscaled business units.
    """
    raw_dict = raw_dict or {}
    shap_arr = np.asarray(shap_row, dtype=float)

    grouped_shap: dict[str, float] = {}
    grouped_val_str: dict[str, str] = {}

    known_cats = sorted(_CAT_PARENT_LABELS.keys(), key=len, reverse=True)

    for idx, fn in enumerate(feature_names):
        s_val = float(shap_arr[idx])

        if fn.startswith("cat__"):
            matched_col = None
            for col in known_cats:
                prefix = f"cat__{col}_"
                if fn.startswith(prefix):
                    matched_col = col
                    break

            if matched_col is not None:
                label = _CAT_PARENT_LABELS[matched_col]
                grouped_shap[label] = grouped_shap.get(label, 0.0) + s_val
                if label not in grouped_val_str:
                    val = raw_dict.get(matched_col, "N/A")
                    grouped_val_str[label] = str(val) if val is not None else "N/A"
            else:
                # Fallback categorical
                rest = fn[len("cat__"):]
                grouped_shap[rest] = grouped_shap.get(rest, 0.0) + s_val
                if rest not in grouped_val_str:
                    grouped_val_str[rest] = "Active"

        elif fn.startswith("num__"):
            col = fn[len("num__"):]
            label = _NUMERIC_LABELS.get(col, col.replace("_", " ").title())
            grouped_shap[label] = grouped_shap.get(label, 0.0) + s_val
            raw_val = raw_dict.get(col, None)
            grouped_val_str[label] = _format_raw_value(col, raw_val)

        else:
            grouped_shap[fn] = grouped_shap.get(fn, 0.0) + s_val
            grouped_val_str[fn] = str(raw_dict.get(fn, "N/A"))

    labels = list(grouped_shap.keys())
    values = np.array([grouped_shap[lbl] for lbl in labels], dtype=float)
    val_strs = [grouped_val_str.get(lbl, "N/A") for lbl in labels]

    return labels, values, val_strs


def get_reason_codes(
    shap_row:       "np.ndarray",
    feature_names:  list[str],
    feature_values: "np.ndarray | list | dict | None" = None,
    k: int = 3,
    raw_dict: dict | None = None,
) -> list[str]:
    """
    Return k plain-English reason strings with grouped parent features and raw values.

    Parameters
    ----------
    shap_row : 1-D array of SHAP values across all 252 encoded features.
    feature_names : list of feature name strings (from preprocessor).
    feature_values : optional transformed feature values or raw_dict.
    k : number of top reasons to return (default 3).
    raw_dict : dict containing raw (unscaled) order features.

    Returns
    -------
    list[str] of length <= k, ordered by |SHAP| descending.
    Example: ["Shipping mode: Second Class raises risk", "Scheduled shipping days: 2 raises risk"]
    """
    if isinstance(feature_values, dict) and raw_dict is None:
        raw_dict = feature_values
        feature_values = None

    labels, values, val_strs = group_shap_values(shap_row, feature_names, raw_dict=raw_dict)

    # Sort by absolute SHAP magnitude descending
    order = np.argsort(np.abs(values))[::-1]
    top_k = order[:k]

    reasons: list[str] = []
    for idx in top_k:
        lbl     = labels[idx]
        shap_v  = values[idx]
        v_str   = val_strs[idx]
        dir_str = "raises" if shap_v > 0 else "lowers"

        reasons.append(f"{lbl}: {v_str} {dir_str} risk")

    return reasons
