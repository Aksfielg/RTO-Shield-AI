"""
ui/data.py
==========
Cached data access for the dashboard.  Every number shown in the app comes from
outputs/final_metrics.json and models/thresholds.json (``fm()`` / ``thr()``) or
from the real orders / model artefacts loaded here.  No synthetic data.

Caching
-------
* ``st.cache_resource`` : model, preprocessor, calibrator, SHAP explainer (shared
  with predict.py so they are loaded once per process)
* ``st.cache_data``     : JSON, parquet, test-set scores, SHAP sample, option lists
"""

from __future__ import annotations

import re

import json
import sys
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import streamlit as st

from config import (  # noqa: E402
    CATEGORICAL_FEATURES, MODELS_DIR, NUMERIC_FEATURES, OUTPUTS_DIR, PROCESSED_DIR,
    SEED, TARGET_COL, TRAIN_FRAC, VAL_FRAC,
)
from features import _select_feature_cols  # noqa: E402
from reasons import group_shap_values  # noqa: E402
import predict  # noqa: E402

FM_PATH = OUTPUTS_DIR / "final_metrics.json"
THR_PATH = MODELS_DIR / "thresholds.json"

MODEL_LABELS = {
    "baseline_lr": "Baseline LR (shipping+days)", "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM (default)",
}


# ---------------------------------------------------------------------------
# JSON sources of truth
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fm() -> dict:
    """outputs/final_metrics.json (re-read automatically when the file changes)."""
    return _read_json(str(FM_PATH), FM_PATH.stat().st_mtime)


def thr() -> dict:
    """models/thresholds.json."""
    return _read_json(str(THR_PATH), THR_PATH.stat().st_mtime)


def mode_stats(name: str) -> dict:
    return next(r for r in fm()["per_mode"] if r["mode"] == name)


# ---------------------------------------------------------------------------
# Model artefacts (shared with predict.py)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def artifacts() -> dict:
    predict._load_artifacts()
    c = predict._CACHE
    return {
        "model": c["model"], "prep": c["prep"], "calibrator": c["cal"], "thresholds": thr(),
        "feat_names": c["feat_names"], "explainer": c["explainer"],
    }


@st.cache_resource(show_spinner=False)
def start_warmup() -> threading.Thread:
    """
    Warm the model + SHAP explainer in a background thread (once per server process) so the first visit to a
    scoring page is instant.  The thread only touches predict.py (no Streamlit calls).
    """
    t = threading.Thread(target=predict._load_artifacts, name="rto-warmup", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Real orders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading orders...")
def datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all orders, chronological test slice) - same split as the pipeline."""
    df = pd.read_parquet(PROCESSED_DIR / "orders.parquet").sort_values("order_date").reset_index(drop=True)
    test_start = int(np.floor(len(df) * (TRAIN_FRAC + VAL_FRAC)))
    return df, df.iloc[test_start:].copy().reset_index(drop=True)


@st.cache_data(show_spinner="Scoring the holdout set...")
def test_scores() -> dict:
    """Raw (ranking) and calibrated (decision) scores for the whole test set, computed once."""
    art = artifacts()
    _, test_df = datasets()
    X = predict._transform(_select_feature_cols(test_df, CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    raw = art["model"].predict_proba(X)[:, 1]
    cal = art["calibrator"].predict(raw)
    return {"raw": raw, "cal": cal, "y": test_df[TARGET_COL].values}


@st.cache_data(show_spinner=False)
def option_lists() -> dict:
    """Selectable values + scheduled days per shipping mode, all taken from the real data."""
    df, _ = datasets()
    days = df.groupby("Shipping_Mode")["Days_for_shipment_scheduled"].agg(lambda s: int(s.mode().iloc[0])).to_dict()
    return {
        "modes": sorted(days, key=lambda m: days[m]),
        "mode_days": days,
        "types": sorted(df["Type"].dropna().unique().tolist()),
        "markets": sorted(df["Market"].dropna().unique().tolist()),
        "departments": sorted(df["Department_Name"].dropna().unique().tolist()),
        "segments": sorted(df["Customer_Segment"].dropna().unique().tolist()),
        "value_range": (float(df["order_value"].min()), float(df["order_value"].max())),
        "items_range": (int(df["n_items"].min()), int(df["n_items"].max())),
    }


def _py(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


@st.cache_data(show_spinner=False)
def representative_orders() -> dict:
    """
    One REAL holdout order per risk tier: the order whose calibrated score is closest to the median score of
    that tier.  Used as presets (no invented order values).
    """
    _, test_df = datasets()
    sc = test_scores()
    cutoffs = thr()["tier_cutoffs"]
    tiers = np.where(sc["cal"] < cutoffs["low_max"], "Low", np.where(sc["cal"] < cutoffs["high_min"], "Medium", "High"))
    out = {}
    for t in ["Low", "Medium", "High"]:
        idx = np.where(tiers == t)[0]
        if len(idx) == 0:
            continue
        med = np.median(sc["cal"][idx])
        pick = int(idx[np.argmin(np.abs(sc["cal"][idx] - med))])
        row = {k: _py(v) for k, v in test_df.iloc[pick].to_dict().items()}
        row["_actual"] = int(sc["y"][pick])
        out[t] = row
    return out


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
def explain_order(order_dict: dict) -> dict:
    """Grouped SHAP attribution for one order (parent features, raw values in business units)."""
    art = artifacts()
    df_row = predict._dict_to_df(order_dict)
    X_enc = predict._transform(_select_feature_cols(df_row, CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    sv = art["explainer"](X_enc)
    labels, values, val_strs = group_shap_values(predict._shap_values(X_enc)[0], art["feat_names"], order_dict)
    val_strs = [en_country(v) if lbl == "Country" else v for lbl, v in zip(labels, val_strs)]      # display only
    top = int(np.argmax(np.abs(values)))
    return {"labels": labels, "values": values, "val_strs": val_strs,
            "base_value": float(np.ravel(sv.base_values)[0]), "top_driver": labels[top],
            "driver_raises": bool(values[top] > 0)}


@st.cache_data(show_spinner="Computing SHAP sample...")
def shap_sample() -> dict:
    """SHAP values for the same 500-row test sample used in the finalized metrics."""
    art = artifacts()
    _, test_df = datasets()
    X_te = predict._transform(_select_feature_cols(test_df, CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    n = fm()["shap"]["sample_n"]
    idx = np.sort(np.random.default_rng(SEED).choice(len(X_te), size=min(n, len(X_te)), replace=False))
    sv = predict._shap_values(X_te[idx])
    return {"sv": sv, "X": X_te[idx], "feat_names": art["feat_names"], "rows": test_df.iloc[idx].reset_index(drop=True)}


@st.cache_data(show_spinner="Computing SHAP importance...")
def grouped_importance() -> list[dict]:
    """Mean |SHAP| per PARENT feature (one-hot columns grouped, readable names), computed from the SHAP sample."""
    s = shap_sample()
    acc: dict[str, float] = {}
    for r in range(len(s["sv"])):
        labels, vals, _ = group_shap_values(s["sv"][r], s["feat_names"], {})
        for lb, v in zip(labels, vals):
            acc[lb] = acc.get(lb, 0.0) + abs(float(v))
    n, tot = len(s["sv"]), sum(acc.values())
    return sorted(({"factor": k, "mean_abs_shap": v / n, "share": v / tot} for k, v in acc.items()), key=lambda d: -d["mean_abs_shap"])


@st.cache_data(show_spinner=False)
def demo_scores() -> pd.DataFrame | None:
    p = OUTPUTS_DIR / "demo_scores.csv"
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
# DISPLAY-ONLY English country names.  DataCo stores Spanish names (and the parquet has lost the accented letters, e.g.
# "Afganist\ufffdn"); model inputs, parquet values and downloadable CSVs keep the ORIGINAL strings.  The lookup key drops every
# character outside [a-z0-9 ()-] so a corrupted ("Afganist\ufffdn") and a proper ("Afganist\u00e1n") spelling hit the same entry.
# ---------------------------------------------------------------------------
_COUNTRY_PAIRS = "Afganistán=Afghanistan|Albania=Albania|Alemania=Germany|Angola=Angola|Arabia Saudí=Saudi Arabia|Argelia=Algeria|Argentina=Argentina|Armenia=Armenia|Australia=Australia|Austria=Austria|Azerbaiyán=Azerbaijan|Bangladés=Bangladesh|Barbados=Barbados|Baréin=Bahrain|Belice=Belize|Benín=Benin|Bielorrusia=Belarus|Bolivia=Bolivia|Bosnia y Herzegovina=Bosnia and Herzegovina|Botsuana=Botswana|Brasil=Brazil|Bulgaria=Bulgaria|Burkina Faso=Burkina Faso|Burundi=Burundi|Bután=Bhutan|Bélgica=Belgium|Camboya=Cambodia|Camerún=Cameroon|Canada=Canada|Chad=Chad|Chile=Chile|China=China|Chipre=Cyprus|Colombia=Colombia|Corea del Sur=South Korea|Costa Rica=Costa Rica|Costa de Marfil=Ivory Coast|Croacia=Croatia|Cuba=Cuba|Dinamarca=Denmark|Ecuador=Ecuador|Egipto=Egypt|El Salvador=El Salvador|Emiratos Árabes Unidos=United Arab Emirates|Eritrea=Eritrea|Eslovaquia=Slovakia|Eslovenia=Slovenia|España=Spain|Estados Unidos=United States|Estonia=Estonia|Etiopía=Ethiopia|Filipinas=Philippines|Finlandia=Finland|Francia=France|Gabón=Gabon|Georgia=Georgia|Ghana=Ghana|Grecia=Greece|Guadalupe=Guadeloupe|Guatemala=Guatemala|Guayana Francesa=French Guiana|Guinea=Guinea|Guinea Ecuatorial=Equatorial Guinea|Guinea-Bissau=Guinea-Bissau|Guyana=Guyana|Haití=Haiti|Honduras=Honduras|Hong Kong=Hong Kong|Hungría=Hungary|India=India|Indonesia=Indonesia|Irak=Iraq|Irlanda=Ireland|Irán=Iran|Israel=Israel|Italia=Italy|Jamaica=Jamaica|Japón=Japan|Jordania=Jordan|Kazajistán=Kazakhstan|Kenia=Kenya|Kirguistán=Kyrgyzstan|Kuwait=Kuwait|Laos=Laos|Lesoto=Lesotho|Liberia=Liberia|Libia=Libya|Lituania=Lithuania|Luxemburgo=Luxembourg|Líbano=Lebanon|Macedonia=North Macedonia|Madagascar=Madagascar|Malasia=Malaysia|Mali=Mali|Marruecos=Morocco|Martinica=Martinique|Mauritania=Mauritania|Moldavia=Moldova|Mongolia=Mongolia|Montenegro=Montenegro|Mozambique=Mozambique|Myanmar (Birmania)=Myanmar|México=Mexico|Namibia=Namibia|Nepal=Nepal|Nicaragua=Nicaragua|Nigeria=Nigeria|Noruega=Norway|Nueva Zelanda=New Zealand|Níger=Niger|Omán=Oman|Pakistán=Pakistan|Panamá=Panama|Papúa Nueva Guinea=Papua New Guinea|Paraguay=Paraguay|Países Bajos=Netherlands|Perú=Peru|Polonia=Poland|Portugal=Portugal|Qatar=Qatar|Reino Unido=United Kingdom|República Centroafricana=Central African Republic|República Checa=Czech Republic|República Democrática del Congo=Democratic Republic of the Congo|República Dominicana=Dominican Republic|República de Gambia=Gambia|República del Congo=Republic of the Congo|Ruanda=Rwanda|Rumania=Romania|Rusia=Russia|Senegal=Senegal|Serbia=Serbia|Sierra Leona=Sierra Leone|Singapur=Singapore|Siria=Syria|Somalia=Somalia|Sri Lanka=Sri Lanka|Suazilandia=Eswatini|SudAfrica=South Africa|Sudán=Sudan|Sudán del Sur=South Sudan|Suecia=Sweden|Suiza=Switzerland|Surinam=Suriname|Sáhara Occidental=Western Sahara|Tailandia=Thailand|Taiwán=Taiwan|Tanzania=Tanzania|Tayikistán=Tajikistan|Togo=Togo|Trinidad y Tobago=Trinidad and Tobago|Turkmenistán=Turkmenistan|Turquía=Turkey|Túnez=Tunisia|Ucrania=Ukraine|Uganda=Uganda|Uruguay=Uruguay|Uzbekistán=Uzbekistan|Venezuela=Venezuela|Vietnam=Vietnam|Yemen=Yemen|Yibuti=Djibouti|Zambia=Zambia|Zimbabue=Zimbabwe"


def _country_key(name: str) -> str:
    return re.sub(r"[^a-z0-9 ()-]", "", str(name).lower())


COUNTRY_EN: dict[str, str] = {_country_key(es): en for es, en in (p.split("=", 1) for p in _COUNTRY_PAIRS.split("|"))}


def en_country(name):
    """English display name for a DataCo country (unchanged when it is already English or unknown)."""
    return COUNTRY_EN.get(_country_key(name), name) if isinstance(name, str) else name


_COUNTRY_REASON = re.compile(r"(Country: )(.+?)( (?:raises|lowers) risk)")


def en_text(text):
    """Translate a country name inside a reason string ('Country: Reino Unido raises risk') for display."""
    return _COUNTRY_REASON.sub(lambda m: m.group(1) + en_country(m.group(2)) + m.group(3), text) if isinstance(text, str) else text
