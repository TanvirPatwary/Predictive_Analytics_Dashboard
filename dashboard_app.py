# -*- coding: utf-8 -*-
"""
Predictive Analytics Dashboard (EDA + Predictions) — aligned with model_training.py

Artifacts expected in ./artifacts/:
  - model.pkl
  - metrics.csv
  - metrics_per_class.csv
  - cv_scores.csv
  - confusion_matrix.csv
  - decision_log.csv
  - label_map.csv

Data default:
  - ./data/cleaned_crime_data.csv (set DATA_DIR / ARTIFACTS_DIR env vars if needed)
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

# -----------------------------
# Page & Paths
# -----------------------------
st.set_page_config(page_title="Crime Analytics Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
DATA_DIR  = Path(os.getenv("DATA_DIR",        str(ROOT / "data")))

DEFAULT_MODEL            = ARTIFACTS / "model.pkl"
DEFAULT_METRICS          = ARTIFACTS / "metrics.csv"
DEFAULT_METRICS_PERCLASS = ARTIFACTS / "metrics_per_class.csv"
DEFAULT_CV_SCORES        = ARTIFACTS / "cv_scores.csv"
DEFAULT_CONFMAT          = ARTIFACTS / "confusion_matrix.csv"
DEFAULT_DECISION_LOG     = ARTIFACTS / "decision_log.csv"
DEFAULT_LABEL_MAP        = ARTIFACTS / "label_map.csv"
DEFAULT_DATA             = DATA_DIR / "cleaned_crime_data.csv"  # unchanged

# -----------------------------
# Column alternatives (aligned with training)
# -----------------------------
TARGET_ALTS     = ["Crime type", "CrimeType", "crime_type"]
MONTH_ALTS      = ["Month", "month"]
LAT_COL_ALTS    = ["Latitude", "lat", "LATITUDE"]
LON_COL_ALTS    = ["Longitude", "lon", "LONGITUDE"]
LSOA_CODE_ALTS  = ["LSOA code", "LSOA Code", "lsoa_code"]
LSOA_NAME_ALTS  = ["LSOA name", "LSOA Name", "lsoa_name"]
LOC_ALTS        = ["Location", "location"]

# Time-aware split constants (mirrors training script)
TRAIN_END  = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")

# -----------------------------
# Cacheable loaders & utils
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_or_path) -> pd.DataFrame:
    return pd.read_csv(file_or_path)

@st.cache_data(show_spinner=False)
def load_artifact_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if not path.exists():
        st.error(f"Missing model file: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        return None

def read_label_map(path: Path) -> Optional[Dict[int, str]]:
    lm = load_artifact_csv(path)
    if lm is None or lm.empty:
        return None
    if {"label", "class"}.issubset(lm.columns):
        key_col, val_col = "label", "class"
    elif {"class_index", "class_label"}.issubset(lm.columns):
        key_col, val_col = "class_index", "class_label"
    else:
        key_col, val_col = lm.columns[:2]
    return {int(k): str(v) for k, v in zip(lm[key_col], lm[val_col])}

def first_existing(df: pd.DataFrame, alts: List[str]) -> Optional[str]:
    for c in alts:
        if c in df.columns:
            return c
    return None

# -----------------------------
# Training-aligned feature engineering
# -----------------------------

def to_month_end(s: pd.Series) -> pd.Series:
    # identical to training: period M -> timestamp at month end
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("M")


def add_calendar_features(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    # identical mapping: season 1..4
    df["month_num"] = df[month_col].dt.month.astype(int)
    df["year"]      = df[month_col].dt.year.astype(int)
    df["quarter"]   = df[month_col].dt.quarter.astype(int)
    season_map = {12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4}
    df["season"]    = df["month_num"].map(season_map).astype(int)
    base_year = int(df["year"].min())
    df["year_idx"]  = (df["year"] - base_year).astype(int)
    return df


def add_geo_bins(df: pd.DataFrame, lat_col: Optional[str], lon_col: Optional[str], ndp: int = 3) -> pd.DataFrame:
    # identical to training: rounding to N dp, else zeros
    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        df["lat_bin"] = np.round(df[lat_col].astype(float), ndp)
        df["lon_bin"] = np.round(df[lon_col].astype(float), ndp)
    else:
        df["lat_bin"] = 0.0
        df["lon_bin"] = 0.0
    return df


def add_lag_density(df: pd.DataFrame, month_col: str, lsoa_code_col: str) -> pd.DataFrame:
    # identical to training: previous-month total crimes per LSOA (no crime-type key)
    grp = df.groupby([lsoa_code_col, month_col]).size().rename("lsoa_month_count").reset_index()
    grp = grp.sort_values(month_col)
    grp["lag1"] = grp.groupby(lsoa_code_col)["lsoa_month_count"].shift(1)
    df = df.merge(grp[[lsoa_code_col, month_col, "lag1"]], on=[lsoa_code_col, month_col], how="left")
    df["lag1"] = df["lag1"].fillna(0.0)
    return df


def frequency_encode_train_window(df_full: pd.DataFrame, lsoa_code_col: str, month_col: str) -> pd.Series:
    # fit on train window only (<= TRAIN_END), apply to all rows
    train_mask = df_full[month_col] <= TRAIN_END
    if not train_mask.any():
        return pd.Series(0.0, index=df_full.index)
    freq = df_full.loc[train_mask, lsoa_code_col].value_counts(normalize=True)
    return df_full[lsoa_code_col].map(freq).fillna(0.0)


def build_training_aligned_features(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Build features exactly like model_training.py:
      - Month coercion to month-end
      - Calendar features (season 1..4, year_idx from min year)
      - Geo rounding to 3 dp
      - Lag1: previous-month total per LSOA code
      - LSOA frequency fitted on train window only (<= 2024-12-31)

    Returns (X_all, month_col_name_used).
    """
    df = df_raw.copy()

    # Clean 'Location' for nicer EDA (not used in features)
    loc = first_existing(df, LOC_ALTS)
    if loc:
        df[loc] = (
            df[loc]
            .astype("string")
            .str.replace(r"^\s*On or Near\s+", "", regex=True)
            .str.strip()
            .str.lower()
        )

    # Identify columns
    month_col = first_existing(df, MONTH_ALTS)
    lsoa_code = first_existing(df, LSOA_CODE_ALTS)
    lat_col   = first_existing(df, LAT_COL_ALTS)
    lon_col   = first_existing(df, LON_COL_ALTS)

    # Month coercion to end of month (required)
    if month_col:
        df[month_col] = to_month_end(df[month_col])
    else:
        # If Month missing, create a dummy constant to keep features valid
        month_col = "Month_missing"
        df[month_col] = pd.Timestamp("1970-01-31")

    # Calendar features
    df = add_calendar_features(df, month_col)

    # Geo rounding to 3 dp
    df = add_geo_bins(df, lat_col, lon_col, ndp=3)

    # Lag density per LSOA code (requires LSOA code)
    if lsoa_code:
        df = add_lag_density(df, month_col, lsoa_code)
        df["lsoa_freq"] = frequency_encode_train_window(df, lsoa_code, month_col)
    else:
        st.warning("LSOA code column not found. Using zeros for lag1 and lsoa_freq (predictions may degrade).")
        df["lag1"] = 0.0
        df["lsoa_freq"] = 0.0

    features = [
        "month_num", "quarter", "season", "year", "year_idx",
        "lat_bin", "lon_bin", "lag1", "lsoa_freq"
    ]
    X = df[features].astype(float).copy()
    return X, month_col


def align_features_to_model(X_feat: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align columns to model expectation:
      - If model has 'feature_names_in_' use it (and create any missing as 0).
      - Otherwise, return as-is (Pipeline models may transform internally).
    """
    if hasattr(model, "feature_names_in_"):
        needed = list(model.feature_names_in_)
        missing = [c for c in needed if c not in X_feat.columns]
        if missing:
            for m in missing:
                X_feat[m] = 0.0
        X_feat = X_feat[needed]  # order matters
    return X_feat

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("Controls")
with st.sidebar.expander("Advanced paths (optional)", expanded=False):
    st.caption(f"Artifacts: {ARTIFACTS}")
    st.caption(f"Data dir:  {DATA_DIR}")

data_choice = st.sidebar.radio(
    "Data source",
    ["Use local (default)", "Upload CSV", "Type a path"],
    index=0,
)

uploaded = None
local_path = None
if data_choice == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
elif data_choice == "Type a path":
    local_path = st.sidebar.text_input("Local CSV path", value=str(DEFAULT_DATA))
else:
    local_path = str(DEFAULT_DATA)

show_eda     = st.sidebar.checkbox("Show EDA", value=True)
show_predict = st.sidebar.checkbox("Show Predictions", value=True)
show_metrics = st.sidebar.checkbox("Show Metrics & Logs", value=True)

# -----------------------------
# Load Data
# -----------------------------
with st.spinner("Loading data..."):
    df = None
    if uploaded is not None:
        df = load_csv(uploaded)
    elif local_path:
        try:
            df = load_csv(local_path)
        except Exception as e:
            st.error(f"Could not read local CSV at '{local_path}': {e}")
            st.stop()
    if df is None or len(df) == 0:
        st.error("No data provided or empty CSV. Upload a CSV or set a valid local path.")
        st.stop()

# -----------------------------
# Light schema awareness & filters (FIXED)
# -----------------------------
# Build full-dataset features first (so lag/frequency use full context)
X_all, month_col_used = build_training_aligned_features(df)

# Human columns
type_col = first_existing(df, TARGET_ALTS)
lsoa_code_col = first_existing(df, LSOA_CODE_ALTS)
area_name_col = first_existing(df, LSOA_NAME_ALTS)  # optional friendly display filter

# Always make masks as Series aligned to df.index
always_true = pd.Series(True, index=df.index)

# Month-range filter (uses month periods, not exact dates)
if month_col_used:
    month_periods = pd.to_datetime(df[month_col_used], errors="coerce").dt.to_period("M")
    month_opts = sorted(month_periods.dropna().unique().tolist())
    if month_opts:
        default_start, default_end = month_opts[0], month_opts[-1]
        m_start, m_end = st.sidebar.select_slider(
            "Month range",
            options=month_opts,
            value=(default_start, default_end),
        )
        mask_date = (month_periods >= m_start) & (month_periods <= m_end)
    else:
        mask_date = always_true
else:
    mask_date = always_true

# Crime type filter (normalized + stable)
if type_col and type_col in df:
    type_series = df[type_col].astype("string").str.strip().str.lower()
    types = sorted(type_series.dropna().unique().tolist())
    chosen_types = st.sidebar.multiselect(
        "Crime type",
        options=types,
        default=types,
        key="crime_type_ms"
    )
    mask_type = type_series.isin(chosen_types) if chosen_types else always_true
else:
    mask_type = always_true

# --- Removed: LSOA CODE (actual Area Code) UI ---
mask_code = always_true

# OPTIONAL: LSOA NAME (normalized + stable)
if area_name_col and area_name_col in df:
    name_series = df[area_name_col].astype("string").str.strip().str.lower()
    names = sorted(name_series.dropna().unique().tolist())
    chosen_names = st.sidebar.multiselect(
        "Area (LSOA name)",
        options=names,
        default=[],
        key="lsoa_name_ms"
    )
    mask_name = name_series.isin(chosen_names) if chosen_names else always_true
else:
    mask_name = always_true

# Final filtered frame (all masks are Series — no silent drops)
df_f = df[mask_date & mask_type & mask_code & mask_name]

# Optional clarity when nothing matches
if len(df_f) == 0:
    st.warning("No rows match the current filters. EDA will be empty and predictions are disabled until you broaden the filters.")

# -----------------------------
# Header
# -----------------------------
st.title("Predictive Analytics Dashboard")
st.caption("Descriptive Data Analysis and Predictions")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Rows (filtered)", f"{len(df_f):,}")
with c2:
    st.metric("Rows (total)", f"{len(df):,}")
with c3:
    uniq = df_f[type_col].nunique() if (type_col and type_col in df_f) else 0
    st.metric("Crime types in view", uniq)

# -----------------------------
# EDA
# -----------------------------
if show_eda:
    tab1, tab2, tab3 = st.tabs(["Distribution by Type", "Monthly Trend", "Quick Peek"])

    with tab1:
        if type_col and type_col in df_f and len(df_f):
            counts = df_f[type_col].value_counts().reset_index()
            counts.columns = ["Crime type", "Count"]
            fig = px.bar(counts, x="Crime type", y="Count", title="Crimes by Type (Filtered)")
            fig.update_layout(xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'Crime type' not found or no rows in view.")

    with tab2:
        if month_col_used and type_col and type_col in df_f and len(df_f):
            monthly = (
                df_f
                .assign(month=lambda x: pd.to_datetime(x[month_col_used], errors="coerce"))
                .groupby(["month", type_col]).size().reset_index(name="Count")
            )
            if len(monthly):
                fig = px.line(monthly, x="month", y="Count", color=type_col, title="Monthly Trend by Crime Type")
                fig.update_layout(xaxis_title="Month", yaxis_title="Count")
                fig.update_xaxes(tickformat="%Y-%m")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data to plot after filtering.")
        else:
            st.info("Need 'Month' and 'Crime type' to plot trend.")

    with tab3:
        #Show a single Month column (YYYY-MM) without dropping it accidentally
        if month_col_used and len(df_f):
            display_df = df_f.copy()
            display_df["Month"] = pd.to_datetime(display_df[month_col_used], errors="coerce").dt.strftime("%Y-%m")
            # Drop other raw month columns, but DO NOT drop the formatted "Month"
            raw_month_cols = [c for c in MONTH_ALTS if (c in display_df.columns and c != "Month")]
            display_df = display_df.drop(columns=raw_month_cols, errors="ignore")

            if "Month" in display_df.columns:
                ordered_cols = ["Month"] + [c for c in display_df.columns if c != "Month"]
                st.dataframe(display_df[ordered_cols].head(20), use_container_width=True)
            else:
                st.dataframe(display_df.head(20), use_container_width=True)

            csv_ready = df_f.copy()
            csv_ready["Month"] = pd.to_datetime(csv_ready[month_col_used], errors="coerce").dt.strftime("%Y-%m")
            csv_ready = csv_ready.drop(columns=[c for c in MONTH_ALTS if (c in csv_ready.columns and c != "Month")], errors="ignore")
            st.download_button("Download filtered CSV", data=csv_ready.to_csv(index=False).encode("utf-8"),
                               file_name="filtered.csv", mime="text/csv")
        elif len(df_f):
            st.dataframe(df_f.head(20), use_container_width=True)
            st.download_button("Download filtered CSV", data=df_f.to_csv(index=False).encode("utf-8"),
                               file_name="filtered.csv", mime="text/csv")
        else:
            st.info("No rows to preview under current filters.")

# -----------------------------
# Predictions (feature-aligned)
# -----------------------------
if show_predict:
    st.subheader("Prediction Demo")
    model = load_model(DEFAULT_MODEL)
    if model is None:
        st.warning("No usable model found. Train and export artifacts to ./artifacts/ first.")
    else:
        label_map = read_label_map(DEFAULT_LABEL_MAP)

        available = len(df_f)
        if available == 0:
            st.info("No rows match the current filters, so there’s nothing to score. Try widening the filters.")
        else:
            # Compute aligned features on the full dataset (training did this before split)
            X_full, _ = build_training_aligned_features(df)

            # UI: slider only when at least 2 rows are available; otherwise fix n_pred
            if available >= 2:
                left, right = st.columns([2, 1])
                with left:
                    n_max = min(50, available)
                    n_pred = st.slider(
                        "How many random rows to score?",
                        min_value=1,
                        max_value=n_max,
                        value=min(5, n_max),
                    )
                with right:
                    show_topk = st.slider("Show top‑k classes", 1, 5, 3)
            else:
                n_pred = 1
                show_topk = 3
                st.caption("Only 1 row in the current filter; scoring that row.")

            # Sample from filtered indices, then align features for those
            sample_raw = df_f.sample(n=n_pred, random_state=42)
            X_sample = X_full.loc[sample_raw.index]
            X_aligned = align_features_to_model(X_sample, model)

            try:
                proba = model.predict_proba(X_aligned)

                classes = getattr(model, "classes_", None)
                if classes is None:
                    raise AttributeError("Model lacks 'classes_'. Ensure the saved classifier supports predict_proba.")

                # Pretty class labels
                if label_map:
                    classes_named = [label_map.get(int(c), str(c)) for c in classes]
                else:
                    classes_named = [str(c) for c in classes]

                # Render top-k table
                def top_k_from_proba(proba_row: np.ndarray, classes: List[str], k: int = 3):
                    idx = np.argsort(proba_row)[::-1][:k]
                    return [(classes[i], float(proba_row[i])) for i in idx]

                rows = []
                for i, p in enumerate(proba):
                    topk = top_k_from_proba(p, classes_named, k=show_topk)
                    rows.append({
                        "row_id": sample_raw.index[i],
                        "prediction": topk[0][0],
                        **{f"top{j+1}": f"{lab} ({prob:.2%})" for j, (lab, prob) in enumerate(topk)}
                    })

                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                with st.expander("What features were used for scoring?"):
                    if hasattr(model, "feature_names_in_"):
                        st.write("Model expects these feature columns (post‑engineering):")
                        st.code(", ".join(list(model.feature_names_in_)) or "(none)")
                    else:
                        st.write("Model is likely a Pipeline; it transformed features internally.")
                    st.write("Engineered features available in app (training‑aligned):")
                    st.code(", ".join(list(X_aligned.columns)))

            except Exception as e:
                st.error(
                    "Model scoring failed. Ensure your saved model expects the training‑aligned feature set "
                    "constructed in this app.\n\nDetails: {}".format(e)
                )

# -----------------------------
# Metrics & Decision Log
# -----------------------------
if show_metrics:
    st.subheader("Model Performance & Decision Log")

    met = load_artifact_csv(DEFAULT_METRICS)
    if met is not None and len(met):
        st.markdown("**Key metrics (overall)**")
        st.dataframe(met, use_container_width=True)
    else:
        st.info("metrics.csv not found.")

    met_pc = load_artifact_csv(DEFAULT_METRICS_PERCLASS)
    if met_pc is not None and len(met_pc):
        st.markdown("**Per‑class metrics**")
        st.dataframe(met_pc, use_container_width=True)

    cv = load_artifact_csv(DEFAULT_CV_SCORES)
    if cv is not None and len(cv):
        st.markdown("**Time‑series CV scores (training window)**")
        st.dataframe(cv, use_container_width=True)

    cm = load_artifact_csv(DEFAULT_CONFMAT)
    if cm is not None and len(cm):
        st.markdown("**Confusion matrix (counts)**")
        st.dataframe(cm, use_container_width=True)

    log = load_artifact_csv(DEFAULT_DECISION_LOG)
    if log is not None and len(log):
        st.markdown("**Decision Log**")
        st.dataframe(log, use_container_width=True)
        st.download_button(
            "Download decision_log.csv",
            data=log.to_csv(index=False).encode("utf-8"),
            file_name="decision_log.csv",
            mime="text/csv",
        )
    else:
        st.info("decision_log.csv not found.")
