# src/model_training.py
# Predicting Crime Type (multiclass) with time-aware validation
# Artifacts -> ../artifacts/: model.pkl, metrics.csv, metrics_per_class.csv,
# cv_scores.csv, confusion_matrix.csv, decision_log.csv, label_map.csv

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, top_k_accuracy_score
)
import joblib

from xgboost import XGBClassifier


# -----------------------------
# Paths / Config
# -----------------------------
ARTIFACTS = Path("../artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Column name fallbacks (handles slight naming differences)
TARGET_COL_ALTS = ["Crime type", "CrimeType", "crime_type"]
MONTH_COL_ALTS  = ["Month", "month"]
LSOA_COL_ALTS   = ["LSOA code", "LSOA Code", "lsoa_code"]
LAT_COL_ALTS    = ["Latitude", "lat", "LATITUDE"]
LON_COL_ALTS    = ["Longitude", "lon", "LONGITUDE"]

# Time-aware split (edit if your dates differ)
TRAIN_END  = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")


# -----------------------------
# Helpers
# -----------------------------
def get_col(df: pd.DataFrame, alts):
    for c in alts:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns found: {alts}")

def load_data() -> pd.DataFrame:
    ROOT = Path(__file__).resolve().parent.parent   # from src/ up to repo root
    DATA_DIR = ROOT / "data"
    return pd.read_csv(DATA_DIR / "cleaned_crime_data.csv")


def to_month(dt_series: pd.Series) -> pd.Series:
    s = pd.to_datetime(dt_series, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("M")

def add_calendar_features(df, month_col):
    df["month_num"] = df[month_col].dt.month.astype(int)
    df["year"]      = df[month_col].dt.year.astype(int)
    df["quarter"]   = df[month_col].dt.quarter.astype(int)
    season_map = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}
    df["season"]    = df["month_num"].map(season_map).astype(int)
    base_year = int(df["year"].min())
    df["year_idx"]  = (df["year"] - base_year).astype(int)
    return df

def add_geo_bins(df, lat_col, lon_col, ndp=3):
    if lat_col in df.columns and lon_col in df.columns:
        df["lat_bin"] = np.round(df[lat_col].astype(float), ndp)
        df["lon_bin"] = np.round(df[lon_col].astype(float), ndp)
    else:
        df["lat_bin"] = 0.0
        df["lon_bin"] = 0.0
    return df

def add_lag_density(df, month_col, lsoa_col):
    # Crimes per LSOA per month; add lag-1 as contextual density
    grp = df.groupby([lsoa_col, month_col]).size().rename("lsoa_month_count").reset_index()
    grp["lag1"] = grp.sort_values(month_col).groupby(lsoa_col)["lsoa_month_count"].shift(1)
    df = df.merge(grp[[lsoa_col, month_col, "lag1"]], on=[lsoa_col, month_col], how="left")
    df["lag1"] = df["lag1"].fillna(0.0)
    return df

def frequency_encode(train_df, full_df, key_col, new_col):
    freq = train_df[key_col].value_counts(normalize=True)
    full_df[new_col] = full_df[key_col].map(freq).fillna(0.0)
    return full_df

def build_feature_frame(df, month_col, target_col, lsoa_col, lat_col, lon_col, train_mask):
    df = add_calendar_features(df, month_col)
    df = add_geo_bins(df, lat_col, lon_col, ndp=3)
    df = add_lag_density(df, month_col, lsoa_col)  # uses only past months implicitly
    # Frequency-encode LSOA using training window ONLY (avoid leakage)
    df = frequency_encode(df.loc[train_mask], df, lsoa_col, "lsoa_freq")
    features = [
        "month_num", "quarter", "season", "year", "year_idx",
        "lat_bin", "lon_bin", "lag1", "lsoa_freq"
    ]
    X = df[features].astype(float).copy()
    y = df[target_col].copy()
    return X, y, features


# -----------------------------
# Main
# -----------------------------
def main():
    # Load & identify columns
    df = load_data().copy()
    target_col = get_col(df, TARGET_COL_ALTS)
    month_col  = get_col(df, MONTH_COL_ALTS)
    lsoa_col   = get_col(df, LSOA_COL_ALTS)
    lat_col    = get_col(df, LAT_COL_ALTS) if any(c in df.columns for c in LAT_COL_ALTS) else ""
    lon_col    = get_col(df, LON_COL_ALTS) if any(c in df.columns for c in LON_COL_ALTS) else ""

    # Basic filtering + month parsing
    df = df[~df[target_col].isna()].copy()
    df[month_col] = to_month(df[month_col])

    # Time-aware split
    train_mask = df[month_col] <= TRAIN_END
    test_mask  = df[month_col] >= TEST_START
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Train or Test split is empty. Adjust TRAIN_END/TEST_START or check Month values.")

    # Features (fit encodings on train only)
    X_all, y_all_raw, feature_list = build_feature_frame(
        df, month_col, target_col, lsoa_col, lat_col, lon_col, train_mask
    )

    # --- Label encoding for stable classes across folds/models ---
    le = LabelEncoder()
    le.fit(y_all_raw.loc[train_mask])              # fit on training labels only
    y_all = pd.Series(le.transform(y_all_raw),     # integer codes 0..K-1
                      index=y_all_raw.index, name=target_col)

    # Split
    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_test,  y_test  = X_all.loc[test_mask],  y_all.loc[test_mask]

    # ------------------ Models ------------------
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto"))
    ])
    # Stepped-up RF capacity (heavier but still under GitHub/Cloud-friendly sizes when saved with compression)
    rf = RandomForestClassifier(
        n_estimators=280,
        max_depth=13,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, tree_method="hist"
    )

    models = [("LogReg", logreg), ("RandomForest", rf), ("XGBoost", xgb)]

    # --------- Expanding TimeSeries CV on training ---------
    tss = TimeSeriesSplit(n_splits=3)
    cv_rows = []
    for model_name, model in models:
        fold_id = 0
        for tr_idx, val_idx in tss.split(X_train):
            fold_id += 1
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            preds_val = model.predict(X_val)
            f1 = f1_score(y_val, preds_val, average="macro")
            acc = accuracy_score(y_val, preds_val)
            cv_rows.append({"model": model_name, "fold": fold_id, "macro_f1": f1, "accuracy": acc})
    pd.DataFrame(cv_rows).to_csv(ARTIFACTS / "cv_scores.csv", index=False)

    # --------------- Fit on full training ---------------
    for _, model in models:
        model.fit(X_train, y_train)

    # --------------- Evaluate on held-out test ---------------
    rows_overall, rows_per_class, confmats = [], [], {}
    for model_name, model in models:
        preds = model.predict(X_test)

        # top-2 accuracy in encoded space (0..K-1)
        try:
            proba = model.predict_proba(X_test)
            top2 = top_k_accuracy_score(y_test, proba, k=2)
        except Exception:
            top2 = np.nan

        # Map to human-readable labels for reporting
        preds_lbl   = le.inverse_transform(preds)
        y_test_lbl  = le.inverse_transform(y_test)

        macro_f1 = f1_score(y_test_lbl, preds_lbl, average="macro")
        acc      = accuracy_score(y_test_lbl, preds_lbl)

        rows_overall += [
            {"model": model_name, "metric": "accuracy",       "value": float(acc)},
            {"model": model_name, "metric": "macro_f1",       "value": float(macro_f1)},
            {"model": model_name, "metric": "top2_accuracy",  "value": float(top2) if not np.isnan(top2) else np.nan},
        ]

        rep = classification_report(y_test_lbl, preds_lbl, output_dict=True, zero_division=0)
        for cls, d in rep.items():
            if cls in ["accuracy", "macro avg", "weighted avg"]:
                continue
            rows_per_class.append({
                "model": model_name, "class": cls,
                "precision": d.get("precision", 0.0),
                "recall":    d.get("recall", 0.0),
                "f1":        d.get("f1-score", 0.0),
                "support":   d.get("support", 0)
            })

        cm = confusion_matrix(y_test_lbl, preds_lbl, labels=le.classes_)
        confmats[model_name] = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    # --------------- Select & persist best model ---------------
    best_model_name = (
        pd.DataFrame(rows_overall)
        .query("metric == 'macro_f1'")
        .sort_values("value", ascending=False)
        .iloc[0]["model"]
    )
    final_model = dict(models)[best_model_name]
    # Save compressed to drastically reduce file size
    joblib.dump(final_model, ARTIFACTS / "model.pkl", compress=4)

    # Save metrics & confusion matrix
    pd.DataFrame(rows_overall).to_csv(ARTIFACTS / "metrics.csv", index=False)
    pd.DataFrame(rows_per_class).to_csv(ARTIFACTS / "metrics_per_class.csv", index=False)
    confmats[best_model_name].to_csv(ARTIFACTS / "confusion_matrix.csv")

    # Save label map for dashboard
    pd.DataFrame({"class_index": np.arange(len(le.classes_)), "class_label": le.classes_}) \
        .to_csv(ARTIFACTS / "label_map.csv", index=False)

    # Decision log (plain English)
    decisions = [
        ("split_strategy", "Used time-aware split: train <= 2024-12, test >= 2025-01 to prevent leakage."),
        ("cv", "Applied expanding TimeSeriesSplit (n_splits=3) on training window."),
        ("encoding_lsoa", "Frequency-encoded LSOA using train window only to avoid leakage."),
        ("geo_binning", "Rounded latitude/longitude to 3dp to reduce overfitting/noise."),
        ("context_lag", "Added lag-1 LSOA-month crime count as contextual density."),
        ("imbalance", "Used class_weight to handle skew across crime types."),
        ("primary_metric", "Macro-F1 chosen to value minority classes."),
        ("model_choice", f"Selected {best_model_name} by highest Macro-F1 on held-out Jan–May 2025 test.")
    ]
    pd.DataFrame(decisions, columns=["decision", "rationale"]).to_csv(
        ARTIFACTS / "decision_log.csv", index=False
    )

    print(f"Training complete. Final model: {best_model_name}")
    print(f"Artifacts written to: {ARTIFACTS.resolve()}")


if __name__ == "__main__":
    main()
