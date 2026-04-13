from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGETS = [
    "y_next_weight_gain_flag",
    "y_next_weight_loss_flag",
]


def log(msg: str) -> None:
    print(f"[daily-weight-dir-v2] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def df_to_markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows_"
    df2 = df.copy()
    df2.columns = [str(c) for c in df2.columns]
    for col in df2.columns:
        df2[col] = df2[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(df2.columns)
    rows = df2.values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + body_lines)


def make_one_hot_encoder() -> OneHotEncoder:
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "period_start" in out.columns:
        t = pd.to_datetime(out["period_start"], errors="coerce")
        out["period_year"] = t.dt.year.astype("float")
        out["period_month"] = t.dt.month.astype("float")
        out["period_quarter"] = t.dt.quarter.astype("float")
        out["period_weekofyear"] = t.dt.isocalendar().week.astype("float")
        out["period_dayofyear"] = t.dt.dayofyear.astype("float")
        angle = 2.0 * math.pi * ((out["period_dayofyear"].fillna(1.0) - 1.0) / 365.25)
        out["period_doy_sin"] = np.sin(angle)
        out["period_doy_cos"] = np.cos(angle)
        if t.notna().any():
            origin = t.min()
            out["period_days_since_start"] = (t - origin).dt.days.astype("float")
        out = out.drop(columns=["period_start"])
    return out


def prepare_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    exclude_cols = [c for c in df.columns if c.startswith("y_")]
    exclude_cols.extend([
        "next_period_id",
        "next_period_start",
        "split_suggested",
        "period_kind",
        "period_id",
        "day_id",
        "date",
    ])
    exclude_cols = [c for c in exclude_cols if c in df.columns]
    x = df.drop(columns=exclude_cols, errors="ignore").copy()
    x = add_time_features(x)
    for col in x.columns:
        if str(x[col].dtype) == "boolean":
            x[col] = x[col].astype("float")
    return x, exclude_cols


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def get_feature_names_from_pipe(pipe: Pipeline) -> List[str]:
    pre = pipe.named_steps["preprocessor"]
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        model = pipe.named_steps["model"]
        n = getattr(model, "n_features_in_", None)
        if n is None:
            return []
        return [f"feature_{i:04d}" for i in range(n)]


def classify_feature_group(col: str) -> str:
    c = col.lower()
    weather_keys = [
        "temperature", "apparent_temperature", "precip", "rain", "snow", "snowfall",
        "cloud", "wind", "gust", "pressure", "humidity", "daylight", "sunrise", "sunset",
        "uv", "weather", "is_day", "rain_streak", "freeze", "hot_streak",
    ]
    biology_keys = [
        "samsung", "heart", "hr", "stress", "sleep", "steps", "exercise", "active",
        "resting", "vo2", "oxygen", "resp", "calories_burned", "bmr", "weight",
        "noom_weight", "rest_calorie", "active_calorie",
    ]
    meal_keys = [
        "meal_", "noom_food", "restaurant_", "cuisine", "archetype", "protein", "carb",
        "fat", "fiber", "dessert", "beverage", "snack", "breakfast", "lunch", "dinner",
        "satiety", "indulgence", "comfort_food", "service_form", "prep_profile",
        "distinct_meal", "distinct_cuisines", "restaurant_specific", "food_", "starch",
        "veg", "sodium", "meal_event_count",
    ]
    temporal_keys = [
        "period_", "month", "quarter", "weekofyear", "dayofyear", "doy_", "season",
        "days_since_start", "year", "day_of_week", "is_weekend",
    ]
    if any(k in c for k in weather_keys):
        return "weather_daylight"
    if any(k in c for k in biology_keys):
        return "biology"
    if any(k in c for k in meal_keys):
        return "meals"
    if any(k in c for k in temporal_keys):
        return "temporal"
    return "other"


def build_ablation_sets(columns: List[str]) -> Dict[str, List[str]]:
    groups = {c: classify_feature_group(c) for c in columns}
    all_cols = list(columns)
    out = {"full": all_cols}
    for grp in ["meals", "biology", "weather_daylight", "temporal", "other"]:
        out[f"drop_{grp}"] = [c for c in all_cols if groups[c] != grp]
    for grp in ["meals", "biology", "weather_daylight", "temporal"]:
        grp_cols = [c for c in all_cols if groups[c] == grp]
        if grp_cols:
            out[f"{grp}_only"] = grp_cols
    return out


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "positive_rate_true": float(np.mean(y_true)),
    }
    if y_prob is not None and len(np.unique(y_true)) >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            out["brier"] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            pass
        try:
            out["log_loss"] = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)))
        except Exception:
            pass
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = np.digitize(y_prob, bins[1:-1], right=True)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = bucket == b
        if mask.sum() == 0:
            rows.append({
                "bin": b,
                "count": 0,
                "prob_mean": np.nan,
                "empirical_rate": np.nan,
                "abs_gap": np.nan,
            })
            continue
        p_mean = float(np.mean(y_prob[mask]))
        y_mean = float(np.mean(y_true[mask]))
        gap = abs(p_mean - y_mean)
        ece += gap * (mask.sum() / len(y_true))
        rows.append({
            "bin": b,
            "count": int(mask.sum()),
            "prob_mean": p_mean,
            "empirical_rate": y_mean,
            "abs_gap": gap,
        })
    return float(ece), pd.DataFrame(rows)


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "balanced_accuracy") -> Tuple[float, pd.DataFrame]:
    thresholds = np.unique(np.round(np.concatenate([np.linspace(0.05, 0.95, 19), y_prob]), 4))
    rows = []
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        if metric == "macro_f1":
            score = f1_score(y_true, pred, average="macro", zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, pred)
        rows.append({
            "threshold": float(t),
            "score": float(score),
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        })
        if score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t, pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def fit_model(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    if model_name == "logreg":
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "et":
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "dummy_majority":
        model = DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe = Pipeline([
        ("preprocessor", build_preprocessor(x_train)),
        ("model", model),
    ])
    pipe.fit(x_train, y_train)
    return pipe


def extract_feature_drivers(pipe: Pipeline, top_k: int = 30) -> Tuple[pd.DataFrame, str]:
    model = pipe.named_steps["model"]
    names = get_feature_names_from_pipe(pipe)

    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        if len(names) != len(vals):
            names = [f"feature_{i:04d}" for i in range(len(vals))]
        idx = np.argsort(vals)[::-1][:top_k]
        rows = []
        for rank, i in enumerate(idx, start=1):
            rows.append({
                "rank": rank,
                "feature": names[i],
                "score": float(vals[i]),
                "direction": "",
            })
        return pd.DataFrame(rows), "tree_importance"

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[0]
        if len(names) != len(coef):
            names = [f"feature_{i:04d}" for i in range(len(coef))]
        idx = np.argsort(np.abs(coef))[::-1][:top_k]
        rows = []
        for rank, i in enumerate(idx, start=1):
            rows.append({
                "rank": rank,
                "feature": names[i],
                "score": float(coef[i]),
                "direction": "positive" if coef[i] > 0 else "negative",
            })
        return pd.DataFrame(rows), "linear_coefficient"

    return pd.DataFrame(), "none"


def fit_calibrators_on_val(val_prob: np.ndarray, y_val: np.ndarray):
    if len(np.unique(y_val)) < 2:
        return None, None
    platt = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    platt.fit(val_prob.reshape(-1, 1), y_val)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_prob, y_val)
    return platt, isotonic


def apply_platt(model, raw_prob: np.ndarray) -> np.ndarray:
    return model.predict_proba(raw_prob.reshape(-1, 1))[:, 1]


def evaluate_probability_series(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    out = classification_metrics(y_true, pred, prob)
    ece, _ = expected_calibration_error(y_true, prob, n_bins=10)
    out["ece"] = ece
    out["threshold"] = float(threshold)
    return out


def make_period_index(df: pd.DataFrame) -> pd.Series:
    keys = df["period_id"].astype(str)
    if keys.duplicated().any():
        raise ValueError("Duplicate period_id values detected in transition matrix.")
    return pd.Series(df.index.to_numpy(), index=keys)


def run_target_analysis(
    df: pd.DataFrame,
    x_all: pd.DataFrame,
    target: str,
    target_report_dir: Path,
    target_model_dir: Path,
    out_dir: Path,
) -> Dict:
    summary_path = target_report_dir / "test_summary.json"
    pred_path = target_report_dir / "test_predictions.csv"
    comp_path = target_report_dir / "model_comparison.csv"
    meta_path = target_report_dir / "meta.json"
    if not summary_path.exists() or not pred_path.exists() or not comp_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing required files for {target}")

    summary = load_json(summary_path)
    meta = load_json(meta_path)
    pred_df = pd.read_csv(pred_path, low_memory=False)
    comp_df = pd.read_csv(comp_path, low_memory=False)

    best_model = summary["best_model"]
    saved_model_path = target_model_dir / f"{best_model}.joblib"
    if not saved_model_path.exists():
        raise FileNotFoundError(f"Missing saved model: {saved_model_path}")
    saved_pipe = joblib.load(saved_model_path)

    ensure_dir(out_dir)
    comp_df.to_csv(out_dir / "model_comparison.csv", index=False)
    pred_df.to_csv(out_dir / "trainer_test_predictions.csv", index=False)
    save_json(out_dir / "trainer_test_summary.json", summary)
    save_json(out_dir / "trainer_meta.json", meta)

    period_index = make_period_index(df)
    test_positions = period_index.loc[pred_df["period_id"].astype(str)].to_numpy()
    x_test = x_all.loc[test_positions].copy()
    y_test = pred_df["y_true_idx"].astype(int).to_numpy()

    # 1) Truth layer A: saved trainer model on exact held-out rows.
    p_saved = saved_pipe.predict_proba(x_test)[:, 1] if hasattr(saved_pipe, "predict_proba") else saved_pipe.predict(x_test).astype(float)
    pred_saved = (p_saved >= 0.5).astype(int)
    saved_metrics = classification_metrics(y_test, pred_saved, p_saved)
    saved_ece, saved_bins = expected_calibration_error(y_test, p_saved, n_bins=10)
    saved_metrics["ece"] = saved_ece

    if "y_pred_idx" in pred_df.columns:
        trainer_saved_prediction_match_rate = float(np.mean(pred_saved == pred_df["y_pred_idx"].astype(int).to_numpy()))
    else:
        trainer_saved_prediction_match_rate = np.nan

    # 2) Truth layer B: clean calibration pipeline: train only -> calibrate on val -> eval on test.
    train_df = df[df["split_suggested"] == "train"].copy()
    val_df = df[df["split_suggested"] == "val"].copy()
    test_df = df[df["split_suggested"] == "test"].copy()

    y_train = train_df[target].astype("boolean").astype("float")
    y_val = val_df[target].astype("boolean").astype("float")
    y_test_clean = test_df[target].astype("boolean").astype("float")

    finite_train = np.isfinite(y_train.to_numpy())
    finite_val = np.isfinite(y_val.to_numpy())
    finite_test = np.isfinite(y_test_clean.to_numpy())

    x_train = x_all.loc[train_df.index].copy().loc[finite_train]
    x_val = x_all.loc[val_df.index].copy().loc[finite_val]
    x_test_clean = x_all.loc[test_df.index].copy().loc[finite_test]

    y_train_arr = y_train.to_numpy()[finite_train].astype(int)
    y_val_arr = y_val.to_numpy()[finite_val].astype(int)
    y_test_arr = y_test_clean.to_numpy()[finite_test].astype(int)

    clean_pipe = fit_model(best_model, x_train, y_train_arr)
    p_val_raw = clean_pipe.predict_proba(x_val)[:, 1] if hasattr(clean_pipe, "predict_proba") else clean_pipe.predict(x_val).astype(float)
    p_test_raw = clean_pipe.predict_proba(x_test_clean)[:, 1] if hasattr(clean_pipe, "predict_proba") else clean_pipe.predict(x_test_clean).astype(float)

    raw_clean_default_metrics = evaluate_probability_series(y_test_arr, p_test_raw, threshold=0.5)

    thr_raw_bal, scan_raw_bal = choose_threshold(y_val_arr, p_val_raw, metric="balanced_accuracy")
    thr_raw_f1, scan_raw_f1 = choose_threshold(y_val_arr, p_val_raw, metric="macro_f1")
    raw_clean_bal_tuned_metrics = evaluate_probability_series(y_test_arr, p_test_raw, threshold=thr_raw_bal)
    raw_clean_f1_tuned_metrics = evaluate_probability_series(y_test_arr, p_test_raw, threshold=thr_raw_f1)

    platt_model, isotonic_model = fit_calibrators_on_val(p_val_raw, y_val_arr)
    p_test_platt = None
    p_test_iso = None
    platt_default_metrics = None
    iso_default_metrics = None
    platt_bal_tuned_metrics = None
    platt_f1_tuned_metrics = None
    iso_bal_tuned_metrics = None
    iso_f1_tuned_metrics = None
    thr_platt_bal = None
    thr_platt_f1 = None
    thr_iso_bal = None
    thr_iso_f1 = None

    if platt_model is not None:
        p_val_platt = apply_platt(platt_model, p_val_raw)
        p_test_platt = apply_platt(platt_model, p_test_raw)
        thr_platt_bal, scan_platt_bal = choose_threshold(y_val_arr, p_val_platt, metric="balanced_accuracy")
        thr_platt_f1, scan_platt_f1 = choose_threshold(y_val_arr, p_val_platt, metric="macro_f1")
        platt_default_metrics = evaluate_probability_series(y_test_arr, p_test_platt, threshold=0.5)
        platt_bal_tuned_metrics = evaluate_probability_series(y_test_arr, p_test_platt, threshold=thr_platt_bal)
        platt_f1_tuned_metrics = evaluate_probability_series(y_test_arr, p_test_platt, threshold=thr_platt_f1)
        scan_platt_bal.to_csv(out_dir / "threshold_scan_platt_balanced_accuracy.csv", index=False)
        scan_platt_f1.to_csv(out_dir / "threshold_scan_platt_macro_f1.csv", index=False)

    if isotonic_model is not None:
        p_val_iso = isotonic_model.predict(p_val_raw)
        p_test_iso = isotonic_model.predict(p_test_raw)
        thr_iso_bal, scan_iso_bal = choose_threshold(y_val_arr, p_val_iso, metric="balanced_accuracy")
        thr_iso_f1, scan_iso_f1 = choose_threshold(y_val_arr, p_val_iso, metric="macro_f1")
        iso_default_metrics = evaluate_probability_series(y_test_arr, p_test_iso, threshold=0.5)
        iso_bal_tuned_metrics = evaluate_probability_series(y_test_arr, p_test_iso, threshold=thr_iso_bal)
        iso_f1_tuned_metrics = evaluate_probability_series(y_test_arr, p_test_iso, threshold=thr_iso_f1)
        scan_iso_bal.to_csv(out_dir / "threshold_scan_isotonic_balanced_accuracy.csv", index=False)
        scan_iso_f1.to_csv(out_dir / "threshold_scan_isotonic_macro_f1.csv", index=False)

    scan_raw_bal.to_csv(out_dir / "threshold_scan_raw_balanced_accuracy.csv", index=False)
    scan_raw_f1.to_csv(out_dir / "threshold_scan_raw_macro_f1.csv", index=False)

    compare_rows = [
        {"series": "trainer_saved_model_default_0_5", **saved_metrics},
        {"series": "train_only_raw_default_0_5", **raw_clean_default_metrics},
        {"series": "train_only_raw_bal_tuned", **raw_clean_bal_tuned_metrics},
        {"series": "train_only_raw_f1_tuned", **raw_clean_f1_tuned_metrics},
    ]
    if platt_default_metrics is not None:
        compare_rows.extend([
            {"series": "train_only_platt_default_0_5", **platt_default_metrics},
            {"series": "train_only_platt_bal_tuned", **platt_bal_tuned_metrics},
            {"series": "train_only_platt_f1_tuned", **platt_f1_tuned_metrics},
        ])
    if iso_default_metrics is not None:
        compare_rows.extend([
            {"series": "train_only_isotonic_default_0_5", **iso_default_metrics},
            {"series": "train_only_isotonic_bal_tuned", **iso_bal_tuned_metrics},
            {"series": "train_only_isotonic_f1_tuned", **iso_f1_tuned_metrics},
        ])
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(out_dir / "calibration_compare.csv", index=False)

    bins_frames = [saved_bins.assign(series="trainer_saved_model_raw")]
    raw_clean_ece, raw_clean_bins = expected_calibration_error(y_test_arr, p_test_raw, n_bins=10)
    bins_frames.append(raw_clean_bins.assign(series="train_only_raw"))
    if p_test_platt is not None:
        _, platt_bins = expected_calibration_error(y_test_arr, p_test_platt, n_bins=10)
        bins_frames.append(platt_bins.assign(series="train_only_platt"))
    if p_test_iso is not None:
        _, iso_bins = expected_calibration_error(y_test_arr, p_test_iso, n_bins=10)
        bins_frames.append(iso_bins.assign(series="train_only_isotonic"))
    pd.concat(bins_frames, ignore_index=True).to_csv(out_dir / "calibration_bins_compare.csv", index=False)

    # Save scored test predictions for both evaluation paths.
    clean_period_ids = test_df.loc[finite_test, "period_id"].astype(str).tolist()
    clean_period_start = test_df.loc[finite_test, "period_start"].astype(str).tolist()
    clean_pred_df = pd.DataFrame({
        "period_id": clean_period_ids,
        "period_start": clean_period_start,
        "y_true": y_test_arr,
        "p_train_only_raw": p_test_raw,
        "y_pred_train_only_raw_0_5": (p_test_raw >= 0.5).astype(int),
        "y_pred_train_only_raw_bal_tuned": (p_test_raw >= thr_raw_bal).astype(int),
        "y_pred_train_only_raw_f1_tuned": (p_test_raw >= thr_raw_f1).astype(int),
    })
    if p_test_platt is not None:
        clean_pred_df["p_train_only_platt"] = p_test_platt
        clean_pred_df["y_pred_train_only_platt_0_5"] = (p_test_platt >= 0.5).astype(int)
        clean_pred_df["y_pred_train_only_platt_bal_tuned"] = (p_test_platt >= thr_platt_bal).astype(int)
        clean_pred_df["y_pred_train_only_platt_f1_tuned"] = (p_test_platt >= thr_platt_f1).astype(int)
    if p_test_iso is not None:
        clean_pred_df["p_train_only_isotonic"] = p_test_iso
        clean_pred_df["y_pred_train_only_isotonic_0_5"] = (p_test_iso >= 0.5).astype(int)
        clean_pred_df["y_pred_train_only_isotonic_bal_tuned"] = (p_test_iso >= thr_iso_bal).astype(int)
        clean_pred_df["y_pred_train_only_isotonic_f1_tuned"] = (p_test_iso >= thr_iso_f1).astype(int)
    clean_pred_df.to_csv(out_dir / "clean_pipeline_test_predictions.csv", index=False)

    trainer_scored = pred_df.copy()
    trainer_scored["p_saved_model_raw"] = p_saved
    trainer_scored["y_pred_saved_model_0_5"] = pred_saved
    trainer_scored.to_csv(out_dir / "saved_model_test_predictions_scored.csv", index=False)

    fi_df, fi_kind = extract_feature_drivers(saved_pipe, top_k=30)
    if not fi_df.empty:
        fi_df.to_csv(out_dir / "feature_driver_summary.csv", index=False)

    # Ablations with clean train->val->test path using logistic regression.
    ablation_rows = []
    ablation_sets = build_ablation_sets(list(x_all.columns))
    for ablation_name, cols in ablation_sets.items():
        if not cols:
            continue
        try:
            pipe_ab = fit_model("logreg", x_train[cols].copy(), y_train_arr)
            p_val_ab = pipe_ab.predict_proba(x_val[cols].copy())[:, 1]
            p_test_ab = pipe_ab.predict_proba(x_test_clean[cols].copy())[:, 1]
            t_ab, _ = choose_threshold(y_val_arr, p_val_ab, metric="balanced_accuracy")
            m_default = evaluate_probability_series(y_test_arr, p_test_ab, threshold=0.5)
            m_tuned = evaluate_probability_series(y_test_arr, p_test_ab, threshold=t_ab)
            ablation_rows.append({
                "ablation": ablation_name,
                "n_features": len(cols),
                "threshold_balanced_accuracy": t_ab,
                "default_accuracy": m_default["accuracy"],
                "default_balanced_accuracy": m_default["balanced_accuracy"],
                "default_macro_f1": m_default["macro_f1"],
                "default_roc_auc": m_default.get("roc_auc"),
                "default_brier": m_default.get("brier"),
                "tuned_accuracy": m_tuned["accuracy"],
                "tuned_balanced_accuracy": m_tuned["balanced_accuracy"],
                "tuned_macro_f1": m_tuned["macro_f1"],
                "tuned_roc_auc": m_tuned.get("roc_auc"),
                "tuned_brier": m_tuned.get("brier"),
            })
        except Exception:
            continue
    ablation_df = pd.DataFrame(ablation_rows)
    if not ablation_df.empty:
        ablation_df = ablation_df.sort_values("tuned_balanced_accuracy", ascending=False).reset_index(drop=True)
        ablation_df.to_csv(out_dir / "ablation_summary.csv", index=False)

    analysis_summary = {
        "target": target,
        "best_model": best_model,
        "best_model_kind": fi_kind,
        "trainer_metrics_from_training_script": summary["test_metrics"],
        "trainer_saved_model_default_0_5_metrics": saved_metrics,
        "trainer_saved_prediction_match_rate": trainer_saved_prediction_match_rate,
        "clean_pipeline_raw_default_0_5_metrics": raw_clean_default_metrics,
        "clean_pipeline_raw_bal_tuned_metrics": raw_clean_bal_tuned_metrics,
        "clean_pipeline_raw_f1_tuned_metrics": raw_clean_f1_tuned_metrics,
        "clean_pipeline_platt_default_0_5_metrics": platt_default_metrics,
        "clean_pipeline_platt_bal_tuned_metrics": platt_bal_tuned_metrics,
        "clean_pipeline_platt_f1_tuned_metrics": platt_f1_tuned_metrics,
        "clean_pipeline_isotonic_default_0_5_metrics": iso_default_metrics,
        "clean_pipeline_isotonic_bal_tuned_metrics": iso_bal_tuned_metrics,
        "clean_pipeline_isotonic_f1_tuned_metrics": iso_f1_tuned_metrics,
        "recommended_threshold_raw_balanced_accuracy": thr_raw_bal,
        "recommended_threshold_raw_macro_f1": thr_raw_f1,
        "recommended_threshold_platt_balanced_accuracy": thr_platt_bal,
        "recommended_threshold_platt_macro_f1": thr_platt_f1,
        "recommended_threshold_isotonic_balanced_accuracy": thr_iso_bal,
        "recommended_threshold_isotonic_macro_f1": thr_iso_f1,
        "test_rows_saved_model_path": int(len(pred_df)),
        "test_rows_clean_pipeline": int(len(clean_pred_df)),
        "val_rows_clean_pipeline": int(len(y_val_arr)),
    }
    save_json(out_dir / "analysis_summary.json", analysis_summary)

    return {
        "target": target,
        "best_model": best_model,
        "trainer_accuracy": summary["test_metrics"].get("accuracy"),
        "trainer_balanced_accuracy": summary["test_metrics"].get("balanced_accuracy"),
        "trainer_macro_f1": summary["test_metrics"].get("macro_f1"),
        "trainer_roc_auc": summary["test_metrics"].get("roc_auc"),
        "saved_model_raw_brier": saved_metrics.get("brier"),
        "saved_model_raw_ece": saved_metrics.get("ece"),
        "clean_raw_default_balanced_accuracy": raw_clean_default_metrics.get("balanced_accuracy"),
        "clean_raw_default_macro_f1": raw_clean_default_metrics.get("macro_f1"),
        "clean_raw_default_roc_auc": raw_clean_default_metrics.get("roc_auc"),
        "clean_raw_default_brier": raw_clean_default_metrics.get("brier"),
        "clean_raw_bal_tuned_balanced_accuracy": raw_clean_bal_tuned_metrics.get("balanced_accuracy"),
        "clean_raw_f1_tuned_macro_f1": raw_clean_f1_tuned_metrics.get("macro_f1"),
        "trainer_saved_prediction_match_rate": trainer_saved_prediction_match_rate,
    }


def build_report(overall_df: pd.DataFrame, compare_sections: Dict[str, pd.DataFrame]) -> str:
    lines = []
    lines.append("# Daily Weight Direction Analysis v2")
    lines.append("")
    lines.append("This v2 report fixes two issues from the first analyzer:")
    lines.append("- exact test-row alignment for the saved trainer model")
    lines.append("- clean calibration pipeline using train -> val -> test separation")
    lines.append("")
    if not overall_df.empty:
        lines.append("## Overall comparison")
        lines.append("")
        lines.append(df_to_markdown_table(overall_df))
        lines.append("")
    for target, df in compare_sections.items():
        lines.append(f"## Probability comparison: {target}")
        lines.append("")
        lines.append(df_to_markdown_table(df))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze daily weight-direction targets with corrected alignment and clean calibration.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--transition-dir", default="training/daily_transition", help="Relative path to daily transition dir.")
    parser.add_argument("--reports-root", default="reports/backtests/daily_transition", help="Relative target reports root.")
    parser.add_argument("--models-root", default="models/daily_transition", help="Relative trained models root.")
    parser.add_argument("--out-dir", default="reports/analysis/daily_weight_direction_v2", help="Relative output directory.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    transition_csv = project_root / args.transition_dir / "days_transition_matrix.csv"
    reports_root = project_root / args.reports_root
    models_root = project_root / args.models_root
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

    if not transition_csv.exists():
        raise FileNotFoundError(f"Missing transition matrix: {transition_csv}")

    df = pd.read_csv(transition_csv, low_memory=False)
    x_all, _ = prepare_feature_frame(df)

    overall_rows = []
    compare_sections = {}
    for target in TARGETS:
        log(f"Analyzing {target} ...")
        target_out = out_dir / target
        ensure_dir(target_out)
        res = run_target_analysis(
            df=df,
            x_all=x_all,
            target=target,
            target_report_dir=reports_root / target,
            target_model_dir=models_root / target,
            out_dir=target_out,
        )
        overall_rows.append(res)
        compare_path = target_out / "calibration_compare.csv"
        if compare_path.exists():
            compare_sections[target] = pd.read_csv(compare_path)

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / "overall_summary.csv", index=False)
    save_json(out_dir / "overall_summary.json", {
        "targets": TARGETS,
        "rows": len(overall_df),
    })
    (out_dir / "overall_report.md").write_text(
        build_report(overall_df, compare_sections),
        encoding="utf-8",
    )

    log("Done.")
    log(f"Wrote analysis to: {out_dir}")


if __name__ == "__main__":
    main()
