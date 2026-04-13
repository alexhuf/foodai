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
TARGET_SPACE = "weeks"
TARGET_NAME = "y_next_weight_gain_flag"


def log(msg: str) -> None:
    print(f"[weight-gain-cal] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_one_hot_encoder() -> OneHotEncoder:
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "period_start" in out.columns:
        t = pd.to_datetime(out["period_start"], errors="coerce")
        out["period_year"] = t.dt.year.astype("float")
        out["period_month"] = t.dt.month.astype("float")
        out["period_quarter"] = t.dt.quarter.astype("float")
        out["period_weekofyear"] = t.dt.isocalendar().week.astype("float")
        out["period_dayofyear"] = t.dt.dayofyear.astype("float")
        angle = 2.0 * np.pi * ((out["period_dayofyear"].fillna(1.0) - 1.0) / 365.25)
        out["period_doy_sin"] = np.sin(angle)
        out["period_doy_cos"] = np.cos(angle)
        if t.notna().any():
            origin = t.min()
            out["period_days_since_start"] = (t - origin).dt.days.astype("float")
        out = out.drop(columns=["period_start"])
    return out


def prepare_feature_frame(df: pd.DataFrame):
    exclude_cols = [c for c in df.columns if c.startswith("y_")]
    exclude_cols.extend([
        "next_period_id",
        "next_period_start",
        "split_suggested",
        "period_kind",
        "period_id",
        "week_id",
        "weekend_id",
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


def build_windows(n: int, min_train: int, val_size: int, test_size: int, step: int):
    windows = []
    train_end = min_train
    fold = 1
    while True:
        val_start = train_end
        val_end = val_start + val_size
        test_start = val_end
        test_end = test_start + test_size
        if test_end > n:
            break
        windows.append({
            "fold": fold,
            "train_idx": list(range(0, train_end)),
            "val_idx": list(range(val_start, val_end)),
            "test_idx": list(range(test_start, test_end)),
        })
        train_end += step
        fold += 1
    return windows


def encode_binary_target(y: pd.Series) -> pd.Series:
    if str(y.dtype) == "boolean":
        return y.astype("float")
    y2 = y.copy()
    if y2.dropna().isin([True, False, 0, 1, "0", "1", "True", "False", "true", "false"]).all():
        mapping = {
            True: 1.0, False: 0.0,
            1: 1.0, 0: 0.0,
            "1": 1.0, "0": 0.0,
            "True": 1.0, "False": 0.0,
            "true": 1.0, "false": 0.0,
        }
        return y2.map(mapping).astype("float")
    return pd.to_numeric(y2, errors="coerce").astype("float")


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


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
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


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "balanced_accuracy"):
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


def model_family() -> Dict[str, object]:
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def fit_model(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(x_train)),
        ("model", model_family()[model_name]),
    ])
    pipe.fit(x_train, y_train)
    return pipe


def evaluate_pipe(pipe: Pipeline, x: pd.DataFrame, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    if hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(x)[:, 1]
    else:
        pred = pipe.predict(x)
        prob = pred.astype(float)
    pred = (prob >= threshold).astype(int)
    out = classification_metrics(y, pred, prob)
    ece, _ = expected_calibration_error(y, prob, n_bins=10)
    out["ece"] = ece
    return out


def select_best_model(x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray):
    rows = []
    for name in model_family().keys():
        pipe = fit_model(name, x_train, y_train)
        val_m = evaluate_pipe(pipe, x_val, y_val, threshold=0.5)
        rows.append({
            "model_name": name,
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_roc_auc": val_m.get("roc_auc"),
            "val_brier": val_m.get("brier"),
            "val_ece": val_m.get("ece"),
        })
    comp = pd.DataFrame(rows)
    comp = comp.sort_values(["val_macro_f1", "val_balanced_accuracy", "val_accuracy"], ascending=[False, False, False]).reset_index(drop=True)
    return str(comp.iloc[0]["model_name"]), comp


def pooled_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (y_prob >= threshold).astype(int)
    out = classification_metrics(y_true, pred, y_prob)
    ece, _ = expected_calibration_error(y_true, y_prob, n_bins=10)
    out["ece"] = ece
    out["threshold"] = float(threshold)
    return out


def fit_platt(raw_prob: np.ndarray, y_true: np.ndarray):
    x = raw_prob.reshape(-1, 1)
    model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    model.fit(x, y_true)
    return model


def apply_platt(model, raw_prob: np.ndarray) -> np.ndarray:
    return model.predict_proba(raw_prob.reshape(-1, 1))[:, 1]


def fit_isotonic(raw_prob: np.ndarray, y_true: np.ndarray):
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(raw_prob, y_true)
    return model


def choose_canonical_predictions(pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = pred_df.sort_values(["period_start", "fold"]).copy()
    dup = ordered[ordered.duplicated("period_id", keep=False)].copy()
    canonical = ordered.drop_duplicates("period_id", keep="first").reset_index(drop=True)
    return canonical, dup.reset_index(drop=True)


def calibration_bins_for_prob(y_true: np.ndarray, y_prob: np.ndarray, label: str) -> pd.DataFrame:
    ece, bins = expected_calibration_error(y_true, y_prob, n_bins=10)
    bins.insert(0, "series", label)
    bins["ece_total"] = ece
    return bins


def run_mode(df: pd.DataFrame, x_all: pd.DataFrame, y_all: np.ndarray, windows: List[Dict], mode: str, out_dir: Path) -> Dict:
    fold_rows = []
    pred_rows = []

    for window in windows:
        train_idx = np.array(window["train_idx"])
        val_idx = np.array(window["val_idx"])
        test_idx = np.array(window["test_idx"])

        train_mask = np.isfinite(y_all[train_idx])
        val_mask = np.isfinite(y_all[val_idx])
        test_mask = np.isfinite(y_all[test_idx])

        if train_mask.sum() < 12 or val_mask.sum() < 4 or test_mask.sum() < 4:
            continue

        train_ids = train_idx[train_mask]
        val_ids = val_idx[val_mask]
        test_ids = test_idx[test_mask]

        y_train = y_all[train_ids].astype(int)
        y_val = y_all[val_ids].astype(int)
        y_test = y_all[test_ids].astype(int)

        if len(np.unique(y_train)) < 2:
            continue

        x_train = x_all.iloc[train_ids].copy()
        x_val = x_all.iloc[val_ids].copy()
        x_test = x_all.iloc[test_ids].copy()

        if mode == "selected_best":
            chosen_model, selection_df = select_best_model(x_train, y_train, x_val, y_val)
        elif mode == "logreg_fixed":
            chosen_model = "logreg"
            selection_df = pd.DataFrame([{
                "model_name": chosen_model,
                "note": "fixed_model_mode"
            }])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        fold_dir = out_dir / mode / f"fold_{window['fold']:02d}"
        ensure_dir(fold_dir)
        selection_df.to_csv(fold_dir / "model_selection.csv", index=False)

        trainval_ids = list(train_ids) + list(val_ids)
        x_trainval = x_all.iloc[trainval_ids].copy()
        y_trainval = y_all[trainval_ids].astype(int)
        pipe = fit_model(chosen_model, x_trainval, y_trainval)

        if hasattr(pipe, "predict_proba"):
            val_prob = pipe.predict_proba(x_val)[:, 1]
            test_prob = pipe.predict_proba(x_test)[:, 1]
        else:
            val_prob = pipe.predict(x_val).astype(float)
            test_prob = pipe.predict(x_test).astype(float)

        default_metrics = pooled_probability_metrics(y_test, test_prob, threshold=0.5)
        fold_rows.append({
            "mode": mode,
            "fold": window["fold"],
            "chosen_model": chosen_model,
            "train_n": int(len(train_ids)),
            "val_n": int(len(val_ids)),
            "test_n": int(len(test_ids)),
            "train_positive_rate": float(np.mean(y_train)),
            "val_positive_rate": float(np.mean(y_val)),
            "test_positive_rate": float(np.mean(y_test)),
            **{f"default_{k}": v for k, v in default_metrics.items()},
        })

        fold_pred = pd.DataFrame({
            "mode": mode,
            "fold": window["fold"],
            "period_id": df.iloc[test_ids]["period_id"].astype(str).tolist(),
            "period_start": pd.to_datetime(df.iloc[test_ids]["period_start"]).astype(str).tolist(),
            "y_true": y_test,
            "p_raw": test_prob,
            "y_pred_default_0_5": (test_prob >= 0.5).astype(int),
        })
        pred_rows.append(fold_pred)

    if not pred_rows:
        return {"status": "no_valid_folds"}

    fold_df = pd.DataFrame(fold_rows)
    all_preds = pd.concat(pred_rows, ignore_index=True)
    canonical_preds, duplicate_preds = choose_canonical_predictions(all_preds)

    fold_df.to_csv(out_dir / mode / "fold_summary.csv", index=False)
    all_preds.to_csv(out_dir / mode / "all_test_predictions_overlapping.csv", index=False)
    canonical_preds.to_csv(out_dir / mode / "canonical_test_predictions.csv", index=False)
    duplicate_preds.to_csv(out_dir / mode / "duplicate_predictions.csv", index=False)

    y = canonical_preds["y_true"].astype(int).to_numpy()
    p_raw = canonical_preds["p_raw"].astype(float).to_numpy()

    raw_metrics = pooled_probability_metrics(y, p_raw, threshold=0.5)
    raw_ece, raw_bins = expected_calibration_error(y, p_raw, n_bins=10)

    platt_model = fit_platt(p_raw, y)
    p_platt = apply_platt(platt_model, p_raw)
    platt_metrics = pooled_probability_metrics(y, p_platt, threshold=0.5)
    platt_ece, platt_bins = expected_calibration_error(y, p_platt, n_bins=10)

    isotonic_model = fit_isotonic(p_raw, y)
    p_iso = isotonic_model.predict(p_raw)
    iso_metrics = pooled_probability_metrics(y, p_iso, threshold=0.5)
    iso_ece, iso_bins = expected_calibration_error(y, p_iso, n_bins=10)

    canonical_preds["p_platt"] = p_platt
    canonical_preds["p_isotonic"] = p_iso
    canonical_preds.to_csv(out_dir / mode / "canonical_test_predictions_calibrated.csv", index=False)

    pd.concat([
        calibration_bins_for_prob(y, p_raw, "raw"),
        calibration_bins_for_prob(y, p_platt, "platt"),
        calibration_bins_for_prob(y, p_iso, "isotonic"),
    ], ignore_index=True).to_csv(out_dir / mode / "calibration_bins_compare.csv", index=False)

    thr_raw_bal, scan_raw_bal = choose_threshold(y, p_raw, metric="balanced_accuracy")
    thr_raw_f1, scan_raw_f1 = choose_threshold(y, p_raw, metric="macro_f1")
    thr_platt_bal, scan_platt_bal = choose_threshold(y, p_platt, metric="balanced_accuracy")
    thr_platt_f1, scan_platt_f1 = choose_threshold(y, p_platt, metric="macro_f1")
    thr_iso_bal, scan_iso_bal = choose_threshold(y, p_iso, metric="balanced_accuracy")
    thr_iso_f1, scan_iso_f1 = choose_threshold(y, p_iso, metric="macro_f1")

    scan_raw_bal.to_csv(out_dir / mode / "threshold_scan_raw_balanced_accuracy.csv", index=False)
    scan_raw_f1.to_csv(out_dir / mode / "threshold_scan_raw_macro_f1.csv", index=False)
    scan_platt_bal.to_csv(out_dir / mode / "threshold_scan_platt_balanced_accuracy.csv", index=False)
    scan_platt_f1.to_csv(out_dir / mode / "threshold_scan_platt_macro_f1.csv", index=False)
    scan_iso_bal.to_csv(out_dir / mode / "threshold_scan_isotonic_balanced_accuracy.csv", index=False)
    scan_iso_f1.to_csv(out_dir / mode / "threshold_scan_isotonic_macro_f1.csv", index=False)

    summary = {
        "mode": mode,
        "n_folds": int(len(fold_df)),
        "n_overlapping_predictions": int(len(all_preds)),
        "n_canonical_predictions": int(len(canonical_preds)),
        "n_duplicate_predictions_collapsed": int(len(duplicate_preds)),
        "raw_metrics": raw_metrics,
        "platt_metrics": platt_metrics,
        "isotonic_metrics": iso_metrics,
        "raw_ece": raw_ece,
        "platt_ece": platt_ece,
        "isotonic_ece": iso_ece,
        "recommended_threshold_raw_balanced_accuracy": float(thr_raw_bal),
        "recommended_threshold_raw_macro_f1": float(thr_raw_f1),
        "recommended_threshold_platt_balanced_accuracy": float(thr_platt_bal),
        "recommended_threshold_platt_macro_f1": float(thr_platt_f1),
        "recommended_threshold_isotonic_balanced_accuracy": float(thr_iso_bal),
        "recommended_threshold_isotonic_macro_f1": float(thr_iso_f1),
    }
    save_json(out_dir / mode / "calibration_summary.json", summary)
    joblib.dump(platt_model, out_dir / mode / "platt_calibrator.joblib")
    joblib.dump(isotonic_model, out_dir / mode / "isotonic_calibrator.joblib")

    return summary


def build_report(summaries: List[Dict], fold_tables: Dict[str, pd.DataFrame]) -> str:
    lines = []
    lines.append("# Weekly Weight-Gain Probability Calibration Report")
    lines.append("")
    lines.append("- target: weeks / y_next_weight_gain_flag")
    lines.append("- goal: compare raw probabilities versus pooled Platt and isotonic calibration using canonical rolling out-of-fold predictions")
    lines.append("")
    for summary in summaries:
        mode = summary["mode"]
        lines.append(f"## Mode: {mode}")
        lines.append("")
        lines.append(f"- folds: {summary['n_folds']}")
        lines.append(f"- overlapping predictions: {summary['n_overlapping_predictions']}")
        lines.append(f"- canonical predictions: {summary['n_canonical_predictions']}")
        lines.append(f"- duplicates collapsed: {summary['n_duplicate_predictions_collapsed']}")
        lines.append("")
        lines.append("### Raw metrics")
        for k, v in summary["raw_metrics"].items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
        lines.append(f"- ece: {summary['raw_ece']:.4f}")
        lines.append("")
        lines.append("### Platt metrics")
        for k, v in summary["platt_metrics"].items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
        lines.append(f"- ece: {summary['platt_ece']:.4f}")
        lines.append("")
        lines.append("### Isotonic metrics")
        for k, v in summary["isotonic_metrics"].items():
            lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
        lines.append(f"- ece: {summary['isotonic_ece']:.4f}")
        lines.append("")
        lines.append("### Suggested thresholds")
        lines.append(f"- raw / balanced_accuracy: {summary['recommended_threshold_raw_balanced_accuracy']:.4f}")
        lines.append(f"- raw / macro_f1: {summary['recommended_threshold_raw_macro_f1']:.4f}")
        lines.append(f"- platt / balanced_accuracy: {summary['recommended_threshold_platt_balanced_accuracy']:.4f}")
        lines.append(f"- platt / macro_f1: {summary['recommended_threshold_platt_macro_f1']:.4f}")
        lines.append(f"- isotonic / balanced_accuracy: {summary['recommended_threshold_isotonic_balanced_accuracy']:.4f}")
        lines.append(f"- isotonic / macro_f1: {summary['recommended_threshold_isotonic_macro_f1']:.4f}")
        lines.append("")
        if mode in fold_tables:
            lines.append("### Fold summary")
            lines.append("")
            lines.append(df_to_markdown_table(fold_tables[mode]))
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pooled out-of-fold calibration analysis for weekly weight-gain target.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--transition-dir", default="training/regime_transition", help="Relative transition dir.")
    parser.add_argument("--out-dir", default="reports/analysis/weekly_weight_gain_calibration", help="Relative output dir.")
    parser.add_argument("--min-train", type=int, default=36)
    parser.add_argument("--val-size", type=int, default=8)
    parser.add_argument("--test-size", type=int, default=6)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    transition_csv = project_root / args.transition_dir / "weeks_transition_matrix.csv"
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)
    ensure_dir(out_dir / "selected_best")
    ensure_dir(out_dir / "logreg_fixed")

    if not transition_csv.exists():
        raise FileNotFoundError(f"Missing transition matrix: {transition_csv}")

    df = pd.read_csv(transition_csv, low_memory=False)
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df = df.sort_values("period_start").reset_index(drop=True)

    if TARGET_NAME not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_NAME}")

    x_all, dropped_cols = prepare_feature_frame(df)
    y_all = encode_binary_target(df[TARGET_NAME]).to_numpy()
    windows = build_windows(len(df), min_train=args.min_train, val_size=args.val_size, test_size=args.test_size, step=args.step)

    summaries = []
    fold_tables = {}

    for mode in ["selected_best", "logreg_fixed"]:
        log(f"Running calibration mode: {mode} ...")
        summary = run_mode(df, x_all, y_all, windows, mode, out_dir)
        if summary.get("status") == "no_valid_folds":
            continue
        summaries.append(summary)
        fold_path = out_dir / mode / "fold_summary.csv"
        if fold_path.exists():
            fold_tables[mode] = pd.read_csv(fold_path)

    if not summaries:
        save_json(out_dir / "status.json", {
            "status": "no_valid_modes",
            "target": TARGET_NAME,
        })
        log("No valid modes.")
        return

    overall_rows = []
    for s in summaries:
        row = {
            "mode": s["mode"],
            "n_folds": s["n_folds"],
            "n_canonical_predictions": s["n_canonical_predictions"],
            "duplicates_collapsed": s["n_duplicate_predictions_collapsed"],
            "raw_accuracy": s["raw_metrics"].get("accuracy"),
            "raw_balanced_accuracy": s["raw_metrics"].get("balanced_accuracy"),
            "raw_macro_f1": s["raw_metrics"].get("macro_f1"),
            "raw_roc_auc": s["raw_metrics"].get("roc_auc"),
            "raw_brier": s["raw_metrics"].get("brier"),
            "raw_ece": s["raw_ece"],
            "platt_accuracy": s["platt_metrics"].get("accuracy"),
            "platt_balanced_accuracy": s["platt_metrics"].get("balanced_accuracy"),
            "platt_macro_f1": s["platt_metrics"].get("macro_f1"),
            "platt_roc_auc": s["platt_metrics"].get("roc_auc"),
            "platt_brier": s["platt_metrics"].get("brier"),
            "platt_ece": s["platt_ece"],
            "isotonic_accuracy": s["isotonic_metrics"].get("accuracy"),
            "isotonic_balanced_accuracy": s["isotonic_metrics"].get("balanced_accuracy"),
            "isotonic_macro_f1": s["isotonic_metrics"].get("macro_f1"),
            "isotonic_roc_auc": s["isotonic_metrics"].get("roc_auc"),
            "isotonic_brier": s["isotonic_metrics"].get("brier"),
            "isotonic_ece": s["isotonic_ece"],
        }
        overall_rows.append(row)

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / "overall_summary.csv", index=False)
    save_json(out_dir / "overall_summary.json", {
        "target_space": TARGET_SPACE,
        "target_name": TARGET_NAME,
        "min_train": args.min_train,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "step": args.step,
        "dropped_feature_cols": dropped_cols,
        "modes": summaries,
    })

    (out_dir / "overall_report.md").write_text(
        build_report(summaries, fold_tables),
        encoding="utf-8",
    )

    log("Done.")
    log(f"Wrote calibration analysis to: {out_dir}")


if __name__ == "__main__":
    main()
