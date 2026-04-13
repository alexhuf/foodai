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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_SPACE = "weeks"
TARGET_NAME = "y_next_weight_gain_flag"
KEEP_GROUPS = {"meals", "biology", "weather_daylight"}


def log(msg: str) -> None:
    print(f"[weekly-gain-refined] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
        angle = 2.0 * np.pi * ((out["period_dayofyear"].fillna(1.0) - 1.0) / 365.25)
        out["period_doy_sin"] = np.sin(angle)
        out["period_doy_cos"] = np.cos(angle)
        if t.notna().any():
            origin = t.min()
            out["period_days_since_start"] = (t - origin).dt.days.astype("float")
        out = out.drop(columns=["period_start"])
    return out


def classify_feature_group(col: str) -> str:
    c = col.lower()
    weather_keys = [
        "temperature", "apparent_temperature", "precip", "rain", "snow", "snowfall",
        "cloud", "wind", "gust", "pressure", "humidity", "daylight", "sunrise", "sunset",
        "uv", "weather", "is_day", "rain_streak", "freeze", "hot_streak",
    ]
    biology_keys = [
        "samsung", "heart", "hr", "stress", "sleep", "steps", "exercise", "active",
        "resting", "vo2", "oxygen", "resp", "calories_burned", "bmr", "weight_",
        "weightvelocity", "weight_velocity", "noom_weight",
    ]
    meal_keys = [
        "meal_", "noom_food", "restaurant_", "cuisine", "archetype", "protein", "carb",
        "fat", "fiber", "dessert", "beverage", "snack", "breakfast", "lunch", "dinner",
        "satiety", "indulgence", "comfort_food", "service_form", "prep_profile",
        "distinct_meal", "distinct_cuisines", "restaurant_specific", "food_",
    ]
    temporal_keys = [
        "period_", "month", "quarter", "weekofyear", "dayofyear", "doy_", "season",
        "days_since_start", "year",
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


def prepare_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
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
    manifest = pd.DataFrame({
        "feature": list(x.columns),
        "feature_group": [classify_feature_group(c) for c in x.columns],
    })
    keep_cols = manifest.loc[manifest["feature_group"].isin(KEEP_GROUPS), "feature"].tolist()
    x = x[keep_cols].copy()
    manifest = manifest[manifest["feature"].isin(keep_cols)].reset_index(drop=True)
    return x, exclude_cols, manifest


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


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "positive_rate_true": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["brier"] = float(brier_score_loss(y_true, y_prob))
    return out


def fit_logreg(x_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(x_train)),
        ("model", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(x_train, y_train)
    return pipe


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


def extract_coefficients(pipe: Pipeline, top_k: int = 60) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim == 2:
        coef = coef[0]
    names = get_feature_names_from_pipe(pipe)
    if len(names) != len(coef):
        names = [f"feature_{i:04d}" for i in range(len(coef))]
    idx = np.argsort(np.abs(coef))[::-1][:top_k]
    rows = []
    for rank, i in enumerate(idx, start=1):
        rows.append({
            "rank": rank,
            "feature": names[i],
            "coefficient": float(coef[i]),
            "direction": "positive" if coef[i] > 0 else "negative",
        })
    return pd.DataFrame(rows)


def summarize_rows(df: pd.DataFrame) -> Dict:
    out = {"n_rows": int(len(df))}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        out[f"{col}_mean"] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std(ddof=0))
        out[f"{col}_min"] = float(df[col].min())
        out[f"{col}_max"] = float(df[col].max())
    return out


def build_report(summary: Dict, fold_df: pd.DataFrame, coef_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Weekly Weight-Gain Refined Baseline Report")
    lines.append("")
    lines.append("- target: weeks / y_next_weight_gain_flag")
    lines.append(f"- kept feature groups: {sorted(list(KEEP_GROUPS))}")
    lines.append("")
    lines.append("## Aggregate metrics")
    for k, v in summary.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    if not fold_df.empty:
        lines.append("## Rolling fold metrics")
        lines.append("")
        lines.append(df_to_markdown_table(fold_df))
        lines.append("")
    if not coef_df.empty:
        lines.append("## Top coefficients")
        lines.append("")
        lines.append(df_to_markdown_table(coef_df.head(20)))
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train refined weekly next-weight-gain baseline using the best ablation feature groups.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--transition-dir", default="training/regime_transition", help="Relative transition dir.")
    parser.add_argument("--out-model-dir", default="models/weekly_weight_gain_refined", help="Relative model output dir.")
    parser.add_argument("--out-report-dir", default="reports/weekly_weight_gain_refined", help="Relative report output dir.")
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument("--val-size", type=int, default=8)
    parser.add_argument("--test-size", type=int, default=8)
    parser.add_argument("--step", type=int, default=6)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    transition_csv = project_root / args.transition_dir / "weeks_transition_matrix.csv"
    model_dir = project_root / args.out_model_dir
    report_dir = project_root / args.out_report_dir
    ensure_dir(model_dir)
    ensure_dir(report_dir)

    if not transition_csv.exists():
        raise FileNotFoundError(f"Missing transition matrix: {transition_csv}")

    df = pd.read_csv(transition_csv, low_memory=False)
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df = df.sort_values("period_start").reset_index(drop=True)

    if TARGET_NAME not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_NAME}")

    x_all, dropped_cols, feature_manifest = prepare_feature_frame(df)
    y_all = encode_binary_target(df[TARGET_NAME]).to_numpy()

    feature_manifest.to_csv(report_dir / "feature_manifest.csv", index=False)
    windows = build_windows(len(df), min_train=args.min_train, val_size=args.val_size, test_size=args.test_size, step=args.step)

    fold_rows = []
    pred_frames = []

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

        pipe = fit_logreg(x_train, y_train)
        val_prob = pipe.predict_proba(x_val)[:, 1]
        test_prob = pipe.predict_proba(x_test)[:, 1]

        thr_bal, scan_bal = choose_threshold(y_val, val_prob, metric="balanced_accuracy")
        thr_f1, scan_f1 = choose_threshold(y_val, val_prob, metric="macro_f1")

        default_pred = (test_prob >= 0.5).astype(int)
        tuned_pred = (test_prob >= thr_bal).astype(int)

        default_m = classification_metrics(y_test, default_pred, test_prob)
        tuned_m = classification_metrics(y_test, tuned_pred, test_prob)

        fold_rows.append({
            "fold": window["fold"],
            "threshold_balanced_accuracy": thr_bal,
            "threshold_macro_f1": thr_f1,
            **{f"default_{k}": v for k, v in default_m.items()},
            **{f"tuned_bal_{k}": v for k, v in tuned_m.items()},
            "train_positive_rate": float(np.mean(y_train)),
            "val_positive_rate": float(np.mean(y_val)),
            "test_positive_rate": float(np.mean(y_test)),
            "train_n": int(len(train_ids)),
            "val_n": int(len(val_ids)),
            "test_n": int(len(test_ids)),
        })

        fold_dir = report_dir / f"fold_{window['fold']:02d}"
        ensure_dir(fold_dir)
        scan_bal.to_csv(fold_dir / "threshold_scan_balanced_accuracy.csv", index=False)
        scan_f1.to_csv(fold_dir / "threshold_scan_macro_f1.csv", index=False)

        pred_df = pd.DataFrame({
            "fold": window["fold"],
            "period_id": df.iloc[test_ids]["period_id"].astype(str).tolist(),
            "period_start": pd.to_datetime(df.iloc[test_ids]["period_start"]).astype(str).tolist(),
            "y_true": y_test,
            "p_positive": test_prob,
            "y_pred_default_0_5": default_pred,
            "y_pred_tuned_bal": tuned_pred,
        })
        pred_df["correct_default_0_5"] = pred_df["y_true"] == pred_df["y_pred_default_0_5"]
        pred_df["correct_tuned_bal"] = pred_df["y_true"] == pred_df["y_pred_tuned_bal"]
        pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)
        pred_frames.append(pred_df)

    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        fold_df.to_csv(report_dir / "rolling_fold_metrics.csv", index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(report_dir / "rolling_predictions.csv", index=False)

    valid_mask = np.isfinite(y_all)
    x_train_final = x_all.loc[valid_mask].copy()
    y_train_final = y_all[valid_mask].astype(int)

    final_pipe = fit_logreg(x_train_final, y_train_final)
    joblib.dump(final_pipe, model_dir / "logreg_refined.joblib")

    coef_df = extract_coefficients(final_pipe, top_k=80)
    coef_df.to_csv(report_dir / "coefficient_summary.csv", index=False)

    p_all = final_pipe.predict_proba(x_train_final)[:, 1]
    train_scored = df.loc[valid_mask, ["period_id", "period_start", TARGET_NAME]].copy()
    train_scored["risk_probability"] = p_all
    train_scored["risk_band"] = pd.cut(
        train_scored["risk_probability"],
        bins=[-np.inf, 0.10, 0.20, np.inf],
        labels=["low", "watch", "high"],
    )
    train_scored.to_csv(report_dir / "all_scored_weeks.csv", index=False)

    recent = train_scored.sort_values("period_start", ascending=False).head(20).copy()
    recent.to_csv(report_dir / "recent_scored_weeks.csv", index=False)

    summary = {
        "target_space": TARGET_SPACE,
        "target_name": TARGET_NAME,
        "kept_feature_groups": sorted(list(KEEP_GROUPS)),
        "dropped_feature_cols": dropped_cols,
        "model_path": str(model_dir / "logreg_refined.joblib"),
        "n_training_rows": int(len(x_train_final)),
        "positive_rate_training": float(np.mean(y_train_final)),
        "rolling_folds": int(len(fold_df)),
    }
    if not fold_df.empty:
        summary.update(summarize_rows(fold_df))
    save_json(report_dir / "overall_summary.json", summary)
    pd.DataFrame([summary]).to_csv(report_dir / "overall_summary.csv", index=False)

    (report_dir / "overall_report.md").write_text(
        build_report(summary, fold_df, coef_df),
        encoding="utf-8",
    )

    log("Done.")
    log(f"Wrote refined model to: {model_dir}")
    log(f"Wrote refined reports to: {report_dir}")


if __name__ == "__main__":
    main()
