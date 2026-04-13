from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_SPACE = "weeks"
TARGET_NAME = "y_next_weight_gain_flag"


def log(msg: str) -> None:
    print(f"[weight-gain-focus] {msg}")


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
    out["positive_rate_pred"] = float(np.mean(pred))
    out["positive_rate_true"] = float(np.mean(y))
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


def extract_logreg_drivers(pipe: Pipeline, top_k: int = 25) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame()
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


def compute_class_balance(y: np.ndarray) -> Dict[str, float]:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    total = int(len(y))
    return {
        "n_total": total,
        "n_positive": positives,
        "n_negative": negatives,
        "positive_rate": float(positives / total) if total else np.nan,
    }


def run_fold(df: pd.DataFrame, x_all: pd.DataFrame, y_all: np.ndarray, window: Dict, out_dir: Path):
    train_idx = window["train_idx"]
    val_idx = window["val_idx"]
    test_idx = window["test_idx"]

    train_mask = np.isfinite(y_all[train_idx])
    val_mask = np.isfinite(y_all[val_idx])
    test_mask = np.isfinite(y_all[test_idx])

    if train_mask.sum() < 12 or val_mask.sum() < 4 or test_mask.sum() < 4:
        return None

    train_ids = np.array(train_idx)[train_mask]
    val_ids = np.array(val_idx)[val_mask]
    test_ids = np.array(test_idx)[test_mask]

    y_train = y_all[train_ids].astype(int)
    y_val = y_all[val_ids].astype(int)
    y_test = y_all[test_ids].astype(int)

    if len(np.unique(y_train)) < 2:
        return None

    x_train = x_all.iloc[train_ids].copy()
    x_val = x_all.iloc[val_ids].copy()
    x_test = x_all.iloc[test_ids].copy()

    fold_dir = out_dir / f"fold_{window['fold']:02d}"
    ensure_dir(fold_dir)

    best_name, comp_df = select_best_model(x_train, y_train, x_val, y_val)
    comp_df.to_csv(fold_dir / "model_selection.csv", index=False)

    trainval_ids = list(train_ids) + list(val_ids)
    x_trainval = x_all.iloc[trainval_ids].copy()
    y_trainval = y_all[trainval_ids].astype(int)
    best_pipe = fit_model(best_name, x_trainval, y_trainval)

    if hasattr(best_pipe, "predict_proba"):
        val_prob = best_pipe.predict_proba(x_val)[:, 1]
        test_prob = best_pipe.predict_proba(x_test)[:, 1]
    else:
        val_prob = best_pipe.predict(x_val).astype(float)
        test_prob = best_pipe.predict(x_test).astype(float)

    default_metrics = evaluate_pipe(best_pipe, x_test, y_test, threshold=0.5)

    best_thr_bal, thr_scan_bal = choose_threshold(y_val, val_prob, metric="balanced_accuracy")
    best_thr_f1, thr_scan_f1 = choose_threshold(y_val, val_prob, metric="macro_f1")
    thr_scan_bal.to_csv(fold_dir / "threshold_scan_balanced_accuracy.csv", index=False)
    thr_scan_f1.to_csv(fold_dir / "threshold_scan_macro_f1.csv", index=False)

    tuned_bal_metrics = evaluate_pipe(best_pipe, x_test, y_test, threshold=best_thr_bal)
    tuned_f1_metrics = evaluate_pipe(best_pipe, x_test, y_test, threshold=best_thr_f1)

    ece_test, calib_bins = expected_calibration_error(y_test, test_prob, n_bins=10)
    calib_bins.to_csv(fold_dir / "calibration_bins_test.csv", index=False)

    pred_df = pd.DataFrame({
        "period_id": df.iloc[test_ids]["period_id"].astype(str).tolist(),
        "period_start": pd.to_datetime(df.iloc[test_ids]["period_start"]).astype(str).tolist(),
        "y_true": y_test,
        "p_positive": test_prob,
        "y_pred_default_0_5": (test_prob >= 0.5).astype(int),
        "y_pred_tuned_bal": (test_prob >= best_thr_bal).astype(int),
        "y_pred_tuned_f1": (test_prob >= best_thr_f1).astype(int),
    })
    pred_df["correct_default_0_5"] = pred_df["y_true"] == pred_df["y_pred_default_0_5"]
    pred_df["correct_tuned_bal"] = pred_df["y_true"] == pred_df["y_pred_tuned_bal"]
    pred_df["correct_tuned_f1"] = pred_df["y_true"] == pred_df["y_pred_tuned_f1"]
    pred_df.to_csv(fold_dir / "test_predictions_scored.csv", index=False)

    logreg_pipe = fit_model("logreg", x_trainval, y_trainval)
    logreg_driver_df = extract_logreg_drivers(logreg_pipe, top_k=25)
    if not logreg_driver_df.empty:
        logreg_driver_df.to_csv(fold_dir / "logreg_feature_drivers.csv", index=False)

    ablation_sets = build_ablation_sets(list(x_all.columns))
    ablation_rows = []
    for name, cols in ablation_sets.items():
        if len(cols) == 0:
            continue
        x_train_ab = x_train[cols].copy()
        x_val_ab = x_val[cols].copy()
        x_test_ab = x_test[cols].copy()
        try:
            pipe_ab = fit_model("logreg", pd.concat([x_train_ab, x_val_ab], axis=0), np.concatenate([y_train, y_val]))
            val_prob_ab = pipe_ab.predict_proba(x_val_ab)[:, 1]
            test_prob_ab = pipe_ab.predict_proba(x_test_ab)[:, 1]
            thr_ab, _ = choose_threshold(y_val, val_prob_ab, metric="balanced_accuracy")
            met_ab = evaluate_pipe(pipe_ab, x_test_ab, y_test, threshold=thr_ab)
            ablation_rows.append({
                "fold": window["fold"],
                "ablation": name,
                "n_features": len(cols),
                "threshold": thr_ab,
                "test_accuracy": met_ab["accuracy"],
                "test_balanced_accuracy": met_ab["balanced_accuracy"],
                "test_macro_f1": met_ab["macro_f1"],
                "test_roc_auc": met_ab.get("roc_auc"),
                "test_brier": met_ab.get("brier"),
                "test_ece": met_ab.get("ece"),
            })
        except Exception:
            continue
    ablation_df = pd.DataFrame(ablation_rows)
    if not ablation_df.empty:
        ablation_df.to_csv(fold_dir / "ablation_results.csv", index=False)

    fold_summary = {
        "fold": window["fold"],
        "best_model": best_name,
        **{f"train_{k}": v for k, v in compute_class_balance(y_train).items()},
        **{f"val_{k}": v for k, v in compute_class_balance(y_val).items()},
        **{f"test_{k}": v for k, v in compute_class_balance(y_test).items()},
        "selected_threshold_balanced_accuracy": best_thr_bal,
        "selected_threshold_macro_f1": best_thr_f1,
        **{f"default_{k}": v for k, v in default_metrics.items()},
        **{f"tuned_bal_{k}": v for k, v in tuned_bal_metrics.items()},
        **{f"tuned_f1_{k}": v for k, v in tuned_f1_metrics.items()},
        "test_ece_raw": ece_test,
    }
    return fold_summary


def summarize_fold_table(fold_df: pd.DataFrame) -> Dict:
    out = {
        "n_folds": int(len(fold_df)),
        "models_chosen": fold_df["best_model"].value_counts().to_dict() if "best_model" in fold_df.columns else {},
    }
    numeric_cols = [c for c in fold_df.columns if c not in {"fold", "best_model"} and pd.api.types.is_numeric_dtype(fold_df[c])]
    for col in numeric_cols:
        out[f"{col}_mean"] = float(fold_df[col].mean())
        out[f"{col}_std"] = float(fold_df[col].std(ddof=0))
        out[f"{col}_min"] = float(fold_df[col].min())
        out[f"{col}_max"] = float(fold_df[col].max())
    return out


def build_report(summary: Dict, fold_df: pd.DataFrame, ablation_summary: pd.DataFrame) -> str:
    lines = []
    lines.append("# Weekly Weight-Gain Transition Stability Report")
    lines.append("")
    lines.append(f"- target: {TARGET_SPACE} / {TARGET_NAME}")
    lines.append(f"- folds: {summary.get('n_folds', 0)}")
    lines.append(f"- models chosen: {summary.get('models_chosen', {})}")
    lines.append("")
    lines.append("## Aggregate fold metrics")
    for k, v in summary.items():
        if k in {"n_folds", "models_chosen"}:
            continue
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Fold diagnostics")
    lines.append("")
    lines.append(df_to_markdown_table(fold_df))
    lines.append("")
    if not ablation_summary.empty:
        lines.append("## Ablation summary")
        lines.append("")
        lines.append(df_to_markdown_table(ablation_summary))
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Focused rolling analysis for weekly next-weight-gain target.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--transition-dir", default="training/regime_transition", help="Relative transition dir.")
    parser.add_argument("--out-dir", default="reports/analysis/weekly_weight_gain_focus", help="Relative output dir.")
    parser.add_argument("--min-train", type=int, default=36)
    parser.add_argument("--val-size", type=int, default=8)
    parser.add_argument("--test-size", type=int, default=6)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    transition_csv = project_root / args.transition_dir / "weeks_transition_matrix.csv"
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

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

    feature_group_manifest = pd.DataFrame({
        "feature": list(x_all.columns),
        "feature_group": [classify_feature_group(c) for c in x_all.columns],
    }).sort_values(["feature_group", "feature"]).reset_index(drop=True)
    feature_group_manifest.to_csv(out_dir / "feature_group_manifest.csv", index=False)

    fold_rows = []
    all_ablation = []
    for window in windows:
        log(f"Running fold {window['fold']} ...")
        res = run_fold(df, x_all, y_all, window, out_dir)
        if res is not None:
            fold_rows.append(res)
            fold_dir = out_dir / f"fold_{window['fold']:02d}"
            ablation_path = fold_dir / "ablation_results.csv"
            if ablation_path.exists():
                all_ablation.append(pd.read_csv(ablation_path))
        else:
            log(f"Skipped fold {window['fold']} due to insufficient class support.")

    if not fold_rows:
        save_json(out_dir / "status.json", {
            "status": "skipped_no_valid_folds",
            "target": TARGET_NAME,
            "min_train": args.min_train,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "step": args.step,
        })
        log("No valid folds.")
        return

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fold_diagnostics.csv", index=False)

    ablation_summary = pd.DataFrame()
    if all_ablation:
        ab_all = pd.concat(all_ablation, ignore_index=True)
        ab_all.to_csv(out_dir / "ablation_by_fold.csv", index=False)
        num_cols = [c for c in ["test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_roc_auc", "test_brier", "test_ece"] if c in ab_all.columns]
        agg = {c: ["mean", "std", "min", "max"] for c in num_cols}
        ablation_summary = ab_all.groupby("ablation").agg(agg)
        ablation_summary.columns = [f"{a}_{b}" for a, b in ablation_summary.columns]
        ablation_summary = ablation_summary.reset_index()
        if "test_balanced_accuracy_mean" in ablation_summary.columns:
            ablation_summary = ablation_summary.sort_values("test_balanced_accuracy_mean", ascending=False)
        ablation_summary.to_csv(out_dir / "ablation_summary.csv", index=False)

    summary = summarize_fold_table(fold_df)
    summary.update({
        "target_space": TARGET_SPACE,
        "target_name": TARGET_NAME,
        "min_train": args.min_train,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "step": args.step,
        "dropped_feature_cols": dropped_cols,
    })
    save_json(out_dir / "overall_summary.json", summary)
    pd.DataFrame([summary]).to_csv(out_dir / "overall_summary.csv", index=False)

    (out_dir / "overall_report.md").write_text(
        build_report(summary, fold_df, ablation_summary),
        encoding="utf-8",
    )

    log("Done.")
    log(f"Wrote focused analysis to: {out_dir}")


if __name__ == "__main__":
    main()
