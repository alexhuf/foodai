from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_datetime64_any_dtype
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

EXCLUDE_AT_TRAIN_TIME = {
    "meal_id",
    "date",
    "decision_time",
    "is_last_meal_of_day",
    "hours_until_next_meal",
    "day_meal_count",
    "state_prior_meal_id",
    "state_prior_meal_text",
}

DEFAULT_CLASSIFICATION_TARGETS = [
    "y_next_meal_family_coarse",
    "y_next_meal_archetype_collapsed",
    "y_next_restaurant_meal",
    "y_post_meal_budget_breach",
]

DEFAULT_REGRESSION_TARGETS = [
    "y_next_meal_kcal_log1p",
]


def log(msg: str) -> None:
    print(f"[train-meal] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV file: {path}")
    return pd.read_csv(path, low_memory=False)


def safe_mkdirs(project_root: Path) -> Dict[str, Path]:
    base_model_dir = project_root / "models" / "baselines" / "meal"
    report_backtest_dir = project_root / "reports" / "backtests" / "meal"
    report_feat_dir = project_root / "reports" / "feature_importance" / "meal"
    split_dir = project_root / "training" / "splits"

    for p in [base_model_dir, report_backtest_dir, report_feat_dir, split_dir]:
        ensure_dir(p)

    return {
        "model": base_model_dir,
        "backtest": report_backtest_dir,
        "feat": report_feat_dir,
        "split": split_dir,
    }


def derive_meal_family_coarse(archetype: object, service_form: object, cuisine: object) -> str:
    a = "" if pd.isna(archetype) else str(archetype).strip().lower()
    s = "" if pd.isna(service_form) else str(service_form).strip().lower()
    c = "" if pd.isna(cuisine) else str(cuisine).strip().lower()

    text = " | ".join([a, s, c])

    if any(k in text for k in ["breakfast", "omelet", "eggs benedict", "pancake", "waffle", "bagel breakfast"]):
        return "breakfast"

    if any(k in text for k in ["dessert", "cookie", "ice cream", "cake", "brownie", "baklava"]):
        return "dessert"

    if any(k in text for k in ["snack", "acai", "fruit bowl", "chips", "pretzel", "cracker"]):
        return "snack"

    if any(k in text for k in ["beverage", "drink", "smoothie", "coffee", "tea", "juice"]):
        return "beverage"

    if any(k in text for k in ["burger", "sandwich", "sub", "wrap", "hot dog", "grilled cheese", "handheld"]):
        return "handheld_savory"

    if any(k in text for k in ["pizza", "pasta", "ramen", "noodle", "mac", "macaroni"]):
        return "pizza_pasta_noodle"

    if any(k in text for k in ["tex_mex", "taco", "burrito", "quesadilla", "fajita", "nacho"]):
        return "tex_mex"

    if any(k in text for k in ["sushi", "asian", "stir fry", "fried rice", "teriyaki", "dumpling", "gyoza"]):
        return "asian"

    if any(k in text for k in ["salad", "bowl", "plate", "mixed_plate", "protein_centered", "protein-centered", "grain bowl"]):
        return "plate_bowl"

    return "other"


def build_walkforward_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 3,
    min_train_days: int = 180,
    val_days: int = 28,
    test_days: int = 42,
) -> List[Dict[str, str]]:
    unique_dates = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.floor("D").sort_values().unique()
    unique_dates = list(pd.to_datetime(unique_dates))

    folds = []
    test_end_idx = len(unique_dates) - 1

    for _ in range(n_splits):
        test_start_idx = test_end_idx - test_days + 1
        val_end_idx = test_start_idx - 1
        val_start_idx = val_end_idx - val_days + 1
        train_end_idx = val_start_idx - 1

        if train_end_idx + 1 < min_train_days:
            break
        if test_start_idx < 0 or val_start_idx < 0:
            break

        fold = {
            "train_start": str(unique_dates[0].date()),
            "train_end": str(unique_dates[train_end_idx].date()),
            "val_start": str(unique_dates[val_start_idx].date()),
            "val_end": str(unique_dates[val_end_idx].date()),
            "test_start": str(unique_dates[test_start_idx].date()),
            "test_end": str(unique_dates[test_end_idx].date()),
        }
        folds.append(fold)
        test_end_idx = test_start_idx - 1

    folds.reverse()
    if not folds:
        raise ValueError("Could not build walk-forward splits with the current dataset size and requested parameters.")
    return folds


def make_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for c in X.columns:
        s = X[c]
        if is_datetime64_any_dtype(s):
            categorical_cols.append(c)
        elif is_bool_dtype(s):
            categorical_cols.append(c)
        elif is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        for name, _, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            names.extend([f"{name}__{c}" for c in cols])
        return names


def classification_models(n_classes: int):
    class_weight = "balanced" if n_classes <= 2 else None
    models = {
        "logreg": LogisticRegression(
            max_iter=5000,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            solver="saga",
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample" if n_classes <= 2 else None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced" if n_classes <= 2 else None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return models


def regression_models():
    return {
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "rf": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "et": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def top_k_accuracy_from_proba(y_true: np.ndarray, y_proba: np.ndarray, class_labels: np.ndarray, k: int) -> float:
    if y_proba is None or y_proba.ndim != 2:
        return float("nan")
    top_idx = np.argsort(-y_proba, axis=1)[:, :k]
    top_labels = class_labels[top_idx]
    hits = [(str(y_true[i]) in set(map(str, top_labels[i]))) for i in range(len(y_true))]
    return float(np.mean(hits)) if hits else float("nan")


def compute_classification_metrics(y_true, y_pred, y_proba=None, class_labels=None) -> Dict[str, float]:
    labels_union = np.unique(np.concatenate([np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str)]))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recall_score(y_true, y_pred, average="macro", labels=labels_union, zero_division=0)),
        "macro_f1": float(
            __import__("sklearn.metrics").metrics.f1_score(y_true, y_pred, average="macro", labels=labels_union, zero_division=0)
        ),
    }
    if y_proba is not None and class_labels is not None:
        try:
            out["log_loss"] = float(log_loss(y_true, y_proba, labels=class_labels))
        except Exception:
            pass
        try:
            out["top3_accuracy"] = top_k_accuracy_from_proba(np.asarray(y_true, dtype=str), y_proba, np.asarray(class_labels, dtype=str), k=3)
        except Exception:
            pass
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                out["avg_precision"] = float(average_precision_score(y_true, y_proba[:, 1]))
        except Exception:
            pass
    return out


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def extract_importance(pipeline: Pipeline) -> pd.DataFrame:
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feat_names = get_feature_names(pre)

    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 1:
            imp = np.abs(coef)
        else:
            imp = np.mean(np.abs(coef), axis=0)
        df = pd.DataFrame({"feature": feat_names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        df = pd.DataFrame({"feature": feat_names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    return pd.DataFrame({"feature": feat_names, "importance": np.nan})


def prepare_X(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in X.columns:
        if is_bool_dtype(X[c]):
            X[c] = X[c].astype("object")
        elif is_datetime64_any_dtype(X[c]):
            X[c] = X[c].astype(str)
        elif not is_numeric_dtype(X[c]):
            X[c] = X[c].astype("object")
    return X


def train_one_task_classification(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    folds: List[Dict[str, str]],
    out_dirs: Dict[str, Path],
) -> None:
    task_dir = out_dirs["model"] / target_col
    ensure_dir(task_dir)

    task_df = df[df[target_col].notna()].copy()
    if task_df[target_col].nunique() < 2:
        log(f"Skipping classification target {target_col}: not enough classes.")
        return

    y_all = task_df[target_col].astype(str)
    X_all = prepare_X(task_df, feature_cols)
    n_classes = y_all.nunique()

    model_defs = classification_models(n_classes)
    val_records = []
    test_pred_records = []

    for model_name, model in model_defs.items():
        for i, fold in enumerate(folds, start=1):
            train_mask = (task_df["date"] >= fold["train_start"]) & (task_df["date"] <= fold["train_end"])
            val_mask = (task_df["date"] >= fold["val_start"]) & (task_df["date"] <= fold["val_end"])
            test_mask = (task_df["date"] >= fold["test_start"]) & (task_df["date"] <= fold["test_end"])

            X_train = X_all.loc[train_mask]
            y_train = y_all.loc[train_mask]
            X_val = X_all.loc[val_mask]
            y_val = y_all.loc[val_mask]
            X_test = X_all.loc[test_mask]
            y_test = y_all.loc[test_mask]

            if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                continue

            preprocessor, _, _ = make_preprocessor(X_train)
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clone(model))])
            pipe.fit(X_train, y_train)

            y_val_pred = pipe.predict(X_val)
            try:
                y_val_proba = pipe.predict_proba(X_val)
                class_labels = pipe.named_steps["model"].classes_
            except Exception:
                y_val_proba = None
                class_labels = None
            val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba, class_labels)
            val_records.append({
                "target": target_col,
                "model": model_name,
                "fold": i,
                "split": "val",
                **fold,
                **val_metrics,
                "n_rows": int(len(X_val)),
            })

            y_test_pred = pipe.predict(X_test)
            try:
                y_test_proba = pipe.predict_proba(X_test)
                class_labels = pipe.named_steps["model"].classes_
            except Exception:
                y_test_proba = None
                class_labels = None
            test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_proba, class_labels)
            val_records.append({
                "target": target_col,
                "model": model_name,
                "fold": i,
                "split": "test",
                **fold,
                **test_metrics,
                "n_rows": int(len(X_test)),
            })

            pred_df = pd.DataFrame({
                "target": target_col,
                "model": model_name,
                "fold": i,
                "date": task_df.loc[test_mask, "date"].astype(str).values,
                "decision_time": task_df.loc[test_mask, "decision_time"].astype(str).values if "decision_time" in task_df.columns else None,
                "meal_id": task_df.loc[test_mask, "meal_id"].values if "meal_id" in task_df.columns else None,
                "y_true": y_test.values,
                "y_pred": y_test_pred,
            })
            test_pred_records.append(pred_df)

    val_df = pd.DataFrame(val_records)
    if val_df.empty:
        log(f"Skipping target {target_col}: no fold results.")
        return

    choice_col = "macro_f1" if "macro_f1" in val_df.columns else "balanced_accuracy"
    model_scores = (
        val_df[val_df["split"] == "val"]
        .groupby("model", as_index=False)[choice_col]
        .mean()
        .sort_values(choice_col, ascending=False)
    )
    best_model_name = model_scores.iloc[0]["model"]
    best_model = model_defs[best_model_name]

    preprocessor, _, _ = make_preprocessor(X_all)
    final_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clone(best_model))])
    final_pipe.fit(X_all, y_all)

    joblib.dump(final_pipe, task_dir / "best_model.joblib")

    importance_df = extract_importance(final_pipe)
    importance_df.to_csv(out_dirs["feat"] / f"{target_col}_feature_importance.csv", index=False)

    val_df.to_csv(out_dirs["backtest"] / f"{target_col}_fold_metrics.csv", index=False)
    if test_pred_records:
        pd.concat(test_pred_records, ignore_index=True).to_csv(
            out_dirs["backtest"] / f"{target_col}_test_predictions.csv", index=False
        )

    summary = {
        "task_type": "classification",
        "target": target_col,
        "best_model": best_model_name,
        "model_scores_validation_mean": model_scores.to_dict(orient="records"),
        "n_rows": int(len(task_df)),
        "n_classes": int(n_classes),
        "features_used": feature_cols,
    }
    (task_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Finished classification target: {target_col} (best={best_model_name})")


def train_one_task_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    folds: List[Dict[str, str]],
    out_dirs: Dict[str, Path],
) -> None:
    task_dir = out_dirs["model"] / target_col
    ensure_dir(task_dir)

    task_df = df[df[target_col].notna()].copy()
    if len(task_df) < 100:
        log(f"Skipping regression target {target_col}: too few rows.")
        return

    y_all = pd.to_numeric(task_df[target_col], errors="coerce")
    keep = y_all.notna()
    task_df = task_df.loc[keep].copy()
    y_all = y_all.loc[keep]
    X_all = prepare_X(task_df, feature_cols)

    model_defs = regression_models()
    records = []
    test_pred_records = []

    for model_name, model in model_defs.items():
        for i, fold in enumerate(folds, start=1):
            train_mask = (task_df["date"] >= fold["train_start"]) & (task_df["date"] <= fold["train_end"])
            val_mask = (task_df["date"] >= fold["val_start"]) & (task_df["date"] <= fold["val_end"])
            test_mask = (task_df["date"] >= fold["test_start"]) & (task_df["date"] <= fold["test_end"])

            X_train = X_all.loc[train_mask]
            y_train = y_all.loc[train_mask]
            X_val = X_all.loc[val_mask]
            y_val = y_all.loc[val_mask]
            X_test = X_all.loc[test_mask]
            y_test = y_all.loc[test_mask]

            if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                continue

            preprocessor, _, _ = make_preprocessor(X_train)
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clone(model))])
            pipe.fit(X_train, y_train)

            y_val_pred = pipe.predict(X_val)
            val_metrics = compute_regression_metrics(y_val, y_val_pred)
            records.append({
                "target": target_col,
                "model": model_name,
                "fold": i,
                "split": "val",
                **fold,
                **val_metrics,
                "n_rows": int(len(X_val)),
            })

            y_test_pred = pipe.predict(X_test)
            test_metrics = compute_regression_metrics(y_test, y_test_pred)
            records.append({
                "target": target_col,
                "model": model_name,
                "fold": i,
                "split": "test",
                **fold,
                **test_metrics,
                "n_rows": int(len(X_test)),
            })

            pred_df = pd.DataFrame({
                "target": target_col,
                "model": model_name,
                "fold": i,
                "date": task_df.loc[test_mask, "date"].astype(str).values,
                "decision_time": task_df.loc[test_mask, "decision_time"].astype(str).values if "decision_time" in task_df.columns else None,
                "meal_id": task_df.loc[test_mask, "meal_id"].values if "meal_id" in task_df.columns else None,
                "y_true": y_test.values,
                "y_pred": y_test_pred,
            })
            test_pred_records.append(pred_df)

    rec_df = pd.DataFrame(records)
    if rec_df.empty:
        log(f"Skipping target {target_col}: no fold results.")
        return

    model_scores = (
        rec_df[rec_df["split"] == "val"]
        .groupby("model", as_index=False)["mae"]
        .mean()
        .sort_values("mae", ascending=True)
    )
    best_model_name = model_scores.iloc[0]["model"]
    best_model = model_defs[best_model_name]

    preprocessor, _, _ = make_preprocessor(X_all)
    final_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clone(best_model))])
    final_pipe.fit(X_all, y_all)

    joblib.dump(final_pipe, task_dir / "best_model.joblib")

    importance_df = extract_importance(final_pipe)
    importance_df.to_csv(out_dirs["feat"] / f"{target_col}_feature_importance.csv", index=False)

    rec_df.to_csv(out_dirs["backtest"] / f"{target_col}_fold_metrics.csv", index=False)
    if test_pred_records:
        pd.concat(test_pred_records, ignore_index=True).to_csv(
            out_dirs["backtest"] / f"{target_col}_test_predictions.csv", index=False
        )

    summary = {
        "task_type": "regression",
        "target": target_col,
        "best_model": best_model_name,
        "model_scores_validation_mean": model_scores.to_dict(orient="records"),
        "n_rows": int(len(task_df)),
        "features_used": feature_cols,
    }
    (task_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Finished regression target: {target_col} (best={best_model_name})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train refined meal baseline models with walk-forward splits.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of walk-forward folds.")
    parser.add_argument("--min-train-days", type=int, default=180, help="Minimum training days in earliest fold.")
    parser.add_argument("--val-days", type=int, default=28, help="Validation window length in days.")
    parser.add_argument("--test-days", type=int, default=42, help="Test window length in days.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    paths = safe_mkdirs(project_root)

    view_path = project_root / "training" / "predictive_views" / "meal_prediction_view.csv"
    target_spec_path = project_root / "training" / "targets" / "target_spec_meal_prediction.json"

    log("Loading predictive view and target spec...")
    df = read_csv(view_path)
    target_spec = read_json(target_spec_path)

    if "date" not in df.columns:
        raise ValueError("meal_prediction_view.csv must contain a 'date' column for walk-forward splitting.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")

    # Derive a coarser meal-family target to reduce fragmentation.
    if {"target_meal_archetype_primary", "target_service_form_primary", "target_cuisine_primary"}.issubset(df.columns):
        df["y_next_meal_family_coarse"] = [
            derive_meal_family_coarse(a, s, c)
            for a, s, c in zip(df["target_meal_archetype_primary"], df["target_service_form_primary"], df["target_cuisine_primary"])
        ]

    folds = build_walkforward_splits(
        df=df,
        date_col="date",
        n_splits=args.n_splits,
        min_train_days=args.min_train_days,
        val_days=args.val_days,
        test_days=args.test_days,
    )
    (paths["split"] / "meal_walkforward_splits.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")

    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_AT_TRAIN_TIME:
            continue
        if col.startswith("state_"):
            feature_cols.append(col)
        elif col in {
            "decision_hour",
            "time_slot",
            "time_slot_label",
            "meal_order_in_day",
            "is_first_meal_of_day",
            "hours_since_prior_meal",
            "cumulative_meal_calories_before_meal",
            "remaining_budget_before_meal_kcal",
        }:
            feature_cols.append(col)

    classification_targets = [
        t for t in DEFAULT_CLASSIFICATION_TARGETS if (t in df.columns or t in target_spec.get("classification_targets", []))
    ]
    regression_targets = [
        t for t in DEFAULT_REGRESSION_TARGETS if t in target_spec.get("regression_targets", []) and t in df.columns
    ]

    run_manifest = {
        "feature_columns": feature_cols,
        "classification_targets": classification_targets,
        "regression_targets": regression_targets,
        "excluded_at_train_time": sorted(EXCLUDE_AT_TRAIN_TIME),
        "derived_targets": ["y_next_meal_family_coarse"] if "y_next_meal_family_coarse" in df.columns else [],
        "n_rows": int(len(df)),
        "split_file": str((paths["split"] / "meal_walkforward_splits.json").name),
    }
    (paths["model"] / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    log("Training classification baselines...")
    for target in classification_targets:
        if target in df.columns:
            train_one_task_classification(df, feature_cols, target, folds, paths)

    log("Training regression baselines...")
    for target in regression_targets:
        train_one_task_regression(df, feature_cols, target, folds, paths)

    log("Done.")
    log(f"Models written under: {paths['model']}")
    log(f"Backtest reports written under: {paths['backtest']}")
    log(f"Feature importance reports written under: {paths['feat']}")


if __name__ == "__main__":
    main()
