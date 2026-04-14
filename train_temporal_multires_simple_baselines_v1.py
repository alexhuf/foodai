from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

DEFAULT_BINARY_TARGETS = [
    "y_next_weight_loss_flag",
]
DEFAULT_REGRESSION_TARGETS = [
    "y_next_weight_delta_lb",
]

DEFAULT_WINDOWS = {
    "days": 7,
    "weeks": 4,
    "meals": 10,
}


def log(msg: str) -> None:
    print(f"[temporal-simple] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_npz_bundle(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def align_bundle_to_anchors(
    anchors_df: pd.DataFrame,
    bundle: Dict[str, np.ndarray],
    name: str,
) -> Dict[str, np.ndarray]:
    anchor_ids = anchors_df["anchor_id"].astype(str).to_numpy()
    bundle_ids = np.array(bundle["anchor_ids"]).astype(str)
    pos = {aid: i for i, aid in enumerate(bundle_ids)}
    missing = [aid for aid in anchor_ids if aid not in pos]
    if missing:
        raise ValueError(f"{name}: missing {len(missing)} anchor_ids from bundle alignment. Example: {missing[:5]}")
    idx = np.array([pos[aid] for aid in anchor_ids], dtype=int)
    out = {}
    for k, v in bundle.items():
        if k == "feature_names":
            out[k] = v
        else:
            out[k] = v[idx]
    return out


def _binary_value(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (bool, np.bool_)):
        return float(int(x))
    sx = str(x).strip().lower()
    if sx in {"true", "1", "1.0", "yes", "y"}:
        return 1.0
    if sx in {"false", "0", "0.0", "no", "n"}:
        return 0.0
    return np.nan


def coerce_binary_series(series: pd.Series) -> pd.Series:
    return series.map(_binary_value).astype(float)


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.unique(np.round(np.concatenate([np.linspace(0.05, 0.95, 19), prob]), 4))
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (prob >= t).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t


def prediction_distribution(prob: np.ndarray) -> Dict[str, float]:
    if prob.size == 0:
        return {}
    return {
        "prob_mean": float(np.mean(prob)),
        "prob_std": float(np.std(prob)),
        "prob_min": float(np.min(prob)),
        "prob_q05": float(np.quantile(prob, 0.05)),
        "prob_q50": float(np.quantile(prob, 0.50)),
        "prob_q95": float(np.quantile(prob, 0.95)),
        "prob_max": float(np.max(prob)),
    }


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(pred)),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, prob))
        except Exception:
            pass
    out.update(prediction_distribution(prob))
    out["positive_rate_pred_at_0_5"] = float(np.mean(prob >= 0.5))
    out["positive_rate_pred_at_threshold"] = float(np.mean(prob >= threshold))
    return out


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
    }


def classification_models(y_train: np.ndarray) -> Dict[str, object]:
    models: Dict[str, object] = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
    }
    if len(np.unique(y_train)) < 2:
        return models
    models["logreg"] = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    models["rf"] = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    models["et"] = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return models


def regression_models() -> Dict[str, object]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "et": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_numeric_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def positive_class_probability(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(pipe, "predict_proba"):
        return pipe.predict(X).astype(float)

    prob = pipe.predict_proba(X)
    if prob.ndim == 1:
        return prob.astype(float)
    if prob.shape[1] == 1:
        classes = getattr(pipe.named_steps["model"], "classes_", np.array([0]))
        positive_class = float(classes[0]) if len(classes) else 0.0
        return np.full(prob.shape[0], positive_class, dtype=float)
    return prob[:, 1].astype(float)


def flatten_modality_bundle(
    bundle: Dict[str, np.ndarray],
    modality: str,
    window: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    X = bundle["X"].astype(np.float32)
    mask = bundle["mask"].astype(np.float32)
    age_days = bundle["age_days"].astype(np.float32)
    feature_names = np.array(bundle["feature_names"]).astype(str)

    seq_len = int(X.shape[1])
    feat_dim = int(X.shape[2])
    take = min(int(window), seq_len)
    start = seq_len - take

    X_recent = X[:, start:, :]
    mask_recent = mask[:, start:]
    age_recent = age_days[:, start:]

    X_recent = np.where(mask_recent[..., None] > 0, X_recent, np.nan)
    age_recent = np.where(mask_recent > 0, age_recent, np.nan)

    value_cols: List[str] = []
    mask_cols: List[str] = []
    age_cols: List[str] = []
    step_labels = list(range(take - 1, -1, -1))

    for lag in step_labels:
        for feat in feature_names:
            value_cols.append(f"{modality}__t_minus_{lag}__{feat}")
        mask_cols.append(f"{modality}__t_minus_{lag}__mask")
        age_cols.append(f"{modality}__t_minus_{lag}__age_days")

    value_flat = X_recent.reshape(X_recent.shape[0], take * feat_dim)
    mask_flat = mask_recent.reshape(mask_recent.shape[0], take)
    age_flat = age_recent.reshape(age_recent.shape[0], take)

    df = pd.concat(
        [
            pd.DataFrame(value_flat, columns=value_cols),
            pd.DataFrame(mask_flat, columns=mask_cols),
            pd.DataFrame(age_flat, columns=age_cols),
        ],
        axis=1,
    )
    meta = {
        "window_used": take,
        "sequence_length_available": seq_len,
        "feature_count_raw": feat_dim,
        "feature_count_flattened": int(df.shape[1]),
    }
    return df, meta


def build_feature_frame(
    anchors: pd.DataFrame,
    dataset_dir: Path,
    enabled_modalities: List[str],
    windows: Dict[str, int],
) -> Tuple[pd.DataFrame, Dict]:
    masks_df = pd.read_csv(dataset_dir / "modality_masks.csv", low_memory=False)
    masks_df["anchor_id"] = masks_df["anchor_id"].astype(str)
    anchors = anchors.copy()
    anchors["anchor_id"] = anchors["anchor_id"].astype(str)
    anchors = anchors.merge(masks_df, on="anchor_id", how="left")

    feature_parts: List[pd.DataFrame] = []
    modality_meta: Dict[str, Dict] = {}

    static_cols = [
        "has_meals",
        "has_days",
        "has_weeks",
        "n_meals_steps_observed",
        "n_days_steps_observed",
        "n_weeks_steps_observed",
    ]
    for col in static_cols:
        if col not in anchors.columns:
            anchors[col] = 0.0
    feature_parts.append(anchors[static_cols].copy().astype(float))

    bundle_names = {
        "days": "days_numeric_sequences.npz",
        "weeks": "weeks_numeric_sequences.npz",
        "meals": "meals_numeric_sequences.npz",
    }

    for modality in enabled_modalities:
        if modality not in bundle_names:
            raise ValueError(f"Unsupported modality: {modality}")
        bundle = load_npz_bundle(dataset_dir / bundle_names[modality])
        bundle = align_bundle_to_anchors(anchors, bundle, modality)
        part_df, part_meta = flatten_modality_bundle(bundle=bundle, modality=modality, window=windows[modality])
        feature_parts.append(part_df)
        modality_meta[modality] = {
            **part_meta,
            "bundle_shape_X": list(bundle["X"].shape),
            "bundle_shape_mask": list(bundle["mask"].shape),
        }

    feature_df = pd.concat(feature_parts, axis=1)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df, {
        "modality_meta": modality_meta,
        "static_cols": static_cols,
        "n_features_total": int(feature_df.shape[1]),
    }


def evaluate_binary_target(
    target_name: str,
    feature_df: pd.DataFrame,
    anchors: pd.DataFrame,
    model_dir: Path,
) -> Dict:
    split = anchors["split_suggested"].astype(str)
    y_all = coerce_binary_series(anchors[target_name])
    valid = y_all.notna()

    train_mask = (split == "train") & valid
    val_mask = (split == "val") & valid
    test_mask = (split == "test") & valid
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(f"{target_name}: missing valid train/val/test rows.")

    X_train = feature_df.loc[train_mask]
    X_val = feature_df.loc[val_mask]
    X_test = feature_df.loc[test_mask]
    y_train = y_all.loc[train_mask].astype(int).to_numpy()
    y_val = y_all.loc[val_mask].astype(int).to_numpy()
    y_test = y_all.loc[test_mask].astype(int).to_numpy()

    rows: List[Dict] = []
    fitted_models: Dict[str, object] = {}
    val_prob_map: Dict[str, np.ndarray] = {}
    test_prob_map: Dict[str, np.ndarray] = {}

    for model_name, model in classification_models(y_train).items():
        pipe = build_numeric_pipeline(model)
        pipe.fit(X_train, y_train)

        val_prob = positive_class_probability(pipe, X_val)
        test_prob = positive_class_probability(pipe, X_test)

        thr = choose_threshold(y_val, val_prob)
        val_fixed = binary_metrics(y_val, val_prob, threshold=0.5)
        val_tuned = binary_metrics(y_val, val_prob, threshold=thr)
        test_fixed = binary_metrics(y_test, test_prob, threshold=0.5)
        test_tuned = binary_metrics(y_test, test_prob, threshold=thr)

        rows.append({
            "target": target_name,
            "task_kind": "binary",
            "model_name": model_name,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "threshold_tuned": float(thr),
            "val_balanced_accuracy_fixed_0_5": val_fixed.get("balanced_accuracy"),
            "val_roc_auc": val_fixed.get("roc_auc"),
            "val_balanced_accuracy_tuned": val_tuned.get("balanced_accuracy"),
            "test_balanced_accuracy_fixed_0_5": test_fixed.get("balanced_accuracy"),
            "test_roc_auc": test_fixed.get("roc_auc"),
            "test_balanced_accuracy_tuned": test_tuned.get("balanced_accuracy"),
            "test_positive_rate_pred_fixed_0_5": test_fixed.get("positive_rate_pred"),
            "test_positive_rate_pred_tuned": test_tuned.get("positive_rate_pred"),
            "test_prob_std": test_fixed.get("prob_std"),
        })

        fitted_models[model_name] = pipe
        val_prob_map[model_name] = val_prob
        test_prob_map[model_name] = test_prob

    comparison_df = pd.DataFrame(rows).sort_values(
        by=["val_balanced_accuracy_tuned", "val_roc_auc", "test_balanced_accuracy_tuned"],
        ascending=[False, False, False],
    )
    best_model_name = str(comparison_df.iloc[0]["model_name"])
    best_threshold = float(comparison_df.iloc[0]["threshold_tuned"])
    best_model = fitted_models[best_model_name]
    best_val_prob = val_prob_map[best_model_name]
    best_test_prob = test_prob_map[best_model_name]

    artifact_path = model_dir / f"{target_name}__{best_model_name}.joblib"
    joblib.dump(best_model, artifact_path)

    val_pred_df = pd.DataFrame({
        "anchor_id": anchors.loc[val_mask, "anchor_id"].astype(str).to_numpy(),
        "split": "val",
        f"{target_name}__mask": 1.0,
        f"{target_name}__true": y_val.astype(float),
        f"{target_name}__prob": best_val_prob.astype(float),
        f"{target_name}__pred_fixed_0_5": (best_val_prob >= 0.5).astype(float),
        f"{target_name}__pred_tuned": (best_val_prob >= best_threshold).astype(float),
    })
    test_pred_df = pd.DataFrame({
        "anchor_id": anchors.loc[test_mask, "anchor_id"].astype(str).to_numpy(),
        "split": "test",
        f"{target_name}__mask": 1.0,
        f"{target_name}__true": y_test.astype(float),
        f"{target_name}__prob": best_test_prob.astype(float),
        f"{target_name}__pred_fixed_0_5": (best_test_prob >= 0.5).astype(float),
        f"{target_name}__pred_tuned": (best_test_prob >= best_threshold).astype(float),
    })

    return {
        "comparison": comparison_df,
        "selected_model": {
            "target": target_name,
            "task_kind": "binary",
            "model_name": best_model_name,
            "threshold": best_threshold,
            "model_artifact": str(artifact_path),
        },
        "val_metrics_fixed_0_5": {target_name: binary_metrics(y_val, best_val_prob, threshold=0.5)},
        "val_metrics_tuned": {target_name: binary_metrics(y_val, best_val_prob, threshold=best_threshold)},
        "test_metrics_fixed_0_5": {target_name: binary_metrics(y_test, best_test_prob, threshold=0.5)},
        "test_metrics_tuned": {target_name: binary_metrics(y_test, best_test_prob, threshold=best_threshold)},
        "val_predictions": val_pred_df,
        "test_predictions": test_pred_df,
    }


def evaluate_regression_target(
    target_name: str,
    feature_df: pd.DataFrame,
    anchors: pd.DataFrame,
    model_dir: Path,
) -> Dict:
    split = anchors["split_suggested"].astype(str)
    y_all = pd.to_numeric(anchors[target_name], errors="coerce")
    valid = y_all.notna()

    train_mask = (split == "train") & valid
    val_mask = (split == "val") & valid
    test_mask = (split == "test") & valid
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(f"{target_name}: missing valid train/val/test rows.")

    X_train = feature_df.loc[train_mask]
    X_val = feature_df.loc[val_mask]
    X_test = feature_df.loc[test_mask]
    y_train = y_all.loc[train_mask].to_numpy(dtype=float)
    y_val = y_all.loc[val_mask].to_numpy(dtype=float)
    y_test = y_all.loc[test_mask].to_numpy(dtype=float)

    rows: List[Dict] = []
    fitted_models: Dict[str, object] = {}
    val_pred_map: Dict[str, np.ndarray] = {}
    test_pred_map: Dict[str, np.ndarray] = {}

    for model_name, model in regression_models().items():
        pipe = build_numeric_pipeline(model)
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val).astype(float)
        test_pred = pipe.predict(X_test).astype(float)

        val_metrics = regression_metrics(y_val, val_pred)
        test_metrics = regression_metrics(y_test, test_pred)

        rows.append({
            "target": target_name,
            "task_kind": "regression",
            "model_name": model_name,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "val_r2": val_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
        })

        fitted_models[model_name] = pipe
        val_pred_map[model_name] = val_pred
        test_pred_map[model_name] = test_pred

    comparison_df = pd.DataFrame(rows).sort_values(
        by=["val_r2", "val_mae", "test_r2"],
        ascending=[False, True, False],
    )
    best_model_name = str(comparison_df.iloc[0]["model_name"])
    best_model = fitted_models[best_model_name]
    best_val_pred = val_pred_map[best_model_name]
    best_test_pred = test_pred_map[best_model_name]

    artifact_path = model_dir / f"{target_name}__{best_model_name}.joblib"
    joblib.dump(best_model, artifact_path)

    val_pred_df = pd.DataFrame({
        "anchor_id": anchors.loc[val_mask, "anchor_id"].astype(str).to_numpy(),
        "split": "val",
        f"{target_name}__mask": 1.0,
        f"{target_name}__true": y_val.astype(float),
        f"{target_name}__pred": best_val_pred.astype(float),
    })
    test_pred_df = pd.DataFrame({
        "anchor_id": anchors.loc[test_mask, "anchor_id"].astype(str).to_numpy(),
        "split": "test",
        f"{target_name}__mask": 1.0,
        f"{target_name}__true": y_test.astype(float),
        f"{target_name}__pred": best_test_pred.astype(float),
    })

    best_val_metrics = regression_metrics(y_val, best_val_pred)
    best_test_metrics = regression_metrics(y_test, best_test_pred)

    return {
        "comparison": comparison_df,
        "selected_model": {
            "target": target_name,
            "task_kind": "regression",
            "model_name": best_model_name,
            "model_artifact": str(artifact_path),
        },
        "val_metrics_fixed_0_5": {target_name: best_val_metrics},
        "val_metrics_tuned": {target_name: best_val_metrics},
        "test_metrics_fixed_0_5": {target_name: best_test_metrics},
        "test_metrics_tuned": {target_name: best_test_metrics},
        "val_predictions": val_pred_df,
        "test_predictions": test_pred_df,
    }


def summarize_metrics(
    binary_metrics_map: Dict[str, Dict[str, float]],
    regression_metrics_map: Dict[str, Dict[str, float]],
) -> Dict:
    bin_bal = [m["balanced_accuracy"] for m in binary_metrics_map.values() if "balanced_accuracy" in m]
    bin_auc = [m["roc_auc"] for m in binary_metrics_map.values() if "roc_auc" in m]
    return {
        "binary": binary_metrics_map,
        "regression": regression_metrics_map,
        "summary": {
            "mean_binary_balanced_accuracy": float(np.mean(bin_bal)) if bin_bal else None,
            "mean_binary_roc_auc": float(np.mean(bin_auc)) if bin_auc else None,
            "n_binary_targets": len(binary_metrics_map),
            "n_regression_targets": len(regression_metrics_map),
        },
    }


def build_binary_diagnostics(binary_metrics_map: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for target_name, metrics in binary_metrics_map.items():
        rows.append({"target": target_name, **metrics, "n": None})
    return pd.DataFrame(rows)


def merge_prediction_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["anchor_id", "split"])
    out = frames[0].copy()
    for frame in frames[1:]:
        out = out.merge(frame, on=["anchor_id", "split"], how="outer")
    return out.sort_values(by=["split", "anchor_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train simple lag-window baselines on the multires temporal dataset.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--dataset-dir", default="training/multires_sequence_dataset", help="Relative path to multires dataset.")
    parser.add_argument("--run-name", default="", help="Optional run name. Auto-generated if omitted.")
    parser.add_argument("--modalities", default="days,weeks", help="Comma-separated enabled modalities.")
    parser.add_argument("--binary-targets", default=",".join(DEFAULT_BINARY_TARGETS), help="Comma-separated binary targets.")
    parser.add_argument("--regression-targets", default=",".join(DEFAULT_REGRESSION_TARGETS), help="Comma-separated regression targets.")
    parser.add_argument("--single-binary-target", default="", help="Optional single binary target override.")
    parser.add_argument("--single-regression-target", default="", help="Optional single regression target override.")
    parser.add_argument("--days-window", type=int, default=DEFAULT_WINDOWS["days"])
    parser.add_argument("--weeks-window", type=int, default=DEFAULT_WINDOWS["weeks"])
    parser.add_argument("--meals-window", type=int, default=DEFAULT_WINDOWS["meals"])
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    dataset_dir = project_root / args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    def _parse_list(raw: str) -> List[str]:
        raw = (raw or "").strip()
        if raw.lower() in {"", "none", "null", "off"}:
            return []
        return [x.strip() for x in raw.split(",") if x.strip()]

    binary_targets = _parse_list(args.binary_targets)
    regression_targets = _parse_list(args.regression_targets)
    if args.single_binary_target:
        binary_targets = [args.single_binary_target.strip()]
    if args.single_regression_target:
        regression_targets = [args.single_regression_target.strip()]
    if not binary_targets and not regression_targets:
        raise ValueError("At least one binary or regression target is required.")

    enabled_modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if not enabled_modalities:
        raise ValueError("At least one modality must be enabled.")

    windows = {
        "days": int(args.days_window),
        "weeks": int(args.weeks_window),
        "meals": int(args.meals_window),
    }

    anchors = pd.read_csv(dataset_dir / "anchors.csv", low_memory=False)
    anchors["anchor_id"] = anchors["anchor_id"].astype(str)

    feature_df, feature_meta = build_feature_frame(
        anchors=anchors,
        dataset_dir=dataset_dir,
        enabled_modalities=enabled_modalities,
        windows=windows,
    )

    target_bits = []
    target_bits.extend([t.replace("y_next_", "").replace("_flag", "").replace("_lb", "") for t in binary_targets])
    target_bits.extend([t.replace("y_next_", "").replace("_flag", "").replace("_lb", "") for t in regression_targets])
    modality_bits = "_".join(enabled_modalities)
    run_name = args.run_name or f"simple_{'_'.join(target_bits)}_{modality_bits}_v1"

    model_dir = project_root / "models" / "temporal_multires" / run_name
    report_dir = project_root / "reports" / "backtests" / "temporal_multires" / run_name
    ensure_dir(model_dir)
    ensure_dir(report_dir)

    save_json(report_dir / "config.json", {
        "project_root": str(project_root),
        "dataset_dir": str(dataset_dir),
        "run_name": run_name,
        "modalities": enabled_modalities,
        "binary_targets": binary_targets,
        "regression_targets": regression_targets,
        "windows": windows,
    })

    split = anchors["split_suggested"].astype(str)
    data_manifest = {
        "n_rows": int(len(anchors)),
        "n_train": int((split == "train").sum()),
        "n_val": int((split == "val").sum()),
        "n_test": int((split == "test").sum()),
        "enabled_modalities": enabled_modalities,
        "windows": windows,
        "feature_build": feature_meta,
    }
    save_json(report_dir / "data_manifest.json", data_manifest)

    comparison_frames: List[pd.DataFrame] = []
    val_prediction_frames: List[pd.DataFrame] = []
    test_prediction_frames: List[pd.DataFrame] = []
    selected_models: List[Dict] = []
    selected_thresholds: Dict[str, float] = {}
    val_binary_fixed: Dict[str, Dict[str, float]] = {}
    val_binary_tuned: Dict[str, Dict[str, float]] = {}
    test_binary_fixed: Dict[str, Dict[str, float]] = {}
    test_binary_tuned: Dict[str, Dict[str, float]] = {}
    val_reg_fixed: Dict[str, Dict[str, float]] = {}
    val_reg_tuned: Dict[str, Dict[str, float]] = {}
    test_reg_fixed: Dict[str, Dict[str, float]] = {}
    test_reg_tuned: Dict[str, Dict[str, float]] = {}

    for target_name in binary_targets:
        log(f"Training binary baselines for {target_name}")
        result = evaluate_binary_target(
            target_name=target_name,
            feature_df=feature_df,
            anchors=anchors,
            model_dir=model_dir,
        )
        comparison_frames.append(result["comparison"])
        val_prediction_frames.append(result["val_predictions"])
        test_prediction_frames.append(result["test_predictions"])
        selected_models.append(result["selected_model"])
        selected_thresholds[target_name] = float(result["selected_model"]["threshold"])
        val_binary_fixed.update(result["val_metrics_fixed_0_5"])
        val_binary_tuned.update(result["val_metrics_tuned"])
        test_binary_fixed.update(result["test_metrics_fixed_0_5"])
        test_binary_tuned.update(result["test_metrics_tuned"])

    for target_name in regression_targets:
        log(f"Training regression baselines for {target_name}")
        result = evaluate_regression_target(
            target_name=target_name,
            feature_df=feature_df,
            anchors=anchors,
            model_dir=model_dir,
        )
        comparison_frames.append(result["comparison"])
        val_prediction_frames.append(result["val_predictions"])
        test_prediction_frames.append(result["test_predictions"])
        selected_models.append(result["selected_model"])
        val_reg_fixed.update(result["val_metrics_fixed_0_5"])
        val_reg_tuned.update(result["val_metrics_tuned"])
        test_reg_fixed.update(result["test_metrics_fixed_0_5"])
        test_reg_tuned.update(result["test_metrics_tuned"])

    comparison_df = pd.concat(comparison_frames, axis=0, ignore_index=True) if comparison_frames else pd.DataFrame()
    comparison_df.to_csv(report_dir / "model_comparison.csv", index=False)

    val_predictions = merge_prediction_frames(val_prediction_frames)
    test_predictions = merge_prediction_frames(test_prediction_frames)
    val_predictions.to_csv(report_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(report_dir / "test_predictions.csv", index=False)

    val_metrics_fixed = summarize_metrics(val_binary_fixed, val_reg_fixed)
    val_metrics_tuned = summarize_metrics(val_binary_tuned, val_reg_tuned)
    test_metrics_fixed = summarize_metrics(test_binary_fixed, test_reg_fixed)
    test_metrics_tuned = summarize_metrics(test_binary_tuned, test_reg_tuned)

    save_json(report_dir / "selected_models.json", {"models": selected_models})
    save_json(report_dir / "selected_thresholds.json", selected_thresholds)
    save_json(report_dir / "val_metrics_fixed_0_5.json", val_metrics_fixed)
    save_json(report_dir / "val_metrics_tuned.json", val_metrics_tuned)
    save_json(report_dir / "test_metrics_fixed_0_5.json", test_metrics_fixed)
    save_json(report_dir / "test_metrics_tuned.json", test_metrics_tuned)

    build_binary_diagnostics(val_binary_tuned).to_csv(report_dir / "val_prediction_diagnostics.csv", index=False)
    build_binary_diagnostics(test_binary_tuned).to_csv(report_dir / "test_prediction_diagnostics.csv", index=False)

    final_summary = {
        "run_name": run_name,
        "model_family": "simple_lag_window_baselines",
        "binary_targets": binary_targets,
        "regression_targets": regression_targets,
        "enabled_modalities": enabled_modalities,
        "windows": windows,
        "selected_models": selected_models,
        "test_metrics_fixed_0_5": test_metrics_fixed,
        "test_metrics_tuned": test_metrics_tuned,
    }
    save_json(report_dir / "final_summary.json", final_summary)

    log(f"Wrote reports to {report_dir}")
    log(f"Wrote model artifacts to {model_dir}")


if __name__ == "__main__":
    main()
