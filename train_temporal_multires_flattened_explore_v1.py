from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_temporal_multires_simple_baselines_v1 import (
    DEFAULT_WINDOWS,
    build_feature_frame,
    ensure_dir,
    log,
    merge_prediction_frames,
    save_json,
)


RANDOM_STATE = 42
TARGET_NAME = "y_next_weight_loss_flag"


def coerce_binary_series(series: pd.Series) -> pd.Series:
    def _coerce(x):
        if pd.isna(x):
            return np.nan
        sx = str(x).strip().lower()
        if sx in {"true", "1", "1.0", "yes", "y"}:
            return 1.0
        if sx in {"false", "0", "0.0", "no", "n"}:
            return 0.0
        return np.nan

    return series.map(_coerce).astype(float)


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.unique(np.round(np.concatenate([np.linspace(0.05, 0.95, 19), prob]), 4))
    best_t = 0.5
    best_score = -1.0
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = float(score)
            best_t = float(thr)
    return best_t


def prediction_distribution(prob: np.ndarray) -> Dict[str, float]:
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
        "positive_rate_pred_at_0_5": float(np.mean(prob >= 0.5)),
        "positive_rate_pred_at_threshold": float(np.mean(prob >= threshold)),
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    out.update(prediction_distribution(prob))
    return out


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def positive_class_probability(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    prob = pipe.predict_proba(X)
    if prob.ndim == 1:
        return prob.astype(float)
    if prob.shape[1] == 1:
        return np.zeros(prob.shape[0], dtype=float)
    return prob[:, 1].astype(float)


def candidate_models() -> Dict[str, object]:
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logreg_balanced": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            C=1.0,
        ),
        "logreg_balanced_c5": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            C=5.0,
        ),
        "rf_balanced": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "et_balanced": ExtraTreesClassifier(
            n_estimators=700,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hgb_depth3": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=300,
            min_samples_leaf=8,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "mlp_small": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            batch_size=32,
            early_stopping=True,
            learning_rate_init=5e-4,
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
        "mlp_wide": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=3e-3,
            batch_size=32,
            early_stopping=True,
            learning_rate_init=5e-4,
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
    }


def parse_candidate_subset(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def evaluate_candidates(
    target_name: str,
    feature_df: pd.DataFrame,
    anchors: pd.DataFrame,
    model_dir: Path,
    candidate_subset: List[str],
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
    fitted_models: Dict[str, Pipeline] = {}
    val_prob_map: Dict[str, np.ndarray] = {}
    test_prob_map: Dict[str, np.ndarray] = {}

    all_candidates = candidate_models()
    selected_names = candidate_subset or list(all_candidates.keys())
    unknown = [name for name in selected_names if name not in all_candidates]
    if unknown:
        raise ValueError(f"Unknown candidate model names: {unknown}")

    for model_name in selected_names:
        model = all_candidates[model_name]
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)

        val_prob = positive_class_probability(pipe, X_val)
        test_prob = positive_class_probability(pipe, X_test)
        thr = choose_threshold(y_val, val_prob)

        val_tuned = binary_metrics(y_val, val_prob, threshold=thr)
        test_tuned = binary_metrics(y_test, test_prob, threshold=thr)

        rows.append(
            {
                "target": target_name,
                "task_kind": "binary",
                "model_name": model_name,
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
                "n_test": int(test_mask.sum()),
                "threshold_tuned": float(thr),
                "val_balanced_accuracy_tuned": val_tuned.get("balanced_accuracy"),
                "val_roc_auc": val_tuned.get("roc_auc"),
                "val_f1_tuned": val_tuned.get("f1"),
                "test_balanced_accuracy_tuned": test_tuned.get("balanced_accuracy"),
                "test_roc_auc": test_tuned.get("roc_auc"),
                "test_f1_tuned": test_tuned.get("f1"),
                "test_positive_rate_pred_tuned": test_tuned.get("positive_rate_pred"),
                "test_prob_std": test_tuned.get("prob_std"),
            }
        )

        fitted_models[model_name] = pipe
        val_prob_map[model_name] = val_prob
        test_prob_map[model_name] = test_prob

    comparison_df = pd.DataFrame(rows).sort_values(
        by=["val_balanced_accuracy_tuned", "val_roc_auc", "test_balanced_accuracy_tuned", "test_roc_auc"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    best_model_name = str(comparison_df.iloc[0]["model_name"])
    best_threshold = float(comparison_df.iloc[0]["threshold_tuned"])
    best_model = fitted_models[best_model_name]
    best_val_prob = val_prob_map[best_model_name]
    best_test_prob = test_prob_map[best_model_name]

    artifact_path = model_dir / f"{target_name}__{best_model_name}.joblib"
    joblib.dump(best_model, artifact_path)

    val_pred_df = pd.DataFrame(
        {
            "anchor_id": anchors.loc[val_mask, "anchor_id"].astype(str).to_numpy(),
            "split": "val",
            f"{target_name}__mask": 1.0,
            f"{target_name}__true": y_val.astype(float),
            f"{target_name}__prob": best_val_prob.astype(float),
            f"{target_name}__pred_fixed_0_5": (best_val_prob >= 0.5).astype(float),
            f"{target_name}__pred_tuned": (best_val_prob >= best_threshold).astype(float),
        }
    )
    test_pred_df = pd.DataFrame(
        {
            "anchor_id": anchors.loc[test_mask, "anchor_id"].astype(str).to_numpy(),
            "split": "test",
            f"{target_name}__mask": 1.0,
            f"{target_name}__true": y_test.astype(float),
            f"{target_name}__prob": best_test_prob.astype(float),
            f"{target_name}__pred_fixed_0_5": (best_test_prob >= 0.5).astype(float),
            f"{target_name}__pred_tuned": (best_test_prob >= best_threshold).astype(float),
        }
    )

    return {
        "comparison": comparison_df,
        "selected_model": {
            "target": target_name,
            "task_kind": "binary",
            "model_name": best_model_name,
            "threshold": best_threshold,
            "model_artifact": str(artifact_path),
        },
        "val_metrics_tuned": {target_name: binary_metrics(y_val, best_val_prob, threshold=best_threshold)},
        "test_metrics_tuned": {target_name: binary_metrics(y_test, best_test_prob, threshold=best_threshold)},
        "val_predictions": val_pred_df,
        "test_predictions": test_pred_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore stronger flattened lag-window classifiers on the multires temporal dataset."
    )
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--dataset-dir", default="training/multires_sequence_dataset")
    parser.add_argument("--run-name", default="flat_loss_daysweeks_explore_v1")
    parser.add_argument("--modalities", default="days,weeks")
    parser.add_argument("--target", default=TARGET_NAME)
    parser.add_argument(
        "--candidate-models",
        default="",
        help="Optional comma-separated subset of candidate model names.",
    )
    parser.add_argument("--days-window", type=int, default=DEFAULT_WINDOWS["days"])
    parser.add_argument("--weeks-window", type=int, default=DEFAULT_WINDOWS["weeks"])
    parser.add_argument("--meals-window", type=int, default=DEFAULT_WINDOWS["meals"])
    args = parser.parse_args()

    candidate_subset = parse_candidate_subset(args.candidate_models)
    project_root = Path(args.project_root).expanduser().resolve()
    dataset_dir = project_root / args.dataset_dir
    enabled_modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    windows: Dict[str, int] = {
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

    model_dir = project_root / "models" / "temporal_multires" / args.run_name
    report_dir = project_root / "reports" / "backtests" / "temporal_multires" / args.run_name
    ensure_dir(model_dir)
    ensure_dir(report_dir)

    save_json(
        report_dir / "config.json",
        {
            "project_root": str(project_root),
            "dataset_dir": str(dataset_dir),
            "run_name": args.run_name,
            "target": args.target,
            "modalities": enabled_modalities,
            "windows": windows,
            "candidate_models": candidate_subset or list(candidate_models().keys()),
        },
    )
    split = anchors["split_suggested"].astype(str)
    save_json(
        report_dir / "data_manifest.json",
        {
            "n_rows": int(len(anchors)),
            "n_train": int((split == "train").sum()),
            "n_val": int((split == "val").sum()),
            "n_test": int((split == "test").sum()),
            "feature_build": feature_meta,
        },
    )

    log(f"Training flattened exploration models for {args.target}")
    result = evaluate_candidates(
        target_name=args.target,
        feature_df=feature_df,
        anchors=anchors,
        model_dir=model_dir,
        candidate_subset=candidate_subset,
    )
    result["comparison"].to_csv(report_dir / "model_comparison.csv", index=False)
    merge_prediction_frames([result["val_predictions"]]).to_csv(report_dir / "val_predictions.csv", index=False)
    merge_prediction_frames([result["test_predictions"]]).to_csv(report_dir / "test_predictions.csv", index=False)
    save_json(report_dir / "selected_models.json", {"models": [result["selected_model"]]})
    save_json(report_dir / "selected_thresholds.json", {args.target: result["selected_model"]["threshold"]})
    save_json(report_dir / "val_metrics_tuned.json", {"binary": result["val_metrics_tuned"], "regression": {}, "summary": {}})
    save_json(report_dir / "test_metrics_tuned.json", {"binary": result["test_metrics_tuned"], "regression": {}, "summary": {}})
    save_json(
        report_dir / "final_summary.json",
        {
            "run_name": args.run_name,
            "model_family": "flattened_window_explore",
            "binary_targets": [args.target],
            "regression_targets": [],
            "enabled_modalities": enabled_modalities,
            "windows": windows,
            "selected_models": [result["selected_model"]],
            "test_metrics_tuned": {"binary": result["test_metrics_tuned"], "regression": {}, "summary": {}},
        },
    )

    top_rows = result["comparison"].head(5)
    top_rows_table = top_rows.to_csv(index=False).strip()
    report_lines = [
        f"# Flattened Temporal Explore: {args.run_name}",
        "",
        f"- target: `{args.target}`",
        f"- modalities: `{','.join(enabled_modalities)}`",
        "",
        "## Top Candidates",
        "",
        "```csv",
        top_rows_table,
        "```",
        "",
    ]
    (report_dir / "summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    log(f"Wrote reports to {report_dir}")
    log(f"Wrote model artifacts to {model_dir}")


if __name__ == "__main__":
    main()
