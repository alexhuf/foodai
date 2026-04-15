from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_temporal_multires_simple_baselines_v1 import (
    build_feature_frame,
    choose_threshold,
    coerce_binary_series,
    ensure_dir,
    prediction_distribution,
    save_json,
)


TARGET_NAME = "y_next_weight_loss_flag"
DEFAULT_WINDOWS = {"days": 7, "weeks": 4, "meals": 10}
DEFAULT_SEEDS = [7, 21, 42, 84, 126]


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def probability_series(pipe, X: pd.DataFrame) -> np.ndarray:
    prob = pipe.predict_proba(X)
    if prob.ndim == 1:
        return prob.astype(float)
    return prob[:, 1].astype(float)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(pred)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    out.update(prediction_distribution(prob))
    return out


def threshold_table(y_true: np.ndarray, prob: np.ndarray, thresholds: List[float], selected_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    seen = set()
    for threshold in sorted(thresholds):
        rounded = round(float(threshold), 4)
        if rounded in seen:
            continue
        seen.add(rounded)
        metrics = classification_metrics(y_true, prob, rounded)
        metrics["is_selected_threshold"] = int(abs(rounded - selected_threshold) < 1e-9)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def summarize_feature_groups(importances: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feature_name, importance in zip(importances["feature"], importances["importance"]):
        parts = str(feature_name).split("__")
        modality = parts[0] if len(parts) > 1 else "static"
        lag = "static"
        feature_core = feature_name
        if len(parts) >= 3 and parts[1].startswith("t_minus_"):
            lag = parts[1]
            feature_core = "__".join(parts[2:])
        elif len(parts) >= 2:
            feature_core = "__".join(parts[1:])
        rows.append(
            {
                "feature": feature_name,
                "modality": modality,
                "lag": lag,
                "feature_core": feature_core,
                "importance": float(importance),
            }
        )
    exploded = pd.DataFrame(rows)
    groupings = [
        ("feature_group_by_modality.csv", ["modality"]),
        ("feature_group_by_lag.csv", ["lag"]),
        ("feature_group_by_modality_and_lag.csv", ["modality", "lag"]),
        ("feature_group_by_feature_core.csv", ["feature_core"]),
    ]
    outputs = {}
    for name, cols in groupings:
        grouped = (
            exploded.groupby(cols, dropna=False)["importance"]
            .sum()
            .reset_index()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        outputs[name] = grouped
    return outputs


def make_probability_diagnostics(pred_df: pd.DataFrame, selected_threshold: float) -> Tuple[pd.DataFrame, Dict]:
    prob_col = f"{TARGET_NAME}__prob"
    true_col = f"{TARGET_NAME}__true"
    by_class = []
    for label in [0.0, 1.0]:
        sub = pred_df[pred_df[true_col] == label]
        if sub.empty:
            continue
        stats = prediction_distribution(sub[prob_col].to_numpy(dtype=float))
        by_class.append({"true_label": int(label), "n": int(len(sub)), **stats})

    bin_edges = np.array([0.0, 0.2, 0.3, 0.35, 0.4, selected_threshold, 0.45, 0.5, 0.6, 1.0], dtype=float)
    bin_edges = np.unique(np.clip(bin_edges, 0.0, 1.0))
    clipped = np.clip(pred_df[prob_col].to_numpy(dtype=float), 0.0, 1.0)
    hist = pd.cut(clipped, bins=bin_edges, include_lowest=True, right=False)
    hist_df = (
        pred_df.assign(prob_bin=hist)
        .groupby(["prob_bin", true_col], dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={true_col: "true_label"})
    )
    summary = {
        "overall": {
            "n": int(len(pred_df)),
            **prediction_distribution(pred_df[prob_col].to_numpy(dtype=float)),
        },
        "by_true_class": by_class,
    }
    return hist_df, summary


def permutation_subset(
    pipe,
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float,
    top_feature_names: List[str],
    n_repeats: int,
    random_state: int,
) -> Dict[str, pd.DataFrame]:
    def score_map(X_eval: pd.DataFrame) -> Dict[str, float]:
        prob = probability_series(pipe, X_eval)
        pred = (prob >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y, prob)),
            "balanced_accuracy_at_selected_threshold": float(balanced_accuracy_score(y, pred)),
        }

    base_scores = score_map(X)
    rng = np.random.default_rng(random_state)
    outputs: Dict[str, List[Dict[str, float]]] = {
        "roc_auc": [],
        "balanced_accuracy_at_selected_threshold": [],
    }
    X_work = X.copy()
    for feature_name in top_feature_names:
        metric_values = {key: [] for key in outputs.keys()}
        original_values = X_work[feature_name].to_numpy(copy=True)
        for _ in range(n_repeats):
            shuffled = original_values[rng.permutation(len(original_values))]
            X_work.loc[:, feature_name] = shuffled
            scores = score_map(X_work)
            for metric_name, base_score in base_scores.items():
                metric_values[metric_name].append(base_score - scores[metric_name])
        X_work.loc[:, feature_name] = original_values
        for metric_name, values in metric_values.items():
            outputs[metric_name].append(
                {
                    "feature": feature_name,
                    "importance_mean": float(np.mean(values)),
                    "importance_std": float(np.std(values, ddof=0)),
                }
            )

    final_outputs = {}
    for metric_name, rows in outputs.items():
        final_outputs[metric_name] = (
            pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        )
    return final_outputs


def false_case_tables(pred_df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    true_col = f"{TARGET_NAME}__true"
    prob_col = f"{TARGET_NAME}__prob"
    pred_df = pred_df.copy()
    pred_df["pred_label"] = (pred_df[prob_col] >= threshold).astype(int)
    pred_df["margin_vs_threshold"] = pred_df[prob_col] - threshold

    false_pos = pred_df[(pred_df[true_col] == 0.0) & (pred_df["pred_label"] == 1)].copy()
    false_neg = pred_df[(pred_df[true_col] == 1.0) & (pred_df["pred_label"] == 0)].copy()

    false_pos = false_pos.sort_values([prob_col, "anchor_id"], ascending=[False, True]).reset_index(drop=True)
    false_neg = false_neg.sort_values([prob_col, "anchor_id"], ascending=[True, True]).reset_index(drop=True)
    return false_pos, false_neg


def comparison_frame(project_root: Path, anchors: pd.DataFrame, run_names: List[str]) -> pd.DataFrame:
    valid_anchor_ids = set(
        anchors.loc[coerce_binary_series(anchors[TARGET_NAME]).notna() & (anchors["split_suggested"].astype(str) == "test"), "anchor_id"]
        .astype(str)
        .tolist()
    )
    rows = []
    for run_name in run_names:
        pred_path = project_root / "reports" / "backtests" / "temporal_multires" / run_name / "test_predictions.csv"
        if not pred_path.exists():
            continue
        df = pd.read_csv(pred_path)
        df["anchor_id"] = df["anchor_id"].astype(str)
        prob_col = f"{TARGET_NAME}__prob"
        true_col = f"{TARGET_NAME}__true"
        pred_col = f"{TARGET_NAME}__pred_tuned"
        if pred_col not in df.columns:
            pred_col = f"{TARGET_NAME}__pred"

        metrics_payload = read_json(
            project_root / "reports" / "backtests" / "temporal_multires" / run_name / "test_metrics_tuned.json"
        )
        reported = metrics_payload["binary"][TARGET_NAME]
        threshold = float(reported["threshold"])

        shared = df[df["anchor_id"].isin(valid_anchor_ids)].copy()
        shared_y = shared[true_col].to_numpy(dtype=float).astype(int)
        shared_prob = shared[prob_col].to_numpy(dtype=float)
        strict = classification_metrics(shared_y, shared_prob, threshold=threshold)
        rows.append(
            {
                "run_name": run_name,
                "n_reported_rows": int(len(df)),
                "n_shared_valid_rows": int(len(shared)),
                "reported_threshold": threshold,
                "reported_balanced_accuracy": float(reported["balanced_accuracy"]),
                "reported_roc_auc": float(reported.get("roc_auc", np.nan)),
                "reported_prob_std": float(reported.get("prob_std", np.nan)),
                "shared_valid_balanced_accuracy": float(strict["balanced_accuracy"]),
                "shared_valid_roc_auc": float(strict.get("roc_auc", np.nan)),
                "shared_valid_prob_std": float(strict.get("prob_std", np.nan)),
                "shared_valid_positive_rate_pred": float(strict["positive_rate_pred"]),
            }
        )
    return pd.DataFrame(rows).sort_values("shared_valid_balanced_accuracy", ascending=False).reset_index(drop=True)


def repeated_seed_robustness(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    seeds: List[int],
) -> Tuple[pd.DataFrame, Dict]:
    rows = []
    for seed in seeds:
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=-1,
        )
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        val_prob = probability_series(pipe, X_val)
        test_prob = probability_series(pipe, X_test)
        threshold = choose_threshold(y_val, val_prob)
        metrics = classification_metrics(y_test, test_prob, threshold=threshold)
        rows.append(
            {
                "seed": int(seed),
                "threshold": float(threshold),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "roc_auc": float(metrics.get("roc_auc", np.nan)),
                "f1": float(metrics["f1"]),
                "positive_rate_pred": float(metrics["positive_rate_pred"]),
                "prob_std": float(metrics["prob_std"]),
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "tn": int(metrics["tn"]),
                "fn": int(metrics["fn"]),
            }
        )

    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary = {
        "n_seeds": int(len(frame)),
        "balanced_accuracy_mean": float(frame["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(frame["balanced_accuracy"].std(ddof=0)),
        "balanced_accuracy_min": float(frame["balanced_accuracy"].min()),
        "balanced_accuracy_max": float(frame["balanced_accuracy"].max()),
        "roc_auc_mean": float(frame["roc_auc"].mean()),
        "roc_auc_std": float(frame["roc_auc"].std(ddof=0)),
        "roc_auc_min": float(frame["roc_auc"].min()),
        "roc_auc_max": float(frame["roc_auc"].max()),
        "prob_std_mean": float(frame["prob_std"].mean()),
        "threshold_mean": float(frame["threshold"].mean()),
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the current flattened temporal winner and bounded robustness.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-name", default="simple_loss_daysweeks_v2")
    parser.add_argument("--target", default=TARGET_NAME)
    parser.add_argument("--analysis-name", default="")
    parser.add_argument("--permutation-top-k", type=int, default=50)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--seed-list", default=",".join(str(x) for x in DEFAULT_SEEDS))
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    run_name = args.run_name
    analysis_name = args.analysis_name or f"{run_name}_winner_analysis_v1"
    target_name = args.target

    report_dir = project_root / "reports" / "backtests" / "temporal_multires" / analysis_name
    ensure_dir(report_dir)

    run_dir = project_root / "reports" / "backtests" / "temporal_multires" / run_name
    config = read_json(run_dir / "config.json")
    selected = read_json(run_dir / "selected_models.json")["models"][0]
    selected_threshold = float(read_json(run_dir / "selected_thresholds.json")[target_name])

    dataset_dir = Path(config["dataset_dir"])
    anchors = pd.read_csv(dataset_dir / "anchors.csv", low_memory=False)
    anchors["anchor_id"] = anchors["anchor_id"].astype(str)
    feature_df, feature_meta = build_feature_frame(
        anchors=anchors,
        dataset_dir=dataset_dir,
        enabled_modalities=list(config["modalities"]),
        windows=dict(config["windows"]),
    )

    y_all = coerce_binary_series(anchors[target_name])
    valid = y_all.notna()
    split = anchors["split_suggested"].astype(str)
    train_mask = (split == "train") & valid
    val_mask = (split == "val") & valid
    test_mask = (split == "test") & valid

    X_train = feature_df.loc[train_mask].copy()
    X_val = feature_df.loc[val_mask].copy()
    X_test = feature_df.loc[test_mask].copy()
    y_train = y_all.loc[train_mask].astype(int).to_numpy()
    y_val = y_all.loc[val_mask].astype(int).to_numpy()
    y_test = y_all.loc[test_mask].astype(int).to_numpy()

    pipe = joblib.load(selected["model_artifact"])
    model = pipe.named_steps["model"]
    test_prob = probability_series(pipe, X_test)
    val_prob = probability_series(pipe, X_val)

    impurity_df = pd.DataFrame(
        {
            "feature": X_train.columns.astype(str),
            "importance": np.asarray(getattr(model, "feature_importances_"), dtype=float),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    impurity_df.to_csv(report_dir / "feature_importance.csv", index=False)

    grouped_outputs = summarize_feature_groups(impurity_df)
    for filename, frame in grouped_outputs.items():
        frame.to_csv(report_dir / filename, index=False)

    top_features = impurity_df["feature"].head(int(args.permutation_top_k)).tolist()
    permutation_outputs = permutation_subset(
        pipe=pipe,
        X=X_test,
        y=y_test,
        threshold=selected_threshold,
        top_feature_names=top_features,
        n_repeats=int(args.permutation_repeats),
        random_state=42,
    )
    for label, frame in permutation_outputs.items():
        frame.to_csv(report_dir / f"permutation_importance_{label}.csv", index=False)

    threshold_candidates = list(np.linspace(0.20, 0.60, 17))
    threshold_candidates.extend(val_prob.tolist())
    threshold_candidates.extend(test_prob.tolist())
    threshold_candidates.append(selected_threshold)
    threshold_df = threshold_table(
        y_true=y_test,
        prob=test_prob,
        thresholds=threshold_candidates,
        selected_threshold=selected_threshold,
    )
    threshold_df.to_csv(report_dir / "threshold_sweep_test.csv", index=False)

    test_pred_df = pd.read_csv(run_dir / "test_predictions.csv")
    test_pred_df["anchor_id"] = test_pred_df["anchor_id"].astype(str)
    context_cols = ["anchor_id", "anchor_period_start", "anchor_next_period_start", "y_next_weight_delta_lb"]
    context_df = anchors[context_cols].copy()
    merged_test = test_pred_df.merge(context_df, on="anchor_id", how="left")

    confusion_metrics = classification_metrics(y_test, test_prob, threshold=selected_threshold)
    confusion_df = pd.DataFrame(
        [
            {"outcome": "true_negative", "count": confusion_metrics["tn"]},
            {"outcome": "false_positive", "count": confusion_metrics["fp"]},
            {"outcome": "false_negative", "count": confusion_metrics["fn"]},
            {"outcome": "true_positive", "count": confusion_metrics["tp"]},
        ]
    )
    confusion_df["share_of_test"] = confusion_df["count"] / max(int(len(y_test)), 1)
    confusion_df.to_csv(report_dir / "confusion_breakdown_test.csv", index=False)

    false_pos, false_neg = false_case_tables(merged_test, selected_threshold)
    false_pos.to_csv(report_dir / "hardest_false_positives.csv", index=False)
    false_neg.to_csv(report_dir / "hardest_false_negatives.csv", index=False)

    hist_df, prob_summary = make_probability_diagnostics(merged_test, selected_threshold=selected_threshold)
    hist_df.to_csv(report_dir / "probability_histogram_test.csv", index=False)
    save_json(report_dir / "probability_diagnostics.json", prob_summary)

    compare_runs = [
        "simple_loss_daysweeks_v2",
        "flat_loss_daysweeks_followup_pilot_v1",
        "gru_loss_daysweeks_smoke_v4_1",
        "tcn_loss_daysweeks_compare_smoke_v1_check",
    ]
    comparison_df = comparison_frame(project_root=project_root, anchors=anchors, run_names=compare_runs)
    comparison_df.to_csv(report_dir / "winner_comparison.csv", index=False)

    seeds = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]
    robustness_df, robustness_summary = repeated_seed_robustness(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        seeds=seeds,
    )
    robustness_df.to_csv(report_dir / "repeated_seed_robustness.csv", index=False)
    save_json(report_dir / "repeated_seed_robustness_summary.json", robustness_summary)

    top_drivers = impurity_df.head(10)["feature"].tolist()
    top_perm = permutation_outputs["roc_auc"].head(10)["feature"].tolist()
    hard_fp_count = int(len(false_pos))
    hard_fn_count = int(len(false_neg))
    best_compare_row = comparison_df.iloc[0].to_dict() if not comparison_df.empty else {}

    summary_lines = [
        f"# Winner Analysis: {run_name}",
        "",
        f"- target: `{target_name}`",
        f"- modalities: `{','.join(config['modalities'])}`",
        f"- winning model artifact: `{selected['model_artifact']}`",
        f"- selected threshold: `{selected_threshold:.4f}`",
        "",
        "## Current Winner",
        "",
        f"- test balanced accuracy: `{confusion_metrics['balanced_accuracy']:.4f}`",
        f"- test ROC AUC: `{confusion_metrics.get('roc_auc', float('nan')):.4f}`",
        f"- test probability std: `{confusion_metrics['prob_std']:.4f}`",
        f"- test confusion: `TN={confusion_metrics['tn']}, FP={confusion_metrics['fp']}, FN={confusion_metrics['fn']}, TP={confusion_metrics['tp']}`",
        "",
        "## Top Drivers",
        "",
        f"- top impurity features: `{', '.join(top_drivers[:5])}`",
        f"- top permutation ROC-AUC features: `{', '.join(top_perm[:5])}`",
        "",
        "## Robustness",
        "",
        f"- repeated-seed balanced accuracy mean/std: `{robustness_summary['balanced_accuracy_mean']:.4f} +/- {robustness_summary['balanced_accuracy_std']:.4f}`",
        f"- repeated-seed ROC AUC mean/std: `{robustness_summary['roc_auc_mean']:.4f} +/- {robustness_summary['roc_auc_std']:.4f}`",
        "",
        "## Comparison",
        "",
        (
            f"- best shared-valid comparison row: `{best_compare_row.get('run_name', 'n/a')}` "
            f"(balanced_accuracy={best_compare_row.get('shared_valid_balanced_accuracy', float('nan')):.4f}, "
            f"roc_auc={best_compare_row.get('shared_valid_roc_auc', float('nan')):.4f})"
            if best_compare_row
            else "- comparison unavailable"
        ),
        "",
        "## Remaining Failure Modes",
        "",
        f"- false positives on test: `{hard_fp_count}`",
        f"- false negatives on test: `{hard_fn_count}`",
        "- neural comparison files include 7 test rows whose target is missing in the anchor table, so strict cross-run comparisons should use the shared-valid table here.",
        "",
    ]
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    save_json(
        report_dir / "analysis_manifest.json",
        {
            "source_run_name": run_name,
            "analysis_name": analysis_name,
            "target": target_name,
            "selected_threshold": selected_threshold,
            "feature_build": feature_meta,
            "permutation_top_k": int(args.permutation_top_k),
            "permutation_repeats": int(args.permutation_repeats),
            "robustness_seeds": seeds,
            "report_files": sorted([p.name for p in report_dir.iterdir() if p.is_file()]),
        },
    )


if __name__ == "__main__":
    main()
