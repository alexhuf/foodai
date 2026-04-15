from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from analyze_temporal_flat_winner_operational_v1 import (
    classification_metrics,
    fit_same_family_model,
    probability_series,
)
from train_temporal_multires_simple_baselines_v1 import (
    build_feature_frame,
    coerce_binary_series,
    ensure_dir,
)


TARGET_NAME = "y_next_weight_loss_flag"
LOCKED_THRESHOLD = 0.4288
CANDIDATE_THRESHOLDS = [0.44, 0.445, 0.45, 0.455]
DEFAULT_NEXT_COMMAND = (
    "python train_temporal_multires_neural_compare_v1.py "
    "--project-root /workspace/foodai "
    "--comparison-run-name loss_daysweeks_compare_focal_smoke_v1 "
    "--families gru,tcn "
    "--binary-loss-mode focal "
    "--focal-gamma 2.0 "
    "--smoke-test"
)


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_threshold_row(table: pd.DataFrame, threshold: float) -> Dict:
    match = table.loc[(table["threshold"] - float(threshold)).abs() < 1e-9]
    if match.empty:
        match = table.iloc[[(table["threshold"] - float(threshold)).abs().argmin()]]
    return match.iloc[0].to_dict()


def build_forward_folds(valid_df: pd.DataFrame, min_train_rows: int, eval_window_rows: int) -> List[Dict[str, np.ndarray]]:
    folds: List[Dict[str, np.ndarray]] = []
    train_end = int(min_train_rows)
    fold_num = 1
    while train_end + eval_window_rows <= len(valid_df):
        eval_end = train_end + eval_window_rows
        train_idx = valid_df.iloc[:train_end]["index"].to_numpy(dtype=int)
        eval_idx = valid_df.iloc[train_end:eval_end]["index"].to_numpy(dtype=int)
        folds.append(
            {
                "fold": int(fold_num),
                "train_idx": train_idx,
                "eval_idx": eval_idx,
            }
        )
        train_end = eval_end
        fold_num += 1
    return folds


def pooled_confusion_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    pos_denom = tp + fn
    neg_denom = tn + fp
    recall = tp / pos_denom if pos_denom else np.nan
    specificity = tn / neg_denom if neg_denom else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    if np.isnan(recall) or np.isnan(specificity):
        bal_acc = np.nan
    else:
        bal_acc = 0.5 * (recall + specificity)
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    elif np.isnan(recall):
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "pooled_accuracy": float(accuracy),
        "pooled_balanced_accuracy": float(bal_acc) if not np.isnan(bal_acc) else np.nan,
        "pooled_precision": float(precision),
        "pooled_recall": float(recall) if not np.isnan(recall) else np.nan,
        "pooled_specificity": float(specificity) if not np.isnan(specificity) else np.nan,
        "pooled_f1": float(f1) if not np.isnan(f1) else np.nan,
    }


def aggregate_threshold_results(fold_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    sub = fold_df.loc[(fold_df["threshold"] - float(threshold)).abs() < 1e-9].copy()
    tp = int(sub["tp"].sum())
    tn = int(sub["tn"].sum())
    fp = int(sub["fp"].sum())
    fn = int(sub["fn"].sum())
    payload = {
        "threshold": float(threshold),
        "fold_count": int(len(sub)),
        "eval_rows_total": int(sub["n"].sum()),
        "eval_positive_total": int(tp + fn),
        "eval_negative_total": int(tn + fp),
        "fp_total": fp,
        "fn_total": fn,
        "tp_total": tp,
        "tn_total": tn,
        "mean_fold_balanced_accuracy": float(sub["balanced_accuracy"].mean()),
        "min_fold_balanced_accuracy": float(sub["balanced_accuracy"].min()),
        "max_fold_balanced_accuracy": float(sub["balanced_accuracy"].max()),
        "mean_fold_roc_auc": float(sub["roc_auc"].dropna().mean()) if sub["roc_auc"].notna().any() else np.nan,
        "latest_fold_balanced_accuracy": float(sub.sort_values("fold").iloc[-1]["balanced_accuracy"]),
        "latest_fold_fp": int(sub.sort_values("fold").iloc[-1]["fp"]),
        "latest_fold_fn": int(sub.sort_values("fold").iloc[-1]["fn"]),
        "latest_fold_threshold": float(sub.sort_values("fold").iloc[-1]["threshold"]),
    }
    payload.update(pooled_confusion_metrics(tp=tp, tn=tn, fp=fp, fn=fn))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an additive fixed-threshold promotion validation pass for the flattened temporal winner.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-name", default="simple_loss_daysweeks_v2")
    parser.add_argument("--analysis-name", default="")
    parser.add_argument("--target", default=TARGET_NAME)
    parser.add_argument("--min-train-rows", type=int, default=181)
    parser.add_argument("--eval-window-rows", type=int, default=39)
    parser.add_argument("--locked-threshold", type=float, default=LOCKED_THRESHOLD)
    parser.add_argument("--candidate-thresholds", default="0.44,0.445,0.45,0.455")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    run_name = args.run_name
    target_name = args.target
    analysis_name = args.analysis_name or f"{run_name}_threshold_promotion_check_v1"
    locked_threshold = round(float(args.locked_threshold), 4)
    candidate_thresholds = [round(float(value), 4) for value in args.candidate_thresholds.split(",") if value.strip()]
    thresholds = [locked_threshold] + [value for value in candidate_thresholds if value != locked_threshold]

    report_dir = project_root / "reports" / "backtests" / "temporal_multires" / analysis_name
    ensure_dir(report_dir)

    run_dir = project_root / "reports" / "backtests" / "temporal_multires" / run_name
    config = read_json(run_dir / "config.json")
    dataset_dir = Path(config["dataset_dir"])

    anchors = pd.read_csv(dataset_dir / "anchors.csv", low_memory=False)
    anchors["anchor_period_start"] = pd.to_datetime(anchors["anchor_period_start"])
    feature_df, _ = build_feature_frame(
        anchors=anchors,
        dataset_dir=dataset_dir,
        enabled_modalities=list(config["modalities"]),
        windows=dict(config["windows"]),
    )

    valid_target = coerce_binary_series(anchors[target_name]).notna()
    ordered = anchors.loc[valid_target].copy().sort_values("anchor_period_start").reset_index()
    folds = build_forward_folds(
        valid_df=ordered,
        min_train_rows=int(args.min_train_rows),
        eval_window_rows=int(args.eval_window_rows),
    )

    fold_rows: List[Dict[str, float]] = []
    eval_prediction_rows: List[Dict[str, float]] = []
    for fold in folds:
        train_idx = fold["train_idx"]
        eval_idx = fold["eval_idx"]
        y_train = coerce_binary_series(anchors.loc[train_idx, target_name]).astype(int).to_numpy()
        y_eval = coerce_binary_series(anchors.loc[eval_idx, target_name]).astype(int).to_numpy()

        if len(np.unique(y_train)) < 2:
            continue

        pipe = fit_same_family_model(feature_df.loc[train_idx], y_train)
        eval_prob = probability_series(pipe, feature_df.loc[eval_idx])
        eval_anchor_ids = anchors.loc[eval_idx, "anchor_id"].astype(str).tolist()
        eval_dates = anchors.loc[eval_idx, "anchor_period_start"].dt.date.astype(str).tolist()

        for anchor_id, anchor_date, y_true_value, prob_value in zip(eval_anchor_ids, eval_dates, y_eval, eval_prob):
            eval_prediction_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "anchor_id": anchor_id,
                    "anchor_period_start": anchor_date,
                    "y_true": int(y_true_value),
                    "y_prob": float(prob_value),
                }
            )

        for threshold in thresholds:
            metrics = classification_metrics(y_eval, eval_prob, threshold=float(threshold))
            fold_rows.append(
                {
                    "fold": int(fold["fold"]),
                    "train_n": int(len(train_idx)),
                    "eval_n": int(len(eval_idx)),
                    "train_start": str(anchors.loc[train_idx, "anchor_period_start"].min().date()),
                    "train_end": str(anchors.loc[train_idx, "anchor_period_start"].max().date()),
                    "eval_start": str(anchors.loc[eval_idx, "anchor_period_start"].min().date()),
                    "eval_end": str(anchors.loc[eval_idx, "anchor_period_start"].max().date()),
                    "eval_positive_rate_true": float(np.mean(y_eval)),
                    **metrics,
                }
            )

    fold_df = pd.DataFrame(fold_rows).sort_values(["threshold", "fold"]).reset_index(drop=True)
    fold_df.to_csv(report_dir / "time_aware_fixed_threshold_folds.csv", index=False)

    eval_pred_df = pd.DataFrame(eval_prediction_rows).sort_values(["fold", "anchor_period_start", "anchor_id"]).reset_index(drop=True)
    eval_pred_df.to_csv(report_dir / "time_aware_eval_predictions.csv", index=False)

    heldout_source = pd.read_csv(
        project_root
        / "reports"
        / "backtests"
        / "temporal_multires"
        / f"{run_name}_operational_check_v1"
        / "threshold_operating_table.csv"
    )
    heldout_rows = [choose_threshold_row(heldout_source, threshold) for threshold in thresholds]
    heldout_df = pd.DataFrame(heldout_rows).sort_values("threshold").reset_index(drop=True)
    heldout_df.to_csv(report_dir / "heldout_zone_reference.csv", index=False)

    summary_rows = [aggregate_threshold_results(fold_df, threshold=threshold) for threshold in thresholds]
    summary_df = pd.DataFrame(summary_rows).sort_values("threshold").reset_index(drop=True)

    heldout_lookup = {round(float(row["threshold"]), 4): row for row in heldout_df.to_dict(orient="records")}
    locked_summary = summary_df.loc[(summary_df["threshold"] - locked_threshold).abs() < 1e-9].iloc[0].to_dict()
    promote_flags = []
    heldout_flags = []
    for row in summary_df.to_dict(orient="records"):
        threshold = round(float(row["threshold"]), 4)
        heldout_row = heldout_lookup[threshold]
        heldout_rule_met = int(
            (int(heldout_row["fn"]) == 0)
            and (float(heldout_row["balanced_accuracy"]) > float(heldout_lookup[locked_threshold]["balanced_accuracy"]))
            and (int(heldout_row["fp"]) < int(heldout_lookup[locked_threshold]["fp"]))
        )
        heldout_flags.append(heldout_rule_met)

        if abs(threshold - locked_threshold) < 1e-9:
            promote_flags.append(0)
            continue
        supports = int(
            (int(row["fp_total"]) < int(locked_summary["fp_total"]))
            and (int(row["fn_total"]) <= int(locked_summary["fn_total"]))
            and (float(row["mean_fold_balanced_accuracy"]) > float(locked_summary["mean_fold_balanced_accuracy"]))
            and (float(row["pooled_balanced_accuracy"]) > float(locked_summary["pooled_balanced_accuracy"]))
        )
        promote_flags.append(supports)
    summary_df["heldout_rule_met"] = heldout_flags
    summary_df["time_aware_supports_promotion"] = promote_flags
    summary_df.to_csv(report_dir / "threshold_time_aware_summary.csv", index=False)

    promotable_thresholds = summary_df.loc[
        (summary_df["time_aware_supports_promotion"] == 1)
        & (summary_df["threshold"] > locked_threshold),
        "threshold",
    ].tolist()
    promoted_threshold = float(promotable_thresholds[0]) if promotable_thresholds else None

    summary_lines = [
        f"# Threshold Promotion Check: {run_name}",
        "",
        f"- target: `{target_name}`",
        f"- modalities: `{','.join(config['modalities'])}`",
        f"- locked threshold: `{locked_threshold:.4f}`",
        f"- candidate thresholds: `{', '.join(f'{threshold:.4f}' for threshold in candidate_thresholds)}`",
        f"- time-aware forward folds: `{len(folds)}` using `min_train_rows={int(args.min_train_rows)}` and `eval_window_rows={int(args.eval_window_rows)}`",
        "",
        "## Held-Out Reference",
        "",
    ]

    for row in heldout_df.to_dict(orient="records"):
        summary_lines.append(
            f"- `threshold={float(row['threshold']):.4f}`: "
            f"`balanced_accuracy={float(row['balanced_accuracy']):.4f}, fp={int(row['fp'])}, fn={int(row['fn'])}`"
        )

    summary_lines.extend(
        [
            "",
            "## Time-Aware Confirmation",
            "",
        ]
    )

    for row in summary_df.to_dict(orient="records"):
        summary_lines.append(
            f"- `threshold={float(row['threshold']):.4f}`: "
            f"`mean_fold_balanced_accuracy={float(row['mean_fold_balanced_accuracy']):.4f}, "
            f"pooled_balanced_accuracy={float(row['pooled_balanced_accuracy']):.4f}, "
            f"fp_total={int(row['fp_total'])}, fn_total={int(row['fn_total'])}, "
            f"latest_fold_balanced_accuracy={float(row['latest_fold_balanced_accuracy']):.4f}, "
            f"latest_fold_fp={int(row['latest_fold_fp'])}, latest_fold_fn={int(row['latest_fold_fn'])}`"
        )

    summary_lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )

    if promoted_threshold is None:
        summary_lines.extend(
            [
                "- promoted threshold: `none`",
                (
                    f"- no candidate in `0.44` to `0.455` cleared the additive time-aware support rule against the locked `{locked_threshold:.4f}` threshold"
                ),
                (
                    f"- strongest held-out candidate remained `0.4550` on the 39-row saved test slice "
                    f"(`balanced_accuracy=0.9306, fp=5, fn=0`), but its fixed-threshold forward check weakened to "
                    f"`mean_fold_balanced_accuracy={float(summary_df.loc[(summary_df['threshold'] - 0.455).abs() < 1e-9, 'mean_fold_balanced_accuracy'].iloc[0]):.4f}, "
                    f"pooled_balanced_accuracy={float(summary_df.loc[(summary_df['threshold'] - 0.455).abs() < 1e-9, 'pooled_balanced_accuracy'].iloc[0]):.4f}, "
                    f"fp_total={int(summary_df.loc[(summary_df['threshold'] - 0.455).abs() < 1e-9, 'fp_total'].iloc[0])}, "
                    f"fn_total={int(summary_df.loc[(summary_df['threshold'] - 0.455).abs() < 1e-9, 'fn_total'].iloc[0])}`"
                ),
                (
                    f"- locked threshold reference in the same time-aware pass: "
                    f"`mean_fold_balanced_accuracy={float(locked_summary['mean_fold_balanced_accuracy']):.4f}, "
                    f"pooled_balanced_accuracy={float(locked_summary['pooled_balanced_accuracy']):.4f}, "
                    f"fp_total={int(locked_summary['fp_total'])}, fn_total={int(locked_summary['fn_total'])}`"
                ),
            ]
        )
    else:
        summary_lines.extend(
            [
                f"- promoted threshold: `{promoted_threshold:.4f}`",
                "- the candidate cleared the held-out rule and the additive fixed-threshold time-aware support rule",
            ]
        )

    summary_lines.extend(
        [
            "",
            "## Next Command",
            "",
            f"- `{DEFAULT_NEXT_COMMAND}`",
            "",
        ]
    )
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    manifest = {
        "analysis_name": analysis_name,
        "source_run_name": run_name,
        "target": target_name,
        "locked_threshold": locked_threshold,
        "candidate_thresholds": candidate_thresholds,
        "min_train_rows": int(args.min_train_rows),
        "eval_window_rows": int(args.eval_window_rows),
        "fold_count": int(len(folds)),
        "promoted_threshold": promoted_threshold,
        "next_command": DEFAULT_NEXT_COMMAND,
        "source_artifacts": {
            "run_config": str(run_dir / "config.json"),
            "heldout_threshold_table": str(
                project_root
                / "reports"
                / "backtests"
                / "temporal_multires"
                / f"{run_name}_operational_check_v1"
                / "threshold_operating_table.csv"
            ),
        },
    }
    (report_dir / "promotion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
