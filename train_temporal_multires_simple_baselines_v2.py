from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from train_temporal_multires_simple_baselines_v1 import (
    DEFAULT_BINARY_TARGETS,
    DEFAULT_WINDOWS,
    build_binary_diagnostics,
    build_feature_frame,
    ensure_dir,
    evaluate_binary_target,
    evaluate_regression_target,
    log,
    merge_prediction_frames,
    save_json,
    summarize_metrics,
)


DEFAULT_REGRESSION_TARGETS: List[str] = []


def parse_target_list(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if raw.lower() in {"", "none", "null", "off"}:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train simple lag-window baselines on the multires temporal dataset."
    )
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument(
        "--dataset-dir",
        default="training/multires_sequence_dataset",
        help="Relative path to multires dataset.",
    )
    parser.add_argument("--run-name", default="", help="Optional run name. Auto-generated if omitted.")
    parser.add_argument("--modalities", default="days,weeks", help="Comma-separated enabled modalities.")
    parser.add_argument(
        "--binary-targets",
        default=",".join(DEFAULT_BINARY_TARGETS),
        help="Comma-separated binary targets.",
    )
    parser.add_argument(
        "--regression-targets",
        default="",
        help="Comma-separated regression targets. Empty by default; set explicitly to train regression targets.",
    )
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

    binary_targets = parse_target_list(args.binary_targets)
    regression_targets = parse_target_list(args.regression_targets)
    if args.single_binary_target:
        binary_targets = [args.single_binary_target.strip()]
    if args.single_regression_target:
        regression_targets = [args.single_regression_target.strip()]
    if not binary_targets and not regression_targets:
        raise ValueError("At least one binary or regression target is required.")

    enabled_modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if not enabled_modalities:
        raise ValueError("At least one modality must be enabled.")

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

    target_bits = []
    target_bits.extend([t.replace("y_next_", "").replace("_flag", "").replace("_lb", "") for t in binary_targets])
    target_bits.extend([t.replace("y_next_", "").replace("_flag", "").replace("_lb", "") for t in regression_targets])
    modality_bits = "_".join(enabled_modalities)
    run_name = args.run_name or f"simple_{'_'.join(target_bits)}_{modality_bits}_v2"

    model_dir = project_root / "models" / "temporal_multires" / run_name
    report_dir = project_root / "reports" / "backtests" / "temporal_multires" / run_name
    ensure_dir(model_dir)
    ensure_dir(report_dir)

    save_json(
        report_dir / "config.json",
        {
            "project_root": str(project_root),
            "dataset_dir": str(dataset_dir),
            "run_name": run_name,
            "modalities": enabled_modalities,
            "binary_targets": binary_targets,
            "regression_targets": regression_targets,
            "windows": windows,
        },
    )

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
