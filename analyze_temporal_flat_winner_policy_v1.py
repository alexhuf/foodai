from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


LOCKED_THRESHOLD = 0.4288
CANDIDATE_ZONE_LOW = 0.44
CANDIDATE_ZONE_HIGH = 0.455
DEFAULT_NEXT_COMMAND = (
    "python train_temporal_multires_flattened_explore_v1.py "
    "--project-root /workspace/foodai "
    "--run-name flat_loss_daysweeks_et_windowpilot_v1 "
    "--target y_next_weight_loss_flag "
    "--modalities days,weeks "
    "--candidate-models et_balanced "
    "--days-window 7 "
    "--weeks-window 2"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_threshold_row(table: pd.DataFrame, threshold: float) -> Dict:
    match = table.loc[(table["threshold"] - threshold).abs() < 1e-9]
    if match.empty:
        match = table.iloc[[(table["threshold"] - threshold).abs().argmin()]]
    return match.iloc[0].to_dict()


def band_rows() -> List[Dict]:
    return [
        {
            "band_name": "below_locked_threshold",
            "score_min": 0.0,
            "score_max": LOCKED_THRESHOLD,
            "decision_label": "below_current_action_threshold",
            "policy_action": "Do not treat as a threshold-positive case.",
            "framing": "Use only as relative ranking context; this score is below the locked operating point.",
        },
        {
            "band_name": "locked_positive_band",
            "score_min": LOCKED_THRESHOLD,
            "score_max": CANDIDATE_ZONE_LOW,
            "decision_label": "current_positive_signal",
            "policy_action": "Treat as threshold-positive under the current operating policy.",
            "framing": "This is a positive ranking/threshold signal, not a calibrated probability claim.",
        },
        {
            "band_name": "candidate_promotion_zone",
            "score_min": CANDIDATE_ZONE_LOW,
            "score_max": CANDIDATE_ZONE_HIGH,
            "decision_label": "positive_signal_with_promotion_interest",
            "policy_action": "Treat as threshold-positive now and track as the explicit candidate promotion zone.",
            "framing": "This band improved the held-out slice while keeping zero false negatives, but it is not promoted yet.",
        },
        {
            "band_name": "upper_positive_tail",
            "score_min": CANDIDATE_ZONE_HIGH,
            "score_max": 1.0,
            "decision_label": "stronger_positive_signal_still_unpromoted",
            "policy_action": "Treat as threshold-positive under the current lock; do not reinterpret as calibrated risk.",
            "framing": "Scores above the candidate zone are stronger rank positions, not a separate approved threshold.",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an operational policy bundle for the current flattened temporal winner.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-name", default="simple_loss_daysweeks_v2")
    parser.add_argument("--analysis-name", default="")
    parser.add_argument("--locked-threshold", type=float, default=LOCKED_THRESHOLD)
    parser.add_argument("--candidate-zone-low", type=float, default=CANDIDATE_ZONE_LOW)
    parser.add_argument("--candidate-zone-high", type=float, default=CANDIDATE_ZONE_HIGH)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    run_name = args.run_name
    analysis_name = args.analysis_name or f"{run_name}_operational_policy_v1"
    locked_threshold = float(args.locked_threshold)
    candidate_zone_low = float(args.candidate_zone_low)
    candidate_zone_high = float(args.candidate_zone_high)

    report_dir = project_root / "reports" / "backtests" / "temporal_multires" / analysis_name
    ensure_dir(report_dir)

    run_dir = project_root / "reports" / "backtests" / "temporal_multires" / run_name
    winner_dir = project_root / "reports" / "backtests" / "temporal_multires" / f"{run_name}_winner_analysis_v1"
    op_dir = project_root / "reports" / "backtests" / "temporal_multires" / f"{run_name}_operational_check_v1"
    split_dir = project_root / "reports" / "backtests" / "temporal_multires" / f"{run_name}_operational_check_splitmimic_v1"

    config = read_json(run_dir / "config.json")
    selected_thresholds = read_json(run_dir / "selected_thresholds.json")
    winner_summary = (winner_dir / "summary.md").read_text(encoding="utf-8")
    op_summary = (op_dir / "summary.md").read_text(encoding="utf-8")
    split_summary = (split_dir / "summary.md").read_text(encoding="utf-8")

    op_threshold_df = pd.read_csv(op_dir / "threshold_operating_table.csv")
    split_rolling_df = pd.read_csv(split_dir / "time_aware_rolling_check.csv")
    test_predictions = pd.read_csv(run_dir / "test_predictions.csv")

    selected_row = choose_threshold_row(op_threshold_df, locked_threshold)
    zone_rows = []
    for threshold in [locked_threshold, candidate_zone_low, 0.445, candidate_zone_high]:
        zone_rows.append(choose_threshold_row(op_threshold_df, threshold))
    zone_df = pd.DataFrame(zone_rows).drop_duplicates(subset=["threshold"]).sort_values("threshold").reset_index(drop=True)
    zone_df.to_csv(report_dir / "promotion_zone_evidence.csv", index=False)

    band_df = pd.DataFrame(band_rows())
    band_df.loc[0, "score_max"] = locked_threshold
    band_df.loc[1, "score_min"] = locked_threshold
    band_df.loc[1, "score_max"] = candidate_zone_low
    band_df.loc[2, "score_min"] = candidate_zone_low
    band_df.loc[2, "score_max"] = candidate_zone_high
    band_df.loc[3, "score_min"] = candidate_zone_high
    band_df.to_csv(report_dir / "decision_bands.csv", index=False)

    split_fold_mean = float(split_rolling_df["balanced_accuracy"].mean()) if not split_rolling_df.empty else None
    split_fold_min = float(split_rolling_df["balanced_accuracy"].min()) if not split_rolling_df.empty else None
    split_fold_max = float(split_rolling_df["balanced_accuracy"].max()) if not split_rolling_df.empty else None

    policy_payload = {
        "winner_run_name": run_name,
        "target": "y_next_weight_loss_flag",
        "modalities": list(config["modalities"]),
        "model_family": "flattened ExtraTrees",
        "locked_threshold": locked_threshold,
        "candidate_promotion_zone": {
            "low": candidate_zone_low,
            "high": candidate_zone_high,
        },
        "decision_framing": {
            "primary": "Treat the score as a ranking-plus-threshold signal, not as a calibrated probability.",
            "threshold_positive_rule": f"Positive when score >= {locked_threshold:.4f}.",
            "threshold_lock": "Do not promote the threshold yet.",
        },
        "current_locked_operating_point": {
            "threshold": float(selected_row["threshold"]),
            "balanced_accuracy": float(selected_row["balanced_accuracy"]),
            "fp": int(selected_row["fp"]),
            "fn": int(selected_row["fn"]),
            "positive_rate_pred": float(selected_row["positive_rate_pred"]),
        },
        "candidate_zone_heldout_evidence": zone_df.to_dict(orient="records"),
        "promotion_condition": {
            "exact_rule": (
                "Promote above 0.4288 only after one specific threshold in the 0.44 to 0.455 zone "
                "reproduces FN=0 and better-than-0.4288 held-out balanced accuracy with fewer false positives, "
                "and that same upward-threshold claim is supported by an additional additive time-aware check "
                "rather than only the current favorable held-out slice."
            ),
            "minimum_heldout_requirements": {
                "threshold_gt": locked_threshold,
                "threshold_zone_low": candidate_zone_low,
                "threshold_zone_high": candidate_zone_high,
                "fn_equals": 0,
                "balanced_accuracy_gt": float(selected_row["balanced_accuracy"]),
                "fp_lt": int(selected_row["fp"]),
            },
            "current_blocker": (
                "The split-mimic rolling validation remains weaker than the original favorable held-out picture, "
                "so the repo does not yet have enough robustness evidence to approve a higher production threshold."
            ),
        },
        "split_mimic_summary": {
            "fold_count": int(len(split_rolling_df)),
            "balanced_accuracy_mean": split_fold_mean,
            "balanced_accuracy_min": split_fold_min,
            "balanced_accuracy_max": split_fold_max,
        },
        "next_command_after_policy": DEFAULT_NEXT_COMMAND,
        "source_artifacts": {
            "winner_summary": str(winner_dir / "summary.md"),
            "operational_summary": str(op_dir / "summary.md"),
            "splitmimic_summary": str(split_dir / "summary.md"),
            "selected_threshold_path": str(run_dir / "selected_thresholds.json"),
        },
    }
    (report_dir / "operational_policy.json").write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

    summary_lines = [
        f"# Operational Policy: {run_name}",
        "",
        f"- target: `y_next_weight_loss_flag`",
        f"- modalities: `{','.join(config['modalities'])}`",
        f"- model family: `flattened ExtraTrees`",
        f"- locked threshold: `{locked_threshold:.4f}`",
        f"- candidate promotion zone: `{candidate_zone_low:.2f}` to `{candidate_zone_high:.3f}`",
        "",
        "## Current Policy",
        "",
        f"- classify as operationally positive only when `score >= {locked_threshold:.4f}`",
        "- interpret the score as a ranking/threshold signal, not as a calibrated probability",
        (
            f"- locked operating point reference: `balanced_accuracy={float(selected_row['balanced_accuracy']):.4f}, "
            f"fp={int(selected_row['fp'])}, fn={int(selected_row['fn'])}, positive_rate_pred={float(selected_row['positive_rate_pred']):.4f}`"
        ),
        "",
        "## Decision Bands",
        "",
        f"- `< {locked_threshold:.4f}`: below the current action threshold",
        f"- `{locked_threshold:.4f}` to `< {candidate_zone_low:.2f}`: current positive signal under the locked policy",
        f"- `{candidate_zone_low:.2f}` to `{candidate_zone_high:.3f}`: explicit candidate promotion zone; positive now, but still unpromoted",
        f"- `> {candidate_zone_high:.3f}`: stronger positive rank position, still governed by the same locked threshold policy",
        "",
        "## Promotion Rule",
        "",
        "- do not promote the threshold yet",
        (
            f"- exact condition for promotion above `{locked_threshold:.4f}`: one specific threshold in `{candidate_zone_low:.2f}` to `{candidate_zone_high:.3f}` "
            f"must reproduce `FN=0`, improve held-out balanced accuracy above `{float(selected_row['balanced_accuracy']):.4f}`, "
            f"and reduce false positives below `{int(selected_row['fp'])}`, with that same upward-threshold claim supported by an additional additive time-aware check"
        ),
        (
            f"- current blocker: split-mimic rolling validation is still weaker than the favorable held-out slice "
            f"(folds={int(len(split_rolling_df))}, balanced_accuracy mean/min/max = "
            f"{split_fold_mean:.4f} / {split_fold_min:.4f} / {split_fold_max:.4f})"
            if split_fold_mean is not None
            else "- current blocker: split-mimic rolling validation unavailable"
        ),
        "",
        "## Held-Out Zone Evidence",
        "",
    ]

    for row in zone_df.to_dict(orient="records"):
        summary_lines.append(
            f"- `threshold={float(row['threshold']):.4f}`: "
            f"`balanced_accuracy={float(row['balanced_accuracy']):.4f}, fp={int(row['fp'])}, fn={int(row['fn'])}, positive_rate_pred={float(row['positive_rate_pred']):.4f}`"
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
        "locked_threshold_from_run": float(selected_thresholds["y_next_weight_loss_flag"]),
        "locked_threshold_policy": locked_threshold,
        "candidate_zone_low": candidate_zone_low,
        "candidate_zone_high": candidate_zone_high,
        "test_rows": int(len(test_predictions)),
        "report_files": sorted(p.name for p in report_dir.iterdir() if p.is_file()),
        "source_summaries_loaded": {
            "winner_summary_md": bool(winner_summary),
            "operational_summary_md": bool(op_summary),
            "splitmimic_summary_md": bool(split_summary),
        },
    }
    (report_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
