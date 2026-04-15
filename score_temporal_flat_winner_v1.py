from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from train_temporal_multires_simple_baselines_v1 import (
    build_feature_frame,
    ensure_dir,
    positive_class_probability,
)


DEFAULT_RUN_NAME = "simple_loss_daysweeks_v2"
DEFAULT_POLICY_NAME = "simple_loss_daysweeks_v2_operational_policy_v1"
DEFAULT_SCORING_NAME = "simple_loss_daysweeks_v2_operational_scoring_v1"
DEFAULT_TARGET = "y_next_weight_loss_flag"


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe_value(value):
    if pd.isna(value):
        return None
    return value


def policy_band(score: float, threshold: float, promotion_low: float, promotion_high: float) -> str:
    if score < threshold:
        return f"below {threshold:.4f}"
    if score < promotion_low:
        return f"{threshold:.4f} to <{promotion_low:.2f}"
    if score <= promotion_high:
        return f"{promotion_low:.2f} to {promotion_high:.3f} candidate promotion zone"
    return f">{promotion_high:.3f} stronger positive rank position"


def choose_latest_eligible_row(score_df: pd.DataFrame, enabled_modalities: List[str]) -> pd.Series:
    eligible = filter_eligible_rows(score_df=score_df, enabled_modalities=enabled_modalities)
    if eligible.empty:
        eligible = score_df.copy()

    if "anchor_period_start" in eligible.columns:
        eligible["_sort_ts"] = pd.to_datetime(eligible["anchor_period_start"], errors="coerce")
    else:
        eligible["_sort_ts"] = pd.to_datetime(eligible["anchor_id"], errors="coerce")
    eligible = eligible.sort_values(by=["_sort_ts", "anchor_id"], ascending=[True, True], na_position="last")
    return eligible.iloc[-1]


def filter_eligible_rows(score_df: pd.DataFrame, enabled_modalities: List[str]) -> pd.DataFrame:
    eligible_mask = pd.Series(True, index=score_df.index)
    for modality in enabled_modalities:
        col = f"has_{modality}"
        if col in score_df.columns:
            eligible_mask &= pd.to_numeric(score_df[col], errors="coerce").fillna(0.0) > 0
    return score_df.loc[eligible_mask].copy()


def sort_history_rows(score_df: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    history = score_df.copy()
    if "anchor_period_start" in history.columns:
        history["_sort_ts"] = pd.to_datetime(history["anchor_period_start"], errors="coerce")
    else:
        history["_sort_ts"] = pd.to_datetime(history["anchor_id"], errors="coerce")
    history = history.sort_values(by=["_sort_ts", "anchor_id"], ascending=[ascending, ascending], na_position="last")
    return history.drop(columns=["_sort_ts"], errors="ignore")


def decision_label(value: int) -> str:
    return "positive" if int(value) == 1 else "negative"


def format_recent_cases_markdown(recent_df: pd.DataFrame, recent_n: int) -> List[str]:
    if recent_df.empty:
        return [f"## Recent Cases ({recent_n})", "", "_No eligible recent cases found._", ""]

    display_cols = [
        "anchor_id",
        "anchor_period_start",
        "score",
        "decision_locked_label",
        "policy_band",
        "locked_threshold",
    ]
    display_df = recent_df[[c for c in display_cols if c in recent_df.columns]].copy()
    if "score" in display_df.columns:
        display_df["score"] = display_df["score"].map(lambda x: f"{float(x):.6f}")
    if "locked_threshold" in display_df.columns:
        display_df["locked_threshold"] = display_df["locked_threshold"].map(lambda x: f"{float(x):.4f}")

    headers = list(display_df.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = []
    for _, row in display_df.iterrows():
        body_rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")

    lines = [f"## Recent Cases ({recent_n})", ""]
    lines.append(header_row)
    lines.append(separator_row)
    lines.extend(body_rows)
    lines.append("")
    return lines


def build_summary_markdown(
    selected_row: pd.Series,
    report_dir: Path,
    threshold: float,
    promotion_low: float,
    promotion_high: float,
    selected_anchor_mode: str,
    recent_df: pd.DataFrame | None = None,
    recent_n: int = 0,
) -> str:
    lines = [
        "# Operational Scoring: simple_loss_daysweeks_v2",
        "",
        "## Selected Case",
        "",
        f"- selected_anchor_mode: `{selected_anchor_mode}`",
        f"- anchor_id: `{selected_row['anchor_id']}`",
    ]
    if pd.notna(selected_row.get("anchor_period_start")):
        lines.append(f"- anchor_period_start: `{selected_row['anchor_period_start']}`")
    lines.extend(
        [
            f"- target: `{DEFAULT_TARGET}`",
            "- modalities: `days,weeks`",
            "- model family: `flattened ExtraTrees`",
            f"- score: `{selected_row['score']:.6f}`",
            f"- locked decision: `{decision_label(selected_row['decision_locked'])}`",
            f"- threshold used: `{threshold:.4f}`",
            f"- policy band: `{selected_row['policy_band']}`",
            "- score is not a calibrated probability: `true`",
            "- score interpretation: `ranking / threshold signal only; not a calibrated probability`",
            "",
            "## Policy",
            "",
            f"- `< {threshold:.4f}`: below the current action threshold",
            f"- `{threshold:.4f}` to `<{promotion_low:.2f}`: current positive signal under the locked policy",
            f"- `{promotion_low:.2f}` to `{promotion_high:.3f}`: candidate promotion zone; positive now, but still unpromoted",
            f"- `>{promotion_high:.3f}`: stronger positive rank position under the same locked threshold",
            "",
            "## Bundle Files",
            "",
            f"- history scores: `{report_dir / 'history_scores.csv'}`",
            f"- selected case JSON: `{report_dir / 'selected_case.json'}`",
            f"- scoring manifest: `{report_dir / 'scoring_manifest.json'}`",
        ]
    )
    if recent_df is not None and recent_n > 0:
        lines.extend(
            [
                f"- recent cases CSV: `{report_dir / 'recent_cases.csv'}`",
                f"- recent cases Markdown: `{report_dir / 'recent_cases.md'}`",
            ]
        )
        lines.extend([""] + format_recent_cases_markdown(recent_df=recent_df, recent_n=recent_n))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the locked flattened ET temporal winner for operational use.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument(
        "--dataset-dir",
        default="training/multires_sequence_dataset",
        help="Relative path to the multires dataset when not overridden by saved run config.",
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Winner training run name.")
    parser.add_argument(
        "--policy-name",
        default=DEFAULT_POLICY_NAME,
        help="Operational policy bundle name that defines the locked threshold and bands.",
    )
    parser.add_argument(
        "--scoring-name",
        default=DEFAULT_SCORING_NAME,
        help="Output report bundle name for this scorer run.",
    )
    parser.add_argument("--anchor-id", default="", help="Optional exact anchor_id to score and select.")
    parser.add_argument(
        "--recent-n",
        type=int,
        default=0,
        help="Optional count of latest eligible cases to write as a compact operational recent-history report.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    run_name = args.run_name
    policy_name = args.policy_name
    scoring_name = args.scoring_name

    run_report_dir = project_root / "reports" / "backtests" / "temporal_multires" / run_name
    policy_dir = project_root / "reports" / "backtests" / "temporal_multires" / policy_name
    scoring_dir = project_root / "reports" / "backtests" / "temporal_multires" / scoring_name
    ensure_dir(scoring_dir)

    config = load_json(run_report_dir / "config.json")
    selected_models_payload = load_json(run_report_dir / "selected_models.json")
    selected_thresholds = load_json(run_report_dir / "selected_thresholds.json")
    policy_payload = load_json(policy_dir / "operational_policy.json")

    dataset_dir = Path(config.get("dataset_dir", project_root / args.dataset_dir))
    if not dataset_dir.is_absolute():
        dataset_dir = (project_root / dataset_dir).resolve()

    enabled_modalities = list(config["modalities"])
    windows = dict(config["windows"])
    threshold = float(selected_thresholds[DEFAULT_TARGET])
    promotion_low = float(policy_payload["candidate_promotion_zone"]["low"])
    promotion_high = float(policy_payload["candidate_promotion_zone"]["high"])

    model_rows = selected_models_payload.get("models", [])
    winner_model_row = next((row for row in model_rows if row.get("target") == DEFAULT_TARGET), None)
    if winner_model_row is None:
        raise ValueError(f"Could not find selected model for {DEFAULT_TARGET} in {run_report_dir / 'selected_models.json'}")

    model_path = Path(winner_model_row["model_artifact"])
    model = joblib.load(model_path)

    anchors = pd.read_csv(dataset_dir / "anchors.csv", low_memory=False)
    anchors["anchor_id"] = anchors["anchor_id"].astype(str)

    feature_df, feature_meta = build_feature_frame(
        anchors=anchors,
        dataset_dir=dataset_dir,
        enabled_modalities=enabled_modalities,
        windows=windows,
    )
    score = positive_class_probability(model, feature_df).astype(float)

    score_df = anchors.copy()
    for modality in enabled_modalities:
        static_col = f"has_{modality}"
        if static_col in feature_df.columns:
            score_df[static_col] = pd.to_numeric(feature_df[static_col], errors="coerce")
    score_df["score"] = score
    score_df["locked_threshold"] = threshold
    score_df["decision_locked"] = (score_df["score"] >= threshold).astype(int)
    score_df["decision_locked_label"] = score_df["decision_locked"].map(decision_label)
    score_df["policy_band"] = score_df["score"].map(
        lambda x: policy_band(
            score=float(x),
            threshold=threshold,
            promotion_low=promotion_low,
            promotion_high=promotion_high,
        )
    )
    score_df["score_not_calibrated_probability"] = True
    score_df["threshold_used"] = threshold

    selected_anchor_mode = "latest_eligible"
    if args.anchor_id:
        selected_anchor_mode = "explicit_anchor_id"
        selected = score_df.loc[score_df["anchor_id"] == str(args.anchor_id)].copy()
        if selected.empty:
            raise ValueError(f"Unknown anchor_id: {args.anchor_id}")
        selected_row = selected.iloc[-1]
    else:
        selected_row = choose_latest_eligible_row(score_df=score_df, enabled_modalities=enabled_modalities)

    history_cols = [
        "anchor_id",
        "anchor_period_start",
        "anchor_next_period_start",
        "split_suggested",
        DEFAULT_TARGET,
        "score",
        "locked_threshold",
        "threshold_used",
        "decision_locked",
        "decision_locked_label",
        "policy_band",
        "score_not_calibrated_probability",
    ]
    for modality in enabled_modalities:
        col = f"has_{modality}"
        if col in score_df.columns:
            history_cols.append(col)
    history_cols = [c for c in history_cols if c in score_df.columns]
    history_df = sort_history_rows(score_df[history_cols], ascending=True)
    history_df.to_csv(scoring_dir / "history_scores.csv", index=False)

    recent_df = None
    if args.recent_n > 0:
        recent_history_source = filter_eligible_rows(score_df=score_df, enabled_modalities=enabled_modalities)
        if recent_history_source.empty:
            recent_history_source = score_df.copy()
        recent_df = sort_history_rows(recent_history_source[history_cols], ascending=False).head(args.recent_n).copy()
        recent_df.to_csv(scoring_dir / "recent_cases.csv", index=False)
        recent_md_lines = [f"# Recent Operational Cases: simple_loss_daysweeks_v2", ""]
        recent_md_lines.extend(format_recent_cases_markdown(recent_df=recent_df, recent_n=args.recent_n))
        (scoring_dir / "recent_cases.md").write_text("\n".join(recent_md_lines) + "\n", encoding="utf-8")

    selected_payload = {
        "selected_anchor_mode": selected_anchor_mode,
        "anchor_id": str(selected_row["anchor_id"]),
        "anchor_period_start": json_safe_value(selected_row.get("anchor_period_start")),
        "anchor_next_period_start": json_safe_value(selected_row.get("anchor_next_period_start")),
        "split_suggested": json_safe_value(selected_row.get("split_suggested")),
        "target": DEFAULT_TARGET,
        "target_observed_value": json_safe_value(selected_row.get(DEFAULT_TARGET)),
        "score": float(selected_row["score"]),
        "locked_threshold": threshold,
        "threshold_used": threshold,
        "decision_locked": bool(int(selected_row["decision_locked"])),
        "decision_locked_label": str(selected_row["decision_locked_label"]),
        "policy_band": str(selected_row["policy_band"]),
        "score_not_calibrated_probability": True,
        "score_interpretation": "ranking / threshold signal only; not a calibrated probability",
        "modalities": enabled_modalities,
        "windows": windows,
        "winner_run_name": run_name,
        "policy_name": policy_name,
        "winner_model_artifact": str(model_path),
    }

    (scoring_dir / "selected_case.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    pd.DataFrame([selected_payload]).to_csv(scoring_dir / "selected_case.csv", index=False)

    scoring_manifest = {
        "winner_run_name": run_name,
        "policy_name": policy_name,
        "scoring_name": scoring_name,
        "dataset_dir": str(dataset_dir),
        "target": DEFAULT_TARGET,
        "modalities": enabled_modalities,
        "windows": windows,
        "locked_threshold": threshold,
        "candidate_promotion_zone": {"low": promotion_low, "high": promotion_high},
        "selected_anchor_mode": selected_anchor_mode,
        "selected_anchor_id": str(selected_row["anchor_id"]),
        "feature_build": feature_meta,
        "source_artifacts": {
            "model_artifact": str(model_path),
            "run_config": str(run_report_dir / "config.json"),
            "selected_models": str(run_report_dir / "selected_models.json"),
            "selected_thresholds": str(run_report_dir / "selected_thresholds.json"),
            "policy_json": str(policy_dir / "operational_policy.json"),
        },
        "outputs": {
            "history_scores_csv": str(scoring_dir / "history_scores.csv"),
            "selected_case_json": str(scoring_dir / "selected_case.json"),
            "selected_case_csv": str(scoring_dir / "selected_case.csv"),
            "summary_md": str(scoring_dir / "summary.md"),
        },
    }
    if args.recent_n > 0:
        scoring_manifest["recent_history"] = {
            "recent_n": int(args.recent_n),
            "recent_cases_csv": str(scoring_dir / "recent_cases.csv"),
            "recent_cases_md": str(scoring_dir / "recent_cases.md"),
        }
    (scoring_dir / "scoring_manifest.json").write_text(json.dumps(scoring_manifest, indent=2), encoding="utf-8")
    (scoring_dir / "summary.md").write_text(
        build_summary_markdown(
            selected_row=selected_row,
            report_dir=scoring_dir,
            threshold=threshold,
            promotion_low=promotion_low,
            promotion_high=promotion_high,
            selected_anchor_mode=selected_anchor_mode,
            recent_df=recent_df,
            recent_n=args.recent_n,
        ),
        encoding="utf-8",
    )

    print(f"anchor_id={selected_payload['anchor_id']}")
    print(f"score={selected_payload['score']:.6f}")
    print(f"locked_decision={selected_payload['decision_locked_label']}")
    print(f"threshold_used={selected_payload['threshold_used']:.4f}")
    print(f"policy_band={selected_payload['policy_band']}")
    print("score_not_calibrated_probability=true")
    print("score_interpretation=ranking / threshold signal only; not a calibrated probability")
    if args.recent_n > 0:
        print(f"recent_cases_written={min(args.recent_n, len(recent_df)) if recent_df is not None else 0}")
    print(f"report_dir={scoring_dir}")


if __name__ == "__main__":
    main()
