from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from score_temporal_flat_winner_v1 import (
    DEFAULT_POLICY_NAME,
    DEFAULT_RUN_NAME,
    DEFAULT_SCORING_NAME,
    filter_eligible_rows,
    load_json,
)
from train_temporal_multires_simple_baselines_v1 import ensure_dir


DEFAULT_REFRESH_NAME = "simple_loss_daysweeks_v2_operational_refresh_v1"
DEFAULT_SPLITMIMIC_NAME = "simple_loss_daysweeks_v2_operational_check_splitmimic_v1"
DEFAULT_REFERENCE_CHECK_NAME = "simple_loss_daysweeks_v2_operational_check_v1"


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def percentile_rank(series: pd.Series, value: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((clean <= value).mean())


def summarize_score_distribution(history_df: pd.DataFrame) -> Dict[str, float | None]:
    scores = pd.to_numeric(history_df["score"], errors="coerce").dropna()
    if scores.empty:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q10": None,
            "median": None,
            "q90": None,
            "max": None,
        }
    return {
        "n": int(scores.shape[0]),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "min": float(scores.min()),
        "q10": float(scores.quantile(0.10)),
        "median": float(scores.quantile(0.50)),
        "q90": float(scores.quantile(0.90)),
        "max": float(scores.max()),
    }


def summarize_recent_window(recent_df: pd.DataFrame, threshold: float) -> Dict[str, float | int | None]:
    if recent_df.empty:
        return {
            "n": 0,
            "score_mean": None,
            "score_std": None,
            "score_min": None,
            "score_max": None,
            "positive_rate_locked": None,
            "positive_count_locked": 0,
            "threshold": threshold,
        }
    scores = pd.to_numeric(recent_df["score"], errors="coerce")
    decisions = pd.to_numeric(recent_df["decision_locked"], errors="coerce").fillna(0.0)
    return {
        "n": int(recent_df.shape[0]),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=0)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "positive_rate_locked": float(decisions.mean()),
        "positive_count_locked": int(decisions.sum()),
        "threshold": threshold,
    }


def rolling_positive_rate_summary(history_df: pd.DataFrame, window_n: int) -> Dict[str, object]:
    decisions = pd.to_numeric(history_df["decision_locked"], errors="coerce").fillna(0.0)
    if window_n <= 0 or decisions.shape[0] < window_n:
        return {
            "window_n": int(window_n),
            "window_count": 0,
            "distribution": {
                "mean": None,
                "std": None,
                "min": None,
                "q10": None,
                "median": None,
                "q90": None,
                "max": None,
            },
        }

    rolling = decisions.rolling(window=window_n).mean().dropna()
    return {
        "window_n": int(window_n),
        "window_count": int(rolling.shape[0]),
        "distribution": {
            "mean": float(rolling.mean()),
            "std": float(rolling.std(ddof=0)),
            "min": float(rolling.min()),
            "q10": float(rolling.quantile(0.10)),
            "median": float(rolling.quantile(0.50)),
            "q90": float(rolling.quantile(0.90)),
            "max": float(rolling.max()),
        },
    }


def watch_level(flag: bool) -> str:
    return "watch" if flag else "ok"


def build_watch_payload(
    latest_case: Dict,
    recent_summary: Dict,
    trailing_distribution: Dict,
    historical_distribution: Dict,
    rolling_positive_rate: Dict,
    promotion_low: float,
    promotion_high: float,
) -> Dict[str, object]:
    latest_score = float(latest_case["score"])
    recent_positive_rate = recent_summary["positive_rate_locked"]
    trailing_q10 = trailing_distribution["q10"]
    trailing_q90 = trailing_distribution["q90"]
    hist_q10 = historical_distribution["q10"]
    hist_q90 = historical_distribution["q90"]
    rolling_q10 = rolling_positive_rate["distribution"]["q10"]
    rolling_q90 = rolling_positive_rate["distribution"]["q90"]

    latest_vs_recent_flag = False
    if trailing_q10 is not None and trailing_q90 is not None:
        latest_vs_recent_flag = latest_score < trailing_q10 or latest_score > trailing_q90

    recent_positive_rate_flag = False
    if (
        recent_positive_rate is not None
        and rolling_q10 is not None
        and rolling_q90 is not None
    ):
        recent_positive_rate_flag = (
            recent_positive_rate < rolling_q10 or recent_positive_rate > rolling_q90
        )

    promotion_zone_flag = promotion_low <= latest_score <= promotion_high

    return {
        "latest_score_vs_recent_window": {
            "status": watch_level(latest_vs_recent_flag),
            "latest_score": latest_score,
            "recent_window_q10": trailing_q10,
            "recent_window_q90": trailing_q90,
            "reason": "Watch when the latest score falls outside the inner recent-window band.",
        },
        "recent_positive_rate_shift": {
            "status": watch_level(recent_positive_rate_flag),
            "recent_positive_rate_locked": recent_positive_rate,
            "historical_window_q10": rolling_q10,
            "historical_window_q90": rolling_q90,
            "reason": "Watch when the latest-N positive rate sits outside the historical rolling band for the same N.",
        },
        "candidate_promotion_zone": {
            "status": watch_level(promotion_zone_flag),
            "latest_score": latest_score,
            "zone_low": promotion_low,
            "zone_high": promotion_high,
            "in_zone": bool(promotion_zone_flag),
            "reason": "Watch when the latest score enters the still-unpromoted candidate promotion zone.",
        },
        "reference_score_context": {
            "historical_q10": hist_q10,
            "historical_q90": hist_q90,
            "historical_mean": historical_distribution["mean"],
            "historical_std": historical_distribution["std"],
        },
    }


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_summary_markdown(
    refresh_payload: Dict,
    report_dir: Path,
) -> str:
    latest = refresh_payload["latest_case"]
    recent = refresh_payload["recent_summary"]
    watches = refresh_payload["watch_checks"]
    policy = refresh_payload["policy"]
    splitmimic = refresh_payload["reference_checks"]["splitmimic"]

    lines: List[str] = [
        "# Operational Refresh: simple_loss_daysweeks_v2",
        "",
        "## Latest Case",
        "",
        f"- anchor_id: `{latest['anchor_id']}`",
        f"- anchor_period_start: `{latest.get('anchor_period_start')}`",
        f"- score: `{latest['score']:.6f}`",
        f"- locked decision: `{latest['decision_locked_label']}`",
        f"- policy band: `{latest['policy_band']}`",
        "",
        "## Recent Window",
        "",
        f"- recent_n: `{recent['n']}`",
        f"- recent score mean/std: `{format_float(recent['score_mean'])}` / `{format_float(recent['score_std'])}`",
        f"- recent score min/max: `{format_float(recent['score_min'])}` / `{format_float(recent['score_max'])}`",
        f"- recent locked positive rate: `{format_float(recent['positive_rate_locked'])}`",
        "",
        "## Policy",
        "",
        f"- locked threshold: `{policy['locked_threshold']:.4f}`",
        f"- positive rule: `score >= {policy['locked_threshold']:.4f}`",
        "- score interpretation: `ranking / threshold signal only; not a calibrated probability`",
        f"- candidate promotion zone: `{policy['candidate_promotion_zone']['low']:.2f}` to `{policy['candidate_promotion_zone']['high']:.3f}`",
        "",
        "## Watch Checks",
        "",
        f"- latest score vs recent window: `{watches['latest_score_vs_recent_window']['status']}`",
        f"- recent positive-rate shift: `{watches['recent_positive_rate_shift']['status']}`",
        f"- latest score in candidate promotion zone: `{watches['candidate_promotion_zone']['in_zone']}`",
        "",
        "## Reference Check",
        "",
        f"- split-mimic rolling folds: `{splitmimic['fold_count']}`",
        f"- split-mimic balanced-accuracy mean/min/max: `{format_float(splitmimic['balanced_accuracy_mean'])}` / `{format_float(splitmimic['balanced_accuracy_min'])}` / `{format_float(splitmimic['balanced_accuracy_max'])}`",
        "",
        "## Bundle Files",
        "",
        f"- current state JSON: `{report_dir / 'current_state.json'}`",
        f"- latest case Markdown: `{report_dir / 'latest_case_summary.md'}`",
        f"- recent summary Markdown: `{report_dir / 'recent_summary.md'}`",
        f"- watch checks Markdown: `{report_dir / 'watch_checks.md'}`",
        f"- refresh manifest: `{report_dir / 'refresh_manifest.json'}`",
    ]
    return "\n".join(lines) + "\n"


def build_latest_case_markdown(latest_case: Dict) -> str:
    lines = [
        "# Latest Case Summary",
        "",
        f"- anchor_id: `{latest_case['anchor_id']}`",
        f"- anchor_period_start: `{latest_case.get('anchor_period_start')}`",
        f"- anchor_next_period_start: `{latest_case.get('anchor_next_period_start')}`",
        f"- score: `{latest_case['score']:.6f}`",
        f"- locked threshold: `{latest_case['locked_threshold']:.4f}`",
        f"- locked decision: `{latest_case['decision_locked_label']}`",
        f"- policy band: `{latest_case['policy_band']}`",
        "- interpretation: `ranking / threshold signal only; not a calibrated probability`",
        "",
    ]
    return "\n".join(lines)


def build_recent_summary_markdown(recent_summary: Dict, trailing_distribution: Dict, rolling_positive_rate: Dict) -> str:
    lines = [
        "# Recent Summary",
        "",
        f"- recent_n: `{recent_summary['n']}`",
        f"- score mean/std: `{format_float(recent_summary['score_mean'])}` / `{format_float(recent_summary['score_std'])}`",
        f"- score min/max: `{format_float(recent_summary['score_min'])}` / `{format_float(recent_summary['score_max'])}`",
        f"- locked positive count/rate: `{recent_summary['positive_count_locked']}` / `{format_float(recent_summary['positive_rate_locked'])}`",
        f"- recent-window q10/q90 reference: `{format_float(trailing_distribution['q10'])}` / `{format_float(trailing_distribution['q90'])}`",
        f"- historical rolling positive-rate q10/q90: `{format_float(rolling_positive_rate['distribution']['q10'])}` / `{format_float(rolling_positive_rate['distribution']['q90'])}`",
        "",
    ]
    return "\n".join(lines)


def build_watch_markdown(watch_payload: Dict) -> str:
    lines = [
        "# Watch Checks",
        "",
        f"- latest score vs recent window: `{watch_payload['latest_score_vs_recent_window']['status']}`",
        f"- recent positive-rate shift: `{watch_payload['recent_positive_rate_shift']['status']}`",
        f"- latest score in candidate promotion zone: `{watch_payload['candidate_promotion_zone']['in_zone']}`",
        "",
        "## Detail",
        "",
        f"- latest score: `{format_float(watch_payload['candidate_promotion_zone']['latest_score'])}`",
        f"- recent score q10/q90: `{format_float(watch_payload['latest_score_vs_recent_window']['recent_window_q10'])}` / `{format_float(watch_payload['latest_score_vs_recent_window']['recent_window_q90'])}`",
        f"- recent positive-rate locked: `{format_float(watch_payload['recent_positive_rate_shift']['recent_positive_rate_locked'])}`",
        f"- historical rolling positive-rate q10/q90: `{format_float(watch_payload['recent_positive_rate_shift']['historical_window_q10'])}` / `{format_float(watch_payload['recent_positive_rate_shift']['historical_window_q90'])}`",
        f"- candidate promotion zone: `{format_float(watch_payload['candidate_promotion_zone']['zone_low'])}` to `{format_float(watch_payload['candidate_promotion_zone']['zone_high'], digits=3)}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact operational refresh around the locked flattened ET scorer.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Locked winner run name.")
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME, help="Operational policy bundle name.")
    parser.add_argument("--scoring-name", default=DEFAULT_SCORING_NAME, help="Scoring bundle name written by the locked scorer.")
    parser.add_argument("--refresh-name", default=DEFAULT_REFRESH_NAME, help="Refresh bundle name written by this wrapper.")
    parser.add_argument("--reference-check-name", default=DEFAULT_REFERENCE_CHECK_NAME, help="Reference operational check bundle name.")
    parser.add_argument("--splitmimic-name", default=DEFAULT_SPLITMIMIC_NAME, help="Split-mimic operational check bundle name.")
    parser.add_argument("--recent-n", type=int, default=10, help="Recent eligible cases to include in the refresh bundle.")
    parser.add_argument("--anchor-id", default="", help="Optional exact anchor_id forwarded to the scorer.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    refresh_dir = project_root / "reports" / "backtests" / "temporal_multires" / args.refresh_name
    scoring_dir = project_root / "reports" / "backtests" / "temporal_multires" / args.scoring_name
    policy_dir = project_root / "reports" / "backtests" / "temporal_multires" / args.policy_name
    reference_check_dir = project_root / "reports" / "backtests" / "temporal_multires" / args.reference_check_name
    splitmimic_dir = project_root / "reports" / "backtests" / "temporal_multires" / args.splitmimic_name
    ensure_dir(refresh_dir)

    scorer_cmd = [
        sys.executable,
        str(project_root / "score_temporal_flat_winner_v1.py"),
        "--project-root",
        str(project_root),
        "--run-name",
        args.run_name,
        "--policy-name",
        args.policy_name,
        "--scoring-name",
        args.scoring_name,
        "--recent-n",
        str(args.recent_n),
    ]
    if args.anchor_id:
        scorer_cmd.extend(["--anchor-id", args.anchor_id])

    scorer_run = subprocess.run(
        scorer_cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )

    selected_case = load_json(scoring_dir / "selected_case.json")
    scoring_manifest = load_json(scoring_dir / "scoring_manifest.json")
    policy_payload = load_json(policy_dir / "operational_policy.json")
    reference_check_manifest = load_json(reference_check_dir / "analysis_manifest.json")
    splitmimic_manifest = load_json(splitmimic_dir / "analysis_manifest.json")

    history_df = pd.read_csv(scoring_dir / "history_scores.csv", low_memory=False)
    history_df["anchor_id"] = history_df["anchor_id"].astype(str)
    eligible_df = filter_eligible_rows(history_df, enabled_modalities=scoring_manifest["modalities"])

    if "anchor_period_start" in eligible_df.columns:
        eligible_df["_sort_ts"] = pd.to_datetime(eligible_df["anchor_period_start"], errors="coerce")
        eligible_df = eligible_df.sort_values(by=["_sort_ts", "anchor_id"], ascending=[True, True], na_position="last")
        eligible_df = eligible_df.drop(columns=["_sort_ts"], errors="ignore")

    recent_df = eligible_df.tail(args.recent_n).copy()
    trailing_history_df = eligible_df.iloc[:-1].tail(args.recent_n).copy()
    historical_prior_df = eligible_df.iloc[:-1].copy()

    recent_summary = summarize_recent_window(recent_df=recent_df, threshold=float(selected_case["locked_threshold"]))
    trailing_distribution = summarize_score_distribution(trailing_history_df)
    historical_distribution = summarize_score_distribution(historical_prior_df)
    rolling_positive_rate = rolling_positive_rate_summary(historical_prior_df, window_n=args.recent_n)

    splitmimic_time_aware = pd.read_csv(splitmimic_dir / "time_aware_rolling_check.csv", low_memory=False)
    splitmimic_summary = {
        "fold_count": int(splitmimic_time_aware["fold"].nunique()),
        "balanced_accuracy_mean": float(splitmimic_time_aware["balanced_accuracy"].mean()),
        "balanced_accuracy_min": float(splitmimic_time_aware["balanced_accuracy"].min()),
        "balanced_accuracy_max": float(splitmimic_time_aware["balanced_accuracy"].max()),
        "latest_fold_balanced_accuracy": float(splitmimic_time_aware.iloc[-1]["balanced_accuracy"]),
        "latest_fold_roc_auc": float(splitmimic_time_aware.iloc[-1]["roc_auc"]),
    }

    watch_payload = build_watch_payload(
        latest_case=selected_case,
        recent_summary=recent_summary,
        trailing_distribution=trailing_distribution,
        historical_distribution=historical_distribution,
        rolling_positive_rate=rolling_positive_rate,
        promotion_low=float(policy_payload["candidate_promotion_zone"]["low"]),
        promotion_high=float(policy_payload["candidate_promotion_zone"]["high"]),
    )

    refresh_payload = {
        "refresh_name": args.refresh_name,
        "refresh_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "winner_run_name": args.run_name,
        "policy_name": args.policy_name,
        "scoring_name": args.scoring_name,
        "latest_case": {key: json_safe(value) for key, value in selected_case.items()},
        "recent_summary": recent_summary,
        "score_context": {
            "recent_trailing_distribution": trailing_distribution,
            "historical_prior_distribution": historical_distribution,
            "historical_rolling_positive_rate": rolling_positive_rate,
        },
        "policy": {
            "target": policy_payload["target"],
            "modalities": policy_payload["modalities"],
            "model_family": policy_payload["model_family"],
            "locked_threshold": float(policy_payload["locked_threshold"]),
            "candidate_promotion_zone": {
                "low": float(policy_payload["candidate_promotion_zone"]["low"]),
                "high": float(policy_payload["candidate_promotion_zone"]["high"]),
            },
            "decision_framing": policy_payload["decision_framing"],
            "promotion_condition": policy_payload["promotion_condition"],
        },
        "watch_checks": watch_payload,
        "reference_checks": {
            "reference_check_name": args.reference_check_name,
            "reference_check_time_aware": reference_check_manifest["time_aware_check"],
            "splitmimic": splitmimic_summary,
            "splitmimic_time_aware": splitmimic_manifest["time_aware_check"],
        },
        "source_artifacts": {
            "selected_case_json": str(scoring_dir / "selected_case.json"),
            "history_scores_csv": str(scoring_dir / "history_scores.csv"),
            "recent_cases_csv": str(scoring_dir / "recent_cases.csv"),
            "policy_json": str(policy_dir / "operational_policy.json"),
            "reference_check_summary": str(reference_check_dir / "summary.md"),
            "splitmimic_summary": str(splitmimic_dir / "summary.md"),
        },
    }

    latest_case_md = build_latest_case_markdown(refresh_payload["latest_case"])
    recent_summary_md = build_recent_summary_markdown(recent_summary, trailing_distribution, rolling_positive_rate)
    watch_checks_md = build_watch_markdown(watch_payload)
    summary_md = build_summary_markdown(refresh_payload=refresh_payload, report_dir=refresh_dir)

    (refresh_dir / "current_state.json").write_text(json.dumps(refresh_payload, indent=2), encoding="utf-8")
    pd.DataFrame([refresh_payload["latest_case"]]).to_csv(refresh_dir / "latest_case_summary.csv", index=False)
    pd.DataFrame([recent_summary]).to_csv(refresh_dir / "recent_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "watch_name": name,
                "status": payload["status"],
                "reason": payload["reason"],
            }
            for name, payload in watch_payload.items()
            if isinstance(payload, dict) and "status" in payload
        ]
    ).to_csv(refresh_dir / "watch_checks.csv", index=False)
    (refresh_dir / "latest_case_summary.md").write_text(latest_case_md, encoding="utf-8")
    (refresh_dir / "recent_summary.md").write_text(recent_summary_md, encoding="utf-8")
    (refresh_dir / "watch_checks.md").write_text(watch_checks_md, encoding="utf-8")
    (refresh_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    refresh_manifest = {
        "refresh_name": args.refresh_name,
        "winner_run_name": args.run_name,
        "policy_name": args.policy_name,
        "scoring_name": args.scoring_name,
        "recent_n": int(args.recent_n),
        "anchor_id": args.anchor_id or None,
        "refresh_timestamp_utc": refresh_payload["refresh_timestamp_utc"],
        "scorer_command": scorer_cmd,
        "scorer_stdout_path": str(refresh_dir / "scorer_stdout.txt"),
        "scorer_stderr_path": str(refresh_dir / "scorer_stderr.txt"),
        "outputs": {
            "current_state_json": str(refresh_dir / "current_state.json"),
            "latest_case_summary_md": str(refresh_dir / "latest_case_summary.md"),
            "recent_summary_md": str(refresh_dir / "recent_summary.md"),
            "watch_checks_md": str(refresh_dir / "watch_checks.md"),
            "summary_md": str(refresh_dir / "summary.md"),
        },
    }
    (refresh_dir / "refresh_manifest.json").write_text(json.dumps(refresh_manifest, indent=2), encoding="utf-8")
    (refresh_dir / "scorer_stdout.txt").write_text(scorer_run.stdout, encoding="utf-8")
    (refresh_dir / "scorer_stderr.txt").write_text(scorer_run.stderr, encoding="utf-8")

    print(f"anchor_id={selected_case['anchor_id']}")
    print(f"score={float(selected_case['score']):.6f}")
    print(f"locked_decision={selected_case['decision_locked_label']}")
    print(f"recent_positive_rate_locked={format_float(recent_summary['positive_rate_locked'])}")
    print(f"watch_latest_score_vs_recent={watch_payload['latest_score_vs_recent_window']['status']}")
    print(f"watch_recent_positive_rate_shift={watch_payload['recent_positive_rate_shift']['status']}")
    print(f"watch_candidate_promotion_zone={watch_payload['candidate_promotion_zone']['in_zone']}")
    print(f"report_dir={refresh_dir}")


if __name__ == "__main__":
    main()
