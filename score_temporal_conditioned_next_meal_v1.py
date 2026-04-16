from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from meal_scenario_planning_core_v1 import PlanningContext
from meal_scenario_planning_core_v2 import (
    add_bounded_day_variants,
    build_day_action_library,
    build_meal_action_library,
    build_planning_context,
    ensure_dir,
    load_source_tables,
    markdown_table,
    save_json,
    score_next_meal_candidates_v2,
)
from score_temporal_flat_winner_v1 import (
    DEFAULT_POLICY_NAME,
    DEFAULT_RUN_NAME,
    DEFAULT_SCORING_NAME,
    load_json,
)


DEFAULT_BRIDGE_NAME = "temporal_conditioned_next_meal_v1"


def parse_datetime(raw: str) -> datetime:
    if raw:
        return datetime.fromisoformat(raw)
    return datetime.now()


def maybe_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def run_locked_temporal_scorer(
    project_root: Path,
    run_name: str,
    policy_name: str,
    scoring_name: str,
    recent_n: int,
    anchor_id: str,
) -> Tuple[List[str], str, str]:
    cmd = [
        sys.executable,
        str(project_root / "score_temporal_flat_winner_v1.py"),
        "--project-root",
        str(project_root),
        "--run-name",
        run_name,
        "--policy-name",
        policy_name,
        "--scoring-name",
        scoring_name,
        "--recent-n",
        str(recent_n),
    ]
    if anchor_id:
        cmd.extend(["--anchor-id", anchor_id])
    completed = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=True)
    return cmd, completed.stdout, completed.stderr


def condition_planning_context(
    context: PlanningContext,
    current_weight_lb: float | None,
    recent_steps_mean: float | None,
    recent_food_kcal_mean: float | None,
) -> PlanningContext:
    return replace(
        context,
        latest_weight_lb=current_weight_lb if current_weight_lb is not None else context.latest_weight_lb,
        recent_steps_mean=recent_steps_mean if recent_steps_mean is not None else context.recent_steps_mean,
        recent_food_kcal_mean=recent_food_kcal_mean if recent_food_kcal_mean is not None else context.recent_food_kcal_mean,
    )


def temporal_pressure(selected_case: Dict, current_state: Dict | None) -> Dict[str, float | str | bool | None]:
    score = float(selected_case["score"])
    threshold = float(selected_case["locked_threshold"])
    policy_band = str(selected_case["policy_band"])
    decision = bool(selected_case["decision_locked"])
    q90 = None
    if current_state:
        q90 = (
            current_state.get("score_context", {})
            .get("historical_prior_distribution", {})
            .get("q90")
        )
    high_ref = float(q90) if q90 is not None else max(threshold + 0.20, 0.70)
    below_threshold_pressure = float(np.clip((threshold - score) / max(threshold, 1e-6), 0.0, 1.0))
    positive_strength = float(np.clip((score - threshold) / max(high_ref - threshold, 1e-6), 0.0, 1.0))
    return {
        "temporal_loss_score": score,
        "locked_threshold": threshold,
        "temporal_decision_locked": decision,
        "temporal_decision_label": str(selected_case["decision_locked_label"]),
        "temporal_policy_band": policy_band,
        "temporal_score_not_calibrated_probability": bool(selected_case["score_not_calibrated_probability"]),
        "temporal_low_loss_support_pressure": below_threshold_pressure,
        "temporal_positive_strength": positive_strength,
    }


def rerank_with_temporal_state(scored: pd.DataFrame, pressure: Dict) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    risk_pressure = float(pressure["temporal_low_loss_support_pressure"])
    positive_strength = float(pressure["temporal_positive_strength"])

    health_w = 0.12 + 0.08 * risk_pressure
    weight_w = 0.14 + 0.12 * risk_pressure
    enjoyment_w = 0.17 - 0.05 * risk_pressure + 0.03 * positive_strength
    consistency_w = 0.10
    realism_w = 0.07
    base_w = 0.50 - 0.08 * risk_pressure
    kcal_penalty_w = 0.04 + 0.08 * risk_pressure

    out["temporal_loss_score"] = float(pressure["temporal_loss_score"])
    out["temporal_low_loss_support_pressure"] = risk_pressure
    out["temporal_positive_strength"] = positive_strength
    out["temporal_policy_band"] = pressure["temporal_policy_band"]
    out["temporal_decision_label"] = pressure["temporal_decision_label"]
    out["temporal_conditioning_mode"] = np.where(
        risk_pressure > 0,
        "below locked loss-support threshold: weight health and robust support more heavily",
        "at or above locked loss-support threshold: preserve enjoyment while maintaining support",
    )
    out["bridge_score"] = (
        base_w * pd.to_numeric(out["next_action_score"], errors="coerce").fillna(0.0)
        + enjoyment_w * pd.to_numeric(out["meal_enjoyment"], errors="coerce").fillna(0.0)
        + health_w * pd.to_numeric(out["meal_health"], errors="coerce").fillna(0.0)
        + weight_w * pd.to_numeric(out["projected_robust_weight_support"], errors="coerce").fillna(0.0)
        + consistency_w * pd.to_numeric(out["slot_archetype_frequency"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        + realism_w * (1.0 - pd.to_numeric(out["projected_fragility"], errors="coerce").fillna(1.0).clip(0.0, 1.0))
        - kcal_penalty_w * pd.to_numeric(out["high_kcal_pressure"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["bridge_rank"] = out["bridge_score"].rank(method="first", ascending=False).astype(int)
    out["temporal_adjustment"] = out["bridge_score"] - pd.to_numeric(out["next_action_score"], errors="coerce").fillna(0.0)
    out["bridge_explanation"] = out.apply(explain_bridge_row, axis=1)
    return out.sort_values(["bridge_score", "next_action_score"], ascending=[False, False]).reset_index(drop=True)


def explain_bridge_row(row: pd.Series) -> str:
    reasons = []
    if float(row.get("temporal_low_loss_support_pressure", 0.0)) > 0:
        reasons.append("current temporal state is below the locked loss-support threshold")
    else:
        reasons.append("current temporal state is at or above the locked loss-support threshold")
    if float(row.get("meal_health", 0.0)) >= 0.65:
        reasons.append("health score is strong")
    if float(row.get("projected_robust_weight_support", 0.0)) >= 0.75:
        reasons.append("projected day support is robust")
    if float(row.get("meal_enjoyment", 0.0)) >= 0.70:
        reasons.append("enjoyment/familiarity remains high")
    if float(row.get("high_kcal_pressure", 0.0)) >= 0.60:
        reasons.append("kcal pressure is a limiting factor")
    return "; ".join(reasons)


def build_context_payload(context: PlanningContext, current_dt: datetime) -> Dict:
    return {
        "current_datetime": current_dt.isoformat(),
        "start_date": context.start_date.date().isoformat(),
        "latest_observed_date": context.latest_observed_date,
        "latest_weight_lb": context.latest_weight_lb,
        "latest_weight_velocity_7d_lb": context.latest_weight_velocity_7d_lb,
        "recent_steps_mean": context.recent_steps_mean,
        "recent_food_kcal_mean": context.recent_food_kcal_mean,
        "recent_restaurant_fraction": context.recent_restaurant_fraction,
        "recent_dominant_archetypes": list(context.recent_dominant_archetypes),
        "season": context.season,
        "weekday": context.weekday,
    }


def build_summary(
    ranked: pd.DataFrame,
    context_payload: Dict,
    temporal_payload: Dict,
    report_dir: Path,
) -> str:
    best = ranked.iloc[0] if not ranked.empty else None
    lines = [
        "# Temporal-Conditioned Next Meal v1",
        "",
        "## Purpose",
        "",
        "This bridge keeps the v2 observed-meal action space and conditions its ranking on the locked `simple_loss_daysweeks_v2` temporal state. It does not retrain the temporal winner and does not generate unconstrained meals.",
        "",
        "## Current Context",
        "",
        f"- current_datetime: `{context_payload['current_datetime']}`",
        f"- latest_observed_date: `{context_payload['latest_observed_date']}`",
        f"- latest_weight_lb: `{context_payload['latest_weight_lb']}`",
        f"- latest_weight_velocity_7d_lb: `{context_payload['latest_weight_velocity_7d_lb']}`",
        f"- recent_steps_mean: `{context_payload['recent_steps_mean']}`",
        f"- recent_food_kcal_mean: `{context_payload['recent_food_kcal_mean']}`",
        f"- recent_dominant_archetypes: `{', '.join(context_payload['recent_dominant_archetypes'])}`",
        "",
        "## Locked Temporal State",
        "",
        f"- anchor_id: `{temporal_payload['selected_case']['anchor_id']}`",
        f"- score: `{temporal_payload['pressure']['temporal_loss_score']:.6f}`",
        f"- locked_threshold: `{temporal_payload['pressure']['locked_threshold']:.4f}`",
        f"- locked_decision: `{temporal_payload['pressure']['temporal_decision_label']}`",
        f"- policy_band: `{temporal_payload['pressure']['temporal_policy_band']}`",
        "- interpretation: `ranking / threshold signal only; not a calibrated probability`",
        f"- low_loss_support_pressure: `{temporal_payload['pressure']['temporal_low_loss_support_pressure']:.3f}`",
        "",
        "## Ranked Recommendation",
        "",
        markdown_table(
            ranked,
            [
                "bridge_rank",
                "meal_cluster_id",
                "current_slot",
                "archetype",
                "meal_text",
                "cluster_kcal_median",
                "bridge_score",
                "next_action_score",
                "meal_enjoyment",
                "meal_health",
                "projected_robust_weight_support",
                "temporal_adjustment",
                "bridge_explanation",
            ],
            max_rows=20,
        ),
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Recommended Action",
                "",
                f"- meal_cluster_id: `{best['meal_cluster_id']}`",
                f"- representative meal_action_id: `{best['meal_action_id']}`",
                f"- slot: `{best['current_slot']}`",
                f"- archetype: `{best['archetype']}`",
                f"- representative observed example: `{best['meal_text']}`",
                f"- bridge_score: `{best['bridge_score']:.3f}`",
                f"- original_v2_score: `{best['next_action_score']:.3f}`",
                f"- why: {best['bridge_explanation']}",
                f"- portion guidance: {best['portion_guidance']}",
                f"- projected day template: `{best['projection_template_id']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Bundle Files",
            "",
            f"- bridge-ranked meals: `{report_dir / 'temporal_conditioned_next_meal_scores.csv'}`",
            f"- projection stress tests: `{report_dir / 'temporal_conditioned_projection_stress_tests.csv'}`",
            f"- bridge manifest: `{report_dir / 'temporal_conditioned_next_meal_manifest.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank observed next-meal options conditioned on the locked temporal winner.")
    parser.add_argument("--project-root", default=".", help="Path to the FoodAI repo root.")
    parser.add_argument("--run-name", default=DEFAULT_BRIDGE_NAME, help="Bridge report bundle name.")
    parser.add_argument("--current-datetime", default="", help="ISO datetime. Defaults to now.")
    parser.add_argument("--start-date", default="", help="Optional planning start date. Defaults to day after latest observed data.")
    parser.add_argument("--current-weight-lb", type=float, default=None, help="Optional current/recent weight override.")
    parser.add_argument("--recent-steps-mean", type=float, default=None, help="Optional recent step mean override.")
    parser.add_argument("--recent-food-kcal-mean", type=float, default=None, help="Optional recent food kcal mean override.")
    parser.add_argument("--top-n", type=int, default=12, help="Ranked recommendations to write.")
    parser.add_argument("--candidate-pool-n", type=int, default=30, help="Observed v2 clusters to generate before bridge re-ranking.")
    parser.add_argument("--temporal-run-name", default=DEFAULT_RUN_NAME, help="Locked temporal winner run name.")
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME, help="Locked temporal policy bundle name.")
    parser.add_argument("--scoring-name", default=DEFAULT_SCORING_NAME, help="Operational temporal scoring bundle name.")
    parser.add_argument("--anchor-id", default="", help="Optional exact temporal anchor_id forwarded to the locked scorer.")
    parser.add_argument("--recent-n", type=int, default=10, help="Recent cases requested from temporal scorer.")
    parser.add_argument(
        "--skip-temporal-refresh",
        action="store_true",
        help="Read the existing operational scoring artifact instead of rerunning the locked scorer.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    report_dir = project_root / "reports" / "backtests" / "meal_scenario_planning" / args.run_name
    ensure_dir(report_dir)
    current_dt = parse_datetime(args.current_datetime)

    scoring_dir = project_root / "reports" / "backtests" / "temporal_multires" / args.scoring_name
    current_state_path = (
        project_root
        / "reports"
        / "backtests"
        / "temporal_multires"
        / "simple_loss_daysweeks_v2_operational_refresh_v1"
        / "current_state.json"
    )
    scorer_cmd: List[str] = []
    scorer_stdout = ""
    scorer_stderr = ""
    if not args.skip_temporal_refresh:
        scorer_cmd, scorer_stdout, scorer_stderr = run_locked_temporal_scorer(
            project_root=project_root,
            run_name=args.temporal_run_name,
            policy_name=args.policy_name,
            scoring_name=args.scoring_name,
            recent_n=args.recent_n,
            anchor_id=args.anchor_id,
        )

    selected_case = load_json(scoring_dir / "selected_case.json")
    current_state = load_json(current_state_path) if current_state_path.exists() else None
    pressure = temporal_pressure(selected_case=selected_case, current_state=current_state)

    tables = load_source_tables(project_root)
    context = build_planning_context(tables["transitions"], start_date=args.start_date or None)
    context = condition_planning_context(
        context=context,
        current_weight_lb=maybe_float(args.current_weight_lb),
        recent_steps_mean=maybe_float(args.recent_steps_mean),
        recent_food_kcal_mean=maybe_float(args.recent_food_kcal_mean),
    )
    day_actions_base, day_metadata = build_day_action_library(tables)
    day_actions, variant_metadata = add_bounded_day_variants(day_actions_base, day_metadata["bounds"])
    meal_actions, meal_metadata = build_meal_action_library(tables)
    scored, stress = score_next_meal_candidates_v2(
        meal_actions=meal_actions,
        day_actions=day_actions,
        context=context,
        current_dt=current_dt,
        top_n=max(args.top_n, args.candidate_pool_n),
        bounds=day_metadata["bounds"],
    )
    ranked = rerank_with_temporal_state(scored=scored, pressure=pressure).head(args.top_n).copy()
    if not ranked.empty:
        keep_ids = set(ranked["meal_action_id"].astype(str))
        stress = stress.loc[stress["meal_action_id"].astype(str).isin(keep_ids)].copy()

    ranked.to_csv(report_dir / "temporal_conditioned_next_meal_scores.csv", index=False)
    stress.to_csv(report_dir / "temporal_conditioned_projection_stress_tests.csv", index=False)
    temporal_payload = {
        "selected_case": selected_case,
        "pressure": pressure,
        "scorer_command": scorer_cmd,
        "scorer_stdout_path": str(report_dir / "temporal_scorer_stdout.txt"),
        "scorer_stderr_path": str(report_dir / "temporal_scorer_stderr.txt"),
    }
    context_payload = build_context_payload(context=context, current_dt=current_dt)
    manifest = {
        "run_name": args.run_name,
        "version": "v1",
        "project_root": str(project_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bridge_scope": "observed v2 next-meal clusters re-ranked by locked simple_loss_daysweeks_v2 temporal state",
        "constraints": {
            "no_temporal_retraining": True,
            "no_unconstrained_meal_generation": True,
            "action_space": "observed meal clusters from score_next_meal_scenario_v2 logic",
            "temporal_score_interpretation": "ranking / threshold signal only; not a calibrated probability",
        },
        "context": context_payload,
        "temporal": temporal_payload,
        "day_action_metadata": day_metadata,
        "day_variant_metadata": variant_metadata,
        "meal_action_metadata": meal_metadata,
        "bridge_weights": {
            "rule": "When the temporal loss score is below the locked threshold, increase health and robust weight-support weights and increase the high-kcal penalty; otherwise preserve more enjoyment weight.",
        },
        "source_artifacts": {
            "temporal_winner": str(project_root / "reports" / "backtests" / "temporal_multires" / args.temporal_run_name),
            "temporal_policy": str(project_root / "reports" / "backtests" / "temporal_multires" / args.policy_name),
            "temporal_scoring": str(scoring_dir),
            "temporal_refresh": str(current_state_path.parent),
            "meal_planner_v2_reference": str(project_root / "reports" / "backtests" / "meal_scenario_planning" / "next_meal_scenario_scoring_v2"),
        },
        "outputs": {
            "ranked_scores": str(report_dir / "temporal_conditioned_next_meal_scores.csv"),
            "projection_stress_tests": str(report_dir / "temporal_conditioned_projection_stress_tests.csv"),
            "summary": str(report_dir / "summary.md"),
        },
    }
    save_json(report_dir / "temporal_conditioned_next_meal_manifest.json", manifest)
    (report_dir / "summary.md").write_text(
        build_summary(ranked=ranked, context_payload=context_payload, temporal_payload=temporal_payload, report_dir=report_dir),
        encoding="utf-8",
    )
    (report_dir / "temporal_scorer_stdout.txt").write_text(scorer_stdout, encoding="utf-8")
    (report_dir / "temporal_scorer_stderr.txt").write_text(scorer_stderr, encoding="utf-8")

    print(f"report_dir={report_dir}")
    print(f"temporal_anchor_id={selected_case['anchor_id']}")
    print(f"temporal_score={float(selected_case['score']):.6f}")
    print(f"temporal_locked_decision={selected_case['decision_locked_label']}")
    if not ranked.empty:
        best = ranked.iloc[0]
        print(f"recommended_meal_cluster_id={best['meal_cluster_id']}")
        print(f"representative_meal_action_id={best['meal_action_id']}")
        print(f"current_slot={best['current_slot']}")
        print(f"archetype={best['archetype']}")
        print(f"bridge_score={best['bridge_score']:.3f}")
        print(f"original_v2_score={best['next_action_score']:.3f}")
        print(f"meal_text={best['meal_text']}")


if __name__ == "__main__":
    main()
