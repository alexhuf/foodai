from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

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


DEFAULT_RUN_NAME = "next_meal_scenario_scoring_v2"


def parse_datetime(raw: str) -> datetime:
    if raw:
        return datetime.fromisoformat(raw)
    return datetime.now()


def build_summary(scored, context_payload: dict, current_dt: datetime, report_dir: Path) -> str:
    best = scored.iloc[0] if not scored.empty else None
    lines = [
        "# Next Meal Scenario Scoring v2",
        "",
        "## What Changed From v1",
        "",
        "- keeps the observed-meal realism constraint: recommendations are clusters of historical meal records, not generated meals",
        "- de-duplicates near-identical options by archetype, service form, protein anchor, and canonical components",
        "- reports observed calorie ranges for each cluster as bounded portion guidance",
        "- adds plain-language explanations for why each option scored well",
        "",
        "## Context",
        "",
        f"- current_datetime: `{current_dt.isoformat(sep=' ', timespec='minutes')}`",
        f"- planning_start_date: `{context_payload['start_date']}`",
        f"- latest_observed_date: `{context_payload['latest_observed_date']}`",
        f"- recent_steps_mean: `{context_payload['recent_steps_mean']}`",
        f"- recent_food_kcal_mean: `{context_payload['recent_food_kcal_mean']}`",
        "",
        "## Ranked Next-Meal Option Clusters",
        "",
        markdown_table(
            scored,
            [
                "meal_cluster_id",
                "current_slot",
                "archetype",
                "meal_text",
                "cluster_observed_examples",
                "cluster_kcal_min",
                "cluster_kcal_median",
                "cluster_kcal_max",
                "next_action_score",
                "meal_health",
                "projected_robust_weight_support",
                "plain_language_explanation",
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
                f"- score: `{best['next_action_score']:.3f}`",
                f"- why: {best['plain_language_explanation']}",
                f"- portion guidance: {best['portion_guidance']}",
                f"- projected day template: `{best['projection_template_id']}`",
                f"- projected day pattern: `{best['projected_day_slot_summary']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Bundle Files",
            "",
            f"- scored option clusters: `{report_dir / 'next_meal_scores.csv'}`",
            f"- projection stress tests: `{report_dir / 'next_meal_projection_stress_tests.csv'}`",
            f"- manifest: `{report_dir / 'next_meal_manifest.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score de-duplicated realistic next-meal option clusters.")
    parser.add_argument("--project-root", default=".", help="Path to the FoodAI repo root.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Report bundle name.")
    parser.add_argument("--current-datetime", default="", help="ISO datetime. Defaults to now.")
    parser.add_argument("--start-date", default="", help="Optional planning start date. Defaults to day after latest observed data.")
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    report_dir = project_root / "reports" / "backtests" / "meal_scenario_planning" / args.run_name
    ensure_dir(report_dir)
    current_dt = parse_datetime(args.current_datetime)

    tables = load_source_tables(project_root)
    context = build_planning_context(tables["transitions"], start_date=args.start_date or None)
    day_actions_base, day_metadata = build_day_action_library(tables)
    day_actions, variant_metadata = add_bounded_day_variants(day_actions_base, day_metadata["bounds"])
    meal_actions, meal_metadata = build_meal_action_library(tables)
    scored, stress = score_next_meal_candidates_v2(
        meal_actions=meal_actions,
        day_actions=day_actions,
        context=context,
        current_dt=current_dt,
        top_n=args.top_n,
        bounds=day_metadata["bounds"],
    )

    scored.to_csv(report_dir / "next_meal_scores.csv", index=False)
    stress.to_csv(report_dir / "next_meal_projection_stress_tests.csv", index=False)
    context_payload = {
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
    manifest = {
        "run_name": args.run_name,
        "project_root": str(project_root),
        "current_datetime": current_dt.isoformat(),
        "version": "v2",
        "context": context_payload,
        "day_action_metadata": day_metadata,
        "day_variant_metadata": variant_metadata,
        "meal_action_metadata": meal_metadata,
        "outputs": {
            "next_meal_scores": str(report_dir / "next_meal_scores.csv"),
            "next_meal_projection_stress_tests": str(report_dir / "next_meal_projection_stress_tests.csv"),
            "summary": str(report_dir / "summary.md"),
        },
    }
    save_json(report_dir / "next_meal_manifest.json", manifest)
    (report_dir / "summary.md").write_text(build_summary(scored, context_payload, current_dt, report_dir), encoding="utf-8")

    print(f"report_dir={report_dir}")
    if not scored.empty:
        best = scored.iloc[0]
        print(f"recommended_meal_cluster_id={best['meal_cluster_id']}")
        print(f"representative_meal_action_id={best['meal_action_id']}")
        print(f"current_slot={best['current_slot']}")
        print(f"archetype={best['archetype']}")
        print(f"next_action_score={best['next_action_score']:.3f}")
        print(f"meal_text={best['meal_text']}")


if __name__ == "__main__":
    main()

