from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from meal_scenario_planning_core_v1 import (
    build_day_action_library,
    build_meal_action_library,
    build_planning_context,
    ensure_dir,
    load_source_tables,
    markdown_table,
    save_json,
    score_next_meal_candidates,
)


DEFAULT_RUN_NAME = "next_meal_scenario_scoring_v1"


def parse_datetime(raw: str) -> datetime:
    if raw:
        return datetime.fromisoformat(raw)
    return datetime.now()


def build_summary(scored, context_payload: dict, current_dt: datetime, report_dir: Path) -> str:
    best = scored.iloc[0] if not scored.empty else None
    lines = [
        "# Next Meal Scenario Scoring v1",
        "",
        "## Mode",
        "",
        "- action: observed meal records only, filtered by the inferred current slot and historically repeated archetype/slot combinations",
        "- projection: each next-meal option is linked to observed full-day templates containing that archetype, then stress-tested under the same robustness logic as horizon planning",
        "- reward: meal enjoyment, meal health, robust projected weight-support, routine frequency, and realism",
        "",
        "## Context",
        "",
        f"- current_datetime: `{current_dt.isoformat(sep=' ', timespec='minutes')}`",
        f"- planning_start_date: `{context_payload['start_date']}`",
        f"- latest_observed_date: `{context_payload['latest_observed_date']}`",
        f"- latest_weight_lb: `{context_payload['latest_weight_lb']}`",
        f"- latest_weight_velocity_7d_lb: `{context_payload['latest_weight_velocity_7d_lb']}`",
        f"- recent_steps_mean: `{context_payload['recent_steps_mean']}`",
        f"- recent_food_kcal_mean: `{context_payload['recent_food_kcal_mean']}`",
        "",
        "## Ranked Next-Meal Options",
        "",
        markdown_table(
            scored,
            [
                "meal_action_id",
                "current_slot",
                "archetype",
                "calories_kcal",
                "protein_g",
                "next_action_score",
                "meal_enjoyment",
                "meal_health",
                "projected_robust_weight_support",
                "projected_fragility",
                "meal_text",
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
                f"- meal_action_id: `{best['meal_action_id']}`",
                f"- slot: `{best['current_slot']}`",
                f"- archetype: `{best['archetype']}`",
                f"- calories_kcal: `{best['calories_kcal']:.0f}`",
                f"- score: `{best['next_action_score']:.3f}`",
                f"- observed example: `{best['meal_text']}`",
                f"- projected day template: `{best['projection_template_id']}`",
                f"- projected day pattern: `{best['projected_day_slot_summary']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Bundle Files",
            "",
            f"- scored options: `{report_dir / 'next_meal_scores.csv'}`",
            f"- projection stress tests: `{report_dir / 'next_meal_projection_stress_tests.csv'}`",
            f"- manifest: `{report_dir / 'next_meal_manifest.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score realistic next-meal options under the scenario reward.")
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
    day_actions, day_metadata = build_day_action_library(tables)
    meal_actions, meal_metadata = build_meal_action_library(tables)
    scored, stress = score_next_meal_candidates(
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
        "context": context_payload,
        "day_action_metadata": day_metadata,
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
        print(f"recommended_meal_action_id={best['meal_action_id']}")
        print(f"current_slot={best['current_slot']}")
        print(f"archetype={best['archetype']}")
        print(f"next_action_score={best['next_action_score']:.3f}")
        print(f"meal_text={best['meal_text']}")


if __name__ == "__main__":
    main()
