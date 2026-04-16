from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from meal_scenario_planning_core_v2 import (
    DEFAULT_HORIZONS,
    add_bounded_day_variants,
    build_day_action_library,
    build_planning_context,
    build_scenario_search_v2,
    ensure_dir,
    load_source_tables,
    markdown_table,
    save_json,
)


DEFAULT_RUN_NAME = "meal_scenario_planning_v2"


def parse_horizons(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_summary(
    rankings: pd.DataFrame,
    plan_details: pd.DataFrame,
    context_payload: dict,
    action_metadata: dict,
    variant_metadata: dict,
    report_dir: Path,
) -> str:
    promoted = rankings.loc[rankings["promoted"]].copy()
    lines = [
        "# Meal Scenario Planning v2",
        "",
        "## What Changed From v1",
        "",
        "- keeps the observed-template realism constraint: every plan is still anchored to historical full-day meal archetype patterns",
        "- adds bounded portion variants by scaling observed templates only within observed archetype-signature calorie ranges",
        "- applies horizon-aware repeat limits to source templates and archetype signatures",
        "- writes plain-language plan and day explanations",
        "- exposes repeat diagnostics in the ranked output",
        "",
        "## Context",
        "",
        f"- planning_start_date: `{context_payload['start_date']}`",
        f"- latest_observed_date: `{context_payload['latest_observed_date']}`",
        f"- recent_steps_mean: `{context_payload['recent_steps_mean']}`",
        f"- recent_food_kcal_mean: `{context_payload['recent_food_kcal_mean']}`",
        f"- recent_dominant_archetypes: `{', '.join(context_payload['recent_dominant_archetypes'])}`",
        "",
        "## Action Space",
        "",
        f"- observed base templates after required-slot filter: `{action_metadata['templates_after_required_slot_filter']}`",
        f"- bounded portion variants added: `{variant_metadata['bounded_variants']}`",
        f"- total day actions in v2 library: `{variant_metadata['total_day_actions_v2']}`",
        f"- portion multiplier range: `{variant_metadata['portion_multiplier_min']:.2f}` to `{variant_metadata['portion_multiplier_max']:.2f}`",
        f"- core calorie band q05/q95: `{action_metadata['bounds']['total_kcal_q05']:.0f}` to `{action_metadata['bounds']['total_kcal_q95']:.0f}` kcal",
        "",
        "## Ranked Promoted Plans",
        "",
        markdown_table(
            promoted,
            [
                "plan_id",
                "horizon_days",
                "strategy",
                "robust_score",
                "robust_weight_support",
                "fragility",
                "unique_source_templates",
                "unique_archetype_signatures",
                "max_source_template_count",
                "max_signature_share",
                "bounded_variant_days",
                "plain_language_explanation",
            ],
            max_rows=20,
        ),
        "",
        "## Best Plan By Horizon",
        "",
    ]
    for horizon in sorted(rankings["horizon_days"].unique()):
        h = rankings.loc[(rankings["horizon_days"] == horizon) & (rankings["promoted"])].copy()
        if h.empty:
            h = rankings.loc[rankings["horizon_days"] == horizon].copy()
        best = h.sort_values("robust_score", ascending=False).iloc[0]
        lines.extend(
            [
                f"### {int(horizon)} days",
                "",
                f"- plan_id: `{best['plan_id']}`",
                f"- robust_score: `{best['robust_score']:.3f}`",
                f"- robust_weight_support: `{best['robust_weight_support']:.3f}`",
                f"- fragility: `{best['fragility']:.3f}`",
                f"- repeat diagnostics: `{int(best['unique_source_templates'])}` source templates, `{int(best['unique_archetype_signatures'])}` archetype signatures, max signature share `{best['max_signature_share']:.2f}`",
                f"- explanation: {best['plain_language_explanation']}",
                f"- promoted: `{bool(best['promoted'])}`",
                f"- rejection_reasons: `{best['rejection_reasons'] or 'none'}`",
                "",
            ]
        )
        details = plan_details.loc[plan_details["plan_id"] == best["plan_id"]]
        lines.append(
            markdown_table(
                details,
                [
                    "day_index",
                    "planned_date",
                    "planned_day_of_week",
                    "source_date",
                    "source_template_id",
                    "portion_variant",
                    "total_kcal",
                    "dominant_archetype",
                    "loss_support_raw",
                    "slot_summary",
                    "day_explanation",
                ],
                max_rows=min(int(horizon), 14),
            )
        )
        if int(horizon) > 14:
            lines.append("")
            lines.append("_First 14 days shown; full detail is in `plan_details.csv`._")
        lines.append("")
    lines.extend(
        [
            "## Bundle Files",
            "",
            f"- rankings: `{report_dir / 'scenario_rankings.csv'}`",
            f"- plan details: `{report_dir / 'plan_details.csv'}`",
            f"- robustness stress table: `{report_dir / 'robustness_stress_tests.csv'}`",
            f"- day action library: `{report_dir / 'day_action_library.csv'}`",
            f"- manifest: `{report_dir / 'planning_manifest.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search observed meal-template scenarios with v2 repeat/explanation improvements.")
    parser.add_argument("--project-root", default=".", help="Path to the FoodAI repo root.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Report bundle name.")
    parser.add_argument("--horizons", default=",".join(str(x) for x in DEFAULT_HORIZONS))
    parser.add_argument("--candidates-per-horizon", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default="", help="Optional planning start date. Defaults to day after latest observed data.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    report_dir = project_root / "reports" / "backtests" / "meal_scenario_planning" / args.run_name
    ensure_dir(report_dir)

    tables = load_source_tables(project_root)
    context = build_planning_context(tables["transitions"], start_date=args.start_date or None)
    day_actions_base, action_metadata = build_day_action_library(tables)
    day_actions, variant_metadata = add_bounded_day_variants(day_actions_base, action_metadata["bounds"])
    rankings, plan_details, stress = build_scenario_search_v2(
        actions=day_actions,
        context=context,
        horizons=parse_horizons(args.horizons),
        candidates_per_horizon=args.candidates_per_horizon,
        seed=args.seed,
        metadata=action_metadata,
    )

    rankings.to_csv(report_dir / "scenario_rankings.csv", index=False)
    plan_details.to_csv(report_dir / "plan_details.csv", index=False)
    stress.to_csv(report_dir / "robustness_stress_tests.csv", index=False)
    day_actions.to_csv(report_dir / "day_action_library.csv", index=False)

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
        "horizons": parse_horizons(args.horizons),
        "candidates_per_horizon": int(args.candidates_per_horizon),
        "seed": int(args.seed),
        "version": "v2",
        "context": context_payload,
        "action_metadata": action_metadata,
        "variant_metadata": variant_metadata,
        "outputs": {
            "scenario_rankings": str(report_dir / "scenario_rankings.csv"),
            "plan_details": str(report_dir / "plan_details.csv"),
            "robustness_stress_tests": str(report_dir / "robustness_stress_tests.csv"),
            "day_action_library": str(report_dir / "day_action_library.csv"),
            "summary": str(report_dir / "summary.md"),
        },
    }
    save_json(report_dir / "planning_manifest.json", manifest)
    (report_dir / "summary.md").write_text(
        build_summary(rankings, plan_details, context_payload, action_metadata, variant_metadata, report_dir),
        encoding="utf-8",
    )

    promoted = rankings.loc[rankings["promoted"]]
    print(f"report_dir={report_dir}")
    print(f"candidate_plans={len(rankings)}")
    print(f"promoted_plans={len(promoted)}")
    if not promoted.empty:
        best = promoted.sort_values(["horizon_days", "robust_score"], ascending=[True, False]).groupby("horizon_days").head(1)
        print(
            best[
                [
                    "horizon_days",
                    "plan_id",
                    "robust_score",
                    "robust_weight_support",
                    "fragility",
                    "unique_source_templates",
                    "max_signature_share",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()

