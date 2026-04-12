from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[training-decision] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def build_meal_decision_points(project_root: Path) -> None:
    fused_dir = project_root / "fused"
    meal_dir = project_root / "meal_db" / "final_repaired"
    weather_dir = project_root / "weather"
    training_dir = project_root / "training"
    ensure_dir(training_dir)

    log("Loading meal semantics, component edges, daily context, intraday telemetry, and weather...")
    meals = read_required(meal_dir / "meal_semantic_features.csv")
    comps = read_required(meal_dir / "meal_component_edge.csv")
    daily = read_required(training_dir / "day_feature_matrix.csv")
    intraday = read_required(fused_dir / "master_15min_telemetry_active.csv")
    weather15 = read_required(weather_dir / "weather_context_15min.csv")

    meals["date"] = pd.to_datetime(meals["date"], errors="coerce").dt.floor("D")
    meals["datetime_local_approx"] = pd.to_datetime(meals["datetime_local_approx"], errors="coerce")
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.floor("D")
    intraday["datetime_local"] = pd.to_datetime(intraday["datetime_local"], errors="coerce")
    weather15["datetime_local"] = pd.to_datetime(weather15["datetime_local"], errors="coerce")
    comps["date"] = pd.to_datetime(comps["date"], errors="coerce").dt.floor("D")
    comps["datetime_local_approx"] = pd.to_datetime(comps["datetime_local_approx"], errors="coerce")

    meals = meals.sort_values("datetime_local_approx").reset_index(drop=True)
    intraday = intraday.sort_values("datetime_local").reset_index(drop=True)
    weather15 = weather15.sort_values("datetime_local").reset_index(drop=True)

    # Component summaries per meal for target metadata
    comp_summary = comps.groupby("meal_id", as_index=False).agg(
        target_component_rows=("food_entry_uuid", "count"),
        target_distinct_entities=("canonical_entity_id", lambda s: s.dropna().nunique()),
        target_component_role_main_count=("component_role_final", lambda s: (s.astype(str) == "main").sum()),
        target_component_role_side_count=("component_role_final", lambda s: (s.astype(str) == "side").sum()),
        target_component_role_beverage_count=("component_role_final", lambda s: (s.astype(str) == "beverage").sum()),
        target_component_role_condiment_count=("component_role_final", lambda s: (s.astype(str) == "condiment").sum()),
        target_component_role_dessert_count=("component_role_final", lambda s: (s.astype(str) == "dessert").sum()),
        target_component_role_protein_anchor_count=("component_role_final", lambda s: (s.astype(str) == "protein_anchor").sum()),
        target_component_role_starch_base_count=("component_role_final", lambda s: (s.astype(str) == "starch_base").sum()),
        target_top_canonical_entities=("canonical_display_name", lambda s: " | ".join(pd.Series(s.dropna().astype(str)).value_counts().head(5).index.tolist())),
    )

    # Prior meal context from meal sequence itself
    prior_cols = [
        "meal_id",
        "meal_archetype_primary",
        "cuisine_primary",
        "service_form_primary",
        "prep_profile",
        "principal_protein",
        "principal_starch",
        "principal_veg",
        "principal_fat_source",
        "energy_density_style",
        "satiety_style",
        "comfort_food_score",
        "fresh_light_score",
        "indulgence_score",
        "restaurant_specific_flag",
        "generic_standin_flag",
        "calories_kcal",
        "protein_g",
        "carbs_g",
        "fat_g",
    ]
    prior_lookup = meals[[c for c in prior_cols if c in meals.columns]].copy()
    prior_lookup = prior_lookup.rename(columns={c: f"prior_{c}" for c in prior_lookup.columns if c != "meal_id"})
    prior_lookup = prior_lookup.rename(columns={"meal_id": "prior_meal_id"})

    # Merge nearest prior intraday telemetry row
    decision = meals.copy()
    decision = pd.merge_asof(
        decision.sort_values("datetime_local_approx"),
        intraday.sort_values("datetime_local"),
        left_on="datetime_local_approx",
        right_on="datetime_local",
        direction="backward",
        tolerance=pd.Timedelta("2H"),
        suffixes=("", "_intraday"),
    )

    # Merge nearest prior weather row
    decision = pd.merge_asof(
        decision.sort_values("datetime_local_approx"),
        weather15.sort_values("datetime_local"),
        left_on="datetime_local_approx",
        right_on="datetime_local",
        direction="backward",
        tolerance=pd.Timedelta("2H"),
        suffixes=("", "_weather"),
    )

    # Merge day context
    decision = decision.merge(daily, on="date", how="left", suffixes=("", "_day"))

    # Merge target component summaries
    decision = decision.merge(comp_summary, on="meal_id", how="left")

    # Merge prior meal semantics
    if "prior_meal_id" in decision.columns:
        decision = decision.merge(prior_lookup, on="prior_meal_id", how="left")

    # Build decision-point style columns
    out = pd.DataFrame()
    out["meal_id"] = decision["meal_id"]
    out["date"] = decision["date"]
    out["decision_time"] = decision["datetime_local_approx"]
    out["decision_hour"] = decision["datetime_local_approx"].dt.hour + decision["datetime_local_approx"].dt.minute.fillna(0) / 60.0
    out["time_slot"] = decision.get("time_slot")
    out["time_slot_label"] = decision.get("time_slot_label")
    out["meal_order_in_day"] = decision.get("meal_order_in_day")
    out["day_meal_count"] = decision.get("day_meal_count")
    out["is_first_meal_of_day"] = decision.get("is_first_meal_of_day")
    out["is_last_meal_of_day"] = decision.get("is_last_meal_of_day")
    out["hours_since_prior_meal"] = decision.get("hours_since_prior_meal")
    out["hours_until_next_meal"] = decision.get("hours_until_next_meal")
    out["cumulative_meal_calories_before_meal"] = decision.get("cumulative_meal_calories_before_meal")
    out["remaining_budget_before_meal_kcal"] = decision.get("remaining_budget_before_meal_kcal")

    # Daily state before meal (copied from daily matrix)
    daily_state_cols = [
        "true_weight_lb",
        "weight_ema_7d_lb",
        "weight_velocity_7d_lb",
        "weight_ema_14d_lb",
        "weight_velocity_14d_lb",
        "weight_ema_30d_lb",
        "weight_velocity_30d_lb",
        "noom_steps",
        "samsung_pedometer_steps",
        "samsung_activity_steps",
        "samsung_rest_calorie_kcal",
        "samsung_active_calorie_kcal",
        "samsung_exercise_session_count",
        "samsung_exercise_duration_ms",
        "samsung_exercise_calorie_kcal",
        "calorie_budget_kcal",
        "base_calorie_budget_kcal",
        "weight_loss_zone_lower_kcal",
        "weight_loss_zone_upper_kcal",
        "manual_calorie_adjustment_kcal",
        "noom_food_calories_kcal",
        "noom_food_protein_g",
        "noom_food_carbs_g",
        "noom_food_fat_g",
        "noom_food_fiber_g",
        "noom_food_sodium_mg",
        "dominant_meal_archetype",
        "dominant_cuisine",
        "dominant_service_form",
        "dominant_prep_profile",
        "dominant_principal_protein",
        "dominant_principal_starch",
        "dominant_principal_veg",
        "dominant_energy_density_style",
        "dominant_satiety_style",
        "first_meal_hour",
        "last_meal_hour",
        "eating_window_hours",
        "budget_minus_noom_food_calories_kcal",
        "steps_gap_samsung_minus_noom",
        "day_of_week",
        "day_of_week_num",
        "is_weekend",
        "month",
        "month_name",
        "quarter",
        "season",
    ]
    for c in daily_state_cols:
        if c in decision.columns:
            out[f"state_{c}"] = decision[c]

    # Intraday physiology/activity at decision time
    intraday_cols = [
        "heart_rate_bpm",
        "stress_score",
        "step_count",
        "distance_m",
        "calorie_kcal",
        "bmr_15min_kcal",
        "total_burn_15min_kcal",
        "cumulative_daily_burn_kcal",
        "is_meal_event",
    ]
    for c in intraday_cols:
        if c in decision.columns:
            out[f"state_intraday_{c}"] = decision[c]

    # Weather/daylight at decision time
    weather_cols = [
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "snow_depth",
        "weather_code",
        "cloud_cover",
        "wind_speed_10m",
        "wind_gusts_10m",
        "shortwave_radiation",
        "is_day_corrected",
        "is_dark_hour",
        "is_cloudy_hour",
        "is_gloomy_hour",
        "is_hot_hour",
        "is_cold_hour",
        "temp_band_f",
        "apparent_temp_band_f",
        "daylight_hours",
        "sunshine_hours",
        "is_precip_day",
        "is_rain_day",
        "is_snow_day",
        "is_dark_early",
        "is_short_day",
        "is_long_day",
        "is_gloomy_day",
        "precip_streak_days",
        "rain_streak_days",
        "snow_streak_days",
        "gloomy_streak_days",
        "dark_early_streak_days",
        "hot_streak_days",
        "cold_streak_days",
    ]
    for c in weather_cols:
        if c in decision.columns:
            out[f"state_weather_{c}"] = decision[c]

    # Prior meal context
    prior_state_cols = [c for c in decision.columns if c.startswith("prior_")]
    for c in prior_state_cols:
        out[f"state_{c}"] = decision[c]

    # Targets = the meal to predict/recommend
    target_cols = [
        "meal_text",
        "calories_kcal",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "sodium_mg",
        "item_count",
        "distinct_alias_count",
        "distinct_entity_count",
        "meal_archetype_primary",
        "meal_archetype_secondary",
        "cuisine_primary",
        "cuisine_secondary",
        "service_form_primary",
        "prep_profile",
        "principal_protein",
        "principal_starch",
        "principal_veg",
        "principal_fat_source",
        "comfort_food_score",
        "fresh_light_score",
        "indulgence_score",
        "energy_density_style",
        "satiety_style",
        "coherence_score",
        "restaurant_specific_flag",
        "generic_standin_flag",
        "semantic_confidence",
        "semantic_source",
        "target_component_rows",
        "target_distinct_entities",
        "target_component_role_main_count",
        "target_component_role_side_count",
        "target_component_role_beverage_count",
        "target_component_role_condiment_count",
        "target_component_role_dessert_count",
        "target_component_role_protein_anchor_count",
        "target_component_role_starch_base_count",
        "target_top_canonical_entities",
    ]
    for c in target_cols:
        if c in decision.columns:
            out[f"target_{c}"] = decision[c]

    # Useful post-meal outcome labels for supervised tasks
    outcome_cols = [
        "remaining_budget_after_meal_kcal",
        "next_meal_id",
        "next_meal_text",
    ]
    for c in outcome_cols:
        if c in decision.columns:
            out[f"outcome_{c}"] = decision[c]

    # Simple flags for recommendation modes
    if "target_restaurant_specific_flag" in out.columns:
        out["target_is_restaurant_meal"] = out["target_restaurant_specific_flag"]
    if "target_generic_standin_flag" in out.columns:
        out["target_is_generic_standin"] = out["target_generic_standin_flag"]

    out = out.sort_values("decision_time").reset_index(drop=True)

    manifest: Dict[str, object] = {
        "rows": int(len(out)),
        "decision_time_min": str(out["decision_time"].min()) if len(out) else None,
        "decision_time_max": str(out["decision_time"].max()) if len(out) else None,
        "source_tables": {
            "meal_semantic_features": "meal_semantic_features.csv",
            "meal_component_edge": "meal_component_edge.csv",
            "day_feature_matrix": "day_feature_matrix.csv",
            "master_15min_telemetry_active": "master_15min_telemetry_active.csv",
            "weather_context_15min": "weather_context_15min.csv",
        },
        "coverage": {
            "rows_with_intraday_hr": int(out["state_intraday_heart_rate_bpm"].notna().sum()) if "state_intraday_heart_rate_bpm" in out.columns else 0,
            "rows_with_intraday_stress": int(out["state_intraday_stress_score"].notna().sum()) if "state_intraday_stress_score" in out.columns else 0,
            "rows_with_weather": int(out["state_weather_temperature_2m"].notna().sum()) if "state_weather_temperature_2m" in out.columns else 0,
            "rows_with_prior_meal": int(out["state_prior_meal_id"].notna().sum()) if "state_prior_meal_id" in out.columns else 0,
        },
        "columns": list(out.columns),
    }

    log("Writing meal decision points dataset...")
    out.to_csv(training_dir / "meal_decision_points.csv", index=False)
    (training_dir / "meal_decision_points_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log("Done.")
    log(f"Wrote: {training_dir / 'meal_decision_points.csv'}")
    log(f"Wrote: {training_dir / 'meal_decision_points_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one-row-per-meal decision point dataset from meal semantics + daily/intraday state.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    args = parser.parse_args()
    build_meal_decision_points(Path(args.project_root).expanduser().resolve())


if __name__ == "__main__":
    main()
