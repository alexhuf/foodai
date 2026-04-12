from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[training-weekend] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def dominant_mode(series: pd.Series):
    s = series.dropna().astype(str)
    if s.empty:
        return pd.NA
    vc = s.value_counts()
    return vc.index[0]


def safe_first(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan


def safe_last(series: pd.Series):
    s = series.dropna()
    return s.iloc[-1] if not s.empty else np.nan


def build_weekend_summary_matrix(project_root: Path) -> None:
    training_dir = project_root / "training"
    ensure_dir(training_dir)

    day_path = training_dir / "day_feature_matrix.csv"
    log("Loading day feature matrix...")
    day = read_required(day_path)
    day["date"] = pd.to_datetime(day["date"], errors="coerce").dt.floor("D")
    day = day.sort_values("date").reset_index(drop=True)

    # Friday-based weekends: Fri/Sat/Sun
    day["day_of_week_num"] = day["date"].dt.dayofweek  # Mon=0 ... Sun=6
    weekend = day[day["day_of_week_num"].isin([4, 5, 6])].copy()
    weekend["days_since_friday"] = weekend["day_of_week_num"] - 4
    weekend["weekend_start"] = weekend["date"] - pd.to_timedelta(weekend["days_since_friday"], unit="D")
    weekend["weekend_end"] = weekend["weekend_start"] + pd.Timedelta(days=2)
    weekend["weekend_id"] = weekend["weekend_start"].dt.strftime("%Y-%m-%d")
    weekend["year"] = weekend["weekend_start"].dt.year
    weekend["month"] = weekend["weekend_start"].dt.month
    weekend["month_name"] = weekend["weekend_start"].dt.month_name()
    weekend["season"] = np.select(
        [
            weekend["month"].isin([12, 1, 2]),
            weekend["month"].isin([3, 4, 5]),
            weekend["month"].isin([6, 7, 8]),
            weekend["month"].isin([9, 10, 11]),
        ],
        ["winter", "spring", "summer", "fall"],
        default="unknown",
    )

    numeric_cols = [
        # core daily/fused
        "noom_food_calories_kcal",
        "noom_food_protein_g",
        "noom_food_carbs_g",
        "noom_food_fat_g",
        "noom_food_fiber_g",
        "noom_food_sodium_mg",
        "noom_meal_event_count",
        "noom_steps",
        "noom_water_liters",
        "calorie_budget_kcal",
        "base_calorie_budget_kcal",
        "weight_loss_zone_lower_kcal",
        "weight_loss_zone_upper_kcal",
        "manual_calorie_adjustment_kcal",
        "true_weight_lb",
        "weight_ema_7d_lb",
        "weight_velocity_7d_lb",
        "weight_ema_14d_lb",
        "weight_velocity_14d_lb",
        "weight_ema_30d_lb",
        "weight_velocity_30d_lb",
        "samsung_pedometer_steps",
        "samsung_activity_steps",
        "samsung_rest_calorie_kcal",
        "samsung_active_calorie_kcal",
        "samsung_exercise_session_count",
        "samsung_exercise_duration_ms",
        "samsung_exercise_calorie_kcal",
        "samsung_sleep_duration_ms",
        "samsung_sleep_score",
        # semantic day rollups
        "meal_event_count",
        "distinct_meal_archetypes",
        "distinct_cuisines",
        "restaurant_specific_meal_count",
        "generic_standin_meal_count",
        "first_meal_hour",
        "last_meal_hour",
        "eating_window_hours",
        "meal_calories_kcal_sum",
        "meal_calories_kcal_mean",
        "meal_protein_g_sum",
        "meal_carbs_g_sum",
        "meal_fat_g_sum",
        "meal_fiber_g_sum",
        "meal_sodium_mg_sum",
        "meal_component_count_sum",
        "meal_main_component_count_sum",
        "meal_protein_anchor_count_sum",
        "meal_starch_base_count_sum",
        "meal_side_component_count_from_roles_sum",
        "meal_beverage_component_count_from_roles_sum",
        "meal_condiment_component_count_from_roles_sum",
        "meal_dessert_component_count_from_roles_sum",
        "meal_comfort_food_score_sum",
        "meal_comfort_food_score_mean",
        "meal_fresh_light_score_sum",
        "meal_fresh_light_score_mean",
        "meal_indulgence_score_sum",
        "meal_indulgence_score_mean",
        "meal_semantic_confidence_mean",
        "meal_hours_since_prior_meal_mean",
        "meal_hours_until_next_meal_mean",
        "meal_count_breakfast",
        "meal_count_morning_snack",
        "meal_count_lunch",
        "meal_count_afternoon_snack",
        "meal_count_dinner",
        "meal_count_evening_snack",
        # weather
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_mean",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "daylight_hours",
        "sunshine_hours",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "cloud_cover_mean",
        "cloud_cover_max",
        "precip_streak_days",
        "rain_streak_days",
        "snow_streak_days",
        "gloomy_streak_days",
        "dark_early_streak_days",
        "hot_streak_days",
        "cold_streak_days",
        # derived day
        "budget_minus_noom_food_calories_kcal",
        "steps_gap_samsung_minus_noom",
    ]
    numeric_cols = [c for c in numeric_cols if c in weekend.columns]

    agg_numeric: Dict[str, list[str]] = {}
    for c in numeric_cols:
        funcs = ["sum", "mean", "max"]
        if c in {
            "true_weight_lb", "weight_ema_7d_lb", "weight_ema_14d_lb", "weight_ema_30d_lb",
            "temperature_2m_mean", "apparent_temperature_mean", "daylight_hours",
            "sunshine_hours", "cloud_cover_mean"
        }:
            funcs += ["min"]
        agg_numeric[c] = funcs

    wk_num = weekend.groupby(["weekend_id", "weekend_start", "weekend_end", "year", "month", "month_name", "season"], as_index=False).agg(agg_numeric)
    wk_num.columns = [
        "weekend_id" if c[0] == "weekend_id" else
        "weekend_start" if c[0] == "weekend_start" else
        "weekend_end" if c[0] == "weekend_end" else
        "year" if c[0] == "year" else
        "month" if c[0] == "month" else
        "month_name" if c[0] == "month_name" else
        "season" if c[0] == "season" else
        f"{c[0]}_{c[1]}"
        for c in wk_num.columns.to_flat_index()
    ]

    log("Aggregating weekend categorical summaries...")
    wk_cat = weekend.groupby(["weekend_id", "weekend_start", "weekend_end", "year", "month", "month_name", "season"], as_index=False).agg(
        n_days=("date", "count"),
        friday_present=("day_of_week_num", lambda s: int((s == 4).any())),
        saturday_present=("day_of_week_num", lambda s: int((s == 5).any())),
        sunday_present=("day_of_week_num", lambda s: int((s == 6).any())),
        days_with_meals=("has_meal_semantics_day", lambda s: s.fillna(False).astype(bool).sum() if "has_meal_semantics_day" in weekend.columns else 0),
        days_with_weight=("has_weight_day", lambda s: s.fillna(False).astype(bool).sum() if "has_weight_day" in weekend.columns else 0),
        days_with_activity=("has_activity_day", lambda s: s.fillna(False).astype(bool).sum() if "has_activity_day" in weekend.columns else 0),
        days_with_weather=("has_weather_day", lambda s: s.fillna(False).astype(bool).sum() if "has_weather_day" in weekend.columns else 0),
        dominant_meal_archetype_weekend=("dominant_meal_archetype", dominant_mode if "dominant_meal_archetype" in weekend.columns else safe_first),
        dominant_cuisine_weekend=("dominant_cuisine", dominant_mode if "dominant_cuisine" in weekend.columns else safe_first),
        dominant_service_form_weekend=("dominant_service_form", dominant_mode if "dominant_service_form" in weekend.columns else safe_first),
        dominant_prep_profile_weekend=("dominant_prep_profile", dominant_mode if "dominant_prep_profile" in weekend.columns else safe_first),
        dominant_protein_weekend=("dominant_principal_protein", dominant_mode if "dominant_principal_protein" in weekend.columns else safe_first),
        dominant_starch_weekend=("dominant_principal_starch", dominant_mode if "dominant_principal_starch" in weekend.columns else safe_first),
        dominant_energy_density_weekend=("dominant_energy_density_style", dominant_mode if "dominant_energy_density_style" in weekend.columns else safe_first),
        dominant_satiety_style_weekend=("dominant_satiety_style", dominant_mode if "dominant_satiety_style" in weekend.columns else safe_first),
        first_date_in_weekend=("date", safe_first),
        last_date_in_weekend=("date", safe_last),
        first_weight_lb=("true_weight_lb", safe_first if "true_weight_lb" in weekend.columns else lambda s: np.nan),
        last_weight_lb=("true_weight_lb", safe_last if "true_weight_lb" in weekend.columns else lambda s: np.nan),
        first_weight_ema_7d_lb=("weight_ema_7d_lb", safe_first if "weight_ema_7d_lb" in weekend.columns else lambda s: np.nan),
        last_weight_ema_7d_lb=("weight_ema_7d_lb", safe_last if "weight_ema_7d_lb" in weekend.columns else lambda s: np.nan),
        gloomy_day_count=("is_gloomy_day", lambda s: s.fillna(False).astype(bool).sum() if "is_gloomy_day" in weekend.columns else 0),
        rain_day_count=("is_rain_day", lambda s: s.fillna(False).astype(bool).sum() if "is_rain_day" in weekend.columns else 0),
        snow_day_count=("is_snow_day", lambda s: s.fillna(False).astype(bool).sum() if "is_snow_day" in weekend.columns else 0),
        hot_day_count=("is_hot_day", lambda s: s.fillna(False).astype(bool).sum() if "is_hot_day" in weekend.columns else 0),
        cold_day_count=("is_cold_day", lambda s: s.fillna(False).astype(bool).sum() if "is_cold_day" in weekend.columns else 0),
        dark_early_day_count=("is_dark_early", lambda s: s.fillna(False).astype(bool).sum() if "is_dark_early" in weekend.columns else 0),
        noom_finished_day_count=("noom_finished_day", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum() if "noom_finished_day" in weekend.columns else 0),
    )

    out = wk_cat.merge(
        wk_num,
        on=["weekend_id", "weekend_start", "weekend_end", "year", "month", "month_name", "season"],
        how="left",
    ).sort_values("weekend_start").reset_index(drop=True)

    # Derived weekend fields
    out["weight_delta_lb"] = out["last_weight_lb"] - out["first_weight_lb"]
    out["weight_ema_7d_delta_lb"] = out["last_weight_ema_7d_lb"] - out["first_weight_ema_7d_lb"]

    if "noom_food_calories_kcal_sum" in out.columns and "calorie_budget_kcal_sum" in out.columns:
        out["budget_minus_logged_food_kcal_weekend"] = out["calorie_budget_kcal_sum"] - out["noom_food_calories_kcal_sum"]

    if "samsung_pedometer_steps_sum" in out.columns and "noom_steps_sum" in out.columns:
        out["steps_gap_samsung_minus_noom_weekend"] = out["samsung_pedometer_steps_sum"] - out["noom_steps_sum"]

    if "meal_event_count_sum" in out.columns and "n_days" in out.columns:
        out["meal_events_per_day_weekend"] = out["meal_event_count_sum"] / out["n_days"].replace(0, np.nan)

    if "restaurant_specific_meal_count_sum" in out.columns and "meal_event_count_sum" in out.columns:
        out["restaurant_meal_fraction_weekend"] = out["restaurant_specific_meal_count_sum"] / out["meal_event_count_sum"].replace(0, np.nan)

    if "generic_standin_meal_count_sum" in out.columns and "meal_event_count_sum" in out.columns:
        out["generic_standin_fraction_weekend"] = out["generic_standin_meal_count_sum"] / out["meal_event_count_sum"].replace(0, np.nan)

    if "gloomy_day_count" in out.columns and "n_days" in out.columns:
        out["gloomy_day_fraction_weekend"] = out["gloomy_day_count"] / out["n_days"].replace(0, np.nan)

    if "snow_day_count" in out.columns and "n_days" in out.columns:
        out["snow_day_fraction_weekend"] = out["snow_day_count"] / out["n_days"].replace(0, np.nan)

    out["weekend_label"] = out["weekend_start"].dt.strftime("%Y-%m-%d") + " to " + out["weekend_end"].dt.strftime("%Y-%m-%d")

    manifest = {
        "rows": int(len(out)),
        "weekend_start_min": str(out["weekend_start"].min().date()) if len(out) else None,
        "weekend_start_max": str(out["weekend_start"].max().date()) if len(out) else None,
        "definition": "Friday-Saturday-Sunday weekends, partial weekends allowed at dataset edges",
        "source_table": str(day_path.name),
        "coverage": {
            "weekends_with_any_meals": int(out["days_with_meals"].gt(0).sum()) if "days_with_meals" in out.columns else 0,
            "weekends_with_any_weight": int(out["days_with_weight"].gt(0).sum()) if "days_with_weight" in out.columns else 0,
            "weekends_with_full_weather": int(out["days_with_weather"].eq(out["n_days"]).sum()) if "days_with_weather" in out.columns else 0,
        },
        "columns": list(out.columns),
    }

    log("Writing weekend summary matrix...")
    out.to_csv(training_dir / "weekend_summary_matrix.csv", index=False)
    (training_dir / "weekend_summary_matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log("Done.")
    log(f"Wrote: {training_dir / 'weekend_summary_matrix.csv'}")
    log(f"Wrote: {training_dir / 'weekend_summary_matrix_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Friday-Sunday weekend summary matrix from day feature matrix.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    args = parser.parse_args()
    build_weekend_summary_matrix(Path(args.project_root).expanduser().resolve())


if __name__ == "__main__":
    main()
