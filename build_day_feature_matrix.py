from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[training-day] {msg}")


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


def first_non_null(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return pd.NA
    return s.iloc[0]


def build_day_feature_matrix(project_root: Path) -> None:
    fused_dir = project_root / "fused"
    meal_dir = project_root / "meal_db" / "final_repaired"
    weather_dir = project_root / "weather"
    training_dir = project_root / "training"
    ensure_dir(training_dir)

    log("Loading fused daily, meal semantics, and weather daily tables...")
    daily = read_required(fused_dir / "master_daily_features.csv")
    meal = read_required(meal_dir / "meal_semantic_features.csv")
    weather = read_required(weather_dir / "weather_context_daily.csv")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.floor("D")
    meal["date"] = pd.to_datetime(meal["date"], errors="coerce").dt.floor("D")
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.floor("D")

    log("Aggregating meal semantics to day level...")
    # Numeric meal-day rollups
    numeric_cols = [
        "calories_kcal",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "sodium_mg",
        "item_count",
        "component_count",
        "main_component_count",
        "protein_anchor_count",
        "starch_base_count",
        "side_component_count_from_roles",
        "beverage_component_count_from_roles",
        "condiment_component_count_from_roles",
        "dessert_component_count_from_roles",
        "comfort_food_score",
        "fresh_light_score",
        "indulgence_score",
        "semantic_confidence",
        "hours_since_prior_meal",
        "hours_until_next_meal",
    ]
    numeric_cols = [c for c in numeric_cols if c in meal.columns]

    agg_numeric = {}
    for c in numeric_cols:
        agg_numeric[c] = ["sum", "mean", "max"]

    meal_num = meal.groupby("date", as_index=False).agg(agg_numeric)
    meal_num.columns = [
        "date" if c[0] == "date" else f"meal_{c[0]}_{c[1]}"
        for c in meal_num.columns.to_flat_index()
    ]

    # Categorical dominant modes / counts
    cat_cols = [
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
        "coherence_score",
        "semantic_source",
    ]
    cat_cols = [c for c in cat_cols if c in meal.columns]

    cat_rollups = {"date": []}
    meal_cat = meal.groupby("date", as_index=False).agg(
        meal_event_count=("meal_id", "count"),
        distinct_meal_archetypes=("meal_archetype_primary", lambda s: s.dropna().nunique() if "meal_archetype_primary" in meal.columns else 0),
        distinct_cuisines=("cuisine_primary", lambda s: s.dropna().nunique() if "cuisine_primary" in meal.columns else 0),
        restaurant_specific_meal_count=("restaurant_specific_flag", lambda s: s.fillna(False).astype(bool).sum() if "restaurant_specific_flag" in meal.columns else 0),
        generic_standin_meal_count=("generic_standin_flag", lambda s: s.fillna(False).astype(bool).sum() if "generic_standin_flag" in meal.columns else 0),
        dominant_meal_archetype=("meal_archetype_primary", dominant_mode if "meal_archetype_primary" in meal.columns else first_non_null),
        dominant_cuisine=("cuisine_primary", dominant_mode if "cuisine_primary" in meal.columns else first_non_null),
        dominant_service_form=("service_form_primary", dominant_mode if "service_form_primary" in meal.columns else first_non_null),
        dominant_prep_profile=("prep_profile", dominant_mode if "prep_profile" in meal.columns else first_non_null),
        dominant_principal_protein=("principal_protein", dominant_mode if "principal_protein" in meal.columns else first_non_null),
        dominant_principal_starch=("principal_starch", dominant_mode if "principal_starch" in meal.columns else first_non_null),
        dominant_principal_veg=("principal_veg", dominant_mode if "principal_veg" in meal.columns else first_non_null),
        dominant_energy_density_style=("energy_density_style", dominant_mode if "energy_density_style" in meal.columns else first_non_null),
        dominant_satiety_style=("satiety_style", dominant_mode if "satiety_style" in meal.columns else first_non_null),
        dominant_semantic_source=("semantic_source", dominant_mode if "semantic_source" in meal.columns else first_non_null),
        first_meal_time=("datetime_local_approx", lambda s: pd.to_datetime(s, errors="coerce").min()),
        last_meal_time=("datetime_local_approx", lambda s: pd.to_datetime(s, errors="coerce").max()),
    )

    # Meal-slot counts
    if "time_slot_label" in meal.columns:
        slot_counts = (
            meal.pivot_table(index="date", columns="time_slot_label", values="meal_id", aggfunc="count", fill_value=0)
            .reset_index()
        )
        slot_counts.columns = ["date"] + [f"meal_count_{c}" for c in slot_counts.columns[1:]]
    else:
        slot_counts = pd.DataFrame({"date": meal["date"].drop_duplicates().sort_values()})

    # Derived meal-day features
    if "first_meal_time" in meal_cat.columns:
        meal_cat["first_meal_hour"] = pd.to_datetime(meal_cat["first_meal_time"], errors="coerce").dt.hour + (
            pd.to_datetime(meal_cat["first_meal_time"], errors="coerce").dt.minute.fillna(0) / 60.0
        )
        meal_cat["last_meal_hour"] = pd.to_datetime(meal_cat["last_meal_time"], errors="coerce").dt.hour + (
            pd.to_datetime(meal_cat["last_meal_time"], errors="coerce").dt.minute.fillna(0) / 60.0
        )
        meal_cat["eating_window_hours"] = meal_cat["last_meal_hour"] - meal_cat["first_meal_hour"]

    meal_daily = meal_cat.merge(meal_num, on="date", how="left").merge(slot_counts, on="date", how="left")

    log("Joining into day feature matrix...")
    out = daily.merge(meal_daily, on="date", how="left").merge(weather, on="date", how="left", suffixes=("", "_weather"))

    out = out.sort_values("date").reset_index(drop=True)

    # Calendar fields
    out["day_of_week"] = out["date"].dt.day_name()
    out["day_of_week_num"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["day_of_week_num"] >= 5
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["month_name"] = out["date"].dt.month_name()
    out["quarter"] = out["date"].dt.quarter
    out["day_of_year"] = out["date"].dt.dayofyear

    # Season
    month = out["month"]
    out["season"] = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
        ],
        ["winter", "spring", "summer", "fall"],
        default="unknown",
    )

    # A few derived behavioral/context features
    if "meal_event_count" in out.columns and "day_total_meal_calories_kcal" in out.columns:
        out["mean_meal_calories_from_day_total"] = out["day_total_meal_calories_kcal"] / out["meal_event_count"].replace(0, np.nan)

    if "calorie_budget_kcal" in out.columns and "day_total_meal_calories_kcal" in out.columns:
        out["budget_minus_logged_meal_calories_kcal"] = out["calorie_budget_kcal"] - out["day_total_meal_calories_kcal"]

    if "noom_food_calories_kcal" in out.columns and "calorie_budget_kcal" in out.columns:
        out["budget_minus_noom_food_calories_kcal"] = out["calorie_budget_kcal"] - out["noom_food_calories_kcal"]

    if "samsung_pedometer_steps" in out.columns and "noom_steps" in out.columns:
        out["steps_gap_samsung_minus_noom"] = out["samsung_pedometer_steps"] - out["noom_steps"]

    # Coverage flags
    out["has_meal_semantics_day"] = out["meal_event_count"].fillna(0) > 0
    out["has_weather_day"] = out["temperature_2m_mean"].notna() if "temperature_2m_mean" in out.columns else False
    out["has_weight_day"] = out["true_weight_lb"].notna() if "true_weight_lb" in out.columns else False
    out["has_activity_day"] = (
        out["samsung_pedometer_steps"].notna() | out["samsung_activity_steps"].notna()
    ) if "samsung_pedometer_steps" in out.columns and "samsung_activity_steps" in out.columns else False

    manifest: Dict[str, object] = {
        "rows": int(len(out)),
        "date_min": str(out["date"].min().date()) if len(out) else None,
        "date_max": str(out["date"].max().date()) if len(out) else None,
        "source_tables": {
            "fused_master_daily_features": str((fused_dir / "master_daily_features.csv").name),
            "meal_semantic_features": str((meal_dir / "meal_semantic_features.csv").name),
            "weather_context_daily": str((weather_dir / "weather_context_daily.csv").name),
        },
        "coverage": {
            "days_with_meal_events": int(out["meal_event_count"].fillna(0).gt(0).sum()) if "meal_event_count" in out.columns else 0,
            "days_with_weather": int(out["has_weather_day"].fillna(False).sum()),
            "days_with_weight": int(out["has_weight_day"].fillna(False).sum()),
            "days_with_activity": int(out["has_activity_day"].fillna(False).sum()),
        },
        "columns": list(out.columns),
    }

    log("Writing day feature matrix...")
    out.to_csv(training_dir / "day_feature_matrix.csv", index=False)
    (training_dir / "day_feature_matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log("Done.")
    log(f"Wrote: {training_dir / 'day_feature_matrix.csv'}")
    log(f"Wrote: {training_dir / 'day_feature_matrix_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build day-level training feature matrix from fused daily + meal semantics + weather.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    args = parser.parse_args()
    build_day_feature_matrix(Path(args.project_root).expanduser().resolve())


if __name__ == "__main__":
    main()
