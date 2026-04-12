from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def log(msg: str) -> None:
    print(f"[training-meal-view] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


SAFE_ID_COLS = [
    "meal_id",
    "date",
    "decision_time",
    "decision_hour",
    "time_slot",
    "time_slot_label",
    "meal_order_in_day",
    "day_meal_count",
    "is_first_meal_of_day",
    "is_last_meal_of_day",
    "hours_since_prior_meal",
    "hours_until_next_meal",
    "cumulative_meal_calories_before_meal",
    "remaining_budget_before_meal_kcal",
]

SAFE_STATE_PREFIXES = [
    "state_weight_",
    "state_true_weight_lb",
    "state_calorie_budget_",
    "state_base_calorie_budget_",
    "state_weight_loss_zone_",
    "state_manual_calorie_adjustment_",
    "state_day_of_week",
    "state_day_of_week_num",
    "state_is_weekend",
    "state_month",
    "state_month_name",
    "state_quarter",
    "state_season",
    "state_intraday_",
    "state_weather_",
    "state_prior_",
]

SAFE_STATE_EXACT = [
    "state_samsung_rest_calorie_kcal",
    "state_samsung_active_calorie_kcal",
    "state_samsung_exercise_session_count",
    "state_samsung_exercise_duration_ms",
    "state_samsung_exercise_calorie_kcal",
]

LEAKY_STATE_EXACT = [
    "state_noom_steps",
    "state_samsung_pedometer_steps",
    "state_samsung_activity_steps",
    "state_noom_food_calories_kcal",
    "state_noom_food_protein_g",
    "state_noom_food_carbs_g",
    "state_noom_food_fat_g",
    "state_noom_food_fiber_g",
    "state_noom_food_sodium_mg",
    "state_dominant_meal_archetype",
    "state_dominant_cuisine",
    "state_dominant_service_form",
    "state_dominant_prep_profile",
    "state_dominant_principal_protein",
    "state_dominant_principal_starch",
    "state_dominant_principal_veg",
    "state_dominant_energy_density_style",
    "state_dominant_satiety_style",
    "state_first_meal_hour",
    "state_last_meal_hour",
    "state_eating_window_hours",
    "state_budget_minus_noom_food_calories_kcal",
    "state_steps_gap_samsung_minus_noom",
]

TARGET_COLS = [
    "target_meal_text",
    "target_calories_kcal",
    "target_protein_g",
    "target_carbs_g",
    "target_fat_g",
    "target_fiber_g",
    "target_sodium_mg",
    "target_item_count",
    "target_distinct_alias_count",
    "target_distinct_entity_count",
    "target_meal_archetype_primary",
    "target_meal_archetype_secondary",
    "target_cuisine_primary",
    "target_cuisine_secondary",
    "target_service_form_primary",
    "target_prep_profile",
    "target_principal_protein",
    "target_principal_starch",
    "target_principal_veg",
    "target_principal_fat_source",
    "target_comfort_food_score",
    "target_fresh_light_score",
    "target_indulgence_score",
    "target_energy_density_style",
    "target_satiety_style",
    "target_coherence_score",
    "target_restaurant_specific_flag",
    "target_generic_standin_flag",
    "target_semantic_confidence",
    "target_semantic_source",
    "target_target_component_rows",
    "target_target_distinct_entities",
    "target_target_component_role_main_count",
    "target_target_component_role_side_count",
    "target_target_component_role_beverage_count",
    "target_target_component_role_condiment_count",
    "target_target_component_role_dessert_count",
    "target_target_component_role_protein_anchor_count",
    "target_target_component_role_starch_base_count",
    "target_target_top_canonical_entities",
    "target_is_restaurant_meal",
    "target_is_generic_standin",
]

OUTCOME_COLS = [
    "outcome_remaining_budget_after_meal_kcal",
    "outcome_next_meal_id",
    "outcome_next_meal_text",
]


def col_is_safe_state(col: str) -> bool:
    if col in LEAKY_STATE_EXACT:
        return False
    if col in SAFE_STATE_EXACT:
        return True
    return any(col.startswith(prefix) for prefix in SAFE_STATE_PREFIXES)


def build_meal_prediction_view(project_root: Path) -> None:
    training_dir = project_root / "training"
    pred_dir = training_dir / "predictive_views"
    targets_dir = training_dir / "targets"
    ensure_dir(pred_dir)
    ensure_dir(targets_dir)

    src_path = training_dir / "meal_decision_points.csv"
    log("Loading meal decision points...")
    df = read_required(src_path)

    if "decision_time" in df.columns:
        df["decision_time"] = pd.to_datetime(df["decision_time"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")

    available_cols = set(df.columns)
    keep_cols: List[str] = []
    dropped_cols: List[str] = []

    for c in SAFE_ID_COLS:
        if c in available_cols:
            keep_cols.append(c)

    for c in df.columns:
        if c in keep_cols:
            continue
        if c.startswith("state_"):
            if col_is_safe_state(c):
                keep_cols.append(c)
            else:
                dropped_cols.append(c)
        elif c in TARGET_COLS or c in OUTCOME_COLS:
            keep_cols.append(c)

    view = df[keep_cols].copy()

    if "target_calories_kcal" in view.columns:
        kcal = pd.to_numeric(view["target_calories_kcal"], errors="coerce")
        view["y_next_meal_kcal_log1p"] = kcal.clip(lower=0).map(lambda x: pd.NA if pd.isna(x) else __import__("math").log1p(x))
        view["y_next_meal_kcal_band"] = pd.cut(
            kcal,
            bins=[-1, 250, 500, 800, 1200, 5000],
            labels=["very_light", "light", "medium", "large", "very_large"],
        ).astype("object")

    if "target_meal_archetype_primary" in view.columns:
        vc = view["target_meal_archetype_primary"].astype("object").value_counts(dropna=True)
        common = set(vc[vc >= 15].index.tolist())
        view["y_next_meal_archetype_collapsed"] = view["target_meal_archetype_primary"].where(
            view["target_meal_archetype_primary"].isin(common),
            other="OTHER",
        )

    if "outcome_remaining_budget_after_meal_kcal" in view.columns:
        rb = pd.to_numeric(view["outcome_remaining_budget_after_meal_kcal"], errors="coerce")
        view["y_post_meal_budget_breach"] = rb < 0
        view["y_post_meal_budget_breach_200"] = rb < -200

    if "target_is_restaurant_meal" in view.columns:
        view["y_next_restaurant_meal"] = view["target_is_restaurant_meal"]

    if {"target_indulgence_score", "target_fresh_light_score", "target_comfort_food_score"}.issubset(view.columns):
        indul = pd.to_numeric(view["target_indulgence_score"], errors="coerce")
        fresh = pd.to_numeric(view["target_fresh_light_score"], errors="coerce")
        comfort = pd.to_numeric(view["target_comfort_food_score"], errors="coerce")
        view["y_enjoyment_proxy_v1"] = (comfort.fillna(0) + indul.fillna(0) * 0.5).clip(lower=0)
        view["y_stability_proxy_v1"] = (fresh.fillna(0) - indul.fillna(0) * 0.5).clip(lower=-10, upper=10)

    feature_cols = [c for c in view.columns if c.startswith("state_") or c in SAFE_ID_COLS]
    target_cols_present = [c for c in view.columns if c.startswith("target_") or c.startswith("y_")]
    outcome_cols_present = [c for c in view.columns if c.startswith("outcome_")]

    baseline_targets = {
        "classification": [
            c for c in [
                "y_next_meal_archetype_collapsed",
                "y_next_restaurant_meal",
                "y_post_meal_budget_breach",
                "y_post_meal_budget_breach_200",
                "target_meal_archetype_primary",
                "target_cuisine_primary",
                "target_service_form_primary",
            ] if c in view.columns
        ],
        "regression": [
            c for c in [
                "target_calories_kcal",
                "y_next_meal_kcal_log1p",
                "target_protein_g",
                "target_carbs_g",
                "target_fat_g",
                "target_fiber_g",
                "target_sodium_mg",
                "target_comfort_food_score",
                "target_fresh_light_score",
                "target_indulgence_score",
                "y_enjoyment_proxy_v1",
                "y_stability_proxy_v1",
            ] if c in view.columns
        ],
    }

    manifest: Dict[str, object] = {
        "rows": int(len(view)),
        "source_table": str(src_path.name),
        "coverage": {
            "rows_with_prior_meal": int(view["state_prior_meal_id"].notna().sum()) if "state_prior_meal_id" in view.columns else 0,
            "rows_with_intraday_hr": int(view["state_intraday_heart_rate_bpm"].notna().sum()) if "state_intraday_heart_rate_bpm" in view.columns else 0,
            "rows_with_intraday_stress": int(view["state_intraday_stress_score"].notna().sum()) if "state_intraday_stress_score" in view.columns else 0,
            "rows_with_weather": int(view["state_weather_temperature_2m"].notna().sum()) if "state_weather_temperature_2m" in view.columns else 0,
        },
        "feature_columns": feature_cols,
        "target_columns": target_cols_present,
        "outcome_columns": outcome_cols_present,
        "dropped_as_potentially_leaky": dropped_cols,
        "suggested_baseline_targets": baseline_targets,
    }

    target_spec = {
        "dataset": "meal_prediction_view.csv",
        "feature_policy": {
            "allowed_prefixes": ["state_"],
            "allowed_exact": SAFE_ID_COLS,
            "dropped_as_potentially_leaky": dropped_cols,
            "note": "Use only state_* and safe decision metadata as predictive inputs. target_* and outcome_* are labels, not inputs.",
        },
        "classification_targets": baseline_targets["classification"],
        "regression_targets": baseline_targets["regression"],
    }

    log("Writing meal prediction view...")
    view.to_csv(pred_dir / "meal_prediction_view.csv", index=False)
    (pred_dir / "meal_prediction_view_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (targets_dir / "target_spec_meal_prediction.json").write_text(json.dumps(target_spec, indent=2), encoding="utf-8")

    log("Done.")
    log(f"Wrote: {pred_dir / 'meal_prediction_view.csv'}")
    log(f"Wrote: {pred_dir / 'meal_prediction_view_manifest.json'}")
    log(f"Wrote: {targets_dir / 'target_spec_meal_prediction.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe meal prediction view from meal_decision_points.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    args = parser.parse_args()
    build_meal_prediction_view(Path(args.project_root).expanduser().resolve())


if __name__ == "__main__":
    main()
