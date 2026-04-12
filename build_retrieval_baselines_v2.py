from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

# ---------------------------
# Curated feature sets
# ---------------------------

MEAL_CONTEXT_FEATURES = [
    "decision_hour",
    "time_slot",
    "time_slot_label",
    "meal_order_in_day",
    "is_first_meal_of_day",
    "hours_since_prior_meal",
    "cumulative_meal_calories_before_meal",
    "remaining_budget_before_meal_kcal",
    "state_true_weight_lb",
    "state_weight_ema_7d_lb",
    "state_weight_velocity_7d_lb",
    "state_weight_ema_14d_lb",
    "state_weight_velocity_14d_lb",
    "state_weight_ema_30d_lb",
    "state_weight_velocity_30d_lb",
    "state_samsung_rest_calorie_kcal",
    "state_samsung_active_calorie_kcal",
    "state_samsung_exercise_session_count",
    "state_samsung_exercise_duration_ms",
    "state_samsung_exercise_calorie_kcal",
    "state_calorie_budget_kcal",
    "state_base_calorie_budget_kcal",
    "state_weight_loss_zone_lower_kcal",
    "state_weight_loss_zone_upper_kcal",
    "state_manual_calorie_adjustment_kcal",
    "state_day_of_week",
    "state_day_of_week_num",
    "state_is_weekend",
    "state_month",
    "state_month_name",
    "state_quarter",
    "state_season",
    "state_intraday_heart_rate_bpm",
    "state_intraday_stress_score",
    "state_intraday_step_count",
    "state_intraday_distance_m",
    "state_intraday_bmr_15min_kcal",
    "state_intraday_total_burn_15min_kcal",
    "state_intraday_cumulative_daily_burn_kcal",
    "state_weather_temperature_2m",
    "state_weather_apparent_temperature",
    "state_weather_cloud_cover",
    "state_weather_wind_speed_10m",
    "state_weather_shortwave_radiation",
    "state_weather_is_day_corrected",
    "state_weather_is_dark_hour",
    "state_weather_is_gloomy_hour",
    "state_weather_temp_band_f",
    "state_weather_apparent_temp_band_f",
    "state_weather_daylight_hours",
    "state_weather_is_precip_day",
    "state_weather_is_snow_day",
    "state_weather_is_dark_early",
    "state_weather_is_gloomy_day",
    "state_weather_snow_streak_days",
    "state_weather_gloomy_streak_days",
    "state_weather_dark_early_streak_days",
    "state_prior_meal_archetype_primary",
    "state_prior_cuisine_primary",
    "state_prior_service_form_primary",
    "state_prior_prep_profile",
    "state_prior_principal_protein",
    "state_prior_principal_starch",
    "state_prior_principal_veg",
    "state_prior_principal_fat_source",
    "state_prior_energy_density_style",
    "state_prior_satiety_style",
    "state_prior_comfort_food_score",
    "state_prior_fresh_light_score",
    "state_prior_indulgence_score",
    "state_prior_restaurant_specific_flag",
    "state_prior_generic_standin_flag",
    "state_prior_calories_kcal",
    "state_prior_protein_g",
    "state_prior_carbs_g",
    "state_prior_fat_g",
]

MEAL_SEMANTIC_FEATURES = [
    "time_slot_label",
    "meal_order_in_day",
    "calories_kcal",
    "protein_g",
    "carbs_g",
    "fat_g",
    "fiber_g",
    "sodium_mg",
    "item_count",
    "distinct_entity_count",
    "component_count",
    "main_component_count",
    "protein_anchor_count",
    "starch_base_count",
    "side_component_count_from_roles",
    "beverage_component_count_from_roles",
    "condiment_component_count_from_roles",
    "dessert_component_count_from_roles",
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
]

DAY_CURATED_FEATURES = [
    "true_weight_lb",
    "weight_ema_7d_lb",
    "weight_velocity_7d_lb",
    "weight_ema_14d_lb",
    "weight_velocity_14d_lb",
    "weight_ema_30d_lb",
    "weight_velocity_30d_lb",
    "noom_food_calories_kcal",
    "noom_food_protein_g",
    "noom_food_carbs_g",
    "noom_food_fat_g",
    "noom_food_fiber_g",
    "noom_meal_event_count",
    "calorie_budget_kcal",
    "base_calorie_budget_kcal",
    "manual_calorie_adjustment_kcal",
    "budget_minus_noom_food_calories_kcal",
    "noom_finished_day",
    "samsung_pedometer_steps",
    "samsung_activity_steps",
    "samsung_rest_calorie_kcal",
    "samsung_active_calorie_kcal",
    "samsung_exercise_session_count",
    "samsung_exercise_duration_ms",
    "samsung_exercise_calorie_kcal",
    "samsung_sleep_duration_ms",
    "samsung_sleep_score",
    "meal_event_count",
    "distinct_meal_archetypes",
    "distinct_cuisines",
    "restaurant_specific_meal_count",
    "generic_standin_meal_count",
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
    "meal_calories_kcal_sum",
    "meal_protein_g_sum",
    "meal_carbs_g_sum",
    "meal_fat_g_sum",
    "meal_fiber_g_sum",
    "meal_sodium_mg_sum",
    "meal_component_count_sum",
    "meal_main_component_count_sum",
    "meal_side_component_count_from_roles_sum",
    "meal_beverage_component_count_from_roles_sum",
    "meal_dessert_component_count_from_roles_sum",
    "meal_comfort_food_score_mean",
    "meal_fresh_light_score_mean",
    "meal_indulgence_score_mean",
    "temperature_2m_mean",
    "apparent_temperature_mean",
    "daylight_hours",
    "sunshine_hours",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "cloud_cover_mean",
    "is_precip_day",
    "is_snow_day",
    "is_dark_early",
    "is_short_day",
    "is_hot_day",
    "is_cold_day",
    "is_gloomy_day",
    "snow_streak_days",
    "gloomy_streak_days",
    "dark_early_streak_days",
    "day_of_week",
    "day_of_week_num",
    "is_weekend",
    "month",
    "season",
]

WEEK_CURATED_FEATURES = [
    "n_days",
    "n_weekend_days",
    "days_with_meals",
    "days_with_weight",
    "days_with_activity",
    "days_with_weather",
    "dominant_meal_archetype_week",
    "dominant_cuisine_week",
    "dominant_service_form_week",
    "dominant_prep_profile_week",
    "dominant_protein_week",
    "dominant_starch_week",
    "dominant_energy_density_week",
    "dominant_satiety_style_week",
    "dominant_season",
    "dominant_month_name",
    "gloomy_day_count",
    "rain_day_count",
    "snow_day_count",
    "hot_day_count",
    "cold_day_count",
    "dark_early_day_count",
    "noom_finished_day_count",
    "noom_food_calories_kcal_sum",
    "noom_food_protein_g_sum",
    "noom_food_carbs_g_sum",
    "noom_food_fat_g_sum",
    "noom_meal_event_count_sum",
    "noom_steps_sum",
    "calorie_budget_kcal_sum",
    "base_calorie_budget_kcal_sum",
    "manual_calorie_adjustment_kcal_sum",
    "true_weight_lb_mean",
    "weight_ema_7d_lb_mean",
    "weight_velocity_7d_lb_mean",
    "weight_ema_14d_lb_mean",
    "weight_velocity_14d_lb_mean",
    "weight_ema_30d_lb_mean",
    "weight_velocity_30d_lb_mean",
    "samsung_pedometer_steps_sum",
    "samsung_activity_steps_sum",
    "samsung_rest_calorie_kcal_sum",
    "samsung_active_calorie_kcal_sum",
    "samsung_exercise_session_count_sum",
    "samsung_exercise_duration_ms_sum",
    "samsung_exercise_calorie_kcal_sum",
    "samsung_sleep_duration_ms_sum",
    "samsung_sleep_score_mean",
    "meal_event_count_sum",
    "distinct_meal_archetypes_mean",
    "distinct_cuisines_mean",
    "restaurant_specific_meal_count_sum",
    "generic_standin_meal_count_sum",
    "first_meal_hour_mean",
    "last_meal_hour_mean",
    "eating_window_hours_mean",
    "meal_calories_kcal_sum_sum",
    "meal_protein_g_sum_sum",
    "meal_carbs_g_sum_sum",
    "meal_fat_g_sum_sum",
    "meal_component_count_sum_sum",
    "meal_main_component_count_sum_sum",
    "meal_side_component_count_from_roles_sum_sum",
    "meal_beverage_component_count_from_roles_sum_sum",
    "meal_dessert_component_count_from_roles_sum_sum",
    "meal_comfort_food_score_mean_mean",
    "meal_fresh_light_score_mean_mean",
    "meal_indulgence_score_mean_mean",
    "temperature_2m_mean_mean",
    "apparent_temperature_mean_mean",
    "daylight_hours_mean",
    "sunshine_hours_mean",
    "precipitation_sum_sum",
    "rain_sum_sum",
    "snowfall_sum_sum",
    "cloud_cover_mean_mean",
    "precip_streak_days_max",
    "rain_streak_days_max",
    "snow_streak_days_max",
    "gloomy_streak_days_max",
    "dark_early_streak_days_max",
    "hot_streak_days_max",
    "cold_streak_days_max",
    "weight_delta_lb",
    "weight_ema_7d_delta_lb",
    "weight_ema_14d_delta_lb",
    "budget_minus_logged_food_kcal_week",
    "steps_gap_samsung_minus_noom_week",
    "meal_events_per_day_week",
    "restaurant_meal_fraction_week",
    "generic_standin_fraction_week",
    "gloomy_day_fraction_week",
    "dark_early_fraction_week",
    "snow_day_fraction_week",
    "weight_coverage_fraction_week",
]

WEEKEND_CURATED_FEATURES = [
    "year",
    "month",
    "month_name",
    "season",
    "n_days",
    "friday_present",
    "saturday_present",
    "sunday_present",
    "days_with_meals",
    "days_with_weight",
    "days_with_activity",
    "days_with_weather",
    "dominant_meal_archetype_weekend",
    "dominant_cuisine_weekend",
    "dominant_service_form_weekend",
    "dominant_prep_profile_weekend",
    "dominant_protein_weekend",
    "dominant_starch_weekend",
    "dominant_energy_density_weekend",
    "dominant_satiety_style_weekend",
    "gloomy_day_count",
    "rain_day_count",
    "snow_day_count",
    "hot_day_count",
    "cold_day_count",
    "dark_early_day_count",
    "noom_finished_day_count",
    "noom_food_calories_kcal_sum",
    "noom_food_protein_g_sum",
    "noom_food_carbs_g_sum",
    "noom_food_fat_g_sum",
    "noom_meal_event_count_sum",
    "noom_steps_sum",
    "calorie_budget_kcal_sum",
    "base_calorie_budget_kcal_sum",
    "manual_calorie_adjustment_kcal_sum",
    "true_weight_lb_mean",
    "weight_ema_7d_lb_mean",
    "weight_velocity_7d_lb_mean",
    "weight_ema_14d_lb_mean",
    "weight_velocity_14d_lb_mean",
    "weight_ema_30d_lb_mean",
    "weight_velocity_30d_lb_mean",
    "samsung_pedometer_steps_sum",
    "samsung_activity_steps_sum",
    "samsung_rest_calorie_kcal_sum",
    "samsung_active_calorie_kcal_sum",
    "samsung_exercise_session_count_sum",
    "samsung_exercise_duration_ms_sum",
    "samsung_exercise_calorie_kcal_sum",
    "samsung_sleep_duration_ms_sum",
    "samsung_sleep_score_mean",
    "meal_event_count_sum",
    "distinct_meal_archetypes_mean",
    "distinct_cuisines_mean",
    "restaurant_specific_meal_count_sum",
    "generic_standin_meal_count_sum",
    "first_meal_hour_mean",
    "last_meal_hour_mean",
    "eating_window_hours_mean",
    "meal_calories_kcal_sum_sum",
    "meal_protein_g_sum_sum",
    "meal_carbs_g_sum_sum",
    "meal_fat_g_sum_sum",
    "meal_component_count_sum_sum",
    "meal_main_component_count_sum_sum",
    "meal_side_component_count_from_roles_sum_sum",
    "meal_beverage_component_count_from_roles_sum_sum",
    "meal_dessert_component_count_from_roles_sum_sum",
    "meal_comfort_food_score_mean_mean",
    "meal_fresh_light_score_mean_mean",
    "meal_indulgence_score_mean_mean",
    "temperature_2m_mean_mean",
    "apparent_temperature_mean_mean",
    "daylight_hours_mean",
    "sunshine_hours_mean",
    "precipitation_sum_sum",
    "rain_sum_sum",
    "snowfall_sum_sum",
    "cloud_cover_mean_mean",
    "precip_streak_days_max",
    "rain_streak_days_max",
    "snow_streak_days_max",
    "gloomy_streak_days_max",
    "dark_early_streak_days_max",
    "hot_streak_days_max",
    "cold_streak_days_max",
    "weight_delta_lb",
    "weight_ema_7d_delta_lb",
    "budget_minus_logged_food_kcal_weekend",
    "steps_gap_samsung_minus_noom_weekend",
    "meal_events_per_day_weekend",
    "restaurant_meal_fraction_weekend",
    "generic_standin_fraction_weekend",
    "gloomy_day_fraction_weekend",
    "snow_day_fraction_weekend",
]

# ---------------------------
# Helpers
# ---------------------------


def log(msg: str) -> None:
    print(f"[retrieval-v2] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def read_json_if_exists(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for c in X.columns:
        s = X[c]
        if is_datetime64_any_dtype(s):
            categorical_cols.append(c)
        elif is_bool_dtype(s):
            categorical_cols.append(c)
        elif is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return pre, numeric_cols, categorical_cols


def prepare_X(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].copy()
    for c in X.columns:
        if is_bool_dtype(X[c]):
            X[c] = X[c].astype("object")
        elif is_datetime64_any_dtype(X[c]):
            X[c] = X[c].astype(str)
        elif not is_numeric_dtype(X[c]):
            X[c] = X[c].astype("object")
    return X


def choose_svd_components(n_rows: int, n_features: int, requested: int) -> int:
    upper = min(requested, n_rows - 1, n_features - 1)
    return max(0, upper)


def fit_embedding_pipeline(X: pd.DataFrame, max_components: int):
    preprocessor, numeric_cols, categorical_cols = make_preprocessor(X)
    Xt = preprocessor.fit_transform(X)

    n_rows = X.shape[0]
    n_features = Xt.shape[1]
    n_components = choose_svd_components(n_rows, n_features, max_components)

    svd = None
    embedding = Xt
    if sparse.issparse(Xt) and n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        embedding = svd.fit_transform(Xt)
    else:
        if sparse.issparse(Xt):
            embedding = Xt.toarray()
        elif hasattr(Xt, "toarray"):
            embedding = Xt.toarray()
        else:
            embedding = np.asarray(Xt)

    if sparse.issparse(embedding):
        embedding = embedding.toarray()

    return preprocessor, svd, embedding, numeric_cols, categorical_cols


def build_neighbor_table(
    df: pd.DataFrame,
    id_col: str,
    label_cols: List[str],
    embedding: np.ndarray,
    top_k: int = 10,
) -> Tuple[pd.DataFrame, NearestNeighbors]:
    nn = NearestNeighbors(n_neighbors=min(top_k + 1, len(df)), metric="cosine")
    nn.fit(embedding)
    distances, indices = nn.kneighbors(embedding)

    records = []
    for i in range(len(df)):
        src = df.iloc[i]
        rank = 0
        for d, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            rank += 1
            nbr = df.iloc[j]
            rec = {
                "source_id": src[id_col],
                "neighbor_rank": rank,
                "neighbor_id": nbr[id_col],
                "cosine_distance": float(d),
            }
            for col in label_cols:
                if col in df.columns:
                    rec[f"source_{col}"] = src[col]
                    rec[f"neighbor_{col}"] = nbr[col]
            records.append(rec)
            if rank >= top_k:
                break
    return pd.DataFrame(records), nn


def save_embedding_csv(df: pd.DataFrame, id_col: str, embedding: np.ndarray, out_path: Path) -> None:
    emb_cols = [f"emb_{i:03d}" for i in range(embedding.shape[1])]
    emb_df = pd.DataFrame(embedding, columns=emb_cols)
    emb_df.insert(0, id_col, df[id_col].values)
    emb_df.to_csv(out_path, index=False)


def build_space(project_root: Path, spec: Dict, top_k: int = 10) -> None:
    name = spec["name"]
    path = spec["path"]
    id_col = spec["id_col"]
    label_cols = spec["label_cols"]
    max_components = spec["max_components"]
    feature_cols = spec["feature_cols"]

    out_dir = project_root / "models" / "retrieval_v2" / name
    ensure_dir(out_dir)

    log(f"Building retrieval space: {name}")
    df = read_csv_required(path)

    if id_col not in df.columns:
        raise ValueError(f"{name}: missing id column '{id_col}' in {path}")

    for c in ["date", "decision_time", "week_start", "week_end", "weekend_start", "weekend_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    X = prepare_X(df, feature_cols)
    preprocessor, svd, embedding, numeric_cols, categorical_cols = fit_embedding_pipeline(X, max_components=max_components)
    neighbors_df, nn_model = build_neighbor_table(df, id_col=id_col, label_cols=label_cols, embedding=embedding, top_k=top_k)

    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")
    joblib.dump(nn_model, out_dir / "nearest_neighbors.joblib")
    if svd is not None:
        joblib.dump(svd, out_dir / "svd.joblib")

    save_embedding_csv(df, id_col=id_col, embedding=embedding, out_path=out_dir / "embeddings.csv")
    neighbors_df.to_csv(out_dir / "neighbors_topk.csv", index=False)

    index_cols = [id_col] + [c for c in label_cols if c in df.columns]
    df[index_cols].to_csv(out_dir / "index_rows.csv", index=False)

    manifest = {
        "name": name,
        "source_table": str(path.name),
        "rows": int(len(df)),
        "id_col": id_col,
        "feature_count": int(X.shape[1]),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "embedding_dim": int(embedding.shape[1]),
        "requested_max_components": int(max_components),
        "top_k_neighbors": int(top_k),
        "feature_columns": list(X.columns),
        "label_columns": [c for c in label_cols if c in df.columns],
        "notes": spec["notes"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(f"Wrote retrieval artifacts to: {out_dir}")


def meal_context_spec(project_root: Path) -> Dict:
    # Prefer the trainer's current safe feature set if available.
    manifest = read_json_if_exists(project_root / "models" / "baselines" / "meal" / "run_manifest.json")
    feature_cols = manifest.get("feature_columns") if manifest else None
    if not feature_cols:
        feature_cols = MEAL_CONTEXT_FEATURES

    return {
        "name": "meal_state_context",
        "path": project_root / "training" / "predictive_views" / "meal_prediction_view.csv",
        "id_col": "meal_id",
        "feature_cols": feature_cols,
        "label_cols": [
            "decision_time",
            "time_slot_label",
            "target_meal_text",
            "target_meal_archetype_primary",
            "target_cuisine_primary",
            "target_service_form_primary",
            "target_calories_kcal",
            "y_next_meal_family_coarse",
        ],
        "max_components": 48,
        "notes": "Context-state meal retrieval for similar situations before a meal decision. Good for explanation and context matching, not final semantic meal adjacency.",
    }


def meal_semantic_spec(project_root: Path) -> Dict:
    return {
        "name": "meal_target_semantics",
        "path": project_root / "meal_db" / "final_repaired" / "meal_semantic_features.csv",
        "id_col": "meal_id",
        "feature_cols": MEAL_SEMANTIC_FEATURES,
        "label_cols": [
            "datetime_local_approx",
            "time_slot_label",
            "meal_text",
            "meal_archetype_primary",
            "cuisine_primary",
            "service_form_primary",
            "principal_protein",
            "principal_starch",
            "principal_veg",
            "calories_kcal",
        ],
        "max_components": 24,
        "notes": "Semantic meal retrieval for adjacent meals, novelty, and recommendation candidate generation. Uses meal meaning rather than only decision-state context.",
    }


def day_spec(project_root: Path) -> Dict:
    return {
        "name": "days_curated",
        "path": project_root / "training" / "day_feature_matrix.csv",
        "id_col": "date",
        "feature_cols": DAY_CURATED_FEATURES,
        "label_cols": [
            "day_of_week",
            "season",
            "meal_event_count",
            "dominant_meal_archetype",
            "dominant_cuisine",
            "true_weight_lb",
        ],
        "max_components": 32,
        "notes": "Curated day retrieval focused on behavioral similarity rather than raw export artifacts or duplicated solar fields.",
    }


def week_spec(project_root: Path) -> Dict:
    return {
        "name": "weeks_curated",
        "path": project_root / "training" / "week_summary_matrix.csv",
        "id_col": "week_id",
        "feature_cols": WEEK_CURATED_FEATURES,
        "label_cols": [
            "week_start",
            "week_end",
            "week_label",
            "dominant_meal_archetype_week",
            "dominant_cuisine_week",
            "weight_delta_lb",
            "meal_events_per_day_week",
        ],
        "max_components": 16,
        "notes": "Curated weekly retrieval emphasizing meaningful behavioral, physiological, and environmental summaries with forced compression to reduce noise.",
    }


def weekend_spec(project_root: Path) -> Dict:
    return {
        "name": "weekends_curated",
        "path": project_root / "training" / "weekend_summary_matrix.csv",
        "id_col": "weekend_id",
        "feature_cols": WEEKEND_CURATED_FEATURES,
        "label_cols": [
            "weekend_start",
            "weekend_end",
            "weekend_label",
            "dominant_meal_archetype_weekend",
            "dominant_cuisine_weekend",
            "weight_delta_lb",
            "meal_events_per_day_weekend",
        ],
        "max_components": 16,
        "notes": "Curated weekend retrieval emphasizing drift/recovery structure with stronger dimensionality control than v1.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build refined retrieval baselines for context-state and semantic analogy.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--top-k", type=int, default=10, help="Neighbors to save per row.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    ensure_dir(project_root / "models" / "retrieval_v2")

    specs = [
        meal_context_spec(project_root),
        meal_semantic_spec(project_root),
        day_spec(project_root),
        week_spec(project_root),
        weekend_spec(project_root),
    ]

    for spec in specs:
        build_space(project_root, spec, top_k=args.top_k)

    log("Done.")


if __name__ == "__main__":
    main()
