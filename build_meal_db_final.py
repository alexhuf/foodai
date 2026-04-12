from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def log(msg: str) -> None:
    print(f"[meal-db-final] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def choose_snapshot_or_seed(meal_db: Path, basename: str) -> Path:
    current = meal_db / "current" / basename.replace("_seed", "_current")
    seed = meal_db / "seed" / basename
    if current.exists():
        return current
    if seed.exists():
        return seed
    raise FileNotFoundError(f"Could not find current or seed version for {basename}")


def normalize_bool(series: pd.Series) -> pd.Series:
    true_vals = {"true", "1", "yes", "y", "t"}
    false_vals = {"false", "0", "no", "n", "f", ""}
    def conv(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip().lower()
        if s in true_vals:
            return True
        if s in false_vals:
            return False
        return pd.NA
    return series.map(conv).astype("boolean")


def build_final_tables(project_root: Path, meal_db_dir: str = "meal_db") -> None:
    meal_db = project_root / meal_db_dir
    final_dir = meal_db / "final"
    ensure_dir(final_dir)

    log("Loading reviewed meal DB tables...")
    alias_path = choose_snapshot_or_seed(meal_db, "food_alias_seed.csv")
    entity_path = choose_snapshot_or_seed(meal_db, "canonical_food_entity_seed.csv")
    component_path = choose_snapshot_or_seed(meal_db, "meal_component_seed.csv")
    event_path = choose_snapshot_or_seed(meal_db, "meal_event_seed.csv")
    raw_entry_path = choose_snapshot_or_seed(meal_db, "raw_food_entry_enriched_seed.csv")

    alias = read_csv_required(alias_path)
    entity = read_csv_required(entity_path)
    comp = read_csv_required(component_path)
    event = read_csv_required(event_path)
    raw_entry = read_csv_required(raw_entry_path)

    log("Standardizing reviewed fields...")

    for df in [alias, entity, comp, event, raw_entry]:
        for col in df.columns:
            if "flag" in col or col.startswith("is_"):
                try:
                    df[col] = normalize_bool(df[col])
                except Exception:
                    pass

    # Canonical food entity final
    entity_final = entity.copy()
    if "llm_review_status" in entity_final.columns:
        entity_final["is_reviewed"] = entity_final["llm_review_status"].fillna("").astype(str).str.lower().eq("done")
    else:
        entity_final["is_reviewed"] = False

    entity_final["canonical_display_name"] = entity_final["canonical_display_name"].fillna(entity_final.get("provisional_display_name"))
    entity_final = entity_final.sort_values(
        ["is_reviewed", "entry_count", "meal_count", "canonical_display_name"],
        ascending=[False, False, False, True],
        na_position="last"
    )

    keep_entity_cols = [c for c in [
        "canonical_entity_id",
        "canonical_display_name",
        "provisional_display_name",
        "entity_type",
        "dish_family",
        "cuisine_family",
        "meal_association_classic",
        "service_form",
        "prep_primary",
        "temperature_mode",
        "protein_primary",
        "starch_primary",
        "vegetable_primary",
        "fat_source_primary",
        "restaurant_style",
        "processing_level",
        "restaurant_specific_flag",
        "generic_standin_flag",
        "semantic_confidence",
        "inference_basis",
        "seed_alias_count",
        "seed_aliases",
        "entry_count",
        "meal_count",
        "days_seen",
        "first_seen",
        "last_seen",
        "is_reviewed",
        "llm_review_status",
        "llm_notes",
    ] if c in entity_final.columns]
    entity_final = entity_final[keep_entity_cols]

    # Meal component edge final
    comp_final = comp.copy()
    entity_lookup_cols = [c for c in [
        "canonical_entity_id",
        "canonical_display_name",
        "entity_type",
        "dish_family",
        "cuisine_family",
        "meal_association_classic",
        "service_form",
        "prep_primary",
        "protein_primary",
        "starch_primary",
        "vegetable_primary",
        "fat_source_primary",
        "restaurant_style",
        "processing_level",
        "restaurant_specific_flag",
        "generic_standin_flag",
    ] if c in entity_final.columns]

    comp_final = comp_final.merge(
        entity_final[entity_lookup_cols].drop_duplicates(subset=["canonical_entity_id"]),
        on="canonical_entity_id",
        how="left"
    )

    # Backfill canonical entity id from provisional if needed
    if "provisional_canonical_entity_id" in comp_final.columns:
        comp_final["canonical_entity_id"] = comp_final["canonical_entity_id"].fillna(comp_final["provisional_canonical_entity_id"])

    if "component_role" in comp_final.columns:
        comp_final["component_role_final"] = comp_final["component_role"]
    elif "heuristic_role_hint" in comp_final.columns:
        comp_final["component_role_final"] = comp_final["heuristic_role_hint"]
    else:
        comp_final["component_role_final"] = pd.NA

    if "role_confidence" not in comp_final.columns:
        comp_final["role_confidence"] = pd.NA

    keep_comp_cols = [c for c in [
        "meal_id",
        "food_entry_uuid",
        "date",
        "time_slot",
        "time_slot_label",
        "datetime_local_approx",
        "meal_text",
        "meal_order_in_day",
        "day_meal_count",
        "display_name_raw",
        "normalized_name_clean",
        "normalized_name_core",
        "alias_id",
        "canonical_entity_id",
        "canonical_display_name",
        "entity_type",
        "dish_family",
        "cuisine_family",
        "meal_association_classic",
        "service_form",
        "prep_primary",
        "protein_primary",
        "starch_primary",
        "vegetable_primary",
        "fat_source_primary",
        "restaurant_style",
        "processing_level",
        "restaurant_specific_flag",
        "generic_standin_flag",
        "brand_candidate",
        "restaurant_candidate",
        "genericity_hint",
        "is_unknown_item",
        "is_beverage_hint",
        "is_condiment_hint",
        "is_side_hint",
        "is_dessert_hint",
        "calories_kcal",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "sodium_mg",
        "quantity_proxy",
        "meal_calories_kcal",
        "calorie_share_of_meal",
        "component_rank_by_calories",
        "heuristic_role_hint",
        "component_role_final",
        "role_confidence",
        "llm_review_status",
    ] if c in comp_final.columns]
    comp_final = comp_final[keep_comp_cols]

    # Meal semantic features final
    event_final = event.copy()
    if "llm_review_status" in event_final.columns:
        event_final["is_semantically_reviewed"] = event_final["llm_review_status"].fillna("").astype(str).str.lower().eq("done")
    else:
        event_final["is_semantically_reviewed"] = False

    # derive component summaries
    comp_summary = comp_final.groupby("meal_id", dropna=False).agg(
        component_count=("food_entry_uuid", "count"),
        distinct_entity_count_from_components=("canonical_entity_id", lambda s: s.dropna().nunique()),
        main_component_count=("component_role_final", lambda s: (s.astype(str) == "main").sum()),
        protein_anchor_count=("component_role_final", lambda s: (s.astype(str) == "protein_anchor").sum()),
        starch_base_count=("component_role_final", lambda s: (s.astype(str) == "starch_base").sum()),
        side_component_count_from_roles=("component_role_final", lambda s: (s.astype(str) == "side").sum()),
        beverage_component_count_from_roles=("component_role_final", lambda s: (s.astype(str) == "beverage").sum()),
        condiment_component_count_from_roles=("component_role_final", lambda s: (s.astype(str) == "condiment").sum()),
        dessert_component_count_from_roles=("component_role_final", lambda s: (s.astype(str) == "dessert").sum()),
    ).reset_index()

    event_final = event_final.merge(comp_summary, on="meal_id", how="left")

    keep_event_cols = [c for c in [
        "meal_id",
        "date",
        "datetime_local_approx",
        "time_slot",
        "time_slot_label",
        "meal_order_in_day",
        "day_meal_count",
        "is_first_meal_of_day",
        "is_last_meal_of_day",
        "hours_since_prior_meal",
        "hours_until_next_meal",
        "calories_kcal",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "sodium_mg",
        "item_count",
        "distinct_alias_count",
        "distinct_entity_count",
        "distinct_entity_count_from_components",
        "component_count",
        "main_component_count",
        "protein_anchor_count",
        "starch_base_count",
        "side_component_count",
        "beverage_component_count",
        "condiment_component_count",
        "dessert_component_count",
        "side_component_count_from_roles",
        "beverage_component_count_from_roles",
        "condiment_component_count_from_roles",
        "dessert_component_count_from_roles",
        "meal_text",
        "cooccurrence_signature",
        "prior_meal_id",
        "prior_meal_text",
        "next_meal_id",
        "next_meal_text",
        "cumulative_meal_calories_before_meal",
        "cumulative_meal_calories_after_meal",
        "day_total_meal_calories_kcal",
        "remaining_budget_before_meal_kcal",
        "remaining_budget_after_meal_kcal",
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
        "noom_steps",
        "calorie_budget_kcal",
        "base_calorie_budget_kcal",
        "weight_loss_zone_lower_kcal",
        "weight_loss_zone_upper_kcal",
        "manual_calorie_adjustment_kcal",
        "noom_finished_day",
        "noom_app_open_count",
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
        "novelty_cluster_id",
        "similarity_cluster_id",
        "semantic_confidence",
        "is_semantically_reviewed",
        "llm_review_status",
        "llm_notes",
    ] if c in event_final.columns]
    event_final = event_final[keep_event_cols]

    # Food entry semantic view
    raw_final = raw_entry.copy()

    alias_lookup_cols = [c for c in [
        "alias_id",
        "canonical_entity_id",
        "normalized_name_clean",
        "normalized_name_core_mode",
        "brand_candidate_modes",
        "restaurant_candidate_modes",
        "canonical_entity_id",
        "llm_review_status",
        "llm_notes",
    ] if c in alias.columns]
    raw_final = raw_final.merge(
        alias[alias_lookup_cols].drop_duplicates(subset=["alias_id"]),
        on="alias_id",
        how="left",
        suffixes=("", "_alias")
    )

    raw_final["canonical_entity_id"] = raw_final["canonical_entity_id"].fillna(raw_final.get("provisional_canonical_entity_id"))
    raw_final = raw_final.merge(
        entity_final[[c for c in [
            "canonical_entity_id",
            "canonical_display_name",
            "entity_type",
            "dish_family",
            "cuisine_family",
            "meal_association_classic",
            "service_form",
            "prep_primary",
            "protein_primary",
            "starch_primary",
            "vegetable_primary",
            "fat_source_primary",
            "restaurant_style",
            "processing_level",
            "restaurant_specific_flag",
            "generic_standin_flag",
            "semantic_confidence",
        ] if c in entity_final.columns]].drop_duplicates(subset=["canonical_entity_id"]),
        on="canonical_entity_id",
        how="left"
    )

    comp_role_lookup = comp_final[[c for c in [
        "food_entry_uuid",
        "component_role_final",
        "role_confidence",
    ] if c in comp_final.columns]].drop_duplicates(subset=["food_entry_uuid"])

    raw_final = raw_final.merge(
        comp_role_lookup,
        left_on="uuid",
        right_on="food_entry_uuid",
        how="left"
    )

    keep_raw_cols = [c for c in [
        "uuid",
        "user_id",
        "date",
        "time_slot",
        "time_slot_label",
        "datetime_local_approx",
        "client_time_inserted",
        "food_type",
        "food_category_code",
        "amount",
        "servings",
        "calories_kcal",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "sodium_mg",
        "logged_name",
        "query_text",
        "unit_name",
        "meal_id",
        "meal_text",
        "meal_order_in_day",
        "day_meal_count",
        "display_name_raw",
        "normalized_name_clean",
        "normalized_name_core",
        "alias_id",
        "canonical_entity_id",
        "canonical_display_name",
        "entity_type",
        "dish_family",
        "cuisine_family",
        "meal_association_classic",
        "service_form",
        "prep_primary",
        "protein_primary",
        "starch_primary",
        "vegetable_primary",
        "fat_source_primary",
        "restaurant_style",
        "processing_level",
        "restaurant_specific_flag",
        "generic_standin_flag",
        "semantic_confidence",
        "brand_candidate",
        "restaurant_candidate",
        "genericity_hint",
        "is_unknown_item",
        "is_beverage_hint",
        "is_condiment_hint",
        "is_side_hint",
        "is_dessert_hint",
        "component_role_final",
        "role_confidence",
    ] if c in raw_final.columns]
    raw_final = raw_final[keep_raw_cols]

    # Coverage / manifest
    alias_done = 0
    if "llm_review_status" in alias.columns:
        alias_done = int(alias["llm_review_status"].fillna("").astype(str).str.lower().eq("done").sum())
    manifest: Dict[str, object] = {
        "tables": {
            "canonical_food_entity": {
                "rows": int(len(entity_final)),
                "reviewed_rows": int(entity_final["is_reviewed"].fillna(False).sum()) if "is_reviewed" in entity_final.columns else None,
            },
            "meal_component_edge": {
                "rows": int(len(comp_final)),
                "reviewed_role_rows": int(comp_final["component_role_final"].notna().sum()) if "component_role_final" in comp_final.columns else None,
            },
            "meal_semantic_features": {
                "rows": int(len(event_final)),
                "reviewed_meal_rows": int(event_final["is_semantically_reviewed"].fillna(False).sum()) if "is_semantically_reviewed" in event_final.columns else None,
            },
            "food_entry_semantic_view": {
                "rows": int(len(raw_final)),
                "rows_with_canonical_entity": int(raw_final["canonical_entity_id"].notna().sum()) if "canonical_entity_id" in raw_final.columns else None,
            },
        },
        "alias_review_progress": {
            "total_aliases": int(len(alias)),
            "aliases_done": alias_done,
            "aliases_pending": int(len(alias) - alias_done),
        },
    }

    log("Writing final meal DB tables...")
    entity_final.to_csv(final_dir / "canonical_food_entity.csv", index=False)
    comp_final.to_csv(final_dir / "meal_component_edge.csv", index=False)
    event_final.to_csv(final_dir / "meal_semantic_features.csv", index=False)
    raw_final.to_csv(final_dir / "food_entry_semantic_view.csv", index=False)
    (final_dir / "meal_db_final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log("Done.")
    log(f"Wrote final meal DB tables to: {final_dir}")
    log(f"Alias review progress: {alias_done}/{len(alias)} complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize reviewed meal DB seed tables into stable final outputs.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--meal-db-dir", default="meal_db", help="Meal DB directory under project root.")
    args = parser.parse_args()
    build_final_tables(Path(args.project_root).expanduser().resolve(), args.meal_db_dir)


if __name__ == "__main__":
    main()
