
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ALIAS_FILE = "food_alias_batch_001_llm_enriched.csv"
ENTITY_FILE = "canonical_food_entity_batch_001_llm.csv"
COMP_FILE = "food_alias_batch_001_components_llm.csv"
MEAL_FILE = "food_alias_batch_001_meal_examples_llm.csv"


def log(msg: str) -> None:
    print(f"[meal-db-repair] {msg}")


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


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")].copy()


def normalize_bool_series(series: pd.Series) -> pd.Series:
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


def first_nonnull(values: pd.Series):
    for v in values:
        if pd.notna(v) and str(v).strip() != "":
            return v
    return pd.NA


def weighted_mode(values: pd.Series, weights: pd.Series, exclude: Optional[set] = None):
    temp = pd.DataFrame({"value": values, "weight": weights})
    temp = temp[temp["value"].notna()].copy()
    if exclude:
        temp = temp[~temp["value"].astype(str).str.lower().isin({str(x).lower() for x in exclude})]
    if temp.empty:
        return pd.NA
    grouped = temp.groupby("value", dropna=False)["weight"].sum().sort_values(ascending=False)
    if grouped.empty:
        return pd.NA
    return grouped.index[0]


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def collect_review_tables(reviews_root: Path) -> Dict[str, pd.DataFrame]:
    review_dirs = sorted([p for p in reviews_root.iterdir() if p.is_dir()]) if reviews_root.exists() else []
    alias_parts: List[pd.DataFrame] = []
    entity_parts: List[pd.DataFrame] = []
    comp_parts: List[pd.DataFrame] = []
    meal_parts: List[pd.DataFrame] = []

    for d in review_dirs:
        alias_path = d / ALIAS_FILE
        entity_path = d / ENTITY_FILE
        comp_path = d / COMP_FILE
        meal_path = d / MEAL_FILE
        if not (alias_path.exists() and entity_path.exists() and comp_path.exists() and meal_path.exists()):
            continue

        alias_df = dedupe_columns(read_csv_required(alias_path))
        entity_df = dedupe_columns(read_csv_required(entity_path))
        comp_df = dedupe_columns(read_csv_required(comp_path))
        meal_df = dedupe_columns(read_csv_required(meal_path))

        alias_df["review_batch_label"] = d.name
        entity_df["review_batch_label"] = d.name
        comp_df["review_batch_label"] = d.name
        meal_df["review_batch_label"] = d.name

        alias_parts.append(alias_df)
        entity_parts.append(entity_df)
        comp_parts.append(comp_df)
        meal_parts.append(meal_df)

    if not alias_parts:
        raise FileNotFoundError(f"No complete review folders found under {reviews_root}")

    alias_reviews = pd.concat(alias_parts, ignore_index=True)
    entity_reviews = pd.concat(entity_parts, ignore_index=True)
    comp_reviews = pd.concat(comp_parts, ignore_index=True)
    meal_reviews = pd.concat(meal_parts, ignore_index=True)

    # Normalize booleans
    for df in [alias_reviews, entity_reviews, comp_reviews, meal_reviews]:
        for col in df.columns:
            if "flag" in col or col.startswith("is_"):
                try:
                    df[col] = normalize_bool_series(df[col])
                except Exception:
                    pass

    # Deduplicate with "last write wins" by review order
    alias_reviews = alias_reviews.drop_duplicates(subset=["alias_id"], keep="last")
    entity_reviews = entity_reviews.drop_duplicates(subset=["canonical_entity_id"], keep="last")
    comp_reviews = comp_reviews.drop_duplicates(subset=["food_entry_uuid"], keep="last")
    meal_reviews = meal_reviews.drop_duplicates(subset=["meal_id"], keep="last")

    return {
        "alias": alias_reviews,
        "entity": entity_reviews,
        "component": comp_reviews,
        "meal": meal_reviews,
    }


def derive_meal_archetype(service_form, dish_family, component_roles, calories, item_count):
    sf = "" if pd.isna(service_form) else str(service_form).lower()
    dfam = "" if pd.isna(dish_family) else str(dish_family).lower()
    roles = set(str(x).lower() for x in component_roles if pd.notna(x))

    if sf in {"beverage", "drink", "smoothie", "coffee"}:
        return "beverage_only"
    if sf in {"burger", "sandwich", "sub", "wrap", "hot_dog"}:
        return "sandwich_meal"
    if sf in {"pizza", "slice"}:
        return "pizza_meal"
    if sf in {"taco", "quesadilla", "burrito", "bowl"} or "tex_mex" in dfam or "fajita" in dfam:
        return "tex_mex_meal"
    if "sushi" in sf or "sushi" in dfam or "roll" in sf:
        return "sushi_meal"
    if sf in {"ramen", "soup", "stew"}:
        return "soup_meal"
    if sf in {"salad"}:
        return "salad_meal"
    if sf in {"pasta", "noodle_plate"} or "pasta" in dfam or "mac" in dfam:
        return "pasta_meal"
    if "dessert" in roles and len(roles) <= 2 and (pd.isna(calories) or calories < 500):
        return "dessert_snack"
    if roles == {"beverage"}:
        return "beverage_only"
    if "main" in roles:
        return "mixed_plate"
    if "protein_anchor" in roles and "starch_base" in roles and item_count <= 4:
        return "assembled_meal"
    if "side" in roles and item_count <= 3 and (pd.isna(calories) or calories < 450):
        return "snack_plate"
    return "mixed_plate"


def derive_scores(row: pd.Series) -> Dict[str, object]:
    prep = str(row.get("prep_profile", "")).lower()
    arche = str(row.get("meal_archetype_primary", "")).lower()
    calories = pd.to_numeric(pd.Series([row.get("calories_kcal")]), errors="coerce").iloc[0]
    protein = pd.to_numeric(pd.Series([row.get("protein_g")]), errors="coerce").iloc[0]
    fiber = pd.to_numeric(pd.Series([row.get("fiber_g")]), errors="coerce").iloc[0]
    item_count = pd.to_numeric(pd.Series([row.get("item_count")]), errors="coerce").iloc[0]
    dessert_ct = pd.to_numeric(pd.Series([row.get("dessert_component_count_from_roles", row.get("dessert_component_count", np.nan))]), errors="coerce").iloc[0]
    bev_ct = pd.to_numeric(pd.Series([row.get("beverage_component_count_from_roles", row.get("beverage_component_count", np.nan))]), errors="coerce").iloc[0]

    comfort = 0.45
    fresh = 0.40
    indulgence = 0.40

    if any(k in prep for k in ["fried", "crispy", "breaded", "packaged"]):
        indulgence += 0.2
        comfort += 0.1
        fresh -= 0.15
    if any(k in arche for k in ["salad", "bowl"]):
        fresh += 0.2
    if any(k in arche for k in ["pizza", "burger", "pasta", "dessert"]):
        comfort += 0.2
        indulgence += 0.2
        fresh -= 0.1
    if pd.notna(dessert_ct) and dessert_ct > 0:
        indulgence += 0.15
    if pd.notna(bev_ct) and bev_ct > 0 and (pd.isna(item_count) or item_count == 1):
        fresh += 0.05

    comfort = float(np.clip(comfort, 0, 1))
    fresh = float(np.clip(fresh, 0, 1))
    indulgence = float(np.clip(indulgence, 0, 1))

    energy_density = pd.NA
    if pd.notna(calories):
        if calories >= 900:
            energy_density = "high"
        elif calories >= 500:
            energy_density = "moderate"
        else:
            energy_density = "light"

    satiety = pd.NA
    if pd.notna(protein) or pd.notna(fiber):
        p = 0 if pd.isna(protein) else protein
        f = 0 if pd.isna(fiber) else fiber
        if p >= 30 or f >= 8:
            satiety = "high"
        elif p >= 15 or f >= 4:
            satiety = "moderate"
        else:
            satiety = "light"

    coherence = 0.5
    roles_count = sum([
        1 if pd.notna(row.get("main_component_count")) and row.get("main_component_count") > 0 else 0,
        1 if pd.notna(row.get("protein_anchor_count")) and row.get("protein_anchor_count") > 0 else 0,
        1 if pd.notna(row.get("starch_base_count")) and row.get("starch_base_count") > 0 else 0,
        1 if pd.notna(row.get("side_component_count_from_roles")) and row.get("side_component_count_from_roles") > 0 else 0,
        1 if pd.notna(row.get("dessert_component_count_from_roles")) and row.get("dessert_component_count_from_roles") > 0 else 0,
    ])
    if pd.notna(item_count):
        if item_count <= 2:
            coherence = 0.8
        elif item_count <= 4:
            coherence = 0.65
        else:
            coherence = 0.5
    if roles_count >= 4:
        coherence -= 0.1
    coherence = float(np.clip(coherence, 0, 1))

    return {
        "comfort_food_score": comfort,
        "fresh_light_score": fresh,
        "indulgence_score": indulgence,
        "energy_density_style": energy_density,
        "satiety_style": satiety,
        "coherence_score": coherence,
    }


def build_repaired_final(project_root: Path, meal_db_dir: str = "meal_db") -> None:
    meal_db = project_root / meal_db_dir
    reviews_root = meal_db / "reviews"
    final_dir = meal_db / "final_repaired"
    ensure_dir(final_dir)

    log("Loading reviewed batch files...")
    reviews = collect_review_tables(reviews_root)
    alias_reviews = reviews["alias"]
    entity_reviews = reviews["entity"]
    comp_reviews = reviews["component"]
    meal_reviews = reviews["meal"]

    log("Loading current/seed base tables...")
    alias_base = dedupe_columns(read_csv_required(choose_snapshot_or_seed(meal_db, "food_alias_seed.csv")))
    entity_base = dedupe_columns(read_csv_required(choose_snapshot_or_seed(meal_db, "canonical_food_entity_seed.csv")))
    comp_base = dedupe_columns(read_csv_required(choose_snapshot_or_seed(meal_db, "meal_component_seed.csv")))
    event_base = dedupe_columns(read_csv_required(choose_snapshot_or_seed(meal_db, "meal_event_seed.csv")))
    raw_entry = dedupe_columns(read_csv_required(choose_snapshot_or_seed(meal_db, "raw_food_entry_enriched_seed.csv")))

    # ---------- reviewed alias master ----------
    alias_master = alias_base.merge(
        alias_reviews.drop(columns=[c for c in ["review_batch_label"] if c in alias_reviews.columns]),
        on="alias_id",
        how="left",
        suffixes=("", "_review"),
    )
    # Fill reviewed semantic columns back into master
    semantic_alias_cols = [
        "canonical_entity_id", "canonical_display_name", "entity_type", "dish_family", "cuisine_family",
        "meal_association_classic", "service_form", "prep_primary", "temperature_mode", "protein_primary",
        "starch_primary", "vegetable_primary", "fat_source_primary", "restaurant_style", "processing_level",
        "restaurant_specific_flag", "generic_standin_flag", "semantic_confidence", "inference_basis",
        "llm_review_status", "llm_notes"
    ]
    for col in semantic_alias_cols:
        review_col = f"{col}_review"
        if review_col in alias_master.columns:
            alias_master[col] = alias_master[review_col].combine_first(alias_master.get(col))
    alias_master = dedupe_columns(alias_master)

    # ---------- component master ----------
    comp_master = comp_base.merge(
        comp_reviews.drop(columns=[c for c in ["review_batch_label"] if c in comp_reviews.columns]),
        on="food_entry_uuid",
        how="left",
        suffixes=("", "_review"),
    )
    if "canonical_entity_id_review" in comp_master.columns:
        comp_master["canonical_entity_id"] = comp_master["canonical_entity_id_review"].combine_first(
            comp_master.get("canonical_entity_id")
        )
    if "component_role_suggested" in comp_master.columns:
        comp_master["component_role_final"] = comp_master["component_role_suggested"].combine_first(
            comp_master.get("component_role")
        ).combine_first(comp_master.get("heuristic_role_hint"))
    elif "component_role_review" in comp_master.columns:
        comp_master["component_role_final"] = comp_master["component_role_review"].combine_first(
            comp_master.get("component_role")
        ).combine_first(comp_master.get("heuristic_role_hint"))
    else:
        comp_master["component_role_final"] = comp_master.get("component_role", comp_master.get("heuristic_role_hint"))
    if "role_confidence_suggested" in comp_master.columns:
        comp_master["role_confidence"] = comp_master["role_confidence_suggested"].combine_first(comp_master.get("role_confidence"))
    comp_master["llm_review_status"] = comp_master.get("llm_review_status_review", comp_master.get("llm_review_status"))

    # Attach alias semantics to components
    alias_sem_cols = ["alias_id"] + [c for c in semantic_alias_cols if c in alias_master.columns]
    comp_master = comp_master.merge(
        dedupe_columns(alias_master[alias_sem_cols]).drop_duplicates(subset=["alias_id"]),
        on="alias_id",
        how="left",
        suffixes=("", "_alias"),
    )
    if "canonical_entity_id_alias" in comp_master.columns:
        comp_master["canonical_entity_id"] = comp_master["canonical_entity_id"].combine_first(comp_master["canonical_entity_id_alias"])
    comp_master = dedupe_columns(comp_master)

    # ---------- canonical entity rebuilt from reviewed aliases/components ----------
    log("Rebuilding canonical food entities from reviewed alias/component ground truth...")
    # exact stats from component membership
    comp_master["date"] = parse_date(comp_master["date"])
    entity_stats = comp_master.groupby("canonical_entity_id", dropna=False).agg(
        entry_count=("food_entry_uuid", "count"),
        meal_count=("meal_id", pd.Series.nunique),
        days_seen=("date", pd.Series.nunique),
        first_seen=("date", "min"),
        last_seen=("date", "max"),
        seed_alias_count=("alias_id", pd.Series.nunique),
    ).reset_index()

    seed_aliases = (
        comp_master.groupby("canonical_entity_id", dropna=False)["normalized_name_clean"]
        .agg(lambda s: json.dumps(sorted(pd.Series(s.dropna().astype(str).unique()).tolist())[:50]))
        .reset_index()
        .rename(columns={"normalized_name_clean": "seed_aliases"})
    )

    # semantic fields from entity reviews if present, else weighted alias mode
    entity_sem_from_reviews = entity_reviews.copy()
    if "review_batch_label" in entity_sem_from_reviews.columns:
        entity_sem_from_reviews = entity_sem_from_reviews.drop(columns=["review_batch_label"])
    entity_sem_from_reviews = entity_sem_from_reviews.drop_duplicates(subset=["canonical_entity_id"], keep="last")

    # weighted semantic fallback from alias/component rows
    comp_master["weight_for_semantics"] = pd.to_numeric(comp_master.get("calories_kcal"), errors="coerce").fillna(1).clip(lower=1)
    group = comp_master.groupby("canonical_entity_id", dropna=False)

    fallback_rows = []
    for canon_id, g in group:
        if pd.isna(canon_id):
            continue
        row = {"canonical_entity_id": canon_id}
        row["canonical_display_name"] = weighted_mode(g.get("canonical_display_name_alias", g.get("normalized_name_clean")), g["weight_for_semantics"])
        row["entity_type"] = weighted_mode(g.get("entity_type"), g["weight_for_semantics"])
        row["dish_family"] = weighted_mode(g.get("dish_family"), g["weight_for_semantics"])
        row["cuisine_family"] = weighted_mode(g.get("cuisine_family"), g["weight_for_semantics"], exclude={"general", "none", ""})
        if pd.isna(row["cuisine_family"]):
            row["cuisine_family"] = weighted_mode(g.get("cuisine_family"), g["weight_for_semantics"])
        row["meal_association_classic"] = weighted_mode(g.get("meal_association_classic"), g["weight_for_semantics"])
        row["service_form"] = weighted_mode(g.get("service_form"), g["weight_for_semantics"])
        row["prep_primary"] = weighted_mode(g.get("prep_primary"), g["weight_for_semantics"])
        row["temperature_mode"] = weighted_mode(g.get("temperature_mode"), g["weight_for_semantics"], exclude={"none", ""})
        row["protein_primary"] = weighted_mode(g.get("protein_primary"), g["weight_for_semantics"], exclude={"none", ""})
        row["starch_primary"] = weighted_mode(g.get("starch_primary"), g["weight_for_semantics"], exclude={"none", ""})
        row["vegetable_primary"] = weighted_mode(g.get("vegetable_primary"), g["weight_for_semantics"], exclude={"none", ""})
        row["fat_source_primary"] = weighted_mode(g.get("fat_source_primary"), g["weight_for_semantics"], exclude={"none", ""})
        row["restaurant_style"] = weighted_mode(g.get("restaurant_style"), g["weight_for_semantics"])
        row["processing_level"] = weighted_mode(g.get("processing_level"), g["weight_for_semantics"])
        row["restaurant_specific_flag"] = bool(pd.Series(g.get("restaurant_specific_flag", pd.Series(dtype="boolean"))).fillna(False).any())
        row["generic_standin_flag"] = bool(pd.Series(g.get("generic_standin_flag", pd.Series(dtype="boolean"))).fillna(False).any())
        sc = pd.to_numeric(g.get("semantic_confidence"), errors="coerce")
        row["semantic_confidence"] = float(sc.mean()) if sc.notna().any() else pd.NA
        row["inference_basis"] = "reviewed_alias_aggregate"
        row["llm_review_status"] = "done"
        row["llm_notes"] = "Rebuilt from reviewed alias/component semantics."
        fallback_rows.append(row)
    entity_fallback = pd.DataFrame(fallback_rows)

    entity_final = entity_stats.merge(seed_aliases, on="canonical_entity_id", how="left")
    entity_final = entity_final.merge(entity_fallback, on="canonical_entity_id", how="left")
    entity_final = entity_final.merge(
        entity_sem_from_reviews[
            [c for c in entity_sem_from_reviews.columns if c in [
                "canonical_entity_id", "canonical_display_name", "provisional_display_name", "entity_type", "dish_family",
                "cuisine_family", "meal_association_classic", "service_form", "prep_primary", "temperature_mode",
                "protein_primary", "starch_primary", "vegetable_primary", "fat_source_primary", "restaurant_style",
                "processing_level", "restaurant_specific_flag", "generic_standin_flag", "semantic_confidence",
                "inference_basis", "llm_review_status", "llm_notes"
            ]]
        ],
        on="canonical_entity_id",
        how="left",
        suffixes=("", "_entityreview"),
    )

    # prefer explicit entity review over aggregate
    for col in [
        "canonical_display_name", "provisional_display_name", "entity_type", "dish_family", "cuisine_family",
        "meal_association_classic", "service_form", "prep_primary", "temperature_mode", "protein_primary",
        "starch_primary", "vegetable_primary", "fat_source_primary", "restaurant_style", "processing_level",
        "restaurant_specific_flag", "generic_standin_flag", "semantic_confidence", "inference_basis",
        "llm_review_status", "llm_notes"
    ]:
        review_col = f"{col}_entityreview"
        if review_col in entity_final.columns:
            entity_final[col] = entity_final[review_col].combine_first(entity_final.get(col))

    entity_final["is_reviewed"] = True
    entity_final = dedupe_columns(entity_final)

    # ---------- enrich components with rebuilt entity semantics ----------
    entity_join_cols = [
        "canonical_entity_id", "canonical_display_name", "entity_type", "dish_family", "cuisine_family",
        "meal_association_classic", "service_form", "prep_primary", "temperature_mode", "protein_primary",
        "starch_primary", "vegetable_primary", "fat_source_primary", "restaurant_style", "processing_level",
        "restaurant_specific_flag", "generic_standin_flag", "semantic_confidence"
    ]
    comp_final = comp_master.merge(
        entity_final[[c for c in entity_join_cols if c in entity_final.columns]].drop_duplicates(subset=["canonical_entity_id"]),
        on="canonical_entity_id",
        how="left",
        suffixes=("", "_entity"),
    )
    comp_final = dedupe_columns(comp_final)

    # ---------- meal semantics rebuilt ----------
    log("Recomputing meal semantics from reviewed components + reviewed meal examples...")
    event_final = event_base.copy()
    meal_explicit = meal_reviews.drop(columns=[c for c in ["review_batch_label"] if c in meal_reviews.columns]).copy()
    # preserve explicit reviewed semantics
    explicit_cols = [
        "meal_archetype_primary", "meal_archetype_secondary", "cuisine_primary", "cuisine_secondary",
        "service_form_primary", "prep_profile", "principal_protein", "principal_starch", "principal_veg",
        "principal_fat_source", "comfort_food_score", "fresh_light_score", "indulgence_score",
        "energy_density_style", "satiety_style", "coherence_score", "restaurant_specific_flag",
        "generic_standin_flag", "novelty_cluster_id", "similarity_cluster_id", "semantic_confidence",
        "llm_review_status", "llm_notes"
    ]
    meal_explicit = meal_explicit[["meal_id"] + [c for c in explicit_cols if c in meal_explicit.columns]].drop_duplicates(subset=["meal_id"])
    event_final = event_final.merge(meal_explicit, on="meal_id", how="left", suffixes=("", "_explicit"))
    event_final = dedupe_columns(event_final)

    # derive fallback from components for all meals
    comp_final["weight_for_semantics"] = pd.to_numeric(comp_final.get("calorie_share_of_meal"), errors="coerce").fillna(0)
    comp_final.loc[comp_final["weight_for_semantics"] <= 0, "weight_for_semantics"] = pd.to_numeric(comp_final.get("calories_kcal"), errors="coerce").fillna(1).clip(lower=1)

    meal_rows = []
    for meal_id, g in comp_final.groupby("meal_id", dropna=False):
        if pd.isna(meal_id):
            continue
        roles = g.get("component_role_final", pd.Series(dtype=object))
        service_form = weighted_mode(g.get("service_form"), g["weight_for_semantics"], exclude={"portion", "ingredient", "", "none"})
        if pd.isna(service_form):
            service_form = weighted_mode(g.get("service_form"), g["weight_for_semantics"])
        dish_family = weighted_mode(g.get("dish_family"), g["weight_for_semantics"])
        cuisine = weighted_mode(g.get("cuisine_family"), g["weight_for_semantics"], exclude={"general", "none", ""})
        if pd.isna(cuisine):
            cuisine = weighted_mode(g.get("cuisine_family"), g["weight_for_semantics"])
        prep = weighted_mode(g.get("prep_primary"), g["weight_for_semantics"])
        pprot = weighted_mode(g.get("protein_primary"), g["weight_for_semantics"], exclude={"none", ""})
        pstarch = weighted_mode(g.get("starch_primary"), g["weight_for_semantics"], exclude={"none", ""})
        pveg = weighted_mode(g.get("vegetable_primary"), g["weight_for_semantics"], exclude={"none", ""})
        pfat = weighted_mode(g.get("fat_source_primary"), g["weight_for_semantics"], exclude={"none", ""})
        calories = pd.to_numeric(g.get("meal_calories_kcal"), errors="coerce")
        meal_cal = float(calories.dropna().iloc[0]) if calories.notna().any() else float(pd.to_numeric(g.get("calories_kcal"), errors="coerce").sum())
        item_count = int(g["food_entry_uuid"].nunique())

        row = {
            "meal_id": meal_id,
            "meal_archetype_primary_fallback": derive_meal_archetype(service_form, dish_family, roles.tolist(), meal_cal, item_count),
            "meal_archetype_secondary_fallback": pd.NA,
            "cuisine_primary_fallback": cuisine,
            "cuisine_secondary_fallback": pd.NA,
            "service_form_primary_fallback": service_form,
            "prep_profile_fallback": prep,
            "principal_protein_fallback": pprot,
            "principal_starch_fallback": pstarch,
            "principal_veg_fallback": pveg,
            "principal_fat_source_fallback": pfat,
            "restaurant_specific_flag_fallback": bool(pd.Series(g.get("restaurant_specific_flag", pd.Series(dtype="boolean"))).fillna(False).any()),
            "generic_standin_flag_fallback": bool(pd.Series(g.get("generic_standin_flag", pd.Series(dtype="boolean"))).fillna(False).any()),
            "semantic_confidence_fallback": float(pd.to_numeric(g.get("semantic_confidence"), errors="coerce").mean()) if pd.to_numeric(g.get("semantic_confidence"), errors="coerce").notna().any() else 0.75,
        }
        score_row = pd.Series({
            "prep_profile": row["prep_profile_fallback"],
            "meal_archetype_primary": row["meal_archetype_primary_fallback"],
            "calories_kcal": meal_cal,
            "protein_g": pd.to_numeric(g.get("protein_g"), errors="coerce").sum(),
            "fiber_g": pd.to_numeric(g.get("fiber_g"), errors="coerce").sum(),
            "item_count": item_count,
            "dessert_component_count_from_roles": (roles.astype(str).str.lower() == "dessert").sum(),
            "beverage_component_count_from_roles": (roles.astype(str).str.lower() == "beverage").sum(),
            "main_component_count": (roles.astype(str).str.lower() == "main").sum(),
            "protein_anchor_count": (roles.astype(str).str.lower() == "protein_anchor").sum(),
            "starch_base_count": (roles.astype(str).str.lower() == "starch_base").sum(),
            "side_component_count_from_roles": (roles.astype(str).str.lower() == "side").sum(),
        })
        row.update(derive_scores(score_row))
        meal_rows.append(row)

    meal_fallback = pd.DataFrame(meal_rows)
    event_final = event_final.merge(meal_fallback, on="meal_id", how="left")
    event_final = dedupe_columns(event_final)

    # fill from explicit reviewed semantics first, then fallback
    for base_col in [
        "meal_archetype_primary", "meal_archetype_secondary", "cuisine_primary", "cuisine_secondary",
        "service_form_primary", "prep_profile", "principal_protein", "principal_starch",
        "principal_veg", "principal_fat_source", "comfort_food_score", "fresh_light_score",
        "indulgence_score", "energy_density_style", "satiety_style", "coherence_score",
        "restaurant_specific_flag", "generic_standin_flag", "semantic_confidence"
    ]:
        fallback_col = f"{base_col}_fallback"
        explicit_col = f"{base_col}_explicit"
        if explicit_col in event_final.columns:
            event_final[base_col] = event_final[explicit_col].combine_first(event_final.get(base_col))
        if fallback_col in event_final.columns:
            event_final[base_col] = event_final.get(base_col).combine_first(event_final[fallback_col])

    # mark all meals as semantically reviewed/materialized
    event_final["is_semantically_reviewed"] = True
    event_final["semantic_source"] = np.where(
        event_final[[c for c in ["meal_archetype_primary_explicit", "cuisine_primary_explicit", "service_form_primary_explicit"] if c in event_final.columns]].notna().any(axis=1),
        "explicit_batch_review",
        "component_derived"
    )
    event_final["llm_review_status"] = event_final.get("llm_review_status_explicit", event_final.get("llm_review_status")).fillna("done")
    event_final["llm_notes"] = event_final.get("llm_notes_explicit", event_final.get("llm_notes"))
    event_final["novelty_cluster_id"] = event_final.get("novelty_cluster_id", pd.Series([pd.NA] * len(event_final)))
    event_final["similarity_cluster_id"] = event_final.get("similarity_cluster_id", pd.Series([pd.NA] * len(event_final)))

    # component summaries from roles
    role_summary = comp_final.groupby("meal_id").agg(
        distinct_entity_count_from_components=("canonical_entity_id", pd.Series.nunique),
        component_count=("food_entry_uuid", "count"),
        main_component_count=("component_role_final", lambda s: (s.astype(str).str.lower() == "main").sum()),
        protein_anchor_count=("component_role_final", lambda s: (s.astype(str).str.lower() == "protein_anchor").sum()),
        starch_base_count=("component_role_final", lambda s: (s.astype(str).str.lower() == "starch_base").sum()),
        side_component_count_from_roles=("component_role_final", lambda s: (s.astype(str).str.lower() == "side").sum()),
        beverage_component_count_from_roles=("component_role_final", lambda s: (s.astype(str).str.lower() == "beverage").sum()),
        condiment_component_count_from_roles=("component_role_final", lambda s: (s.astype(str).str.lower() == "condiment").sum()),
        dessert_component_count_from_roles=("component_role_final", lambda s: (s.astype(str).str.lower() == "dessert").sum()),
    ).reset_index()
    event_final = event_final.drop(columns=[c for c in role_summary.columns if c != "meal_id" and c in event_final.columns], errors="ignore")
    event_final = event_final.merge(role_summary, on="meal_id", how="left")

    event_cols_keep = [c for c in [
        "meal_id","date","datetime_local_approx","time_slot","time_slot_label","meal_order_in_day","day_meal_count",
        "is_first_meal_of_day","is_last_meal_of_day","hours_since_prior_meal","hours_until_next_meal",
        "calories_kcal","protein_g","carbs_g","fat_g","fiber_g","sodium_mg","item_count","distinct_alias_count",
        "distinct_entity_count","distinct_entity_count_from_components","component_count","main_component_count",
        "protein_anchor_count","starch_base_count","side_component_count","beverage_component_count",
        "condiment_component_count","dessert_component_count","side_component_count_from_roles",
        "beverage_component_count_from_roles","condiment_component_count_from_roles","dessert_component_count_from_roles",
        "meal_text","cooccurrence_signature","prior_meal_id","prior_meal_text","next_meal_id","next_meal_text",
        "cumulative_meal_calories_before_meal","cumulative_meal_calories_after_meal","day_total_meal_calories_kcal",
        "remaining_budget_before_meal_kcal","remaining_budget_after_meal_kcal","true_weight_lb","weight_ema_7d_lb",
        "weight_velocity_7d_lb","weight_ema_14d_lb","weight_velocity_14d_lb","weight_ema_30d_lb",
        "weight_velocity_30d_lb","samsung_pedometer_steps","samsung_activity_steps","samsung_rest_calorie_kcal",
        "samsung_active_calorie_kcal","noom_steps","calorie_budget_kcal","base_calorie_budget_kcal",
        "weight_loss_zone_lower_kcal","weight_loss_zone_upper_kcal","manual_calorie_adjustment_kcal",
        "noom_finished_day","noom_app_open_count","meal_archetype_primary","meal_archetype_secondary",
        "cuisine_primary","cuisine_secondary","service_form_primary","prep_profile","principal_protein",
        "principal_starch","principal_veg","principal_fat_source","comfort_food_score","fresh_light_score",
        "indulgence_score","energy_density_style","satiety_style","coherence_score","restaurant_specific_flag",
        "generic_standin_flag","novelty_cluster_id","similarity_cluster_id","semantic_confidence",
        "semantic_source","is_semantically_reviewed","llm_review_status","llm_notes"
    ] if c in event_final.columns]
    event_final = dedupe_columns(event_final[event_cols_keep])

    # ---------- food entry semantic view ----------
    raw_final = raw_entry.merge(
        comp_final[["food_entry_uuid","canonical_entity_id","component_role_final","role_confidence"]].drop_duplicates(subset=["food_entry_uuid"]),
        left_on="uuid", right_on="food_entry_uuid", how="left"
    )
    raw_final = dedupe_columns(raw_final)
    raw_final = raw_final.merge(
        entity_final[[c for c in [
            "canonical_entity_id","canonical_display_name","entity_type","dish_family","cuisine_family",
            "meal_association_classic","service_form","prep_primary","temperature_mode","protein_primary",
            "starch_primary","vegetable_primary","fat_source_primary","restaurant_style","processing_level",
            "restaurant_specific_flag","generic_standin_flag","semantic_confidence"
        ] if c in entity_final.columns]].drop_duplicates(subset=["canonical_entity_id"]),
        on="canonical_entity_id", how="left"
    )
    raw_final = dedupe_columns(raw_final)

    raw_cols_keep = [c for c in [
        "uuid","user_id","date","time_slot","time_slot_label","datetime_local_approx","client_time_inserted",
        "food_type","food_category_code","amount","servings","calories_kcal","protein_g","carbs_g","fat_g",
        "fiber_g","sodium_mg","logged_name","query_text","unit_name","meal_id","meal_text","meal_order_in_day",
        "day_meal_count","display_name_raw","normalized_name_clean","normalized_name_core","alias_id",
        "canonical_entity_id","canonical_display_name","entity_type","dish_family","cuisine_family",
        "meal_association_classic","service_form","prep_primary","temperature_mode","protein_primary",
        "starch_primary","vegetable_primary","fat_source_primary","restaurant_style","processing_level",
        "restaurant_specific_flag","generic_standin_flag","semantic_confidence","brand_candidate",
        "restaurant_candidate","genericity_hint","is_unknown_item","is_beverage_hint","is_condiment_hint",
        "is_side_hint","is_dessert_hint","component_role_final","role_confidence"
    ] if c in raw_final.columns]
    raw_final = dedupe_columns(raw_final[raw_cols_keep])

    # ---------- manifest ----------
    alias_done = int(alias_master.get("llm_review_status", pd.Series(dtype=object)).fillna("").astype(str).str.lower().eq("done").sum())
    entity_reviewed = int(entity_final.get("is_reviewed", pd.Series(dtype=bool)).fillna(False).sum())
    meal_sem_fields = ["meal_archetype_primary","cuisine_primary","service_form_primary","prep_profile","principal_protein"]
    fully_populated_meals = int(event_final[meal_sem_fields].notna().all(axis=1).sum())

    manifest = {
        "tables": {
            "canonical_food_entity": {
                "rows": int(len(entity_final)),
                "reviewed_rows": entity_reviewed,
                "rows_with_entity_type": int(entity_final.get("entity_type", pd.Series(dtype=object)).notna().sum()),
            },
            "meal_component_edge": {
                "rows": int(len(comp_final)),
                "reviewed_role_rows": int(comp_final.get("component_role_final", pd.Series(dtype=object)).notna().sum()),
            },
            "meal_semantic_features": {
                "rows": int(len(event_final)),
                "reviewed_meal_rows": int(event_final.get("is_semantically_reviewed", pd.Series(dtype=bool)).fillna(False).sum()),
                "rows_with_core_meal_semantics": fully_populated_meals,
            },
            "food_entry_semantic_view": {
                "rows": int(len(raw_final)),
                "rows_with_canonical_entity": int(raw_final.get("canonical_entity_id", pd.Series(dtype=object)).notna().sum()),
            },
        },
        "alias_review_progress": {
            "total_aliases": int(len(alias_master)),
            "aliases_done": alias_done,
            "aliases_pending": int(len(alias_master) - alias_done),
        },
        "build_note": "Rebuilt from reviewed alias/component/meal batch files plus current seed tables. This repaired finalization avoids stale placeholder entities and derives missing meal semantics from reviewed components.",
    }

    log("Writing repaired final meal DB tables...")
    entity_final.to_csv(final_dir / "canonical_food_entity.csv", index=False)
    comp_final.to_csv(final_dir / "meal_component_edge.csv", index=False)
    event_final.to_csv(final_dir / "meal_semantic_features.csv", index=False)
    raw_final.to_csv(final_dir / "food_entry_semantic_view.csv", index=False)
    (final_dir / "meal_db_final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log("Done.")
    log(f"Wrote repaired final meal DB to: {final_dir}")
    log(f"Alias review progress: {alias_done}/{len(alias_master)} complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair and rebuild final meal DB from reviewed batch files.")
    parser.add_argument("--project-root", default=".", help="Path to the project root.")
    parser.add_argument("--meal-db-dir", default="meal_db", help="Meal DB directory under project root.")
    args = parser.parse_args()
    build_repaired_final(Path(args.project_root).expanduser().resolve(), args.meal_db_dir)


if __name__ == "__main__":
    main()
