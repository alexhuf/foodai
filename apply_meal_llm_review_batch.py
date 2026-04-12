
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


def log(msg: str) -> None:
    print(f"[meal-db] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def backup_files(files: Iterable[Path], backup_dir: Path) -> None:
    ensure_dir(backup_dir)
    for src in files:
        if src.exists():
            dst = backup_dir / src.name
            src.replace(dst)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def safe_json_list(text: str) -> list[str]:
    if pd.isna(text):
        return []
    try:
        val = json.loads(text)
        return [str(x) for x in val] if isinstance(val, list) else []
    except Exception:
        return []


def build_placeholder_entities_from_aliases(alias_df: pd.DataFrame, reviewed_ids: set[str]) -> pd.DataFrame:
    # Any alias without a reviewed canonical entity keeps its fallback canonical id.
    temp = alias_df.copy()

    # Guarantee every alias has some canonical ID for export continuity.
    temp["canonical_entity_id"] = temp["canonical_entity_id"].fillna(temp["provisional_canonical_entity_id"])

    pending = temp[~temp["canonical_entity_id"].isin(reviewed_ids)].copy()
    if pending.empty:
        return pd.DataFrame(columns=[
            "canonical_entity_id","seed_alias_count","seed_aliases","example_logged_names","entry_count","meal_count",
            "days_seen","first_seen","last_seen","provisional_display_name","canonical_display_name","entity_type",
            "dish_family","cuisine_family","meal_association_classic","service_form","prep_primary","temperature_mode",
            "protein_primary","starch_primary","vegetable_primary","fat_source_primary","restaurant_style",
            "processing_level","semantic_confidence","restaurant_specific_flag","generic_standin_flag",
            "inference_basis","llm_review_status","llm_notes"
        ])

    def join_unique(series: pd.Series) -> str:
        vals = []
        for x in series.dropna().astype(str):
            vals.append(x)
        # preserve first-seen order
        seen = set()
        out = []
        for x in vals:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return json.dumps(out, ensure_ascii=False)

    rows = []
    for canon_id, g in pending.groupby("canonical_entity_id", dropna=False):
        if pd.isna(canon_id):
            continue
        ex_names = []
        for s in g["example_logged_names"].dropna().astype(str):
            ex_names.append(s)
        rows.append({
            "canonical_entity_id": canon_id,
            "seed_alias_count": int(g["alias_id"].nunique()),
            "seed_aliases": json.dumps(sorted(g["normalized_name_clean"].dropna().astype(str).unique().tolist()), ensure_ascii=False),
            "example_logged_names": ex_names[0] if ex_names else "[]",
            "entry_count": int(pd.to_numeric(g["entry_count"], errors="coerce").fillna(0).sum()),
            "meal_count": int(pd.to_numeric(g["meal_count"], errors="coerce").fillna(0).sum()),
            "days_seen": int(pd.to_numeric(g["days_seen"], errors="coerce").fillna(0).max()),
            "first_seen": g["first_seen"].dropna().astype(str).min() if g["first_seen"].notna().any() else None,
            "last_seen": g["last_seen"].dropna().astype(str).max() if g["last_seen"].notna().any() else None,
            "provisional_display_name": g["normalized_name_core_mode"].dropna().astype(str).mode().iloc[0] if g["normalized_name_core_mode"].notna().any() else None,
            "canonical_display_name": None,
            "entity_type": None,
            "dish_family": None,
            "cuisine_family": None,
            "meal_association_classic": None,
            "service_form": None,
            "prep_primary": None,
            "temperature_mode": None,
            "protein_primary": None,
            "starch_primary": None,
            "vegetable_primary": None,
            "fat_source_primary": None,
            "restaurant_style": None,
            "processing_level": None,
            "semantic_confidence": None,
            "restaurant_specific_flag": None,
            "generic_standin_flag": None,
            "inference_basis": None,
            "llm_review_status": "pending",
            "llm_notes": None,
        })
    return pd.DataFrame(rows)


def apply_review_batch(project_root: Path, meal_db_dir: str, batch_label: str) -> None:
    root = project_root / meal_db_dir
    seed_dir = root / "seed"
    review_dir = root / "reviews" / batch_label
    backup_dir = root / "_archive" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{batch_label}"

    alias_seed_path = seed_dir / "food_alias_seed.csv"
    canon_seed_path = seed_dir / "canonical_food_entity_seed.csv"
    comp_seed_path = seed_dir / "meal_component_seed.csv"
    meal_seed_path = seed_dir / "meal_event_seed.csv"

    required_seed = [alias_seed_path, canon_seed_path, comp_seed_path, meal_seed_path]
    for p in required_seed:
        if not p.exists():
            raise FileNotFoundError(f"Missing seed file: {p}")

    alias_review_path = review_dir / "food_alias_batch_001_llm_enriched.csv"
    canon_review_path = review_dir / "canonical_food_entity_batch_001_llm.csv"
    comp_review_path = review_dir / "food_alias_batch_001_components_llm.csv"
    meal_review_path = review_dir / "food_alias_batch_001_meal_examples_llm.csv"

    # Generic fallback: if exact batch_001 names aren't present, use any *_llm.csv file patterns.
    if not alias_review_path.exists():
        matches = list(review_dir.glob("food_alias_batch_*_llm_enriched.csv"))
        if not matches:
            raise FileNotFoundError(f"Missing alias review file in {review_dir}")
        alias_review_path = matches[0]
    if not canon_review_path.exists():
        matches = list(review_dir.glob("canonical_food_entity_batch_*_llm.csv"))
        if not matches:
            raise FileNotFoundError(f"Missing canonical entity review file in {review_dir}")
        canon_review_path = matches[0]
    if not comp_review_path.exists():
        matches = list(review_dir.glob("food_alias_batch_*_components_llm.csv"))
        if not matches:
            raise FileNotFoundError(f"Missing component review file in {review_dir}")
        comp_review_path = matches[0]
    if not meal_review_path.exists():
        matches = list(review_dir.glob("food_alias_batch_*_meal_examples_llm.csv"))
        if not matches:
            raise FileNotFoundError(f"Missing meal example review file in {review_dir}")
        meal_review_path = matches[0]

    log("Loading seed tables...")
    alias_seed = read_csv(alias_seed_path)
    canon_seed = read_csv(canon_seed_path)
    comp_seed = read_csv(comp_seed_path)
    meal_seed = read_csv(meal_seed_path)

    log("Loading reviewed batch files...")
    alias_review = read_csv(alias_review_path)
    canon_review = read_csv(canon_review_path)
    comp_review = read_csv(comp_review_path)
    meal_review = read_csv(meal_review_path)

    log("Archiving current seed tables before applying review batch...")
    ensure_dir(backup_dir)
    for src in required_seed:
        dst = backup_dir / src.name
        src.replace(dst)

    # Reload backup copies for mutation/writeback.
    alias_seed = read_csv(backup_dir / "food_alias_seed.csv")
    canon_seed = read_csv(backup_dir / "canonical_food_entity_seed.csv")
    comp_seed = read_csv(backup_dir / "meal_component_seed.csv")
    meal_seed = read_csv(backup_dir / "meal_event_seed.csv")

    log("Applying reviewed alias semantics back to food_alias_seed.csv ...")
    alias_key = "alias_id"
    alias_review_cols = [c for c in alias_review.columns if c != alias_key]
    alias_seed = alias_seed.merge(alias_review[[alias_key] + alias_review_cols], on=alias_key, how="left", suffixes=("", "__review"))

    # Fields we want to write back into the seed alias table.
    alias_fields_to_apply = [
        "canonical_entity_id",
        "llm_review_status",
        "llm_notes",
        "canonical_display_name",
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
    ]
    for field in alias_fields_to_apply:
        rev = f"{field}__review"
        if rev in alias_seed.columns:
            if field not in alias_seed.columns:
                alias_seed[field] = pd.NA
            alias_seed[field] = alias_seed[rev].combine_first(alias_seed[field])

    # Mark reviewed aliases as done so the batch exporter moves on.
    reviewed_alias_ids = set(alias_review["alias_id"].dropna().astype(str))
    if "llm_review_status" not in alias_seed.columns:
        alias_seed["llm_review_status"] = "pending"
    alias_seed.loc[alias_seed["alias_id"].astype(str).isin(reviewed_alias_ids), "llm_review_status"] = "done"

    # Fill fallback canonical ids for still-unreviewed aliases.
    alias_seed["canonical_entity_id"] = alias_seed["canonical_entity_id"].fillna(alias_seed["provisional_canonical_entity_id"])

    # Drop merge helper columns.
    alias_seed = alias_seed[[c for c in alias_seed.columns if not c.endswith("__review")]]

    log("Rebuilding canonical_food_entity_seed.csv from reviewed entities + pending placeholders ...")
    reviewed_entity_ids = set(canon_review["canonical_entity_id"].dropna().astype(str))
    placeholder_entities = build_placeholder_entities_from_aliases(alias_seed, reviewed_entity_ids)
    canon_out = pd.concat([canon_review, placeholder_entities], ignore_index=True, sort=False)

    # Guarantee stable ordering.
    sort_cols = [c for c in ["llm_review_status", "entry_count", "meal_count", "days_seen", "canonical_display_name", "provisional_display_name"] if c in canon_out.columns]
    ascending = [True, False, False, False, True, True][:len(sort_cols)]
    canon_out = canon_out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    log("Applying reviewed component mappings back to meal_component_seed.csv ...")
    comp_key_cols = ["food_entry_uuid"]
    comp_review_cols = [c for c in comp_review.columns if c not in comp_key_cols]
    comp_seed = comp_seed.merge(comp_review[comp_key_cols + comp_review_cols], on="food_entry_uuid", how="left", suffixes=("", "__review"))

    comp_fields_to_apply = [
        "canonical_entity_id",
        "canonical_display_name",
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
        "component_role_suggested",
        "role_confidence_suggested",
        "component_role",
        "role_confidence",
        "llm_review_status",
    ]
    for field in comp_fields_to_apply:
        rev = f"{field}__review"
        if rev in comp_seed.columns:
            if field not in comp_seed.columns:
                comp_seed[field] = pd.NA
            comp_seed[field] = comp_seed[rev].combine_first(comp_seed[field])

    # Promote suggested role into component_role when main field is still empty.
    if "component_role_suggested" in comp_seed.columns:
        if "component_role" not in comp_seed.columns:
            comp_seed["component_role"] = pd.NA
        empty = comp_seed["component_role"].isna()
        comp_seed.loc[empty, "component_role"] = comp_seed.loc[empty, "component_role_suggested"]
    if "role_confidence_suggested" in comp_seed.columns:
        if "role_confidence" not in comp_seed.columns:
            comp_seed["role_confidence"] = pd.NA
        empty = comp_seed["role_confidence"].isna()
        comp_seed.loc[empty, "role_confidence"] = comp_seed.loc[empty, "role_confidence_suggested"]

    comp_seed = comp_seed[[c for c in comp_seed.columns if not c.endswith("__review")]]

    log("Applying reviewed meal-level semantics back to meal_event_seed.csv ...")
    meal_key = "meal_id"
    meal_review_cols = [c for c in meal_review.columns if c != meal_key]
    meal_seed = meal_seed.merge(meal_review[[meal_key] + meal_review_cols], on=meal_key, how="left", suffixes=("", "__review"))

    # Map suggested fields back into main semantic columns.
    suggested_to_main = {
        "meal_archetype_primary_suggested": "meal_archetype_primary",
        "cuisine_primary_suggested": "cuisine_primary",
        "cuisine_secondary_suggested": "cuisine_secondary",
        "service_form_primary_suggested": "service_form_primary",
        "prep_profile_suggested": "prep_profile",
        "principal_protein_suggested": "principal_protein",
        "principal_starch_suggested": "principal_starch",
        "principal_veg_suggested": "principal_veg",
        "principal_fat_source_suggested": "principal_fat_source",
        "comfort_food_score_suggested": "comfort_food_score",
        "fresh_light_score_suggested": "fresh_light_score",
        "indulgence_score_suggested": "indulgence_score",
        "energy_density_style_suggested": "energy_density_style",
        "satiety_style_suggested": "satiety_style",
        "coherence_score_suggested": "coherence_score",
        "restaurant_specific_flag_suggested": "restaurant_specific_flag",
        "generic_standin_flag_suggested": "generic_standin_flag",
        "semantic_confidence_suggested": "semantic_confidence",
    }
    for suggested, main in suggested_to_main.items():
        rev = f"{suggested}__review"
        if rev in meal_seed.columns:
            if main not in meal_seed.columns:
                meal_seed[main] = pd.NA
            meal_seed[main] = meal_seed[rev].combine_first(meal_seed[main])

    for field in ["llm_review_status", "llm_notes"]:
        rev = f"{field}__review"
        if rev in meal_seed.columns:
            if field not in meal_seed.columns:
                meal_seed[field] = pd.NA
            meal_seed[field] = meal_seed[rev].combine_first(meal_seed[field])

    reviewed_meal_ids = set(meal_review["meal_id"].dropna().astype(str))
    if "llm_review_status" not in meal_seed.columns:
        meal_seed["llm_review_status"] = "pending"
    meal_seed.loc[meal_seed["meal_id"].astype(str).isin(reviewed_meal_ids), "llm_review_status"] = "done"

    meal_seed = meal_seed[[c for c in meal_seed.columns if not c.endswith("__review")]]

    log("Writing updated seed tables back into meal_db/seed ...")
    ensure_dir(seed_dir)
    alias_seed.to_csv(alias_seed_path, index=False)
    canon_out.to_csv(canon_seed_path, index=False)
    comp_seed.to_csv(comp_seed_path, index=False)
    meal_seed.to_csv(meal_seed_path, index=False)

    # Also write easy-to-browse current snapshots.
    current_dir = root / "current"
    ensure_dir(current_dir)
    alias_seed.to_csv(current_dir / "food_alias_current.csv", index=False)
    canon_out.to_csv(current_dir / "canonical_food_entity_current.csv", index=False)
    comp_seed.to_csv(current_dir / "meal_component_current.csv", index=False)
    meal_seed.to_csv(current_dir / "meal_event_current.csv", index=False)

    manifest = {
        "batch_label": batch_label,
        "review_dir": str(review_dir),
        "backup_dir": str(backup_dir),
        "reviewed_alias_count": int(len(reviewed_alias_ids)),
        "reviewed_entity_count": int(len(reviewed_entity_ids)),
        "reviewed_component_count": int(comp_review["food_entry_uuid"].nunique()),
        "reviewed_meal_count": int(len(reviewed_meal_ids)),
    }
    with open(current_dir / f"apply_manifest_{batch_label}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log("Done.")
    log(f"Archived previous seed tables to: {backup_dir}")
    log(f"Updated seed tables in: {seed_dir}")
    log(f"Wrote current snapshots to: {current_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply one reviewed meal-ontology batch back into meal_db seed tables.")
    parser.add_argument("--project-root", default=".", help="Project root containing meal_db/")
    parser.add_argument("--meal-db-dir", default="meal_db", help="Meal DB directory name under project root")
    parser.add_argument("--batch-label", default="batch_001", help="Review batch label, e.g. batch_001")
    args = parser.parse_args()

    apply_review_batch(
        project_root=Path(args.project_root).expanduser().resolve(),
        meal_db_dir=args.meal_db_dir,
        batch_label=args.batch_label,
    )
