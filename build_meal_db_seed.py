from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


KNOWN_BRANDS = [
    "trader joe's", "tillamook", "ihop", "chili's", "mcdonald's", "taco bell",
    "burger king", "wendy's", "starbucks", "panera", "subway", "jersey mike's",
    "jimmy john's", "costco", "whole foods", "safeway", "ralphs", "chipotle",
    "panda express", "domino's", "pizza hut", "little caesars", "kfc",
    "popeyes", "arbys", "chick-fil-a", "in-n-out", "dunkin", "7-eleven",
    "celsius", "gatorade", "coca-cola", "pepsi", "la croix", "fairlife",
]

BEVERAGE_PATTERNS = [
    r"\bcoffee\b", r"\blatte\b", r"\btea\b", r"\bchai\b", r"\bsoda\b",
    r"\bcoke\b", r"\bpepsi\b", r"\bwater\b", r"\bjuice\b", r"\bmilk\b",
    r"\bshake\b", r"\bsmoothie\b", r"\bcelsius\b", r"\bdrink\b",
]
CONDIMENT_PATTERNS = [
    r"\bsauce\b", r"\bdressing\b", r"\bmayo\b", r"\bmustard\b", r"\bketchup\b",
    r"\baioli\b", r"\bsalsa\b", r"\bdip\b", r"\bbutter\b", r"\bcream cheese\b",
    r"\bjam\b", r"\bjelly\b", r"\bhot sauce\b",
]
SIDE_PATTERNS = [
    r"\bfries\b", r"\bhash browns\b", r"\btots\b", r"\bchips\b", r"\bfruit cup\b",
    r"\bside salad\b", r"\bcoleslaw\b", r"\bmac and cheese\b", r"\bmashed potatoes\b",
]
DESSERT_PATTERNS = [
    r"\bcookie\b", r"\bbrownie\b", r"\bcake\b", r"\bice cream\b", r"\bdonut\b",
    r"\bmuffin\b", r"\bcandy\b", r"\bpie\b",
]


MEAL_CONTEXT_COLUMNS = [
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
]


def log(msg: str) -> None:
    print(f"[meal-db] {msg}")


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def stable_id(prefix: str, *parts: Any, length: int = 12) -> str:
    joined = "||".join("" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x) for x in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def pick_display_name(row: pd.Series) -> str:
    for col in ["logged_name", "query_text", "unit_name"]:
        val = clean_text(row.get(col))
        if val:
            return val
    return "unknown_item"


def normalize_text_basic(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("’", "'")
    text = text.replace("&", " and ")
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\([^\)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9+'\-/ ]+", " ", text)
    text = re.sub(r"\b(extra|large|small|medium|regular|original)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "unknown_item"


def extract_brand_candidate(text: str) -> str:
    t = normalize_text_basic(text)
    for brand in sorted(KNOWN_BRANDS, key=len, reverse=True):
        if t.startswith(brand + " ") or t == brand:
            return brand
    # conservative fallback for possessive names like "x's ..."
    m = re.match(r"^([a-z0-9]+(?:\s+[a-z0-9]+)?'s)\s+", t)
    if m:
        candidate = m.group(1)
        if len(candidate) <= 24:
            return candidate
    return ""


def strip_brand_prefix(text: str, brand_candidate: str) -> str:
    t = normalize_text_basic(text)
    if brand_candidate and (t.startswith(brand_candidate + " ") or t == brand_candidate):
        t = t[len(brand_candidate):].strip(" -:")
    t = re.sub(r"^(brand|generic)\s+", "", t)
    return t or normalize_text_basic(text)


def pattern_flag(text: str, patterns: List[str]) -> int:
    t = normalize_text_basic(text)
    return int(any(re.search(p, t) for p in patterns))


def top_values_json(series: pd.Series, n: int = 5) -> str:
    vals = [clean_text(x) for x in series if clean_text(x)]
    counts = Counter(vals)
    top = [{"value": v, "count": c} for v, c in counts.most_common(n)]
    return json.dumps(top, ensure_ascii=False)


def counter_json(counter: Counter, n: int = 10) -> str:
    top = [{"value": v, "count": c} for v, c in counter.most_common(n)]
    return json.dumps(top, ensure_ascii=False)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_inputs(project_root: Path) -> Dict[str, pd.DataFrame]:
    paths = {
        "food_entries": project_root / "canonical" / "noom" / "noom_food_entries.csv",
        "meal_events": project_root / "canonical" / "noom" / "noom_meal_events.csv",
        "master_daily": project_root / "fused" / "master_daily_features.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))

    food = pd.read_csv(paths["food_entries"], low_memory=False)
    meals = pd.read_csv(paths["meal_events"], low_memory=False)
    daily = pd.read_csv(paths["master_daily"], low_memory=False)

    # parse dates / datetimes
    for col in ["date", "client_time_inserted", "server_time_created", "server_time_updated", "datetime_local_approx"]:
        if col in food.columns:
            food[col] = pd.to_datetime(food[col], errors="coerce")
    for col in ["date", "datetime_local_approx"]:
        if col in meals.columns:
            meals[col] = pd.to_datetime(meals[col], errors="coerce")
    if "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    return {"food_entries": food, "meal_events": meals, "master_daily": daily}


def build_meal_ids(meals: pd.DataFrame) -> pd.DataFrame:
    meals = meals.copy()
    meals["meal_id"] = [
        stable_id(
            "meal",
            row.date.date() if pd.notna(row.date) else None,
            row.time_slot,
            row.datetime_local_approx,
            row.meal_text,
            row.calories_kcal,
            row.item_count,
        )
        for row in meals.itertuples(index=False)
    ]
    # deterministic in-day order and context
    meals = meals.sort_values(["date", "datetime_local_approx", "time_slot", "meal_id"]).reset_index(drop=True)
    meals["meal_order_in_day"] = meals.groupby("date").cumcount() + 1
    meals["day_meal_count"] = meals.groupby("date")["meal_id"].transform("count")
    meals["day_total_meal_calories_kcal"] = meals.groupby("date")["calories_kcal"].transform("sum")
    meals["cumulative_meal_calories_before_meal"] = meals.groupby("date")["calories_kcal"].cumsum().shift(fill_value=0)
    meals["cumulative_meal_calories_after_meal"] = meals.groupby("date")["calories_kcal"].cumsum()
    meals["is_first_meal_of_day"] = (meals["meal_order_in_day"] == 1).astype(int)
    meals["is_last_meal_of_day"] = (meals["meal_order_in_day"] == meals["day_meal_count"]).astype(int)

    meals["prior_meal_id"] = meals.groupby("date")["meal_id"].shift(1)
    meals["prior_meal_text"] = meals.groupby("date")["meal_text"].shift(1)
    meals["prior_meal_time_slot"] = meals.groupby("date")["time_slot_label"].shift(1)
    meals["next_meal_id"] = meals.groupby("date")["meal_id"].shift(-1)
    meals["next_meal_text"] = meals.groupby("date")["meal_text"].shift(-1)
    meals["next_meal_time_slot"] = meals.groupby("date")["time_slot_label"].shift(-1)

    prior_dt = meals.groupby("date")["datetime_local_approx"].shift(1)
    meals["hours_since_prior_meal"] = (meals["datetime_local_approx"] - prior_dt).dt.total_seconds() / 3600.0
    meals["hours_until_next_meal"] = (meals.groupby("date")["datetime_local_approx"].shift(-1) - meals["datetime_local_approx"]).dt.total_seconds() / 3600.0
    return meals


def attach_meal_ids_to_entries(food: pd.DataFrame, meals: pd.DataFrame) -> pd.DataFrame:
    food = food.copy()
    join_cols = ["date", "time_slot", "time_slot_label", "datetime_local_approx"]
    meal_key = meals[join_cols + ["meal_id", "meal_text", "meal_order_in_day", "day_meal_count"]].drop_duplicates()
    merged = food.merge(meal_key, on=join_cols, how="left", validate="m:1")
    if merged["meal_id"].isna().any():
        # fallback to date + time_slot only if exact datetime did not match
        fallback_key = meals[["date", "time_slot", "meal_id", "meal_text", "meal_order_in_day", "day_meal_count"]].drop_duplicates()
        missing = merged["meal_id"].isna()
        recovered = merged.loc[missing].drop(columns=["meal_id", "meal_text", "meal_order_in_day", "day_meal_count"]).merge(
            fallback_key, on=["date", "time_slot"], how="left"
        )
        for col in ["meal_id", "meal_text", "meal_order_in_day", "day_meal_count"]:
            merged.loc[missing, col] = recovered[col].values
    return merged


def enrich_food_entries(food: pd.DataFrame) -> pd.DataFrame:
    food = food.copy()
    food["display_name_raw"] = food.apply(pick_display_name, axis=1)
    food["normalized_name_clean"] = food["display_name_raw"].map(normalize_text_basic)
    food["brand_candidate"] = food["display_name_raw"].map(extract_brand_candidate)
    food["restaurant_candidate"] = food["brand_candidate"]
    food["normalized_name_core"] = [
        strip_brand_prefix(name, brand) for name, brand in zip(food["display_name_raw"], food["brand_candidate"])
    ]
    food["alias_id"] = food["normalized_name_clean"].map(lambda x: stable_id("alias", x))
    food["provisional_canonical_entity_id"] = food["normalized_name_core"].map(lambda x: stable_id("ent", x))
    food["genericity_hint"] = ((food["brand_candidate"] == "") & (food["restaurant_candidate"] == "")).astype(int)
    food["is_unknown_item"] = food["normalized_name_clean"].isin(["unknown_item", "unknown item"]).astype(int)
    food["is_beverage_hint"] = food["display_name_raw"].map(lambda x: pattern_flag(x, BEVERAGE_PATTERNS))
    food["is_condiment_hint"] = food["display_name_raw"].map(lambda x: pattern_flag(x, CONDIMENT_PATTERNS))
    food["is_side_hint"] = food["display_name_raw"].map(lambda x: pattern_flag(x, SIDE_PATTERNS))
    food["is_dessert_hint"] = food["display_name_raw"].map(lambda x: pattern_flag(x, DESSERT_PATTERNS))
    food["calories_kcal"] = safe_numeric(food.get("calories_kcal"))
    food["protein_g"] = safe_numeric(food.get("protein_g"))
    food["carbs_g"] = safe_numeric(food.get("carbs_g"))
    food["fat_g"] = safe_numeric(food.get("fat_g"))
    food["fiber_g"] = safe_numeric(food.get("fiber_g"))
    food["sodium_mg"] = safe_numeric(food.get("sodium_mg"))
    return food


def build_component_seed(food: pd.DataFrame, meals: pd.DataFrame) -> pd.DataFrame:
    meal_cal = meals[["meal_id", "calories_kcal"]].rename(columns={"calories_kcal": "meal_calories_kcal"})
    comp = food.merge(meal_cal, on="meal_id", how="left")
    comp["meal_calories_kcal"] = safe_numeric(comp["meal_calories_kcal"])
    comp["calorie_share_of_meal"] = np.where(
        comp["meal_calories_kcal"].gt(0),
        comp["calories_kcal"] / comp["meal_calories_kcal"],
        np.nan,
    )
    comp["quantity_proxy"] = comp["servings"].fillna(1)
    comp["heuristic_role_hint"] = np.select(
        [
            comp["is_beverage_hint"].eq(1),
            comp["is_condiment_hint"].eq(1),
            comp["is_dessert_hint"].eq(1),
            comp["is_side_hint"].eq(1),
            comp["calorie_share_of_meal"].fillna(0).ge(0.45),
        ],
        ["beverage", "condiment", "dessert", "side", "main_candidate"],
        default="unassigned",
    )
    comp["component_rank_by_calories"] = comp.groupby("meal_id")["calories_kcal"].rank(method="dense", ascending=False)
    cols = [
        "meal_id", "uuid", "date", "time_slot", "time_slot_label", "datetime_local_approx",
        "meal_text", "meal_order_in_day", "day_meal_count", "display_name_raw", "normalized_name_clean",
        "normalized_name_core", "alias_id", "provisional_canonical_entity_id", "brand_candidate",
        "restaurant_candidate", "genericity_hint", "is_unknown_item", "is_beverage_hint",
        "is_condiment_hint", "is_side_hint", "is_dessert_hint", "calories_kcal", "protein_g",
        "carbs_g", "fat_g", "fiber_g", "sodium_mg", "quantity_proxy", "meal_calories_kcal",
        "calorie_share_of_meal", "component_rank_by_calories", "heuristic_role_hint",
    ]
    rename = {"uuid": "food_entry_uuid"}
    comp = comp[cols].rename(columns=rename)
    comp["component_role"] = ""
    comp["role_confidence"] = np.nan
    comp["llm_review_status"] = "pending"
    return comp.sort_values(["date", "datetime_local_approx", "meal_id", "component_rank_by_calories", "food_entry_uuid"]).reset_index(drop=True)


def build_alias_seed(food: pd.DataFrame, component_seed: pd.DataFrame) -> pd.DataFrame:
    # cooccurrence counts
    co_counts: Dict[str, Counter] = defaultdict(Counter)
    meal_text_examples: Dict[str, Counter] = defaultdict(Counter)
    for meal_id, group in component_seed.groupby("meal_id"):
        aliases = group["normalized_name_clean"].tolist()
        uniq_aliases = list(dict.fromkeys([a for a in aliases if a]))
        for a in uniq_aliases:
            for b in uniq_aliases:
                if a != b:
                    co_counts[a][b] += 1
            meal_text = clean_text(group["meal_text"].iloc[0])
            if meal_text:
                meal_text_examples[a][meal_text] += 1

    rows = []
    for alias_id, group in food.groupby("alias_id"):
        norm_name = group["normalized_name_clean"].iloc[0]
        rows.append({
            "alias_id": alias_id,
            "normalized_name_clean": norm_name,
            "normalized_name_core_mode": group["normalized_name_core"].mode().iloc[0] if not group["normalized_name_core"].mode().empty else group["normalized_name_core"].iloc[0],
            "provisional_canonical_entity_id": group["provisional_canonical_entity_id"].mode().iloc[0] if not group["provisional_canonical_entity_id"].mode().empty else group["provisional_canonical_entity_id"].iloc[0],
            "example_logged_names": top_values_json(group["display_name_raw"], n=8),
            "example_query_texts": top_values_json(group["query_text"], n=5),
            "example_unit_names": top_values_json(group["unit_name"], n=5),
            "brand_candidate_modes": top_values_json(group.loc[group["brand_candidate"].astype(str).str.len() > 0, "brand_candidate"], n=5),
            "restaurant_candidate_modes": top_values_json(group.loc[group["restaurant_candidate"].astype(str).str.len() > 0, "restaurant_candidate"], n=5),
            "entry_count": int(len(group)),
            "meal_count": int(group["meal_id"].nunique(dropna=True)),
            "days_seen": int(group["date"].dt.date.nunique()),
            "first_seen": group["date"].min(),
            "last_seen": group["date"].max(),
            "median_calories_kcal": float(group["calories_kcal"].median()) if group["calories_kcal"].notna().any() else np.nan,
            "median_protein_g": float(group["protein_g"].median()) if group["protein_g"].notna().any() else np.nan,
            "median_carbs_g": float(group["carbs_g"].median()) if group["carbs_g"].notna().any() else np.nan,
            "median_fat_g": float(group["fat_g"].median()) if group["fat_g"].notna().any() else np.nan,
            "median_fiber_g": float(group["fiber_g"].median()) if group["fiber_g"].notna().any() else np.nan,
            "dominant_time_slot": group["time_slot_label"].mode().iloc[0] if not group["time_slot_label"].mode().empty else "",
            "time_slot_distribution": counter_json(Counter(group["time_slot_label"].fillna("unknown"))),
            "top_cooccurring_aliases": counter_json(co_counts[norm_name], n=10),
            "top_sequence_examples": counter_json(meal_text_examples[norm_name], n=5),
            "genericity_hint_rate": float(group["genericity_hint"].mean()) if len(group) else np.nan,
            "beverage_hint_rate": float(group["is_beverage_hint"].mean()) if len(group) else np.nan,
            "condiment_hint_rate": float(group["is_condiment_hint"].mean()) if len(group) else np.nan,
            "side_hint_rate": float(group["is_side_hint"].mean()) if len(group) else np.nan,
            "dessert_hint_rate": float(group["is_dessert_hint"].mean()) if len(group) else np.nan,
            "canonical_entity_id": "",
            "llm_review_status": "pending",
            "llm_notes": "",
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["entry_count", "meal_count", "days_seen", "normalized_name_clean"], ascending=[False, False, False, True]).reset_index(drop=True)


def build_entity_seed(food: pd.DataFrame, alias_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = food.groupby("provisional_canonical_entity_id")
    for ent_id, group in grouped:
        alias_subset = alias_seed[alias_seed["provisional_canonical_entity_id"] == ent_id]
        rows.append({
            "canonical_entity_id": ent_id,
            "seed_alias_count": int(alias_subset["alias_id"].nunique()),
            "seed_aliases": json.dumps(alias_subset["normalized_name_clean"].head(20).tolist(), ensure_ascii=False),
            "example_logged_names": top_values_json(group["display_name_raw"], n=10),
            "entry_count": int(len(group)),
            "meal_count": int(group["meal_id"].nunique(dropna=True)),
            "days_seen": int(group["date"].dt.date.nunique()),
            "first_seen": group["date"].min(),
            "last_seen": group["date"].max(),
            "provisional_display_name": group["normalized_name_core"].mode().iloc[0] if not group["normalized_name_core"].mode().empty else group["normalized_name_core"].iloc[0],
            "canonical_display_name": "",
            "entity_type": "",
            "dish_family": "",
            "cuisine_family": "",
            "meal_association_classic": "",
            "service_form": "",
            "prep_primary": "",
            "temperature_mode": "",
            "protein_primary": "",
            "starch_primary": "",
            "vegetable_primary": "",
            "fat_source_primary": "",
            "restaurant_style": "",
            "processing_level": "",
            "restaurant_specific_flag": "",
            "generic_standin_flag": "",
            "semantic_confidence": np.nan,
            "inference_basis": "",
            "llm_review_status": "pending",
            "llm_notes": "",
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["entry_count", "meal_count", "seed_alias_count", "provisional_display_name"], ascending=[False, False, False, True]).reset_index(drop=True)


def build_meal_event_seed(meals: pd.DataFrame, component_seed: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    meals = meals.copy()
    # component-derived summaries
    comp_summary = component_seed.groupby("meal_id", as_index=False).agg(
        distinct_alias_count=("alias_id", "nunique"),
        distinct_entity_count=("provisional_canonical_entity_id", "nunique"),
        beverage_component_count=("is_beverage_hint", "sum"),
        condiment_component_count=("is_condiment_hint", "sum"),
        dessert_component_count=("is_dessert_hint", "sum"),
        side_component_count=("is_side_hint", "sum"),
    )
    # readable component signature by descending calorie share then name
    sig_rows = []
    for meal_id, group in component_seed.groupby("meal_id"):
        g = group.sort_values(["calorie_share_of_meal", "calories_kcal", "normalized_name_clean"], ascending=[False, False, True])
        signature = " | ".join(g["normalized_name_clean"].head(8).tolist())
        sig_rows.append({"meal_id": meal_id, "cooccurrence_signature": signature})
    sig_df = pd.DataFrame(sig_rows)

    out = meals.merge(comp_summary, on="meal_id", how="left").merge(sig_df, on="meal_id", how="left")

    # join daily context conservatively
    daily_keep = ["date"] + [c for c in MEAL_CONTEXT_COLUMNS if c in daily.columns]
    daily_ctx = daily[daily_keep].copy()
    out = out.merge(daily_ctx, on="date", how="left")

    out["remaining_budget_after_meal_kcal"] = np.where(
        out.get("calorie_budget_kcal").notna(),
        out.get("calorie_budget_kcal") - out.get("cumulative_meal_calories_after_meal"),
        np.nan,
    )
    out["remaining_budget_before_meal_kcal"] = np.where(
        out.get("calorie_budget_kcal").notna(),
        out.get("calorie_budget_kcal") - out.get("cumulative_meal_calories_before_meal"),
        np.nan,
    )

    # blank semantic fields for later LLM fill
    for col in [
        "meal_archetype_primary", "meal_archetype_secondary", "cuisine_primary", "cuisine_secondary",
        "service_form_primary", "prep_profile", "principal_protein", "principal_starch",
        "principal_veg", "principal_fat_source", "comfort_food_score", "fresh_light_score",
        "indulgence_score", "energy_density_style", "satiety_style", "coherence_score",
        "restaurant_specific_flag", "generic_standin_flag", "novelty_cluster_id", "similarity_cluster_id",
        "semantic_confidence", "llm_review_status", "llm_notes",
    ]:
        if col in ["comfort_food_score", "fresh_light_score", "indulgence_score", "coherence_score", "semantic_confidence"]:
            out[col] = np.nan
        else:
            out[col] = ""
    out["llm_review_status"] = "pending"

    cols_front = [
        "meal_id", "date", "datetime_local_approx", "time_slot", "time_slot_label", "meal_order_in_day",
        "day_meal_count", "is_first_meal_of_day", "is_last_meal_of_day", "hours_since_prior_meal",
        "hours_until_next_meal", "calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g",
        "sodium_mg", "item_count", "distinct_alias_count", "distinct_entity_count", "meal_text",
        "cooccurrence_signature", "prior_meal_id", "prior_meal_text", "next_meal_id", "next_meal_text",
        "cumulative_meal_calories_before_meal", "cumulative_meal_calories_after_meal",
        "day_total_meal_calories_kcal", "remaining_budget_before_meal_kcal", "remaining_budget_after_meal_kcal",
    ]
    remaining = [c for c in out.columns if c not in cols_front]
    out = out[cols_front + remaining]
    return out.sort_values(["date", "datetime_local_approx", "meal_order_in_day", "meal_id"]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_manifest(path: Path, tables: Dict[str, pd.DataFrame]) -> None:
    manifest = []
    for name, df in tables.items():
        manifest.append({
            "table_name": name,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
        })
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def build_all(project_root: Path, output_dir_name: str = "meal_db") -> None:
    out_root = project_root / output_dir_name
    seed_dir = out_root / "seed"
    review_dir = out_root / "review_batches"
    ensure_dirs([seed_dir, review_dir])

    log("Loading canonical/fused inputs...")
    data = load_inputs(project_root)
    food = data["food_entries"]
    meals = data["meal_events"]
    daily = data["master_daily"]

    # Restrict daily context to active meal era if a larger full-range file was accidentally supplied.
    first_meal_date = meals["date"].min()
    last_meal_date = meals["date"].max()
    if pd.notna(first_meal_date) and pd.notna(last_meal_date):
        daily = daily[(daily["date"] >= first_meal_date - pd.Timedelta(days=30)) & (daily["date"] <= last_meal_date + pd.Timedelta(days=1))].copy()

    log("Building stable meal IDs and in-day meal context...")
    meals = build_meal_ids(meals)

    log("Attaching meal IDs to raw food entries...")
    food = attach_meal_ids_to_entries(food, meals)

    log("Building deterministic food-entry enrichment...")
    food = enrich_food_entries(food)

    log("Building meal component seed table...")
    component_seed = build_component_seed(food, meals)

    log("Building alias seed table...")
    alias_seed = build_alias_seed(food, component_seed)

    log("Building provisional canonical food-entity seed table...")
    entity_seed = build_entity_seed(food, alias_seed)

    log("Building meal event seed table with fused context...")
    meal_event_seed = build_meal_event_seed(meals, component_seed, daily)

    log("Writing seed tables...")
    raw_food_entry_enriched_seed = food.sort_values(["date", "datetime_local_approx", "meal_id", "alias_id", "uuid"]).reset_index(drop=True)

    tables = {
        "raw_food_entry_enriched_seed": raw_food_entry_enriched_seed,
        "food_alias_seed": alias_seed,
        "canonical_food_entity_seed": entity_seed,
        "meal_component_seed": component_seed,
        "meal_event_seed": meal_event_seed,
    }
    for name, df in tables.items():
        write_csv(df, seed_dir / f"{name}.csv")

    write_manifest(seed_dir / "seed_manifest.json", tables)

    # Convenience first review batch: top aliases and the meals they appear in.
    batch_size = min(200, len(alias_seed))
    alias_batch = alias_seed.head(batch_size).copy()
    write_csv(alias_batch, review_dir / "food_alias_batch_001.csv")

    batch_alias_ids = set(alias_batch["alias_id"])
    meal_ids = component_seed.loc[component_seed["alias_id"].isin(batch_alias_ids), "meal_id"].dropna().unique().tolist()
    meal_example_batch = meal_event_seed[meal_event_seed["meal_id"].isin(meal_ids)].copy()
    # keep representative meals only
    meal_example_batch = meal_example_batch.sort_values(["date", "meal_order_in_day", "calories_kcal"], ascending=[True, True, False])
    write_csv(meal_example_batch, review_dir / "food_alias_batch_001_meal_examples.csv")

    (out_root / "README_seed_outputs.txt").write_text(
        "Meal DB seed build complete.\n\n"
        "Primary files to upload for LLM review:\n"
        "1) seed/food_alias_seed.csv\n"
        "2) seed/canonical_food_entity_seed.csv\n"
        "3) seed/meal_event_seed.csv\n"
        "4) optionally seed/meal_component_seed.csv\n\n"
        "Convenience batch files:\n"
        "- review_batches/food_alias_batch_001.csv\n"
        "- review_batches/food_alias_batch_001_meal_examples.csv\n",
        encoding="utf-8",
    )

    log("Done.")
    log(f"Wrote meal-db seed tables to: {seed_dir}")
    log(f"Wrote first review batch to: {review_dir}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic meal database seed tables from current canonical/fused FoodAI outputs.")
    parser.add_argument("--project-root", default=".", help="Path to the FoodAI project root containing canonical/ and fused/ folders.")
    parser.add_argument("--output-dir", default="meal_db", help="Output folder name to create under the project root.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    build_all(project_root, output_dir_name=args.output_dir)


if __name__ == "__main__":
    main()
