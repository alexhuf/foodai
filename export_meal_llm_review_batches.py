from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(f"[meal-db] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_batches(project_root: Path, output_dir_name: str, batch_size: int, max_meal_examples: int) -> None:
    root = project_root / output_dir_name
    seed_dir = root / "seed"
    review_dir = root / "review_batches"
    ensure_dir(review_dir)

    alias_path = seed_dir / "food_alias_seed.csv"
    comp_path = seed_dir / "meal_component_seed.csv"
    meal_path = seed_dir / "meal_event_seed.csv"

    for p in [alias_path, comp_path, meal_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    aliases = pd.read_csv(alias_path, low_memory=False)
    components = pd.read_csv(comp_path, low_memory=False)
    meals = pd.read_csv(meal_path, low_memory=False)

    sort_cols = [c for c in ["llm_review_status", "entry_count", "meal_count", "days_seen"] if c in aliases.columns]
    if "llm_review_status" in aliases.columns:
        aliases["needs_review"] = (aliases["llm_review_status"].fillna("pending") != "done").astype(int)
        aliases = aliases.sort_values(["needs_review", "entry_count", "meal_count", "days_seen", "normalized_name_clean"], ascending=[False, False, False, False, True])
        aliases = aliases.drop(columns=["needs_review"])
    else:
        aliases = aliases.sort_values(["entry_count", "meal_count", "days_seen", "normalized_name_clean"], ascending=[False, False, False, True])

    total = len(aliases)
    if total == 0:
        raise ValueError("food_alias_seed.csv is empty")

    batch_count = (total + batch_size - 1) // batch_size
    log(f"Creating {batch_count} review batches from {total} aliases...")

    for i in range(batch_count):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        batch_num = i + 1
        batch = aliases.iloc[start:end].copy()
        batch_file = review_dir / f"food_alias_batch_{batch_num:03d}.csv"
        batch.to_csv(batch_file, index=False)

        batch_alias_ids = set(batch["alias_id"])
        comp_batch = components[components["alias_id"].isin(batch_alias_ids)].copy()
        comp_batch.to_csv(review_dir / f"food_alias_batch_{batch_num:03d}_components.csv", index=False)

        # representative meals: highest-frequency meals per alias, deduped across the batch
        representative_meal_ids = []
        for alias_id, g in comp_batch.groupby("alias_id"):
            counts = g["meal_id"].value_counts().head(max_meal_examples)
            representative_meal_ids.extend(counts.index.tolist())
        representative_meal_ids = list(dict.fromkeys(representative_meal_ids))
        meal_batch = meals[meals["meal_id"].isin(representative_meal_ids)].copy()
        meal_batch = meal_batch.sort_values(["date", "meal_order_in_day", "calories_kcal"], ascending=[True, True, False])
        meal_batch.to_csv(review_dir / f"food_alias_batch_{batch_num:03d}_meal_examples.csv", index=False)

    log(f"Wrote review batches to: {review_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split meal alias seed tables into review batches for LLM enrichment uploads.")
    parser.add_argument("--project-root", default=".", help="Path to the FoodAI project root containing meal_db/")
    parser.add_argument("--output-dir", default="meal_db", help="Meal DB folder name under the project root")
    parser.add_argument("--batch-size", type=int, default=150, help="Aliases per batch")
    parser.add_argument("--max-meal-examples", type=int, default=3, help="Representative meals per alias")
    args = parser.parse_args()

    build_batches(Path(args.project_root).expanduser().resolve(), args.output_dir, args.batch_size, args.max_meal_examples)
