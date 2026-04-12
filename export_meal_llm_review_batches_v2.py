
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import json

import pandas as pd


def log(msg: str) -> None:
    print(f"[meal-db] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_done(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.eq("done")


def build_batches(
    project_root: Path,
    output_dir_name: str,
    batch_size: int,
    max_meal_examples: int,
    pending_only: bool,
    clear_existing: bool,
) -> None:
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

    if "llm_review_status" not in aliases.columns:
        aliases["llm_review_status"] = ""

    aliases["__is_done"] = is_done(aliases["llm_review_status"])

    total_all = len(aliases)
    total_done = int(aliases["__is_done"].sum())
    total_pending = int((~aliases["__is_done"]).sum())

    if pending_only:
        aliases = aliases.loc[~aliases["__is_done"]].copy()

    aliases = aliases.sort_values(
        ["entry_count", "meal_count", "days_seen", "normalized_name_clean"],
        ascending=[False, False, False, True],
    ).drop(columns=["__is_done"])

    total = len(aliases)
    if total == 0:
        raise ValueError("No aliases available for export with the current filter.")

    if clear_existing:
        for p in review_dir.glob("food_alias_batch_*.csv"):
            p.unlink()
        for p in review_dir.glob("food_alias_batch_*_components.csv"):
            p.unlink()
        for p in review_dir.glob("food_alias_batch_*_meal_examples.csv"):
            p.unlink()
        for p in review_dir.glob("review_manifest*.json"):
            p.unlink()

    batch_count = (total + batch_size - 1) // batch_size
    mode = "pending-only" if pending_only else "all-alias"
    log(f"Creating {batch_count} review batches from {total} aliases ({mode}).")
    log(f"Alias status summary: total={total_all}, done={total_done}, pending={total_pending}")

    manifests = []
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

        representative_meal_ids = []
        for alias_id, g in comp_batch.groupby("alias_id"):
            counts = g["meal_id"].value_counts().head(max_meal_examples)
            representative_meal_ids.extend(counts.index.tolist())
        representative_meal_ids = list(dict.fromkeys(representative_meal_ids))
        meal_batch = meals[meals["meal_id"].isin(representative_meal_ids)].copy()
        sort_cols = [c for c in ["date", "meal_order_in_day", "calories_kcal"] if c in meal_batch.columns]
        if sort_cols:
            meal_batch = meal_batch.sort_values(sort_cols, ascending=[True] * len(sort_cols))
        meal_batch.to_csv(review_dir / f"food_alias_batch_{batch_num:03d}_meal_examples.csv", index=False)

        manifests.append({
            "batch_num": batch_num,
            "alias_count": len(batch),
            "component_row_count": len(comp_batch),
            "meal_example_count": len(meal_batch),
            "top_aliases": batch["normalized_name_clean"].head(10).tolist(),
        })

    manifest = {
        "mode": mode,
        "total_aliases_all": total_all,
        "total_done": total_done,
        "total_pending": total_pending,
        "exported_aliases": total,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "max_meal_examples": max_meal_examples,
        "batches": manifests,
    }
    manifest_path = review_dir / f"review_manifest_{mode}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"Wrote review batches to: {review_dir}")
    log(f"Wrote manifest to: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split meal alias seed tables into review batches for LLM enrichment uploads.")
    parser.add_argument("--project-root", default=".", help="Path to the FoodAI project root containing meal_db/")
    parser.add_argument("--output-dir", default="meal_db", help="Meal DB folder name under the project root")
    parser.add_argument("--batch-size", type=int, default=150, help="Aliases per batch")
    parser.add_argument("--max-meal-examples", type=int, default=3, help="Representative meals per alias")
    parser.add_argument("--pending-only", action="store_true", help="Export only aliases not yet marked done")
    parser.add_argument("--clear-existing", action="store_true", help="Delete previously generated batch files before writing new ones")
    args = parser.parse_args()

    build_batches(
        Path(args.project_root).expanduser().resolve(),
        args.output_dir,
        args.batch_size,
        args.max_meal_examples,
        args.pending_only,
        args.clear_existing,
    )
