# Meal DB seed build: first run

## What this does
This creates a new `meal_db/` folder from the **current trusted** project outputs only:
- `canonical/noom/noom_food_entries.csv`
- `canonical/noom/noom_meal_events.csv`
- `fused/master_daily_features.csv`

It does **not** use the older legacy meal files.

## Files to create in your project root
Create these two files in the project root:
- `build_meal_db_seed.py`
- `export_meal_llm_review_batches.py`

## Run order
1. Run the seed build:
   - `python build_meal_db_seed.py --project-root .`
2. Run the batch exporter:
   - `python export_meal_llm_review_batches.py --project-root . --batch-size 150`

## What should appear afterward
A new folder:
- `meal_db/`
  - `seed/`
  - `review_batches/`

## Main outputs
Inside `meal_db/seed/`:
- `raw_food_entry_enriched_seed.csv`
- `food_alias_seed.csv`
- `canonical_food_entity_seed.csv`
- `meal_component_seed.csv`
- `meal_event_seed.csv`
- `seed_manifest.json`

Inside `meal_db/review_batches/`:
- `food_alias_batch_001.csv`
- `food_alias_batch_001_components.csv`
- `food_alias_batch_001_meal_examples.csv`
- more batches if needed

## First upload back to ChatGPT
Upload these three first:
1. `meal_db/seed/food_alias_seed.csv`
2. `meal_db/seed/canonical_food_entity_seed.csv`
3. `meal_db/seed/meal_event_seed.csv`

If the alias file is too large, upload instead:
1. `meal_db/review_batches/food_alias_batch_001.csv`
2. `meal_db/review_batches/food_alias_batch_001_meal_examples.csv`
3. `meal_db/seed/canonical_food_entity_seed.csv`
4. `meal_db/seed/meal_event_seed.csv`

## Do not delete
Keep these folders as-is:
- `noom/`
- `samsung/`
- `canonical/`
- `fused/`

The seed build reads from them and writes a separate `meal_db/` folder.
