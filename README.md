# FoodAI Project README

## Purpose

This project builds a **multi-scale personal food/biology intelligence system** from:

- **Noom behavioral data**: food entries, meal events, weigh-ins, budgets, app engagement
- **Samsung health data**: activity, steps, exercise, energy expenditure, heart rate, stress, weight, limited sleep
- **Derived meal semantics**: canonical food entities, meal archetypes, cuisines, service forms, component roles
- **Environmental context**: weather, daylight, darkness, seasonal and streak effects
- **Fused temporal context**: daily summaries plus 15-minute telemetry-aligned state

The long-term goal is not just prediction. It is:

1. **Comparison**  
   Example: compare one day/week/weekend to another.

2. **Explanation**  
   Example: why did one week work while another failed under similar intentions?

3. **Forecasting**  
   Example: what is the likely next meal type or calorie load in the current state?

4. **Recommendation**  
   Example: what is the best next meal for:
   - stability
   - health
   - enjoyment
   - novelty
   - low weight-gain risk

5. **Emergent meal awareness**  
   Example: suggest not only historically common meals, but also **novel but adjacent** meals and restaurants that fit the same latent need.

---

# 1. Current project state

The project currently has:

- raw source folders:
  - `samsung/`
  - `noom/`

- canonical/fused source-derived layers:
  - `canonical/`
  - `fused/`

- a repaired semantic meal DB:
  - `meal_db/final_repaired/`

- corrected weather/daylight context:
  - `weather/`

- model-ready aggregate and decision datasets:
  - `training/day_feature_matrix.csv`
  - `training/week_summary_matrix.csv`
  - `training/weekend_summary_matrix.csv`
  - `training/meal_decision_points.csv`
  - `training/predictive_views/meal_prediction_view.csv`

This means the project is **past data acquisition and semantic curation** and is now at the boundary between:
- final dataset QA
- leakage-safe target design
- first baseline modeling

---

# 2. Folder structure

Recommended structure:

```text
foodai/
  samsung/
  noom/

  canonical/
    samsung/
    noom/

  fused/
    master_daily_features.csv
    master_daily_features_full.csv
    master_event_ledger.csv
    master_15min_telemetry_active.csv

  meal_db/
    seed/
    current/
    reviews/
      batch_001/
      batch_002/
      batch_003/
      batch_004/
      pending_001/
      pending_002/
      pending_003/
      pending_004/
    final/
    final_repaired/
      canonical_food_entity.csv
      meal_component_edge.csv
      meal_semantic_features.csv
      food_entry_semantic_view.csv
      meal_db_final_manifest.json

  weather/
    weather_hourly_raw.csv
    weather_daily_raw.csv
    weather_context_15min.csv
    weather_context_daily.csv
    weather_context_manifest.json

  training/
    day_feature_matrix.csv
    day_feature_matrix_manifest.json
    week_summary_matrix.csv
    week_summary_matrix_manifest.json
    weekend_summary_matrix.csv
    weekend_summary_matrix_manifest.json
    meal_decision_points.csv
    meal_decision_points_manifest.json

    predictive_views/
      meal_prediction_view.csv
      meal_prediction_view_manifest.json

    targets/
      target_spec_meal_prediction.json

  models/
    baselines/
    retrieval/
    sequence/
    regime/
    ranker/

  reports/
    backtests/
    diagnostics/
    feature_importance/
    retrieval_eval/
    calibration/

  configs/
    model_catalog.yaml
    feature_policies.yaml
```

---

# 3. Script catalog

This section lists every script created during the project so far, what it does, and whether it is still active.

## 3.1 Raw canonical/fused build scripts

### `build_foodai_project.py`
**Role:** original end-to-end raw-source build from Samsung + Noom into canonical/fused outputs.  
**Status:** historical / superseded.

Reads:
- `samsung/`
- `noom/`

Writes:
- `canonical/samsung/`
- `canonical/noom/`
- `fused/`

Why superseded:
- later fixed to improve active-window logic and Noom calorie-budget parsing.

---

### `build_foodai_project_v2.py`
**Role:** improved raw-source build.  
**Status:** current preferred historical source-builder if a full rebuild from raw is needed.

Main fixes:
- fixed Noom `daily_calorie_budgets.csv` parsing
- made `master_daily_features.csv` active-window focused
- preserved a full-range daily table separately
- trimmed fused outputs to the intended active era

Reads:
- `samsung/`
- `noom/`

Writes:
- `canonical/samsung/`
- `canonical/noom/`
- `fused/`

Key outputs:
- `fused/master_daily_features.csv`
- `fused/master_daily_features_full.csv`
- `fused/master_event_ledger.csv`
- `fused/master_15min_telemetry_active.csv`

---

## 3.2 Meal DB seed and review scripts

### `build_meal_db_seed.py`
**Role:** create deterministic seed tables for meal ontology construction from the trusted canonical/fused base.  
**Status:** active historical build step; not needed routinely once seed is built.

Reads:
- `canonical/noom/noom_food_entries.csv`
- `canonical/noom/noom_meal_events.csv`
- fused/current trusted meal-related outputs

Writes:
- `meal_db/seed/raw_food_entry_enriched_seed.csv`
- `meal_db/seed/food_alias_seed.csv`
- `meal_db/seed/canonical_food_entity_seed.csv`
- `meal_db/seed/meal_component_seed.csv`
- `meal_db/seed/meal_event_seed.csv`
- `meal_db/seed/seed_manifest.json`

Purpose:
- stable IDs
- alias grouping
- name normalization
- brand/restaurant hints
- meal/component links
- meal-event scaffold

This stage is **deterministic** and intentionally avoids speculative meal semantics.

---

### `export_meal_llm_review_batches.py`
**Role:** original batch exporter for alias review.  
**Status:** superseded.

Problem:
- exported already-reviewed aliases again.

---

### `export_meal_llm_review_batches_v2.py`
**Role:** export **pending-only** LLM review batches.  
**Status:** active for any future review workflow.

Reads:
- `meal_db/seed/food_alias_seed.csv`

Writes:
- `meal_db/review_batches/food_alias_batch_001.csv` etc.
- component and meal-example companion files
- `review_manifest_pending-only.json`

Purpose:
- stage the next unresolved alias batches for semantic review.

---

### `apply_meal_llm_review_batch.py`
**Role:** original apply script for reviewed batches.  
**Status:** superseded.

Problem:
- dtype issue when writing strings into numeric-inferred columns
- moved files in a way that complicated recovery after failure

---

### `apply_meal_llm_review_batch_v2.py`
**Role:** apply reviewed alias/entity/component/meal-example batches back into seed tables.  
**Status:** active.

Reads:
- `meal_db/reviews/<batch_label>/...`

Writes:
- updated `meal_db/seed/`
- snapshots to `meal_db/current/`
- archives prior seed state to `meal_db/_archive/`

Purpose:
- materialize LLM-reviewed semantics back into the working meal DB.

---

## 3.3 Meal DB finalization scripts

### `build_meal_db_final.py`
**Role:** first attempt to finalize meal DB.  
**Status:** superseded.

Problem:
- duplicate label merge bug
- canonical entity table under-materialized from review results

---

### `build_meal_db_final_v2.py`
**Role:** second attempt to finalize seed/current into stable outputs.  
**Status:** superseded by repaired finalization.

Improved:
- merge stability
- duplicate-column handling

But still not ideal because:
- canonical entity and meal semantic propagation were still incomplete.

---

### `build_meal_db_final_repaired.py`
**Role:** first repair pass rebuilding final outputs from reviewed batch files + seed/current bases.  
**Status:** superseded by v2 repair.

Problem:
- failed when combining into missing event columns.

---

### `build_meal_db_final_repaired_v2.py`
**Role:** **current preferred finalization** for meal DB.  
**Status:** current.

Reads:
- reviewed batch files in `meal_db/reviews/`
- base tables in `meal_db/current/` and `meal_db/seed/`

Writes:
- `meal_db/final_repaired/canonical_food_entity.csv`
- `meal_db/final_repaired/meal_component_edge.csv`
- `meal_db/final_repaired/meal_semantic_features.csv`
- `meal_db/final_repaired/food_entry_semantic_view.csv`
- `meal_db/final_repaired/meal_db_final_manifest.json`

This is the meal DB to use for modeling.

---

## 3.4 Weather/daylight scripts

### `build_weather_context.py`
**Role:** first historical weather/daylight context build using Open-Meteo.  
**Status:** superseded.

Problem:
- solar/daylight fields were effectively shifted by about +1 hour.

---

### `build_weather_context_v2.py`
**Role:** corrected weather/daylight build.  
**Status:** current.

Reads:
- `fused/master_daily_features.csv`
- `fused/master_15min_telemetry_active.csv`

Writes:
- `weather/weather_hourly_raw.csv`
- `weather/weather_daily_raw.csv`
- `weather/weather_context_15min.csv`
- `weather/weather_context_daily.csv`
- `weather/weather_context_manifest.json`

Key logic:
- Open-Meteo historical archive pull
- daily + 15-minute alignment
- corrected sunrise/sunset / `is_day`
- behavioral weather features and streaks

---

## 3.5 Model-ready training dataset scripts

### `build_day_feature_matrix.py`
**Role:** create one-row-per-day model-ready dataset.  
**Status:** current.

Reads:
- `fused/master_daily_features.csv`
- `meal_db/final_repaired/meal_semantic_features.csv`
- `weather/weather_context_daily.csv`

Writes:
- `training/day_feature_matrix.csv`
- `training/day_feature_matrix_manifest.json`

Purpose:
- daily state modeling
- daily similarity / clustering
- weekly/weekend aggregation source
- explanatory comparisons

---

### `build_week_summary_matrix.py`
**Role:** aggregate the day matrix into Monday-based weeks.  
**Status:** current.

Reads:
- `training/day_feature_matrix.csv`

Writes:
- `training/week_summary_matrix.csv`
- `training/week_summary_matrix_manifest.json`

Purpose:
- week vs week comparison
- weekly regime / retrieval modeling
- weekly forecasting targets

---

### `build_weekend_summary_matrix.py`
**Role:** aggregate the day matrix into Friday–Saturday–Sunday weekend blocks.  
**Status:** current.

Reads:
- `training/day_feature_matrix.csv`

Writes:
- `training/weekend_summary_matrix.csv`
- `training/weekend_summary_matrix_manifest.json`

Purpose:
- weekend vs weekend comparison
- drift analysis
- weekend forecasting / retrieval

---

### `build_meal_decision_points.py`
**Role:** build one-row-per-meal decision dataset.  
**Status:** current.

Reads:
- `meal_db/final_repaired/meal_semantic_features.csv`
- `meal_db/final_repaired/meal_component_edge.csv`
- `training/day_feature_matrix.csv`
- `fused/master_15min_telemetry_active.csv`
- `weather/weather_context_15min.csv`

Writes:
- `training/meal_decision_points.csv`
- `training/meal_decision_points_manifest.json`

Purpose:
- recommendation
- next-meal modeling
- meal ranking
- pre-meal state + target meal pairing

---

### `build_meal_prediction_view.py`
**Role:** create the first leakage-safe predictive view from meal decision points.  
**Status:** current.

Reads:
- `training/meal_decision_points.csv`

Writes:
- `training/predictive_views/meal_prediction_view.csv`
- `training/predictive_views/meal_prediction_view_manifest.json`
- `training/targets/target_spec_meal_prediction.json`

Purpose:
- exclude likely leaky post-hoc full-day features
- define baseline classification and regression targets
- provide the first clean training table for next-meal modeling

---

# 4. Data-prep logic, stage by stage

This section explains the full logic of the data prep stack.

## Stage A — Raw data triage

Goal:
- choose only the high-value Samsung and Noom files.

Samsung selected core layers:
- heart rate
- stress
- sleep summary
- sleep stages
- weight
- exercise
- calories burned details
- pedometer day summary
- pedometer step count
- activity day summary
- oxygen saturation kept as secondary enrichment

Noom selected core layers:
- food entries
- actions
- daily calorie budgets
- finish day
- assignments
- goals
- curriculum state
- app opens
- supporting user/app event context as needed

This stage was about **source sufficiency and scope discipline**.

---

## Stage B — Canonicalization and fusion

Goal:
- convert Samsung and Noom raw exports into clean source-specific canonical tables and then fused tables.

Key principles:
- keep provenance
- do not merge too early
- preserve source semantics
- distinguish daily summaries from event tables
- build a 15-minute active telemetry layer for temporal context

Outputs:
- canonical Samsung tables
- canonical Noom tables
- fused daily table
- fused event ledger
- fused 15-minute telemetry

Logic:
- Samsung provides physiological and activity context
- Noom provides behavioral and meal-logging context
- fused tables align them in shared time

---

## Stage C — Meal ontology seed generation

Goal:
- create a deterministic scaffold for meal semantics without hallucinating food meaning.

Why:
- raw Noom food strings are too noisy
- the same meal can be logged as:
  - an exact restaurant item
  - a generic stand-in
  - a partial component
- meal understanding requires:
  - alias grouping
  - canonical entities
  - component roles
  - meal-event structure

The seed builder does:
- name normalization
- alias grouping
- stable IDs
- component links
- meal links
- brand/restaurant hints

It does **not** attempt rich semantics by itself.

---

## Stage D — Batch semantic review

Goal:
- use LLM semantic review to fill in:
  - canonical entity meaning
  - cuisine
  - service form
  - dish family
  - meal archetypes
  - component roles
  - principal protein/starch/veg/fat
  - restaurant specificity vs generic stand-in

Process:
1. export alias batches
2. review alias/entity/component/example files
3. apply them back to the seed
4. repeat until all aliases are reviewed

This is where the project gained:
- emergent meal awareness
- dish/component separation
- semantic hierarchy

---

## Stage E — Final meal DB repair and finalization

Goal:
- rebuild the final ontology from reviewed ground truth rather than stale placeholders.

Outputs:
- `canonical_food_entity.csv`
- `meal_component_edge.csv`
- `meal_semantic_features.csv`
- `food_entry_semantic_view.csv`

These files are now the **semantic meal core** of the project.

---

## Stage F — Weather/daylight context

Goal:
- add exogenous environmental context.

Why:
- weather and daylight were identified as meaningful behavior drivers
- dark early, snow, cold, gloom, heat, etc. can affect meal preference, activity, and behavior

Source:
- Open-Meteo historical archive

Derived features:
- temperature bands
- precipitation/rain/snow flags
- gloom/cloudiness
- early-dark flags
- daylight hours
- streaks (rain, snow, gloom, hot, cold, dark early)

Outputs:
- daily weather context
- 15-minute weather context aligned to the fused timeline

---

## Stage G — Multi-resolution training datasets

Goal:
- build model-ready views at several time scales.

Outputs:
- day feature matrix
- week summary matrix
- weekend summary matrix
- meal decision points
- leakage-safe meal prediction view

This preserves **granularity** instead of destroying it.

The project now has:
- meal-to-meal resolution
- 15-minute context resolution
- day-level resolution
- week-level resolution
- weekend-level resolution

That is deliberate and foundational.

---

# 5. Current major outputs and how to use them

## `meal_db/final_repaired/canonical_food_entity.csv`
One row per canonical food entity.

Use for:
- ontology
- candidate meal semantics
- ingredient/dish/service-form understanding
- recommendation candidate normalization

---

## `meal_db/final_repaired/meal_component_edge.csv`
One row per food-entry-to-meal-role link.

Use for:
- meal decomposition
- component-role analysis
- explanation and rationalization
- recommendation generation

---

## `meal_db/final_repaired/meal_semantic_features.csv`
One row per meal event.

Use for:
- meal retrieval
- semantic meal comparisons
- daily rollups
- recommendation target metadata

---

## `training/day_feature_matrix.csv`
One row per day.

Use for:
- day vs day comparisons
- daily clustering / regime discovery
- daily forecasting
- source for week/weekend aggregation

---

## `training/week_summary_matrix.csv`
One row per Monday-based week.

Use for:
- week vs week comparison
- weekly retrieval / clustering
- weekly forecasting targets
- “trying but gained” vs “not trying but lost”

---

## `training/weekend_summary_matrix.csv`
One row per Friday–Saturday–Sunday weekend.

Use for:
- weekend vs weekend comparison
- weekend drift analysis
- weekend retrieval / forecasting

---

## `training/meal_decision_points.csv`
One row per meal decision point.

Use for:
- recommendation
- meal ranking
- next meal prediction
- state-before-meal modeling

Important:
- useful for analysis and candidate generation
- but not automatically safe for predictive training until leakage filtering

---

## `training/predictive_views/meal_prediction_view.csv`
Leakage-safe predictive view.

Use for:
- first baseline next-meal models
- target-spec driven training
- walk-forward evaluation

---

# 6. Training logic overview

This project should **not** use one monolithic model.

It should use a layered stack:

1. **retrieval**
2. **tabular baselines**
3. **regime discovery**
4. **sequence models**
5. **recommendation ranker**
6. **LLM explanation layer**

The reason is simple:

- retrieval gives interpretable historical analogs
- baselines give robust predictive signals
- regime models identify hidden states
- sequence models capture temporal dynamics
- rankers solve recommendation tradeoffs
- the LLM translates outputs into reasoning and guidance

---

# 7. Model catalog

## Retrieval models

### R1 — Meal retrieval
Dataset:
- `meal_db/final_repaired/meal_semantic_features.csv`
- optionally `meal_component_edge.csv`

Purpose:
- find historically similar meals
- support recommendation candidates
- support meal explanation

Approach:
- semantic embeddings
- cosine similarity
- nearest-neighbor retrieval

Output:
- similar meals
- similar component structures
- nearest adjacent meals for novelty

---

### R2 — Day retrieval
Dataset:
- `training/day_feature_matrix.csv`

Purpose:
- compare one day to another
- retrieve historically similar days
- analyze why one day diverged

Approach:
- embeddings or weighted feature similarity

Output:
- similar days
- matched counterexamples

---

### R3 — Week retrieval
Dataset:
- `training/week_summary_matrix.csv`

Purpose:
- compare one week to another
- find weeks with similar structure but different outcomes

Output:
- similar weeks
- “trying but gained” analogs
- “not trying but lost” analogs

---

### R4 — Weekend retrieval
Dataset:
- `training/weekend_summary_matrix.csv`

Purpose:
- compare one weekend to another
- find weekend drift or recovery analogs

---

## Regime models

### G1 — Daily regime discovery
Dataset:
- `training/day_feature_matrix.csv`

Purpose:
- identify hidden states like:
  - controlled
  - drift
  - indulgent
  - active high-control
  - dark/cold comfort-seeking
  - fatigue/recovery

Approaches:
- clustering
- HMM
- change-point models
- embedding + clustering

Output:
- latent day-state labels
- transition structure

---

### G2 — Weekly regime discovery
Dataset:
- `training/week_summary_matrix.csv`

Purpose:
- identify week-level behavioral regimes and transitions

Output:
- latent weekly states
- trajectory classification

---

## Meal prediction models

### M1 — Next meal archetype
Dataset:
- `training/predictive_views/meal_prediction_view.csv`

Target:
- `y_next_meal_archetype_collapsed`

Type:
- multiclass classification

Purpose:
- predict likely next meal family
- support candidate generation and top-k ranking

---

### M2 — Next meal calories
Dataset:
- `training/predictive_views/meal_prediction_view.csv`

Target:
- `y_next_meal_kcal_log1p`

Type:
- regression

Purpose:
- predict likely next-meal energy load
- support risk estimation and ranking

---

### M3 — Next meal macros
Dataset:
- `training/predictive_views/meal_prediction_view.csv`

Targets:
- `target_protein_g`
- `target_carbs_g`
- `target_fat_g`

Type:
- regression or multi-output regression

Purpose:
- estimate composition of likely next meal

---

### M4 — Restaurant-vs-non-restaurant next meal
Dataset:
- `training/predictive_views/meal_prediction_view.csv`

Target:
- `y_next_restaurant_meal`

Type:
- binary classification

Purpose:
- detect restaurant-seeking state

---

### M5 — Immediate budget-risk model
Dataset:
- `training/predictive_views/meal_prediction_view.csv`

Target:
- `y_post_meal_budget_breach`

Type:
- binary classification

Purpose:
- estimate immediate post-meal risk

---

### M6 — Post-meal stability proxy
Dataset:
- `training/predictive_views/meal_prediction_view.csv`

Targets:
- `y_stability_proxy_v1`
- later more refined labels

Type:
- regression / ranking

Purpose:
- estimate whether a meal sets you up for success vs rebound

---

### M7 — Enjoyment proxy
Dataset:
- `training/predictive_views/meal_prediction_view.csv`

Target:
- `y_enjoyment_proxy_v1`

Type:
- regression / ranking

Purpose:
- estimate likely meal satisfaction

---

## Day models

### D1 — Daily outcome model
Dataset:
- future `day_forecast_view.csv`

Targets:
- next-day weight delta
- budget breach
- daily control state

Purpose:
- forecast day-level outcomes before the day unfolds

---

## Week models

### W1 — Weekly weight-delta model
Dataset:
- future `week_forecast_view.csv`

Target:
- `weight_delta_lb`

Type:
- regression

Purpose:
- forecast weekly outcome

---

### W2 — Weekly success-class model
Dataset:
- future `week_forecast_view.csv`

Target:
- weekly categorical success/failure label

Purpose:
- answer questions like:
  - trying but gained
  - not trying but lost
  - maintain
  - drift

---

## Weekend models

### WE1 — Weekend drift-risk model
Dataset:
- future `weekend_forecast_view.csv`

Target:
- weekend drift risk

Purpose:
- estimate whether a coming weekend is likely to create instability

---

## Ranking / recommendation models

### RR1 — Multi-objective next-meal ranker
Inputs:
- current state
- candidate meals
- outputs of M1/M2/M3/M4/M5/M6/M7
- retrieval neighbors

Outputs:
- ranked meal list for different objectives:
  - safest
  - most enjoyable safe
  - healthiest practical
  - novel but adjacent

This is the final recommendation engine.

---

# 8. Training styles and approaches

## 8.1 Tabular baselines

These should come first.

Algorithms:
- XGBoost / LightGBM / CatBoost
- logistic regression baselines
- elastic net regressors
- random forest sanity checks

Why:
- work well on small/medium structured data
- interpretable
- fast to iterate
- strong baselines before deep learning

Use for:
- M1
- M2
- M3
- M4
- M5
- W1
- W2
- WE1

---

## 8.2 Retrieval embeddings

These can start simple and get more sophisticated.

Approaches:
- standardized numeric feature space + cosine similarity
- PCA/UMAP embedding for exploration
- later self-supervised learned embeddings

Use for:
- similar meals
- similar days
- similar weeks
- similar weekends

These are essential for:
- contextualization
- rationalization
- nearest-neighbor explanations

---

## 8.3 Regime discovery

Approaches:
- k-means / GMM on embeddings
- HMM
- change-point analysis
- temporal clustering

Use for:
- hidden states
- transitions
- higher-level behavioral interpretation

---

## 8.4 Sequence models

These are for later, after tabular baselines.

Approaches:
- GRU / LSTM
- TCN
- modest Transformer encoder
- sequence autoencoder
- contrastive state encoder

Use for:
- meal-to-meal transitions
- intraday dynamics
- next-meal forecasting
- latent state encoding

GPU:
- this is where the RTX 5070 Ti becomes valuable

Caution:
- do not start here
- validate against tabular baselines first

---

## 8.5 Ranking / recommendation

Approaches:
- pointwise scorer
- pairwise ranker
- listwise ranker
- multi-objective weighted scoring

Example score:

\[
S(m \mid x) =
w_1 \cdot \hat E[\text{enjoyment}]
+ w_2 \cdot \hat E[\text{stability}]
+ w_3 \cdot \hat E[\text{health fit}]
- w_4 \cdot P(\text{budget breach})
- w_5 \cdot P(\text{7d drift risk})
+ w_6 \cdot \text{novelty bonus}
\]

where:
- \(m\) is a candidate meal
- \(x\) is the current state

This supports:
- safe familiar meals
- enjoyable safe meals
- novel but adjacent meals
- restaurant suggestions

---

# 9. Leakage policy

This is critical.

## Descriptive datasets
These are allowed to use full retrospective information:
- `day_feature_matrix.csv`
- `week_summary_matrix.csv`
- `weekend_summary_matrix.csv`

Use them for:
- explanation
- comparison
- retrieval
- retrospective pattern analysis

## Predictive datasets
These must use only information known before the prediction horizon.

### Meal prediction
Use:
- `training/predictive_views/meal_prediction_view.csv`

Input policy:
- only safe `state_*` features
- safe decision metadata
- no `target_*` or `outcome_*` as inputs

Dropped because they were considered likely leaky:
- full-day steps totals
- full-day day-semantic dominant summaries
- full-day meal totals
- end-of-day style features

Important note:
the first `meal_prediction_view` is mostly good, but the trainer should additionally exclude:
- `is_last_meal_of_day`
- `hours_until_next_meal`
- `day_meal_count`

These are present for analysis, but they are future-known at decision time.

---

# 10. Target specification

## Classification targets

### `y_next_meal_archetype_collapsed`
Use first.

Reason:
- better class balance
- robust first target
- good for top-k candidate generation

---

### `y_next_restaurant_meal`
Binary target.

Reason:
- captures a key real-world behavior mode

---

### `y_post_meal_budget_breach`
Binary target.

Definition:
- whether remaining budget after meal falls below 0

Reason:
- strong short-horizon risk proxy

---

### `y_post_meal_budget_breach_200`
Stricter version.

Definition:
- remaining budget after meal < -200 kcal

Reason:
- more severe breach

---

## Regression targets

### `y_next_meal_kcal_log1p`
First regression target.

Reason:
- stabilizes scale
- usually easier to model than raw kcal

---

### `target_protein_g`
### `target_carbs_g`
### `target_fat_g`
Macro prediction targets.

Reason:
- meal composition understanding
- recommendation tradeoffs

---

### `target_comfort_food_score`
### `target_fresh_light_score`
### `target_indulgence_score`
Semantic trait targets.

Reason:
- helps the system reason about meal style, not just nutrients

---

### `y_enjoyment_proxy_v1`
Version-1 enjoyment proxy.

Reason:
- useful later for ranking
- not the first target to optimize

---

### `y_stability_proxy_v1`
Version-1 stability proxy.

Reason:
- useful later for recommendation ranking
- should come after basic predictive baselines work

---

# 11. Recommended training order

## Phase 1 — first baseline trainer
Train on `meal_prediction_view.csv`:

Classification:
- `y_next_meal_archetype_collapsed`
- `y_next_restaurant_meal`
- `y_post_meal_budget_breach`

Regression:
- `y_next_meal_kcal_log1p`
- `target_protein_g`
- `target_carbs_g`
- `target_fat_g`

Use:
- walk-forward splits
- tabular baselines
- feature importance
- probability calibration

---

## Phase 2 — retrieval
Build:
- meal retrieval
- day retrieval
- week retrieval
- weekend retrieval

These should power explanation and candidate generation.

---

## Phase 3 — regime models
Use:
- day matrix
- week matrix

Goal:
- identify latent states and transitions

---

## Phase 4 — recommendation ranker
Use:
- outputs of meal models
- retrieval neighbors
- semantic candidate meals

Goal:
- rank meals for multiple user objectives

---

## Phase 5 — GPU sequence models
Only after the tabular baselines are solid.

Use:
- meal-to-meal sequences
- 15-minute windows
- learned embeddings

---

# 12. Explanation logic

Every comparison or recommendation should be rationalized using:

1. **retrieval evidence**  
   “this resembles these historical meals/days/weeks”

2. **feature deltas**  
   “what differs from matched counterexamples”

3. **model outputs**  
   probability, risk, predicted meal form, predicted kcal, etc.

4. **semantic decomposition**  
   principal protein, starch, service form, indulgence profile, restaurant intensity, etc.

The LLM layer should summarize these, not invent them.

---

# 13. Known caveats

- Some weekly/weekend column names are verbose and double-aggregated (`..._sum_sum`). This is not fatal, but future cleanup is recommended.
- Samsung sleep coverage is limited because the watch was rarely worn overnight.
- Samsung physiological intraday context is partial, not continuous for all meals.
- Enjoyment and stability targets are still proxy targets, not direct labels.
- The first predictive view still contains a few analysis-only fields that should be excluded by the trainer (`is_last_meal_of_day`, `hours_until_next_meal`, `day_meal_count`).

---

# 14. Recommended next file to build

The next script after this README should be:

### `train_meal_baselines.py`

It should:
- read `training/predictive_views/meal_prediction_view.csv`
- read `training/targets/target_spec_meal_prediction.json`
- build walk-forward splits
- exclude the last few still-borderline leaky input columns
- train the first baseline models
- write:
  - metrics
  - feature importances
  - calibration summaries
  - saved models

Suggested first tasks:
- next meal archetype classifier
- next meal kcal regressor
- restaurant-vs-non-restaurant classifier
- post-meal budget breach classifier

---

# 15. Short summary

The project now has:

- a repaired semantic meal database
- corrected weather/daylight context
- daily, weekly, weekend, and meal-decision training datasets
- a leakage-aware predictive meal view
- a clear multi-model architecture

The modeling strategy is:

- **retrieval for analogs**
- **tabular baselines for first predictive tasks**
- **regime discovery for latent state understanding**
- **sequence models later for richer temporal dynamics**
- **rankers for recommendation**
- **LLM on top for explanation**

This is the correct structure for:
- retrospective insight
- contextual explanation
- future guidance
- emergent meal awareness
- multi-dimensional reasoning across time, biology, meal structure, and environment
