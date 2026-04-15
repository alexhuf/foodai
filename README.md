# FoodAI

Developer documentation set for the FoodAI repository.

**Purpose:** preserve the full project theory, data/model lineage, script chronology, current findings, and next-step logic in a form that does **not** depend on this chat surviving.

---

## 1. What FoodAI is

FoodAI is a **multi-resolution personal food / biology intelligence system** built from:

- **Noom behavior data**
  - food entries
  - meal events
  - weigh-ins
  - calorie budgets
  - actions, assignments, app usage
- **Samsung health data**
  - steps
  - exercise
  - energy expenditure
  - heart rate
  - stress
  - sleep
  - weight
  - related telemetry
- **Meal semantics**
  - canonical food entities
  - meal archetypes
  - components
  - protein / starch anchors
  - cuisine and meal structure where available
- **Weather / daylight context**
  - daily weather
  - 15-minute aligned context
  - daylight, seasonality, streak effects
- **Temporal structure**
  - meal-level states
  - day-level summaries
  - week-level regimes
  - weekend-level regimes
  - multi-resolution sequence views

The core idea is **not** merely “predict tomorrow’s weight.”

The project is trying to build a reusable personal intelligence layer that can do all of the following:

1. **Comparison**
   - compare one day, week, or weekend to another
   - find analogous states or meal contexts

2. **Explanation**
   - identify why one period worked and another failed
   - separate meal effects, biology effects, and context effects

3. **Forecasting**
   - predict next meal behavior
   - predict next-day direction
   - predict next-week regime shifts

4. **Recommendation**
   - choose the next meal or next-day pattern that best fits stability / enjoyment / novelty / health constraints

5. **Retrieval and memory**
   - recover similar meals, days, weeks, and regimes
   - use those as anchors for explanation and suggestion

6. **Temporal representation learning**
   - learn embeddings and sequence models that go beyond engineered statistical features

---

## 2. Theory of operation

FoodAI is built around a layered theory:

### 2.1 Raw behavior is too noisy to use directly
Food logs, health telemetry, and weather are messy, sparse, duplicated, and differently timed. The first phase of the project therefore focused on canonicalization and fusion.

### 2.2 Meals, days, and weeks are different “resolutions” of the same system
The same human system expresses itself at multiple time scales:
- meals capture acute decisions and immediate contexts
- days capture state accumulation
- weeks capture behavioral regimes
- weekends capture a special social/behavioral regime class

The project therefore builds representations and targets at all four scales.

### 2.3 Engineered anchors are required before neural models
Before running expensive neural sequence models, we needed to establish:
- the data is real
- the targets are leakage-safe
- simple baselines actually find signal
- the signal is stable enough to be worth scaling up

That is why the project spent significant time on:
- retrieval baselines
- representation encoders
- regime transition targets
- daily / weekly weight-direction branches
- calibration and historical band scoring

This was not wasted work. It established the project’s “basis of truth.”

### 2.4 Recommendation requires both discriminative and generative understanding
A good recommendation system cannot rely only on classification. It also needs:
- retrieval of similar prior states
- latent semantic structure of meals
- sensitivity to missing modalities
- awareness of regime transitions
- ability to generalize to adjacent but not identical options

The long-term architecture is therefore expected to combine:
- semantic meal space
- state/retrieval space
- transition models
- temporal encoders / transformers
- recommendation / ranking logic

---

## 3. How the project started

The initial project started from the question:

> Can a personal intelligence system learn from food, biology, and context data well enough to compare periods, explain outcomes, and eventually recommend better next actions?

The starting ingredients were:
- Samsung exports
- Noom exports
- the need to reconcile food semantics
- the need to distinguish meal-level and day-level logic
- the desire to build an auditable system, not a black box only

So the first phase focused on:
- choosing which raw files mattered
- canonicalizing them
- fusing them into stable daily / event / telemetry layers
- building a meal semantic database that could survive downstream modeling

---

## 4. What has been built

At a high level, the project now contains these major layers:

### Layer A — Raw and canonicalized source data
- `samsung/`
- `noom/`
- `canonical/`
- `fused/`

### Layer B — Meal semantic DB
- deterministic seeds
- review/export/apply loops
- final repaired semantic tables
- canonical meal timeline recovery

### Layer C — Environmental context
- daily and 15-minute weather/daylight context

### Layer D — Model-ready tables
- day feature matrix
- week summary matrix
- weekend summary matrix
- meal decision points
- meal prediction view
- daily transition matrix
- weekly regime transition matrices
- canonical meal timeline
- multi-resolution sequence dataset

### Layer E — Statistical / anchor models
- meal baselines
- retrieval baselines
- representation encoders
- regime transition models
- weekly weight-gain-focused branch
- daily weight-direction branch
- historical scorers and calibration layers

### Layer F — First temporal pilot
- multi-resolution temporal dataset
- GRU / TCN / transformer-capable trainer
- diagnostic ablation trainer
- first temporal smoke tests
- bounded flattened-path exploration over the same `days,weeks` loss target
- bounded flattened follow-up confirming extra trees as the preferred surviving flattened family

---

## 5. Current state in one paragraph

The project has **proven nontrivial signal exists** in multiple anchor branches, especially the daily weight-direction branch and the weekly regime branch. The meal modality was recovered correctly after rejecting an earlier telemetry-grid false positive. A full multi-resolution sequence dataset now exists, conservative flattened temporal baselines have set a strong `days,weeks` binary-loss floor, bounded GRU / TCN / transformer smoke comparisons have been run, and a later bounded path-exploration loop plus a direct follow-up pilot confirmed that stronger flattened tree models still beat the weak neural ceiling while `et_balanced` remains the only clearly preferred surviving flattened challenger. The operational winner-analysis and held-out checks still show an attractive candidate threshold zone around `0.44` to `0.455`, but a later additive six-fold fixed-threshold forward validation rejected promotion because those higher thresholds reduced false positives only by giving back too much recall and time-aware balanced accuracy versus the locked `0.4288` threshold. A subsequent focal-loss GRU/TCN smoke probe also failed to improve ranking or probability dispersion enough to change that picture, so GRU/TCN sequence work is now effectively frozen behind the flattened ET winner unless a new setup clearly beats the current floor. The current blocker is therefore **not data availability** and no longer primarily threshold promotion, but **temporal model performance and training design**: neural temporal runs remain well below both the anchor branches and the current simple temporal floor, and even the best flattened follow-up still does not clear that floor. The project is therefore in the stage:

> **diagnose which target / modality / architecture combination is genuinely learnable in temporal form before committing to long multi-hour or multi-day neural runs.**

---

## 6. Documentation map

This documentation set is split across multiple files on purpose.

- `README.md`
  - this file
  - project theory, current state, roadmap, top-level structure

- `DEVELOPER_GUIDE.md`
  - conventions, operating model, data contracts, how to reason about the system

- `SCRIPT_CATALOG.md`
  - chronological and grouped documentation for every important script
  - what each script reads, writes, why it exists, and what consumed its outputs next

- `PROJECT_HISTORY.md`
  - narrative story of the project’s phases and major pivots

- `NEXT_DEVELOPER_HANDOFF.md`
  - current findings, trusted artifacts, current blockers, exact next steps

- `generate_repo_inventory.py`
  - self-documenting utility
  - generates a tree and JSON manifest of the current repo

---

## 7. High-level repo structure

The exact live repo should be inventoried by running `generate_repo_inventory.py`, but conceptually the project looks like this:

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
    final/
    final_repaired/

  weather/

  training/
    predictive_views/
    targets/
    daily_transition/
    regime_transition/
    meal_sequence_source/
    meal_timeline_canonical/
    multires_sequence_dataset/

  models/
    retrieval/
    representation/
    regime_transition/
    daily_transition/
    temporal_multires/

  reports/
    backtests/
    analysis/
    feature_importance/
    scoring/
    diagnostics/
```

---

## 8. What has actually worked

The most important result so far is that the project is **not just fitting noise**.

### Proven anchor branches
1. **Daily weight-direction branch**
   - daily next-day gain / loss signal exists
   - historical band scoring showed strong separation between high / watch / low regimes
   - this is one of the strongest validated branches so far

2. **Weekly regime / weight-gain-focused branch**
   - weekly summaries and regime targets produced useful anchor behavior
   - this branch became a useful “macro” counterpart to the daily branch

3. **Retrieval / representation work**
   - useful for establishing whether states, days, and weeks have meaningful similarity structure
   - created a basis for comparison, not just classification

### What did *not* work yet
1. **First temporal GRU pilots**
   - became numerically stable after v3 fixes
   - but did not beat anchored baselines
   - v4.1 ablations show weak or collapsed class separation on the tested configs

This is a key finding. The neural phase is now an optimization / model-selection problem, not a data-ingestion problem.

---

## 9. Where the project is going

The intended direction remains:

1. keep the **anchor/statistical system** as a trustworthy baseline
2. identify a **temporal architecture** that can beat those anchors on at least one task
3. move from prediction-only to:
   - retrieval-assisted explanation
   - recommendation
   - meal suggestion / ranking
   - eventually novel-but-adjacent meal generation or discovery

The likely future architecture is not a single monolithic model. It is more likely to be a layered system that combines:
- semantic meal space
- daily / weekly anchors
- retrieval
- temporal encoders
- recommendation logic

---

## 10. Immediate next steps

The immediate next steps are **not** “run longer at all costs.”

They are:

1. finish temporal **diagnostic ablations**
   - single-target
   - specific modality mixes
   - compare GRU / TCN / transformer
   - possibly simpler sequence-to-MLP baselines too

2. identify whether the temporal signal is strongest for:
   - `y_next_weight_loss_flag`
   - `y_next_weight_gain_flag`
   - `y_next_weight_delta_lb`

3. only then launch a **longer GPU run** on the best target/modality/architecture combination

See `NEXT_DEVELOPER_HANDOFF.md` for the exact recommended order.

---

## 11. How to use this documentation

### If you are a new developer
Read in this order:

1. `README.md`
2. `NEXT_DEVELOPER_HANDOFF.md`
3. `SCRIPT_CATALOG.md`
4. `DEVELOPER_GUIDE.md`

### If you need the current live repo structure
Run:

```bash
python generate_repo_inventory.py --root . --output-dir docs/repo_inventory
```

### If you need the project story
Read:
- `PROJECT_HISTORY.md`

---

## 12. Final note

This documentation set is deliberately redundant in places.

That is intentional.

The goal is not elegance.  
The goal is that a future developer can lose this chat entirely and still recover:
- what FoodAI is
- why each script exists
- which outputs matter
- which results are trustworthy
- and what should happen next
