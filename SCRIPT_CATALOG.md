# FoodAI Script Catalog

This file documents the important scripts created across the project.

It is organized in approximate creation order and grouped by project phase.

For each script, the goal is to answer:
- why it exists
- what it reads
- what it writes
- which later stage used those outputs
- whether it is current or superseded

---

## Legend

- **current** — preferred script for this job
- **active** — still useful / still used
- **historical** — informative but not usually rerun
- **superseded** — replaced by a later version
- **diagnostic** — analysis / inspection / reporting rather than core production
- **experimental** — useful for iteration but not canonical

---

# 1. Source build and canonical fusion

## `build_foodai_project.py`
**Status:** historical / superseded  
**Purpose:** first end-to-end raw-source build from Samsung + Noom into canonical and fused outputs.

**Reads**
- `samsung/`
- `noom/`

**Writes**
- `canonical/samsung/*`
- `canonical/noom/*`
- `fused/*`

**Why it mattered**
- established the first reliable fused daily/event layer

**Why superseded**
- later fixes improved active-window logic and Noom budget parsing

**Downstream consumers**
- later semantic meal builders
- fused daily/event tables
- weather/day builders

---

## `build_foodai_project_v2.py`
**Status:** current historical source builder  
**Purpose:** corrected raw-source builder.

**Key fixes**
- fixed `daily_calorie_budgets.csv` parsing
- focused `master_daily_features.csv` on the intended active period
- preserved a full-range daily table separately

**Important outputs**
- `fused/master_daily_features.csv`
- `fused/master_daily_features_full.csv`
- `fused/master_event_ledger.csv`
- `fused/master_15min_telemetry_active.csv`

**Downstream consumers**
- weather builders
- day feature matrix
- meal decision points
- meal timeline recovery
- multires sequence work

---

# 2. Meal DB seed, review, and finalization

## `build_meal_db_seed.py`
**Status:** historical / foundational  
**Purpose:** deterministic seed generation for meal ontology construction.

**Reads**
- canonical Noom meal-related sources
- trusted fused meal/event context

**Writes**
- `meal_db/seed/raw_food_entry_enriched_seed.csv`
- `meal_db/seed/food_alias_seed.csv`
- `meal_db/seed/canonical_food_entity_seed.csv`
- `meal_db/seed/meal_component_seed.csv`
- `meal_db/seed/meal_event_seed.csv`
- `meal_db/seed/seed_manifest.json`

**Next consumer**
- review batch exporters
- apply scripts
- finalization scripts

---

## `export_meal_llm_review_batches.py`
**Status:** superseded  
**Purpose:** first batch exporter for alias/entity review.

**Problem**
- re-exported already-reviewed aliases

---

## `export_meal_llm_review_batches_v2.py`
**Status:** active historical tool  
**Purpose:** export only pending/unresolved review batches.

**Writes**
- `meal_db/review_batches/*`
- pending-only review manifests

**Next consumer**
- review process
- apply scripts

---

## `apply_meal_llm_review_batch.py`
**Status:** superseded  
**Purpose:** initial batch apply script.

**Problem**
- dtype issues
- awkward move/archive behavior after failure

---

## `apply_meal_llm_review_batch_v2.py`
**Status:** active  
**Purpose:** apply reviewed batch semantics into seed/current tables.

**Reads**
- `meal_db/reviews/<batch_label>/*`

**Writes**
- updated `meal_db/seed/*`
- snapshots to `meal_db/current/*`
- archive state where needed

**Next consumer**
- finalization scripts

---

## `build_meal_db_final.py`
**Status:** superseded  
**Purpose:** first final meal DB assembly.

**Problem**
- merge and propagation issues

---

## `build_meal_db_final_v2.py`
**Status:** superseded  
**Purpose:** second finalization attempt.

**Improved**
- duplicate-column handling
- merge stability

**Still insufficient because**
- final semantic propagation was incomplete

---

## `build_meal_db_final_repaired.py`
**Status:** superseded  
**Purpose:** first repair-oriented final rebuild.

**Problem**
- failed on missing event-column combinations

---

## `build_meal_db_final_repaired_v2.py`
**Status:** current  
**Purpose:** preferred repaired final meal DB builder.

**Important outputs**
- `meal_db/final_repaired/canonical_food_entity.csv`
- `meal_db/final_repaired/meal_component_edge.csv`
- `meal_db/final_repaired/meal_semantic_features.csv`
- `meal_db/final_repaired/food_entry_semantic_view.csv`
- `meal_db/final_repaired/meal_db_final_manifest.json`

**Next consumers**
- meal decision points
- meal prediction view
- canonical meal timeline recovery

---

# 3. Weather and exogenous context

## `build_weather_context.py`
**Status:** superseded  
**Purpose:** first weather/daylight context build.

**Problem**
- solar/daylight fields were shifted incorrectly

---

## `build_weather_context_v2.py`
**Status:** current  
**Purpose:** corrected weather/daylight build.

**Reads**
- `fused/master_daily_features.csv`
- `fused/master_15min_telemetry_active.csv`

**Writes**
- `weather/weather_hourly_raw.csv`
- `weather/weather_daily_raw.csv`
- `weather/weather_context_15min.csv`
- `weather/weather_context_daily.csv`
- `weather/weather_context_manifest.json`

**Next consumers**
- day feature matrix
- meal decision points

---

# 4. Model-ready daily/weekly/meal tables

## `build_day_feature_matrix.py`
**Status:** current  
**Purpose:** create the one-row-per-day master modeling table.

**Reads**
- fused daily layer
- repaired meal semantic features
- daily weather context

**Writes**
- `training/day_feature_matrix.csv`
- `training/day_feature_matrix_manifest.json`

**Next consumers**
- weekly and weekend summary builders
- daily transition targets
- weekly regime targets
- scoring / analysis
- multires sequence pipeline indirectly

---

## `build_week_summary_matrix.py`
**Status:** current  
**Purpose:** aggregate day matrix into Monday-based weekly regime rows.

**Writes**
- `training/week_summary_matrix.csv`
- manifest

**Next consumers**
- weekly retrieval
- representation encoders
- regime transitions
- weekly anchor branch

---

## `build_weekend_summary_matrix.py`
**Status:** current  
**Purpose:** aggregate day matrix into Friday–Sunday weekend blocks.

**Writes**
- `training/weekend_summary_matrix.csv`
- manifest

**Next consumers**
- weekend retrieval
- weekend representation
- weekend regime transitions

---

## `build_meal_decision_points.py`
**Status:** current  
**Purpose:** build one-row-per-meal decision dataset.

**Reads**
- repaired meal semantic outputs
- day matrix
- 15-minute telemetry
- 15-minute weather

**Writes**
- `training/meal_decision_points.csv`
- manifest

**Next consumers**
- meal prediction view
- meal baselines

---

## `build_meal_prediction_view.py`
**Status:** current  
**Purpose:** leakage-aware predictive view for next-meal modeling.

**Reads**
- `training/meal_decision_points.csv`

**Writes**
- `training/predictive_views/meal_prediction_view.csv`
- manifest
- target specification JSON

**Next consumers**
- meal baseline training scripts

---

# 5. Meal baseline modeling

## `train_meal_baselines.py`
**Status:** historical / experimental  
**Purpose:** initial meal prediction baseline training.

## `train_meal_baselines_v2.py`
**Status:** superseded / experimental  
**Purpose:** second meal-baseline iteration.

## `train_meal_baselines_v3.py`
**Status:** superseded / experimental  
**Purpose:** third refinement pass.

## `train_meal_baselines_v4.py`
**Status:** latest meal-baseline branch  
**Purpose:** current best version of the meal baseline stack from this branch.

**Notes**
- useful as a meal-side reference branch
- less central to the current project bottleneck than the daily / weekly / temporal work

---

# 6. Retrieval baselines

## `build_retrieval_baselines.py`
**Status:** superseded  
**Purpose:** first retrieval baseline builder over meal/day/week/weekend spaces.

**Spaces created**
- early meal state / day / week / weekend retrieval artifacts

**Why it mattered**
- proved the value of similarity-based comparison as a first-class capability

---

## `build_retrieval_baselines_v2.py`
**Status:** superseded  
**Purpose:** improved retrieval spaces with cleaner semantics and curated variants.

---

## `build_retrieval_baselines_v3.py`
**Status:** current retrieval baseline branch  
**Purpose:** further refined retrieval spaces and artifact organization.

**Next consumers**
- representation learning
- comparison workflows
- future recommendation logic

---

# 7. Representation encoders

## `train_representation_encoders.py`
**Status:** historical / superseded  
**Purpose:** first GPU representation-encoder training stage.

**What it established**
- representation learning can run on the local GPU
- fixed-length training is possible and resumability matters

---

## `train_representation_encoders_v2.py`
**Status:** superseded  
**Purpose:** second pass with improved semantic loss logic.

**Key issue encountered**
- half-precision overflow in contrastive loss path

---

## `train_representation_encoders_v2_1.py`
**Status:** superseded  
**Purpose:** repair pass after v2 instability.

---

## `train_representation_encoders_v3.py`
**Status:** superseded  
**Purpose:** stronger representation-learning phase with better task structure.

## `train_representation_encoders_v3_1.py`
**Status:** superseded  
**Purpose:** diagnostic improvement / trainer fix pass.

## `train_representation_encoders_v3_2.py`
**Status:** superseded  
**Purpose:** extended regime-structure representation training.

**Issue encountered**
- missing dominant-week categorical columns in some smoke tests

## `train_representation_encoders_v3_2_1.py`
**Status:** latest repaired representation trainer  
**Purpose:** patch over v3.2 issues and continue regime representation learning.

**Next consumers**
- audit layer
- regime transition modeling

---

## `audit_regime_representation_results.py`
**Status:** diagnostic  
**Purpose:** audit classification behavior and class balance for regime representations.

**Writes**
- audits
- class-balance views
- feature summaries

**Next consumer**
- regime transition target and model design

---

# 8. Weekly regime transition stack

## `build_regime_transition_targets.py`
**Status:** superseded  
**Purpose:** first build of weekly regime transition targets.

**Issue**
- boolean/NaN dtype assignment bug

---

## `build_regime_transition_targets_v2.py`
**Status:** current  
**Purpose:** corrected weekly regime transition target builder.

**Writes**
- weekly transition matrices
- manifests
- target-ready supervised next-step labels

**Next consumer**
- `train_regime_transition_models.py`

---

## `train_regime_transition_models.py`
**Status:** current anchor branch  
**Purpose:** train weekly regime transition models.

**Outputs**
- trained per-target models
- feature-importance views
- test predictions / summaries

**Next consumers**
- inspection / analysis
- backtests
- weekly focused branch

---

## `inspect_regime_transition_targets.py`
**Status:** superseded  
**Purpose:** first inspection/reporting layer.

**Issue**
- dependency on `tabulate`

## `inspect_regime_transition_targets_v2.py`
**Status:** superseded  
**Purpose:** remove `tabulate` dependency.

**Issue**
- markdown rendering bug on boolean headers

## `inspect_regime_transition_targets_v3.py`
**Status:** current inspection layer  
**Purpose:** reliable inspection report layer for regime transition targets.

---

## `backtest_regime_transition_targets.py`
**Status:** current diagnostic/backtest tool  
**Purpose:** rolling or held-out backtests for regime-transition targets.

**Next consumers**
- threshold scans
- refined weekly focus branch

---

# 9. Weekly weight-gain focused branch

## `analyze_weekly_weight_gain_focus.py`
**Status:** diagnostic  
**Purpose:** isolate the weekly weight-gain target behavior and strongest predictors.

## `train_weekly_weight_gain_refined.py`
**Status:** superseded  
**Purpose:** build a refined weekly weight-gain model.

**Issue**
- report helper missing `df_to_markdown_table`

## `train_weekly_weight_gain_refined_v2.py`
**Status:** current  
**Purpose:** repaired refined weekly weight-gain trainer.

## `calibrate_weekly_weight_gain_probabilities.py`
**Status:** current  
**Purpose:** calibrate weekly weight-gain probabilities and produce thresholded/operational views.

**Role in project**
- provides a macro anchor branch complementary to the daily branch

---

# 10. Daily transition and daily weight-direction branch

## `build_daily_transition_targets.py`
**Status:** superseded  
**Purpose:** first daily transition target build.

**Issue**
- could not locate the actual daily source file used in this repo

## `build_daily_transition_targets_v2.py`
**Status:** superseded  
**Purpose:** second pass with adjusted source logic.

## `build_daily_transition_targets_v3.py`
**Status:** current  
**Purpose:** robust daily transition target builder.

**Writes**
- `training/daily_transition/days_transition_matrix.csv`
- manifest
- daily next-step supervised targets

**Next consumers**
- daily transition model trainer
- daily analysis/scoring

---

## `train_daily_transition_models.py`
**Status:** current anchor branch  
**Purpose:** train daily transition models for key next-day targets.

**Outputs**
- trained models
- test summaries
- feature importance
- prediction files

**Next consumers**
- daily target analysis
- scoring layers
- historical scorer

---

## `analyze_daily_weight_direction_targets.py`
**Status:** superseded  
**Purpose:** first analysis/calibration layer for daily gain/loss targets.

**Issues**
- row alignment mismatch
- calibration leakage problems

## `analyze_daily_weight_direction_targets_v2.py`
**Status:** current  
**Purpose:** corrected analysis layer:
- exact held-out row alignment
- clean train→val→test calibration path

**Outputs**
- comparison tables
- calibration views
- threshold scans
- ablation summaries

**Next consumers**
- operational scorers
- project decision logic

---

## `score_daily_weight_direction.py`
**Status:** superseded  
**Purpose:** first operational scorer for latest day.

**Issue**
- missing `split_suggested` in rebuilt transition-like frame

## `score_daily_weight_direction_v2.py`
**Status:** superseded  
**Purpose:** fixed missing split logic.

**Issue**
- missing metric imports

## `score_daily_weight_direction_v3.py`
**Status:** current single-day scorer  
**Purpose:** operational score for the latest day or a selected day.

**Outputs**
- current-day gain/loss risk bands
- local/global driver proxies
- summary JSON/Markdown

---

## `score_daily_weight_direction_history.py`
**Status:** current historical scorer  
**Purpose:** batch-score the full daily history using the validated daily branch.

**Why it mattered**
- proved the daily bands correspond to real recurring historical regimes
- major project milestone

---

# 11. Meal modality recovery for temporal modeling

## `build_meal_event_sequence_source.py`
**Status:** superseded  
**Purpose:** first attempt to discover a meal-event source for sequence modeling.

**Problem**
- report writer bug

## `build_meal_event_sequence_source_v2.py`
**Status:** superseded  
**Purpose:** report-fix pass.

**Problem**
- list-like cell rendering bug

## `build_meal_event_sequence_source_v3.py`
**Status:** superseded logically  
**Purpose:** robust report rendering.

**Problem**
- source ranking still selected telemetry-like data rather than a true meal table

---

## `build_canonical_meal_timeline.py`
**Status:** current  
**Purpose:** semantically correct meal-timeline recovery.

**Key logic**
- strongly prefer true meal-event tables
- penalize telemetry / 15-minute density
- synthesize meal times only when needed
- produce canonical meal-event timeline for multires sequences

**Outputs**
- `training/meal_timeline_canonical/canonical_meal_timeline.csv`
- manifest / summary / candidate ranking report

**Next consumer**
- multires sequence dataset builder

---

# 12. Multi-resolution sequence dataset

## `build_multires_sequence_dataset.py`
**Status:** superseded  
**Purpose:** first multires sequence packer.

**Issue**
- duplicate `period_start` label in long-sequence export path

## `build_multires_sequence_dataset_v2.py`
**Status:** current  
**Purpose:** corrected multires sequence builder.

**Writes**
- `anchors.csv`
- `modality_masks.csv`
- meals/day/week long sequence exports
- meals/day/week numeric sequence NPZ bundles
- manifests and summary reports

**Role in project**
- direct bridge from anchor/statistical work into temporal deep learning

---

# 13. Temporal multi-resolution model training

## `train_temporal_multires_models.py`
**Status:** superseded  
**Purpose:** first real temporal trainer.

**Issue**
- AMP GradScaler compatibility bug across PyTorch versions

## `train_temporal_multires_models_v2.py`
**Status:** superseded  
**Purpose:** AMP compatibility fix.

**Issue**
- NaN training instability remained

## `train_temporal_multires_models_v3.py`
**Status:** superseded / diagnostic  
**Purpose:** stabilize training:
- GRU AMP safety
- nonfinite handling improvements

**Result**
- numerically stable, but not yet competitive with anchors

## `train_temporal_multires_models_v4.py`
**Status:** superseded  
**Purpose:** diagnostic ablation trainer.

**Issue**
- PowerShell empty-argument CLI parsing annoyance for disabled target lists

## `train_temporal_multires_models_v4_1.py`
**Status:** current temporal diagnostic trainer  
**Purpose:** current best diagnostic temporal trainer.

**Key features**
- single-head overrides
- modality ablations
- balanced sampler
- safer smoke tests
- dual-threshold evaluation
- prediction diagnostics
- skipped-batch accounting

**Current project role**
- determine whether any temporal target/modality combination is promising enough to justify longer runs

## `train_temporal_multires_simple_baselines_v1.py`
**Status:** superseded conservative temporal baseline trainer  
**Purpose:** first simple non-recurrent lag-window baseline trainer on the multires sequence pack.

**Key features**
- flattens recent modality windows into tabular temporal features
- reuses `anchors.csv` and `split_suggested` from the multires dataset
- supports days / weeks / meals modality toggles
- trains dummy / linear / tree baselines for binary and regression targets
- writes comparable backtest artifacts under `reports/backtests/temporal_multires/<run_name>/`

**Why superseded**
- a single-binary CLI run could still train the default regression target unless regression was explicitly disabled

**Historical project role**
- provide a data-efficient non-neural reference for `y_next_weight_loss_flag` and `y_next_weight_delta_lb`
- test whether simple lag-window structure is learnable before escalating to more complex temporal architectures

## `train_temporal_multires_simple_baselines_v2.py`
**Status:** current conservative temporal baseline trainer  
**Purpose:** fix-forward simple non-recurrent lag-window baseline trainer on the multires sequence pack.

**Key features**
- preserves the v1 feature build, model selection, and artifact layout
- reuses `anchors.csv` and `split_suggested` from the multires dataset
- supports days / weeks / meals modality toggles
- defaults to binary-target training without silently adding regression
- still allows explicit regression runs through `--regression-targets` or `--single-regression-target`
- writes compatible backtest artifacts under `reports/backtests/temporal_multires/<run_name>/`

**Current project role**
- provide the conservative temporal comparison floor for binary temporal diagnostics
- keep regression runs explicit rather than accidental

## `train_temporal_multires_neural_compare_v1.py`
**Status:** current focused temporal comparison wrapper  
**Purpose:** run a bounded neural comparison only on the strongest current temporal setup:
- target: `y_next_weight_loss_flag`
- modalities: `days,weeks`
- binary only
- no regression head

**Key features**
- reuses `train_temporal_multires_models_v4_1.py` rather than forking trainer internals
- runs one or more neural families with the same constrained setup
- supports bounded loss-mode checks such as BCE versus focal loss on the same target/modality slice
- writes an aggregate comparison bundle under `reports/backtests/temporal_multires/<comparison_run_name>/`
- makes the comparison explicit against:
  - `simple_loss_daysweeks_v2`
  - `gru_loss_daysweeks_smoke_v4_1`
- keeps probability-dispersion diagnostics in the comparison table because under-dispersed outputs were the prior neural failure mode

**Current project role**
- provide the next bounded neural architecture comparison without widening scope to meals, regression, or multi-head training
- force future neural smoke tests and pilots to be read against the current simple temporal floor
- record negative bounded retries such as `loss_daysweeks_compare_focal_smoke_v1`, where focal loss failed to improve GRU/TCN dispersion or ranking enough to unfreeze the neural path

## `train_temporal_multires_flattened_explore_v1.py`
**Status:** current bounded flattened-path explorer  
**Purpose:** run a wider but still cheap set of flattened `days,weeks` binary-loss classifiers on the same lag-window representation.

**Key features**
- reuses the multires lag-window feature build from the simple baseline path
- evaluates stronger direct comparators such as extra trees, random forest, histogram boosting, and small MLP variants
- writes candidate rankings plus the selected best-model artifact under `reports/backtests/temporal_multires/<run_name>/`
- keeps outputs directly comparable to `simple_loss_daysweeks_v2`

**Current project role**
- test whether better flattened classifiers, rather than recurrent sequence models, are the most plausible next bounded path
- prune the flattened MLP hypothesis quickly against stronger tree references

## `analyze_temporal_flat_winner_v1.py`
**Status:** current diagnostic winner-analysis stage  
**Purpose:** inspect the saved `simple_loss_daysweeks_v2` extra-trees winner without rerunning the same plain confirmation training path.

**Key features**
- rebuilds the exact flattened `days,weeks` feature frame from the multires dataset
- loads the saved winner artifact and emits impurity-based feature importance plus grouped summaries
- runs a bounded permutation-importance pass on the top-ranked winner features
- writes threshold-sweep, confusion-style, false-positive/false-negative, and probability-diagnostic tables
- compares the winner against:
  - `flat_loss_daysweeks_followup_pilot_v1`
  - `gru_loss_daysweeks_smoke_v4_1`
  - `tcn_loss_daysweeks_compare_smoke_v1_check`
- adds a cheap repeated-seed robustness check for the same ET config

**Writes**
- one bounded analysis bundle under `reports/backtests/temporal_multires/<analysis_run_name>/`

**Current project role**
- determine whether `simple_loss_daysweeks_v2` is stable enough to treat as the current operational best flattened temporal path
- surface the top drivers and failure modes before any nearby ET window or split follow-on

## `analyze_temporal_flat_winner_operational_v1.py`
**Status:** current diagnostic operational-validation stage  
**Purpose:** stress the saved `simple_loss_daysweeks_v2` winner as an operating classifier under rolling calibration/eval splits without changing the underlying trained artifact.

**Key features**
- loads the saved `simple_loss_daysweeks_v2` ET winner and its selected threshold
- runs time-aware rolling operational checks with configurable minimum train, calibration, and eval windows
- writes threshold operating tables, calibration comparisons, segmented error slices, and rolling-fold summaries under `reports/backtests/temporal_multires/<analysis_name>/`
- makes it easy to compare the saved `0.4288` operating point against nearby threshold zones such as `0.44` to `0.455`
- supports split-mimic validation so operating-threshold claims can be tested under slightly different rolling geometry before promotion

**Writes**
- one bounded operational-analysis bundle under `reports/backtests/temporal_multires/<analysis_name>/`

**Current project role**
- test whether the current ET winner remains credible under alternate rolling split choices
- decide whether false-positive-heavy threshold behavior is stable enough to justify promoting a new operating point above `0.4288`

## `analyze_temporal_flat_winner_policy_v1.py`
**Status:** current diagnostic policy-bundle stage  
**Purpose:** turn the saved winner plus the two existing operational checks into one conservative operational policy bundle without retraining, changing family, or promoting the threshold.

**Key features**
- reads the locked winner artifacts for `simple_loss_daysweeks_v2`
- keeps the operating threshold fixed at `0.4288`
- records the candidate promotion zone at `0.44` to `0.455`
- writes a compact decision-band table so the score can be used as a ranking/threshold signal rather than a calibrated probability
- states the exact evidence still required before any upward threshold promotion is allowed
- records the exact next bounded follow-on command after the policy layer is added

**Writes**
- one bounded operational-policy bundle under `reports/backtests/temporal_multires/<analysis_name>/`

**Current project role**
- freeze the current winner into an auditable operational policy artifact while keeping the threshold locked
- separate “use now” policy from “promotion later” criteria so future developers do not over-read the favorable held-out slice

## `analyze_temporal_flat_threshold_promotion_v1.py`
**Status:** current diagnostic threshold-confirmation stage  
**Purpose:** run one more additive time-aware threshold confirmation pass for `simple_loss_daysweeks_v2` without reopening the already-rejected ET window-pilot path.

**Key features**
- keeps the model, target, and modalities locked at:
  - `simple_loss_daysweeks_v2`
  - `y_next_weight_loss_flag`
  - `days,weeks`
- reuses the existing held-out threshold table so the locked `0.4288` operating point and the `0.44` to `0.455` candidate zone stay directly comparable
- fits the same ET family on expanding forward folds and scores each eval window with fixed thresholds only
- writes fold-level and pooled threshold summaries under `reports/backtests/temporal_multires/<analysis_name>/`
- records whether any single candidate threshold is actually promotable once additive time-aware evidence is included
- replaces the stale post-policy next step with a modeling-focused follow-on after the threshold question is rechecked

**Writes**
- one bounded threshold-confirmation bundle under `reports/backtests/temporal_multires/<analysis_name>/`

**Current project role**
- close the current threshold-promotion question with forward fixed-threshold evidence rather than another held-out-only argument
- determine whether the repo should keep the threshold locked at `0.4288` and return attention to temporal training design

## `score_temporal_flat_winner_v1.py`
**Status:** current operational scoring stage  
**Purpose:** turn the locked `simple_loss_daysweeks_v2` flattened ET winner into a direct scoring artifact without retraining or changing the threshold.

**Key features**
- rebuilds the exact saved `days,weeks` lag-window feature frame from the multires dataset
- loads the authoritative winner artifact plus the locked policy files
- scores every anchor cheaply and writes a compact history-scoring table
- selects either an explicit `anchor_id` or the latest eligible anchor row for single-case operational output
- can also write a compact latest-`N` eligible recent-cases report for routine operational review via `--recent-n`
- emits the locked threshold decision and the current policy band:
  - below `0.4288`
  - `0.4288` to `<0.44`
  - `0.44` to `0.455` candidate promotion zone
  - `>0.455` stronger positive rank position
- states explicitly that the score is a ranking/threshold signal, not a calibrated probability

**Writes**
- one operational scoring bundle under `reports/backtests/temporal_multires/<scoring_name>/`

**Current project role**
- operationalize the frozen flattened ET winner as a usable scorer
- keep future single-case and batch/history scoring aligned to the same saved model, feature build, and locked threshold policy

## `run_temporal_operational_refresh_v1.py`
**Status:** current operational refresh stage  
**Purpose:** wrap the locked scorer in one compact routine-use monitoring bundle without changing the model, threshold, target, or modalities.

**Key features**
- calls `score_temporal_flat_winner_v1.py` directly so the authoritative scorer output remains the source of truth
- reads the locked policy bundle plus the existing operational check bundles
- writes a compact current-state bundle with:
  - latest case summary
  - recent-`N` score/decision summary
  - simple policy wording
  - lightweight watch checks around score position, recent positive-rate drift, and promotion-zone entry
- preserves the current reporting structure under `reports/backtests/temporal_multires/<refresh_name>/`

**Writes**
- one operational refresh bundle under `reports/backtests/temporal_multires/<refresh_name>/`

**Current project role**
- give operators a single refresh command and a small set of first-read files
- keep routine monitoring additive and auditable while the winning scorer stays locked

## `scripts/run_operational_refresh.sh`
**Status:** current host helper  
**Purpose:** run the locked operational refresh from Linux / WSL with one repo-root command.

**Key features**
- computes the repo root from the helper location
- calls `run_temporal_operational_refresh_v1.py` with only `--project-root`
- prints the generated latest-case summary to the terminal
- points the operator to the refresh bundle and first-read `summary.md`
- does not retrain, change the model, change the threshold, or override policy

**Current project role**
- make routine host-side operational refresh easier while preserving the locked Python entry point as the source of truth

## `scripts/run_operational_refresh.ps1`
**Status:** current host helper  
**Purpose:** run the locked operational refresh from Windows PowerShell with one repo-root command.

**Key features**
- computes the repo root from the helper location
- calls `run_temporal_operational_refresh_v1.py` with only `--project-root`
- prints the generated latest-case summary to the terminal
- points the operator to the refresh bundle and first-read `summary.md`
- does not retrain, change the model, change the threshold, or override policy

**Current project role**
- provide the Windows host equivalent of the Linux / WSL refresh helper

## `run_temporal_path_exploration_v1.py`
**Status:** current bounded temporal path-search orchestrator  
**Purpose:** automate a small, evidence-driven experiment matrix over the current most plausible temporal branches.

**Key features**
- writes an explicit experiment plan with promotion criteria and stop rules
- executes flattened and neural smoke candidates in a single loop
- ranks all explored paths against the required reference runs:
  - `simple_loss_daysweeks_v2`
  - `gru_loss_daysweeks_smoke_v4_1`
  - `tcn_loss_daysweeks_compare_smoke_v1_check`
  - `tcn_loss_daysweeks_compare_pilot_v1`
- emits one aggregate ranking bundle under `reports/backtests/temporal_multires/temporal_path_explore_v1/`

**Current project role**
- identify the strongest next bounded direction without widening scope to meals, regression, or long pilots
- make path pruning auditable rather than conversational

---

# 14. Meal scenario planning / recommendation layer

## `meal_scenario_planning_core_v1.py`
**Status:** active historical planning helper  
**Purpose:** shared implementation for bounded scenario search and immediate next-meal scoring.

**Reads**
- `training/meal_decision_points.csv`
- `training/daily_transition/days_transition_matrix.csv`
- `reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_scoring_v1/history_scores.csv` when available

**Core behavior**
- builds an observed full-day action library from real meal templates only
- requires lunch, dinner, and at least one snack period for day-plan actions
- builds observed meal-level actions for next-meal scoring
- computes explicit enjoyment, healthfulness, consistency, weight-support, and realism rewards
- stress-tests candidates under nearby step-count, weekday/weekend, seasonal, and recent-intake perturbations

**Current project role**
- first bounded recommendation/planning layer on top of existing meal semantics and anchor/temporal scoring artifacts

## `run_meal_scenario_planning_v1.py`
**Status:** active historical bounded planning entry point  
**Purpose:** search realistic observed-template eating patterns for 3, 5, 7, 14, and 30 day horizons.

**Writes**
- `reports/backtests/meal_scenario_planning/<run_name>/scenario_rankings.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/plan_details.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/robustness_stress_tests.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/day_action_library.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/planning_manifest.json`
- `reports/backtests/meal_scenario_planning/<run_name>/summary.md`

**Current reference run**
- `python run_meal_scenario_planning_v1.py --project-root /workspace/foodai --run-name meal_scenario_planning_v1 --candidates-per-horizon 80 --seed 42`

**Current project role**
- provides the first auditable ranked scenario bundle for horizon planning without training a new model

## `score_next_meal_scenario_v1.py`
**Status:** active historical immediate recommendation entry point  
**Purpose:** score realistic “what should I eat right now?” options using observed meal records and the same reward/projection logic as horizon planning.

**Writes**
- `reports/backtests/meal_scenario_planning/<run_name>/next_meal_scores.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/next_meal_projection_stress_tests.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/next_meal_manifest.json`
- `reports/backtests/meal_scenario_planning/<run_name>/summary.md`

**Current reference run**
- `python score_next_meal_scenario_v1.py --project-root /workspace/foodai --run-name next_meal_scenario_scoring_v1 --current-datetime 2026-04-16T12:00:00 --top-n 12`

**Current project role**
- gives an operator-facing next-meal scoring command grounded in observed archetypes and projected short-horizon effects

## `meal_scenario_planning_core_v2.py`
**Status:** current planning helper  
**Purpose:** additive v2 improvement layer for observed-template scenario search and immediate next-meal scoring.

**Reads**
- same source tables as `meal_scenario_planning_core_v1.py`

**Core behavior**
- preserves the realism constraint: no unconstrained meal generation
- expands day actions only through bounded portion variants inside observed archetype-signature calorie ranges
- applies horizon-aware repeat limits to source templates and archetype signatures
- clusters near-identical next-meal records by archetype, service form, protein anchor, and canonical components
- adds plain-language plan, day, and meal explanations

**Current project role**
- preferred helper for planner-quality refinement while keeping the action space observed and auditable

## `run_meal_scenario_planning_v2.py`
**Status:** current bounded planning entry point  
**Purpose:** search realistic observed-template eating patterns with v2 repeat constraints, bounded portion variants, and explanations.

**Writes**
- `reports/backtests/meal_scenario_planning/<run_name>/scenario_rankings.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/plan_details.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/robustness_stress_tests.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/day_action_library.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/planning_manifest.json`
- `reports/backtests/meal_scenario_planning/<run_name>/summary.md`

**Current reference run**
- `python run_meal_scenario_planning_v2.py --project-root /workspace/foodai --run-name meal_scenario_planning_v2 --candidates-per-horizon 80 --seed 42`

**Current project role**
- preferred auditable horizon planner; less repetitive than v1, with explicit repeat diagnostics and explanation fields

## `score_next_meal_scenario_v2.py`
**Status:** current immediate recommendation entry point  
**Purpose:** score de-duplicated observed next-meal option clusters with projected day support and portion guidance.

**Writes**
- `reports/backtests/meal_scenario_planning/<run_name>/next_meal_scores.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/next_meal_projection_stress_tests.csv`
- `reports/backtests/meal_scenario_planning/<run_name>/next_meal_manifest.json`
- `reports/backtests/meal_scenario_planning/<run_name>/summary.md`

**Current reference run**
- `python score_next_meal_scenario_v2.py --project-root /workspace/foodai --run-name next_meal_scenario_scoring_v2 --current-datetime 2026-04-16T12:00:00 --top-n 12`

**Current project role**
- preferred operator-facing next-meal scorer; turns repeated meal records into actionable observed clusters with kcal ranges and explanations

---

# 15. Codex runtime support

## `Dockerfile.codex`
**Status:** current runtime definition  
**Purpose:** build a small Codex-focused container image for the active temporal workflow.

**Includes**
- system Python runtime
- `git`
- `ripgrep`
- pinned temporal workflow dependencies from `requirements-codex-temporal.txt`
- `scripts/start_codex.sh` as the default container entrypoint

**Does not do**
- copy the full repo into the image
- create a repo-local virtualenv
- bundle historical artifacts into the build context

**Current project role**
- provide a reproducible local runtime for documentation, dataset inspection, simple baselines, and temporal smoke tests

## `.dockerignore`
**Status:** current runtime support file  
**Purpose:** keep Docker build context small by excluding the data- and artifact-heavy repo contents from the image build.

**Current project role**
- make the Codex container build fast enough to be practical without mutating project data

## `requirements-codex-temporal.txt`
**Status:** current runtime dependency lock  
**Purpose:** pin the Python packages needed by the active temporal workflow.

**Includes**
- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `scipy`
- CPU `torch`

**Current project role**
- support `build_multires_sequence_dataset_v2.py`
- support `train_temporal_multires_simple_baselines_v2.py`
- support CPU smoke-test use of `train_temporal_multires_models_v4_1.py`

## `scripts/start_codex.sh`
**Status:** current runtime helper  
**Purpose:** container entrypoint for the Codex image.

**Behavior**
- switches into `/workspace/foodai`
- keeps Python output unbuffered
- starts `bash` by default, or executes the command provided to the container

**Current project role**
- make the runtime predictable without introducing extra repo state

---

# 16. Repo inventory and documentation support

## `generate_repo_inventory.py`
**Status:** current utility  
**Purpose:** generate a machine-readable and human-readable directory inventory of the repo.

**Why it exists**
- to keep documentation from depending on chat memory
- to help future developers maintain an up-to-date repo tree and manifest

---

# 17. Practical “current best path” through the repo

If a future developer wants the most relevant current path rather than the full history, it is:

1. raw/canonical/fused source builder
   - `build_foodai_project_v2.py`

2. repaired meal semantics
   - `build_meal_db_final_repaired_v2.py`

3. corrected weather
   - `build_weather_context_v2.py`

4. model-ready aggregate tables
   - `build_day_feature_matrix.py`
   - `build_week_summary_matrix.py`
   - `build_weekend_summary_matrix.py`
   - `build_meal_decision_points.py`
   - `build_meal_prediction_view.py`

5. validated anchor branches
   - daily transition targets / daily transition models
   - regime transition targets / regime transition models
   - daily scoring and historical scoring

6. corrected meal timeline
   - `build_canonical_meal_timeline.py`

7. multires sequence pack
   - `build_multires_sequence_dataset_v2.py`

8. temporal diagnostics and conservative baselines
   - `train_temporal_multires_models_v4_1.py`
   - `train_temporal_multires_simple_baselines_v2.py`

9. bounded scenario planning and immediate next-meal scoring
   - `run_meal_scenario_planning_v1.py`
   - `score_next_meal_scenario_v1.py`

---

# 18. Final note

This catalog includes more scripts than a new developer should use on day one.

That is intentional.

The point is not only to say what the latest script is.  
The point is to explain **why there are multiple versions at all**, and which failures or discoveries produced them.
