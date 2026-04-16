# FoodAI — Next Developer Handoff

This file is the current operational handoff.

If you are the next developer, start here.

---

## 1. Project status

The repository is in a **post-baseline / pre-breakout temporal modeling** stage.

That means:

- the raw and canonical data layers are built
- the meal semantic DB exists
- weather/daylight context exists
- daily / weekly / weekend model-ready tables exist
- transition targets exist
- historical scoring exists
- a corrected canonical meal timeline exists
- a multiresolution sequence dataset exists
- temporal model training exists
- but temporal models are **not yet better than anchor baselines**

---

## 2. Trusted artifacts and findings

## 2.1 Trust these
### Daily anchor branch
This is one of the strongest validated parts of the project.

It established:
- real daily gain/loss signal
- historical regime separation
- strong basis for operational scoring

### Canonical meal timeline
The meal modality was first recovered incorrectly from a telemetry grid and then corrected.

The current canonical meal timeline is the trustworthy meal source.

### Multires sequence dataset
The sequence pack that uses:
- canonical meal timeline
- daily sequences
- weekly sequences
- explicit modality masks

is the correct dataset base for temporal work.

---

## 3. Do **not** trust these as “winning models” yet

### First temporal GRU pilots
They are useful as diagnostics, not as production-worthy improvements.

What is true:
- training is now numerically stable
- outputs are inspectable
- ablation trainer works

What is not true:
- temporal GRU has beaten anchor baselines

---

## 4. Current empirical picture

### 4.1 What the anchor system proved
The anchor/statistical branches found real signal.

That includes:
- daily next-day weight-direction signal
- weekly regime signal
- strong historical band separation

### 4.2 What the first temporal ablations showed
The v4.1 GRU smoke tests suggest:

- `gain + days only` is poor
- `gain + days,weeks` is slightly less poor, but still not useful
- `loss + days,weeks` is the least bad of the tested GRU binary setups so far, but still weak

The common failure mode is:
- under-dispersed probabilities
- narrow score ranges
- effectively one-class prediction behavior

This means the current temporal GRU setup is not learning sharp ranking structure.

### 4.3 What the conservative simple baselines showed
The non-recurrent lag-window baseline path has now been run.

Most important result:
- `simple_loss_daysweeks_v2` clearly outperformed the current GRU smoke tests for binary loss prediction
- test ROC AUC = `0.9167`
- test balanced accuracy = `0.8611`
- selected model = extra trees on flattened `days,weeks`

Other takeaways:
- `simple_loss_days_v2` was also materially stronger than the GRU loss smokes
- `days,weeks` beat `days` for the binary loss target
- regression on `y_next_weight_delta_lb` remained weak for both simple baselines and GRU
- the simple regression runs were only slightly better than the GRU regression smoke and still had negative test `R²`

Interpretation:
- for the current sequence pack and sample size, conservative tabularized temporal baselines are currently stronger diagnostics than the GRU smoke configuration
- binary loss looks more learnable than next-day weight-delta regression in the present temporal setup

### 4.4 What the bounded neural family comparison showed
The focused `loss + days,weeks` neural comparison has now been run via:
- `train_temporal_multires_neural_compare_v1.py`

Reference floor:
- `simple_loss_daysweeks_v2`
  - balanced accuracy = `0.8611`
  - ROC AUC = `0.9167`

Neural comparison results:
- `gru_loss_daysweeks_compare_smoke_v1_check`
  - balanced accuracy = `0.5000`
  - ROC AUC = `0.4722`
  - `prob_std` = `0.0124`
- `tcn_loss_daysweeks_compare_smoke_v1_check`
  - balanced accuracy = `0.5833`
  - ROC AUC = `0.5000`
  - `prob_std` = `0.0090`
- `transformer_loss_daysweeks_compare_smoke_v1_check`
  - balanced accuracy = `0.5000`
  - ROC AUC = `nan`
  - `prob_std` = `nan`

Interpretation:
- TCN was the least bad new neural family in this bounded comparison, but still far below the conservative simple floor
- the new GRU compare run did not improve the ranking picture in a meaningful way
- transformer remained effectively unusable in this smoke setup
- probability dispersion is still extremely narrow for GRU and TCN, so the old collapse problem remains active

### 4.5 What the bounded path-exploration loop showed
The broader bounded exploration loop has now been run via:
- `run_temporal_path_exploration_v1.py`

New directly comparable outcomes:
- `flat_loss_daysweeks_explore_v1:et_balanced`
  - balanced accuracy = `0.8194`
  - ROC AUC = `0.8611`
  - `prob_std` = `0.0748`
- `flat_loss_daysweeks_explore_v1:hgb_depth3`
  - balanced accuracy = `0.6806`
  - ROC AUC = `0.8056`
  - `prob_std` = `0.0656`
- `flat_loss_daysweeks_explore_v1:rf_balanced`
  - balanced accuracy = `0.6944`
  - ROC AUC = `0.7963`
  - `prob_std` = `0.0462`
- `flat_loss_daysweeks_explore_v1:logreg_balanced_c5`
  - balanced accuracy = `0.7222`
  - ROC AUC = `0.6944`
  - `prob_std` = `0.3304`
- `flat_loss_daysweeks_explore_v1:mlp_small`
  - balanced accuracy = `0.6111`
  - ROC AUC = `0.4352`
- `flat_loss_daysweeks_explore_v1:mlp_wide`
  - balanced accuracy = `0.5000`
  - ROC AUC = `0.6481`
  - effectively one-class at the tuned threshold
- `gru_loss_daysweeks_bce_smoke_v1`
  - balanced accuracy = `0.5000`
  - ROC AUC = `0.6852`
  - `prob_std` = `0.0171`
- `tcn_loss_daysweeks_bce_smoke_v1`
  - balanced accuracy = `0.4861`
  - ROC AUC = `0.5278`
  - `prob_std` = `0.0094`
- `tcn_loss_daysweeks_bce_deep_smoke_v1`
  - balanced accuracy = `0.5000`
  - ROC AUC = `0.2870`
  - tuned threshold collapsed to all-negative

Interpretation:
- the strongest surviving next-step direction is still flattened tabular `days,weeks` modeling, not neural sequence modeling
- extra trees remained the best bounded direct challenger to the current simple floor, but it still did **not** beat `simple_loss_daysweeks_v2`
- histogram boosting also survived the bounded criteria and is the clearest non-ET alternate path worth a follow-on pilot
- flattened MLP did not justify promotion
- cheap BCE-only GRU improved ROC AUC versus the older weak neural references, but balanced accuracy stayed stuck at `0.5000`, so it is still not a credible pilot candidate
- cheap BCE-only TCN variants remained collapsed or worse than the old TCN smoke

### 4.6 What the bounded flattened follow-up pilot showed
The requested narrowed follow-up has now been run via:
- `train_temporal_multires_flattened_explore_v1.py --project-root /workspace/foodai --run-name flat_loss_daysweeks_followup_pilot_v1 --candidate-models et_balanced,hgb_depth3`

Directly comparable follow-up outcomes:
- `flat_loss_daysweeks_followup_pilot_v1:et_balanced`
  - balanced accuracy = `0.8194`
  - ROC AUC = `0.8611`
  - `prob_std` = `0.0748`
- `flat_loss_daysweeks_followup_pilot_v1:hgb_depth3`
  - balanced accuracy = `0.6806`
  - ROC AUC = `0.8056`
  - `prob_std` = `0.0656`

Interpretation:
- neither follow-up candidate matched or beat `simple_loss_daysweeks_v2`
- the follow-up exactly reproduced the earlier ranking and metric gap, so the flattened tree ordering now looks stable rather than incidental
- `et_balanced` is now the clearly preferred flattened follow-on family
- `hgb_depth3` should be deprioritized because it remained materially worse than `et_balanced` on both balanced accuracy and ROC AUC in the direct retry

### 4.7 What the bounded winner-analysis and robustness pass showed
The requested winner-analysis bundle has now been run via:
- `analyze_temporal_flat_winner_v1.py --project-root /workspace/foodai`

Bundle location:
- `reports/backtests/temporal_multires/simple_loss_daysweeks_v2_winner_analysis_v1/`

Most important outcomes:
- `simple_loss_daysweeks_v2` remains the best shared-valid test result among the directly requested references
  - balanced accuracy = `0.8611`
  - ROC AUC = `0.9167`
- direct strict comparison still ranks:
  1. `simple_loss_daysweeks_v2`
  2. `flat_loss_daysweeks_followup_pilot_v1:et_balanced`
  3. `tcn_loss_daysweeks_compare_smoke_v1_check`
  4. `gru_loss_daysweeks_smoke_v4_1`
- the saved winner is high-recall but false-positive-heavy at the selected validation threshold
  - test confusion = `TN=26, FP=10, FN=0, TP=3`
  - there were no false negatives on the valid 39-row test slice
- the threshold sweep shows the selected validation threshold is not test-optimal
  - saved threshold = `0.4288`
  - best test balanced accuracy in the sweep occurs around `0.4603`
  - this means ranking is strong, but the operating point is sensitive
- repeated-seed retrains of the same ET config were directionally stable but not perfectly tight
  - balanced accuracy mean/std = `0.7972 +/- 0.0722`
  - balanced accuracy range = `0.6667` to `0.8611`
  - ROC AUC mean/std = `0.8519 +/- 0.0628`
- feature importance is dominated by `days` rather than `weeks`
  - summed impurity importance: `days = 0.6789`, `weeks = 0.3211`
  - strongest grouped drivers include weekend/day-of-week structure, recent-week recency, restaurant-meal intensity, weather, and calorie-budget gap features
- the neural comparison files still contain 7 test rows whose target is missing in `anchors.csv`
  - use the shared-valid comparison table in the winner-analysis bundle for strict apples-to-apples comparison

Interpretation:
- `simple_loss_daysweeks_v2` is now strong enough to treat as the current operational best flattened temporal path
- that does **not** mean the threshold is operationally settled or that the ET family is fully robust to seed/decision-boundary variance
- the next bounded development step should therefore stay on the same target/family/modality path, but probe a nearby window choice rather than rerunning the exact same config again

### 4.8 What the split-mimic operational validation showed
The requested split-mimic operational recheck has now been run via:
- `analyze_temporal_flat_winner_operational_v1.py --project-root /workspace/foodai --analysis-name simple_loss_daysweeks_v2_operational_check_splitmimic_v1 --min-train-rows 245 --calibration-window-rows 46 --eval-window-rows 39`

Bundle location:
- `reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_check_splitmimic_v1/`

Most important outcomes:
- the held-out split-mimic eval slice reproduced the same raw saved-model result seen in the winner analysis
  - threshold = `0.4288`
  - balanced accuracy = `0.8611`
  - ROC AUC = `0.9167`
  - test confusion = `TN=26, FP=10, FN=0, TP=3`
- the nearby higher threshold zone remained favorable on that eval slice
  - `0.44` gave balanced accuracy = `0.8889` with `FP=8`, `FN=0`
  - `0.445` to `0.4545` gave balanced accuracy = `0.9167` with `FP=6`, `FN=0`
  - `0.455` gave balanced accuracy = `0.9306` with `FP=5`, `FN=0`
  - `0.4603` remained the best zero-FN point with balanced accuracy = `0.9444` and `FP=4`
- false-positive heaviness improved meaningfully on the eval slice once thresholds moved above the saved `0.4288`
  - false-positive rate fell from `0.2778` at `0.4288` to `0.1667` at `0.445` and `0.1389` at `0.455`
- calibration did not overturn the raw-threshold conclusion
  - isotonic-on-val improved ECE to `0.0944`, but reproduced the same `FP=10`, `FN=0` operating behavior at its tuned threshold
- the split-mimic time-aware check was weaker than the earlier operational pass
  - rolling folds dropped from `4` to `2`
  - balanced-accuracy mean/min/max became `0.6852 / 0.5093 / 0.8611`
  - fold 1 was only `0.5093` balanced accuracy with a much higher positive rate in that eval window

Interpretation:
- the `0.44` to `0.455` candidate zone is still defensible as a **held-out-slice threshold-improvement zone**
- the new split-mimic check strengthens the claim that the saved `0.4288` threshold is more false-positive-heavy than necessary on the current eval slice
- it does **not** strengthen the broader claim that a threshold above `0.4288` is already stable enough for operational promotion, because the split-mimic rolling results are weaker than the original operational check and only one of the two folds resembles the favorable latest-slice picture
- the next bounded step should therefore stay additive and nearby:
  - keep `y_next_weight_loss_flag`
  - keep `days,weeks`
  - keep the ET family
  - test a small window or split variant before adopting a new operating threshold

### 4.9 What the operational policy bundle now says
The requested conservative policy layer has now been added via:
- `analyze_temporal_flat_winner_policy_v1.py --project-root /workspace/foodai`

Bundle location:
- `reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_policy_v1/`

Most important outcomes:
- the operational threshold remains locked at `0.4288`
- the candidate promotion zone remains `0.44` to `0.455`
- the score is now framed explicitly as a ranking/threshold signal, not a calibrated probability
- the new decision bands are:
  - `< 0.4288` = below current action threshold
  - `0.4288` to `< 0.44` = current positive signal under the locked policy
  - `0.44` to `0.455` = candidate promotion zone, but still unpromoted
  - `> 0.455` = stronger positive rank position, still governed by the same locked threshold policy
- the bundle makes the promotion rule explicit:
  - do **not** move above `0.4288` until one specific threshold in `0.44` to `0.455` reproduces `FN=0`, improves held-out balanced accuracy above `0.8611`, and reduces false positives below `10`, with the same upward-threshold claim supported by an additional additive time-aware check rather than only the current favorable held-out slice

Interpretation:
- the project now has an auditable operational policy artifact for the current winner
- this is a policy freeze, not a model-family change and not a threshold promotion
- the next command should continue the nearby same-family window check rather than reopen threshold promotion prematurely

### 4.10 What the additive threshold-promotion confirmation pass showed
The requested additive threshold-promotion confirmation pass has now been run via:
- `python analyze_temporal_flat_threshold_promotion_v1.py --project-root /workspace/foodai`

Bundle location:
- `reports/backtests/temporal_multires/simple_loss_daysweeks_v2_threshold_promotion_check_v1/`

Most important outcomes:
- the held-out 39-row slice still favors the higher candidate zone
  - `0.4400` gave balanced accuracy = `0.8889` with `FP=8`, `FN=0`
  - `0.4450` and `0.4500` gave balanced accuracy = `0.9167` with `FP=6`, `FN=0`
  - `0.4550` gave balanced accuracy = `0.9306` with `FP=5`, `FN=0`
- the new fixed-threshold time-aware confirmation rejected promotion across 6 forward folds of 39 eval rows each
  - locked `0.4288`: mean fold balanced accuracy = `0.5718`, pooled balanced accuracy = `0.5832`, `FP_total=43`, `FN_total=30`
  - `0.4400`: mean fold balanced accuracy = `0.5580`, pooled balanced accuracy = `0.5613`, `FP_total=40`, `FN_total=33`
  - `0.4450`: mean fold balanced accuracy = `0.5639`, pooled balanced accuracy = `0.5667`, `FP_total=38`, `FN_total=33`
  - `0.4500`: mean fold balanced accuracy = `0.5429`, pooled balanced accuracy = `0.5495`, `FP_total=37`, `FN_total=35`
  - `0.4550`: mean fold balanced accuracy = `0.5514`, pooled balanced accuracy = `0.5576`, `FP_total=34`, `FN_total=35`
- no candidate threshold in `0.44` to `0.455` cleared the additive support rule against the lock
  - every candidate reduced false positives
  - every candidate also increased false negatives and failed to beat the locked threshold on mean fold balanced accuracy and pooled balanced accuracy
- the threshold remains unpromoted
  - promoted threshold = `none`
  - operational threshold stays locked at `0.4288`

Interpretation:
- the threshold-promotion question is now materially more settled for the current winner
- the held-out slice still says the score ranking is useful, but the new forward check says the higher candidate zone is not robust enough to replace the lock
- do **not** spend the next cycle on more threshold-promotion arguments for `simple_loss_daysweeks_v2` unless new model behavior changes the score distribution itself
- the next bounded step should move back to temporal training design on the same locked target/modality pair
- exact next command after this validation pass:
  - `python train_temporal_multires_neural_compare_v1.py --project-root /workspace/foodai --comparison-run-name loss_daysweeks_compare_focal_smoke_v1 --families gru,tcn --binary-loss-mode focal --focal-gamma 2.0 --smoke-test`

### 4.11 What the focal-loss GRU/TCN smoke probe showed
The requested focal-loss comparison has now been run via:
- `python train_temporal_multires_neural_compare_v1.py --project-root /workspace/foodai --comparison-run-name loss_daysweeks_compare_focal_smoke_v1 --families gru,tcn --binary-loss-mode focal --focal-gamma 2.0 --smoke-test`

Bundle location:
- `reports/backtests/temporal_multires/loss_daysweeks_compare_focal_smoke_v1/`

Direct focal-loss outcomes:
- `gru_loss_daysweeks_compare_smoke_v1`
  - balanced accuracy = `0.5417`
  - ROC AUC = `0.5000`
  - `prob_std` = `0.0139`
  - fixed `0.5` threshold still predicted all-negative
  - tuned threshold lifted positive rate only to `0.2564`
- `tcn_loss_daysweeks_compare_smoke_v1`
  - balanced accuracy = `0.5000`
  - ROC AUC = `0.4722`
  - `prob_std` = `0.0083`
  - fixed and tuned thresholds both stayed all-positive

Direct comparison against the requested references:
- versus `simple_loss_daysweeks_v2`
  - focal GRU trailed by `0.3194` balanced accuracy and `0.4167` ROC AUC
  - focal TCN trailed by `0.3611` balanced accuracy and `0.4444` ROC AUC
- versus `gru_loss_daysweeks_smoke_v4_1`
  - focal GRU improved balanced accuracy from `0.4444` to `0.5417`, but ROC AUC fell from `0.5278` to `0.5000` and `prob_std` fell from `0.0202` to `0.0139`
- versus `tcn_loss_daysweeks_compare_smoke_v1_check`
  - focal TCN lost balanced accuracy from `0.5833` to `0.5000`, lost ROC AUC from `0.5000` to `0.4722`, and slightly narrowed dispersion from `0.0090` to `0.0083`
- versus `tcn_loss_daysweeks_compare_pilot_v1`
  - focal TCN exactly matched the same all-positive collapse pattern, with identical tuned balanced accuracy = `0.5000` and identical ROC AUC = `0.4722`

Interpretation:
- focal loss did **not** improve probability dispersion
- focal loss did **not** resolve collapse or one-class behavior
  - GRU still collapsed to all-negative at `0.5`
  - TCN still collapsed to all-positive even after threshold tuning
- focal loss did **not** improve balanced accuracy or ROC AUC enough to keep GRU/TCN neural sequence work as the leading path
- for the current repo state, GRU/TCN sequence work should now be treated as **frozen behind the flattened ET winner** unless a new architecture or training design clears `simple_loss_daysweeks_v2` on both ranking and tuned classification behavior

### 4.12 What the first meal scenario-planning layer added
The first bounded scenario-planning / recommendation layer has now been added via:
- `run_meal_scenario_planning_v1.py`
- `score_next_meal_scenario_v1.py`
- shared helper: `meal_scenario_planning_core_v1.py`

Reference horizon-planning run:
- `python run_meal_scenario_planning_v1.py --project-root /workspace/foodai --run-name meal_scenario_planning_v1 --candidates-per-horizon 80 --seed 42`

Bundle location:
- `reports/backtests/meal_scenario_planning/meal_scenario_planning_v1/`

Reference immediate next-meal run:
- `python score_next_meal_scenario_v1.py --project-root /workspace/foodai --run-name next_meal_scenario_scoring_v1 --current-datetime 2026-04-16T12:00:00 --top-n 12`

Bundle location:
- `reports/backtests/meal_scenario_planning/next_meal_scenario_scoring_v1/`

Most important implementation choices:
- the day-plan action space is observed full-day meal templates only
- each promoted day template must include lunch, dinner, and at least one snack period
- immediate next-meal actions are observed meal records from historically repeated slot/archetype combinations
- scoring is a weighted multi-objective reward:
  - enjoyment
  - healthfulness
  - consistency
  - weight-support
  - realism
- robustness is handled by stress-testing step-count variation, weekday/weekend variation, adjacent season, and recent heavier/lighter intake
- promotion rejects plans with missing required slots, out-of-core calories, consecutive duplicate day templates, repeat-heavy patterns, weak robust weight-support, or excessive fragility

Current reference outcomes:
- observed templates after required-slot filtering: `229`
- core observed calorie band used for promotion: about `1363` to `2643` kcal/day
- promoted candidate counts in the reference run:
  - 3 days: `38 / 43`
  - 5 days: `33 / 45`
  - 7 days: `31 / 45`
  - 14 days: `24 / 45`
  - 30 days: `11 / 45`
- best promoted robust scores by horizon:
  - 3 days: `0.853`
  - 5 days: `0.854`
  - 7 days: `0.789`
  - 14 days: `0.744`
  - 30 days: `0.740`

Interpretation:
- this is a concrete planning layer, not a new predictive model promotion
- its recommendations are constrained to observed meal archetypes and observed day templates
- it should be treated as a first auditable planner that can rank realistic options using existing signals, not as proof that the reward weights are final

### 4.13 What meal scenario-planning v2 improved
The targeted planner-quality pass has now been added via:
- `run_meal_scenario_planning_v2.py`
- `score_next_meal_scenario_v2.py`
- shared helper: `meal_scenario_planning_core_v2.py`

Reference horizon-planning run:
- `python run_meal_scenario_planning_v2.py --project-root /workspace/foodai --run-name meal_scenario_planning_v2 --candidates-per-horizon 80 --seed 42`

Bundle location:
- `reports/backtests/meal_scenario_planning/meal_scenario_planning_v2/`

Reference immediate next-meal run:
- `python score_next_meal_scenario_v2.py --project-root /workspace/foodai --run-name next_meal_scenario_scoring_v2 --current-datetime 2026-04-16T12:00:00 --top-n 12`

Bundle location:
- `reports/backtests/meal_scenario_planning/next_meal_scenario_scoring_v2/`

Most important implementation choices:
- v2 keeps the core realism constraint: all actions are still observed day templates or observed meal records/archetype clusters
- day planning adds bounded portion variants by scaling observed templates only within observed archetype-signature calorie ranges
- horizon planning applies repeat limits that tighten by horizon and reports source-template/signature diversity
- next-meal scoring clusters near-identical observed meal records by archetype, service form, protein anchor, and canonical components
- summaries and CSVs now include plain-language explanations and observed kcal ranges for portion guidance

Reference v2 outcomes:
- observed base day templates after required-slot filtering: `229`
- bounded day variants added: `363`
- total v2 day actions: `592`
- promoted candidate counts in the reference run:
  - total promoted: `170 / 253`
- best promoted robust scores by horizon:
  - 3 days: `0.856`
  - 5 days: `0.803`
  - 7 days: `0.763`
  - 14 days: `0.756`
  - 30 days: `0.698`
- best-plan diversity improved versus v1:
  - 5-day best plan moved from `3` to `5` unique source templates
  - 7-day best plan moved from `6` to `7` unique source templates
  - 30-day best plan moved from `14` to `19` unique source templates
- next-meal reference output moved from `12` rows / `9` unique meal texts to `12` de-duplicated clusters / `12` unique representative meal texts

Remaining planner refinements:
- optionally wire current-context overrides for manually supplied weight, steps, already-eaten meals, and budget state
- improve analog explanations further by retrieving nearest historical days/meals, not only score-component explanations

---

## 5. Most likely causes of current temporal underperformance

These are hypotheses, not final truths, but they are the best current explanation set.

1. **Data volume vs model class**
   - ~456 anchors is not huge for sequence models
   - simple anchors may currently exploit the data better than the GRU does

2. **Architecture mismatch**
   - GRU does not appear to be the right inductive bias here
   - TCN improved slightly over GRU in the bounded comparison, but not enough to change the project state
   - the focal-loss GRU/TCN retry did not rescue either family
   - the current transformer smoke setup is not yet viable
   - the new bounded path search confirms flattened boosted-tree models are still the strongest near-term class
   - the flattened MLP path did not validate as the next-best direction

3. **Weak temporal signal relative to engineered summaries**
   - daily anchors may already compress the useful structure efficiently

4. **Synthetic meal timing noise**
   - meals are semantically valid, but timing is only partially real

5. **Class-imbalance / threshold issues**
   - the v4.1 trainer improved diagnostics, but ranking quality remains weak

---

## 6. Recommended next steps

### Highest-priority next work
Use the new simple baseline results as the comparison floor and treat bounded flattened-tree refinement as the leading next path. After the focal-loss failure, GRU/TCN sequence work should stay frozen unless a materially different setup earns another bounded retry.

Current empirical floor:
1. `simple_loss_daysweeks_v2`
2. `simple_loss_days_v2`

Current best directly comparable follow-on candidate:
1. `flat_loss_daysweeks_explore_v1:et_balanced`

Why:
- this is now the strongest non-anchor temporal-style result currently recorded in-repo after a direct follow-up retry
- they show the current neural branch is still underperforming a more conservative baseline on the same dataset
- they make binary loss on flattened `days,weeks` the cleanest target for the next bounded comparison
- they collapse the earlier shortlist to one clearly preferred flattened family
- the new split-mimic check supports a nearby higher threshold zone on the current eval slice, but not yet enough rolling stability to promote it
- the new policy bundle locks that interpretation into an explicit operational artifact so future work does not blur current use and future promotion criteria

### After that
Choose one path:

#### Path A — flattened tabular follow-on
- keep the follow-on centered on `et_balanced`
- keep target = `y_next_weight_loss_flag`
- keep modalities = `days,weeks`
- do not widen scope to meals or regression
- use `simple_loss_daysweeks_v2_winner_analysis_v1` to read threshold sensitivity and dominant lag groups before touching the next training command
- use `simple_loss_daysweeks_v2_operational_check_splitmimic_v1` to read how threshold behavior changes under the shorter rolling split before promoting any new operating point
- use `simple_loss_daysweeks_v2_operational_policy_v1` as the authoritative current operating policy:
  - threshold stays at `0.4288`
  - the score is ranking-oriented, not probability-calibrated
  - `0.44` to `0.455` stays only a candidate promotion zone until a new additive robustness check clears it
- use the exploration bundle to rank any follow-on directly against:
  - `simple_loss_daysweeks_v2`
  - `gru_loss_daysweeks_smoke_v4_1`
  - `tcn_loss_daysweeks_compare_smoke_v1_check`
  - `tcn_loss_daysweeks_compare_pilot_v1`
- treat `hgb_depth3` as deprioritized unless a later bounded window-sensitivity check gives a concrete reason to revive it
- exact next model-development command:
  - `python train_temporal_multires_flattened_explore_v1.py --project-root /workspace/foodai --run-name flat_loss_daysweeks_et_windowpilot_v1 --target y_next_weight_loss_flag --modalities days,weeks --candidate-models et_balanced --days-window 7 --weeks-window 2`
  - reason: keep the same winning family/target/modality mix, avoid a plain confirmation rerun, and test whether a slightly shorter weekly context stabilizes the false-positive-heavy operating point without widening scope

#### Path B — continue neural ablations only after a concrete reason appears
- if continuing neural work, use `gru_loss_daysweeks_bce_smoke_v1` only as a diagnostic sign that ranking may be salvageable, not as a pilot candidate
- treat `tcn_loss_daysweeks_compare_smoke_v1_check` as the current TCN neural ceiling until a new run clears it on both ranking and balanced accuracy
- treat `loss_daysweeks_compare_focal_smoke_v1` as a negative result, not as a promotion path
- do not revisit transformer until there is a concrete optimization reason rather than a generic architecture retry

Operational note:
- use `run_temporal_path_exploration_v1.py` when the goal is to rerun or extend the bounded path-search loop in a directly comparable way
- use `train_temporal_multires_flattened_explore_v1.py` when the goal is to inspect stronger flattened classifiers on the same lag-window representation
- use `run_temporal_operational_refresh_v1.py` when the goal is routine monitoring around the locked winner without retraining:
  - `python run_temporal_operational_refresh_v1.py --project-root /workspace/foodai`
  - host Linux / WSL helper from repo root:
    - `./scripts/run_operational_refresh.sh`
  - host Windows PowerShell helper from repo root:
    - `.\scripts\run_operational_refresh.ps1`
  - authoritative refresh bundle: `reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_refresh_v1/`
  - first-read files:
    - `summary.md`
    - `latest_case_summary.md`
    - `watch_checks.md`
- use `score_temporal_flat_winner_v1.py` when the goal is to score the frozen winner operationally on the latest eligible anchor or a specific `anchor_id` without retraining:
  - `python score_temporal_flat_winner_v1.py --project-root /workspace/foodai`
  - optional recent-history operational report:
    - `python score_temporal_flat_winner_v1.py --project-root /workspace/foodai --recent-n 10`
  - authoritative scoring bundle: `reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_scoring_v1/`
  - authoritative winner artifacts remain:
    - `models/temporal_multires/simple_loss_daysweeks_v2/y_next_weight_loss_flag__et.joblib`
    - `reports/backtests/temporal_multires/simple_loss_daysweeks_v2/config.json`
    - `reports/backtests/temporal_multires/simple_loss_daysweeks_v2/selected_models.json`
    - `reports/backtests/temporal_multires/simple_loss_daysweeks_v2/selected_thresholds.json`
    - `reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_policy_v1/operational_policy.json`

---

## 7. What should *not* happen yet

Do **not**:
- start a multi-day brute-force GPU run
- declare temporal models superior
- keep spending bounded cycles on the same GRU/TCN loss-days,weeks setup without a concrete new training idea
- retire the anchor branch
- add more complexity before isolating a setup that clearly beats chance and approaches anchor performance

---

## 8. Suggested success criteria for temporal progression

Only escalate to long real pilots once a smoke-test configuration clears something like:

### Binary target
- ROC AUC materially above 0.60
- balanced accuracy above 0.55
- no one-class collapse
- probability distribution visibly broader / better ranked than the current GRU / TCN runs

### Regression target
- MAE / RMSE better than the current simple temporal baselines
- improve beyond roughly `0.312` MAE / `0.574` RMSE while moving `R²` materially upward from the current negative range

### Overall
- must be worth comparing directly against the daily or weekly anchor baselines

---

## 9. Key scripts to know right now

### Data / sequence side
- `build_canonical_meal_timeline.py`
- `build_multires_sequence_dataset_v2.py`

### Temporal training side
- `train_temporal_multires_models_v4_1.py`
- `train_temporal_multires_simple_baselines_v2.py`
- `train_temporal_multires_neural_compare_v1.py`
- `train_temporal_multires_flattened_explore_v1.py`
- `analyze_temporal_flat_winner_v1.py`
- `analyze_temporal_flat_winner_operational_v1.py`
- `analyze_temporal_flat_threshold_promotion_v1.py`
- `run_temporal_path_exploration_v1.py`

### Current focused neural comparison entry point
Use `train_temporal_multires_neural_compare_v1.py` when the goal is to compare neural families on the current best-bounded temporal setup only:
- target: `y_next_weight_loss_flag`
- modalities: `days,weeks`
- binary only
- explicit reference runs:
  - `simple_loss_daysweeks_v2`
  - `gru_loss_daysweeks_smoke_v4_1`
- latest bounded comparison report:
  - `reports/backtests/temporal_multires/loss_daysweeks_compare_smoke_v1_check/comparison_report.md`

### Anchor / reference side
- daily scoring and historical scorer scripts
- weekly weight-gain-focused branch
- regime transition model stack

### Scenario planning / recommendation side
- `run_meal_scenario_planning_v1.py`
- `score_next_meal_scenario_v1.py`
- `run_meal_scenario_planning_v2.py`
- `score_next_meal_scenario_v2.py`
- current planning bundle:
  - `reports/backtests/meal_scenario_planning/meal_scenario_planning_v2/summary.md`
- current immediate next-meal bundle:
  - `reports/backtests/meal_scenario_planning/next_meal_scenario_scoring_v2/summary.md`

See `SCRIPT_CATALOG.md` for the full map.

---

## 10. One-sentence handoff summary

FoodAI is now a **real multi-resolution personal food/biology modeling system with strong anchored signal, a valid sequence dataset, and a first bounded observed-template scenario planner; the current temporal neural branch has still not beaten the anchor models and should remain an active diagnosis problem rather than a solved modeling stack.**
