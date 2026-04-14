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
- `simple_loss_daysweeks_v1` clearly outperformed the current GRU smoke tests for binary loss prediction
- test ROC AUC = `0.9167`
- test balanced accuracy = `0.8611`
- selected model = extra trees on flattened `days,weeks`

Other takeaways:
- `simple_loss_days_v1` was also materially stronger than the GRU loss smokes
- `days,weeks` beat `days` for the binary loss target
- regression on `y_next_weight_delta_lb` remained weak for both simple baselines and GRU
- the simple regression runs were only slightly better than the GRU regression smoke and still had negative test `R²`

Interpretation:
- for the current sequence pack and sample size, conservative tabularized temporal baselines are currently stronger diagnostics than the GRU smoke configuration
- binary loss looks more learnable than next-day weight-delta regression in the present temporal setup

---

## 5. Most likely causes of current temporal underperformance

These are hypotheses, not final truths, but they are the best current explanation set.

1. **Data volume vs model class**
   - ~456 anchors is not huge for sequence models
   - simple anchors may currently exploit the data better than the GRU does

2. **Architecture mismatch**
   - GRU may not be the right inductive bias here
   - TCN or transformer may still do better
   - a simpler flattened-sequence MLP or boosted-tree on lagged summaries may also be worth testing

3. **Weak temporal signal relative to engineered summaries**
   - daily anchors may already compress the useful structure efficiently

4. **Synthetic meal timing noise**
   - meals are semantically valid, but timing is only partially real

5. **Class-imbalance / threshold issues**
   - the v4.1 trainer improved diagnostics, but ranking quality remains weak

---

## 6. Recommended next steps

### Highest-priority next work
Use the new simple baseline results as the comparison floor before any further temporal escalation.

Current empirical floor:
1. `simple_loss_daysweeks_v1`
2. `simple_loss_days_v1`

Why:
- these are now the strongest temporal-style results currently recorded in-repo
- they show the current GRU branch is underperforming a more conservative baseline on the same dataset
- they make binary loss the cleanest target for the next architecture comparison

### After that
Choose one path:

#### Path A — continue with temporal ablations
- run TCN versions of the same smoke tests
- run transformer versions only after TCN / GRU smoke tests are interpretable
- compare days-only vs days+weeks systematically

#### Path B — simpler sequence baseline
A conservative non-recurrent baseline path now exists via:
- `train_temporal_multires_simple_baselines_v2.py`

Current completed runs:
- `simple_loss_days_v1`
- `simple_loss_daysweeks_v1`
- `simple_delta_days_v1`
- `simple_delta_daysweeks_v1`

What they established:
- keep `days,weeks` as the preferred conservative baseline for `y_next_weight_loss_flag`
- do not treat `y_next_weight_delta_lb` as the leading temporal target yet
- any new GRU / TCN / transformer smoke should be compared directly against `simple_loss_daysweeks_v1`

This path is intentionally more data-efficient than the current neural branch and should be checked before any broader temporal escalation.

Operational note:
- use `train_temporal_multires_simple_baselines_v2.py` for future binary-only baseline runs because it no longer trains the default regression target unless regression is explicitly requested

---

## 7. What should *not* happen yet

Do **not**:
- start a multi-day brute-force GPU run
- declare temporal models superior
- retire the anchor branch
- add more complexity before isolating a setup that clearly beats chance and approaches anchor performance

---

## 8. Suggested success criteria for temporal progression

Only escalate to long real pilots once a smoke-test configuration clears something like:

### Binary target
- ROC AUC materially above 0.60
- balanced accuracy above 0.55
- no one-class collapse
- probability distribution visibly broader / better ranked than current GRU runs

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

### Anchor / reference side
- daily scoring and historical scorer scripts
- weekly weight-gain-focused branch
- regime transition model stack

See `SCRIPT_CATALOG.md` for the full map.

---

## 10. One-sentence handoff summary

FoodAI is now a **real multi-resolution personal food/biology modeling system with strong anchored signal and a valid sequence dataset, but the current temporal neural branch has not yet beaten the anchor models and should still be treated as an active diagnosis problem rather than a solved modeling stack.**
