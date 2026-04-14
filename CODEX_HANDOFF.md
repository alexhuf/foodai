# CODEX_HANDOFF.md — FoodAI Local Codex Handoff

## 1. Project state summary
FoodAI is a multi-resolution personal food / biology intelligence system built from:
- Noom behavior data
- Samsung health data
- semantic meal layers
- weather/daylight context
- daily/weekly/weekend/meal-level training surfaces

Current state:
- source/canonical/fused layers built
- meal semantic DB built and repaired
- weather context built
- model-ready daily/weekly/weekend/meal tables built
- daily and weekly transition/anchor branches built
- historical daily scoring validated
- canonical meal timeline recovered correctly
- multires sequence dataset built
- temporal trainer built and debugged through multiple versions
- temporal branch still underperforming anchors

## 2. Current trusted artifacts
- Daily anchor branch
- Weekly regime / weight-gain-focused branch
- Canonical meal timeline
- Multires sequence pack

## 3. Current problem
The temporal GRU branch became numerically stable after fixes, but v4.1 ablation smoke tests still indicate weak ranking / collapsed probability structure.

Observed pattern:
- narrow probability bands
- many settings collapse toward near one-class behavior
- no clear win over anchor baselines yet

Interpretation:
- infrastructure is working
- sequence data is usable
- the training recipe / architecture / task framing is still wrong or not yet good enough

## 4. Immediate next tasks
1. review the existing v4.1 ablation outputs
2. continue focused single-head temporal experiments
3. compare modality mixes
4. identify whether binary loss or regression is more learnable than gain

Suggested next implementation direction:
- `train_temporal_multires_models_v4_2.py`
or
- a simpler lag-window baseline trainer if the recurrent branch continues to underperform

## 5. How to operate in this repo
Before coding:
1. read the project docs listed in `AGENTS.md`
2. inspect current relevant outputs under:
   - `reports/backtests/`
   - `reports/scoring/`
   - `models/`
   - `training/`
3. decide whether the next step is:
   - data/build
   - model training
   - scoring
   - analysis
   - documentation

When adding a new script:
- version it instead of mutating old behavior unpredictably
- document it in `SCRIPT_CATALOG.md`
- record what outputs it creates

## 6. Known pitfalls
1. Reporting helpers have repeatedly failed after otherwise-correct analysis logic.
2. PowerShell argument handling can differ from Linux shell assumptions.
3. “Meal modality present” is not enough; source semantics matter.
4. Temporal smoke tests can look operationally clean while still being scientifically weak.
5. Broken temporal runs must not be benchmarked against anchors.

## 7. Working definition of success
A temporal configuration becomes worth escalating only if it shows:
- finite stable training
- no one-class collapse
- materially improved ranking/separation
- metrics that begin to approach or beat anchor baselines on at least one target

Until then, the anchor system remains the primary truth standard.
