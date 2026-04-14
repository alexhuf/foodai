# FoodAI Project History

This file tells the project story in chronological order.

It is intentionally narrative rather than only technical.

---

## Phase 0 — Project framing

The project began with a broad ambition:

> build a personal food / biology intelligence system that is useful for comparison, explanation, forecasting, and eventually recommendation.

From the beginning, the system was meant to operate across multiple scales:
- meals
- days
- weeks
- weekends

The design intent was always larger than a single predictive target.

---

## Phase 1 — Raw-source triage and canonicalization

The first job was deciding which Samsung and Noom files were worth keeping.

This phase produced:
- raw-source builders
- canonicalized outputs
- fused daily and telemetry layers

The biggest conceptual move here was recognizing that downstream modeling would need:
- a daily master layer
- an event ledger
- a telemetry-aligned active window

rather than just one giant flat table.

This produced the first major build scripts:
- `build_foodai_project.py`
- `build_foodai_project_v2.py`

`v2` became the trusted historical source builder because it fixed active-window and budget parsing issues.

---

## Phase 2 — Meal semantics and ontology

The project then confronted a hard problem:
raw food logs are not enough.

A separate semantic meal database was built so that the system could reason about:
- canonical food entities
- meal components
- meal events
- meal archetypes
- protein / starch anchors
- cuisine or service form when available

This phase required:
- deterministic seed generation
- export/apply loops for reviewed batches
- multiple finalization attempts
- repaired final tables

This phase mattered because it transformed the project from “calorie logs” into something that could reason about meals as structured events.

---

## Phase 3 — Weather and exogenous context

Weather/daylight was added at both daily and 15-minute resolutions.

This turned out to matter because:
- many behaviors are state/context dependent
- daylight and seasonality can influence habits
- step counts, exercise, and meal behavior interact with weather

The first weather script had a solar/daylight alignment bug, which led to a corrected `v2`.

---

## Phase 4 — Model-ready aggregate tables

With canonical data, meal semantics, and weather in place, the repo then built the core model-ready tables:

- day feature matrix
- week summary matrix
- weekend summary matrix
- meal decision points
- leakage-aware meal prediction view

This stage established the project’s first real training surfaces.

The logic was:
- meal-level tasks need pre-meal state
- day-level tasks need one-row-per-day summaries
- week/weekend tasks need aggregate regime views

---

## Phase 5 — Anchor and baseline modeling

This phase was deliberately statistical and diagnostic.

It included:
- meal baselines
- retrieval baselines
- representation encoders
- regime transition targets and models
- audit and inspection layers

This was the “prove signal exists” phase.

It was also the phase where the project learned that:
- datasets and targets need many rounds of repair
- report-writing bugs can hide correct discovery logic
- smoke tests are mandatory

---

## Phase 6 — Weekly anchor focus

The project then narrowed briefly into a weekly weight-gain-focused branch.

This did **not** mean the project’s ambition shrank to “a weekly calculator.”

Instead, it served as:
- a tractable macro target
- a sanity anchor
- a way to prove regime-level signal before sequence modeling

This branch produced:
- focused analysis
- refined weekly trainer variants
- calibration logic

---

## Phase 7 — Daily transition and weight-direction branch

This became one of the most important phases.

Daily transition targets were built and trained, leading to:
- next-day gain direction
- next-day loss direction
- daily scoring layers
- historical batch scoring

This phase proved something critical:

> the project’s scores were not just plausible-looking snapshots; the daily bands mapped onto real recurring regimes across history.

That validation moved the project from “possible” to “real.”

---

## Phase 8 — Meal recovery for multires sequences

When the project moved toward temporal multires models, a missing meal modality became the main blocker.

An initial meal-source recovery step did technically fill the gap, but it selected the wrong source:
a 15-minute telemetry grid masquerading as meal events.

That false positive forced a correction:
- stricter source ranking
- preference for true meal-event tables
- density-aware scoring
- canonical meal timeline creation

This was one of the most important quality-control moments in the project.

---

## Phase 9 — Multi-resolution sequence dataset

With canonical meal events recovered, the repo built the sequence pack:

- meal sequences
- daily sequences
- weekly sequences
- masks
- anchor-level targets

This was the first real bridge to temporal deep learning.

At that point, the project was finally able to say:
- meals are present
- days are present
- weeks are present
- masks are explicit
- targets are aligned

---

## Phase 10 — First temporal pilots

The repo then built:
- `train_temporal_multires_models.py`
- followed by v2, v3, v4, and v4.1 repairs

This phase taught several hard lessons.

### First lesson: GPU plumbing is not enough
A numerically valid trainer is not the same as a good model.

### Second lesson: NaN instability can be fixed without solving learning
After GRU AMP instability was removed, the model still underperformed anchor baselines.

### Third lesson: diagnostic ablations matter
The first meaningful next step was not “train longer.”
It was:
- isolate target
- isolate modality
- inspect probability collapse
- compare days vs days+weeks

### Fourth lesson: anchors remain essential
The daily anchor models are currently stronger than the temporal GRU smoke tests.
That means the project’s earlier statistical work remains essential, not obsolete.

---

## Current chapter

The project is now in a mature intermediate state:

- data layers exist
- sequence pack exists
- anchor models have proven signal
- meal modality has been corrected
- temporal training is real but not yet strong enough

The repo is no longer blocked by missing data.
It is blocked by identifying the right temporal modeling strategy.

That is a much better problem to have.
