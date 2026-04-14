# FoodAI Developer Guide

This file is the operational guide for future developers.

It explains how to think about the repo, how data flows through it, which outputs are trustworthy, and what conventions should be followed when adding new stages.

---

## 1. Core design principles

### 1.1 Build auditable layers
Every stage should produce inspectable outputs:
- CSVs / JSON manifests
- backtest reports
- diagnostics
- feature summaries
- calibration or threshold artifacts where relevant

A future developer should be able to inspect a stage **without rerunning everything**.

### 1.2 Prefer staged trust over end-to-end cleverness
The project deliberately used:
- deterministic data builders
- semantic seed tables
- retrieval baselines
- anchor models
- historical scorers

before heavy temporal modeling.

That is a feature, not a weakness.

### 1.3 Never skip data contracts
Every new script should clearly define:
- what files it reads
- what columns it expects
- what files it writes
- who consumes those outputs next

### 1.4 Missing modalities must be explicit
The project learned this the hard way with the meal modality:
- “technically present” is not enough
- modality masks must represent missingness honestly
- false-positive sources can poison downstream training

### 1.5 The best model is not necessarily the most complex model
As of the current state:
- the anchored daily branch is stronger than the temporal GRU pilots
- therefore the baseline / anchor system remains the truth standard

---

## 2. Major data layers

## 2.1 Raw inputs
Primary raw sources:
- Samsung export CSVs
- Noom export CSVs

These are not model-ready and should not be used directly downstream.

## 2.2 Canonical / fused layer
Purpose:
- normalize timestamps
- unify schemas
- extract daily and telemetry-aligned master layers

Key outputs:
- `fused/master_daily_features.csv`
- `fused/master_daily_features_full.csv`
- `fused/master_event_ledger.csv`
- `fused/master_15min_telemetry_active.csv`

## 2.3 Meal semantic DB
Purpose:
- move from raw food logging into interpretable meal entities and structures

Key outputs:
- canonical food entities
- component edges
- meal semantic features
- food entry semantic view

## 2.4 Weather / daylight layer
Purpose:
- provide exogenous context at daily and 15-minute resolutions

## 2.5 Model-ready aggregate layers
These form the backbone of almost all later modeling:
- day feature matrix
- week summary matrix
- weekend summary matrix
- meal decision points
- meal prediction view

## 2.6 Transition layers
Purpose:
- convert static summaries into supervised next-step targets

Examples:
- daily transitions
- weekly regime transitions

## 2.7 Sequence layer
Purpose:
- align meal/day/week context into a multi-resolution supervised sequence pack for temporal modeling

---

## 3. Modeling philosophy

The project uses three broad model families.

### 3.1 Retrieval / similarity models
Purpose:
- compare states, days, or weeks
- provide “memory” of similar prior contexts
- support explanation and recommendation later

### 3.2 Anchor / statistical predictive models
Purpose:
- establish trustworthy target behavior
- determine whether signal exists
- define what “good” looks like before deep models

### 3.3 Temporal / neural models
Purpose:
- go beyond engineered summary features
- learn interactions across time and across modalities

Current status:
- dataset ready
- first temporal pilots run
- not yet better than anchor models

---

## 4. Project conventions

### 4.1 File naming
Builder scripts generally follow these patterns:
- `build_*` — deterministic data/table creation
- `train_*` — model training
- `analyze_*` — analysis of one target/model family
- `score_*` — operational scoring
- `backtest_*` — out-of-sample or pseudo-out-of-sample evaluation
- `inspect_*` / `audit_*` — diagnosis and artifact quality checks
- `*_v2`, `*_v3`, etc. — fix-forward iterations after smoke tests

### 4.2 Outputs
A good script usually writes:
- primary artifact(s)
- manifest JSON
- report Markdown
- supporting CSVs
- diagnostics if the stage is modeling-related

### 4.3 Status labels for scripts
When documenting or refactoring, use:
- **current**
- **active**
- **historical**
- **superseded**
- **diagnostic**
- **experimental**

### 4.4 Never trust a script version just because it exists
Many scripts in this repo are iterative repairs.
The latest version is not always the best version for every job.
Use the script catalog and handoff notes.

---

## 5. Current trusted findings

These are the findings that a new developer should treat as most trustworthy.

### 5.1 Daily anchor branch is real
The daily weight-direction branch demonstrated strong historical separation:
- gain high-band and loss high-band behavior mapped to real recurring outcome regimes
- this is one of the clearest proofs of signal in the project

### 5.2 Meal timeline is now semantically credible
The canonical meal timeline now comes from a real meal-event table rather than a 15-minute telemetry grid.
That was a critical correction.

### 5.3 Multires sequence dataset is real
The sequence pack now has:
- meals
- days
- weeks
- modality masks
- aligned anchor rows

### 5.4 Temporal pilots are not yet competitive
The first GRU temporal runs stabilized after debugging, and a later bounded GRU / TCN / transformer comparison was added, but the neural family still underperforms both anchor models and the current simple temporal floor.
That means:
- do not discard the anchor branch
- use anchors as the benchmark
- use `simple_loss_daysweeks_v2` as the conservative floor for bounded temporal comparisons

---

## 6. Current known weak points

### 6.1 Meal timing is partially synthetic
The canonical meal timeline currently uses synthetic within-day timing when true event timestamps are unavailable.

Implication:
- meals are usable as events
- but precise intra-day timing should not be over-trusted

### 6.2 Temporal logits are under-dispersed
The current GRU and TCN ablations produce overly narrow probability distributions and often collapse to effectively one-class behavior, and the bounded transformer smoke was not interpretable enough to change that assessment.

Implication:
- current temporal setup is not yet expressive or well-trained enough

### 6.3 Data volume is not huge
The temporal sequence pack currently has ~456 anchors.
That is enough for meaningful experiments, but not enough to justify careless model complexity.

---

## 7. Recommended developer workflow

When adding or changing a modeling stage:

1. **Run a smoke test first**
2. **Inspect generated diagnostics**
3. **Do not compare broken runs to baselines**
4. **Write a compact summary artifact**
5. **Update documentation**
   - README if the project state changed
   - script catalog if a new stage was added
   - handoff if the next step changed

---

## 8. What *not* to do

Do not:
- treat the repo as if only the newest script versions matter
- assume the temporal branch is already superior
- trust any modality source without checking row density and semantics
- run large multi-day jobs before a smoke test wins on a small configuration
- let output reports silently fail; reporting bugs repeatedly masked working discovery logic during this project

---

## 9. Recommended next modeling logic

The most defensible next order is:

1. temporal diagnostic ablations
2. identify the best single target
3. identify the best modality mix
4. compare GRU / TCN / transformer
5. only then run a longer GPU pilot

If temporal models continue to underperform, consider a hybrid path:
- anchor models for decision reliability
- temporal embeddings for representation / retrieval enrichment
- ranking / recommendation on top of both

---

## 10. Developer handoff principle

If you are taking over this project, assume:
- the chat may be gone
- the person before you is unavailable
- the docs are the only memory

That means any future major work should also leave behind:
- clear manifests
- reports
- chronology
- rationale
- and “why this version exists” notes
