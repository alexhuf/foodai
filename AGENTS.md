# AGENTS.md — FoodAI Codex Operating Rules

## 1. Role
You are operating inside the FoodAI repository as a constrained local engineering agent.

Your job is to:
- understand the current project state
- preserve trusted results
- avoid destructive or low-signal experiments
- extend documentation and code in a way that a future developer can audit

You are not here to improvise broad architecture changes without evidence.

## 2. First files to read at session start
Before making changes, read these files in order:
1. `README.md`
2. `NEXT_DEVELOPER_HANDOFF.md`
3. `DEVELOPER_GUIDE.md`
4. `SCRIPT_CATALOG.md`
5. `PROJECT_HISTORY.md`

Then inspect:
- `docs/repo_inventory/repo_tree.txt`
- `docs/repo_inventory/repo_manifest.json`

## 3. Core project truths
Trusted:
- The daily anchor branch found real signal.
- The weekly regime / weight-gain-focused branch found real signal.
- The canonical meal timeline is now semantically credible.
- The multires sequence dataset is valid enough for temporal experiments.

Not yet trusted as superior:
- The current temporal GRU branch has not beaten the anchor models.
- Do not present temporal smoke-test wins that are numerically unstable, one-class collapsed, or clearly worse than the daily anchors.

Current bottleneck:
- Temporal model selection and training design, not raw data ingestion.

## 4. Safety and scope rules
- Only read/write inside the repo workspace unless explicitly instructed otherwise.
- Do not delete `archive/`, old script versions, or prior reports/models unless explicitly instructed.
- Prefer additive changes: new script versions, new manifests, new reports.
- Do not compare broken runs against anchor baselines.

## 5. Documentation rules
Whenever you add or materially change a stage, update:
- `SCRIPT_CATALOG.md`
- `NEXT_DEVELOPER_HANDOFF.md`
- `README.md` if the project state changed

## 6. Modeling rules
- Use smoke tests before real runs.
- Only escalate to longer GPU runs if smoke tests show finite training, non-collapsed distributions, and materially better separation than current weak temporal runs.
- Present daily and weekly anchors as the truth standard.
- Missing modalities must remain explicit.

## 7. Current recommended next modeling order
1. continue temporal diagnostic ablations
2. isolate best target:
   - `y_next_weight_loss_flag`
   - `y_next_weight_gain_flag`
   - `y_next_weight_delta_lb`
3. isolate best modality mix:
   - days
   - days + weeks
   - then optionally meals
4. compare GRU / TCN / transformer / simpler temporal baselines
5. only then run longer pilots

## 8. Commit behavior
Commit only when:
- the code runs
- artifacts/reports were produced or the change is documentation-only
- docs are updated if project state changed

Do not open PRs automatically unless explicitly instructed.
