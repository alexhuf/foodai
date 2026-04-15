# FoodAI Operator Runbook

This is the concise operator runbook for the current locked temporal winner:
`simple_loss_daysweeks_v2`.

## Refresh Commands

Run from the repo root.

Linux / WSL:

```bash
./scripts/run_operational_refresh.sh
```

Windows PowerShell:

```powershell
.\scripts\run_operational_refresh.ps1
```

Both commands call the locked refresh wrapper:

```bash
python run_temporal_operational_refresh_v1.py --project-root <repo-root>
```

They do not retrain, change the model, change the threshold, or override the operational policy.

## First File To Read

After each refresh, read this first:

```text
reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_refresh_v1/summary.md
```

Then inspect, as needed:

```text
reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_refresh_v1/latest_case_summary.md
reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_refresh_v1/watch_checks.md
```

## Locked Policy

- target: `y_next_weight_loss_flag`
- modalities: `days,weeks`
- model family: `flattened ExtraTrees`
- locked threshold: `0.4288`
- positive rule: `score >= 0.4288`
- candidate promotion zone: `0.44` to `0.455`
- score meaning: ranking / threshold signal only
- the score is not a calibrated probability

Decision bands:

- `< 0.4288`: below current action threshold
- `0.4288` to `< 0.44`: current positive signal under the locked policy
- `0.44` to `0.455`: candidate promotion zone; still unpromoted
- `> 0.455`: stronger positive rank position under the same locked threshold

## Watch Conditions

Current refresh watch checks are:

- latest score vs recent window
- recent positive-rate shift
- latest score in candidate promotion zone

Treat a watch as an operator review flag, not as permission to change the policy.

Watch rules:

- latest score vs recent window: watch when the latest score falls outside the recent-window q10/q90 band
- recent positive-rate shift: watch when the latest-N locked positive rate falls outside the historical rolling q10/q90 band
- candidate promotion zone: watch when the latest score is between `0.44` and `0.455`

The latest recorded refresh had:

- latest score vs recent window: `ok`
- recent positive-rate shift: `watch`
- latest score in candidate promotion zone: `False`

## Do Not Change Casually

Do not casually change:

- the model artifact under `models/temporal_multires/simple_loss_daysweeks_v2/`
- `score_temporal_flat_winner_v1.py`
- `run_temporal_operational_refresh_v1.py`
- the locked threshold `0.4288`
- the candidate promotion zone `0.44` to `0.455`
- the policy bundle `simple_loss_daysweeks_v2_operational_policy_v1`
- the target, modality mix, scoring interpretation, or operational decision bands

Threshold promotion requires a new additive robustness check. The candidate zone is evidence to monitor, not the active operating threshold.
