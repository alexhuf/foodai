# Operational Scoring: simple_loss_daysweeks_v2

- selected_anchor_mode: `latest_eligible`
- anchor_id: `2026-03-29`
- anchor_period_start: `2026-03-29 00:00:00`
- target: `y_next_weight_loss_flag`
- modalities: `days,weeks`
- model family: `flattened ExtraTrees`
- score: `0.381743`
- locked threshold decision: `negative` at `0.4288`
- policy band: `below 0.4288`
- score interpretation: `ranking / threshold signal only; not a calibrated probability`

## Policy

- `< 0.4288`: below the current action threshold
- `0.4288` to `<0.44`: current positive signal under the locked policy
- `0.44` to `0.455`: candidate promotion zone; positive now, but still unpromoted
- `>0.455`: stronger positive rank position under the same locked threshold

## Bundle Files

- history scores: `/workspace/foodai/reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_scoring_v1/history_scores.csv`
- selected case JSON: `/workspace/foodai/reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_scoring_v1/selected_case.json`
- scoring manifest: `/workspace/foodai/reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_scoring_v1/scoring_manifest.json`
