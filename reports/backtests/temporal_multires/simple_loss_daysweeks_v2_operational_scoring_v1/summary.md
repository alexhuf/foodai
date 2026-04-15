# Operational Scoring: simple_loss_daysweeks_v2

## Selected Case

- selected_anchor_mode: `latest_eligible`
- anchor_id: `2026-03-29`
- anchor_period_start: `2026-03-29 00:00:00`
- target: `y_next_weight_loss_flag`
- modalities: `days,weeks`
- model family: `flattened ExtraTrees`
- score: `0.381743`
- locked decision: `negative`
- threshold used: `0.4288`
- policy band: `below 0.4288`
- score is not a calibrated probability: `true`
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
- recent cases CSV: `/workspace/foodai/reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_scoring_v1/recent_cases.csv`
- recent cases Markdown: `/workspace/foodai/reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_scoring_v1/recent_cases.md`

## Recent Cases (10)

| anchor_id | anchor_period_start | score | decision_locked_label | policy_band | locked_threshold |
| --- | --- | --- | --- | --- | --- |
| 2026-03-29 | 2026-03-29 00:00:00 | 0.381743 | negative | below 0.4288 | 0.4288 |
| 2026-03-28 | 2026-03-28 00:00:00 | 0.303756 | negative | below 0.4288 | 0.4288 |
| 2026-03-27 | 2026-03-27 00:00:00 | 0.327177 | negative | below 0.4288 | 0.4288 |
| 2026-03-26 | 2026-03-26 00:00:00 | 0.380896 | negative | below 0.4288 | 0.4288 |
| 2026-03-25 | 2026-03-25 00:00:00 | 0.460595 | positive | >0.455 stronger positive rank position | 0.4288 |
| 2026-03-24 | 2026-03-24 00:00:00 | 0.412245 | negative | below 0.4288 | 0.4288 |
| 2026-03-23 | 2026-03-23 00:00:00 | 0.409637 | negative | below 0.4288 | 0.4288 |
| 2026-03-22 | 2026-03-22 00:00:00 | 0.337815 | negative | below 0.4288 | 0.4288 |
| 2026-03-21 | 2026-03-21 00:00:00 | 0.289030 | negative | below 0.4288 | 0.4288 |
| 2026-03-20 | 2026-03-20 00:00:00 | 0.297987 | negative | below 0.4288 | 0.4288 |

