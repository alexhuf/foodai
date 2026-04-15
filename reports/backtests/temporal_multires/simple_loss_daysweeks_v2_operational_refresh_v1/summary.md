# Operational Refresh: simple_loss_daysweeks_v2

## Latest Case

- anchor_id: `2026-03-29`
- anchor_period_start: `2026-03-29 00:00:00`
- score: `0.381743`
- locked decision: `negative`
- policy band: `below 0.4288`

## Recent Window

- recent_n: `10`
- recent score mean/std: `0.3601` / `0.0546`
- recent score min/max: `0.2890` / `0.4606`
- recent locked positive rate: `0.1000`

## Policy

- locked threshold: `0.4288`
- positive rule: `score >= 0.4288`
- score interpretation: `ranking / threshold signal only; not a calibrated probability`
- candidate promotion zone: `0.44` to `0.455`

## Watch Checks

- latest score vs recent window: `ok`
- recent positive-rate shift: `watch`
- latest score in candidate promotion zone: `False`

## Reference Check

- split-mimic rolling folds: `2`
- split-mimic balanced-accuracy mean/min/max: `0.6852` / `0.5093` / `0.8611`

## Bundle Files

- current state JSON: `/workspace/foodai/reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_refresh_v1/current_state.json`
- latest case Markdown: `/workspace/foodai/reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_refresh_v1/latest_case_summary.md`
- recent summary Markdown: `/workspace/foodai/reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_refresh_v1/recent_summary.md`
- watch checks Markdown: `/workspace/foodai/reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_refresh_v1/watch_checks.md`
- refresh manifest: `/workspace/foodai/reports/backtests/temporal_multires/simple_loss_daysweeks_v2_operational_refresh_v1/refresh_manifest.json`
