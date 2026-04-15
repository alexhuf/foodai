# Operational Policy: simple_loss_daysweeks_v2

- target: `y_next_weight_loss_flag`
- modalities: `days,weeks`
- model family: `flattened ExtraTrees`
- locked threshold: `0.4288`
- candidate promotion zone: `0.44` to `0.455`

## Current Policy

- classify as operationally positive only when `score >= 0.4288`
- interpret the score as a ranking/threshold signal, not as a calibrated probability
- locked operating point reference: `balanced_accuracy=0.8611, fp=10, fn=0, positive_rate_pred=0.3333`

## Decision Bands

- `< 0.4288`: below the current action threshold
- `0.4288` to `< 0.44`: current positive signal under the locked policy
- `0.44` to `0.455`: explicit candidate promotion zone; positive now, but still unpromoted
- `> 0.455`: stronger positive rank position, still governed by the same locked threshold policy

## Promotion Rule

- do not promote the threshold yet
- exact condition for promotion above `0.4288`: one specific threshold in `0.44` to `0.455` must reproduce `FN=0`, improve held-out balanced accuracy above `0.8611`, and reduce false positives below `10`, with that same upward-threshold claim supported by an additional additive time-aware check
- current blocker: split-mimic rolling validation is still weaker than the favorable held-out slice (folds=2, balanced_accuracy mean/min/max = 0.6852 / 0.5093 / 0.8611)

## Held-Out Zone Evidence

- `threshold=0.4288`: `balanced_accuracy=0.8611, fp=10, fn=0, positive_rate_pred=0.3333`
- `threshold=0.4400`: `balanced_accuracy=0.8889, fp=8, fn=0, positive_rate_pred=0.2821`
- `threshold=0.4450`: `balanced_accuracy=0.9167, fp=6, fn=0, positive_rate_pred=0.2308`
- `threshold=0.4550`: `balanced_accuracy=0.9306, fp=5, fn=0, positive_rate_pred=0.2051`

## Next Command

- `python train_temporal_multires_flattened_explore_v1.py --project-root /workspace/foodai --run-name flat_loss_daysweeks_et_windowpilot_v1 --target y_next_weight_loss_flag --modalities days,weeks --candidate-models et_balanced --days-window 7 --weeks-window 2`
