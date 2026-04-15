# Threshold Promotion Check: simple_loss_daysweeks_v2

- target: `y_next_weight_loss_flag`
- modalities: `days,weeks`
- locked threshold: `0.4288`
- candidate thresholds: `0.4400, 0.4450, 0.4500, 0.4550`
- time-aware forward folds: `6` using `min_train_rows=181` and `eval_window_rows=39`

## Held-Out Reference

- `threshold=0.4288`: `balanced_accuracy=0.8611, fp=10, fn=0`
- `threshold=0.4400`: `balanced_accuracy=0.8889, fp=8, fn=0`
- `threshold=0.4450`: `balanced_accuracy=0.9167, fp=6, fn=0`
- `threshold=0.4500`: `balanced_accuracy=0.9167, fp=6, fn=0`
- `threshold=0.4550`: `balanced_accuracy=0.9306, fp=5, fn=0`

## Time-Aware Confirmation

- `threshold=0.4288`: `mean_fold_balanced_accuracy=0.5718, pooled_balanced_accuracy=0.5832, fp_total=43, fn_total=30, latest_fold_balanced_accuracy=0.4861, latest_fold_fp=1, latest_fold_fn=3`
- `threshold=0.4400`: `mean_fold_balanced_accuracy=0.5580, pooled_balanced_accuracy=0.5613, fp_total=40, fn_total=33, latest_fold_balanced_accuracy=0.5000, latest_fold_fp=0, latest_fold_fn=3`
- `threshold=0.4450`: `mean_fold_balanced_accuracy=0.5639, pooled_balanced_accuracy=0.5667, fp_total=38, fn_total=33, latest_fold_balanced_accuracy=0.5000, latest_fold_fp=0, latest_fold_fn=3`
- `threshold=0.4500`: `mean_fold_balanced_accuracy=0.5429, pooled_balanced_accuracy=0.5495, fp_total=37, fn_total=35, latest_fold_balanced_accuracy=0.5000, latest_fold_fp=0, latest_fold_fn=3`
- `threshold=0.4550`: `mean_fold_balanced_accuracy=0.5514, pooled_balanced_accuracy=0.5576, fp_total=34, fn_total=35, latest_fold_balanced_accuracy=0.5000, latest_fold_fp=0, latest_fold_fn=3`

## Decision

- promoted threshold: `none`
- no candidate in `0.44` to `0.455` cleared the additive time-aware support rule against the locked `0.4288` threshold
- strongest held-out candidate remained `0.4550` on the 39-row saved test slice (`balanced_accuracy=0.9306, fp=5, fn=0`), but its fixed-threshold forward check weakened to `mean_fold_balanced_accuracy=0.5514, pooled_balanced_accuracy=0.5576, fp_total=34, fn_total=35`
- locked threshold reference in the same time-aware pass: `mean_fold_balanced_accuracy=0.5718, pooled_balanced_accuracy=0.5832, fp_total=43, fn_total=30`

## Next Command

- `python train_temporal_multires_neural_compare_v1.py --project-root /workspace/foodai --comparison-run-name loss_daysweeks_compare_focal_smoke_v1 --families gru,tcn --binary-loss-mode focal --focal-gamma 2.0 --smoke-test`
