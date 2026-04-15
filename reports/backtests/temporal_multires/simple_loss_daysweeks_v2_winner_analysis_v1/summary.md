# Winner Analysis: simple_loss_daysweeks_v2

- target: `y_next_weight_loss_flag`
- modalities: `days,weeks`
- winning model artifact: `/workspace/foodai/models/temporal_multires/simple_loss_daysweeks_v2/y_next_weight_loss_flag__et.joblib`
- selected threshold: `0.4288`

## Current Winner

- test balanced accuracy: `0.8611`
- test ROC AUC: `0.9167`
- test probability std: `0.0659`
- test confusion: `TN=26, FP=10, FN=0, TP=3`

## Top Drivers

- top impurity features: `days__t_minus_0__is_weekend, weeks__t_minus_0__age_days, weeks__t_minus_2__age_days, weeks__t_minus_3__age_days, days__t_minus_6__day_of_week_num`
- top permutation ROC-AUC features: `days__t_minus_0__is_weekend, days__t_minus_6__is_weekend, days__t_minus_2__day_of_week_num, days__t_minus_6__weather_code, days__t_minus_3__day_of_week_num`

## Robustness

- repeated-seed balanced accuracy mean/std: `0.7972 +/- 0.0722`
- repeated-seed ROC AUC mean/std: `0.8519 +/- 0.0628`

## Comparison

- best shared-valid comparison row: `simple_loss_daysweeks_v2` (balanced_accuracy=0.8611, roc_auc=0.9167)

## Remaining Failure Modes

- false positives on test: `10`
- false negatives on test: `0`
- neural comparison files include 7 test rows whose target is missing in the anchor table, so strict cross-run comparisons should use the shared-valid table here.
