# Historical score report: y_next_weight_loss_flag

## Score channels

| channel          | threshold_balanced_accuracy | threshold_macro_f1 |
| ---------------- | --------------------------- | ------------------ |
| saved_raw        | 0.5                         | 0.5                |
| clean_raw        | 0.5079                      | 0.5079             |
| clean_platt      | 0.2219                      | 0.2219             |
| clean_isotonic   | 0.25                        | 0.25               |
| primary_selected | 0.25                        | 0.25               |

## Realized event rate by primary band

| band  | rows | event_rate           | prob_mean            | prob_median         | period_start_min    | period_start_max    |
| ----- | ---- | -------------------- | -------------------- | ------------------- | ------------------- | ------------------- |
| low   | 249  | 0.008032128514056224 | 0.023707925175417546 | 0.0                 | 2025-01-27 00:00:00 | 2026-03-27 00:00:00 |
| watch | 41   | 0.0975609756097561   | 0.1805045088206555   | 0.16666666666666682 | 2025-02-04 00:00:00 | 2026-03-25 00:00:00 |
| high  | 125  | 0.864                | 0.49910083217678336  | 0.5                 | 2025-01-28 00:00:00 | 2026-03-04 00:00:00 |

## Realized metrics by split

| split_suggested | rows | event_rate          | prob_mean           | accuracy           | balanced_accuracy  | precision          | recall             | tp  | tn  | fp | fn |
| --------------- | ---- | ------------------- | ------------------- | ------------------ | ------------------ | ------------------ | ------------------ | --- | --- | -- | -- |
| test            | 39   | 0.07692307692307693 | 0.11950338913604447 | 0.8717948717948718 | 0.625              | 0.25               | 0.3333333333333333 | 1   | 33  | 3  | 2  |
| train           | 330  | 0.30606060606060603 | 0.1849422123518728  | 0.9757575757575757 | 0.9797656621557352 | 0.9345794392523364 | 0.9900990099009901 | 100 | 222 | 7  | 1  |
| val             | 46   | 0.21739130434782608 | 0.21739130434782616 | 0.782608695652174  | 0.7527777777777778 | 0.5                | 0.7                | 7   | 29  | 7  | 3  |

## Summary

- target: y_next_weight_loss_flag
- best_model: et
- trainer_metrics_reference: {'accuracy': 0.8974358974358975, 'balanced_accuracy': 0.6388888888888888, 'macro_f1': 0.6388888888888888, 'roc_auc': 0.8611111111111112}
- analysis_v2_reference_target: y_next_weight_loss_flag
- rows_scored: 415
- primary_channel: p_clean_isotonic
- primary_threshold: 0.25
- experimental_channel: p_saved_raw
- experimental_threshold: 0.5
- primary_event_rate_overall: 0.2746987951807229
- primary_probability_mean: 0.18238930663234645
- high_band_rows: 125
- watch_band_rows: 41
- low_band_rows: 249
