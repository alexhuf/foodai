# Historical score report: y_next_weight_gain_flag

## Score channels

| channel          | threshold_balanced_accuracy | threshold_macro_f1 |
| ---------------- | --------------------------- | ------------------ |
| saved_raw        | 0.5                         | 0.5                |
| clean_raw        | 0.4166                      | 0.4997             |
| clean_platt      | 0.2368                      | 0.2415             |
| clean_isotonic   | 0.05                        | 0.35               |
| primary_selected | 0.5                         | 0.5                |

## Realized event rate by primary band

| band  | rows | event_rate           | prob_mean           | prob_median         | period_start_min    | period_start_max    |
| ----- | ---- | -------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| low   | 6    | 0.0                  | 0.2316017006331846  | 0.236219593704521   | 2025-01-30 00:00:00 | 2025-08-01 00:00:00 |
| watch | 288  | 0.003472222222222222 | 0.37465063945470484 | 0.37593260938956274 | 2025-01-28 00:00:00 | 2026-03-27 00:00:00 |
| high  | 121  | 0.8842975206611571   | 0.6190672382881045  | 0.6144395173969832  | 2025-01-27 00:00:00 | 2026-03-21 00:00:00 |

## Realized metrics by split

| split_suggested | rows | event_rate          | prob_mean           | accuracy           | balanced_accuracy  | precision           | recall             | tp | tn  | fp | fn |
| --------------- | ---- | ------------------- | ------------------- | ------------------ | ------------------ | ------------------- | ------------------ | -- | --- | -- | -- |
| test            | 39   | 0.07692307692307693 | 0.467715301084865   | 0.6923076923076923 | 0.6805555555555556 | 0.15384615384615385 | 0.6666666666666666 | 2  | 25  | 11 | 1  |
| train           | 330  | 0.28484848484848485 | 0.44450141333485854 | 0.990909090909091  | 0.9936440677966102 | 0.9690721649484536  | 1.0                | 94 | 233 | 3  | 0  |
| val             | 46   | 0.2391304347826087  | 0.41890797949568886 | 1.0                | 1.0                | 1.0                 | 1.0                | 11 | 35  | 0  | 0  |

## Summary

- target: y_next_weight_gain_flag
- best_model: et
- trainer_metrics_reference: {'accuracy': 0.6923076923076923, 'balanced_accuracy': 0.6805555555555556, 'macro_f1': 0.5282258064516129, 'roc_auc': 0.8148148148148148}
- analysis_v2_reference_target: y_next_weight_gain_flag
- rows_scored: 415
- primary_channel: p_saved_raw
- primary_threshold: 0.5
- experimental_channel: p_clean_isotonic
- experimental_threshold: 0.35
- primary_event_rate_overall: 0.26024096385542167
- primary_probability_mean: 0.44384609686654153
- high_band_rows: 121
- watch_band_rows: 288
- low_band_rows: 6
