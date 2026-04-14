# Temporal Neural Comparison: loss_daysweeks_compare_smoke_v1_check

- target: `y_next_weight_loss_flag`
- modalities: `days,weeks`
- binary-only: `true`
- regression-heads: `none`
- smoke_test: `true`
- executed_neural_runs: `gru_loss_daysweeks_compare_smoke_v1_check, tcn_loss_daysweeks_compare_smoke_v1_check, transformer_loss_daysweeks_compare_smoke_v1_check`
- required_reference_runs: `simple_loss_daysweeks_v2`, `gru_loss_daysweeks_smoke_v4_1`

## Comparison

| run_name | source_kind | model_family | balanced_accuracy_tuned | roc_auc_tuned | f1_tuned | prob_std | delta_vs_simple_floor_balanced_accuracy | delta_vs_simple_floor_roc_auc | delta_vs_gru_v4_1_balanced_accuracy | delta_vs_gru_v4_1_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simple_loss_daysweeks_v2 | reference_baseline | simple_floor | 0.8611111111111112 | 0.9166666666666666 | 0.375 | 0.06593768411412655 | 0.0 | 0.0 | 0.41666666666666674 | 0.38888888888888884 |
| gru_loss_daysweeks_smoke_v4_1 | historical_neural_reference | gru | 0.4444444444444444 | 0.5277777777777778 | 0.0 | 0.020189868640548156 | -0.41666666666666674 | -0.38888888888888884 | 0.0 | 0.0 |
| gru_loss_daysweeks_compare_smoke_v1_check | new_neural_run | gru | 0.5 | 0.4722222222222222 | 0.0 | 0.012389408649820327 | -0.36111111111111116 | -0.4444444444444444 | 0.05555555555555558 | -0.05555555555555558 |
| tcn_loss_daysweeks_compare_smoke_v1_check | new_neural_run | tcn | 0.5833333333333334 | 0.5 | 0.2 | 0.00903124458920409 | -0.2777777777777778 | -0.41666666666666663 | 0.13888888888888895 | -0.02777777777777779 |
| transformer_loss_daysweeks_compare_smoke_v1_check | new_neural_run | transformer | 0.5 | nan | 0.0 | nan | -0.36111111111111116 | nan | 0.05555555555555558 | nan |

## Interpretation

- `delta_vs_simple_floor_*` shows whether a neural run cleared the current conservative floor.
- `delta_vs_gru_v4_1_*` shows whether a neural run improved on the best existing GRU loss smoke reference.
- `prob_std` remains in the comparison because under-dispersed probabilities were the main prior neural failure mode.
