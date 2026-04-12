# Inspection Report: weeks / y_next_weight_loss_flag

- target kind: binary_classification
- best model: et
- description: Whether next period weight delta is <= -0.5 lb.
- n_test: 6

## Test metrics
- accuracy: 0.6667
- balanced_accuracy: 0.6667
- macro_f1: 0.6667
- roc_auc: 0.5556

## Test label balance
- True: 3 (0.500)
- False: 3 (0.500)

## Confusion table

| true  | False | True |
| ----- | ----- | ---- |
| False | 2     | 1    |
| True  | 1     | 2    |

## Top feature drivers (tree_importance)

| rank | feature                                              | score                | direction |
| ---- | ---------------------------------------------------- | -------------------- | --------- |
| 1    | num__meal_count_afternoon_snack_max                  | 0.030752694629628437 |           |
| 2    | num__meal_count_afternoon_snack_sum                  | 0.02124169122672808  |           |
| 3    | num__meal_count_afternoon_snack_mean                 | 0.017379521849183174 |           |
| 4    | num__snowfall_sum_mean                               | 0.013951202092268675 |           |
| 5    | cat__dominant_month_name_August                      | 0.012327929711739774 |           |
| 6    | num__snowfall_sum_sum                                | 0.011098738533184122 |           |
| 7    | num__meal_indulgence_score_sum_max                   | 0.010819447679339591 |           |
| 8    | num__meal_dessert_component_count_from_roles_sum_sum | 0.010519091843905393 |           |
| 9    | num__gloomy_day_count                                | 0.00933562455754825  |           |
| 10   | num__meal_dessert_component_count_from_roles_sum_max | 0.008928892920570782 |           |
| 11   | num__noom_food_fiber_g_mean                          | 0.008313805385541271 |           |
| 12   | num__precipitation_hours_sum                         | 0.007990896203660164 |           |

## Highest-confidence wrong predictions

| period_id  | period_start | y_true_label | y_pred_label | pred_confidence    | true_label_probability |
| ---------- | ------------ | ------------ | ------------ | ------------------ | ---------------------- |
| 2026-03-09 | 2026-03-09   | False        | True         | 0.7779479998745124 |                        |
| 2026-02-09 | 2026-02-09   | True         | False        | 0.6574810836059859 |                        |

## Highest-confidence correct predictions

| period_id  | period_start | y_true_label | y_pred_label | pred_confidence    | true_label_probability |
| ---------- | ------------ | ------------ | ------------ | ------------------ | ---------------------- |
| 2026-03-16 | 2026-03-16   | True         | True         | 0.7889606374806047 |                        |
| 2026-02-23 | 2026-02-23   | False        | False        | 0.7302578192487885 |                        |
| 2026-03-02 | 2026-03-02   | True         | True         | 0.6096926785686336 |                        |
| 2026-02-16 | 2026-02-16   | False        | False        | 0.570767540659938  |                        |
