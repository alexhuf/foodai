# Inspection Report: weeks / y_next_weight_gain_flag

- target kind: binary_classification
- best model: logreg
- description: Whether next period weight delta is >= 0.5 lb.
- n_test: 6

## Test metrics
- accuracy: 1.0000
- balanced_accuracy: 1.0000
- macro_f1: 1.0000
- roc_auc: 1.0000

## Test label balance
- False: 5 (0.833)
- True: 1 (0.167)

## Confusion table

| true  | False | True |
| ----- | ----- | ---- |
| False | 5     | 0    |
| True  | 0     | 1    |

## Top feature drivers (linear_coefficient)

| rank | feature                                              | score                | direction |
| ---- | ---------------------------------------------------- | -------------------- | --------- |
| 1    | num__meal_dessert_component_count_from_roles_sum_max | -0.29939223462343245 | negative  |
| 2    | num__meal_starch_base_count_sum_mean                 | 0.27618332542988044  | positive  |
| 3    | num__meal_main_component_count_sum_max               | -0.2725388655240849  | negative  |
| 4    | num__samsung_sleep_score_sum                         | 0.25872406585756336  | positive  |
| 5    | num__meal_starch_base_count_sum_sum                  | 0.24800411319807988  | positive  |
| 6    | num__meal_fiber_g_sum_mean                           | 0.24463989844572834  | positive  |
| 7    | num__noom_food_fiber_g_mean                          | 0.24463989844572828  | positive  |
| 8    | num__samsung_sleep_duration_ms_sum                   | 0.24176713483171866  | positive  |
| 9    | num__rain_sum_max                                    | 0.2365948022029089   | positive  |
| 10   | num__samsung_sleep_score_max                         | -0.23545487874758403 | negative  |
| 11   | num__samsung_sleep_score_mean                        | -0.22078004586349756 | negative  |
| 12   | num__precipitation_sum_max                           | 0.2202449017699747   | positive  |

## Highest-confidence correct predictions

| period_id  | period_start | y_true_label | y_pred_label | pred_confidence    | true_label_probability |
| ---------- | ------------ | ------------ | ------------ | ------------------ | ---------------------- |
| 2026-02-09 | 2026-02-09   | False        | False        | 0.999999139720879  |                        |
| 2026-02-23 | 2026-02-23   | False        | False        | 0.9995296368962574 |                        |
| 2026-03-09 | 2026-03-09   | True         | True         | 0.9991817588830317 |                        |
| 2026-02-16 | 2026-02-16   | False        | False        | 0.997871614567094  |                        |
| 2026-03-02 | 2026-03-02   | False        | False        | 0.9966656377107384 |                        |
| 2026-03-16 | 2026-03-16   | False        | False        | 0.9885475985240234 |                        |
