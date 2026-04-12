# Inspection Report: weekends / y_next_restaurant_heavy_flag

- target kind: binary_classification
- best model: rf
- description: Whether next period restaurant meal fraction is >= 0.50.
- n_test: 7

## Test metrics
- accuracy: 0.8571
- balanced_accuracy: 0.8750
- macro_f1: 0.8571
- roc_auc: 0.7500

## Test label balance
- False: 4 (0.571)
- True: 3 (0.429)

## Confusion table

| true  | False | True |
| ----- | ----- | ---- |
| False | 3     | 1    |
| True  | 0     | 3    |

## Top feature drivers (tree_importance)

| rank | feature                                       | score                | direction |
| ---- | --------------------------------------------- | -------------------- | --------- |
| 1    | num__noom_food_fat_g_max                      | 0.01443586005783751  |           |
| 2    | num__meal_fat_g_sum_max                       | 0.01356473075501722  |           |
| 3    | num__distinct_cuisines_sum                    | 0.012696780776497885 |           |
| 4    | num__wind_gusts_10m_max_mean                  | 0.012492741247299903 |           |
| 5    | num__meal_component_count_sum_max             | 0.011056557547023133 |           |
| 6    | num__restaurant_specific_meal_count_mean      | 0.010071436170781625 |           |
| 7    | num__weight_velocity_30d_lb_mean              | 0.009840899566180058 |           |
| 8    | num__period_dayofyear                         | 0.009449353026431274 |           |
| 9    | num__wind_gusts_10m_max_sum                   | 0.009378342947891638 |           |
| 10   | num__restaurant_meal_fraction_weekend         | 0.0093424048760637   |           |
| 11   | num__meal_protein_g_sum_mean                  | 0.00922876871742996  |           |
| 12   | num__budget_minus_noom_food_calories_kcal_max | 0.009142614880298659 |           |

## Highest-confidence wrong predictions

| period_id  | period_start | y_true_label | y_pred_label | pred_confidence    | true_label_probability |
| ---------- | ------------ | ------------ | ------------ | ------------------ | ---------------------- |
| 2026-02-06 | 2026-02-06   | False        | True         | 0.6046999186917482 |                        |

## Highest-confidence correct predictions

| period_id  | period_start | y_true_label | y_pred_label | pred_confidence    | true_label_probability |
| ---------- | ------------ | ------------ | ------------ | ------------------ | ---------------------- |
| 2026-03-06 | 2026-03-06   | False        | False        | 0.7756883653767238 |                        |
| 2026-02-27 | 2026-02-27   | True         | True         | 0.7589549774938028 |                        |
| 2026-03-13 | 2026-03-13   | True         | True         | 0.7540375029151285 |                        |
| 2026-02-13 | 2026-02-13   | True         | True         | 0.7533218074258375 |                        |
| 2026-02-20 | 2026-02-20   | False        | False        | 0.6883294272985683 |                        |
| 2026-03-20 | 2026-03-20   | False        | False        | 0.6647117733946304 |                        |
