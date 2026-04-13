# Rolling Backtest Report: weeks / y_next_weight_loss_flag

- target kind: classification
- folds: 5
- models chosen: {'dummy_majority': 3, 'et': 2}

## Aggregate metrics
- test_accuracy_mean: 0.4600
- test_accuracy_std: 0.1818
- test_accuracy_min: 0.3333
- test_accuracy_max: 0.8000
- test_balanced_accuracy_mean: 0.5667
- test_balanced_accuracy_std: 0.1333
- test_balanced_accuracy_min: 0.5000
- test_balanced_accuracy_max: 0.8333
- test_macro_f1_mean: 0.3767
- test_macro_f1_std: 0.2141
- test_macro_f1_min: 0.2500
- test_macro_f1_max: 0.8000
- test_roc_auc_mean: 0.5917
- test_roc_auc_std: 0.1302
- test_roc_auc_min: 0.5000
- test_roc_auc_max: 0.8333
- space: weeks
- target: y_next_weight_loss_flag
- kind: classification
- min_train: 36
- val_size: 8
- test_size: 6
- step: 4
- dropped_feature_cols: ['y_next_weight_delta_lb', 'y_delta_next_weight_delta_lb', 'y_next_meal_events_per_day_week', 'y_delta_next_meal_events_per_day_week', 'y_next_restaurant_meal_fraction_week', 'y_delta_next_restaurant_meal_fraction_week', 'y_next_budget_minus_logged_food_kcal_week', 'y_delta_next_budget_minus_logged_food_kcal_week', 'y_next_dominant_meal_archetype_week', 'y_same_dominant_meal_archetype_week', 'y_next_dominant_cuisine_week', 'y_same_dominant_cuisine_week', 'y_next_dominant_service_form_week', 'y_same_dominant_service_form_week', 'y_next_dominant_prep_profile_week', 'y_same_dominant_prep_profile_week', 'y_next_dominant_protein_week', 'y_same_dominant_protein_week', 'y_next_dominant_starch_week', 'y_same_dominant_starch_week', 'y_next_dominant_energy_density_week', 'y_same_dominant_energy_density_week', 'y_next_dominant_satiety_style_week', 'y_same_dominant_satiety_style_week', 'y_next_weight_loss_flag', 'y_next_weight_gain_flag', 'y_next_restaurant_heavy_flag', 'y_next_budget_breach_flag', 'y_next_high_meal_frequency_flag', 'next_period_id', 'next_period_start', 'split_suggested', 'period_kind', 'period_id', 'week_id']

## Fold-by-fold results

| fold | best_model     | n_train | n_val | n_test | test_accuracy      | test_balanced_accuracy | test_macro_f1      | n_test_classes | test_roc_auc       |
| ---- | -------------- | ------- | ----- | ------ | ------------------ | ---------------------- | ------------------ | -------------- | ------------------ |
| 1    | dummy_majority | 32      | 8     | 6      | 0.3333333333333333 | 0.5                    | 0.25               | 2              | 0.5                |
| 2    | dummy_majority | 36      | 8     | 6      | 0.5                | 0.5                    | 0.3333333333333333 | 2              | 0.5                |
| 3    | dummy_majority | 40      | 8     | 6      | 0.3333333333333333 | 0.5                    | 0.25               | 2              | 0.5                |
| 4    | et             | 44      | 8     | 6      | 0.3333333333333333 | 0.5                    | 0.25               | 2              | 0.625              |
| 5    | et             | 48      | 8     | 5      | 0.8                | 0.8333333333333333     | 0.8                | 2              | 0.8333333333333334 |

## Feature drivers from best fold

| fold | rank | feature                                              | score                 | driver_type     |
| ---- | ---- | ---------------------------------------------------- | --------------------- | --------------- |
| 5    | 1    | num__meal_count_afternoon_snack_max                  | 0.029742535867031857  | tree_importance |
| 5    | 2    | cat__dominant_month_name_August                      | 0.01779503150339212   | tree_importance |
| 5    | 3    | num__meal_count_afternoon_snack_mean                 | 0.016865668024195996  | tree_importance |
| 5    | 4    | num__meal_count_afternoon_snack_sum                  | 0.016493499585849707  | tree_importance |
| 5    | 5    | num__snowfall_sum_mean                               | 0.012215263945231177  | tree_importance |
| 5    | 6    | num__meal_dessert_component_count_from_roles_sum_max | 0.010689247538648392  | tree_importance |
| 5    | 7    | num__snowfall_sum_sum                                | 0.009664419315761195  | tree_importance |
| 5    | 8    | num__precipitation_hours_mean                        | 0.00913796982340306   | tree_importance |
| 5    | 9    | num__meal_indulgence_score_sum_max                   | 0.008749888786414174  | tree_importance |
| 5    | 10   | num__meal_dessert_component_count_from_roles_sum_sum | 0.007889971383094907  | tree_importance |
| 5    | 11   | num__meal_fiber_g_sum_mean                           | 0.007817284736104376  | tree_importance |
| 5    | 12   | num__noom_food_fiber_g_mean                          | 0.007255553460446018  | tree_importance |
| 5    | 13   | num__distinct_meal_archetypes_sum                    | 0.007139091015151739  | tree_importance |
| 5    | 14   | num__meal_comfort_food_score_sum_max                 | 0.0068446199382647425 | tree_importance |
| 5    | 15   | num__meal_side_component_count_from_roles_sum_mean   | 0.006765302060825117  | tree_importance |
