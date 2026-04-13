# Rolling Backtest Report: weekends / y_next_restaurant_heavy_flag

- target kind: classification
- folds: 4
- models chosen: {'rf': 2, 'et': 1, 'dummy_majority': 1}

## Aggregate metrics
- test_accuracy_mean: 0.3750
- test_accuracy_std: 0.1816
- test_accuracy_min: 0.1667
- test_accuracy_max: 0.6667
- test_balanced_accuracy_mean: 0.5208
- test_balanced_accuracy_std: 0.1488
- test_balanced_accuracy_min: 0.3333
- test_balanced_accuracy_max: 0.7500
- test_macro_f1_mean: 0.3274
- test_macro_f1_std: 0.2007
- test_macro_f1_min: 0.1429
- test_macro_f1_max: 0.6667
- test_roc_auc_mean: 0.6146
- test_roc_auc_std: 0.2454
- test_roc_auc_min: 0.3333
- test_roc_auc_max: 1.0000
- space: weekends
- target: y_next_restaurant_heavy_flag
- kind: classification
- min_train: 36
- val_size: 8
- test_size: 6
- step: 4
- dropped_feature_cols: ['y_next_weight_delta_lb', 'y_delta_next_weight_delta_lb', 'y_next_meal_events_per_day_weekend', 'y_delta_next_meal_events_per_day_weekend', 'y_next_restaurant_meal_fraction_weekend', 'y_delta_next_restaurant_meal_fraction_weekend', 'y_next_budget_minus_logged_food_kcal_weekend', 'y_delta_next_budget_minus_logged_food_kcal_weekend', 'y_next_dominant_meal_archetype_weekend', 'y_same_dominant_meal_archetype_weekend', 'y_next_dominant_cuisine_weekend', 'y_same_dominant_cuisine_weekend', 'y_next_dominant_service_form_weekend', 'y_same_dominant_service_form_weekend', 'y_next_dominant_prep_profile_weekend', 'y_same_dominant_prep_profile_weekend', 'y_next_dominant_protein_weekend', 'y_same_dominant_protein_weekend', 'y_next_dominant_starch_weekend', 'y_same_dominant_starch_weekend', 'y_next_dominant_energy_density_weekend', 'y_same_dominant_energy_density_weekend', 'y_next_dominant_satiety_style_weekend', 'y_same_dominant_satiety_style_weekend', 'y_next_weight_loss_flag', 'y_next_weight_gain_flag', 'y_next_restaurant_heavy_flag', 'y_next_budget_breach_flag', 'y_next_high_meal_frequency_flag', 'next_period_id', 'next_period_start', 'split_suggested', 'period_kind', 'period_id', 'weekend_id']

## Fold-by-fold results

| fold | best_model     | n_train | n_val | n_test | test_accuracy       | test_balanced_accuracy | test_macro_f1       | n_test_classes | test_roc_auc        |
| ---- | -------------- | ------- | ----- | ------ | ------------------- | ---------------------- | ------------------- | -------------- | ------------------- |
| 1    | et             | 32      | 8     | 6      | 0.16666666666666666 | 0.5                    | 0.14285714285714285 | 2              | 1.0                 |
| 2    | rf             | 36      | 8     | 6      | 0.6666666666666666  | 0.75                   | 0.6666666666666666  | 2              | 0.625               |
| 3    | dummy_majority | 40      | 8     | 6      | 0.3333333333333333  | 0.5                    | 0.25                | 2              | 0.5                 |
| 4    | rf             | 44      | 8     | 6      | 0.3333333333333333  | 0.3333333333333333     | 0.25                | 2              | 0.33333333333333337 |

## Feature drivers from best fold

| fold | rank | feature                                 | score                | driver_type     |
| ---- | ---- | --------------------------------------- | -------------------- | --------------- |
| 2    | 1    | num__wind_gusts_10m_max_mean            | 0.019676485559052135 | tree_importance |
| 2    | 2    | num__distinct_cuisines_sum              | 0.017545678803827874 | tree_importance |
| 2    | 3    | num__wind_gusts_10m_max_sum             | 0.015226422755611773 | tree_importance |
| 2    | 4    | num__weight_velocity_30d_lb_mean        | 0.012818678329333104 | tree_importance |
| 2    | 5    | num__noom_food_fiber_g_max              | 0.012258372299461277 | tree_importance |
| 2    | 6    | num__meal_indulgence_score_mean_sum     | 0.011934894337321155 | tree_importance |
| 2    | 7    | num__steps_gap_samsung_minus_noom_mean  | 0.010742411887538425 | tree_importance |
| 2    | 8    | num__wind_speed_10m_max_mean            | 0.01059058250194446  | tree_importance |
| 2    | 9    | num__distinct_cuisines_mean             | 0.010333518408803729 | tree_importance |
| 2    | 10   | num__meal_fat_g_sum_max                 | 0.009737317826901236 | tree_importance |
| 2    | 11   | num__restaurant_specific_meal_count_sum | 0.009712345152986333 | tree_importance |
| 2    | 12   | num__meal_indulgence_score_sum_mean     | 0.009684438265657767 | tree_importance |
| 2    | 13   | num__noom_food_fat_g_max                | 0.009660215839635434 | tree_importance |
| 2    | 14   | num__meal_protein_g_sum_mean            | 0.00886126112735073  | tree_importance |
| 2    | 15   | num__rain_streak_days_sum               | 0.008736107521497343 | tree_importance |
