# Rolling Backtest Report: weeks / y_next_weight_gain_flag

- target kind: classification
- folds: 5
- models chosen: {'dummy_majority': 3, 'logreg': 2}

## Aggregate metrics
- test_accuracy_mean: 0.8667
- test_accuracy_std: 0.1247
- test_accuracy_min: 0.6667
- test_accuracy_max: 1.0000
- test_balanced_accuracy_mean: 0.7500
- test_balanced_accuracy_std: 0.2236
- test_balanced_accuracy_min: 0.5000
- test_balanced_accuracy_max: 1.0000
- test_macro_f1_mean: 0.7265
- test_macro_f1_std: 0.2580
- test_macro_f1_min: 0.4000
- test_macro_f1_max: 1.0000
- test_roc_auc_mean: 0.6875
- test_roc_auc_std: 0.2073
- test_roc_auc_min: 0.5000
- test_roc_auc_max: 1.0000
- space: weeks
- target: y_next_weight_gain_flag
- kind: classification
- min_train: 36
- val_size: 8
- test_size: 6
- step: 4
- dropped_feature_cols: ['y_next_weight_delta_lb', 'y_delta_next_weight_delta_lb', 'y_next_meal_events_per_day_week', 'y_delta_next_meal_events_per_day_week', 'y_next_restaurant_meal_fraction_week', 'y_delta_next_restaurant_meal_fraction_week', 'y_next_budget_minus_logged_food_kcal_week', 'y_delta_next_budget_minus_logged_food_kcal_week', 'y_next_dominant_meal_archetype_week', 'y_same_dominant_meal_archetype_week', 'y_next_dominant_cuisine_week', 'y_same_dominant_cuisine_week', 'y_next_dominant_service_form_week', 'y_same_dominant_service_form_week', 'y_next_dominant_prep_profile_week', 'y_same_dominant_prep_profile_week', 'y_next_dominant_protein_week', 'y_same_dominant_protein_week', 'y_next_dominant_starch_week', 'y_same_dominant_starch_week', 'y_next_dominant_energy_density_week', 'y_same_dominant_energy_density_week', 'y_next_dominant_satiety_style_week', 'y_same_dominant_satiety_style_week', 'y_next_weight_loss_flag', 'y_next_weight_gain_flag', 'y_next_restaurant_heavy_flag', 'y_next_budget_breach_flag', 'y_next_high_meal_frequency_flag', 'next_period_id', 'next_period_start', 'split_suggested', 'period_kind', 'period_id', 'week_id']

## Fold-by-fold results

| fold | best_model     | n_train | n_val | n_test | test_accuracy      | test_balanced_accuracy | test_macro_f1       | n_test_classes | test_roc_auc |
| ---- | -------------- | ------- | ----- | ------ | ------------------ | ---------------------- | ------------------- | -------------- | ------------ |
| 1    | dummy_majority | 32      | 8     | 6      | 1.0                | 1.0                    | 1.0                 | 1              |              |
| 2    | dummy_majority | 36      | 8     | 6      | 0.8333333333333334 | 0.5                    | 0.45454545454545453 | 2              | 0.5          |
| 3    | dummy_majority | 40      | 8     | 6      | 0.6666666666666666 | 0.5                    | 0.4                 | 2              | 0.5          |
| 4    | logreg         | 44      | 8     | 6      | 0.8333333333333334 | 0.75                   | 0.7777777777777777  | 2              | 0.75         |
| 5    | logreg         | 48      | 8     | 5      | 1.0                | 1.0                    | 1.0                 | 2              | 1.0          |
