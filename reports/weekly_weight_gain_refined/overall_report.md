# Weekly Weight-Gain Refined Baseline Report

- target: weeks / y_next_weight_gain_flag
- kept feature groups: ['biology', 'meals', 'weather_daylight']

## Aggregate metrics
- target_space: weeks
- target_name: y_next_weight_gain_flag
- kept_feature_groups: ['biology', 'meals', 'weather_daylight']
- dropped_feature_cols: ['y_next_weight_delta_lb', 'y_delta_next_weight_delta_lb', 'y_next_meal_events_per_day_week', 'y_delta_next_meal_events_per_day_week', 'y_next_restaurant_meal_fraction_week', 'y_delta_next_restaurant_meal_fraction_week', 'y_next_budget_minus_logged_food_kcal_week', 'y_delta_next_budget_minus_logged_food_kcal_week', 'y_next_dominant_meal_archetype_week', 'y_same_dominant_meal_archetype_week', 'y_next_dominant_cuisine_week', 'y_same_dominant_cuisine_week', 'y_next_dominant_service_form_week', 'y_same_dominant_service_form_week', 'y_next_dominant_prep_profile_week', 'y_same_dominant_prep_profile_week', 'y_next_dominant_protein_week', 'y_same_dominant_protein_week', 'y_next_dominant_starch_week', 'y_same_dominant_starch_week', 'y_next_dominant_energy_density_week', 'y_same_dominant_energy_density_week', 'y_next_dominant_satiety_style_week', 'y_same_dominant_satiety_style_week', 'y_next_weight_loss_flag', 'y_next_weight_gain_flag', 'y_next_restaurant_heavy_flag', 'y_next_budget_breach_flag', 'y_next_high_meal_frequency_flag', 'next_period_id', 'next_period_start', 'split_suggested', 'period_kind', 'period_id', 'week_id']
- model_path: E:\Users\Alex\Documents\GitHub\foodai\models\weekly_weight_gain_refined\logreg_refined.joblib
- n_training_rows: 61
- positive_rate_training: 0.1475
- rolling_folds: 2
- n_rows: 2
- fold_mean: 1.5000
- fold_std: 0.5000
- fold_min: 1.0000
- fold_max: 2.0000
- threshold_balanced_accuracy_mean: 0.0968
- threshold_balanced_accuracy_std: 0.0467
- threshold_balanced_accuracy_min: 0.0500
- threshold_balanced_accuracy_max: 0.1435
- threshold_macro_f1_mean: 0.0968
- threshold_macro_f1_std: 0.0467
- threshold_macro_f1_min: 0.0500
- threshold_macro_f1_max: 0.1435
- default_accuracy_mean: 0.8750
- default_accuracy_std: 0.0000
- default_accuracy_min: 0.8750
- default_accuracy_max: 0.8750
- default_balanced_accuracy_mean: 0.6250
- default_balanced_accuracy_std: 0.1250
- default_balanced_accuracy_min: 0.5000
- default_balanced_accuracy_max: 0.7500
- default_macro_f1_mean: 0.6308
- default_macro_f1_std: 0.1641
- default_macro_f1_min: 0.4667
- default_macro_f1_max: 0.7949
- default_positive_rate_pred_mean: 0.0625
- default_positive_rate_pred_std: 0.0625
- default_positive_rate_pred_min: 0.0000
- default_positive_rate_pred_max: 0.1250
- default_positive_rate_true_mean: 0.1875
- default_positive_rate_true_std: 0.0625
- default_positive_rate_true_min: 0.1250
- default_positive_rate_true_max: 0.2500
- default_roc_auc_mean: 0.8333
- default_roc_auc_std: 0.1667
- default_roc_auc_min: 0.6667
- default_roc_auc_max: 1.0000
- default_brier_mean: 0.0796
- default_brier_std: 0.0455
- default_brier_min: 0.0342
- default_brier_max: 0.1251
- tuned_bal_accuracy_mean: 0.9375
- tuned_bal_accuracy_std: 0.0625
- tuned_bal_accuracy_min: 0.8750
- tuned_bal_accuracy_max: 1.0000
- tuned_bal_balanced_accuracy_mean: 0.8750
- tuned_bal_balanced_accuracy_std: 0.1250
- tuned_bal_balanced_accuracy_min: 0.7500
- tuned_bal_balanced_accuracy_max: 1.0000
- tuned_bal_macro_f1_mean: 0.8974
- tuned_bal_macro_f1_std: 0.1026
- tuned_bal_macro_f1_min: 0.7949
- tuned_bal_macro_f1_max: 1.0000
- tuned_bal_positive_rate_pred_mean: 0.1250
- tuned_bal_positive_rate_pred_std: 0.0000
- tuned_bal_positive_rate_pred_min: 0.1250
- tuned_bal_positive_rate_pred_max: 0.1250
- tuned_bal_positive_rate_true_mean: 0.1875
- tuned_bal_positive_rate_true_std: 0.0625
- tuned_bal_positive_rate_true_min: 0.1250
- tuned_bal_positive_rate_true_max: 0.2500
- tuned_bal_roc_auc_mean: 0.8333
- tuned_bal_roc_auc_std: 0.1667
- tuned_bal_roc_auc_min: 0.6667
- tuned_bal_roc_auc_max: 1.0000
- tuned_bal_brier_mean: 0.0796
- tuned_bal_brier_std: 0.0455
- tuned_bal_brier_min: 0.0342
- tuned_bal_brier_max: 0.1251
- train_positive_rate_mean: 0.1151
- train_positive_rate_std: 0.0040
- train_positive_rate_min: 0.1111
- train_positive_rate_max: 0.1190
- val_positive_rate_mean: 0.1250
- val_positive_rate_std: 0.0000
- val_positive_rate_min: 0.1250
- val_positive_rate_max: 0.1250
- test_positive_rate_mean: 0.1875
- test_positive_rate_std: 0.0625
- test_positive_rate_min: 0.1250
- test_positive_rate_max: 0.2500
- train_n_mean: 39.0000
- train_n_std: 3.0000
- train_n_min: 36.0000
- train_n_max: 42.0000
- val_n_mean: 8.0000
- val_n_std: 0.0000
- val_n_min: 8.0000
- val_n_max: 8.0000
- test_n_mean: 8.0000
- test_n_std: 0.0000
- test_n_min: 8.0000
- test_n_max: 8.0000

## Rolling fold metrics

| fold | threshold_balanced_accuracy | threshold_macro_f1 | default_accuracy | default_balanced_accuracy | default_macro_f1   | default_positive_rate_pred | default_positive_rate_true | default_roc_auc    | default_brier        | tuned_bal_accuracy | tuned_bal_balanced_accuracy | tuned_bal_macro_f1 | tuned_bal_positive_rate_pred | tuned_bal_positive_rate_true | tuned_bal_roc_auc  | tuned_bal_brier      | train_positive_rate | val_positive_rate | test_positive_rate | train_n | val_n | test_n |
| ---- | --------------------------- | ------------------ | ---------------- | ------------------------- | ------------------ | -------------------------- | -------------------------- | ------------------ | -------------------- | ------------------ | --------------------------- | ------------------ | ---------------------------- | ---------------------------- | ------------------ | -------------------- | ------------------- | ----------------- | ------------------ | ------- | ----- | ------ |
| 1    | 0.1435                      | 0.1435             | 0.875            | 0.5                       | 0.4666666666666667 | 0.0                        | 0.125                      | 1.0                | 0.034150034964054464 | 1.0                | 1.0                         | 1.0                | 0.125                        | 0.125                        | 1.0                | 0.034150034964054464 | 0.1111111111111111  | 0.125             | 0.125              | 36      | 8     | 8      |
| 2    | 0.05                        | 0.05               | 0.875            | 0.75                      | 0.7948717948717949 | 0.125                      | 0.25                       | 0.6666666666666667 | 0.1251413245689146   | 0.875              | 0.75                        | 0.7948717948717949 | 0.125                        | 0.25                         | 0.6666666666666667 | 0.1251413245689146   | 0.11904761904761904 | 0.125             | 0.25               | 42      | 8     | 8      |

## Top coefficients

| rank | feature                                              | coefficient          | direction |
| ---- | ---------------------------------------------------- | -------------------- | --------- |
| 1    | num__samsung_sleep_score_sum                         | 0.3519904503269743   | positive  |
| 2    | num__meal_dessert_component_count_from_roles_sum_max | -0.3497721483902372  | negative  |
| 3    | num__samsung_sleep_duration_ms_sum                   | 0.33618735740559763  | positive  |
| 4    | num__meal_starch_base_count_sum_mean                 | 0.29403151806597577  | positive  |
| 5    | num__samsung_sleep_duration_ms_max                   | -0.28885873867046    | negative  |
| 6    | num__meal_main_component_count_sum_max               | -0.27984148416028964 | negative  |
| 7    | num__samsung_sleep_duration_ms_mean                  | -0.27351376394248217 | negative  |
| 8    | num__rain_sum_max                                    | 0.271510792611458    | positive  |
| 9    | num__samsung_sleep_score_max                         | -0.2695591022940471  | negative  |
| 10   | num__meal_starch_base_count_sum_sum                  | 0.25905278214417937  | positive  |
| 11   | num__restaurant_meal_fraction_week                   | -0.25368205406140165 | negative  |
| 12   | num__meal_fiber_g_sum_mean                           | 0.2532059223797545   | positive  |
| 13   | num__noom_food_fiber_g_mean                          | 0.2532059223797544   | positive  |
| 14   | num__meal_starch_base_count_sum_max                  | 0.2526441493101051   | positive  |
| 15   | num__meal_dessert_component_count_from_roles_sum_sum | -0.2500620866917205  | negative  |
| 16   | num__precipitation_sum_max                           | 0.2487614081334074   | positive  |
| 17   | num__samsung_sleep_score_mean                        | -0.24693310872359336 | negative  |
| 18   | num__noom_meal_event_count_max                       | 0.23966724703802725  | positive  |
| 19   | num__meal_event_count_max                            | 0.23966724703802725  | positive  |
| 20   | num__snow_streak_days_mean                           | 0.23231123870318615  | positive  |
