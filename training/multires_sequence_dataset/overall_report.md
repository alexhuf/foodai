# Multi-Resolution Sequence Dataset Build

## Source manifest
- daily_transition_csv: E:\Users\Alex\Documents\GitHub\foodai\training\daily_transition\days_transition_matrix.csv
- weekly_transition_csv: E:\Users\Alex\Documents\GitHub\foodai\training\regime_transition\weeks_transition_matrix.csv
- meal_source_csv: None
- meal_source_detected: False
- daily_targets: ['y_next_weight_gain_flag', 'y_next_weight_loss_flag', 'y_next_weight_delta_lb', 'y_next_logged_food_kcal_day', 'y_next_restaurant_meal_fraction_day', 'y_next_budget_breach_flag']
- weekly_targets: ['y_next_weight_gain_flag', 'y_next_restaurant_heavy_flag', 'y_next_budget_breach_flag']
- meal_id_col: None
- meal_time_col: None

## Overall summary

| anchors | daily_targets | weekly_targets | day_numeric_features | meal_numeric_features | week_numeric_features | meal_modality_detected | week_modality_detected | pct_has_meals | pct_has_days | pct_has_weeks |
| ------- | ------------- | -------------- | -------------------- | --------------------- | --------------------- | ---------------------- | ---------------------- | ------------- | ------------ | ------------- |
| 456     | 6             | 3              | 224                  | 0                     | 334                   | False                  | True                   | 0.0           | 1.0          | 1.0           |
