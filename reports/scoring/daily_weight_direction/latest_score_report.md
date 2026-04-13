# Daily Weight Direction Score

- scored day: 2026-03-29
- timestamp column: 2026-03-29 00:00:00

## Score summary

| target                  | best_model | primary_channel | primary_probability | primary_threshold | primary_band | experimental_probability |
| ----------------------- | ---------- | --------------- | ------------------- | ----------------- | ------------ | ------------------------ |
| y_next_weight_gain_flag | et         | saved_raw       | 0.5329672061009567  | 0.5               | high         | 0.6666666666666666       |
| y_next_weight_loss_flag | et         | clean_isotonic  | 0.0                 | 0.25              | low          | 0.3788514831221851       |

## y_next_weight_gain_flag

next-day weight gain risk looks elevated today. This is based on the saved ET model's raw score. Local driver proxies suggest: `is_weekend` is above its recent baseline; `day_of_week_num` is above its recent baseline; `day_of_week` is in an active categorical state; `noom_food_carbs_g` is below its recent baseline.

## y_next_weight_loss_flag

next-day weight loss risk looks relatively low today. This is based on the clean train→val→test isotonic-calibrated channel. Local driver proxies suggest: `is_weekend` is above its recent baseline; `day_of_week_num` is above its recent baseline; `meal_hours_since_prior_meal_sum` is below its recent baseline; `meal_hours_until_next_meal_sum` is below its recent baseline.
