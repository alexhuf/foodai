# Regime Representation Audit

## weeks_structure

- Source: `E:\Users\Alex\Documents\GitHub\foodai\training\week_summary_matrix.csv`
- Model dir: `E:\Users\Alex\Documents\GitHub\foodai\models\representation_v3_2_1\weeks_structure`
- Neutralized columns: 8

### Split sizes
- train: 53
- val: 7
- test: 7

### Flags
- weeks_structure/weight_delta_lb: representation regressor still has negative test R² (-0.063).
- weeks_structure/restaurant_meal_fraction_week: representation regressor still has negative test R² (-0.091).
- weeks_structure/budget_minus_logged_food_kcal_week: representation regressor still has negative test R² (-0.219).

## weekends_structure

- Source: `E:\Users\Alex\Documents\GitHub\foodai\training\weekend_summary_matrix.csv`
- Model dir: `E:\Users\Alex\Documents\GitHub\foodai\models\representation_v3_2_1\weekends_structure`
- Neutralized columns: 8

### Split sizes
- train: 52
- val: 7
- test: 7

### Flags
- weekends_structure/dominant_cuisine_weekend: test split has only one class.
- weekends_structure/dominant_cuisine_weekend: both representation and simple proxy are near-perfect; target may still be too easy.
- weekends_structure/weight_delta_lb: proxy regressor still has negative test R² (-2.910).
- weekends_structure/weight_delta_lb: representation regressor still has negative test R² (-15.373).
- weekends_structure/meal_events_per_day_weekend: representation regressor still has negative test R² (-12.778).
- weekends_structure/restaurant_meal_fraction_weekend: representation regressor still has negative test R² (-0.206).
- weekends_structure/budget_minus_logged_food_kcal_weekend: representation regressor still has negative test R² (-0.182).
