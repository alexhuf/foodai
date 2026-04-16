# Next Meal Scenario Scoring v1

## Mode

- action: observed meal records only, filtered by the inferred current slot and historically repeated archetype/slot combinations
- projection: each next-meal option is linked to observed full-day templates containing that archetype, then stress-tested under the same robustness logic as horizon planning
- reward: meal enjoyment, meal health, robust projected weight-support, routine frequency, and realism

## Context

- current_datetime: `2026-04-16 12:00`
- planning_start_date: `2026-03-30`
- latest_observed_date: `2026-03-29`
- latest_weight_lb: `None`
- latest_weight_velocity_7d_lb: `0.0`
- recent_steps_mean: `2210.8571428571427`
- recent_food_kcal_mean: `2129.3571428571427`

## Ranked Next-Meal Options

| meal_action_id | current_slot | archetype | calories_kcal | protein_g | next_action_score | meal_enjoyment | meal_health | projected_robust_weight_support | projected_fragility | meal_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meal_00031 | lunch | mixed_plate | 614.000 | 42.000 | 0.771 | 0.704 | 0.715 | 0.797 | 0.085 | Carne Asada Street Tacos + Mexican Rice And Beans |
| meal_00875 | lunch | mixed_plate | 701.000 | 41.636 | 0.765 | 0.721 | 0.658 | 0.797 | 0.085 | Carne Asada Street Tacos + Tortilla Chip + Mexican Rice And Beans |
| meal_00137 | lunch | mixed_plate | 811.000 | 58.000 | 0.764 | 0.717 | 0.715 | 0.797 | 0.085 | Carne Asada Street Tacos + Mexican Rice And Beans |
| meal_00539 | lunch | mixed_plate | 662.000 | 39.737 | 0.754 | 0.713 | 0.616 | 0.797 | 0.085 | Carne Asada Street Tacos + Tortilla Chip |
| meal_00145 | lunch | mixed_plate | 735.000 | 44.308 | 0.750 | 0.728 | 0.573 | 0.797 | 0.085 | Chicken Shawarma Pita + French Fries |
| meal_00140 | lunch | mixed_plate | 396.000 | 17.003 | 0.750 | 0.717 | 0.589 | 0.797 | 0.085 | Pasta with Meat Sauce + White Tortilla |
| meal_01118 | lunch | mixed_plate | 697.000 | 43.607 | 0.748 | 0.720 | 0.573 | 0.797 | 0.085 | Chicken Shawarma Pita + French Fries |
| meal_01036 | lunch | mixed_plate | 697.000 | 43.607 | 0.748 | 0.720 | 0.573 | 0.797 | 0.085 | Chicken Shawarma Pita + French Fries |
| meal_00513 | lunch | mixed_plate | 826.000 | 45.189 | 0.747 | 0.714 | 0.644 | 0.797 | 0.085 | Carne Asada Street Tacos + Mexican Rice And Beans + Tortilla Chip |
| meal_00789 | lunch | mixed_plate | 745.000 | 32.955 | 0.746 | 0.730 | 0.549 | 0.797 | 0.085 | Philly Steak And Cheese Sub + French Fries |
| meal_00730 | lunch | mixed_plate | 646.000 | 37.119 | 0.740 | 0.710 | 0.548 | 0.797 | 0.085 | Bbq Chicken + Mashed Potatoes |
| meal_00705 | lunch | mixed_plate | 601.000 | 38.887 | 0.737 | 0.701 | 0.550 | 0.797 | 0.085 | Fried Chicken Breast + Mashed Potatoes |

## Recommended Action

- meal_action_id: `meal_00031`
- slot: `lunch`
- archetype: `mixed_plate`
- calories_kcal: `614`
- score: `0.771`
- observed example: `Carne Asada Street Tacos + Mexican Rice And Beans`
- projected day template: `day_2025-05-22`
- projected day pattern: `breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal)`

## Bundle Files

- scored options: `/workspace/foodai/reports/backtests/meal_scenario_planning/next_meal_scenario_scoring_v1/next_meal_scores.csv`
- projection stress tests: `/workspace/foodai/reports/backtests/meal_scenario_planning/next_meal_scenario_scoring_v1/next_meal_projection_stress_tests.csv`
- manifest: `/workspace/foodai/reports/backtests/meal_scenario_planning/next_meal_scenario_scoring_v1/next_meal_manifest.json`
