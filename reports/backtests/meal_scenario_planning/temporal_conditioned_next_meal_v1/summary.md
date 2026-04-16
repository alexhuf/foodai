# Temporal-Conditioned Next Meal v1

## Purpose

This bridge keeps the v2 observed-meal action space and conditions its ranking on the locked `simple_loss_daysweeks_v2` temporal state. It does not retrain the temporal winner and does not generate unconstrained meals.

## Current Context

- current_datetime: `2026-04-16T12:00:00`
- latest_observed_date: `2026-03-29`
- latest_weight_lb: `None`
- latest_weight_velocity_7d_lb: `0.0`
- recent_steps_mean: `2210.8571428571427`
- recent_food_kcal_mean: `2129.3571428571427`
- recent_dominant_archetypes: `beverage_only, dessert_snack`

## Locked Temporal State

- anchor_id: `2026-03-29`
- score: `0.381743`
- locked_threshold: `0.4288`
- locked_decision: `negative`
- policy_band: `below 0.4288`
- interpretation: `ranking / threshold signal only; not a calibrated probability`
- low_loss_support_pressure: `0.110`

## Ranked Recommendation

| bridge_rank | meal_cluster_id | current_slot | archetype | meal_text | cluster_kcal_median | bridge_score | next_action_score | meal_enjoyment | meal_health | projected_robust_weight_support | temporal_adjustment | bridge_explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | cluster_000 | lunch | mixed_plate | Carne Asada Street Tacos + Mexican Rice And Beans | 826.000 | 0.823 | 0.783 | 0.704 | 0.715 | 0.844 | 0.040 | current temporal state is below the locked loss-support threshold; health score is strong; projected day support is robust; enjoyment/familiarity remains high |
| 2 | cluster_002 | lunch | mixed_plate | Pasta with Meat Sauce + White Tortilla | 396.000 | 0.799 | 0.762 | 0.717 | 0.589 | 0.844 | 0.037 | current temporal state is below the locked loss-support threshold; projected day support is robust; enjoyment/familiarity remains high |
| 3 | cluster_001 | lunch | mixed_plate | Chicken Shawarma Pita + French Fries | 787.500 | 0.798 | 0.762 | 0.728 | 0.573 | 0.844 | 0.037 | current temporal state is below the locked loss-support threshold; projected day support is robust; enjoyment/familiarity remains high |
| 4 | cluster_003 | lunch | mixed_plate | Philly Steak And Cheese Sub + French Fries | 745.000 | 0.794 | 0.758 | 0.730 | 0.549 | 0.844 | 0.036 | current temporal state is below the locked loss-support threshold; projected day support is robust; enjoyment/familiarity remains high |
| 5 | cluster_004 | lunch | mixed_plate | Bbq Chicken + Mashed Potatoes | 646.000 | 0.787 | 0.752 | 0.710 | 0.548 | 0.844 | 0.036 | current temporal state is below the locked loss-support threshold; projected day support is robust; enjoyment/familiarity remains high |
| 6 | cluster_005 | lunch | mixed_plate | Fried Chicken Breast + Mashed Potatoes | 601.000 | 0.785 | 0.749 | 0.701 | 0.550 | 0.844 | 0.036 | current temporal state is below the locked loss-support threshold; projected day support is robust; enjoyment/familiarity remains high |
| 7 | cluster_006 | lunch | mixed_plate | Beef Vegetable Stir Fry + Steamed White Rice | 634.500 | 0.778 | 0.744 | 0.717 | 0.550 | 0.844 | 0.035 | current temporal state is below the locked loss-support threshold; projected day support is robust; enjoyment/familiarity remains high |
| 8 | cluster_007 | lunch | mixed_plate | Panda Express Chow Mein + Panda Express Super Greens + Panda Express Beef Black Pepper Angus Steak | 660.000 | 0.777 | 0.743 | 0.713 | 0.500 | 0.844 | 0.034 | current temporal state is below the locked loss-support threshold; projected day support is robust; enjoyment/familiarity remains high |
| 9 | cluster_008 | lunch | mixed_plate | Baked Chicken Wing | 831.000 | 0.772 | 0.739 | 0.722 | 0.466 | 0.844 | 0.034 | current temporal state is below the locked loss-support threshold; projected day support is robust; enjoyment/familiarity remains high |
| 10 | cluster_009 | lunch | mixed_plate | Chicken Quesadilla + Tortilla Chip | 1116.500 | 0.769 | 0.736 | 0.717 | 0.512 | 0.844 | 0.034 | current temporal state is below the locked loss-support threshold; projected day support is robust; enjoyment/familiarity remains high |
| 11 | cluster_010 | lunch | mixed_plate | Scrambled Eggs + Bacon | 451.000 | 0.769 | 0.735 | 0.671 | 0.523 | 0.844 | 0.035 | current temporal state is below the locked loss-support threshold; projected day support is robust |
| 12 | cluster_011 | lunch | mixed_plate | Panda Express Chow Mein + Panda Express Orange Chicken + Panda Express Super Greens + Panda Express Beef Black Pepper Angus Steak | 969.000 | 0.765 | 0.732 | 0.728 | 0.425 | 0.844 | 0.033 | current temporal state is below the locked loss-support threshold; projected day support is robust; enjoyment/familiarity remains high |

## Recommended Action

- meal_cluster_id: `cluster_000`
- representative meal_action_id: `meal_00031`
- slot: `lunch`
- archetype: `mixed_plate`
- representative observed example: `Carne Asada Street Tacos + Mexican Rice And Beans`
- bridge_score: `0.823`
- original_v2_score: `0.783`
- why: current temporal state is below the locked loss-support threshold; health score is strong; projected day support is robust; enjoyment/familiarity remains high
- portion guidance: Observed lunch examples in this archetype cluster range 614-1173 kcal; choose a portion near 826 kcal unless already over budget.
- projected day template: `day_2025-05-22__lighter_observed_portion`

## Bundle Files

- bridge-ranked meals: `/workspace/foodai/reports/backtests/meal_scenario_planning/temporal_conditioned_next_meal_v1/temporal_conditioned_next_meal_scores.csv`
- projection stress tests: `/workspace/foodai/reports/backtests/meal_scenario_planning/temporal_conditioned_next_meal_v1/temporal_conditioned_projection_stress_tests.csv`
- bridge manifest: `/workspace/foodai/reports/backtests/meal_scenario_planning/temporal_conditioned_next_meal_v1/temporal_conditioned_next_meal_manifest.json`
