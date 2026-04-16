# Next Meal Scenario Scoring v2

## What Changed From v1

- keeps the observed-meal realism constraint: recommendations are clusters of historical meal records, not generated meals
- de-duplicates near-identical options by archetype, service form, protein anchor, and canonical components
- reports observed calorie ranges for each cluster as bounded portion guidance
- adds plain-language explanations for why each option scored well

## Context

- current_datetime: `2026-04-16 12:00`
- planning_start_date: `2026-03-30`
- latest_observed_date: `2026-03-29`
- recent_steps_mean: `2210.8571428571427`
- recent_food_kcal_mean: `2129.3571428571427`

## Ranked Next-Meal Option Clusters

| meal_cluster_id | current_slot | archetype | meal_text | cluster_observed_examples | cluster_kcal_min | cluster_kcal_median | cluster_kcal_max | next_action_score | meal_health | projected_robust_weight_support | plain_language_explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cluster_000 | lunch | mixed_plate | Carne Asada Street Tacos + Mexican Rice And Beans | 8 | 614.000 | 818.500 | 925.000 | 0.783 | 0.715 | 0.844 | solid protein/health profile; familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |
| cluster_001 | lunch | mixed_plate | Chicken Shawarma Pita + French Fries | 5 | 697.000 | 735.000 | 840.000 | 0.762 | 0.573 | 0.844 | familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |
| cluster_002 | lunch | mixed_plate | Pasta with Meat Sauce + White Tortilla | 1 | 396.000 | 396.000 | 396.000 | 0.762 | 0.589 | 0.844 | familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |
| cluster_003 | lunch | mixed_plate | Philly Steak And Cheese Sub + French Fries | 1 | 745.000 | 745.000 | 745.000 | 0.758 | 0.549 | 0.844 | familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |
| cluster_004 | lunch | mixed_plate | Bbq Chicken + Mashed Potatoes | 1 | 646.000 | 646.000 | 646.000 | 0.752 | 0.548 | 0.844 | familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |
| cluster_005 | lunch | mixed_plate | Fried Chicken Breast + Mashed Potatoes | 1 | 601.000 | 601.000 | 601.000 | 0.749 | 0.550 | 0.844 | familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |
| cluster_006 | lunch | mixed_plate | Beef Vegetable Stir Fry + Steamed White Rice | 2 | 460.000 | 634.500 | 809.000 | 0.744 | 0.550 | 0.844 | familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |
| cluster_007 | lunch | mixed_plate | Panda Express Chow Mein + Panda Express Super Greens + Panda Express Beef Black Pepper Angus Steak | 5 | 585.000 | 660.000 | 708.000 | 0.743 | 0.500 | 0.844 | familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |
| cluster_008 | lunch | mixed_plate | Baked Chicken Wing | 2 | 704.000 | 831.000 | 958.000 | 0.739 | 0.466 | 0.844 | familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |
| cluster_009 | lunch | mixed_plate | Chicken Quesadilla + Tortilla Chip | 1 | 810.000 | 810.000 | 810.000 | 0.736 | 0.512 | 0.844 | familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |
| cluster_010 | lunch | mixed_plate | Scrambled Eggs + Bacon | 1 | 451.000 | 451.000 | 451.000 | 0.735 | 0.523 | 0.844 | links to day templates with robust weight support; low stress-test fragility |
| cluster_011 | lunch | mixed_plate | Panda Express Chow Mein + Panda Express Orange Chicken + Panda Express Super Greens + Panda Express Beef Black Pepper Angus Steak | 3 | 733.000 | 858.000 | 969.000 | 0.732 | 0.425 | 0.844 | familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility |

## Recommended Action

- meal_cluster_id: `cluster_000`
- representative meal_action_id: `meal_00031`
- slot: `lunch`
- archetype: `mixed_plate`
- representative observed example: `Carne Asada Street Tacos + Mexican Rice And Beans`
- score: `0.783`
- why: solid protein/health profile; familiar high-fit lunch pattern; links to day templates with robust weight support; low stress-test fragility
- portion guidance: Observed lunch examples in this archetype cluster range 614-925 kcal; choose a portion near 818 kcal unless already over budget.
- projected day template: `day_2025-05-22__lighter_observed_portion`
- projected day pattern: `breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal)`

## Bundle Files

- scored option clusters: `/workspace/foodai/reports/backtests/meal_scenario_planning/next_meal_scenario_scoring_v2/next_meal_scores.csv`
- projection stress tests: `/workspace/foodai/reports/backtests/meal_scenario_planning/next_meal_scenario_scoring_v2/next_meal_projection_stress_tests.csv`
- manifest: `/workspace/foodai/reports/backtests/meal_scenario_planning/next_meal_scenario_scoring_v2/next_meal_manifest.json`
