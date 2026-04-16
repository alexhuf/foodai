# Meal Scenario Planning v1

## Planning Matrix

- state: latest observed weight/trend, recent steps, recent intake, recent restaurant intensity, recent dominant archetypes, start date, weekday/weekend, and season
- action: observed full-day meal templates only, filtered to days that include lunch, dinner, and at least one snack period
- reward: 22% enjoyment, 20% healthfulness, 20% consistency, 28% weight-support, 10% realism
- robustness: re-score each candidate under low/high steps, weekday/weekend shifts, adjacent season, and recent heavier/lighter intake
- promotion: required slots present, calories inside the observed core band, repeat frequency controlled, robust score >= 0.50, robust weight support >= 0.45, fragility <= 0.22

## Context

- planning_start_date: `2026-03-30`
- latest_observed_date: `2026-03-29`
- latest_weight_lb: `None`
- latest_weight_velocity_7d_lb: `0.0`
- recent_steps_mean: `2210.8571428571427`
- recent_food_kcal_mean: `2129.3571428571427`
- recent_dominant_archetypes: `beverage_only, dessert_snack`

## Action Space

- observed templates after required-slot filter: `229`
- core calorie band q05/q95: `1363` to `2643` kcal

## Ranked Promoted Plans

| plan_id | horizon_days | strategy | robust_score | robust_weight_support | fragility | enjoyment | health | consistency | mean_kcal | mean_steps | restaurant_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scenario_h3_0035 | 3 | sampled | 0.853 | 0.838 | 0.080 | 0.978 | 0.973 | 0.811 | 1706.667 | 3975.333 | 0.500 |
| scenario_h3_0003 | 3 | enjoyment | 0.848 | 0.819 | 0.078 | 0.986 | 0.970 | 0.811 | 1856.333 | 4765.667 | 0.500 |
| scenario_h3_0009 | 3 | sampled | 0.817 | 0.764 | 0.078 | 0.991 | 0.886 | 0.811 | 1829.333 | 4467.333 | 0.500 |
| scenario_h3_0022 | 3 | sampled | 0.817 | 0.764 | 0.078 | 0.991 | 0.886 | 0.811 | 1829.333 | 4467.333 | 0.500 |
| scenario_h3_0025 | 3 | sampled | 0.817 | 0.764 | 0.078 | 0.991 | 0.886 | 0.811 | 1829.333 | 4467.333 | 0.500 |
| scenario_h3_0032 | 3 | sampled | 0.817 | 0.785 | 0.081 | 0.978 | 0.864 | 0.811 | 1663.667 | 3967.000 | 0.500 |
| scenario_h3_0040 | 3 | sampled | 0.800 | 0.648 | 0.082 | 0.986 | 0.973 | 0.811 | 1769.667 | 3670.000 | 0.417 |
| scenario_h3_0019 | 3 | sampled | 0.772 | 0.847 | 0.083 | 0.736 | 0.812 | 0.811 | 1581.000 | 3495.333 | 0.417 |
| scenario_h3_0033 | 3 | sampled | 0.770 | 0.863 | 0.084 | 0.735 | 0.779 | 0.811 | 1576.333 | 3327.667 | 0.583 |
| scenario_h3_0034 | 3 | sampled | 0.766 | 0.828 | 0.078 | 0.745 | 0.809 | 0.811 | 1730.667 | 4285.667 | 0.417 |
| scenario_h3_0001 | 3 | weight | 0.764 | 0.845 | 0.079 | 0.744 | 0.776 | 0.811 | 1726.000 | 4118.000 | 0.583 |
| scenario_h3_0028 | 3 | sampled | 0.763 | 0.829 | 0.074 | 0.785 | 0.747 | 0.811 | 1859.000 | 4676.000 | 0.583 |
| scenario_h3_0031 | 3 | sampled | 0.759 | 0.797 | 0.078 | 0.810 | 0.784 | 0.771 | 1940.000 | 4193.000 | 0.667 |
| scenario_h3_0038 | 3 | sampled | 0.759 | 0.847 | 0.077 | 0.739 | 0.751 | 0.811 | 1710.000 | 4408.000 | 0.583 |
| scenario_h3_0039 | 3 | sampled | 0.758 | 0.766 | 0.084 | 0.754 | 0.833 | 0.811 | 1664.000 | 3287.000 | 0.500 |
| scenario_h3_0030 | 3 | sampled | 0.756 | 0.836 | 0.084 | 0.760 | 0.760 | 0.771 | 1657.333 | 3350.000 | 0.583 |
| scenario_h3_0027 | 3 | sampled | 0.755 | 0.825 | 0.083 | 0.753 | 0.783 | 0.771 | 1743.333 | 3676.000 | 0.500 |
| scenario_h3_0018 | 3 | sampled | 0.752 | 0.810 | 0.081 | 0.745 | 0.761 | 0.811 | 1731.667 | 3949.333 | 0.667 |
| scenario_h3_0007 | 3 | sampled | 0.748 | 0.723 | 0.071 | 0.827 | 0.772 | 0.811 | 1994.333 | 6075.333 | 0.500 |
| scenario_h3_0023 | 3 | sampled | 0.730 | 0.775 | 0.078 | 0.745 | 0.701 | 0.811 | 1687.667 | 4277.333 | 0.417 |

## Best Plan By Horizon

### 3 days

- plan_id: `scenario_h3_0035`
- robust_score: `0.853`
- robust_weight_support: `0.838`
- fragility: `0.080`
- promoted: `True`
- rejection_reasons: `none`

| day_index | planned_date | planned_day_of_week | source_date | total_kcal | steps_day | dominant_archetype | loss_support_raw | gain_risk_raw | slot_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-30 | Monday | 2025-05-22 | 2022.000 | 5266.000 | beverage_only | 0.776 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) |
| 2 | 2026-03-31 | Tuesday | 2025-03-12 | 1573.000 | 2895.000 | beverage_only | 0.901 | 0.000 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) |
| 3 | 2026-04-01 | Wednesday | 2025-04-14 | 1525.000 | 3765.000 | beverage_only | 0.853 | 0.000 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) |

### 5 days

- plan_id: `scenario_h5_0077`
- robust_score: `0.854`
- robust_weight_support: `0.846`
- fragility: `0.082`
- promoted: `True`
- rejection_reasons: `none`

| day_index | planned_date | planned_day_of_week | source_date | total_kcal | steps_day | dominant_archetype | loss_support_raw | gain_risk_raw | slot_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-30 | Monday | 2025-03-12 | 1573.000 | 2895.000 | beverage_only | 0.901 | 0.000 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) |
| 2 | 2026-03-31 | Tuesday | 2025-04-14 | 1525.000 | 3765.000 | beverage_only | 0.853 | 0.000 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) |
| 3 | 2026-04-01 | Wednesday | 2025-05-22 | 2022.000 | 5266.000 | beverage_only | 0.776 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) |
| 4 | 2026-04-02 | Thursday | 2025-03-12 | 1573.000 | 2895.000 | beverage_only | 0.901 | 0.000 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) |
| 5 | 2026-04-03 | Friday | 2025-04-14 | 1525.000 | 3765.000 | beverage_only | 0.853 | 0.000 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) |

### 7 days

- plan_id: `scenario_h7_0120`
- robust_score: `0.789`
- robust_weight_support: `0.758`
- fragility: `0.068`
- promoted: `True`
- rejection_reasons: `none`

| day_index | planned_date | planned_day_of_week | source_date | total_kcal | steps_day | dominant_archetype | loss_support_raw | gain_risk_raw | slot_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-30 | Monday | 2025-03-12 | 1573.000 | 2895.000 | beverage_only | 0.901 | 0.000 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) |
| 2 | 2026-03-31 | Tuesday | 2025-05-22 | 2022.000 | 5266.000 | beverage_only | 0.776 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) |
| 3 | 2026-04-01 | Wednesday | 2025-03-13 | 1635.000 | 2867.000 | mixed_plate | 0.854 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:protein_component (550 kcal); evening_snack:mixed_plate (105 kcal) |
| 4 | 2026-04-02 | Thursday | 2025-03-12 | 1573.000 | 2895.000 | beverage_only | 0.901 | 0.000 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) |
| 5 | 2026-04-03 | Friday | 2025-04-14 | 1525.000 | 3765.000 | beverage_only | 0.853 | 0.000 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) |
| 6 | 2026-04-04 | Saturday | 2026-03-15 | 2593.000 | 3045.000 | beverage_only | 0.297 | 0.000 | breakfast:beverage_only (10 kcal); lunch:deli_sandwich (1167 kcal); dinner:sandwich_meal (1070 kcal); evening_snack:berries (346 kcal) |
| 7 | 2026-04-05 | Sunday | 2025-05-11 | 2178.000 | 2772.000 | beverage_only | 0.779 | 0.000 | breakfast:beverage_only (10 kcal); lunch:breakfast_plate (763 kcal); dinner:mixed_plate (685 kcal); evening_snack:bagel_and_spread (720 kcal) |

### 14 days

- plan_id: `scenario_h14_0149`
- robust_score: `0.744`
- robust_weight_support: `0.688`
- fragility: `0.062`
- promoted: `True`
- rejection_reasons: `none`

| day_index | planned_date | planned_day_of_week | source_date | total_kcal | steps_day | dominant_archetype | loss_support_raw | gain_risk_raw | slot_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-30 | Monday | 2025-04-07 | 1645.000 | 3826.000 | beverage_only | 0.800 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (0 kcal); dinner:sandwich_meal (950 kcal); evening_snack:sushi_meal (685 kcal) |
| 2 | 2026-03-31 | Tuesday | 2025-05-22 | 2022.000 | 5266.000 | beverage_only | 0.776 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) |
| 3 | 2026-04-01 | Wednesday | 2025-04-14 | 1525.000 | 3765.000 | beverage_only | 0.853 | 0.000 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) |
| 4 | 2026-04-02 | Thursday | 2025-03-21 | 2436.000 | 9195.000 | beverage_only | 0.125 | 0.000 | breakfast:beverage_only (10 kcal); lunch:pizza_and_wings (1115 kcal); dinner:steak_plate (1174 kcal); evening_snack:alcoholic_beverage (137 kcal) |
| 5 | 2026-04-03 | Friday | 2025-03-12 | 1573.000 | 2895.000 | beverage_only | 0.901 | 0.000 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) |
| 6 | 2026-04-04 | Saturday | 2025-05-31 | 1942.000 | 5380.000 | beverage_only | 0.123 | 1.000 | breakfast:beverage_only (10 kcal); lunch:tea_beverage (462 kcal); afternoon_snack:mixed_plate (453 kcal); dinner:fried_chicken_piece (787 kcal); evening_snack:candy_bar (230 kcal) |
| 7 | 2026-04-05 | Sunday | 2026-03-01 | 1908.000 | 755.000 | beverage_only | 0.336 | 0.000 | breakfast:beverage_only (10 kcal); lunch:breakfast_plate (938 kcal); dinner:enchilada_plate (540 kcal); evening_snack:dessert_snack (420 kcal) |
| 8 | 2026-04-06 | Monday | 2025-03-12 | 1573.000 | 2895.000 | beverage_only | 0.901 | 0.000 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) |
| 9 | 2026-04-07 | Tuesday | 2025-05-22 | 2022.000 | 5266.000 | beverage_only | 0.776 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) |
| 10 | 2026-04-08 | Wednesday | 2025-04-14 | 1525.000 | 3765.000 | beverage_only | 0.853 | 0.000 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) |
| 11 | 2026-04-09 | Thursday | 2025-03-21 | 2436.000 | 9195.000 | beverage_only | 0.125 | 0.000 | breakfast:beverage_only (10 kcal); lunch:pizza_and_wings (1115 kcal); dinner:steak_plate (1174 kcal); evening_snack:alcoholic_beverage (137 kcal) |
| 12 | 2026-04-10 | Friday | 2025-03-12 | 1573.000 | 2895.000 | beverage_only | 0.901 | 0.000 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) |
| 13 | 2026-04-11 | Saturday | 2025-05-11 | 2178.000 | 2772.000 | beverage_only | 0.779 | 0.000 | breakfast:beverage_only (10 kcal); lunch:breakfast_plate (763 kcal); dinner:mixed_plate (685 kcal); evening_snack:bagel_and_spread (720 kcal) |
| 14 | 2026-04-12 | Sunday | 2026-03-15 | 2593.000 | 3045.000 | beverage_only | 0.297 | 0.000 | breakfast:beverage_only (10 kcal); lunch:deli_sandwich (1167 kcal); dinner:sandwich_meal (1070 kcal); evening_snack:berries (346 kcal) |

### 30 days

- plan_id: `scenario_h30_0207`
- robust_score: `0.740`
- robust_weight_support: `0.738`
- fragility: `0.068`
- promoted: `True`
- rejection_reasons: `none`

| day_index | planned_date | planned_day_of_week | source_date | total_kcal | steps_day | dominant_archetype | loss_support_raw | gain_risk_raw | slot_summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-30 | Monday | 2025-04-14 | 1525.000 | 3765.000 | beverage_only | 0.853 | 0.000 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) |
| 2 | 2026-03-31 | Tuesday | 2025-04-07 | 1645.000 | 3826.000 | beverage_only | 0.800 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (0 kcal); dinner:sandwich_meal (950 kcal); evening_snack:sushi_meal (685 kcal) |
| 3 | 2026-04-01 | Wednesday | 2025-04-14 | 1525.000 | 3765.000 | beverage_only | 0.853 | 0.000 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) |
| 4 | 2026-04-02 | Thursday | 2025-03-05 | 1835.000 | 2862.000 | beverage_only | 0.782 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:protein_component (492 kcal); evening_snack:savory_snack (363 kcal) |
| 5 | 2026-04-03 | Friday | 2025-05-22 | 2022.000 | 5266.000 | beverage_only | 0.776 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) |
| 6 | 2026-04-04 | Saturday | 2026-03-15 | 2593.000 | 3045.000 | beverage_only | 0.297 | 0.000 | breakfast:beverage_only (10 kcal); lunch:deli_sandwich (1167 kcal); dinner:sandwich_meal (1070 kcal); evening_snack:berries (346 kcal) |
| 7 | 2026-04-05 | Sunday | 2025-05-11 | 2178.000 | 2772.000 | beverage_only | 0.779 | 0.000 | breakfast:beverage_only (10 kcal); lunch:breakfast_plate (763 kcal); dinner:mixed_plate (685 kcal); evening_snack:bagel_and_spread (720 kcal) |
| 8 | 2026-04-06 | Monday | 2025-03-12 | 1573.000 | 2895.000 | beverage_only | 0.901 | 0.000 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) |
| 9 | 2026-04-07 | Tuesday | 2025-03-13 | 1635.000 | 2867.000 | mixed_plate | 0.854 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:protein_component (550 kcal); evening_snack:mixed_plate (105 kcal) |
| 10 | 2026-04-08 | Wednesday | 2025-03-12 | 1573.000 | 2895.000 | beverage_only | 0.901 | 0.000 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) |
| 11 | 2026-04-09 | Thursday | 2025-04-14 | 1525.000 | 3765.000 | beverage_only | 0.853 | 0.000 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) |
| 12 | 2026-04-10 | Friday | 2025-05-27 | 1540.000 | 4460.000 | beverage_only | 0.395 | 0.000 | breakfast:beverage_only (10 kcal); lunch:burger_sliders (190 kcal); dinner:spicy_chicken_entree (1120 kcal); evening_snack:bread (220 kcal) |
| 13 | 2026-04-11 | Saturday | 2025-04-20 | 1760.000 | 2218.000 | beverage_only | 0.820 | 0.000 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (0 kcal); dinner:salad_meal (858 kcal); evening_snack:sushi_meal (892 kcal) |
| 14 | 2026-04-12 | Sunday | 2025-05-11 | 2178.000 | 2772.000 | beverage_only | 0.779 | 0.000 | breakfast:beverage_only (10 kcal); lunch:breakfast_plate (763 kcal); dinner:mixed_plate (685 kcal); evening_snack:bagel_and_spread (720 kcal) |

_First 14 days shown; full detail is in `plan_details.csv`._

## Why Weaker Paths Were Rejected

- plans outside the observed core calorie band are not promotable even when they score well on one objective
- repeat-heavy plans are penalized because they collapse into one narrow historical slice
- plans that depend on high steps or a favorable recent-intake assumption lose robust score under stress testing
- plans with strong enjoyment but weak weight-support remain ranked but unpromoted

## Bundle Files

- rankings: `/workspace/foodai/reports/backtests/meal_scenario_planning/meal_scenario_planning_v1/scenario_rankings.csv`
- plan details: `/workspace/foodai/reports/backtests/meal_scenario_planning/meal_scenario_planning_v1/plan_details.csv`
- robustness stress table: `/workspace/foodai/reports/backtests/meal_scenario_planning/meal_scenario_planning_v1/robustness_stress_tests.csv`
- manifest: `/workspace/foodai/reports/backtests/meal_scenario_planning/meal_scenario_planning_v1/planning_manifest.json`
