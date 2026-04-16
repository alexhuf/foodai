# Meal Scenario Planning v2

## What Changed From v1

- keeps the observed-template realism constraint: every plan is still anchored to historical full-day meal archetype patterns
- adds bounded portion variants by scaling observed templates only within observed archetype-signature calorie ranges
- applies horizon-aware repeat limits to source templates and archetype signatures
- writes plain-language plan and day explanations
- exposes repeat diagnostics in the ranked output

## Context

- planning_start_date: `2026-03-30`
- latest_observed_date: `2026-03-29`
- recent_steps_mean: `2210.8571428571427`
- recent_food_kcal_mean: `2129.3571428571427`
- recent_dominant_archetypes: `beverage_only, dessert_snack`

## Action Space

- observed base templates after required-slot filter: `229`
- bounded portion variants added: `363`
- total day actions in v2 library: `592`
- portion multiplier range: `0.88` to `1.12`
- core calorie band q05/q95: `1363` to `2643` kcal

## Ranked Promoted Plans

| plan_id | horizon_days | strategy | robust_score | robust_weight_support | fragility | unique_source_templates | unique_archetype_signatures | max_source_template_count | max_signature_share | bounded_variant_days | plain_language_explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scenario_v2_h3_0001 | 3 | weight | 0.856 | 0.843 | 0.083 | 3 | 3 | 1 | 0.333 | 2 | Scores well because enjoyment 0.99, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0023 | 3 | sampled | 0.852 | 0.828 | 0.083 | 3 | 3 | 1 | 0.333 | 1 | Scores well because enjoyment 0.99, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0036 | 3 | sampled | 0.851 | 0.827 | 0.082 | 3 | 3 | 1 | 0.333 | 1 | Scores well because enjoyment 0.99, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0046 | 3 | sampled | 0.851 | 0.827 | 0.082 | 3 | 3 | 1 | 0.333 | 1 | Scores well because enjoyment 0.99, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0003 | 3 | enjoyment | 0.851 | 0.827 | 0.082 | 3 | 3 | 1 | 0.333 | 1 | Scores well because enjoyment 0.99, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0016 | 3 | sampled | 0.849 | 0.817 | 0.085 | 3 | 3 | 1 | 0.333 | 2 | Scores well because enjoyment 1.00, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0026 | 3 | sampled | 0.849 | 0.817 | 0.085 | 3 | 3 | 1 | 0.333 | 2 | Scores well because enjoyment 1.00, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0000 | 3 | balanced | 0.849 | 0.824 | 0.080 | 3 | 3 | 1 | 0.333 | 1 | Scores well because enjoyment 0.98, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0015 | 3 | sampled | 0.846 | 0.803 | 0.085 | 3 | 3 | 1 | 0.333 | 3 | Scores well because enjoyment 1.00, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0027 | 3 | sampled | 0.821 | 0.778 | 0.085 | 3 | 3 | 1 | 0.333 | 3 | Scores well because enjoyment 1.00, health 0.89 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0009 | 3 | sampled | 0.817 | 0.785 | 0.081 | 3 | 3 | 1 | 0.333 | 0 | Scores well because enjoyment 0.98, health 0.86 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0020 | 3 | sampled | 0.814 | 0.775 | 0.081 | 3 | 3 | 1 | 0.333 | 1 | Scores well because enjoyment 0.99, health 0.86 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0045 | 3 | sampled | 0.810 | 0.738 | 0.084 | 3 | 3 | 1 | 0.333 | 2 | Scores well because enjoyment 1.00, health 0.89 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0044 | 3 | sampled | 0.809 | 0.736 | 0.078 | 3 | 3 | 1 | 0.333 | 2 | Scores well because enjoyment 0.99, health 0.89 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0039 | 3 | sampled | 0.808 | 0.650 | 0.087 | 3 | 3 | 1 | 0.333 | 2 | Scores well because enjoyment 1.00, health 1.00 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0033 | 3 | sampled | 0.805 | 0.740 | 0.086 | 3 | 3 | 1 | 0.333 | 2 | Scores well because enjoyment 1.00, health 0.86 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0012 | 3 | sampled | 0.800 | 0.641 | 0.089 | 3 | 3 | 1 | 0.333 | 2 | Scores well because enjoyment 0.99, health 0.98 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0034 | 3 | sampled | 0.795 | 0.623 | 0.085 | 3 | 3 | 1 | 0.333 | 2 | Scores well because enjoyment 1.00, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0041 | 3 | sampled | 0.770 | 0.836 | 0.085 | 3 | 3 | 1 | 0.333 | 1 | Scores well because weight support 0.84, health 0.81 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |
| scenario_v2_h3_0028 | 3 | sampled | 0.770 | 0.834 | 0.079 | 3 | 3 | 1 | 0.333 | 1 | Scores well because weight support 0.83, consistency 0.81 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only. |

## Best Plan By Horizon

### 3 days

- plan_id: `scenario_v2_h3_0001`
- robust_score: `0.856`
- robust_weight_support: `0.843`
- fragility: `0.083`
- repeat diagnostics: `3` source templates, `3` archetype signatures, max signature share `0.33`
- explanation: Scores well because enjoyment 0.99, health 0.97 while staying inside observed calorie bands. Uses 3 source templates and 3 archetype patterns; main observed archetypes: beverage_only.
- promoted: `True`
- rejection_reasons: `none`

| day_index | planned_date | planned_day_of_week | source_date | source_template_id | portion_variant | total_kcal | dominant_archetype | loss_support_raw | slot_summary | day_explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-30 | Monday | 2025-03-12 | day_2025-03-12 | observed | 1573.000 | beverage_only | 0.901 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) | strong projected weight support; good health score; high familiarity/enjoyment |
| 2 | 2026-03-31 | Tuesday | 2025-05-22 | day_2025-05-22 | lighter_observed_portion | 1779.360 | beverage_only | 0.776 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |
| 3 | 2026-04-01 | Wednesday | 2025-04-14 | day_2025-04-14 | lighter_observed_portion | 1708.000 | beverage_only | 0.853 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |

### 5 days

- plan_id: `scenario_v2_h5_0081`
- robust_score: `0.803`
- robust_weight_support: `0.696`
- fragility: `0.083`
- repeat diagnostics: `5` source templates, `5` archetype signatures, max signature share `0.20`
- explanation: Scores well because enjoyment 0.99, health 0.92 while staying inside observed calorie bands. Uses 5 source templates and 5 archetype patterns; main observed archetypes: beverage_only.
- promoted: `True`
- rejection_reasons: `none`

| day_index | planned_date | planned_day_of_week | source_date | source_template_id | portion_variant | total_kcal | dominant_archetype | loss_support_raw | slot_summary | day_explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-30 | Monday | 2025-04-14 | day_2025-04-14 | observed | 1525.000 | beverage_only | 0.853 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) | strong projected weight support; good health score; high familiarity/enjoyment |
| 2 | 2026-03-31 | Tuesday | 2025-03-12 | day_2025-03-12 | lighter_observed_portion | 1753.000 | beverage_only | 0.901 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |
| 3 | 2026-04-01 | Wednesday | 2025-05-22 | day_2025-05-22 | lighter_observed_portion | 1779.360 | beverage_only | 0.776 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |
| 4 | 2026-04-02 | Thursday | 2025-05-14 | day_2025-05-14 | lighter_observed_portion | 1753.000 | beverage_only | 0.320 | breakfast:beverage_only (10 kcal); lunch:falafel (607 kcal); dinner:tex_mex_meal (876 kcal); evening_snack:mixed_plate (400 kcal) | high familiarity/enjoyment; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |
| 5 | 2026-04-03 | Friday | 2025-04-23 | day_2025-04-23 | observed | 1762.000 | beverage_only | 0.316 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (0 kcal); dinner:potato_casserole (1427 kcal); evening_snack:dessert_snack (325 kcal) | good health score; high familiarity/enjoyment |

### 7 days

- plan_id: `scenario_v2_h7_0133`
- robust_score: `0.763`
- robust_weight_support: `0.610`
- fragility: `0.071`
- repeat diagnostics: `7` source templates, `7` archetype signatures, max signature share `0.14`
- explanation: Scores well because enjoyment 0.91, health 0.89 while staying inside observed calorie bands. Uses 7 source templates and 7 archetype patterns; main observed archetypes: beverage_only.
- promoted: `True`
- rejection_reasons: `none`

| day_index | planned_date | planned_day_of_week | source_date | source_template_id | portion_variant | total_kcal | dominant_archetype | loss_support_raw | slot_summary | day_explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-30 | Monday | 2025-03-12 | day_2025-03-12 | heartier_observed_portion | 1761.760 | beverage_only | 0.901 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with higher portion level bounded to observed cluster kcal range (1753-2235) |
| 2 | 2026-03-31 | Tuesday | 2025-05-22 | day_2025-05-22 | heartier_observed_portion | 2234.750 | beverage_only | 0.776 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with higher portion level bounded to observed cluster kcal range (1753-2235) |
| 3 | 2026-04-01 | Wednesday | 2025-04-14 | day_2025-04-14 | lighter_observed_portion | 1708.000 | beverage_only | 0.853 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |
| 4 | 2026-04-02 | Thursday | 2025-04-23 | day_2025-04-23 | observed | 1762.000 | beverage_only | 0.316 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (0 kcal); dinner:potato_casserole (1427 kcal); evening_snack:dessert_snack (325 kcal) | good health score; high familiarity/enjoyment |
| 5 | 2026-04-03 | Friday | 2026-03-17 | day_2026-03-17 | observed | 2504.000 | beverage_only | 0.400 | breakfast:beverage_only (10 kcal); lunch:breakfast_burrito_plus_chips (720 kcal); dinner:mixed_plate (1023 kcal); evening_snack:breakfast_cereal (751 kcal) | balanced observed template |
| 6 | 2026-04-04 | Saturday | 2025-05-11 | day_2025-05-11 | observed | 2178.000 | beverage_only | 0.779 | breakfast:beverage_only (10 kcal); lunch:breakfast_plate (763 kcal); dinner:mixed_plate (685 kcal); evening_snack:bagel_and_spread (720 kcal) | good health score; high familiarity/enjoyment |
| 7 | 2026-04-05 | Sunday | 2026-03-15 | day_2026-03-15 | lighter_observed_portion | 2281.840 | beverage_only | 0.297 | breakfast:beverage_only (10 kcal); lunch:deli_sandwich (1167 kcal); dinner:sandwich_meal (1070 kcal); evening_snack:berries (346 kcal) | good health score; high familiarity/enjoyment; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |

### 14 days

- plan_id: `scenario_v2_h14_0161`
- robust_score: `0.756`
- robust_weight_support: `0.710`
- fragility: `0.064`
- repeat diagnostics: `9` source templates, `9` archetype signatures, max signature share `0.14`
- explanation: Scores well because enjoyment 0.84, consistency 0.80 while staying inside observed calorie bands. Uses 9 source templates and 9 archetype patterns; main observed archetypes: beverage_only, snack_plate.
- promoted: `True`
- rejection_reasons: `none`

| day_index | planned_date | planned_day_of_week | source_date | source_template_id | portion_variant | total_kcal | dominant_archetype | loss_support_raw | slot_summary | day_explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-30 | Monday | 2025-03-12 | day_2025-03-12 | observed | 1573.000 | beverage_only | 0.901 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) | strong projected weight support; good health score; high familiarity/enjoyment |
| 2 | 2026-03-31 | Tuesday | 2025-04-14 | day_2025-04-14 | observed | 1525.000 | beverage_only | 0.853 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) | strong projected weight support; good health score; high familiarity/enjoyment |
| 3 | 2026-04-01 | Wednesday | 2025-05-14 | day_2025-05-14 | lighter_observed_portion | 1753.000 | beverage_only | 0.320 | breakfast:beverage_only (10 kcal); lunch:falafel (607 kcal); dinner:tex_mex_meal (876 kcal); evening_snack:mixed_plate (400 kcal) | high familiarity/enjoyment; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |
| 4 | 2026-04-02 | Thursday | 2025-05-22 | day_2025-05-22 | observed | 2022.000 | beverage_only | 0.776 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) | strong projected weight support; good health score; high familiarity/enjoyment |
| 5 | 2026-04-03 | Friday | 2025-05-28 | day_2025-05-28 | observed | 2225.000 | snack_plate | 0.833 | breakfast:snack_plate (375 kcal); lunch:tex_mex_meal (1290 kcal); dinner:salad_meal (420 kcal); evening_snack:protein_component (140 kcal) | strong projected weight support |
| 6 | 2026-04-04 | Saturday | 2025-05-11 | day_2025-05-11 | lighter_observed_portion | 1916.640 | beverage_only | 0.779 | breakfast:beverage_only (10 kcal); lunch:breakfast_plate (763 kcal); dinner:mixed_plate (685 kcal); evening_snack:bagel_and_spread (720 kcal) | strong projected weight support; good health score; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |
| 7 | 2026-04-05 | Sunday | 2025-05-31 | day_2025-05-31 | observed | 1942.000 | beverage_only | 0.123 | breakfast:beverage_only (10 kcal); lunch:tea_beverage (462 kcal); afternoon_snack:mixed_plate (453 kcal); dinner:fried_chicken_piece (787 kcal); evening_snack:candy_bar (230 kcal) | good health score; high familiarity/enjoyment |
| 8 | 2026-04-06 | Monday | 2025-04-14 | day_2025-04-14 | heartier_observed_portion | 1708.000 | beverage_only | 0.853 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with higher portion level bounded to observed cluster kcal range (1753-2235) |
| 9 | 2026-04-07 | Tuesday | 2025-03-12 | day_2025-03-12 | heartier_observed_portion | 1761.760 | beverage_only | 0.901 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with higher portion level bounded to observed cluster kcal range (1753-2235) |
| 10 | 2026-04-08 | Wednesday | 2025-05-22 | day_2025-05-22 | observed | 2022.000 | beverage_only | 0.776 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) | strong projected weight support; good health score; high familiarity/enjoyment |
| 11 | 2026-04-09 | Thursday | 2025-05-14 | day_2025-05-14 | lighter_observed_portion | 1753.000 | beverage_only | 0.320 | breakfast:beverage_only (10 kcal); lunch:falafel (607 kcal); dinner:tex_mex_meal (876 kcal); evening_snack:mixed_plate (400 kcal) | high familiarity/enjoyment; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |
| 12 | 2026-04-10 | Friday | 2026-03-17 | day_2026-03-17 | observed | 2504.000 | beverage_only | 0.400 | breakfast:beverage_only (10 kcal); lunch:breakfast_burrito_plus_chips (720 kcal); dinner:mixed_plate (1023 kcal); evening_snack:breakfast_cereal (751 kcal) | balanced observed template |
| 13 | 2026-04-11 | Saturday | 2025-05-11 | day_2025-05-11 | observed | 2178.000 | beverage_only | 0.779 | breakfast:beverage_only (10 kcal); lunch:breakfast_plate (763 kcal); dinner:mixed_plate (685 kcal); evening_snack:bagel_and_spread (720 kcal) | good health score; high familiarity/enjoyment |
| 14 | 2026-04-12 | Sunday | 2025-04-20 | day_2025-04-20 | observed | 1760.000 | beverage_only | 0.820 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (0 kcal); dinner:salad_meal (858 kcal); evening_snack:sushi_meal (892 kcal) | strong projected weight support |

### 30 days

- plan_id: `scenario_v2_h30_0241`
- robust_score: `0.698`
- robust_weight_support: `0.618`
- fragility: `0.065`
- repeat diagnostics: `19` source templates, `19` archetype signatures, max signature share `0.10`
- explanation: Scores well because consistency 0.80, enjoyment 0.77 while staying inside observed calorie bands. Uses 19 source templates and 19 archetype patterns; main observed archetypes: beverage_only, mixed_plate, snack_plate.
- promoted: `True`
- rejection_reasons: `none`

| day_index | planned_date | planned_day_of_week | source_date | source_template_id | portion_variant | total_kcal | dominant_archetype | loss_support_raw | slot_summary | day_explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-30 | Monday | 2025-03-12 | day_2025-03-12 | observed | 1573.000 | beverage_only | 0.901 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) | strong projected weight support; good health score; high familiarity/enjoyment |
| 2 | 2026-03-31 | Tuesday | 2025-04-23 | day_2025-04-23 | heartier_observed_portion | 1973.440 | beverage_only | 0.316 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (0 kcal); dinner:potato_casserole (1427 kcal); evening_snack:dessert_snack (325 kcal) | good health score; high familiarity/enjoyment; same observed archetype pattern with higher portion level bounded to observed cluster kcal range (1753-2235) |
| 3 | 2026-04-01 | Wednesday | 2025-05-22 | day_2025-05-22 | observed | 2022.000 | beverage_only | 0.776 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) | strong projected weight support; good health score; high familiarity/enjoyment |
| 4 | 2026-04-02 | Thursday | 2025-04-14 | day_2025-04-14 | observed | 1525.000 | beverage_only | 0.853 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) | strong projected weight support; good health score; high familiarity/enjoyment |
| 5 | 2026-04-03 | Friday | 2025-05-28 | day_2025-05-28 | observed | 2225.000 | snack_plate | 0.833 | breakfast:snack_plate (375 kcal); lunch:tex_mex_meal (1290 kcal); dinner:salad_meal (420 kcal); evening_snack:protein_component (140 kcal) | strong projected weight support |
| 6 | 2026-04-04 | Saturday | 2025-05-10 | day_2025-05-10 | observed | 2145.000 | mixed_plate | 0.173 | breakfast:mixed_plate (0 kcal); lunch:mixed_plate (905 kcal); dinner:burger_meal (898 kcal); evening_snack:pretzel_bites_with_cheese (342 kcal) | balanced observed template |
| 7 | 2026-04-05 | Sunday | 2026-03-08 | day_2026-03-08 | observed | 2368.000 | beverage_only | 0.384 | breakfast:beverage_only (10 kcal); lunch:burger_meal (1630 kcal); afternoon_snack:mixed_plate (174 kcal); dinner:mexican_casserole (554 kcal) | balanced observed template |
| 8 | 2026-04-06 | Monday | 2025-05-14 | day_2025-05-14 | heartier_observed_portion | 2120.160 | beverage_only | 0.320 | breakfast:beverage_only (10 kcal); lunch:falafel (607 kcal); dinner:tex_mex_meal (876 kcal); evening_snack:mixed_plate (400 kcal) | high familiarity/enjoyment; same observed archetype pattern with higher portion level bounded to observed cluster kcal range (1753-2235) |
| 9 | 2026-04-07 | Tuesday | 2025-05-22 | day_2025-05-22 | heartier_observed_portion | 2234.750 | beverage_only | 0.776 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (970 kcal); dinner:tex_mex_meal (815 kcal); evening_snack:cookie (227 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with higher portion level bounded to observed cluster kcal range (1753-2235) |
| 10 | 2026-04-08 | Wednesday | 2025-04-14 | day_2025-04-14 | heartier_observed_portion | 1708.000 | beverage_only | 0.853 | breakfast:beverage_only (10 kcal); lunch:fajitas (445 kcal); dinner:sandwich_meal (930 kcal); evening_snack:mixed_plate (140 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with higher portion level bounded to observed cluster kcal range (1753-2235) |
| 11 | 2026-04-09 | Thursday | 2025-03-12 | day_2025-03-12 | heartier_observed_portion | 1761.760 | beverage_only | 0.901 | breakfast:beverage_only (10 kcal); lunch:cracker_snack (200 kcal); dinner:flatbread (493 kcal); evening_snack:berries (870 kcal) | strong projected weight support; good health score; high familiarity/enjoyment; same observed archetype pattern with higher portion level bounded to observed cluster kcal range (1753-2235) |
| 12 | 2026-04-10 | Friday | 2025-04-07 | day_2025-04-07 | observed | 1645.000 | beverage_only | 0.800 | breakfast:beverage_only (10 kcal); lunch:mixed_plate (0 kcal); dinner:sandwich_meal (950 kcal); evening_snack:sushi_meal (685 kcal) | strong projected weight support |
| 13 | 2026-04-11 | Saturday | 2026-03-15 | day_2026-03-15 | lighter_observed_portion | 2281.840 | beverage_only | 0.297 | breakfast:beverage_only (10 kcal); lunch:deli_sandwich (1167 kcal); dinner:sandwich_meal (1070 kcal); evening_snack:berries (346 kcal) | good health score; high familiarity/enjoyment; same observed archetype pattern with lower portion level bounded to observed cluster kcal range (1753-2235) |
| 14 | 2026-04-12 | Sunday | 2026-03-07 | day_2026-03-07 | observed | 1788.000 | beverage_only | 0.346 | breakfast:beverage_only (10 kcal); lunch:bread_roll (152 kcal); dinner:side_plate (996 kcal); evening_snack:ice_cream (630 kcal) | balanced observed template |

_First 14 days shown; full detail is in `plan_details.csv`._

## Bundle Files

- rankings: `/workspace/foodai/reports/backtests/meal_scenario_planning/meal_scenario_planning_v2/scenario_rankings.csv`
- plan details: `/workspace/foodai/reports/backtests/meal_scenario_planning/meal_scenario_planning_v2/plan_details.csv`
- robustness stress table: `/workspace/foodai/reports/backtests/meal_scenario_planning/meal_scenario_planning_v2/robustness_stress_tests.csv`
- day action library: `/workspace/foodai/reports/backtests/meal_scenario_planning/meal_scenario_planning_v2/day_action_library.csv`
- manifest: `/workspace/foodai/reports/backtests/meal_scenario_planning/meal_scenario_planning_v2/planning_manifest.json`
