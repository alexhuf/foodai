# Weekly Weight-Gain Transition Stability Report

- target: weeks / y_next_weight_gain_flag
- folds: 5
- models chosen: {'dummy_majority': 3, 'logreg': 2}

## Aggregate fold metrics
- train_n_total_mean: 40.0000
- train_n_total_std: 5.6569
- train_n_total_min: 32.0000
- train_n_total_max: 48.0000
- train_n_positive_mean: 4.6000
- train_n_positive_std: 0.4899
- train_n_positive_min: 4.0000
- train_n_positive_max: 5.0000
- train_n_negative_mean: 35.4000
- train_n_negative_std: 5.2383
- train_n_negative_min: 28.0000
- train_n_negative_max: 43.0000
- train_positive_rate_mean: 0.1158
- train_positive_rate_std: 0.0081
- train_positive_rate_min: 0.1042
- train_positive_rate_max: 0.1250
- val_n_total_mean: 8.0000
- val_n_total_std: 0.0000
- val_n_total_min: 8.0000
- val_n_total_max: 8.0000
- val_n_positive_mean: 1.2000
- val_n_positive_std: 0.9798
- val_n_positive_min: 0.0000
- val_n_positive_max: 3.0000
- val_n_negative_mean: 6.8000
- val_n_negative_std: 0.9798
- val_n_negative_min: 5.0000
- val_n_negative_max: 8.0000
- val_positive_rate_mean: 0.1500
- val_positive_rate_std: 0.1225
- val_positive_rate_min: 0.0000
- val_positive_rate_max: 0.3750
- test_n_total_mean: 5.8000
- test_n_total_std: 0.4000
- test_n_total_min: 5.0000
- test_n_total_max: 6.0000
- test_n_positive_mean: 1.2000
- test_n_positive_std: 0.7483
- test_n_positive_min: 0.0000
- test_n_positive_max: 2.0000
- test_n_negative_mean: 4.6000
- test_n_negative_std: 0.8000
- test_n_negative_min: 4.0000
- test_n_negative_max: 6.0000
- test_positive_rate_mean: 0.2067
- test_positive_rate_std: 0.1236
- test_positive_rate_min: 0.0000
- test_positive_rate_max: 0.3333
- selected_threshold_balanced_accuracy_mean: 0.0323
- selected_threshold_balanced_accuracy_std: 0.0267
- selected_threshold_balanced_accuracy_min: 0.0000
- selected_threshold_balanced_accuracy_max: 0.0616
- selected_threshold_macro_f1_mean: 0.0523
- selected_threshold_macro_f1_std: 0.0046
- selected_threshold_macro_f1_min: 0.0500
- selected_threshold_macro_f1_max: 0.0616
- default_accuracy_mean: 0.8667
- default_accuracy_std: 0.1247
- default_accuracy_min: 0.6667
- default_accuracy_max: 1.0000
- default_balanced_accuracy_mean: 0.7500
- default_balanced_accuracy_std: 0.2236
- default_balanced_accuracy_min: 0.5000
- default_balanced_accuracy_max: 1.0000
- default_macro_f1_mean: 0.7265
- default_macro_f1_std: 0.2580
- default_macro_f1_min: 0.4000
- default_macro_f1_max: 1.0000
- default_ece_mean: 0.1522
- default_ece_std: 0.1092
- default_ece_min: 0.0000
- default_ece_max: 0.3333
- default_positive_rate_pred_mean: 0.0733
- default_positive_rate_pred_std: 0.0904
- default_positive_rate_pred_min: 0.0000
- default_positive_rate_pred_max: 0.2000
- default_positive_rate_true_mean: 0.2067
- default_positive_rate_true_std: 0.1236
- default_positive_rate_true_min: 0.0000
- default_positive_rate_true_max: 0.3333
- tuned_bal_accuracy_mean: 0.5333
- tuned_bal_accuracy_std: 0.3859
- tuned_bal_accuracy_min: 0.0000
- tuned_bal_accuracy_max: 1.0000
- tuned_bal_balanced_accuracy_mean: 0.5500
- tuned_bal_balanced_accuracy_std: 0.3317
- tuned_bal_balanced_accuracy_min: 0.0000
- tuned_bal_balanced_accuracy_max: 1.0000
- tuned_bal_macro_f1_mean: 0.4641
- tuned_bal_macro_f1_std: 0.3764
- tuned_bal_macro_f1_min: 0.0000
- tuned_bal_macro_f1_max: 1.0000
- tuned_bal_ece_mean: 0.1522
- tuned_bal_ece_std: 0.1092
- tuned_bal_ece_min: 0.0000
- tuned_bal_ece_max: 0.3333
- tuned_bal_positive_rate_pred_mean: 0.4733
- tuned_bal_positive_rate_pred_std: 0.4353
- tuned_bal_positive_rate_pred_min: 0.0000
- tuned_bal_positive_rate_pred_max: 1.0000
- tuned_bal_positive_rate_true_mean: 0.2067
- tuned_bal_positive_rate_true_std: 0.1236
- tuned_bal_positive_rate_true_min: 0.0000
- tuned_bal_positive_rate_true_max: 0.3333
- tuned_f1_accuracy_mean: 0.8667
- tuned_f1_accuracy_std: 0.1247
- tuned_f1_accuracy_min: 0.6667
- tuned_f1_accuracy_max: 1.0000
- tuned_f1_balanced_accuracy_mean: 0.7500
- tuned_f1_balanced_accuracy_std: 0.2236
- tuned_f1_balanced_accuracy_min: 0.5000
- tuned_f1_balanced_accuracy_max: 1.0000
- tuned_f1_macro_f1_mean: 0.7265
- tuned_f1_macro_f1_std: 0.2580
- tuned_f1_macro_f1_min: 0.4000
- tuned_f1_macro_f1_max: 1.0000
- tuned_f1_ece_mean: 0.1522
- tuned_f1_ece_std: 0.1092
- tuned_f1_ece_min: 0.0000
- tuned_f1_ece_max: 0.3333
- tuned_f1_positive_rate_pred_mean: 0.0733
- tuned_f1_positive_rate_pred_std: 0.0904
- tuned_f1_positive_rate_pred_min: 0.0000
- tuned_f1_positive_rate_pred_max: 0.2000
- tuned_f1_positive_rate_true_mean: 0.2067
- tuned_f1_positive_rate_true_std: 0.1236
- tuned_f1_positive_rate_true_min: 0.0000
- tuned_f1_positive_rate_true_max: 0.3333
- test_ece_raw_mean: 0.1522
- test_ece_raw_std: 0.1092
- test_ece_raw_min: 0.0000
- test_ece_raw_max: 0.3333
- default_roc_auc_mean: 0.6875
- default_roc_auc_std: 0.2073
- default_roc_auc_min: 0.5000
- default_roc_auc_max: 1.0000
- default_brier_mean: 0.1741
- default_brier_std: 0.1075
- default_brier_min: 0.0299
- default_brier_max: 0.3333
- tuned_bal_roc_auc_mean: 0.6875
- tuned_bal_roc_auc_std: 0.2073
- tuned_bal_roc_auc_min: 0.5000
- tuned_bal_roc_auc_max: 1.0000
- tuned_bal_brier_mean: 0.1741
- tuned_bal_brier_std: 0.1075
- tuned_bal_brier_min: 0.0299
- tuned_bal_brier_max: 0.3333
- tuned_f1_roc_auc_mean: 0.6875
- tuned_f1_roc_auc_std: 0.2073
- tuned_f1_roc_auc_min: 0.5000
- tuned_f1_roc_auc_max: 1.0000
- tuned_f1_brier_mean: 0.1741
- tuned_f1_brier_std: 0.1075
- tuned_f1_brier_min: 0.0299
- tuned_f1_brier_max: 0.3333
- target_space: weeks
- target_name: y_next_weight_gain_flag
- min_train: 36
- val_size: 8
- test_size: 6
- step: 4
- dropped_feature_cols: ['y_next_weight_delta_lb', 'y_delta_next_weight_delta_lb', 'y_next_meal_events_per_day_week', 'y_delta_next_meal_events_per_day_week', 'y_next_restaurant_meal_fraction_week', 'y_delta_next_restaurant_meal_fraction_week', 'y_next_budget_minus_logged_food_kcal_week', 'y_delta_next_budget_minus_logged_food_kcal_week', 'y_next_dominant_meal_archetype_week', 'y_same_dominant_meal_archetype_week', 'y_next_dominant_cuisine_week', 'y_same_dominant_cuisine_week', 'y_next_dominant_service_form_week', 'y_same_dominant_service_form_week', 'y_next_dominant_prep_profile_week', 'y_same_dominant_prep_profile_week', 'y_next_dominant_protein_week', 'y_same_dominant_protein_week', 'y_next_dominant_starch_week', 'y_same_dominant_starch_week', 'y_next_dominant_energy_density_week', 'y_same_dominant_energy_density_week', 'y_next_dominant_satiety_style_week', 'y_same_dominant_satiety_style_week', 'y_next_weight_loss_flag', 'y_next_weight_gain_flag', 'y_next_restaurant_heavy_flag', 'y_next_budget_breach_flag', 'y_next_high_meal_frequency_flag', 'next_period_id', 'next_period_start', 'split_suggested', 'period_kind', 'period_id', 'week_id']

## Fold diagnostics

| fold | best_model     | train_n_total | train_n_positive | train_n_negative | train_positive_rate | val_n_total | val_n_positive | val_n_negative | val_positive_rate | test_n_total | test_n_positive | test_n_negative | test_positive_rate  | selected_threshold_balanced_accuracy | selected_threshold_macro_f1 | default_accuracy   | default_balanced_accuracy | default_macro_f1    | default_ece         | default_positive_rate_pred | default_positive_rate_true | tuned_bal_accuracy  | tuned_bal_balanced_accuracy | tuned_bal_macro_f1  | tuned_bal_ece       | tuned_bal_positive_rate_pred | tuned_bal_positive_rate_true | tuned_f1_accuracy  | tuned_f1_balanced_accuracy | tuned_f1_macro_f1   | tuned_f1_ece        | tuned_f1_positive_rate_pred | tuned_f1_positive_rate_true | test_ece_raw        | default_roc_auc | default_brier       | tuned_bal_roc_auc | tuned_bal_brier     | tuned_f1_roc_auc | tuned_f1_brier      |
| ---- | -------------- | ------------- | ---------------- | ---------------- | ------------------- | ----------- | -------------- | -------------- | ----------------- | ------------ | --------------- | --------------- | ------------------- | ------------------------------------ | --------------------------- | ------------------ | ------------------------- | ------------------- | ------------------- | -------------------------- | -------------------------- | ------------------- | --------------------------- | ------------------- | ------------------- | ---------------------------- | ---------------------------- | ------------------ | -------------------------- | ------------------- | ------------------- | --------------------------- | --------------------------- | ------------------- | --------------- | ------------------- | ----------------- | ------------------- | ---------------- | ------------------- |
| 1    | dummy_majority | 32            | 4                | 28               | 0.125               | 8           | 1              | 7              | 0.125             | 6            | 0               | 6               | 0.0                 | 0.0                                  | 0.05                        | 1.0                | 1.0                       | 1.0                 | 0.0                 | 0.0                        | 0.0                        | 0.0                 | 0.0                         | 0.0                 | 0.0                 | 1.0                          | 0.0                          | 1.0                | 1.0                        | 1.0                 | 0.0                 | 0.0                         | 0.0                         | 0.0                 |                 |                     |                   |                     |                  |                     |
| 2    | dummy_majority | 36            | 4                | 32               | 0.1111111111111111  | 8           | 1              | 7              | 0.125             | 6            | 1               | 5               | 0.16666666666666666 | 0.0                                  | 0.05                        | 0.8333333333333334 | 0.5                       | 0.45454545454545453 | 0.16666666666666666 | 0.0                        | 0.16666666666666666        | 0.16666666666666666 | 0.5                         | 0.14285714285714285 | 0.16666666666666666 | 1.0                          | 0.16666666666666666          | 0.8333333333333334 | 0.5                        | 0.45454545454545453 | 0.16666666666666666 | 0.0                         | 0.16666666666666666         | 0.16666666666666666 | 0.5             | 0.16666666666666666 | 0.5               | 0.16666666666666666 | 0.5              | 0.16666666666666666 |
| 3    | dummy_majority | 40            | 5                | 35               | 0.125               | 8           | 0              | 8              | 0.0               | 6            | 2               | 4               | 0.3333333333333333  | 0.05                                 | 0.05                        | 0.6666666666666666 | 0.5                       | 0.4                 | 0.3333333333333333  | 0.0                        | 0.3333333333333333         | 0.6666666666666666  | 0.5                         | 0.4                 | 0.3333333333333333  | 0.0                          | 0.3333333333333333           | 0.6666666666666666 | 0.5                        | 0.4                 | 0.3333333333333333  | 0.0                         | 0.3333333333333333          | 0.3333333333333333  | 0.5             | 0.3333333333333333  | 0.5               | 0.3333333333333333  | 0.5              | 0.3333333333333333  |
| 4    | logreg         | 44            | 5                | 39               | 0.11363636363636363 | 8           | 1              | 7              | 0.125             | 6            | 2               | 4               | 0.3333333333333333  | 0.05                                 | 0.05                        | 0.8333333333333334 | 0.75                      | 0.7777777777777777  | 0.16594665520334848 | 0.16666666666666666        | 0.3333333333333333         | 0.8333333333333334  | 0.75                        | 0.7777777777777777  | 0.16594665520334848 | 0.16666666666666666          | 0.3333333333333333           | 0.8333333333333334 | 0.75                       | 0.7777777777777777  | 0.16594665520334848 | 0.16666666666666666         | 0.3333333333333333          | 0.16594665520334848 | 0.75            | 0.16664487472530656 | 0.75              | 0.16664487472530656 | 0.75             | 0.16664487472530656 |
| 5    | logreg         | 48            | 5                | 43               | 0.10416666666666667 | 8           | 3              | 5              | 0.375             | 5            | 1               | 4               | 0.2                 | 0.0616                               | 0.0616                      | 1.0                | 1.0                       | 1.0                 | 0.09510381953535355 | 0.2                        | 0.2                        | 1.0                 | 1.0                         | 1.0                 | 0.09510381953535355 | 0.2                          | 0.2                          | 1.0                | 1.0                        | 1.0                 | 0.09510381953535355 | 0.2                         | 0.2                         | 0.09510381953535355 | 1.0             | 0.02989598863283985 | 1.0               | 0.02989598863283985 | 1.0              | 0.02989598863283985 |

## Ablation summary

| ablation              | test_accuracy_mean | test_accuracy_std   | test_accuracy_min  | test_accuracy_max  | test_balanced_accuracy_mean | test_balanced_accuracy_std | test_balanced_accuracy_min | test_balanced_accuracy_max | test_macro_f1_mean | test_macro_f1_std   | test_macro_f1_min  | test_macro_f1_max  | test_roc_auc_mean | test_roc_auc_std    | test_roc_auc_min | test_roc_auc_max | test_brier_mean     | test_brier_std      | test_brier_min     | test_brier_max     | test_ece_mean       | test_ece_std        | test_ece_min       | test_ece_max       |
| --------------------- | ------------------ | ------------------- | ------------------ | ------------------ | --------------------------- | -------------------------- | -------------------------- | -------------------------- | ------------------ | ------------------- | ------------------ | ------------------ | ----------------- | ------------------- | ---------------- | ---------------- | ------------------- | ------------------- | ------------------ | ------------------ | ------------------- | ------------------- | ------------------ | ------------------ |
| drop_other            | 0.8933333333333333 | 0.09831920802501745 | 0.8                | 1.0                | 0.875                       | 0.12499999999999999        | 0.75                       | 1.0                        | 0.8634920634920634 | 0.12478250163195907 | 0.7619047619047619 | 1.0                | 0.84375           | 0.1875              | 0.625            | 1.0              | 0.08668764965791938 | 0.09282481582822596 | 0.0006248869544024 | 0.1672605202445818 | 0.08727476495651407 | 0.07983803335373577 | 0.0109666283582475 | 0.1758745673371669 |
| meals_only            | 0.7866666666666666 | 0.196638416050035   | 0.6                | 1.0                | 0.775                       | 0.22360679774997896        | 0.5                        | 1.0                        | 0.7216666666666667 | 0.26781004877006054 | 0.4                | 1.0                | 0.78125           | 0.2576941016011038  | 0.5              | 1.0              | 0.14960970755369293 | 0.13870475149542477 | 0.0033302526114254 | 0.331303054722459  | 0.15168112922006666 | 0.13114167724173384 | 0.0166008946049258 | 0.328259467804395  |
| full                  | 0.8                | 0.27386127875258304 | 0.3333333333333333 | 1.0                | 0.7666666666666666          | 0.2725904538966021         | 0.3333333333333333         | 1.0                        | 0.7611111111111111 | 0.3065639924738088  | 0.25               | 1.0                | 0.84375           | 0.1875              | 0.625            | 1.0              | 0.09089539991245914 | 0.08840995726359256 | 0.000209899619821  | 0.1668308366718692 | 0.11164396706666593 | 0.06498222164683444 | 0.009018247407062  | 0.1667993741344546 |
| drop_temporal         | 0.8                | 0.27386127875258304 | 0.3333333333333333 | 1.0                | 0.7666666666666666          | 0.2725904538966021         | 0.3333333333333333         | 1.0                        | 0.7611111111111111 | 0.3065639924738088  | 0.25               | 1.0                | 0.84375           | 0.1875              | 0.625            | 1.0              | 0.09012386306535615 | 0.0891153149949917  | 0.0002404774801231 | 0.1668044078854851 | 0.10992160349712576 | 0.06467736135130642 | 0.0093628497880561 | 0.1666197104464259 |
| drop_biology          | 0.7533333333333333 | 0.16261747890200626 | 0.6                | 1.0                | 0.6583333333333333          | 0.2400810048481322         | 0.375                      | 1.0                        | 0.5905555555555555 | 0.2836300189583623  | 0.375              | 1.0                | 0.75              | 0.2041241452319315  | 0.5              | 1.0              | 0.17172057106665398 | 0.13631191719154864 | 0.0001801251094668 | 0.3332495366292613 | 0.15595372580084962 | 0.12956146623296047 | 0.0069091383875907 | 0.3327409511989403 |
| weather_daylight_only | 0.7266666666666667 | 0.14414498873626438 | 0.5                | 0.8333333333333334 | 0.6383333333333333          | 0.20628997929021073        | 0.375                      | 0.9                        | 0.5466666666666666 | 0.21464872250743555 | 0.3333333333333333 | 0.7777777777777777 | 0.59375           | 0.3442231592053814  | 0.25             | 1.0              | 0.2038492415310504  | 0.10771422906381171 | 0.0780475101091068 | 0.3301642287550921 | 0.29293345939598636 | 0.10152379949609268 | 0.1813719029388327 | 0.4050441996900663 |
| biology_only          | 0.6533333333333333 | 0.20628997929021073 | 0.3333333333333333 | 0.8333333333333334 | 0.6216666666666667          | 0.2166346130136077         | 0.375                      | 0.8333333333333334         | 0.5097979797979797 | 0.17569065833963246 | 0.3333333333333333 | 0.7777777777777777 | 0.79375           | 0.21250000000000002 | 0.5              | 1.0              | 0.29130661117614637 | 0.1448545937370932  | 0.1666130583130527 | 0.4686617763575098 | 0.38912107856709743 | 0.1413371282840179  | 0.2057669556164286 | 0.5888978847160963 |
| drop_weather_daylight | 0.5599999999999999 | 0.27121127475743995 | 0.1666666666666666 | 0.8333333333333334 | 0.5983333333333334          | 0.27653611377580006        | 0.1666666666666666         | 0.875                      | 0.5307936507936508 | 0.25931281165949616 | 0.1428571428571428 | 0.7777777777777777 | 0.875             | 0.17677669529663687 | 0.625            | 1.0              | 0.1210268249307519  | 0.07110472234143933 | 0.0186377516696914 | 0.1725629056220004 | 0.20705921115874654 | 0.1122145186825677  | 0.0950747858308884 | 0.394752736889734  |
| drop_meals            | 0.5666666666666667 | 0.383695481107378   | 0.0                | 1.0                | 0.5900000000000001          | 0.3748332962798262         | 0.0                        | 1.0                        | 0.5498412698412698 | 0.3758815957836707  | 0.0                | 1.0                | 0.8125            | 0.21650635094610965 | 0.625            | 1.0              | 0.14630182205305695 | 0.06743154467274416 | 0.0728654058147447 | 0.2295196904910111 | 0.29467903399021306 | 0.22960838235225667 | 0.1588016013475826 | 0.699845701547773  |
| temporal_only         | 0.7266666666666667 | 0.18618986725025255 | 0.5                | 1.0                | 0.5549999999999999          | 0.2551960031034969         | 0.375                      | 1.0                        | 0.5155555555555555 | 0.27370345311470506 | 0.3333333333333333 | 1.0                | 0.3               | 0.31885210782848317 | 0.0              | 0.75             | 0.29700865560247813 | 0.04571304777834791 | 0.2415655630030531 | 0.3420074539673383 | 0.43489660340131425 | 0.11162018153648627 | 0.3247489460432515 | 0.5911905238192833 |
