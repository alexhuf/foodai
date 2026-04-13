# Regime Transition Rolling Backtest Summary

| space    | target                       | kind           | n_folds | models_chosen                           | test_accuracy_mean  | test_balanced_accuracy_mean | test_macro_f1_mean  | test_roc_auc_mean  |
| -------- | ---------------------------- | -------------- | ------- | --------------------------------------- | ------------------- | --------------------------- | ------------------- | ------------------ |
| weekends | y_next_restaurant_heavy_flag | classification | 4       | {"rf": 2, "et": 1, "dummy_majority": 1} | 0.37499999999999994 | 0.5208333333333334          | 0.3273809523809524  | 0.6145833333333334 |
| weeks    | y_next_weight_loss_flag      | classification | 5       | {"dummy_majority": 3, "et": 2}          | 0.45999999999999996 | 0.5666666666666667          | 0.37666666666666665 | 0.5916666666666667 |
| weeks    | y_next_weight_gain_flag      | classification | 5       | {"dummy_majority": 3, "logreg": 2}      | 0.8666666666666668  | 0.75                        | 0.7264646464646465  | 0.6875             |