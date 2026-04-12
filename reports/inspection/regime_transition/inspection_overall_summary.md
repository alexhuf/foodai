# Regime Transition Inspection Summary

## Target summary

| space    | target                       | kind                  | best_model | test_accuracy      | test_balanced_accuracy | test_macro_f1      | test_roc_auc       | n_test | n_errors | feature_driver_type |
| -------- | ---------------------------- | --------------------- | ---------- | ------------------ | ---------------------- | ------------------ | ------------------ | ------ | -------- | ------------------- |
| weeks    | y_next_weight_gain_flag      | binary_classification | logreg     | 1.0                | 1.0                    | 1.0                | 1.0                | 6      | 0        | linear_coefficient  |
| weeks    | y_next_weight_loss_flag      | binary_classification | et         | 0.6666666666666666 | 0.6666666666666666     | 0.6666666666666666 | 0.5555555555555556 | 6      | 2        | tree_importance     |
| weekends | y_next_restaurant_heavy_flag | binary_classification | rf         | 0.8571428571428571 | 0.875                  | 0.8571428571428571 | 0.75               | 7      | 1        | tree_importance     |

## Initial read

- weeks/y_next_weight_gain_flag (logreg), macro_f1=1.000, balanced_acc=1.000
- weeks/y_next_weight_loss_flag (et), macro_f1=0.667, balanced_acc=0.667
- weekends/y_next_restaurant_heavy_flag (rf), macro_f1=0.857, balanced_acc=0.875
