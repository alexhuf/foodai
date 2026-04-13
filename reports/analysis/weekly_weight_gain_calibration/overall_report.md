# Weekly Weight-Gain Probability Calibration Report

- target: weeks / y_next_weight_gain_flag
- goal: compare raw probabilities versus pooled Platt and isotonic calibration using canonical rolling out-of-fold predictions

## Mode: selected_best

- folds: 2
- overlapping predictions: 16
- canonical predictions: 14
- duplicates collapsed: 4

### Raw metrics
- accuracy: 0.8571
- balanced_accuracy: 0.6667
- macro_f1: 0.7083
- positive_rate_pred: 0.0714
- positive_rate_true: 0.2143
- roc_auc: 0.7121
- brier: 0.1428
- log_loss: 1.5178
- ece: 0.1422
- threshold: 0.5000
- ece: 0.1422

### Platt metrics
- accuracy: 0.7857
- balanced_accuracy: 0.5000
- macro_f1: 0.4400
- positive_rate_pred: 0.0000
- positive_rate_true: 0.2143
- roc_auc: 0.7121
- brier: 0.1550
- log_loss: 0.4851
- ece: 0.0951
- threshold: 0.5000
- ece: 0.0951

### Isotonic metrics
- accuracy: 0.8571
- balanced_accuracy: 0.6667
- macro_f1: 0.7083
- positive_rate_pred: 0.0714
- positive_rate_true: 0.2143
- roc_auc: 0.7727
- brier: 0.1119
- log_loss: 0.3686
- ece: 0.0000
- threshold: 0.5000
- ece: 0.0000

### Suggested thresholds
- raw / balanced_accuracy: 0.0002
- raw / macro_f1: 0.0073
- platt / balanced_accuracy: 0.2050
- platt / macro_f1: 0.2050
- isotonic / balanced_accuracy: 0.1500
- isotonic / macro_f1: 0.1500

### Fold summary

| mode          | fold | chosen_model   | train_n | val_n | test_n | train_positive_rate | val_positive_rate | test_positive_rate | default_accuracy | default_balanced_accuracy | default_macro_f1   | default_positive_rate_pred | default_positive_rate_true | default_roc_auc    | default_brier      | default_log_loss   | default_ece       | default_threshold |
| ------------- | ---- | -------------- | ------- | ----- | ------ | ------------------- | ----------------- | ------------------ | ---------------- | ------------------------- | ------------------ | -------------------------- | -------------------------- | ------------------ | ------------------ | ------------------ | ----------------- | ----------------- |
| selected_best | 1    | dummy_majority | 36      | 8     | 8      | 0.1111111111111111  | 0.125             | 0.125              | 0.875            | 0.5                       | 0.4666666666666667 | 0.0                        | 0.125                      | 0.5                | 0.125              | 1.726939694745972  | 0.125             | 0.5               |
| selected_best | 2    | logreg         | 42      | 8     | 8      | 0.119047619047619   | 0.125             | 0.25               | 0.875            | 0.75                      | 0.7948717948717949 | 0.125                      | 0.25                       | 0.6666666666666667 | 0.1253235890928292 | 0.9397909363977448 | 0.113425348232338 | 0.5               |

## Mode: logreg_fixed

- folds: 2
- overlapping predictions: 16
- canonical predictions: 14
- duplicates collapsed: 4

### Raw metrics
- accuracy: 0.9286
- balanced_accuracy: 0.8333
- macro_f1: 0.8783
- positive_rate_pred: 0.1429
- positive_rate_true: 0.2143
- roc_auc: 0.7576
- brier: 0.0726
- log_loss: 0.5480
- ece: 0.0722
- threshold: 0.5000
- ece: 0.0722

### Platt metrics
- accuracy: 0.7857
- balanced_accuracy: 0.5000
- macro_f1: 0.4400
- positive_rate_pred: 0.0000
- positive_rate_true: 0.2143
- roc_auc: 0.7576
- brier: 0.1250
- log_loss: 0.4106
- ece: 0.1700
- threshold: 0.5000
- ece: 0.1700

### Isotonic metrics
- accuracy: 0.9286
- balanced_accuracy: 0.8333
- macro_f1: 0.8783
- positive_rate_pred: 0.1429
- positive_rate_true: 0.2143
- roc_auc: 0.8788
- brier: 0.0635
- log_loss: 0.2242
- ece: 0.0000
- threshold: 0.5000
- ece: 0.0000

### Suggested thresholds
- raw / balanced_accuracy: 0.1160
- raw / macro_f1: 0.1160
- platt / balanced_accuracy: 0.2000
- platt / macro_f1: 0.2000
- isotonic / balanced_accuracy: 0.1500
- isotonic / macro_f1: 0.1500

### Fold summary

| mode         | fold | chosen_model | train_n | val_n | test_n | train_positive_rate | val_positive_rate | test_positive_rate | default_accuracy | default_balanced_accuracy | default_macro_f1   | default_positive_rate_pred | default_positive_rate_true | default_roc_auc    | default_brier      | default_log_loss   | default_ece        | default_threshold |
| ------------ | ---- | ------------ | ------- | ----- | ------ | ------------------- | ----------------- | ------------------ | ---------------- | ------------------------- | ------------------ | -------------------------- | -------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ----------------- |
| logreg_fixed | 1    | logreg       | 36      | 8     | 8      | 0.1111111111111111  | 0.125             | 0.125              | 1.0              | 1.0                       | 1.0                | 0.125                      | 0.125                      | 1.0                | 0.0022747986251486 | 0.0298623435992282 | 0.0286430590257042 | 0.5               |
| logreg_fixed | 2    | logreg       | 42      | 8     | 8      | 0.119047619047619   | 0.125             | 0.25               | 0.875            | 0.75                      | 0.7948717948717949 | 0.125                      | 0.25                       | 0.6666666666666667 | 0.1253235890928292 | 0.9397909363977448 | 0.113425348232338  | 0.5               |
