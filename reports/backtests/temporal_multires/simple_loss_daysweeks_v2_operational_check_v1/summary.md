# Operational Check: simple_loss_daysweeks_v2

- target: `y_next_weight_loss_flag`
- modalities: `days,weeks`
- model family: `flattened ExtraTrees`
- saved selected threshold: `0.4288`

## Calibration

- lowest ECE series on test: `isotonic_on_val` (ece=0.0944, balanced_accuracy=0.8611, threshold=0.0417)
- raw saved-model ECE: `0.3137`

## Operating Points

- saved selected operating point: `threshold=0.4288, balanced_accuracy=0.8611, fp=10, fn=0, positive_rate_pred=0.3333`
- best zero-FN operating point on test sweep: `threshold=0.4603, balanced_accuracy=0.9444, fp=4, positive_rate_pred=0.1795`

## Time-Aware Check

- rolling folds: `4` (balanced_accuracy mean/min/max = 0.6138 / 0.4152 / 0.7037)
- latest rolling fold: `balanced_accuracy=0.7037, roc_auc=0.8519`

## Segments

- weekend anchor slice: `balanced_accuracy=1.0000, error_rate=0.0000, n=12`
- recent restaurant-heavy slice: `balanced_accuracy=0.9048, error_rate=0.1739, n=23`
