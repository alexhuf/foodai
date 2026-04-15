# Temporal Path Exploration Plan: temporal_path_explore_v1

## Mission

- Explore bounded, directly comparable paths for `y_next_weight_loss_flag` on `days,weeks`.
- Prune collapsed or weak paths quickly.
- Promote at most the top 1-2 surviving directions to the next pilot stage.

## Reference Runs

- `simple_loss_daysweeks_v2`
- `gru_loss_daysweeks_smoke_v4_1`
- `tcn_loss_daysweeks_compare_smoke_v1_check`
- `tcn_loss_daysweeks_compare_pilot_v1`

## Matrix

1. Path A/B: stronger flattened baselines on the current lag-window representation.
2. Path C/D: cheap neural smoke variants only on `days,weeks` and the loss target.
3. No meals, no regression, no multi-head runs in this loop.

## Candidate Runs

- `flat_loss_daysweeks_explore_v1`: logistic, random forest, extra trees, histogram boosting, small MLPs on flattened windows.
- `gru_loss_daysweeks_bce_smoke_v1`: GRU smoke with BCE instead of focal.
- `tcn_loss_daysweeks_bce_smoke_v1`: TCN smoke with BCE instead of focal.
- `tcn_loss_daysweeks_bce_deep_smoke_v1`: TCN smoke with BCE plus a slightly larger hidden state.

## Promotion Criteria

- no obvious collapse
- interpretable metrics
- balanced accuracy at least `0.55`
- ROC AUC at least `0.55`
- materially stronger than the current weak neural ceiling

## Stop Rules

- reject any run with effectively one-class predictions
- reject any run with `prob_std < 0.01`
- reject any run with missing or non-finite metrics
- do not escalate to a longer pilot unless a smoke run survives the criteria above
