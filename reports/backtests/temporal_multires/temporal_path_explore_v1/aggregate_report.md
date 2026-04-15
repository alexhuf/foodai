# Temporal Path Exploration Results: temporal_path_explore_v1

## Ranked Results

```csv
candidate_name,path_family,source_kind,balanced_accuracy_tuned,roc_auc_tuned,f1_tuned,prob_std,positive_rate_pred,status,status_reason,delta_vs_simple_floor_balanced_accuracy,delta_vs_simple_floor_roc_auc,delta_vs_tcn_smoke_balanced_accuracy,delta_vs_tcn_smoke_roc_auc
simple_loss_daysweeks_v2,reference,reference,0.8611111111111112,0.9166666666666666,0.375,0.06593768411412655,0.3333333333333333,promote,clears bounded smoke criteria,0.0,0.0,0.2777777777777778,0.41666666666666663
flat_loss_daysweeks_explore_v1:et_balanced,flattened_explore,new_flattened_candidate,0.8194444444444444,0.8611111111111112,0.3157894736842105,0.0748495472669055,0.4102564102564102,promote,clears bounded smoke criteria,-0.04166666666666674,-0.05555555555555547,0.23611111111111105,0.36111111111111116
flat_loss_daysweeks_explore_v1:hgb_depth3,flattened_explore,new_flattened_candidate,0.6805555555555556,0.8055555555555556,0.25,0.0655727290962292,0.3333333333333333,promote,clears bounded smoke criteria,-0.18055555555555558,-0.11111111111111105,0.09722222222222221,0.3055555555555556
flat_loss_daysweeks_explore_v1:rf_balanced,flattened_explore,new_flattened_candidate,0.6944444444444444,0.7962962962962963,0.2666666666666666,0.0462393708334214,0.3076923076923077,promote,clears bounded smoke criteria,-0.16666666666666674,-0.12037037037037035,0.11111111111111105,0.2962962962962963
flat_loss_daysweeks_explore_v1:logreg_balanced_c5,flattened_explore,new_flattened_candidate,0.7222222222222222,0.6944444444444444,0.2307692307692307,0.3303617073071439,0.5897435897435898,promote,clears bounded smoke criteria,-0.13888888888888895,-0.2222222222222222,0.13888888888888884,0.19444444444444442
flat_loss_daysweeks_explore_v1:logreg_balanced,flattened_explore,new_flattened_candidate,0.5,0.75,0.0,0.3120678119166775,0.0,reject,effectively one-class predictions; balanced accuracy below promotion floor,-0.36111111111111116,-0.16666666666666663,-0.08333333333333337,0.25
gru_loss_daysweeks_bce_smoke_v1,neural_sequence,new_neural_run,0.5,0.6851851851851851,0.125,0.017148344871675318,0.3333333333333333,reject,balanced accuracy below promotion floor,-0.36111111111111116,-0.2314814814814815,-0.08333333333333337,0.18518518518518512
flat_loss_daysweeks_explore_v1:mlp_wide,flattened_explore,new_flattened_candidate,0.5,0.6481481481481481,0.0,0.2664132256792146,0.0,reject,effectively one-class predictions; balanced accuracy below promotion floor,-0.36111111111111116,-0.2685185185185185,-0.08333333333333337,0.14814814814814814
tcn_loss_daysweeks_bce_smoke_v1,neural_sequence,new_neural_run,0.4861111111111111,0.5277777777777778,0.13333333333333333,0.009400225661983161,0.6923076923076923,reject,under-dispersed probabilities; ranking too weak; balanced accuracy below promotion floor,-0.37500000000000006,-0.38888888888888884,-0.09722222222222227,0.02777777777777779
gru_loss_daysweeks_smoke_v4_1,reference,reference,0.4444444444444444,0.5277777777777778,0.0,0.020189868640548156,0.10256410256410256,reject,ranking too weak; balanced accuracy below promotion floor,-0.41666666666666674,-0.38888888888888884,-0.13888888888888895,0.02777777777777779
tcn_loss_daysweeks_compare_smoke_v1_check,reference,reference,0.5833333333333334,0.5,0.2,0.00903124458920409,0.1794871794871795,reject,under-dispersed probabilities; ranking too weak,-0.2777777777777778,-0.41666666666666663,0.0,0.0
flat_loss_daysweeks_explore_v1:dummy_majority,flattened_explore,new_flattened_candidate,0.5,0.5,0.1428571428571428,0.0,1.0,reject,effectively one-class predictions; under-dispersed probabilities; ranking too weak; balanced accuracy below promotion floor,-0.36111111111111116,-0.41666666666666663,-0.08333333333333337,0.0
tcn_loss_daysweeks_compare_pilot_v1,reference,reference,0.5,0.4722222222222222,0.14285714285714285,0.008320564359854172,1.0,reject,effectively one-class predictions; under-dispersed probabilities; ranking too weak; balanced accuracy below promotion floor,-0.36111111111111116,-0.4444444444444444,-0.08333333333333337,-0.02777777777777779
flat_loss_daysweeks_explore_v1:mlp_small,flattened_explore,new_flattened_candidate,0.611111111111111,0.4351851851851852,0.25,0.1350888356265228,0.1282051282051282,reject,ranking too weak,-0.2500000000000001,-0.48148148148148145,0.02777777777777768,-0.06481481481481483
tcn_loss_daysweeks_bce_deep_smoke_v1,neural_sequence,new_neural_run,0.5,0.28703703703703703,0.0,0.019696565727298117,0.0,reject,effectively one-class predictions; ranking too weak; balanced accuracy below promotion floor,-0.36111111111111116,-0.6296296296296295,-0.08333333333333337,-0.21296296296296297
```

## Surviving Paths

- `flat_loss_daysweeks_explore_v1:et_balanced`: ROC AUC `0.8611`, balanced accuracy `0.8194`, prob_std `0.0748`
- `flat_loss_daysweeks_explore_v1:hgb_depth3`: ROC AUC `0.8056`, balanced accuracy `0.6806`, prob_std `0.0656`
