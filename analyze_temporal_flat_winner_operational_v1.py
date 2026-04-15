from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_temporal_multires_simple_baselines_v1 import (
    build_feature_frame,
    choose_threshold,
    coerce_binary_series,
    ensure_dir,
    prediction_distribution,
    save_json,
)


TARGET_NAME = "y_next_weight_loss_flag"
RANDOM_STATE = 42


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def probability_series(pipe, X: pd.DataFrame) -> np.ndarray:
    prob = pipe.predict_proba(X)
    if prob.ndim == 1:
        return prob.astype(float)
    return prob[:, 1].astype(float)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(pred)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    out.update(prediction_distribution(prob))
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 8) -> Tuple[float, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = np.digitize(y_prob, bins[1:-1], right=True)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = bucket == b
        if mask.sum() == 0:
            rows.append(
                {
                    "bin": int(b),
                    "count": 0,
                    "prob_mean": np.nan,
                    "empirical_rate": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        p_mean = float(np.mean(y_prob[mask]))
        y_mean = float(np.mean(y_true[mask]))
        gap = abs(p_mean - y_mean)
        ece += gap * (mask.sum() / len(y_true))
        rows.append(
            {
                "bin": int(b),
                "count": int(mask.sum()),
                "prob_mean": p_mean,
                "empirical_rate": y_mean,
                "abs_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def fit_calibrators_on_val(val_prob: np.ndarray, y_val: np.ndarray):
    if len(np.unique(y_val)) < 2:
        return None, None
    platt = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    platt.fit(val_prob.reshape(-1, 1), y_val)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_prob, y_val)
    return platt, isotonic


def apply_platt(model, raw_prob: np.ndarray) -> np.ndarray:
    return model.predict_proba(raw_prob.reshape(-1, 1))[:, 1]


def calibration_comparison(
    y_val: np.ndarray,
    val_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
    selected_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bins_frames = []

    raw_ece, raw_bins = expected_calibration_error(y_test, test_prob, n_bins=8)
    raw_metrics = classification_metrics(y_test, test_prob, threshold=selected_threshold)
    rows.append(
        {
            "series": "raw_saved_model",
            "threshold_rule": "saved_selected_threshold",
            "ece": raw_ece,
            **raw_metrics,
        }
    )
    bins_frames.append(raw_bins.assign(series="raw_saved_model"))

    platt_model, isotonic_model = fit_calibrators_on_val(val_prob, y_val)
    if platt_model is not None:
        val_prob_platt = apply_platt(platt_model, val_prob)
        test_prob_platt = apply_platt(platt_model, test_prob)
        thr_platt = choose_threshold(y_val, val_prob_platt)
        platt_ece, platt_bins = expected_calibration_error(y_test, test_prob_platt, n_bins=8)
        platt_metrics = classification_metrics(y_test, test_prob_platt, threshold=thr_platt)
        rows.append(
            {
                "series": "platt_on_val",
                "threshold_rule": "val_balanced_accuracy_tuned",
                "ece": platt_ece,
                **platt_metrics,
            }
        )
        bins_frames.append(platt_bins.assign(series="platt_on_val"))

    if isotonic_model is not None:
        val_prob_iso = isotonic_model.predict(val_prob)
        test_prob_iso = isotonic_model.predict(test_prob)
        thr_iso = choose_threshold(y_val, val_prob_iso)
        iso_ece, iso_bins = expected_calibration_error(y_test, test_prob_iso, n_bins=8)
        iso_metrics = classification_metrics(y_test, test_prob_iso, threshold=thr_iso)
        rows.append(
            {
                "series": "isotonic_on_val",
                "threshold_rule": "val_balanced_accuracy_tuned",
                "ece": iso_ece,
                **iso_metrics,
            }
        )
        bins_frames.append(iso_bins.assign(series="isotonic_on_val"))

    compare_df = pd.DataFrame(rows)
    bins_df = pd.concat(bins_frames, ignore_index=True)
    return compare_df, bins_df


def threshold_operating_table(y_true: np.ndarray, prob: np.ndarray, selected_threshold: float) -> pd.DataFrame:
    thresholds = list(np.linspace(0.35, 0.47, 25))
    thresholds.extend(prob.tolist())
    thresholds.append(selected_threshold)
    rows = []
    seen = set()
    for threshold in sorted(thresholds):
        threshold = round(float(threshold), 4)
        if threshold in seen:
            continue
        seen.add(threshold)
        metrics = classification_metrics(y_true, prob, threshold=threshold)
        metrics["false_positive_rate"] = float(metrics["fp"] / max(metrics["fp"] + metrics["tn"], 1))
        metrics["is_selected_threshold"] = int(abs(threshold - selected_threshold) < 1e-9)
        metrics["zero_false_negative"] = int(metrics["fn"] == 0)
        metrics["operating_zone"] = (
            "recall_preserving"
            if metrics["fn"] == 0 and threshold >= 0.42
            else "aggressive"
            if threshold < 0.42
            else "precision_leaning"
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def build_time_aware_folds(valid_df: pd.DataFrame, min_train: int, calibration_size: int, eval_size: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    folds = []
    n = len(valid_df)
    train_end = min_train
    while train_end + calibration_size + eval_size <= n:
        cal_end = train_end + calibration_size
        eval_end = cal_end + eval_size
        train_idx = valid_df.iloc[:train_end]["index"].to_numpy(dtype=int)
        cal_idx = valid_df.iloc[train_end:cal_end]["index"].to_numpy(dtype=int)
        eval_idx = valid_df.iloc[cal_end:eval_end]["index"].to_numpy(dtype=int)
        folds.append((train_idx, cal_idx, eval_idx))
        train_end = eval_end
    return folds


def fit_same_family_model(X_train: pd.DataFrame, y_train: np.ndarray):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def rolling_time_aware_check(
    feature_df: pd.DataFrame,
    anchors: pd.DataFrame,
    target_name: str,
    min_train: int,
    calibration_size: int,
    eval_size: int,
) -> pd.DataFrame:
    valid_target = coerce_binary_series(anchors[target_name]).notna()
    ordered = anchors.loc[valid_target].copy()
    ordered["anchor_period_start"] = pd.to_datetime(ordered["anchor_period_start"])
    ordered = ordered.sort_values("anchor_period_start").reset_index()

    folds = build_time_aware_folds(
        valid_df=ordered,
        min_train=min_train,
        calibration_size=calibration_size,
        eval_size=eval_size,
    )

    rows = []
    for fold_num, (train_idx, cal_idx, eval_idx) in enumerate(folds, start=1):
        y_train = coerce_binary_series(anchors.loc[train_idx, target_name]).astype(int).to_numpy()
        y_cal = coerce_binary_series(anchors.loc[cal_idx, target_name]).astype(int).to_numpy()
        y_eval = coerce_binary_series(anchors.loc[eval_idx, target_name]).astype(int).to_numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_cal)) < 2 or len(np.unique(y_eval)) < 2:
            continue

        pipe = fit_same_family_model(feature_df.loc[train_idx], y_train)
        cal_prob = probability_series(pipe, feature_df.loc[cal_idx])
        eval_prob = probability_series(pipe, feature_df.loc[eval_idx])
        threshold = choose_threshold(y_cal, cal_prob)
        metrics = classification_metrics(y_eval, eval_prob, threshold=threshold)

        rows.append(
            {
                "fold": int(fold_num),
                "train_n": int(len(train_idx)),
                "calibration_n": int(len(cal_idx)),
                "eval_n": int(len(eval_idx)),
                "train_start": str(pd.to_datetime(anchors.loc[train_idx, "anchor_period_start"]).min().date()),
                "train_end": str(pd.to_datetime(anchors.loc[train_idx, "anchor_period_start"]).max().date()),
                "calibration_start": str(pd.to_datetime(anchors.loc[cal_idx, "anchor_period_start"]).min().date()),
                "calibration_end": str(pd.to_datetime(anchors.loc[cal_idx, "anchor_period_start"]).max().date()),
                "eval_start": str(pd.to_datetime(anchors.loc[eval_idx, "anchor_period_start"]).min().date()),
                "eval_end": str(pd.to_datetime(anchors.loc[eval_idx, "anchor_period_start"]).max().date()),
                "eval_positive_rate_true": float(np.mean(y_eval)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def segmented_error_analysis(
    merged_test: pd.DataFrame,
    selected_threshold: float,
) -> pd.DataFrame:
    prob_col = f"{TARGET_NAME}__prob"
    true_col = f"{TARGET_NAME}__true"
    pred = (merged_test[prob_col].to_numpy(dtype=float) >= selected_threshold).astype(int)
    y_true = merged_test[true_col].to_numpy(dtype=float).astype(int)

    base = merged_test.copy()
    base["pred"] = pred
    base["is_error"] = (pred != y_true).astype(int)
    base["anchor_period_start"] = pd.to_datetime(base["anchor_period_start"])
    base["anchor_is_weekend"] = (base["anchor_period_start"].dt.dayofweek >= 5).astype(int)
    base["recent_restaurant_any"] = (pd.to_numeric(base["days__t_minus_0__restaurant_meal_count_day"], errors="coerce").fillna(0.0) > 0).astype(int)
    base["recent_restaurant_heavy"] = (pd.to_numeric(base["days__t_minus_0__restaurant_meal_fraction_day"], errors="coerce").fillna(0.0) >= 0.5).astype(int)
    restaurant_week = pd.to_numeric(base["weeks__t_minus_0__restaurant_meal_fraction_week"], errors="coerce").fillna(0.0)
    base["recent_week_restaurant_above_median"] = (restaurant_week >= float(restaurant_week.median())).astype(int)

    segment_specs = [
        ("anchor_is_weekend", {0: "weekday_anchor", 1: "weekend_anchor"}),
        ("recent_restaurant_any", {0: "no_recent_restaurant", 1: "recent_restaurant_any"}),
        ("recent_restaurant_heavy", {0: "recent_restaurant_light", 1: "recent_restaurant_heavy"}),
        ("recent_week_restaurant_above_median", {0: "week_restaurant_below_median", 1: "week_restaurant_above_median"}),
    ]

    rows = []
    for segment_name, labels in segment_specs:
        for raw_value, label in labels.items():
            sub = base[base[segment_name] == raw_value].copy()
            if len(sub) < 5:
                continue
            y_sub = sub[true_col].to_numpy(dtype=float).astype(int)
            prob_sub = sub[prob_col].to_numpy(dtype=float)
            metrics = classification_metrics(y_sub, prob_sub, threshold=selected_threshold)
            metrics.update(
                {
                    "segment_name": segment_name,
                    "segment_label": label,
                    "error_rate": float(sub["is_error"].mean()),
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows).sort_values(["segment_name", "segment_label"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Operational robustness and calibration checks for the flattened temporal winner.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-name", default="simple_loss_daysweeks_v2")
    parser.add_argument("--target", default=TARGET_NAME)
    parser.add_argument("--analysis-name", default="")
    parser.add_argument("--min-train-rows", type=int, default=168)
    parser.add_argument("--calibration-window-rows", type=int, default=28)
    parser.add_argument("--eval-window-rows", type=int, default=28)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    run_name = args.run_name
    target_name = args.target
    analysis_name = args.analysis_name or f"{run_name}_operational_check_v1"

    report_dir = project_root / "reports" / "backtests" / "temporal_multires" / analysis_name
    ensure_dir(report_dir)

    run_dir = project_root / "reports" / "backtests" / "temporal_multires" / run_name
    config = read_json(run_dir / "config.json")
    selected = read_json(run_dir / "selected_models.json")["models"][0]
    selected_threshold = float(read_json(run_dir / "selected_thresholds.json")[target_name])

    dataset_dir = Path(config["dataset_dir"])
    anchors = pd.read_csv(dataset_dir / "anchors.csv", low_memory=False)
    anchors["anchor_id"] = anchors["anchor_id"].astype(str)

    feature_df, feature_meta = build_feature_frame(
        anchors=anchors,
        dataset_dir=dataset_dir,
        enabled_modalities=list(config["modalities"]),
        windows=dict(config["windows"]),
    )

    y_all = coerce_binary_series(anchors[target_name])
    valid = y_all.notna()
    split = anchors["split_suggested"].astype(str)
    train_mask = (split == "train") & valid
    val_mask = (split == "val") & valid
    test_mask = (split == "test") & valid

    X_val = feature_df.loc[val_mask].copy()
    X_test = feature_df.loc[test_mask].copy()
    y_val = y_all.loc[val_mask].astype(int).to_numpy()
    y_test = y_all.loc[test_mask].astype(int).to_numpy()

    pipe = joblib.load(selected["model_artifact"])
    val_prob = probability_series(pipe, X_val)
    test_prob = probability_series(pipe, X_test)

    calibration_compare_df, calibration_bins_df = calibration_comparison(
        y_val=y_val,
        val_prob=val_prob,
        y_test=y_test,
        test_prob=test_prob,
        selected_threshold=selected_threshold,
    )
    calibration_compare_df.to_csv(report_dir / "calibration_compare.csv", index=False)
    calibration_bins_df.to_csv(report_dir / "calibration_bins_compare.csv", index=False)

    threshold_df = threshold_operating_table(
        y_true=y_test,
        prob=test_prob,
        selected_threshold=selected_threshold,
    )
    threshold_df.to_csv(report_dir / "threshold_operating_table.csv", index=False)

    test_pred_df = pd.read_csv(run_dir / "test_predictions.csv")
    test_pred_df["anchor_id"] = test_pred_df["anchor_id"].astype(str)
    context_cols = [
        "anchor_id",
        "anchor_period_start",
        "anchor_next_period_start",
        "y_next_weight_delta_lb",
    ]
    context_df = anchors[context_cols].copy()
    merged_test = test_pred_df.merge(context_df, on="anchor_id", how="left")
    merged_test = merged_test.join(feature_df.loc[test_mask, [
        "days__t_minus_0__is_weekend",
        "days__t_minus_0__restaurant_meal_count_day",
        "days__t_minus_0__restaurant_meal_fraction_day",
        "weeks__t_minus_0__restaurant_meal_fraction_week",
    ]].reset_index(drop=True))
    segmented_df = segmented_error_analysis(merged_test, selected_threshold=selected_threshold)
    segmented_df.to_csv(report_dir / "segmented_error_analysis.csv", index=False)

    rolling_df = rolling_time_aware_check(
        feature_df=feature_df,
        anchors=anchors,
        target_name=target_name,
        min_train=int(args.min_train_rows),
        calibration_size=int(args.calibration_window_rows),
        eval_size=int(args.eval_window_rows),
    )
    rolling_df.to_csv(report_dir / "time_aware_rolling_check.csv", index=False)

    selected_row = threshold_df.loc[threshold_df["is_selected_threshold"] == 1].iloc[0].to_dict()
    zero_fn_df = threshold_df[threshold_df["zero_false_negative"] == 1].copy()
    zero_fn_best = (
        zero_fn_df.sort_values(["balanced_accuracy", "fp", "threshold"], ascending=[False, True, False]).iloc[0].to_dict()
        if not zero_fn_df.empty
        else {}
    )
    calibration_best = calibration_compare_df.sort_values(["ece", "balanced_accuracy"], ascending=[True, False]).iloc[0].to_dict()
    rolling_summary = {}
    if not rolling_df.empty:
        rolling_summary = {
            "fold_count": int(len(rolling_df)),
            "balanced_accuracy_mean": float(rolling_df["balanced_accuracy"].mean()),
            "balanced_accuracy_min": float(rolling_df["balanced_accuracy"].min()),
            "balanced_accuracy_max": float(rolling_df["balanced_accuracy"].max()),
            "roc_auc_mean": float(rolling_df["roc_auc"].mean()),
            "latest_fold_balanced_accuracy": float(rolling_df.iloc[-1]["balanced_accuracy"]),
            "latest_fold_roc_auc": float(rolling_df.iloc[-1]["roc_auc"]),
        }

    weekend_segment = segmented_df[segmented_df["segment_label"] == "weekend_anchor"]
    restaurant_heavy_segment = segmented_df[segmented_df["segment_label"] == "recent_restaurant_heavy"]

    summary_lines = [
        f"# Operational Check: {run_name}",
        "",
        f"- target: `{target_name}`",
        f"- modalities: `{','.join(config['modalities'])}`",
        f"- model family: `flattened ExtraTrees`",
        f"- saved selected threshold: `{selected_threshold:.4f}`",
        "",
        "## Calibration",
        "",
        (
            f"- lowest ECE series on test: `{calibration_best.get('series', 'n/a')}` "
            f"(ece={calibration_best.get('ece', float('nan')):.4f}, "
            f"balanced_accuracy={calibration_best.get('balanced_accuracy', float('nan')):.4f}, "
            f"threshold={calibration_best.get('threshold', float('nan')):.4f})"
            if calibration_best
            else "- calibration comparison unavailable"
        ),
        (
            f"- raw saved-model ECE: `{float(calibration_compare_df.loc[calibration_compare_df['series'] == 'raw_saved_model', 'ece'].iloc[0]):.4f}`"
            if "raw_saved_model" in calibration_compare_df["series"].tolist()
            else "- raw saved-model ECE unavailable"
        ),
        "",
        "## Operating Points",
        "",
        f"- saved selected operating point: `threshold={selected_row['threshold']:.4f}, balanced_accuracy={selected_row['balanced_accuracy']:.4f}, fp={int(selected_row['fp'])}, fn={int(selected_row['fn'])}, positive_rate_pred={selected_row['positive_rate_pred']:.4f}`",
        (
            f"- best zero-FN operating point on test sweep: `threshold={zero_fn_best.get('threshold', float('nan')):.4f}, "
            f"balanced_accuracy={zero_fn_best.get('balanced_accuracy', float('nan')):.4f}, "
            f"fp={int(zero_fn_best.get('fp', 0))}, positive_rate_pred={zero_fn_best.get('positive_rate_pred', float('nan')):.4f}`"
            if zero_fn_best
            else "- zero-FN operating point unavailable"
        ),
        "",
        "## Time-Aware Check",
        "",
        (
            f"- rolling folds: `{rolling_summary.get('fold_count', 0)}` "
            f"(balanced_accuracy mean/min/max = {rolling_summary.get('balanced_accuracy_mean', float('nan')):.4f} / "
            f"{rolling_summary.get('balanced_accuracy_min', float('nan')):.4f} / "
            f"{rolling_summary.get('balanced_accuracy_max', float('nan')):.4f})"
            if rolling_summary
            else "- rolling folds unavailable"
        ),
        (
            f"- latest rolling fold: `balanced_accuracy={rolling_summary.get('latest_fold_balanced_accuracy', float('nan')):.4f}, "
            f"roc_auc={rolling_summary.get('latest_fold_roc_auc', float('nan')):.4f}`"
            if rolling_summary
            else ""
        ),
        "",
        "## Segments",
        "",
        (
            f"- weekend anchor slice: `balanced_accuracy={float(weekend_segment['balanced_accuracy'].iloc[0]):.4f}, error_rate={float(weekend_segment['error_rate'].iloc[0]):.4f}, n={int(weekend_segment['n'].iloc[0])}`"
            if not weekend_segment.empty
            else "- weekend slice unavailable"
        ),
        (
            f"- recent restaurant-heavy slice: `balanced_accuracy={float(restaurant_heavy_segment['balanced_accuracy'].iloc[0]):.4f}, error_rate={float(restaurant_heavy_segment['error_rate'].iloc[0]):.4f}, n={int(restaurant_heavy_segment['n'].iloc[0])}`"
            if not restaurant_heavy_segment.empty
            else "- recent restaurant-heavy slice unavailable"
        ),
        "",
    ]
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    save_json(
        report_dir / "analysis_manifest.json",
        {
            "source_run_name": run_name,
            "analysis_name": analysis_name,
            "target": target_name,
            "selected_threshold": selected_threshold,
            "feature_build": feature_meta,
            "time_aware_check": {
                "min_train_rows": int(args.min_train_rows),
                "calibration_window_rows": int(args.calibration_window_rows),
                "eval_window_rows": int(args.eval_window_rows),
            },
            "report_files": sorted([p.name for p in report_dir.iterdir() if p.is_file()]),
        },
    )


if __name__ == "__main__":
    main()
