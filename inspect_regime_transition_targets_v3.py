from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


DEFAULT_TARGET_PAIRS = [
    "weeks:y_next_weight_gain_flag",
    "weeks:y_next_weight_loss_flag",
    "weekends:y_next_restaurant_heavy_flag",
]


def log(msg: str) -> None:
    print(f"[regime-inspect] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "period_start" in out.columns:
        t = pd.to_datetime(out["period_start"], errors="coerce")
        out["period_year"] = t.dt.year.astype("float")
        out["period_month"] = t.dt.month.astype("float")
        out["period_quarter"] = t.dt.quarter.astype("float")
        out["period_weekofyear"] = t.dt.isocalendar().week.astype("float")
        out["period_dayofyear"] = t.dt.dayofyear.astype("float")
        angle = 2.0 * np.pi * ((out["period_dayofyear"].fillna(1.0) - 1.0) / 365.25)
        out["period_doy_sin"] = np.sin(angle)
        out["period_doy_cos"] = np.cos(angle)
        if t.notna().any():
            origin = t.min()
            out["period_days_since_start"] = (t - origin).dt.days.astype("float")
        out = out.drop(columns=["period_start"])
    return out


def prepare_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    exclude_cols = [c for c in df.columns if c.startswith("y_")]
    exclude_cols.extend([
        "next_period_id",
        "next_period_start",
        "split_suggested",
        "period_kind",
        "period_id",
        "week_id",
        "weekend_id",
    ])
    exclude_cols = [c for c in exclude_cols if c in df.columns]
    x = df.drop(columns=exclude_cols, errors="ignore").copy()
    x = add_time_features(x)

    for col in x.columns:
        if str(x[col].dtype) == "boolean":
            x[col] = x[col].astype("float")
    return x, exclude_cols


def parse_target_pairs(items: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Target pair must be in 'space:target' form, got: {item}")
        space, target = item.split(":", 1)
        pairs.append((space, target))
    return pairs


def get_feature_names_from_pipe(pipe) -> List[str]:
    pre = pipe.named_steps["preprocessor"]
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        model = pipe.named_steps["model"]
        n = getattr(model, "n_features_in_", None)
        if n is None:
            return []
        return [f"feature_{i:04d}" for i in range(n)]


def extract_feature_drivers(pipe, top_k: int = 20) -> Tuple[pd.DataFrame, str]:
    model = pipe.named_steps["model"]
    feature_names = get_feature_names_from_pipe(pipe)

    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        if len(feature_names) != len(vals):
            feature_names = [f"feature_{i:04d}" for i in range(len(vals))]
        idx = np.argsort(vals)[::-1][:top_k]
        rows = []
        for rank, i in enumerate(idx, start=1):
            rows.append({
                "rank": rank,
                "feature": feature_names[i],
                "score": float(vals[i]),
                "direction": "",
            })
        return pd.DataFrame(rows), "tree_importance"

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2 and coef.shape[0] == 1:
            vals = coef[0]
            if len(feature_names) != len(vals):
                feature_names = [f"feature_{i:04d}" for i in range(len(vals))]
            idx = np.argsort(np.abs(vals))[::-1][:top_k]
            rows = []
            for rank, i in enumerate(idx, start=1):
                rows.append({
                    "rank": rank,
                    "feature": feature_names[i],
                    "score": float(vals[i]),
                    "direction": "positive" if vals[i] > 0 else "negative",
                })
            return pd.DataFrame(rows), "linear_coefficient"
        elif coef.ndim == 2:
            vals = np.mean(np.abs(coef), axis=0)
            if len(feature_names) != len(vals):
                feature_names = [f"feature_{i:04d}" for i in range(len(vals))]
            idx = np.argsort(vals)[::-1][:top_k]
            rows = []
            for rank, i in enumerate(idx, start=1):
                rows.append({
                    "rank": rank,
                    "feature": feature_names[i],
                    "score": float(vals[i]),
                    "direction": "multiclass_abs_mean",
                })
            return pd.DataFrame(rows), "linear_abs_mean"

    return pd.DataFrame(), "none"


def metrics_to_lines(metrics: Dict) -> List[str]:
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
        else:
            lines.append(f"- {k}: {v}")
    return lines


def df_to_markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows_"
    df2 = df.copy()
    for col in df2.columns:
        df2[col] = df2[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(h) for h in df2.columns]
    rows = df2.values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + body_lines)


def make_confusion_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(
        pred_df["y_true_label"],
        pred_df["y_pred_label"],
        rownames=["true"],
        colnames=["pred"],
        dropna=False,
    ).reset_index()


def inspect_classification_target(
    space: str,
    target: str,
    df: pd.DataFrame,
    meta: Dict,
    summary: Dict,
    pred_df: pd.DataFrame,
    pipe,
    out_dir: Path,
) -> Dict:
    x_all, _ = prepare_feature_frame(df)

    period_ids = pred_df["period_id"].astype(str).tolist()
    test_df = df[df["period_id"].astype(str).isin(period_ids)].copy()
    test_df = test_df.set_index(test_df["period_id"].astype(str)).loc[period_ids].reset_index(drop=True)
    x_test = x_all.loc[test_df.index]

    probs = None
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(x_test)
        labels = summary.get("labels", [str(c) for c in range(probs.shape[1])])
        prob_df = pd.DataFrame(probs, columns=[f"p_{lab}" for lab in labels])
        pred_df = pd.concat([pred_df.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)

        pred_df["pred_confidence"] = probs.max(axis=1)
        label_to_col = {lab: f"p_{lab}" for lab in labels}
        true_probs = []
        for _, row in pred_df.iterrows():
            true_lab = row["y_true_label"]
            col = label_to_col.get(true_lab)
            true_probs.append(row[col] if col in pred_df.columns else np.nan)
        pred_df["true_label_probability"] = true_probs

    pred_df["is_correct"] = pred_df["y_true_label"] == pred_df["y_pred_label"]
    pred_df.to_csv(out_dir / "inspection_scored_predictions.csv", index=False)

    confusion = make_confusion_table(pred_df)
    confusion.to_csv(out_dir / "confusion_table.csv", index=False)

    label_balance = (
        pred_df["y_true_label"]
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
    )
    label_balance["fraction"] = label_balance["count"] / max(1, len(pred_df))
    label_balance.to_csv(out_dir / "test_label_balance.csv", index=False)

    hard_errors = pred_df[~pred_df["is_correct"]].copy()
    if "pred_confidence" in hard_errors.columns:
        hard_errors = hard_errors.sort_values("pred_confidence", ascending=False)
    hard_errors.head(25).to_csv(out_dir / "hard_errors.csv", index=False)

    top_correct = pred_df[pred_df["is_correct"]].copy()
    if "pred_confidence" in top_correct.columns:
        top_correct = top_correct.sort_values("pred_confidence", ascending=False)
    top_correct.head(25).to_csv(out_dir / "top_correct.csv", index=False)

    fi_df, fi_kind = extract_feature_drivers(pipe, top_k=20)
    if not fi_df.empty:
        fi_df.to_csv(out_dir / "feature_driver_summary.csv", index=False)

    report_lines = []
    report_lines.append(f"# Inspection Report: {space} / {target}")
    report_lines.append("")
    report_lines.append(f"- target kind: {summary.get('kind')}")
    report_lines.append(f"- best model: {summary.get('best_model')}")
    report_lines.append(f"- description: {meta.get('description', '')}")
    report_lines.append(f"- n_test: {len(pred_df)}")
    report_lines.append("")
    report_lines.append("## Test metrics")
    report_lines.extend(metrics_to_lines(summary.get("test_metrics", {})))
    report_lines.append("")
    report_lines.append("## Test label balance")
    for _, row in label_balance.iterrows():
        report_lines.append(f"- {row['label']}: {int(row['count'])} ({row['fraction']:.3f})")
    report_lines.append("")
    report_lines.append("## Confusion table")
    report_lines.append("")
    report_lines.append(df_to_markdown_table(confusion))
    report_lines.append("")

    if not fi_df.empty:
        report_lines.append(f"## Top feature drivers ({fi_kind})")
        report_lines.append("")
        report_lines.append(df_to_markdown_table(fi_df.head(12)))
        report_lines.append("")

    if len(hard_errors) > 0:
        report_lines.append("## Highest-confidence wrong predictions")
        report_lines.append("")
        cols = [c for c in ["period_id", "period_start", "y_true_label", "y_pred_label", "pred_confidence", "true_label_probability"] if c in hard_errors.columns]
        report_lines.append(df_to_markdown_table(hard_errors[cols], max_rows=10))
        report_lines.append("")

    if len(top_correct) > 0:
        report_lines.append("## Highest-confidence correct predictions")
        report_lines.append("")
        cols = [c for c in ["period_id", "period_start", "y_true_label", "y_pred_label", "pred_confidence", "true_label_probability"] if c in top_correct.columns]
        report_lines.append(df_to_markdown_table(top_correct[cols], max_rows=10))
        report_lines.append("")

    (out_dir / "inspection_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "space": space,
        "target": target,
        "kind": summary.get("kind"),
        "best_model": summary.get("best_model"),
        **{f"test_{k}": v for k, v in summary.get("test_metrics", {}).items()},
        "n_test": int(len(pred_df)),
        "n_errors": int((~pred_df["is_correct"]).sum()),
        "feature_driver_type": fi_kind,
    }


def inspect_regression_target(
    space: str,
    target: str,
    df: pd.DataFrame,
    meta: Dict,
    summary: Dict,
    pred_df: pd.DataFrame,
    pipe,
    out_dir: Path,
) -> Dict:
    pred_df["abs_residual"] = np.abs(pred_df["residual"])
    pred_df = pred_df.sort_values("abs_residual", ascending=False).reset_index(drop=True)
    pred_df.to_csv(out_dir / "inspection_scored_predictions.csv", index=False)

    worst = pred_df.head(25).copy()
    worst.to_csv(out_dir / "largest_residuals.csv", index=False)

    fi_df, fi_kind = extract_feature_drivers(pipe, top_k=20)
    if not fi_df.empty:
        fi_df.to_csv(out_dir / "feature_driver_summary.csv", index=False)

    report_lines = []
    report_lines.append(f"# Inspection Report: {space} / {target}")
    report_lines.append("")
    report_lines.append(f"- target kind: {summary.get('kind')}")
    report_lines.append(f"- best model: {summary.get('best_model')}")
    report_lines.append(f"- description: {meta.get('description', '')}")
    report_lines.append(f"- n_test: {len(pred_df)}")
    report_lines.append("")
    report_lines.append("## Test metrics")
    report_lines.extend(metrics_to_lines(summary.get("test_metrics", {})))
    report_lines.append("")

    if not fi_df.empty:
        report_lines.append(f"## Top feature drivers ({fi_kind})")
        report_lines.append("")
        report_lines.append(df_to_markdown_table(fi_df.head(12)))
        report_lines.append("")

    if len(worst) > 0:
        report_lines.append("## Largest residuals")
        report_lines.append("")
        cols = [c for c in ["period_id", "period_start", "y_true", "y_pred", "residual", "abs_residual"] if c in worst.columns]
        report_lines.append(df_to_markdown_table(worst[cols], max_rows=10))
        report_lines.append("")

    (out_dir / "inspection_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "space": space,
        "target": target,
        "kind": summary.get("kind"),
        "best_model": summary.get("best_model"),
        **{f"test_{k}": v for k, v in summary.get("test_metrics", {}).items()},
        "n_test": int(len(pred_df)),
        "mean_abs_residual": float(pred_df["abs_residual"].mean()) if len(pred_df) else np.nan,
        "feature_driver_type": fi_kind,
    }


def inspect_one_target(project_root: Path, model_root: Path, reports_root: Path, out_root: Path, space: str, target: str) -> Optional[Dict]:
    target_report_dir = reports_root / space / target
    target_model_dir = model_root / space / target
    out_dir = out_root / space / target
    ensure_dir(out_dir)

    summary_path = target_report_dir / "test_summary.json"
    pred_path = target_report_dir / "test_predictions.csv"
    meta_path = target_report_dir / "meta.json"

    if not summary_path.exists() or not pred_path.exists() or not meta_path.exists():
        return None

    summary = load_json(summary_path)
    meta = load_json(meta_path)
    pred_df = pd.read_csv(pred_path, low_memory=False)

    best_model = summary["best_model"]
    model_path = target_model_dir / f"{best_model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    pipe = joblib.load(model_path)

    transition_csv = project_root / "training" / "regime_transition" / f"{space}_transition_matrix.csv"
    if not transition_csv.exists():
        raise FileNotFoundError(f"Missing transition matrix: {transition_csv}")
    df = pd.read_csv(transition_csv, low_memory=False)

    kind = summary.get("kind", "")
    if "classification" in str(kind):
        result = inspect_classification_target(space, target, df, meta, summary, pred_df, pipe, out_dir)
    else:
        result = inspect_regression_target(space, target, df, meta, summary, pred_df, pipe, out_dir)

    comparison_path = target_report_dir / "model_comparison.csv"
    if comparison_path.exists():
        pd.read_csv(comparison_path).to_csv(out_dir / "model_comparison.csv", index=False)

    return result


def build_overall_markdown(rows: pd.DataFrame) -> str:
    lines = ["# Regime Transition Inspection Summary", ""]
    if rows.empty:
        lines.append("No targets were successfully inspected.")
        return "\n".join(lines)

    lines.append("## Target summary")
    lines.append("")
    lines.append(df_to_markdown_table(rows))
    lines.append("")
    lines.append("## Initial read")
    lines.append("")
    for _, row in rows.iterrows():
        bits = [f"{row['space']}/{row['target']} ({row['best_model']})"]
        if "test_macro_f1" in row and pd.notna(row["test_macro_f1"]):
            bits.append(f"macro_f1={row['test_macro_f1']:.3f}")
        if "test_balanced_accuracy" in row and pd.notna(row["test_balanced_accuracy"]):
            bits.append(f"balanced_acc={row['test_balanced_accuracy']:.3f}")
        if "test_r2" in row and pd.notna(row["test_r2"]):
            bits.append(f"R2={row['test_r2']:.3f}")
        lines.append("- " + ", ".join(bits))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inspection reports for selected regime transition targets.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--model-root", default="models/regime_transition", help="Relative model root.")
    parser.add_argument("--reports-root", default="reports/backtests/regime_transition", help="Relative reports root.")
    parser.add_argument("--out-dir", default="reports/inspection/regime_transition", help="Relative output dir.")
    parser.add_argument(
        "--target-pairs",
        nargs="*",
        default=DEFAULT_TARGET_PAIRS,
        help="Items in 'space:target' form.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    model_root = project_root / args.model_root
    reports_root = project_root / args.reports_root
    out_root = project_root / args.out_dir
    ensure_dir(out_root)

    rows = []
    for space, target in parse_target_pairs(args.target_pairs):
        log(f"Inspecting {space}/{target} ...")
        res = inspect_one_target(project_root, model_root, reports_root, out_root, space, target)
        if res is None:
            log(f"Skipped {space}/{target} because required report files were missing.")
        else:
            rows.append(res)

    overall_df = pd.DataFrame(rows)
    if not overall_df.empty:
        overall_df.to_csv(out_root / "inspection_overall_summary.csv", index=False)
        save_json(out_root / "inspection_overall_summary.json", {
            "rows": len(overall_df),
            "targets": rows,
        })

    (out_root / "inspection_overall_summary.md").write_text(
        build_overall_markdown(overall_df),
        encoding="utf-8",
    )

    log("Done.")
    log(f"Wrote inspection reports to: {out_root}")


if __name__ == "__main__":
    main()
