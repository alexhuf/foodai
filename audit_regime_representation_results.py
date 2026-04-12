from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


DEFAULT_SPACE_CONFIGS = {
    "weeks_structure": {
        "source_csv": "training/week_summary_matrix.csv",
        "manifest_json": "models/retrieval_v3/weeks_structure/manifest.json",
        "preprocessor_joblib": "models/retrieval_v3/weeks_structure/preprocessor.joblib",
        "time_col": "week_start",
        "class_targets": [
            "dominant_meal_archetype_week",
            "dominant_cuisine_week",
        ],
        "reg_targets": [
            "weight_delta_lb",
            "meal_events_per_day_week",
            "restaurant_meal_fraction_week",
            "budget_minus_logged_food_kcal_week",
        ],
        "neutralize_input_cols": [
            "dominant_meal_archetype_week",
            "dominant_cuisine_week",
            "dominant_service_form_week",
            "dominant_prep_profile_week",
            "dominant_protein_week",
            "dominant_starch_week",
            "dominant_energy_density_week",
            "dominant_satiety_style_week",
        ],
    },
    "weekends_structure": {
        "source_csv": "training/weekend_summary_matrix.csv",
        "manifest_json": "models/retrieval_v3/weekends_structure/manifest.json",
        "preprocessor_joblib": "models/retrieval_v3/weekends_structure/preprocessor.joblib",
        "time_col": "weekend_start",
        "class_targets": [
            "dominant_meal_archetype_weekend",
            "dominant_cuisine_weekend",
        ],
        "reg_targets": [
            "weight_delta_lb",
            "meal_events_per_day_weekend",
            "restaurant_meal_fraction_weekend",
            "budget_minus_logged_food_kcal_weekend",
        ],
        "neutralize_input_cols": [
            "dominant_meal_archetype_weekend",
            "dominant_cuisine_weekend",
            "dominant_service_form_weekend",
            "dominant_prep_profile_weekend",
            "dominant_protein_weekend",
            "dominant_starch_weekend",
            "dominant_energy_density_weekend",
            "dominant_satiety_style_weekend",
        ],
    },
}


def log(msg: str) -> None:
    print(f"[audit-regime] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def build_temporal_split_labels(df: pd.DataFrame, time_col: str) -> np.ndarray:
    times = parse_time_col(df, time_col)
    if times.notna().sum() == 0:
        n = len(df)
        labels = np.array(["train"] * n, dtype=object)
        val_start = int(n * 0.8)
        test_start = int(n * 0.9)
        labels[val_start:test_start] = "val"
        labels[test_start:] = "test"
        return labels

    order = pd.DataFrame({"i": np.arange(len(df)), "t": times}).sort_values("t")
    idx = order.loc[order["t"].notna(), "i"].to_numpy()
    n = len(idx)

    labels = np.array(["train"] * len(df), dtype=object)
    if n < 20:
        val_start = max(1, int(n * 0.7))
        test_start = max(val_start + 1, int(n * 0.85))
    else:
        val_start = int(n * 0.8)
        test_start = int(n * 0.9)

    labels[idx[val_start:test_start]] = "val"
    labels[idx[test_start:]] = "test"
    return labels


def build_transform_frame(manifest_cols: List[str], df: pd.DataFrame, neutralize_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in manifest_cols if c in df.columns]
    X = df[cols].copy()
    for c in neutralize_cols:
        if c in X.columns:
            X[c] = np.nan
    return X


@dataclass
class EncodedClassTarget:
    name: str
    values_text: pd.Series
    encoded: np.ndarray
    labels: List[str]


def encode_class_target(df: pd.DataFrame, split_labels: np.ndarray, col: str, min_class_count: int = 5) -> Optional[EncodedClassTarget]:
    if col not in df.columns:
        return None

    raw = df[col].astype("object")
    raw = raw.where(raw.notna(), other=None)
    train_mask = split_labels == "train"

    if raw.dropna().isin([0, 1, True, False, "0", "1", "True", "False"]).all():
        raw = raw.map(lambda x: None if x is None else str(int(bool(x))) if isinstance(x, (bool, np.bool_)) else str(x))

    vc = pd.Series(raw[train_mask]).dropna().astype(str).value_counts()
    common = set(vc[vc >= min_class_count].index.tolist())

    collapsed = []
    for x in raw:
        if x is None:
            collapsed.append(None)
        else:
            sx = str(x)
            collapsed.append(sx if sx in common else "OTHER")

    text = pd.Series(collapsed, index=df.index, dtype="object")
    labels = sorted(text.dropna().unique().tolist())
    if len(labels) < 2:
        return None

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    encoded = np.full(len(text), -100, dtype=np.int64)
    for i, x in enumerate(text):
        if x is not None:
            encoded[i] = label_to_idx[x]

    return EncodedClassTarget(name=col, values_text=text, encoded=encoded, labels=labels)


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def get_feature_names(preprocessor, fallback_n: int) -> List[str]:
    try:
        names = list(preprocessor.get_feature_names_out())
        if len(names) == fallback_n:
            return names
    except Exception:
        pass
    return [f"feature_{i:04d}" for i in range(fallback_n)]


def maybe_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def class_balance_table(text_values: pd.Series, split_labels: np.ndarray, target_name: str) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        mask = split_labels == split
        vc = text_values[mask].fillna("<MISSING>").value_counts(dropna=False)
        denom = max(1, int(mask.sum()))
        for label, count in vc.items():
            rows.append({
                "target": target_name,
                "split": split,
                "label": label,
                "count": int(count),
                "fraction": float(count / denom),
            })
    return pd.DataFrame(rows)


def classification_audit(
    X: np.ndarray,
    feature_names: List[str],
    split_labels: np.ndarray,
    target: EncodedClassTarget,
    learned_metrics: Dict,
) -> Tuple[pd.DataFrame, List[Dict]]:
    rows = []
    top_feature_rows = []

    train_mask = (split_labels == "train") & (target.encoded >= 0)
    val_mask = (split_labels == "val") & (target.encoded >= 0)
    test_mask = (split_labels == "test") & (target.encoded >= 0)

    if train_mask.sum() < 5 or len(np.unique(target.encoded[train_mask])) < 2:
        rows.append({
            "target": target.name,
            "status": "skipped_insufficient_train_classes",
        })
        return pd.DataFrame(rows), top_feature_rows

    x_train = X[train_mask]
    y_train = target.encoded[train_mask]

    model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    train_majority = pd.Series(y_train).value_counts().idxmax()

    for split_name, mask in [("val", val_mask), ("test", test_mask)]:
        if mask.sum() == 0:
            rows.append({
                "target": target.name,
                "split": split_name,
                "status": "no_rows",
            })
            continue

        x_split = X[mask]
        y_split = target.encoded[mask]
        y_pred = model.predict(x_split)
        maj_pred = np.full_like(y_split, train_majority)

        row = {
            "target": target.name,
            "split": split_name,
            "n_rows": int(mask.sum()),
            "n_classes_in_split": int(len(np.unique(y_split))),
            "proxy_accuracy": float(accuracy_score(y_split, y_pred)),
            "proxy_macro_f1": float(f1_score(y_split, y_pred, average="macro", zero_division=0)),
            "majority_accuracy": float(accuracy_score(y_split, maj_pred)),
            "majority_macro_f1": float(f1_score(y_split, maj_pred, average="macro", zero_division=0)),
        }

        learned_cls = learned_metrics.get("classification", {}).get(target.name, {})
        if learned_cls:
            row["repr_accuracy"] = learned_cls.get("accuracy")
            row["repr_macro_f1"] = learned_cls.get("macro_f1")
            row["repr_roc_auc"] = learned_cls.get("roc_auc")

        rows.append(row)

    importances = getattr(model, "feature_importances_", None)
    if importances is not None and len(importances) == len(feature_names):
        top_idx = np.argsort(importances)[::-1][:20]
        for rank, idx in enumerate(top_idx, start=1):
            top_feature_rows.append({
                "target": target.name,
                "rank": rank,
                "feature": feature_names[idx],
                "importance": float(importances[idx]),
            })

    return pd.DataFrame(rows), top_feature_rows


def regression_audit(
    X: np.ndarray,
    feature_names: List[str],
    split_labels: np.ndarray,
    df: pd.DataFrame,
    target_name: str,
    learned_metrics: Dict,
) -> Tuple[pd.DataFrame, List[Dict]]:
    rows = []
    top_feature_rows = []

    if target_name not in df.columns:
        return pd.DataFrame(rows), top_feature_rows

    y = pd.to_numeric(df[target_name], errors="coerce").astype(float).to_numpy()

    train_mask = (split_labels == "train") & np.isfinite(y)
    val_mask = (split_labels == "val") & np.isfinite(y)
    test_mask = (split_labels == "test") & np.isfinite(y)

    if train_mask.sum() < 10:
        rows.append({
            "target": target_name,
            "status": "skipped_insufficient_train_rows",
        })
        return pd.DataFrame(rows), top_feature_rows

    x_train = X[train_mask]
    y_train = y[train_mask]

    model = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    baseline_value = float(np.mean(y_train))

    for split_name, mask in [("val", val_mask), ("test", test_mask)]:
        if mask.sum() == 0:
            rows.append({
                "target": target_name,
                "split": split_name,
                "status": "no_rows",
            })
            continue

        x_split = X[mask]
        y_split = y[mask]
        y_pred = model.predict(x_split)
        y_base = np.full_like(y_split, baseline_value)

        row = {
            "target": target_name,
            "split": split_name,
            "n_rows": int(mask.sum()),
            "proxy_mae": float(mean_absolute_error(y_split, y_pred)),
            "proxy_rmse": float(safe_rmse(y_split, y_pred)),
            "proxy_r2": float(r2_score(y_split, y_pred)),
            "baseline_mae": float(mean_absolute_error(y_split, y_base)),
            "baseline_rmse": float(safe_rmse(y_split, y_base)),
            "baseline_r2": float(r2_score(y_split, y_base)),
        }

        learned_reg = learned_metrics.get("regression", {}).get(target_name, {})
        if learned_reg:
            row["repr_mae"] = learned_reg.get("mae")
            row["repr_rmse"] = learned_reg.get("rmse")
            row["repr_r2"] = learned_reg.get("r2")

        rows.append(row)

    importances = getattr(model, "feature_importances_", None)
    if importances is not None and len(importances) == len(feature_names):
        top_idx = np.argsort(importances)[::-1][:20]
        for rank, idx in enumerate(top_idx, start=1):
            top_feature_rows.append({
                "target": target_name,
                "rank": rank,
                "feature": feature_names[idx],
                "importance": float(importances[idx]),
            })

    return pd.DataFrame(rows), top_feature_rows


def summarize_flags(space: str, class_df: pd.DataFrame, reg_df: pd.DataFrame) -> List[str]:
    flags = []

    if not class_df.empty:
        for _, row in class_df.iterrows():
            if row.get("split") != "test":
                continue
            proxy_acc = row.get("proxy_accuracy")
            maj_acc = row.get("majority_accuracy")
            repr_acc = row.get("repr_accuracy")
            n_classes = row.get("n_classes_in_split")
            target = row.get("target")

            if n_classes == 1:
                flags.append(f"{space}/{target}: test split has only one class.")
            if isinstance(proxy_acc, float) and isinstance(maj_acc, float):
                if proxy_acc >= 0.95 and proxy_acc > maj_acc + 0.10:
                    flags.append(f"{space}/{target}: simple proxy model is near-perfect on test ({proxy_acc:.3f}), suggesting residual determinism or an easy target.")
            if isinstance(repr_acc, float) and isinstance(proxy_acc, float):
                if repr_acc >= 0.99 and proxy_acc < 0.80:
                    flags.append(f"{space}/{target}: representation model is perfect but simple proxy is not; inspect split composition or remaining leakage.")
                if repr_acc >= 0.99 and proxy_acc >= 0.95:
                    flags.append(f"{space}/{target}: both representation and simple proxy are near-perfect; target may still be too easy.")
    if not reg_df.empty:
        for _, row in reg_df.iterrows():
            if row.get("split") != "test":
                continue
            target = row.get("target")
            proxy_r2 = row.get("proxy_r2")
            repr_r2 = row.get("repr_r2")
            base_r2 = row.get("baseline_r2")
            if isinstance(proxy_r2, float) and proxy_r2 < 0:
                flags.append(f"{space}/{target}: proxy regressor still has negative test R² ({proxy_r2:.3f}).")
            if isinstance(repr_r2, float) and repr_r2 < 0:
                flags.append(f"{space}/{target}: representation regressor still has negative test R² ({repr_r2:.3f}).")
            if isinstance(proxy_r2, float) and isinstance(base_r2, float) and proxy_r2 <= base_r2 + 0.02:
                flags.append(f"{space}/{target}: proxy regressor barely beats or does not beat the mean baseline.")
    return flags


def audit_space(project_root: Path, space: str, model_root: str, out_dir: Path) -> Dict:
    default_cfg = DEFAULT_SPACE_CONFIGS[space]
    model_dir = project_root / model_root / space
    cfg_path = model_dir / "config.json"
    metrics_path = model_dir / "test_metrics.json"

    cfg = load_json(cfg_path) if cfg_path.exists() else {}
    learned_metrics = load_json(metrics_path) if metrics_path.exists() else {}

    source_csv = project_root / cfg.get("source_csv", default_cfg["source_csv"])
    manifest_json = project_root / cfg.get("manifest_json", default_cfg["manifest_json"])
    preprocessor_joblib = project_root / cfg.get("preprocessor_joblib", default_cfg["preprocessor_joblib"])

    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {source_csv}")
    if not manifest_json.exists():
        raise FileNotFoundError(f"Missing manifest JSON: {manifest_json}")
    if not preprocessor_joblib.exists():
        raise FileNotFoundError(f"Missing preprocessor: {preprocessor_joblib}")

    df = pd.read_csv(source_csv, low_memory=False)
    split_labels = build_temporal_split_labels(df, time_col=default_cfg["time_col"])

    manifest = load_json(manifest_json)
    neutralized_cols = cfg.get("neutralized_input_cols", default_cfg["neutralize_input_cols"])
    X_raw = build_transform_frame(manifest["feature_columns"], df, neutralized_cols)

    preprocessor = joblib.load(preprocessor_joblib)
    X = maybe_dense(preprocessor.transform(X_raw)).astype(np.float32)
    feature_names = get_feature_names(preprocessor, X.shape[1])

    space_out = out_dir / space
    ensure_dir(space_out)

    split_summary = pd.DataFrame({
        "split": ["train", "val", "test"],
        "n_rows": [
            int((split_labels == "train").sum()),
            int((split_labels == "val").sum()),
            int((split_labels == "test").sum()),
        ],
    })
    split_summary.to_csv(space_out / "split_summary.csv", index=False)

    class_balance_frames = []
    class_audit_frames = []
    class_top_features = []

    for target_name in default_cfg["class_targets"]:
        enc = encode_class_target(df, split_labels, target_name, min_class_count=5)
        if enc is None:
            continue
        class_balance_frames.append(class_balance_table(enc.values_text, split_labels, target_name))
        audit_df, top_feats = classification_audit(X, feature_names, split_labels, enc, learned_metrics)
        if not audit_df.empty:
            class_audit_frames.append(audit_df)
        class_top_features.extend(top_feats)

    reg_audit_frames = []
    reg_top_features = []
    for target_name in default_cfg["reg_targets"]:
        audit_df, top_feats = regression_audit(X, feature_names, split_labels, df, target_name, learned_metrics)
        if not audit_df.empty:
            reg_audit_frames.append(audit_df)
        reg_top_features.extend(top_feats)

    class_balance_df = pd.concat(class_balance_frames, ignore_index=True) if class_balance_frames else pd.DataFrame()
    class_audit_df = pd.concat(class_audit_frames, ignore_index=True) if class_audit_frames else pd.DataFrame()
    reg_audit_df = pd.concat(reg_audit_frames, ignore_index=True) if reg_audit_frames else pd.DataFrame()

    if not class_balance_df.empty:
        class_balance_df.to_csv(space_out / "class_balance.csv", index=False)
    if not class_audit_df.empty:
        class_audit_df.to_csv(space_out / "classification_audit.csv", index=False)
    if not reg_audit_df.empty:
        reg_audit_df.to_csv(space_out / "regression_audit.csv", index=False)
    if class_top_features:
        pd.DataFrame(class_top_features).to_csv(space_out / "classification_top_features.csv", index=False)
    if reg_top_features:
        pd.DataFrame(reg_top_features).to_csv(space_out / "regression_top_features.csv", index=False)

    flags = summarize_flags(space, class_audit_df, reg_audit_df)

    result = {
        "space": space,
        "source_csv": str(source_csv),
        "model_dir": str(model_dir),
        "neutralized_input_cols": neutralized_cols,
        "split_summary": split_summary.to_dict(orient="records"),
        "flags": flags,
    }
    return result


def build_markdown(summary: Dict) -> str:
    lines = ["# Regime Representation Audit", ""]
    for item in summary["spaces"]:
        lines.append(f"## {item['space']}")
        lines.append("")
        lines.append(f"- Source: `{item['source_csv']}`")
        lines.append(f"- Model dir: `{item['model_dir']}`")
        lines.append(f"- Neutralized columns: {len(item['neutralized_input_cols'])}")
        lines.append("")
        lines.append("### Split sizes")
        for row in item["split_summary"]:
            lines.append(f"- {row['split']}: {row['n_rows']}")
        lines.append("")
        if item["flags"]:
            lines.append("### Flags")
            for flag in item["flags"]:
                lines.append(f"- {flag}")
        else:
            lines.append("### Flags")
            lines.append("- No major automatic flags.")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit regime representation results for split balance, proxy recoverability, and leakage risk.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--model-root", default="models/representation_v3_2_1", help="Relative path to the trained model root to audit.")
    parser.add_argument("--spaces", nargs="+", default=["weeks_structure", "weekends_structure"], choices=list(DEFAULT_SPACE_CONFIGS.keys()))
    parser.add_argument("--out-dir", default="audit/regime_representation", help="Relative output directory for audit artifacts.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

    summary = {
        "project_root": str(project_root),
        "model_root": str(project_root / args.model_root),
        "spaces": [],
    }

    for space in args.spaces:
        log(f"Auditing {space} ...")
        result = audit_space(project_root, space, args.model_root, out_dir)
        summary["spaces"].append(result)

    save_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"Wrote audit artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
