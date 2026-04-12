from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

PRIMARY_TARGETS = [
    "y_next_budget_breach_flag",
    "y_next_restaurant_heavy_flag",
    "y_next_weight_loss_flag",
    "y_next_weight_gain_flag",
    "y_next_high_meal_frequency_flag",
    "y_next_weight_delta_lb",
    "y_next_restaurant_meal_fraction_week",
    "y_next_restaurant_meal_fraction_weekend",
    "y_next_budget_minus_logged_food_kcal_week",
    "y_next_budget_minus_logged_food_kcal_weekend",
]

CLASSIFICATION_KINDS = {"classification", "binary_classification"}
REGRESSION_KINDS = {"regression"}


def log(msg: str) -> None:
    print(f"[regime-train] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_one_hot_encoder() -> OneHotEncoder:
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "period_start" in out.columns:
        t = pd.to_datetime(out["period_start"], errors="coerce")
        out["period_year"] = t.dt.year.astype("float")
        out["period_month"] = t.dt.month.astype("float")
        out["period_quarter"] = t.dt.quarter.astype("float")
        out["period_weekofyear"] = t.dt.isocalendar().week.astype("float")
        out["period_dayofyear"] = t.dt.dayofyear.astype("float")
        day = out["period_dayofyear"]
        angle = 2.0 * math.pi * ((day.fillna(1.0) - 1.0) / 365.25)
        out["period_doy_sin"] = np.sin(angle)
        out["period_doy_cos"] = np.cos(angle)
        if t.notna().any():
            origin = t.min()
            out["period_days_since_start"] = (t - origin).dt.days.astype("float")
        out = out.drop(columns=["period_start"])
    return out


def prepare_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    exclude_cols = []
    for col in df.columns:
        if col.startswith("y_"):
            exclude_cols.append(col)
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

    # Cast pandas nullable booleans to float for preprocessing compatibility.
    for col in x.columns:
        if str(x[col].dtype) == "boolean":
            x[col] = x[col].astype("float")
    return x, exclude_cols


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def safe_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        n_num = 0
        try:
            sample = preprocessor.transform(pd.DataFrame([{}]))
            return [f"feature_{i:04d}" for i in range(sample.shape[1])]
        except Exception:
            return []


def classification_models() -> Dict[str, object]:
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def regression_models() -> Dict[str, object]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "rf": RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "et": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


@dataclass
class TargetSpec:
    name: str
    kind: str
    source_current: Optional[str]
    source_next: Optional[str]
    description: str
    notes: str = ""


def load_target_specs(spec_json: Dict, space: str, target_group: str) -> List[TargetSpec]:
    targets = [TargetSpec(**t) for t in spec_json["spaces"][space]["targets"]]
    if target_group == "primary":
        targets = [t for t in targets if t.name in PRIMARY_TARGETS]
    return targets


def encode_class_target(y: pd.Series, min_class_count: int = 3) -> Tuple[np.ndarray, Dict[str, int], List[str], pd.Series]:
    raw = y.astype("object")
    raw = raw.where(raw.notna(), other=None)

    vc = pd.Series(raw).dropna().astype(str).value_counts()
    common = set(vc[vc >= min_class_count].index.tolist())

    collapsed = []
    for x in raw:
        if x is None:
            collapsed.append(None)
        else:
            sx = str(x)
            collapsed.append(sx if sx in common else "OTHER")

    text = pd.Series(collapsed, index=y.index, dtype="object")
    labels = sorted(text.dropna().unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    enc = np.full(len(text), -100, dtype=np.int64)
    for i, val in enumerate(text):
        if val is not None:
            enc[i] = label_to_idx[val]

    return enc, label_to_idx, labels, text


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2 and len(np.unique(y_true)) >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            pass
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def choose_best_classification(rows: List[Dict]) -> str:
    # Prefer macro_f1, break ties with balanced accuracy then accuracy.
    rows2 = [r for r in rows if "val_macro_f1" in r]
    rows2.sort(key=lambda r: (r["val_macro_f1"], r.get("val_balanced_accuracy", -1), r.get("val_accuracy", -1)), reverse=True)
    return rows2[0]["model_name"]


def choose_best_regression(rows: List[Dict]) -> str:
    # Prefer lowest val MAE, then highest val R².
    rows2 = [r for r in rows if "val_mae" in r]
    rows2.sort(key=lambda r: (r["val_mae"], -r.get("val_r2", -9999)))
    return rows2[0]["model_name"]


def fit_and_evaluate_classification(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    rows = []
    fitted = {}
    preprocessor = build_preprocessor(x_train)

    for name, model in classification_models().items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_val)
        prob = pipe.predict_proba(x_val) if hasattr(pipe, "predict_proba") else None
        m = classification_metrics(y_val, pred, prob)
        row = {
            "model_name": name,
            "val_accuracy": m["accuracy"],
            "val_balanced_accuracy": m["balanced_accuracy"],
            "val_macro_f1": m["macro_f1"],
        }
        if "roc_auc" in m:
            row["val_roc_auc"] = m["roc_auc"]
        rows.append(row)
        fitted[name] = pipe

    return pd.DataFrame(rows), fitted


def fit_and_evaluate_regression(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    rows = []
    fitted = {}
    preprocessor = build_preprocessor(x_train)

    for name, model in regression_models().items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_val)
        m = regression_metrics(y_val, pred)
        rows.append({
            "model_name": name,
            "val_mae": m["mae"],
            "val_rmse": m["rmse"],
            "val_r2": m["r2"],
        })
        fitted[name] = pipe

    return pd.DataFrame(rows), fitted


def extract_tree_feature_importance(pipe: Pipeline, top_k: int = 50) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocessor"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    feature_names = get_feature_names(preprocessor)
    importances = getattr(model, "feature_importances_")
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i:04d}" for i in range(len(importances))]
    idx = np.argsort(importances)[::-1][:top_k]
    rows = []
    for rank, i in enumerate(idx, start=1):
        rows.append({
            "rank": rank,
            "feature": feature_names[i],
            "importance": float(importances[i]),
        })
    return pd.DataFrame(rows)


def split_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["split_suggested"] == "train"].copy()
    val = df[df["split_suggested"] == "val"].copy()
    test = df[df["split_suggested"] == "test"].copy()
    return train, val, test


def train_one_target(
    df: pd.DataFrame,
    space: str,
    target: TargetSpec,
    project_root: Path,
    model_root: Path,
    reports_root: Path,
    fi_root: Path,
) -> Optional[Dict]:
    train_df, val_df, test_df = split_frame(df)
    x_all, dropped_cols = prepare_feature_frame(df)
    x_train = x_all.loc[train_df.index]
    x_val = x_all.loc[val_df.index]
    x_test = x_all.loc[test_df.index]
    x_trainval = x_all.loc[df[df["split_suggested"].isin(["train", "val"])].index]

    model_dir = model_root / space / target.name
    report_dir = reports_root / space / target.name
    fi_dir = fi_root / space / target.name
    ensure_dir(model_dir)
    ensure_dir(report_dir)
    ensure_dir(fi_dir)

    meta = {
        "space": space,
        "target_name": target.name,
        "target_kind": target.kind,
        "description": target.description,
        "notes": target.notes,
        "dropped_feature_cols": dropped_cols,
        "feature_columns_used": list(x_all.columns),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }

    if target.kind in CLASSIFICATION_KINDS:
        enc_all, label_to_idx, labels, text_series = encode_class_target(df[target.name], min_class_count=3)
        train_mask = enc_all[train_df.index] >= 0
        val_mask = enc_all[val_df.index] >= 0
        test_mask = enc_all[test_df.index] >= 0

        if train_mask.sum() < 8 or len(np.unique(enc_all[train_df.index][train_mask])) < 2:
            meta["status"] = "skipped_insufficient_train_classes"
            save_json(report_dir / "meta.json", meta)
            return None
        if val_mask.sum() < 2:
            meta["status"] = "skipped_insufficient_val_rows"
            save_json(report_dir / "meta.json", meta)
            return None
        if test_mask.sum() < 2:
            meta["status"] = "skipped_insufficient_test_rows"
            save_json(report_dir / "meta.json", meta)
            return None

        comparison_df, fitted = fit_and_evaluate_classification(
            x_train.loc[train_mask],
            enc_all[train_df.index][train_mask],
            x_val.loc[val_mask],
            enc_all[val_df.index][val_mask],
        )
        best_name = choose_best_classification(comparison_df.to_dict(orient="records"))

        # Refit best on train+val
        trainval_idx = df[df["split_suggested"].isin(["train", "val"])].index
        trainval_mask_full = enc_all[trainval_idx] >= 0
        x_trainval_used = x_trainval.loc[trainval_mask_full]
        y_trainval_used = enc_all[trainval_idx][trainval_mask_full]

        best_model = classification_models()[best_name]
        best_pipe = Pipeline([
            ("preprocessor", build_preprocessor(x_trainval_used)),
            ("model", best_model),
        ])
        best_pipe.fit(x_trainval_used, y_trainval_used)

        x_test_used = x_test.loc[test_mask]
        y_test_used = enc_all[test_df.index][test_mask]
        pred = best_pipe.predict(x_test_used)
        prob = best_pipe.predict_proba(x_test_used) if hasattr(best_pipe, "predict_proba") else None
        test_m = classification_metrics(y_test_used, pred, prob)

        pred_text = [labels[i] for i in pred]
        true_text = [labels[i] for i in y_test_used]

        pred_df = pd.DataFrame({
            "space": space,
            "target": target.name,
            "period_id": df.loc[test_df.index[test_mask], "period_id"].astype(str).tolist(),
            "period_start": df.loc[test_df.index[test_mask], "period_start"].astype(str).tolist(),
            "y_true_idx": y_test_used,
            "y_pred_idx": pred,
            "y_true_label": true_text,
            "y_pred_label": pred_text,
        })
        pred_df.to_csv(report_dir / "test_predictions.csv", index=False)

        comparison_df["selected_best"] = comparison_df["model_name"] == best_name
        comparison_df.to_csv(report_dir / "model_comparison.csv", index=False)

        test_summary = {
            "target": target.name,
            "kind": target.kind,
            "best_model": best_name,
            "labels": labels,
            "test_metrics": test_m,
        }
        save_json(report_dir / "test_summary.json", test_summary)
        save_json(report_dir / "meta.json", meta)
        joblib.dump(best_pipe, model_dir / f"{best_name}.joblib")

        fi_df = extract_tree_feature_importance(best_pipe, top_k=50)
        if not fi_df.empty:
            fi_df.to_csv(fi_dir / "top_features.csv", index=False)

        result = {
            "space": space,
            "target": target.name,
            "kind": target.kind,
            "best_model": best_name,
            **{f"test_{k}": v for k, v in test_m.items()},
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
        return result

    elif target.kind in REGRESSION_KINDS:
        y_all = pd.to_numeric(df[target.name], errors="coerce").astype(float).to_numpy()
        train_mask = np.isfinite(y_all[train_df.index])
        val_mask = np.isfinite(y_all[val_df.index])
        test_mask = np.isfinite(y_all[test_df.index])

        if train_mask.sum() < 10:
            meta["status"] = "skipped_insufficient_train_rows"
            save_json(report_dir / "meta.json", meta)
            return None
        if val_mask.sum() < 3:
            meta["status"] = "skipped_insufficient_val_rows"
            save_json(report_dir / "meta.json", meta)
            return None
        if test_mask.sum() < 3:
            meta["status"] = "skipped_insufficient_test_rows"
            save_json(report_dir / "meta.json", meta)
            return None

        comparison_df, fitted = fit_and_evaluate_regression(
            x_train.loc[train_mask],
            y_all[train_df.index][train_mask],
            x_val.loc[val_mask],
            y_all[val_df.index][val_mask],
        )
        best_name = choose_best_regression(comparison_df.to_dict(orient="records"))

        trainval_idx = df[df["split_suggested"].isin(["train", "val"])].index
        trainval_mask_full = np.isfinite(y_all[trainval_idx])
        x_trainval_used = x_trainval.loc[trainval_mask_full]
        y_trainval_used = y_all[trainval_idx][trainval_mask_full]

        best_model = regression_models()[best_name]
        best_pipe = Pipeline([
            ("preprocessor", build_preprocessor(x_trainval_used)),
            ("model", best_model),
        ])
        best_pipe.fit(x_trainval_used, y_trainval_used)

        x_test_used = x_test.loc[test_mask]
        y_test_used = y_all[test_df.index][test_mask]
        pred = best_pipe.predict(x_test_used)
        test_m = regression_metrics(y_test_used, pred)

        pred_df = pd.DataFrame({
            "space": space,
            "target": target.name,
            "period_id": df.loc[test_df.index[test_mask], "period_id"].astype(str).tolist(),
            "period_start": df.loc[test_df.index[test_mask], "period_start"].astype(str).tolist(),
            "y_true": y_test_used,
            "y_pred": pred,
            "residual": y_test_used - pred,
        })
        pred_df.to_csv(report_dir / "test_predictions.csv", index=False)

        comparison_df["selected_best"] = comparison_df["model_name"] == best_name
        comparison_df.to_csv(report_dir / "model_comparison.csv", index=False)

        test_summary = {
            "target": target.name,
            "kind": target.kind,
            "best_model": best_name,
            "test_metrics": test_m,
        }
        save_json(report_dir / "test_summary.json", test_summary)
        save_json(report_dir / "meta.json", meta)
        joblib.dump(best_pipe, model_dir / f"{best_name}.joblib")

        fi_df = extract_tree_feature_importance(best_pipe, top_k=50)
        if not fi_df.empty:
            fi_df.to_csv(fi_dir / "top_features.csv", index=False)

        result = {
            "space": space,
            "target": target.name,
            "kind": target.kind,
            "best_model": best_name,
            **{f"test_{k}": v for k, v in test_m.items()},
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
        return result

    else:
        meta["status"] = "skipped_unknown_target_kind"
        save_json(report_dir / "meta.json", meta)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forward-looking regime transition baselines.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--spaces", nargs="+", default=["weeks", "weekends"], choices=["weeks", "weekends"])
    parser.add_argument("--target-group", choices=["primary", "all"], default="primary")
    parser.add_argument("--transition-dir", default="training/regime_transition", help="Relative path to transition dataset dir.")
    parser.add_argument("--target-spec-json", default="training/regime_transition/regime_transition_target_spec.json", help="Relative path to combined target spec.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    transition_dir = project_root / args.transition_dir
    target_spec_path = project_root / args.target_spec_json

    if not target_spec_path.exists():
        raise FileNotFoundError(f"Missing target spec JSON: {target_spec_path}")

    spec_json = load_json(target_spec_path)

    model_root = project_root / "models" / "regime_transition"
    reports_root = project_root / "reports" / "backtests" / "regime_transition"
    fi_root = project_root / "reports" / "feature_importance" / "regime_transition"
    ensure_dir(model_root)
    ensure_dir(reports_root)
    ensure_dir(fi_root)

    overall_rows = []

    for space in args.spaces:
        csv_path = transition_dir / f"{space}_transition_matrix.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing transition matrix for {space}: {csv_path}")

        log(f"Loading transition matrix for {space} ...")
        df = pd.read_csv(csv_path, low_memory=False)

        targets = load_target_specs(spec_json, space, args.target_group)
        log(f"Training {len(targets)} targets for {space} ({args.target_group}) ...")

        space_rows = []
        for target in targets:
            res = train_one_target(
                df=df,
                space=space,
                target=target,
                project_root=project_root,
                model_root=model_root,
                reports_root=reports_root,
                fi_root=fi_root,
            )
            if res is not None:
                space_rows.append(res)
                overall_rows.append(res)
                log(f"Finished {space}/{target.name} (best={res['best_model']})")
            else:
                log(f"Skipped {space}/{target.name}")

        if space_rows:
            pd.DataFrame(space_rows).to_csv(reports_root / f"{space}_summary.csv", index=False)

    if overall_rows:
        overall_df = pd.DataFrame(overall_rows)
        overall_df.to_csv(reports_root / "overall_summary.csv", index=False)
        save_json(reports_root / "overall_summary.json", {
            "rows": len(overall_df),
            "spaces": args.spaces,
            "target_group": args.target_group,
        })

    log("Done.")
    log(f"Models written under: {model_root}")
    log(f"Backtest reports written under: {reports_root}")
    log(f"Feature importance written under: {fi_root}")


if __name__ == "__main__":
    main()
