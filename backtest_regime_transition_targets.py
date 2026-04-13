from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
DEFAULT_TARGET_PAIRS = [
    "weekends:y_next_restaurant_heavy_flag",
    "weeks:y_next_weight_loss_flag",
    "weeks:y_next_weight_gain_flag",
]


def log(msg: str) -> None:
    print(f"[regime-backtest] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_target_pairs(items: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Target pair must be in 'space:target' form, got: {item}")
        space, target = item.split(":", 1)
        pairs.append((space, target))
    return pairs


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
    rows2 = [r for r in rows if "val_macro_f1" in r]
    rows2.sort(key=lambda r: (r["val_macro_f1"], r.get("val_balanced_accuracy", -1), r.get("val_accuracy", -1)), reverse=True)
    return rows2[0]["model_name"]


def choose_best_regression(rows: List[Dict]) -> str:
    rows2 = [r for r in rows if "val_mae" in r]
    rows2.sort(key=lambda r: (r["val_mae"], -r.get("val_r2", -9999)))
    return rows2[0]["model_name"]


def encode_class_target(y: pd.Series, train_index: pd.Index, min_class_count: int = 3) -> Tuple[np.ndarray, Dict[str, int], List[str], pd.Series]:
    raw = y.astype("object")
    raw = raw.where(raw.notna(), other=None)
    vc = pd.Series(raw.loc[train_index]).dropna().astype(str).value_counts()
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


def fit_and_eval_classification(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    preprocessor = build_preprocessor(x_train)
    rows = []
    fitted = {}

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


def fit_and_eval_regression(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    preprocessor = build_preprocessor(x_train)
    rows = []
    fitted = {}

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


def get_feature_names_from_pipe(pipe) -> List[str]:
    try:
        return list(pipe.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        model = pipe.named_steps["model"]
        n = getattr(model, "n_features_in_", None)
        if n is None:
            return []
        return [f"feature_{i:04d}" for i in range(n)]


def extract_feature_drivers(pipe, top_k: int = 20) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    feature_names = get_feature_names_from_pipe(pipe)

    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        if len(feature_names) != len(vals):
            feature_names = [f"feature_{i:04d}" for i in range(len(vals))]
        idx = np.argsort(vals)[::-1][:top_k]
        return pd.DataFrame({
            "rank": np.arange(1, len(idx) + 1),
            "feature": [feature_names[i] for i in idx],
            "score": [float(vals[i]) for i in idx],
            "driver_type": "tree_importance",
        })

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2 and coef.shape[0] == 1:
            vals = coef[0]
            if len(feature_names) != len(vals):
                feature_names = [f"feature_{i:04d}" for i in range(len(vals))]
            idx = np.argsort(np.abs(vals))[::-1][:top_k]
            return pd.DataFrame({
                "rank": np.arange(1, len(idx) + 1),
                "feature": [feature_names[i] for i in idx],
                "score": [float(vals[i]) for i in idx],
                "direction": ["positive" if vals[i] > 0 else "negative" for i in idx],
                "driver_type": "linear_coefficient",
            })

    return pd.DataFrame()


def build_rolling_windows(df: pd.DataFrame, min_train: int, val_size: int, test_size: int, step: int) -> List[Dict]:
    n = len(df)
    windows = []
    train_end = min_train
    fold = 1
    while True:
        val_start = train_end
        val_end = val_start + val_size
        test_start = val_end
        test_end = test_start + test_size
        if test_end > n:
            break
        windows.append({
            "fold": fold,
            "train_idx": list(range(0, train_end)),
            "val_idx": list(range(val_start, val_end)),
            "test_idx": list(range(test_start, test_end)),
        })
        train_end += step
        fold += 1
    return windows


def inspect_target_kind(spec_json: Dict, space: str, target: str) -> Optional[str]:
    for t in spec_json["spaces"][space]["targets"]:
        if t["name"] == target:
            return t["kind"]
    return None


def run_one_classification_fold(
    x_all: pd.DataFrame,
    df: pd.DataFrame,
    target: str,
    window: Dict,
) -> Optional[Tuple[Dict, Pipeline, pd.DataFrame]]:
    train_df = df.iloc[window["train_idx"]].copy()
    val_df = df.iloc[window["val_idx"]].copy()
    test_df = df.iloc[window["test_idx"]].copy()

    enc_all, label_to_idx, labels, text_series = encode_class_target(df[target], train_df.index, min_class_count=3)

    train_mask = enc_all[train_df.index] >= 0
    val_mask = enc_all[val_df.index] >= 0
    test_mask = enc_all[test_df.index] >= 0

    if train_mask.sum() < 8 or len(np.unique(enc_all[train_df.index][train_mask])) < 2:
        return None
    if val_mask.sum() < 2 or test_mask.sum() < 2:
        return None

    x_train = x_all.iloc[train_df.index].loc[train_mask]
    x_val = x_all.iloc[val_df.index].loc[val_mask]
    x_test = x_all.iloc[test_df.index].loc[test_mask]

    y_train = enc_all[train_df.index][train_mask]
    y_val = enc_all[val_df.index][val_mask]
    y_test = enc_all[test_df.index][test_mask]

    comparison_df, fitted = fit_and_eval_classification(x_train, y_train, x_val, y_val)
    best_name = choose_best_classification(comparison_df.to_dict(orient="records"))

    trainval_idx = window["train_idx"] + window["val_idx"]
    trainval_df = df.iloc[trainval_idx].copy()
    trainval_mask = enc_all[trainval_df.index] >= 0
    x_trainval = x_all.iloc[trainval_df.index].loc[trainval_mask]
    y_trainval = enc_all[trainval_df.index][trainval_mask]

    best_pipe = Pipeline([
        ("preprocessor", build_preprocessor(x_trainval)),
        ("model", classification_models()[best_name]),
    ])
    best_pipe.fit(x_trainval, y_trainval)

    pred = best_pipe.predict(x_test)
    prob = best_pipe.predict_proba(x_test) if hasattr(best_pipe, "predict_proba") else None
    m = classification_metrics(y_test, pred, prob)

    pred_df = pd.DataFrame({
        "fold": window["fold"],
        "period_id": test_df.loc[test_mask, "period_id"].astype(str).tolist(),
        "period_start": test_df.loc[test_mask, "period_start"].astype(str).tolist(),
        "y_true_idx": y_test,
        "y_pred_idx": pred,
        "y_true_label": [labels[i] for i in y_test],
        "y_pred_label": [labels[i] for i in pred],
        "is_correct": [labels[t] == labels[p] for t, p in zip(y_test, pred)],
    })
    if prob is not None:
        pred_df["pred_confidence"] = prob.max(axis=1)

    row = {
        "fold": window["fold"],
        "best_model": best_name,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "test_accuracy": m["accuracy"],
        "test_balanced_accuracy": m["balanced_accuracy"],
        "test_macro_f1": m["macro_f1"],
        "n_test_classes": int(len(np.unique(y_test))),
    }
    if "roc_auc" in m:
        row["test_roc_auc"] = m["roc_auc"]

    return row, best_pipe, pred_df


def run_one_regression_fold(
    x_all: pd.DataFrame,
    df: pd.DataFrame,
    target: str,
    window: Dict,
) -> Optional[Tuple[Dict, Pipeline, pd.DataFrame]]:
    train_df = df.iloc[window["train_idx"]].copy()
    val_df = df.iloc[window["val_idx"]].copy()
    test_df = df.iloc[window["test_idx"]].copy()

    y_all = pd.to_numeric(df[target], errors="coerce").astype(float).to_numpy()

    train_mask = np.isfinite(y_all[train_df.index])
    val_mask = np.isfinite(y_all[val_df.index])
    test_mask = np.isfinite(y_all[test_df.index])

    if train_mask.sum() < 10 or val_mask.sum() < 3 or test_mask.sum() < 3:
        return None

    x_train = x_all.iloc[train_df.index].loc[train_mask]
    x_val = x_all.iloc[val_df.index].loc[val_mask]
    x_test = x_all.iloc[test_df.index].loc[test_mask]

    y_train = y_all[train_df.index][train_mask]
    y_val = y_all[val_df.index][val_mask]
    y_test = y_all[test_df.index][test_mask]

    comparison_df, fitted = fit_and_eval_regression(x_train, y_train, x_val, y_val)
    best_name = choose_best_regression(comparison_df.to_dict(orient="records"))

    trainval_idx = window["train_idx"] + window["val_idx"]
    trainval_df = df.iloc[trainval_idx].copy()
    trainval_mask = np.isfinite(y_all[trainval_df.index])
    x_trainval = x_all.iloc[trainval_df.index].loc[trainval_mask]
    y_trainval = y_all[trainval_df.index][trainval_mask]

    best_pipe = Pipeline([
        ("preprocessor", build_preprocessor(x_trainval)),
        ("model", regression_models()[best_name]),
    ])
    best_pipe.fit(x_trainval, y_trainval)

    pred = best_pipe.predict(x_test)
    m = regression_metrics(y_test, pred)

    pred_df = pd.DataFrame({
        "fold": window["fold"],
        "period_id": test_df.loc[test_mask, "period_id"].astype(str).tolist(),
        "period_start": test_df.loc[test_mask, "period_start"].astype(str).tolist(),
        "y_true": y_test,
        "y_pred": pred,
        "residual": y_test - pred,
        "abs_residual": np.abs(y_test - pred),
    })

    row = {
        "fold": window["fold"],
        "best_model": best_name,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "test_mae": m["mae"],
        "test_rmse": m["rmse"],
        "test_r2": m["r2"],
    }

    return row, best_pipe, pred_df


def summarize_folds(df: pd.DataFrame, kind: str) -> Dict:
    out = {
        "n_folds": int(len(df)),
        "models_chosen": df["best_model"].value_counts().to_dict() if "best_model" in df.columns else {},
    }
    if kind == "classification":
        for col in ["test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_roc_auc"]:
            if col in df.columns:
                out[f"{col}_mean"] = float(df[col].mean())
                out[f"{col}_std"] = float(df[col].std(ddof=0))
                out[f"{col}_min"] = float(df[col].min())
                out[f"{col}_max"] = float(df[col].max())
    else:
        for col in ["test_mae", "test_rmse", "test_r2"]:
            if col in df.columns:
                out[f"{col}_mean"] = float(df[col].mean())
                out[f"{col}_std"] = float(df[col].std(ddof=0))
                out[f"{col}_min"] = float(df[col].min())
                out[f"{col}_max"] = float(df[col].max())
    return out


def build_markdown_report(space: str, target: str, kind: str, summary: Dict, fold_df: pd.DataFrame, best_driver_df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"# Rolling Backtest Report: {space} / {target}")
    lines.append("")
    lines.append(f"- target kind: {kind}")
    lines.append(f"- folds: {summary.get('n_folds', 0)}")
    lines.append(f"- models chosen: {summary.get('models_chosen', {})}")
    lines.append("")

    lines.append("## Aggregate metrics")
    for k, v in summary.items():
        if k in {"n_folds", "models_chosen"}:
            continue
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Fold-by-fold results")
    lines.append("")
    lines.append(df_to_markdown_table(fold_df))
    lines.append("")

    if not best_driver_df.empty:
        lines.append("## Feature drivers from best fold")
        lines.append("")
        lines.append(df_to_markdown_table(best_driver_df.head(15)))
        lines.append("")

    return "\n".join(lines)


def df_to_markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows_"
    df2 = df.copy()
    df2.columns = [str(c) for c in df2.columns]
    for col in df2.columns:
        df2[col] = df2[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(df2.columns)
    rows = df2.values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body_lines = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body_lines)


def run_target_backtest(
    project_root: Path,
    transition_dir: Path,
    space: str,
    target: str,
    kind: str,
    out_root: Path,
    min_train: int,
    val_size: int,
    test_size: int,
    step: int,
) -> Optional[Dict]:
    csv_path = transition_dir / f"{space}_transition_matrix.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing transition matrix: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df = df.sort_values("period_start").reset_index(drop=True)

    x_all, dropped_cols = prepare_feature_frame(df)
    windows = build_rolling_windows(df, min_train=min_train, val_size=val_size, test_size=test_size, step=step)

    target_out = out_root / space / target
    ensure_dir(target_out)

    fold_rows = []
    pred_frames = []
    driver_rows = []

    for window in windows:
        if kind == "classification":
            result = run_one_classification_fold(x_all, df, target, window)
        else:
            result = run_one_regression_fold(x_all, df, target, window)

        if result is None:
            continue

        row, best_pipe, pred_df = result
        fold_rows.append(row)
        pred_frames.append(pred_df)

        fi_df = extract_feature_drivers(best_pipe, top_k=20)
        if not fi_df.empty:
            fi_df.insert(0, "fold", window["fold"])
            driver_rows.append(fi_df)

    if not fold_rows:
        save_json(target_out / "backtest_status.json", {
            "status": "skipped_no_valid_folds",
            "space": space,
            "target": target,
            "kind": kind,
            "min_train": min_train,
            "val_size": val_size,
            "test_size": test_size,
            "step": step,
        })
        return None

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(target_out / "fold_metrics.csv", index=False)

    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(target_out / "all_fold_predictions.csv", index=False)

    driver_df = pd.concat(driver_rows, ignore_index=True) if driver_rows else pd.DataFrame()
    if not driver_df.empty:
        driver_df.to_csv(target_out / "feature_drivers_by_fold.csv", index=False)

    summary = summarize_folds(fold_df, kind="classification" if kind == "classification" else "regression")
    summary.update({
        "space": space,
        "target": target,
        "kind": kind,
        "min_train": min_train,
        "val_size": val_size,
        "test_size": test_size,
        "step": step,
        "dropped_feature_cols": dropped_cols,
    })
    save_json(target_out / "backtest_summary.json", summary)

    # Choose representative best fold
    if kind == "classification":
        best_fold_num = fold_df.sort_values(["test_macro_f1", "test_balanced_accuracy", "test_accuracy"], ascending=False).iloc[0]["fold"]
    else:
        best_fold_num = fold_df.sort_values(["test_mae", "test_r2"], ascending=[True, False]).iloc[0]["fold"]

    best_driver_df = driver_df[driver_df["fold"] == best_fold_num].copy() if not driver_df.empty else pd.DataFrame()
    (target_out / "backtest_report.md").write_text(
        build_markdown_report(space, target, kind, summary, fold_df, best_driver_df),
        encoding="utf-8",
    )

    out_row = {
        "space": space,
        "target": target,
        "kind": kind,
        "n_folds": summary["n_folds"],
        "models_chosen": json.dumps(summary["models_chosen"]),
    }
    if kind == "classification":
        for col in ["test_accuracy_mean", "test_balanced_accuracy_mean", "test_macro_f1_mean", "test_roc_auc_mean"]:
            if col in summary:
                out_row[col] = summary[col]
    else:
        for col in ["test_mae_mean", "test_rmse_mean", "test_r2_mean"]:
            if col in summary:
                out_row[col] = summary[col]
    return out_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling backtest selected regime transition targets.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--transition-dir", default="training/regime_transition", help="Relative transition dir.")
    parser.add_argument("--target-spec-json", default="training/regime_transition/regime_transition_target_spec.json", help="Relative target spec path.")
    parser.add_argument("--out-dir", default="reports/backtests/regime_transition_rolling", help="Relative output dir.")
    parser.add_argument("--target-pairs", nargs="*", default=DEFAULT_TARGET_PAIRS, help="Items in 'space:target' form.")
    parser.add_argument("--min-train", type=int, default=36, help="Minimum initial train rows.")
    parser.add_argument("--val-size", type=int, default=8, help="Validation rows per fold.")
    parser.add_argument("--test-size", type=int, default=6, help="Test rows per fold.")
    parser.add_argument("--step", type=int, default=4, help="How many rows to advance between folds.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    transition_dir = project_root / args.transition_dir
    out_root = project_root / args.out_dir
    ensure_dir(out_root)

    spec_json = load_json(project_root / args.target_spec_json)
    rows = []

    for space, target in parse_target_pairs(args.target_pairs):
        kind = inspect_target_kind(spec_json, space, target)
        if kind is None:
            log(f"Skipping {space}/{target}: target not found in spec.")
            continue
        mode = "classification" if "classification" in kind else "regression"
        log(f"Backtesting {space}/{target} ...")
        res = run_target_backtest(
            project_root=project_root,
            transition_dir=transition_dir,
            space=space,
            target=target,
            kind=mode,
            out_root=out_root,
            min_train=args.min_train,
            val_size=args.val_size,
            test_size=args.test_size,
            step=args.step,
        )
        if res is None:
            log(f"Skipped {space}/{target}: no valid folds.")
        else:
            rows.append(res)

    overall_df = pd.DataFrame(rows)
    if not overall_df.empty:
        overall_df.to_csv(out_root / "rolling_backtest_overall_summary.csv", index=False)
        save_json(out_root / "rolling_backtest_overall_summary.json", {
            "rows": len(overall_df),
            "min_train": args.min_train,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "step": args.step,
            "targets": rows,
        })
        (out_root / "rolling_backtest_overall_summary.md").write_text(
            "# Regime Transition Rolling Backtest Summary\n\n" + df_to_markdown_table(overall_df),
            encoding="utf-8",
        )

    log("Done.")
    log(f"Wrote rolling backtest artifacts to: {out_root}")


if __name__ == "__main__":
    main()
