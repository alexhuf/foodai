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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGETS = ["y_next_weight_gain_flag", "y_next_weight_loss_flag"]


@dataclass
class DailyConfig:
    source_csv_candidates: List[str]
    id_col_candidates: List[str]
    time_col_candidates: List[str]
    expected_gap_days: int


DAILY_CONFIG = DailyConfig(
    source_csv_candidates=[
        "training/day_feature_matrix.csv",
        "training/day_summary_matrix.csv",
        "training/daily_summary_matrix.csv",
    ],
    id_col_candidates=[
        "day_id",
        "date",
        "date_local",
        "day",
        "day_date",
        "date_est",
    ],
    time_col_candidates=[
        "date",
        "date_local",
        "day_date",
        "day_start",
        "date_est",
        "datetime_local",
    ],
    expected_gap_days=1,
)


CANONICAL_SOURCE_MAP = {
    "true_weight_lb": "true_weight_lb",
    "logged_food_kcal_day": "noom_food_calories_kcal",
    "budget_minus_logged_food_kcal_day": "budget_minus_noom_food_calories_kcal",
    "meal_event_count_day": "meal_event_count",
    "restaurant_meal_count_day": "restaurant_specific_meal_count",
    "samsung_sleep_duration_hours_day": "samsung_sleep_duration_ms",
    "samsung_sleep_score_day": "samsung_sleep_score",
    "steps_day": "samsung_pedometer_steps",
    "exercise_calories_day": "samsung_exercise_calorie_kcal",
    "restaurant_fraction_numerator": "restaurant_specific_meal_count",
    "restaurant_fraction_denominator": "meal_event_count",
    "dominant_meal_archetype_day": "dominant_meal_archetype",
    "dominant_cuisine_day": "dominant_cuisine",
    "dominant_service_form_day": "dominant_service_form",
    "dominant_prep_profile_day": "dominant_prep_profile",
    "dominant_principal_protein_day": "dominant_principal_protein",
    "dominant_principal_starch_day": "dominant_principal_starch",
    "dominant_energy_density_day": "dominant_energy_density_style",
    "dominant_satiety_style_day": "dominant_satiety_style",
}


def log(msg: str) -> None:
    print(f"[daily-history-score] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body_lines = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body_lines)


def find_existing_path(project_root: Path, candidates: List[str]) -> Path:
    for rel in candidates:
        p = project_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find any expected daily feature matrix. Tried: "
        + ", ".join(str(project_root / c) for c in candidates)
    )


def first_present(columns: List[str], candidates: List[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def temporal_split_labels(n: int) -> np.ndarray:
    labels = np.array(["train"] * n, dtype=object)
    if n == 0:
        return labels
    val_start = int(n * 0.8)
    test_start = int(n * 0.9)
    if n < 30:
        val_start = max(2, int(n * 0.7))
        test_start = max(val_start + 2, int(n * 0.85))
    labels[val_start:test_start] = "val"
    labels[test_start:] = "test"
    return labels


def infer_day_id(df: pd.DataFrame, id_col: Optional[str], time_col: str) -> pd.Series:
    if id_col and id_col in df.columns:
        return df[id_col].astype(str)
    t = pd.to_datetime(df[time_col], errors="coerce")
    return t.dt.strftime("%Y-%m-%d").fillna(pd.Series(df.index, index=df.index).astype(str))


def canonicalize_daily_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    work = df.copy()
    mapping_used = {}

    def maybe_copy(dst: str, src: str, transform=None):
        if src in work.columns:
            work[dst] = transform(work[src]) if transform is not None else work[src]
            mapping_used[dst] = src

    maybe_copy("true_weight_lb", CANONICAL_SOURCE_MAP["true_weight_lb"])
    maybe_copy("logged_food_kcal_day", CANONICAL_SOURCE_MAP["logged_food_kcal_day"])
    maybe_copy("budget_minus_logged_food_kcal_day", CANONICAL_SOURCE_MAP["budget_minus_logged_food_kcal_day"])
    maybe_copy("meal_event_count_day", CANONICAL_SOURCE_MAP["meal_event_count_day"])
    maybe_copy("restaurant_meal_count_day", CANONICAL_SOURCE_MAP["restaurant_meal_count_day"])
    maybe_copy(
        "samsung_sleep_duration_hours_day",
        CANONICAL_SOURCE_MAP["samsung_sleep_duration_hours_day"],
        transform=lambda s: pd.to_numeric(s, errors="coerce") / 3_600_000.0,
    )
    maybe_copy("samsung_sleep_score_day", CANONICAL_SOURCE_MAP["samsung_sleep_score_day"])
    maybe_copy("steps_day", CANONICAL_SOURCE_MAP["steps_day"])
    maybe_copy("exercise_calories_day", CANONICAL_SOURCE_MAP["exercise_calories_day"])

    num_col = CANONICAL_SOURCE_MAP["restaurant_fraction_numerator"]
    den_col = CANONICAL_SOURCE_MAP["restaurant_fraction_denominator"]
    if num_col in work.columns and den_col in work.columns:
        num = pd.to_numeric(work[num_col], errors="coerce")
        den = pd.to_numeric(work[den_col], errors="coerce")
        frac = np.where((den > 0) & np.isfinite(den), num / den, np.nan)
        work["restaurant_meal_fraction_day"] = frac
        mapping_used["restaurant_meal_fraction_day"] = f"{num_col}/{den_col}"

    for dst, src in [
        ("dominant_meal_archetype_day", CANONICAL_SOURCE_MAP["dominant_meal_archetype_day"]),
        ("dominant_cuisine_day", CANONICAL_SOURCE_MAP["dominant_cuisine_day"]),
        ("dominant_service_form_day", CANONICAL_SOURCE_MAP["dominant_service_form_day"]),
        ("dominant_prep_profile_day", CANONICAL_SOURCE_MAP["dominant_prep_profile_day"]),
        ("dominant_principal_protein_day", CANONICAL_SOURCE_MAP["dominant_principal_protein_day"]),
        ("dominant_principal_starch_day", CANONICAL_SOURCE_MAP["dominant_principal_starch_day"]),
        ("dominant_energy_density_day", CANONICAL_SOURCE_MAP["dominant_energy_density_day"]),
        ("dominant_satiety_style_day", CANONICAL_SOURCE_MAP["dominant_satiety_style_day"]),
    ]:
        maybe_copy(dst, src)

    return work, mapping_used


def add_transition_context_to_source(df: pd.DataFrame, id_col: Optional[str], time_col: str) -> pd.DataFrame:
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.sort_values(time_col).reset_index(drop=True)
    work["period_kind"] = "day"
    work["period_id"] = infer_day_id(work, id_col, time_col)
    work["period_start"] = work[time_col]
    work["transition_horizon"] = 1
    work["next_period_id"] = work["period_id"].shift(-1)
    work["next_period_start"] = work["period_start"].shift(-1)
    work["days_to_next_period"] = (work["next_period_start"] - work["period_start"]).dt.total_seconds() / 86400.0
    work["gap_vs_expected_days"] = work["days_to_next_period"] - DAILY_CONFIG.expected_gap_days
    work["is_gap_expected"] = pd.Series(work["gap_vs_expected_days"].abs() <= 0.5, dtype="boolean")
    work.loc[work["gap_vs_expected_days"].isna(), "is_gap_expected"] = pd.NA
    work["period_ordinal"] = np.arange(len(work))
    work["y_next_weight_delta_lb"] = pd.to_numeric(work.get("true_weight_lb"), errors="coerce").shift(-1) - pd.to_numeric(work.get("true_weight_lb"), errors="coerce")
    work["y_next_weight_gain_flag"] = pd.Series(work["y_next_weight_delta_lb"] >= 0.5, dtype="boolean")
    work["y_next_weight_loss_flag"] = pd.Series(work["y_next_weight_delta_lb"] <= -0.5, dtype="boolean")
    missing = work["y_next_weight_delta_lb"].isna()
    work.loc[missing, "y_next_weight_gain_flag"] = pd.NA
    work.loc[missing, "y_next_weight_loss_flag"] = pd.NA
    out = work[work["next_period_id"].notna()].copy().reset_index(drop=True)
    out["split_suggested"] = temporal_split_labels(len(out))
    return out


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
        angle = 2.0 * math.pi * ((out["period_dayofyear"].fillna(1.0) - 1.0) / 365.25)
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
        "day_id",
        "date",
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


def fit_model(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    if model_name == "logreg":
        model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_name == "et":
        model = ExtraTreesClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_name == "dummy_majority":
        model = DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(x_train)),
        ("model", model),
    ])
    pipe.fit(x_train, y_train)
    return pipe


def apply_platt(model, raw_prob: np.ndarray) -> np.ndarray:
    return model.predict_proba(raw_prob.reshape(-1, 1))[:, 1]


def fit_calibrators_on_val(val_prob: np.ndarray, y_val: np.ndarray):
    if len(np.unique(y_val)) < 2:
        return None, None
    platt = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    platt.fit(val_prob.reshape(-1, 1), y_val)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_prob, y_val)
    return platt, isotonic


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "balanced_accuracy") -> float:
    thresholds = np.unique(np.round(np.concatenate([np.linspace(0.05, 0.95, 19), y_prob]), 4))
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        if metric == "macro_f1":
            score = f1_score(y_true, pred, average="macro", zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t


def probability_band(prob: float, threshold: float) -> str:
    t = max(float(threshold), 1e-6)
    low_cut = 0.5 * t
    if prob < low_cut:
        return "low"
    if prob < t:
        return "watch"
    return "high"


def compute_binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    return {
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def summarize_by_band(df: pd.DataFrame, prob_col: str, threshold: float, outcome_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["band"] = tmp[prob_col].apply(lambda p: probability_band(float(p), float(threshold)))
    rows = []
    for band, g in tmp.groupby("band", dropna=False):
        y = g[outcome_col].astype(int).to_numpy()
        rows.append({
            "band": band,
            "rows": int(len(g)),
            "event_rate": float(np.mean(y)) if len(g) else np.nan,
            "prob_mean": float(g[prob_col].mean()) if len(g) else np.nan,
            "prob_median": float(g[prob_col].median()) if len(g) else np.nan,
            "period_start_min": str(g["period_start"].min()) if len(g) else "",
            "period_start_max": str(g["period_start"].max()) if len(g) else "",
        })
    if not rows:
        return pd.DataFrame()
    order = {"low": 0, "watch": 1, "high": 2}
    out = pd.DataFrame(rows)
    out["sort_key"] = out["band"].map(order).fillna(999)
    out = out.sort_values(["sort_key", "band"]).drop(columns=["sort_key"]).reset_index(drop=True)
    return out


def summarize_by_split(df: pd.DataFrame, prob_col: str, threshold: float, outcome_col: str) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("split_suggested", dropna=False):
        y = g[outcome_col].astype(int).to_numpy()
        prob = g[prob_col].astype(float).to_numpy()
        metrics = compute_binary_metrics(y, prob, threshold)
        rows.append({
            "split_suggested": split,
            "rows": int(len(g)),
            "event_rate": float(np.mean(y)) if len(g) else np.nan,
            "prob_mean": float(np.mean(prob)) if len(g) else np.nan,
            **metrics,
        })
    return pd.DataFrame(rows).sort_values("split_suggested").reset_index(drop=True)


def summarize_monthly(df: pd.DataFrame, prob_col: str, threshold: float, outcome_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["period_start"]).dt.to_period("M").astype(str)
    rows = []
    for month, g in tmp.groupby("month", dropna=False):
        y = g[outcome_col].astype(int).to_numpy()
        prob = g[prob_col].astype(float).to_numpy()
        metrics = compute_binary_metrics(y, prob, threshold)
        rows.append({
            "month": month,
            "rows": int(len(g)),
            "event_rate": float(np.mean(y)) if len(g) else np.nan,
            "prob_mean": float(np.mean(prob)) if len(g) else np.nan,
            **metrics,
        })
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_target_report(target: str, summary_json: Dict, channels_df: pd.DataFrame, band_df: pd.DataFrame, split_df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"# Historical score report: {target}")
    lines.append("")
    lines.append("## Score channels")
    lines.append("")
    lines.append(df_to_markdown_table(channels_df))
    lines.append("")
    if not band_df.empty:
        lines.append("## Realized event rate by primary band")
        lines.append("")
        lines.append(df_to_markdown_table(band_df))
        lines.append("")
    if not split_df.empty:
        lines.append("## Realized metrics by split")
        lines.append("")
        lines.append(df_to_markdown_table(split_df))
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    for k, v in summary_json.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def build_overall_report(results_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Daily Weight Direction Historical Batch Score")
    lines.append("")
    lines.append("This report summarizes historical daily scoring across both weight-direction targets.")
    lines.append("")
    lines.append("## Overall summary")
    lines.append("")
    lines.append(df_to_markdown_table(results_df))
    lines.append("")
    return "\n".join(lines)


def score_target_history(target: str, transition_df: pd.DataFrame, transition_x: pd.DataFrame, project_root: Path, out_dir: Path) -> Dict:
    reports_root = project_root / "reports" / "backtests" / "daily_transition" / target
    models_root = project_root / "models" / "daily_transition" / target
    analysis_root = project_root / "reports" / "analysis" / "daily_weight_direction_v2" / target

    trainer_summary = load_json(reports_root / "test_summary.json")
    analysis_summary = load_json(analysis_root / "analysis_summary.json")
    best_model = trainer_summary["best_model"]

    saved_pipe = joblib.load(models_root / f"{best_model}.joblib")
    train_df = transition_df[transition_df["split_suggested"] == "train"].copy()
    val_df = transition_df[transition_df["split_suggested"] == "val"].copy()

    y_train = train_df[target].astype("boolean").astype("float")
    y_val = val_df[target].astype("boolean").astype("float")
    finite_train = np.isfinite(y_train.to_numpy())
    finite_val = np.isfinite(y_val.to_numpy())

    x_train = transition_x.loc[train_df.index].copy().loc[finite_train]
    x_val = transition_x.loc[val_df.index].copy().loc[finite_val]
    y_train_arr = y_train.to_numpy()[finite_train].astype(int)
    y_val_arr = y_val.to_numpy()[finite_val].astype(int)

    clean_pipe = fit_model(best_model, x_train, y_train_arr)
    p_val_raw = clean_pipe.predict_proba(x_val)[:, 1] if hasattr(clean_pipe, "predict_proba") else clean_pipe.predict(x_val).astype(float)

    thr_raw_bal = choose_threshold(y_val_arr, p_val_raw, metric="balanced_accuracy")
    thr_raw_f1 = choose_threshold(y_val_arr, p_val_raw, metric="macro_f1")
    platt_model, isotonic_model = fit_calibrators_on_val(p_val_raw, y_val_arr)
    p_val_platt = apply_platt(platt_model, p_val_raw) if platt_model is not None else None
    p_val_iso = isotonic_model.predict(p_val_raw) if isotonic_model is not None else None
    thr_platt_bal = choose_threshold(y_val_arr, p_val_platt, metric="balanced_accuracy") if p_val_platt is not None else None
    thr_platt_f1 = choose_threshold(y_val_arr, p_val_platt, metric="macro_f1") if p_val_platt is not None else None
    thr_iso_bal = choose_threshold(y_val_arr, p_val_iso, metric="balanced_accuracy") if p_val_iso is not None else None
    thr_iso_f1 = choose_threshold(y_val_arr, p_val_iso, metric="macro_f1") if p_val_iso is not None else None

    valid_mask = transition_df[target].astype("boolean").notna().to_numpy()
    score_df = transition_df.loc[valid_mask, ["period_id", "period_start", "split_suggested", target, "y_next_weight_delta_lb"]].copy()
    score_df = score_df.rename(columns={target: "y_true"})
    x_valid = transition_x.loc[valid_mask].copy()

    p_saved_raw = saved_pipe.predict_proba(x_valid)[:, 1] if hasattr(saved_pipe, "predict_proba") else saved_pipe.predict(x_valid).astype(float)
    p_clean_raw = clean_pipe.predict_proba(x_valid)[:, 1] if hasattr(clean_pipe, "predict_proba") else clean_pipe.predict(x_valid).astype(float)
    p_clean_platt = apply_platt(platt_model, p_clean_raw) if platt_model is not None else np.full(len(score_df), np.nan)
    p_clean_iso = isotonic_model.predict(p_clean_raw) if isotonic_model is not None else np.full(len(score_df), np.nan)

    score_df["p_saved_raw"] = p_saved_raw
    score_df["p_clean_raw"] = p_clean_raw
    score_df["p_clean_platt"] = p_clean_platt
    score_df["p_clean_isotonic"] = p_clean_iso

    score_df["band_saved_raw"] = score_df["p_saved_raw"].apply(lambda p: probability_band(float(p), 0.5))
    score_df["band_clean_raw_bal"] = score_df["p_clean_raw"].apply(lambda p: probability_band(float(p), thr_raw_bal))
    score_df["band_clean_raw_f1"] = score_df["p_clean_raw"].apply(lambda p: probability_band(float(p), thr_raw_f1))
    if np.isfinite(score_df["p_clean_platt"]).any() and thr_platt_bal is not None:
        score_df["band_clean_platt_bal"] = score_df["p_clean_platt"].apply(lambda p: probability_band(float(p), thr_platt_bal))
        score_df["band_clean_platt_f1"] = score_df["p_clean_platt"].apply(lambda p: probability_band(float(p), thr_platt_f1))
    if np.isfinite(score_df["p_clean_isotonic"]).any() and thr_iso_bal is not None:
        score_df["band_clean_isotonic_bal"] = score_df["p_clean_isotonic"].apply(lambda p: probability_band(float(p), thr_iso_bal))
        score_df["band_clean_isotonic_f1"] = score_df["p_clean_isotonic"].apply(lambda p: probability_band(float(p), thr_iso_f1))

    if target == "y_next_weight_gain_flag":
        primary_channel = "p_saved_raw"
        primary_threshold = 0.5
        experimental_channel = "p_clean_isotonic"
        experimental_threshold = thr_iso_f1
    else:
        primary_channel = "p_clean_isotonic" if np.isfinite(score_df["p_clean_isotonic"]).any() else "p_saved_raw"
        primary_threshold = thr_iso_bal if primary_channel == "p_clean_isotonic" else 0.5
        experimental_channel = "p_saved_raw"
        experimental_threshold = 0.5

    score_df["primary_probability"] = score_df[primary_channel]
    score_df["primary_band"] = score_df["primary_probability"].apply(lambda p: probability_band(float(p), primary_threshold))
    if experimental_channel in score_df.columns:
        score_df["experimental_probability"] = score_df[experimental_channel]
    score_df.to_csv(out_dir / "historical_scored_rows.csv", index=False)

    channels_df = pd.DataFrame([
        {"channel": "saved_raw", "threshold_balanced_accuracy": 0.5, "threshold_macro_f1": 0.5},
        {"channel": "clean_raw", "threshold_balanced_accuracy": thr_raw_bal, "threshold_macro_f1": thr_raw_f1},
        {"channel": "clean_platt", "threshold_balanced_accuracy": thr_platt_bal, "threshold_macro_f1": thr_platt_f1},
        {"channel": "clean_isotonic", "threshold_balanced_accuracy": thr_iso_bal, "threshold_macro_f1": thr_iso_f1},
        {"channel": "primary_selected", "threshold_balanced_accuracy": primary_threshold, "threshold_macro_f1": primary_threshold},
    ])
    channels_df.to_csv(out_dir / "score_channels_reference.csv", index=False)

    band_df = summarize_by_band(score_df, "primary_probability", primary_threshold, "y_true")
    band_df.to_csv(out_dir / "primary_band_realization_summary.csv", index=False)

    split_df = summarize_by_split(score_df, "primary_probability", primary_threshold, "y_true")
    split_df.to_csv(out_dir / "primary_split_metrics.csv", index=False)

    monthly_df = summarize_monthly(score_df, "primary_probability", primary_threshold, "y_true")
    monthly_df.to_csv(out_dir / "primary_monthly_metrics.csv", index=False)

    score_df.sort_values("primary_probability", ascending=False).head(25).to_csv(out_dir / "top_high_risk_days.csv", index=False)
    score_df[(score_df["y_true"].astype(int) == 0)].sort_values("primary_probability", ascending=False).head(25).to_csv(out_dir / "top_false_alarm_days.csv", index=False)
    score_df[(score_df["y_true"].astype(int) == 1)].sort_values("primary_probability", ascending=True).head(25).to_csv(out_dir / "top_missed_event_days.csv", index=False)

    summary = {
        "target": target,
        "best_model": best_model,
        "trainer_metrics_reference": trainer_summary["test_metrics"],
        "analysis_v2_reference_target": analysis_summary["target"],
        "rows_scored": int(len(score_df)),
        "primary_channel": primary_channel,
        "primary_threshold": float(primary_threshold),
        "experimental_channel": experimental_channel,
        "experimental_threshold": None if experimental_threshold is None else float(experimental_threshold),
        "primary_event_rate_overall": float(score_df["y_true"].astype(int).mean()) if len(score_df) else np.nan,
        "primary_probability_mean": float(score_df["primary_probability"].mean()) if len(score_df) else np.nan,
        "high_band_rows": int((score_df["primary_band"] == "high").sum()),
        "watch_band_rows": int((score_df["primary_band"] == "watch").sum()),
        "low_band_rows": int((score_df["primary_band"] == "low").sum()),
    }
    save_json(out_dir / "historical_summary.json", summary)

    (out_dir / "historical_report.md").write_text(
        build_target_report(target, summary, channels_df, band_df, split_df),
        encoding="utf-8",
    )

    return {
        "target": target,
        "best_model": best_model,
        "rows_scored": int(len(score_df)),
        "primary_channel": primary_channel,
        "primary_threshold": float(primary_threshold),
        "primary_event_rate_overall": float(score_df["y_true"].astype(int).mean()) if len(score_df) else np.nan,
        "primary_probability_mean": float(score_df["primary_probability"].mean()) if len(score_df) else np.nan,
        "high_band_rows": int((score_df["primary_band"] == "high").sum()),
        "watch_band_rows": int((score_df["primary_band"] == "watch").sum()),
        "low_band_rows": int((score_df["primary_band"] == "low").sum()),
        "high_band_event_rate": float(band_df.loc[band_df["band"] == "high", "event_rate"].iloc[0]) if "high" in set(band_df["band"]) else np.nan,
        "watch_band_event_rate": float(band_df.loc[band_df["band"] == "watch", "event_rate"].iloc[0]) if "watch" in set(band_df["band"]) else np.nan,
        "low_band_event_rate": float(band_df.loc[band_df["band"] == "low", "event_rate"].iloc[0]) if "low" in set(band_df["band"]) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch historical scorer for daily weight-direction targets.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--source-csv", default="", help="Optional explicit relative path to the daily feature matrix.")
    parser.add_argument("--out-dir", default="reports/scoring/daily_weight_direction_history", help="Relative output directory.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

    if args.source_csv:
        source_csv = project_root / args.source_csv
        if not source_csv.exists():
            raise FileNotFoundError(f"Missing explicit source CSV: {source_csv}")
    else:
        source_csv = find_existing_path(project_root, DAILY_CONFIG.source_csv_candidates)

    raw_df = pd.read_csv(source_csv, low_memory=False)
    id_col = first_present(list(raw_df.columns), DAILY_CONFIG.id_col_candidates)
    time_col = first_present(list(raw_df.columns), DAILY_CONFIG.time_col_candidates)
    if time_col is None:
        raise ValueError(f"Could not detect daily time column. Tried: {DAILY_CONFIG.time_col_candidates}")

    canonical_df, canonical_mapping = canonicalize_daily_columns(raw_df)
    transition_like_df = add_transition_context_to_source(canonical_df, id_col=id_col, time_col=time_col)
    transition_x, _ = prepare_feature_frame(transition_like_df)

    overall_rows = []
    for target in TARGETS:
        log(f"Scoring historical series for {target} ...")
        target_out = out_dir / target
        ensure_dir(target_out)
        res = score_target_history(
            target=target,
            transition_df=transition_like_df,
            transition_x=transition_x,
            project_root=project_root,
            out_dir=target_out,
        )
        overall_rows.append(res)

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / "overall_summary.csv", index=False)
    save_json(out_dir / "overall_summary.json", {
        "source_csv": str(source_csv),
        "rows": len(overall_df),
        "targets": TARGETS,
        "canonical_source_mapping": canonical_mapping,
    })
    (out_dir / "overall_report.md").write_text(
        build_overall_report(overall_df),
        encoding="utf-8",
    )

    log("Done.")
    log(f"Wrote historical scoring outputs to: {out_dir}")


if __name__ == "__main__":
    main()
