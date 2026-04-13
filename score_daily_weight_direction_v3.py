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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score


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
    print(f"[daily-score] {msg}")


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
    # Critical fix: add split_suggested so clean train/val/test scoring works.
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


def get_feature_names_from_pipe(pipe: Pipeline) -> List[str]:
    pre = pipe.named_steps["preprocessor"]
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        model = pipe.named_steps["model"]
        n = getattr(model, "n_features_in_", None)
        if n is None:
            return []
        return [f"feature_{i:04d}" for i in range(n)]


def classify_feature_group(col: str) -> str:
    c = col.lower()
    weather_keys = [
        "temperature", "apparent_temperature", "precip", "rain", "snow", "snowfall",
        "cloud", "wind", "gust", "pressure", "humidity", "daylight", "sunrise", "sunset",
        "uv", "weather", "is_day", "rain_streak", "freeze", "hot_streak",
    ]
    biology_keys = [
        "samsung", "heart", "hr", "stress", "sleep", "steps", "exercise", "active",
        "resting", "vo2", "oxygen", "resp", "calories_burned", "bmr", "weight",
        "noom_weight", "rest_calorie", "active_calorie",
    ]
    meal_keys = [
        "meal_", "noom_food", "restaurant_", "cuisine", "archetype", "protein", "carb",
        "fat", "fiber", "dessert", "beverage", "snack", "breakfast", "lunch", "dinner",
        "satiety", "indulgence", "comfort_food", "service_form", "prep_profile",
        "distinct_meal", "distinct_cuisines", "restaurant_specific", "food_", "starch",
        "veg", "sodium", "meal_event_count",
    ]
    temporal_keys = [
        "period_", "month", "quarter", "weekofyear", "dayofyear", "doy_", "season",
        "days_since_start", "year", "day_of_week", "is_weekend",
    ]
    if any(k in c for k in weather_keys):
        return "weather_daylight"
    if any(k in c for k in biology_keys):
        return "biology"
    if any(k in c for k in meal_keys):
        return "meals"
    if any(k in c for k in temporal_keys):
        return "temporal"
    return "other"


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


def probability_band(prob: float, threshold: float) -> str:
    t = max(float(threshold), 1e-6)
    low_cut = 0.5 * t
    if prob < low_cut:
        return "low"
    if prob < t:
        return "watch"
    return "high"


def extract_global_drivers(pipe: Pipeline, top_k: int = 20) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    names = get_feature_names_from_pipe(pipe)
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        if len(names) != len(vals):
            names = [f"feature_{i:04d}" for i in range(len(vals))]
        idx = np.argsort(vals)[::-1][:top_k]
        return pd.DataFrame({
            "rank": np.arange(1, len(idx) + 1),
            "feature": [names[i] for i in idx],
            "global_importance": [float(vals[i]) for i in idx],
            "feature_group": [classify_feature_group(names[i]) for i in idx],
        })
    return pd.DataFrame()


def infer_feature_row_from_history(source_row_df: pd.DataFrame, full_transition_df: pd.DataFrame) -> pd.DataFrame:
    x_all, _ = prepare_feature_frame(full_transition_df)
    key = source_row_df["period_id"].iloc[0]
    match = full_transition_df["period_id"].astype(str) == str(key)
    return x_all.loc[match].copy()


def build_local_driver_proxies(
    latest_feature_row: pd.DataFrame,
    history_feature_df: pd.DataFrame,
    global_driver_df: pd.DataFrame,
    lookback_days: int = 30,
    top_k: int = 12,
) -> pd.DataFrame:
    if global_driver_df.empty:
        return pd.DataFrame()

    latest = latest_feature_row.iloc[0]
    history_recent = history_feature_df.tail(lookback_days).copy()
    rows = []

    for _, r in global_driver_df.iterrows():
        feat = str(r["feature"])
        importance = float(r["global_importance"])

        if feat.startswith("num__"):
            base_feat = feat.split("num__", 1)[1]
            if base_feat not in latest_feature_row.columns:
                continue
            series = pd.to_numeric(history_recent[base_feat], errors="coerce")
            cur = pd.to_numeric(pd.Series([latest.get(base_feat)]), errors="coerce").iloc[0]
            mu = float(series.mean()) if series.notna().any() else np.nan
            sd = float(series.std(ddof=0)) if series.notna().any() else np.nan
            if not np.isfinite(cur):
                continue
            z = 0.0 if (not np.isfinite(sd) or sd <= 1e-9) else (cur - mu) / sd
            score_proxy = abs(z) * importance
            rows.append({
                "feature": base_feat,
                "feature_repr": feat,
                "feature_group": classify_feature_group(base_feat),
                "local_signal_type": "numeric_deviation",
                "current_value": float(cur),
                "recent_mean": mu,
                "z_score": float(z),
                "active_flag": "",
                "score_proxy": float(score_proxy),
                "direction_proxy": "above_recent" if z > 0 else "below_recent",
            })

        elif feat.startswith("cat__"):
            body = feat.split("cat__", 1)[1]
            match_col = None
            match_val = None
            for raw_col in latest_feature_row.columns:
                prefix = raw_col + "_"
                if body.startswith(prefix):
                    match_col = raw_col
                    match_val = body[len(prefix):]
                    break
            if match_col is None or match_col not in latest_feature_row.columns:
                continue
            cur_val = str(latest_feature_row.iloc[0][match_col])
            active = cur_val == match_val
            mode_val = None
            if match_col in history_recent.columns and history_recent[match_col].dropna().shape[0] > 0:
                mode_val = str(history_recent[match_col].mode(dropna=True).iloc[0])
            rows.append({
                "feature": match_col,
                "feature_repr": feat,
                "feature_group": classify_feature_group(match_col),
                "local_signal_type": "categorical_active",
                "current_value": cur_val,
                "recent_mean": mode_val,
                "z_score": "",
                "active_flag": bool(active),
                "score_proxy": float(importance if active else 0.0),
                "direction_proxy": "active_match" if active else "inactive",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("score_proxy", ascending=False).head(top_k).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def build_interpretation(target: str, primary_channel: str, primary_prob: float, band: str, top_local: pd.DataFrame) -> str:
    direction = "next-day weight gain" if "gain" in target else "next-day weight loss"
    if band == "high":
        intro = f"{direction} risk looks elevated today."
    elif band == "watch":
        intro = f"{direction} risk is in a watch zone today."
    else:
        intro = f"{direction} risk looks relatively low today."

    channel_text = {
        "saved_raw": "This is based on the saved ET model's raw score.",
        "clean_isotonic": "This is based on the clean train→val→test isotonic-calibrated channel.",
        "clean_raw": "This is based on the clean train→val→test raw model score.",
    }.get(primary_channel, "This is based on the current scoring channel.")

    if top_local.empty:
        return f"{intro} {channel_text}"

    feature_bits = []
    for _, row in top_local.head(4).iterrows():
        feat = str(row["feature"])
        direction_proxy = str(row["direction_proxy"])
        if direction_proxy == "above_recent":
            feature_bits.append(f"`{feat}` is above its recent baseline")
        elif direction_proxy == "below_recent":
            feature_bits.append(f"`{feat}` is below its recent baseline")
        elif direction_proxy == "active_match":
            feature_bits.append(f"`{feat}` is in an active categorical state")
    return f"{intro} {channel_text} Local driver proxies suggest: " + "; ".join(feature_bits) + "." if feature_bits else f"{intro} {channel_text}"


def score_one_target(
    target: str,
    project_root: Path,
    transition_df: pd.DataFrame,
    transition_x: pd.DataFrame,
    latest_feature_row: pd.DataFrame,
    latest_source_row: pd.DataFrame,
    out_dir: Path,
) -> Dict:
    reports_root = project_root / "reports" / "backtests" / "daily_transition" / target
    models_root = project_root / "models" / "daily_transition" / target
    analysis_root = project_root / "reports" / "analysis" / "daily_weight_direction_v2" / target

    trainer_summary = load_json(reports_root / "test_summary.json")
    analysis_summary = load_json(analysis_root / "analysis_summary.json")
    best_model = trainer_summary["best_model"]

    saved_pipe = joblib.load(models_root / f"{best_model}.joblib")
    p_saved_raw = float(saved_pipe.predict_proba(latest_feature_row)[0, 1]) if hasattr(saved_pipe, "predict_proba") else float(saved_pipe.predict(latest_feature_row)[0])

    global_driver_df = extract_global_drivers(saved_pipe, top_k=20)
    local_proxy_df = build_local_driver_proxies(latest_feature_row, transition_x, global_driver_df, lookback_days=30, top_k=12)

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
    p_latest_clean_raw = float(clean_pipe.predict_proba(latest_feature_row)[0, 1]) if hasattr(clean_pipe, "predict_proba") else float(clean_pipe.predict(latest_feature_row)[0])

    thr_raw_bal = choose_threshold(y_val_arr, p_val_raw, metric="balanced_accuracy")
    thr_raw_f1 = choose_threshold(y_val_arr, p_val_raw, metric="macro_f1")

    platt_model, isotonic_model = fit_calibrators_on_val(p_val_raw, y_val_arr)
    p_latest_platt = float(apply_platt(platt_model, np.array([p_latest_clean_raw]))[0]) if platt_model is not None else None
    p_latest_iso = float(isotonic_model.predict(np.array([p_latest_clean_raw]))[0]) if isotonic_model is not None else None

    thr_platt_bal = choose_threshold(y_val_arr, apply_platt(platt_model, p_val_raw), metric="balanced_accuracy") if platt_model is not None else None
    thr_platt_f1 = choose_threshold(y_val_arr, apply_platt(platt_model, p_val_raw), metric="macro_f1") if platt_model is not None else None
    thr_iso_bal = choose_threshold(y_val_arr, isotonic_model.predict(p_val_raw), metric="balanced_accuracy") if isotonic_model is not None else None
    thr_iso_f1 = choose_threshold(y_val_arr, isotonic_model.predict(p_val_raw), metric="macro_f1") if isotonic_model is not None else None

    if target == "y_next_weight_gain_flag":
        primary_channel = "saved_raw"
        primary_prob = p_saved_raw
        primary_threshold = 0.5
        experimental_prob = p_latest_iso
        experimental_threshold = thr_iso_f1
    else:
        primary_channel = "clean_isotonic" if p_latest_iso is not None else "saved_raw"
        primary_prob = p_latest_iso if p_latest_iso is not None else p_saved_raw
        primary_threshold = thr_iso_bal if p_latest_iso is not None else 0.5
        experimental_prob = p_saved_raw
        experimental_threshold = 0.5

    primary_band = probability_band(float(primary_prob), float(primary_threshold if primary_threshold is not None else 0.5))
    interpretation = build_interpretation(target, primary_channel, float(primary_prob), primary_band, local_proxy_df)

    target_out = out_dir / target
    ensure_dir(target_out)
    if not global_driver_df.empty:
        global_driver_df.to_csv(target_out / "global_driver_summary.csv", index=False)
    if not local_proxy_df.empty:
        local_proxy_df.to_csv(target_out / "local_driver_proxies.csv", index=False)

    channel_df = pd.DataFrame([
        {
            "channel": "saved_raw",
            "probability": p_saved_raw,
            "threshold_balanced_accuracy": 0.5,
            "threshold_macro_f1": 0.5,
            "band_balanced_accuracy": probability_band(p_saved_raw, 0.5),
            "band_macro_f1": probability_band(p_saved_raw, 0.5),
        },
        {
            "channel": "clean_raw",
            "probability": p_latest_clean_raw,
            "threshold_balanced_accuracy": thr_raw_bal,
            "threshold_macro_f1": thr_raw_f1,
            "band_balanced_accuracy": probability_band(p_latest_clean_raw, thr_raw_bal),
            "band_macro_f1": probability_band(p_latest_clean_raw, thr_raw_f1),
        },
        {
            "channel": "clean_platt",
            "probability": p_latest_platt,
            "threshold_balanced_accuracy": thr_platt_bal,
            "threshold_macro_f1": thr_platt_f1,
            "band_balanced_accuracy": probability_band(p_latest_platt, thr_platt_bal) if p_latest_platt is not None and thr_platt_bal is not None else None,
            "band_macro_f1": probability_band(p_latest_platt, thr_platt_f1) if p_latest_platt is not None and thr_platt_f1 is not None else None,
        },
        {
            "channel": "clean_isotonic",
            "probability": p_latest_iso,
            "threshold_balanced_accuracy": thr_iso_bal,
            "threshold_macro_f1": thr_iso_f1,
            "band_balanced_accuracy": probability_band(p_latest_iso, thr_iso_bal) if p_latest_iso is not None and thr_iso_bal is not None else None,
            "band_macro_f1": probability_band(p_latest_iso, thr_iso_f1) if p_latest_iso is not None and thr_iso_f1 is not None else None,
        },
    ])
    channel_df.to_csv(target_out / "score_channels.csv", index=False)

    summary = {
        "target": target,
        "best_model": best_model,
        "latest_period_id": str(latest_source_row["period_id"].iloc[0]),
        "latest_period_start": str(latest_source_row["period_start"].iloc[0]),
        "primary_channel": primary_channel,
        "primary_probability": float(primary_prob),
        "primary_threshold": float(primary_threshold if primary_threshold is not None else 0.5),
        "primary_band": primary_band,
        "experimental_probability": None if experimental_prob is None else float(experimental_prob),
        "experimental_threshold": None if experimental_threshold is None else float(experimental_threshold),
        "trainer_metrics_reference": trainer_summary["test_metrics"],
        "analysis_v2_reference": analysis_summary,
        "interpretation": interpretation,
    }
    save_json(target_out / "score_summary.json", summary)

    return {
        "target": target,
        "best_model": best_model,
        "primary_channel": primary_channel,
        "primary_probability": float(primary_prob),
        "primary_threshold": float(primary_threshold if primary_threshold is not None else 0.5),
        "primary_band": primary_band,
        "experimental_probability": None if experimental_prob is None else float(experimental_prob),
    }


def build_overall_report(latest_source_row: pd.DataFrame, results_df: pd.DataFrame, interpretations: Dict[str, str]) -> str:
    lines = []
    lines.append("# Daily Weight Direction Score")
    lines.append("")
    lines.append(f"- scored day: {latest_source_row['period_id'].iloc[0]}")
    lines.append(f"- timestamp column: {latest_source_row['period_start'].iloc[0]}")
    lines.append("")
    lines.append("## Score summary")
    lines.append("")
    lines.append(df_to_markdown_table(results_df))
    lines.append("")
    for target, text in interpretations.items():
        lines.append(f"## {target}")
        lines.append("")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score daily next-weight-gain and next-weight-loss direction for the latest available day.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--source-csv", default="", help="Optional explicit relative path to the daily feature matrix.")
    parser.add_argument("--score-date", default="", help="Optional YYYY-MM-DD date to score instead of the latest day.")
    parser.add_argument("--out-dir", default="reports/scoring/daily_weight_direction", help="Relative output directory.")
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

    if args.score_date:
        score_date = pd.to_datetime(args.score_date).normalize()
        match = pd.to_datetime(transition_like_df["period_start"]).dt.normalize() == score_date
        if match.sum() == 0:
            raise ValueError(f"No daily row found for score date: {args.score_date}")
        latest_source_row = transition_like_df.loc[match].tail(1).copy()
    else:
        latest_source_row = transition_like_df.tail(1).copy()

    latest_feature_row = infer_feature_row_from_history(latest_source_row, transition_like_df)
    latest_source_row[["period_id", "period_start"]].to_csv(out_dir / "scored_day_reference.csv", index=False)

    results = []
    interpretations = {}
    for target in TARGETS:
        log(f"Scoring {target} ...")
        res = score_one_target(
            target=target,
            project_root=project_root,
            transition_df=transition_like_df,
            transition_x=transition_x,
            latest_feature_row=latest_feature_row,
            latest_source_row=latest_source_row,
            out_dir=out_dir,
        )
        results.append(res)
        summary = load_json(out_dir / target / "score_summary.json")
        interpretations[target] = summary["interpretation"]

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "latest_score_summary.csv", index=False)
    save_json(out_dir / "latest_score_summary.json", {
        "source_csv": str(source_csv),
        "scored_period_id": str(latest_source_row["period_id"].iloc[0]),
        "scored_period_start": str(latest_source_row["period_start"].iloc[0]),
        "results": results,
        "canonical_source_mapping": canonical_mapping,
    })
    (out_dir / "latest_score_report.md").write_text(
        build_overall_report(latest_source_row, results_df, interpretations),
        encoding="utf-8",
    )

    log("Done.")
    log(f"Wrote scoring outputs to: {out_dir}")


if __name__ == "__main__":
    main()
