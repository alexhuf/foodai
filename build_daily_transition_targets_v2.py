from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
    print(f"[daily-transition] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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


def find_existing_path(project_root: Path, candidates: Sequence[str]) -> Path:
    for rel in candidates:
        p = project_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find any expected daily feature matrix. Tried: "
        + ", ".join(str(project_root / c) for c in candidates)
    )


def first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def build_nullable_same_flag(current: pd.Series, nxt: pd.Series) -> pd.Series:
    cur_obj = current.astype("object").where(current.notna(), None)
    nxt_obj = nxt.astype("object").where(nxt.notna(), None)
    out = pd.Series(cur_obj == nxt_obj, index=current.index, dtype="boolean")
    missing_mask = current.isna() | nxt.isna()
    out.loc[missing_mask] = pd.NA
    return out


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

    # Derived restaurant meal fraction
    num_col = CANONICAL_SOURCE_MAP["restaurant_fraction_numerator"]
    den_col = CANONICAL_SOURCE_MAP["restaurant_fraction_denominator"]
    if num_col in work.columns and den_col in work.columns:
        num = pd.to_numeric(work[num_col], errors="coerce")
        den = pd.to_numeric(work[den_col], errors="coerce")
        frac = np.where((den > 0) & np.isfinite(den), num / den, np.nan)
        work["restaurant_meal_fraction_day"] = frac
        mapping_used["restaurant_meal_fraction_day"] = f"{num_col}/{den_col}"

    # Categorical canonicals
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

    # Derived weight delta from current -> next day weight
    if "true_weight_lb" in work.columns:
        w = pd.to_numeric(work["true_weight_lb"], errors="coerce")
        work["weight_delta_lb"] = w.shift(-1) - w
        mapping_used["weight_delta_lb"] = "derived_from_true_weight_lb_next_minus_current"

    return work, mapping_used


def build_transition_table(
    df: pd.DataFrame,
    id_col: Optional[str],
    time_col: str,
    cfg: DailyConfig,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.sort_values(time_col).reset_index(drop=True)

    work["period_kind"] = "day"
    work["period_id"] = infer_day_id(work, id_col, time_col)
    work["period_start"] = work[time_col]

    work["next_period_id"] = work["period_id"].shift(-horizon)
    work["next_period_start"] = work["period_start"].shift(-horizon)
    work["transition_horizon"] = int(horizon)
    work["days_to_next_period"] = (work["next_period_start"] - work["period_start"]).dt.total_seconds() / 86400.0
    work["gap_vs_expected_days"] = work["days_to_next_period"] - cfg.expected_gap_days

    numeric_targets = [
        "weight_delta_lb",
        "logged_food_kcal_day",
        "budget_minus_logged_food_kcal_day",
        "meal_event_count_day",
        "restaurant_meal_count_day",
        "restaurant_meal_fraction_day",
        "exercise_calories_day",
        "steps_day",
        "samsung_sleep_duration_hours_day",
        "samsung_sleep_score_day",
    ]
    categorical_targets = [
        "dominant_meal_archetype_day",
        "dominant_cuisine_day",
        "dominant_service_form_day",
        "dominant_prep_profile_day",
        "dominant_principal_protein_day",
        "dominant_principal_starch_day",
        "dominant_energy_density_day",
        "dominant_satiety_style_day",
    ]

    available_numeric = [c for c in numeric_targets if c in work.columns]
    available_categorical = [c for c in categorical_targets if c in work.columns]

    for col in available_numeric:
        work[f"y_next_{col}"] = work[col].shift(-horizon)
        work[f"y_delta_next_{col}"] = work[f"y_next_{col}"] - work[col]

    for col in available_categorical:
        next_col = f"y_next_{col}"
        same_col = f"y_same_{col}"
        work[next_col] = work[col].shift(-horizon)
        work[same_col] = build_nullable_same_flag(work[col], work[next_col])

    if "y_next_weight_delta_lb" in work.columns:
        work["y_next_weight_loss_flag"] = pd.Series(work["y_next_weight_delta_lb"] <= -0.5, dtype="boolean")
        work["y_next_weight_gain_flag"] = pd.Series(work["y_next_weight_delta_lb"] >= 0.5, dtype="boolean")
        missing = work["y_next_weight_delta_lb"].isna()
        work.loc[missing, "y_next_weight_loss_flag"] = pd.NA
        work.loc[missing, "y_next_weight_gain_flag"] = pd.NA

    if "y_next_budget_minus_logged_food_kcal_day" in work.columns:
        work["y_next_budget_breach_flag"] = pd.Series(work["y_next_budget_minus_logged_food_kcal_day"] < 0, dtype="boolean")
        work.loc[work["y_next_budget_minus_logged_food_kcal_day"].isna(), "y_next_budget_breach_flag"] = pd.NA

    if "y_next_restaurant_meal_fraction_day" in work.columns:
        work["y_next_restaurant_heavy_flag"] = pd.Series(work["y_next_restaurant_meal_fraction_day"] >= 0.5, dtype="boolean")
        work.loc[work["y_next_restaurant_meal_fraction_day"].isna(), "y_next_restaurant_heavy_flag"] = pd.NA
    elif "y_next_restaurant_meal_count_day" in work.columns:
        work["y_next_restaurant_heavy_flag"] = pd.Series(work["y_next_restaurant_meal_count_day"] >= 1, dtype="boolean")
        work.loc[work["y_next_restaurant_meal_count_day"].isna(), "y_next_restaurant_heavy_flag"] = pd.NA

    if "y_next_meal_event_count_day" in work.columns:
        work["y_next_high_meal_frequency_flag"] = pd.Series(work["y_next_meal_event_count_day"] >= 4.0, dtype="boolean")
        work.loc[work["y_next_meal_event_count_day"].isna(), "y_next_high_meal_frequency_flag"] = pd.NA

    if "y_next_logged_food_kcal_day" in work.columns:
        work["y_next_high_kcal_flag"] = pd.Series(work["y_next_logged_food_kcal_day"] >= 2500.0, dtype="boolean")
        work.loc[work["y_next_logged_food_kcal_day"].isna(), "y_next_high_kcal_flag"] = pd.NA

    out = work[work["next_period_id"].notna()].copy().reset_index(drop=True)
    out["split_suggested"] = temporal_split_labels(len(out))
    out["is_gap_expected"] = pd.Series(out["gap_vs_expected_days"].abs() <= 0.5, dtype="boolean")
    out.loc[out["gap_vs_expected_days"].isna(), "is_gap_expected"] = pd.NA
    out["period_ordinal"] = np.arange(len(out))

    meta = {
        "detected_id_col": id_col,
        "detected_time_col": time_col,
        "available_numeric_targets": available_numeric,
        "available_categorical_targets": available_categorical,
    }
    return out, meta


def summarize_transition_table(df: pd.DataFrame, meta: Dict, source_csv: Path, canonical_mapping: Dict) -> Dict:
    y_cols = [c for c in df.columns if c.startswith("y_")]
    numeric_y = [c for c in y_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    binary_y = [c for c in y_cols if pd.api.types.is_bool_dtype(df[c])]
    return {
        "space": "days",
        "period_kind": "day",
        "source_csv": str(source_csv),
        "rows": int(len(df)),
        "time_min": None if len(df) == 0 else str(pd.to_datetime(df["period_start"]).min()),
        "time_max": None if len(df) == 0 else str(pd.to_datetime(df["period_start"]).max()),
        "expected_gap_days": 1,
        "pct_gap_expected": None if len(df) == 0 else float(df["is_gap_expected"].astype("float").mean()),
        "target_column_count": len(y_cols),
        "numeric_target_count": len(numeric_y),
        "binary_like_target_count": len(binary_y),
        "split_counts": {k: int(v) for k, v in df["split_suggested"].value_counts(dropna=False).to_dict().items()},
        "canonical_source_mapping": canonical_mapping,
        **meta,
    }


def build_target_spec(df: pd.DataFrame, meta: Dict) -> Dict:
    targets = []

    def maybe_add(name: str, kind: str, source_current: Optional[str], source_next: Optional[str], description: str, notes: str = ""):
        if name in df.columns:
            targets.append({
                "name": name,
                "kind": kind,
                "source_current": source_current,
                "source_next": source_next,
                "description": description,
                "notes": notes,
            })

    for col in meta["available_numeric_targets"]:
        maybe_add(
            name=f"y_next_{col}",
            kind="regression",
            source_current=col,
            source_next=col,
            description=f"Next-day value of `{col}`.",
        )
        maybe_add(
            name=f"y_delta_next_{col}",
            kind="regression",
            source_current=col,
            source_next=col,
            description=f"Change from current `{col}` to next-day `{col}`.",
        )

    for col in meta["available_categorical_targets"]:
        maybe_add(
            name=f"y_next_{col}",
            kind="classification",
            source_current=col,
            source_next=col,
            description=f"Next-day dominant label for `{col}`.",
        )
        maybe_add(
            name=f"y_same_{col}",
            kind="binary_classification",
            source_current=col,
            source_next=col,
            description=f"Whether the next day keeps the same `{col}` label.",
        )

    maybe_add("y_next_weight_loss_flag", "binary_classification", "weight_delta_lb", "weight_delta_lb", "Whether next-day weight delta is <= -0.5 lb.")
    maybe_add("y_next_weight_gain_flag", "binary_classification", "weight_delta_lb", "weight_delta_lb", "Whether next-day weight delta is >= 0.5 lb.")
    maybe_add("y_next_budget_breach_flag", "binary_classification", "budget_minus_logged_food_kcal_day", "budget_minus_logged_food_kcal_day", "Whether next-day budget-minus-food-kcal is negative.")
    maybe_add("y_next_restaurant_heavy_flag", "binary_classification", "restaurant_meal_fraction_day", "restaurant_meal_fraction_day", "Whether next day is restaurant-heavy.")
    maybe_add("y_next_high_meal_frequency_flag", "binary_classification", "meal_event_count_day", "meal_event_count_day", "Whether next day has high meal frequency.")
    maybe_add("y_next_high_kcal_flag", "binary_classification", "logged_food_kcal_day", "logged_food_kcal_day", "Whether next day has high logged kcal.")

    return {
        "space": "days",
        "period_kind": "day",
        "detected_id_col": meta["detected_id_col"],
        "detected_time_col": meta["detected_time_col"],
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward-looking daily transition targets from the daily feature matrix.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--source-csv", default="", help="Optional explicit relative path to daily feature matrix.")
    parser.add_argument("--horizon", type=int, default=1, help="How many days ahead to target. Default: 1.")
    parser.add_argument("--out-dir", default="training/daily_transition", help="Relative output directory.")
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

    log(f"Loading daily feature matrix from: {source_csv}")
    df = pd.read_csv(source_csv, low_memory=False)

    id_col = first_present(df.columns, DAILY_CONFIG.id_col_candidates)
    time_col = first_present(df.columns, DAILY_CONFIG.time_col_candidates)
    if time_col is None:
        raise ValueError(
            "Could not detect a daily time column. "
            f"Tried candidates: {DAILY_CONFIG.time_col_candidates}"
        )

    canonical_df, canonical_mapping = canonicalize_daily_columns(df)
    transition_df, meta = build_transition_table(
        df=canonical_df,
        id_col=id_col,
        time_col=time_col,
        cfg=DAILY_CONFIG,
        horizon=args.horizon,
    )

    out_csv = out_dir / "days_transition_matrix.csv"
    summary_json = out_dir / "days_transition_summary.json"
    target_spec_json = out_dir / "days_transition_target_spec.json"
    preview_csv = out_dir / "days_transition_preview.csv"
    manifest_json = out_dir / "daily_transition_manifest.json"

    transition_df.to_csv(out_csv, index=False)
    transition_df.head(60).to_csv(preview_csv, index=False)

    summary = summarize_transition_table(transition_df, meta, source_csv, canonical_mapping)
    save_json(summary_json, summary)

    target_spec = build_target_spec(transition_df, meta)
    save_json(target_spec_json, target_spec)

    manifest = {
        "project_root": str(project_root),
        "source_csv": str(source_csv),
        "output_csv": str(out_csv),
        "summary_json": str(summary_json),
        "target_spec_json": str(target_spec_json),
        "preview_csv": str(preview_csv),
        "horizon": int(args.horizon),
        "rows": int(len(transition_df)),
        "canonical_source_mapping": canonical_mapping,
        **meta,
    }
    save_json(manifest_json, manifest)

    log("Done.")
    log(f"Wrote transition matrix to: {out_csv}")
    log(f"Wrote manifest to: {manifest_json}")


if __name__ == "__main__":
    main()
