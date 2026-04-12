from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SpaceConfig:
    name: str
    source_csv: str
    id_col: str
    time_col: str
    period_kind: str
    expected_gap_days: int
    key_numeric_targets: List[str]
    key_categorical_targets: List[str]


SPACE_CONFIGS: Dict[str, SpaceConfig] = {
    "weeks": SpaceConfig(
        name="weeks",
        source_csv="training/week_summary_matrix.csv",
        id_col="week_id",
        time_col="week_start",
        period_kind="week",
        expected_gap_days=7,
        key_numeric_targets=[
            "weight_delta_lb",
            "meal_events_per_day_week",
            "restaurant_meal_fraction_week",
            "budget_minus_logged_food_kcal_week",
        ],
        key_categorical_targets=[
            "dominant_meal_archetype_week",
            "dominant_cuisine_week",
            "dominant_service_form_week",
            "dominant_prep_profile_week",
            "dominant_protein_week",
            "dominant_starch_week",
            "dominant_energy_density_week",
            "dominant_satiety_style_week",
        ],
    ),
    "weekends": SpaceConfig(
        name="weekends",
        source_csv="training/weekend_summary_matrix.csv",
        id_col="weekend_id",
        time_col="weekend_start",
        period_kind="weekend",
        expected_gap_days=7,
        key_numeric_targets=[
            "weight_delta_lb",
            "meal_events_per_day_weekend",
            "restaurant_meal_fraction_weekend",
            "budget_minus_logged_food_kcal_weekend",
        ],
        key_categorical_targets=[
            "dominant_meal_archetype_weekend",
            "dominant_cuisine_weekend",
            "dominant_service_form_weekend",
            "dominant_prep_profile_weekend",
            "dominant_protein_weekend",
            "dominant_starch_weekend",
            "dominant_energy_density_weekend",
            "dominant_satiety_style_weekend",
        ],
    ),
}


def log(msg: str) -> None:
    print(f"[regime-transition] {msg}")


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
    if n < 20:
        val_start = max(1, int(n * 0.7))
        test_start = max(val_start + 1, int(n * 0.85))
    labels[val_start:test_start] = "val"
    labels[test_start:] = "test"
    return labels


def build_nullable_same_flag(current: pd.Series, nxt: pd.Series) -> pd.Series:
    """
    Compare two categorical-ish series and return a pandas nullable boolean Series.
    Missing on either side -> <NA>
    Equal -> True
    Not equal -> False
    """
    cur_obj = current.astype("object").where(current.notna(), None)
    nxt_obj = nxt.astype("object").where(nxt.notna(), None)

    out = pd.Series(cur_obj == nxt_obj, index=current.index, dtype="boolean")
    missing_mask = current.isna() | nxt.isna()
    out.loc[missing_mask] = pd.NA
    return out


def build_transition_table(df: pd.DataFrame, cfg: SpaceConfig, horizon: int = 1) -> pd.DataFrame:
    if cfg.id_col not in df.columns:
        raise ValueError(f"Missing required ID column: {cfg.id_col}")
    if cfg.time_col not in df.columns:
        raise ValueError(f"Missing required time column: {cfg.time_col}")

    work = df.copy()
    work[cfg.time_col] = pd.to_datetime(work[cfg.time_col], errors="coerce")
    work = work.sort_values(cfg.time_col).reset_index(drop=True)

    work["period_kind"] = cfg.period_kind
    work["period_id"] = work[cfg.id_col].astype(str)
    work["period_start"] = work[cfg.time_col]

    work["next_period_id"] = work["period_id"].shift(-horizon)
    work["next_period_start"] = work["period_start"].shift(-horizon)
    work["transition_horizon"] = int(horizon)
    work["days_to_next_period"] = (work["next_period_start"] - work["period_start"]).dt.total_seconds() / 86400.0
    work["gap_vs_expected_days"] = work["days_to_next_period"] - cfg.expected_gap_days

    available_numeric = [c for c in cfg.key_numeric_targets if c in work.columns]
    available_categorical = [c for c in cfg.key_categorical_targets if c in work.columns]

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

    restaurant_col = None
    if "y_next_restaurant_meal_fraction_week" in work.columns:
        restaurant_col = "y_next_restaurant_meal_fraction_week"
    elif "y_next_restaurant_meal_fraction_weekend" in work.columns:
        restaurant_col = "y_next_restaurant_meal_fraction_weekend"
    if restaurant_col:
        work["y_next_restaurant_heavy_flag"] = pd.Series(work[restaurant_col] >= 0.50, dtype="boolean")
        work.loc[work[restaurant_col].isna(), "y_next_restaurant_heavy_flag"] = pd.NA

    budget_col = None
    if "y_next_budget_minus_logged_food_kcal_week" in work.columns:
        budget_col = "y_next_budget_minus_logged_food_kcal_week"
    elif "y_next_budget_minus_logged_food_kcal_weekend" in work.columns:
        budget_col = "y_next_budget_minus_logged_food_kcal_weekend"
    if budget_col:
        work["y_next_budget_breach_flag"] = pd.Series(work[budget_col] < 0, dtype="boolean")
        work.loc[work[budget_col].isna(), "y_next_budget_breach_flag"] = pd.NA

    meal_freq_col = None
    if "y_next_meal_events_per_day_week" in work.columns:
        meal_freq_col = "y_next_meal_events_per_day_week"
    elif "y_next_meal_events_per_day_weekend" in work.columns:
        meal_freq_col = "y_next_meal_events_per_day_weekend"
    if meal_freq_col:
        work["y_next_high_meal_frequency_flag"] = pd.Series(work[meal_freq_col] >= 4.0, dtype="boolean")
        work.loc[work[meal_freq_col].isna(), "y_next_high_meal_frequency_flag"] = pd.NA

    out = work[work["next_period_id"].notna()].copy().reset_index(drop=True)
    out["split_suggested"] = temporal_split_labels(len(out))
    out["is_gap_expected"] = pd.Series(out["gap_vs_expected_days"].abs() <= 1.5, dtype="boolean")
    out.loc[out["gap_vs_expected_days"].isna(), "is_gap_expected"] = pd.NA
    out["period_ordinal"] = np.arange(len(out))
    return out


def summarize_transition_table(df: pd.DataFrame, cfg: SpaceConfig) -> Dict:
    y_cols = [c for c in df.columns if c.startswith("y_")]
    numeric_y = [c for c in y_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    binary_y = [c for c in y_cols if pd.api.types.is_bool_dtype(df[c])]
    summary = {
        "space": cfg.name,
        "period_kind": cfg.period_kind,
        "rows": int(len(df)),
        "time_min": None if len(df) == 0 else str(pd.to_datetime(df["period_start"]).min()),
        "time_max": None if len(df) == 0 else str(pd.to_datetime(df["period_start"]).max()),
        "expected_gap_days": cfg.expected_gap_days,
        "pct_gap_expected": None if len(df) == 0 else float(df["is_gap_expected"].astype("float").mean()),
        "target_column_count": len(y_cols),
        "numeric_target_count": len(numeric_y),
        "binary_like_target_count": len(binary_y),
        "split_counts": {k: int(v) for k, v in df["split_suggested"].value_counts(dropna=False).to_dict().items()},
    }
    return summary


def build_target_spec(cfg: SpaceConfig, transition_df: pd.DataFrame) -> Dict:
    targets = []

    def maybe_add(name: str, kind: str, source_current: Optional[str], source_next: Optional[str], description: str, notes: str = ""):
        if name in transition_df.columns:
            targets.append({
                "name": name,
                "kind": kind,
                "source_current": source_current,
                "source_next": source_next,
                "description": description,
                "notes": notes,
            })

    for col in cfg.key_numeric_targets:
        maybe_add(
            name=f"y_next_{col}",
            kind="regression",
            source_current=col,
            source_next=col,
            description=f"Next {cfg.period_kind} value of `{col}`.",
        )
        maybe_add(
            name=f"y_delta_next_{col}",
            kind="regression",
            source_current=col,
            source_next=col,
            description=f"Change from current `{col}` to next {cfg.period_kind} `{col}`.",
        )

    for col in cfg.key_categorical_targets:
        maybe_add(
            name=f"y_next_{col}",
            kind="classification",
            source_current=col,
            source_next=col,
            description=f"Next {cfg.period_kind} dominant label for `{col}`.",
        )
        maybe_add(
            name=f"y_same_{col}",
            kind="binary_classification",
            source_current=col,
            source_next=col,
            description=f"Whether the next {cfg.period_kind} keeps the same `{col}` label.",
        )

    maybe_add(
        name="y_next_weight_loss_flag",
        kind="binary_classification",
        source_current="weight_delta_lb" if "weight_delta_lb" in transition_df.columns else None,
        source_next="weight_delta_lb",
        description="Whether next period weight delta is <= -0.5 lb.",
        notes="Simple domain threshold; useful for ranking and control-oriented models.",
    )
    maybe_add(
        name="y_next_weight_gain_flag",
        kind="binary_classification",
        source_current="weight_delta_lb" if "weight_delta_lb" in transition_df.columns else None,
        source_next="weight_delta_lb",
        description="Whether next period weight delta is >= 0.5 lb.",
        notes="Simple domain threshold; useful for ranking and risk models.",
    )
    maybe_add(
        name="y_next_restaurant_heavy_flag",
        kind="binary_classification",
        source_current=None,
        source_next=None,
        description="Whether next period restaurant meal fraction is >= 0.50.",
    )
    maybe_add(
        name="y_next_budget_breach_flag",
        kind="binary_classification",
        source_current=None,
        source_next=None,
        description="Whether next period budget-minus-food-kcal is negative.",
    )
    maybe_add(
        name="y_next_high_meal_frequency_flag",
        kind="binary_classification",
        source_current=None,
        source_next=None,
        description="Whether next period meal events per day is >= 4.0.",
    )

    return {
        "space": cfg.name,
        "period_kind": cfg.period_kind,
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward-looking week/weekend transition targets for regime modeling.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--spaces", nargs="+", default=["weeks", "weekends"], choices=list(SPACE_CONFIGS.keys()))
    parser.add_argument("--horizon", type=int, default=1, help="How many periods ahead to target. Default: 1.")
    parser.add_argument("--out-dir", default="training/regime_transition", help="Relative output directory.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

    manifest = {
        "project_root": str(project_root),
        "out_dir": str(out_dir),
        "horizon": int(args.horizon),
        "spaces": {},
    }

    global_target_spec = {
        "horizon": int(args.horizon),
        "spaces": {},
    }

    for space_name in args.spaces:
        cfg = SPACE_CONFIGS[space_name]
        source_csv = project_root / cfg.source_csv
        if not source_csv.exists():
            raise FileNotFoundError(f"Missing source CSV for {space_name}: {source_csv}")

        log(f"Building transition targets for {space_name} ...")
        df = pd.read_csv(source_csv, low_memory=False)
        transition_df = build_transition_table(df, cfg, horizon=args.horizon)

        out_csv = out_dir / f"{cfg.name}_transition_matrix.csv"
        summary_json = out_dir / f"{cfg.name}_transition_summary.json"
        target_spec_json = out_dir / f"{cfg.name}_transition_target_spec.json"
        preview_csv = out_dir / f"{cfg.name}_transition_preview.csv"

        transition_df.to_csv(out_csv, index=False)
        transition_df.head(50).to_csv(preview_csv, index=False)

        summary = summarize_transition_table(transition_df, cfg)
        save_json(summary_json, summary)

        target_spec = build_target_spec(cfg, transition_df)
        save_json(target_spec_json, target_spec)

        manifest["spaces"][cfg.name] = {
            "source_csv": str(source_csv),
            "output_csv": str(out_csv),
            "summary_json": str(summary_json),
            "target_spec_json": str(target_spec_json),
            "preview_csv": str(preview_csv),
            "rows": int(len(transition_df)),
        }
        global_target_spec["spaces"][cfg.name] = target_spec

        log(f"Wrote {cfg.name} transition matrix to: {out_csv}")

    save_json(out_dir / "regime_transition_manifest.json", manifest)
    save_json(out_dir / "regime_transition_target_spec.json", global_target_spec)

    log("Done.")
    log(f"Wrote manifest to: {out_dir / 'regime_transition_manifest.json'}")
    log(f"Wrote combined target spec to: {out_dir / 'regime_transition_target_spec.json'}")


if __name__ == "__main__":
    main()
