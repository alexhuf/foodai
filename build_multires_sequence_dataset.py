from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SourceConfig:
    daily_transition_candidates: List[str]
    weekly_transition_candidates: List[str]
    meal_source_candidates: List[str]
    meal_id_candidates: List[str]
    meal_time_candidates: List[str]


CONFIG = SourceConfig(
    daily_transition_candidates=[
        "training/daily_transition/days_transition_matrix.csv",
    ],
    weekly_transition_candidates=[
        "training/regime_transition/weeks_transition_matrix.csv",
        "training/weekly_transition/weeks_transition_matrix.csv",
    ],
    meal_source_candidates=[
        "training/predictive_view.csv",
        "training/meal_feature_matrix.csv",
        "training/meals_feature_matrix.csv",
        "training/meal_event_matrix.csv",
        "training/meal_predictive_view.csv",
        "training/meal_state_context.csv",
        "training/meal_state_matrix.csv",
    ],
    meal_id_candidates=[
        "meal_id", "event_id", "log_id", "row_id", "entry_id",
    ],
    meal_time_candidates=[
        "meal_time", "datetime_local", "timestamp_local", "logged_at",
        "meal_datetime_local", "time_local", "event_time_local", "created_at",
    ],
)


DEFAULT_DAILY_TARGETS = [
    "y_next_weight_gain_flag",
    "y_next_weight_loss_flag",
    "y_next_weight_delta_lb",
    "y_next_logged_food_kcal_day",
    "y_next_restaurant_meal_fraction_day",
    "y_next_budget_breach_flag",
]

DEFAULT_WEEKLY_TARGETS = [
    "y_next_weight_gain_flag",
    "y_next_restaurant_heavy_flag",
    "y_next_budget_breach_flag",
]


def log(msg: str) -> None:
    print(f"[multires-seq] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_existing_path(project_root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for rel in candidates:
        p = project_root / rel
        if p.exists():
            return p
    return None


def first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


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
    return out


def clean_bool_and_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype) == "boolean":
            out[c] = out[c].astype("float")
        elif str(out[c].dtype) == "bool":
            out[c] = out[c].astype("float")
    return out


def prepare_daily_base(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    exclude = [c for c in df.columns if c.startswith("y_")]
    exclude.extend([
        "next_period_id", "next_period_start", "split_suggested",
        "period_kind", "period_id", "day_id", "date",
    ])
    exclude = [c for c in exclude if c in df.columns]
    out = df.drop(columns=exclude, errors="ignore").copy()
    out = add_time_features(out)
    out = clean_bool_and_numeric(out)
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    categorical_cols = [c for c in out.columns if c not in numeric_cols]
    return out, numeric_cols, categorical_cols


def prepare_weekly_base(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    exclude = [c for c in df.columns if c.startswith("y_")]
    exclude.extend([
        "next_period_id", "next_period_start", "split_suggested",
        "period_kind", "period_id", "week_id", "weekend_id",
    ])
    exclude = [c for c in exclude if c in df.columns]
    out = df.drop(columns=exclude, errors="ignore").copy()
    out = add_time_features(out)
    out = clean_bool_and_numeric(out)
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    categorical_cols = [c for c in out.columns if c not in numeric_cols]
    return out, numeric_cols, categorical_cols


def prepare_meal_base(df: pd.DataFrame, meal_id_col: str, meal_time_col: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    exclude = [c for c in df.columns if c.startswith("y_")]
    exclude.extend([
        meal_id_col, meal_time_col,
        "next_period_id", "next_period_start", "split_suggested",
        "period_kind", "period_id", "day_id", "date",
    ])
    exclude = [c for c in exclude if c in df.columns]
    out = df.drop(columns=exclude, errors="ignore").copy()
    out = clean_bool_and_numeric(out)
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    categorical_cols = [c for c in out.columns if c not in numeric_cols]
    return out, numeric_cols, categorical_cols


def pack_numeric_sequence(
    rows_df: pd.DataFrame,
    seq_len: int,
    numeric_cols: List[str],
    time_col: str,
    anchor_time: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.zeros((seq_len, len(numeric_cols)), dtype=np.float32)
    mask = np.zeros(seq_len, dtype=np.float32)
    age_days = np.full(seq_len, np.nan, dtype=np.float32)

    if rows_df.empty or not numeric_cols:
        return X, mask, age_days

    rows_sorted = rows_df.sort_values(time_col, ascending=False).head(seq_len).copy()
    for i, (_, row) in enumerate(rows_sorted.iterrows()):
        vals = pd.to_numeric(row[numeric_cols], errors="coerce").to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        X[i, :] = vals.astype(np.float32)
        mask[i] = 1.0
        t = pd.to_datetime(row[time_col], errors="coerce")
        if pd.notna(t) and pd.notna(anchor_time):
            age_days[i] = float((anchor_time - t).total_seconds() / 86400.0)
    return X, mask, age_days


def longify_sequence(
    rows_df: pd.DataFrame,
    seq_len: int,
    anchor_id: str,
    anchor_time: pd.Timestamp,
    time_col: str,
    keep_cols: List[str],
    modality: str,
) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(columns=["anchor_id", "modality", "step_back", "sequence_time", "age_days"] + keep_cols)
    rows_sorted = rows_df.sort_values(time_col, ascending=False).head(seq_len).copy()
    rows_sorted["anchor_id"] = anchor_id
    rows_sorted["modality"] = modality
    rows_sorted["step_back"] = np.arange(len(rows_sorted))
    rows_sorted["sequence_time"] = pd.to_datetime(rows_sorted[time_col], errors="coerce")
    rows_sorted["age_days"] = (anchor_time - rows_sorted["sequence_time"]).dt.total_seconds() / 86400.0
    cols = ["anchor_id", "modality", "step_back", "sequence_time", "age_days"] + [c for c in keep_cols if c in rows_sorted.columns]
    return rows_sorted[cols].copy()


def join_week_context(anchor_time: pd.Timestamp, week_df: pd.DataFrame) -> Optional[pd.Series]:
    if week_df is None or week_df.empty:
        return None
    candidates = week_df[pd.to_datetime(week_df["period_start"], errors="coerce") <= anchor_time].copy()
    if candidates.empty:
        return None
    if "next_period_start" in candidates.columns:
        within = candidates[
            (pd.to_datetime(candidates["period_start"], errors="coerce") <= anchor_time) &
            (pd.to_datetime(candidates["next_period_start"], errors="coerce") > anchor_time)
        ]
        if not within.empty:
            return within.sort_values("period_start").iloc[-1]
    return candidates.sort_values("period_start").iloc[-1]


def maybe_select_targets(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def write_npz(path: Path, **kwargs) -> None:
    np.savez_compressed(path, **kwargs)


def build_report(overall_df: pd.DataFrame, source_manifest: Dict) -> str:
    lines = []
    lines.append("# Multi-Resolution Sequence Dataset Build")
    lines.append("")
    lines.append("## Source manifest")
    for k, v in source_manifest.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Overall summary")
    lines.append("")
    if overall_df.empty:
        lines.append("_No rows_")
    else:
        cols = overall_df.columns.tolist()
        widths = [len(c) for c in cols]
        rows = overall_df.astype(str).values.tolist()
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        header = "| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |"
        sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
        body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |" for row in rows]
        lines.extend([header, sep] + body)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-resolution supervised sequence dataset.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--daily-transition-csv", default="", help="Optional explicit relative path to daily transition matrix.")
    parser.add_argument("--weekly-transition-csv", default="", help="Optional explicit relative path to weekly transition matrix.")
    parser.add_argument("--meal-source-csv", default="", help="Optional explicit relative path to meal source file.")
    parser.add_argument("--out-dir", default="training/multires_sequence_dataset", help="Relative output directory.")
    parser.add_argument("--meal-seq-len", type=int, default=24)
    parser.add_argument("--day-seq-len", type=int, default=14)
    parser.add_argument("--week-seq-len", type=int, default=8)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

    daily_path = project_root / args.daily_transition_csv if args.daily_transition_csv else find_existing_path(project_root, CONFIG.daily_transition_candidates)
    if daily_path is None or not daily_path.exists():
        raise FileNotFoundError("Daily transition matrix is required.")

    weekly_path = project_root / args.weekly_transition_csv if args.weekly_transition_csv else find_existing_path(project_root, CONFIG.weekly_transition_candidates)
    meal_path = project_root / args.meal_source_csv if args.meal_source_csv else find_existing_path(project_root, CONFIG.meal_source_candidates)

    daily_df = pd.read_csv(daily_path, low_memory=False)
    daily_df["period_start"] = pd.to_datetime(daily_df["period_start"], errors="coerce")
    if "next_period_start" in daily_df.columns:
        daily_df["next_period_start"] = pd.to_datetime(daily_df["next_period_start"], errors="coerce")
    daily_df = daily_df.sort_values("period_start").reset_index(drop=True)

    daily_targets = maybe_select_targets(daily_df, DEFAULT_DAILY_TARGETS)
    if not daily_targets:
        raise ValueError("No expected daily targets found in daily transition matrix.")

    week_df = None
    weekly_targets = []
    if weekly_path is not None and weekly_path.exists():
        week_df = pd.read_csv(weekly_path, low_memory=False)
        if "period_start" in week_df.columns:
            week_df["period_start"] = pd.to_datetime(week_df["period_start"], errors="coerce")
        if "next_period_start" in week_df.columns:
            week_df["next_period_start"] = pd.to_datetime(week_df["next_period_start"], errors="coerce")
        week_df = week_df.sort_values("period_start").reset_index(drop=True)
        weekly_targets = maybe_select_targets(week_df, DEFAULT_WEEKLY_TARGETS)

    meal_df = None
    meal_id_col = None
    meal_time_col = None
    if meal_path is not None and meal_path.exists():
        meal_df = pd.read_csv(meal_path, low_memory=False)
        meal_id_col = first_present(list(meal_df.columns), CONFIG.meal_id_candidates)
        meal_time_col = first_present(list(meal_df.columns), CONFIG.meal_time_candidates)
        if meal_time_col is None:
            log(f"Meal source detected at {meal_path}, but no meal time column was found. Meals modality will be skipped.")
            meal_df = None
        else:
            meal_df[meal_time_col] = pd.to_datetime(meal_df[meal_time_col], errors="coerce")
            meal_df = meal_df.sort_values(meal_time_col).reset_index(drop=True)
            if meal_id_col is None:
                meal_id_col = "row_index_meal_id"
                meal_df[meal_id_col] = np.arange(len(meal_df)).astype(str)

    day_base, day_num_cols, day_cat_cols = prepare_daily_base(daily_df)
    week_base = week_num_cols = week_cat_cols = None
    if week_df is not None:
        week_base, week_num_cols, week_cat_cols = prepare_weekly_base(week_df)
    meal_base = meal_num_cols = meal_cat_cols = None
    if meal_df is not None:
        meal_base, meal_num_cols, meal_cat_cols = prepare_meal_base(meal_df, meal_id_col, meal_time_col)

    anchors = daily_df.copy()
    anchors = anchors[anchors["next_period_start"].notna()].copy().reset_index(drop=True)

    anchor_rows = []
    modality_mask_rows = []
    meal_long_parts = []
    day_long_parts = []
    week_long_parts = []

    n = len(anchors)
    meal_tensor = np.zeros((n, args.meal_seq_len, len(meal_num_cols or [])), dtype=np.float32)
    meal_mask = np.zeros((n, args.meal_seq_len), dtype=np.float32)
    meal_age = np.full((n, args.meal_seq_len), np.nan, dtype=np.float32)

    day_tensor = np.zeros((n, args.day_seq_len, len(day_num_cols or [])), dtype=np.float32)
    day_mask = np.zeros((n, args.day_seq_len), dtype=np.float32)
    day_age = np.full((n, args.day_seq_len), np.nan, dtype=np.float32)

    week_tensor = np.zeros((n, args.week_seq_len, len(week_num_cols or [])), dtype=np.float32) if week_df is not None else np.zeros((n, 0, 0), dtype=np.float32)
    week_mask = np.zeros((n, args.week_seq_len), dtype=np.float32) if week_df is not None else np.zeros((n, 0), dtype=np.float32)
    week_age = np.full((n, args.week_seq_len), np.nan, dtype=np.float32) if week_df is not None else np.zeros((n, 0), dtype=np.float32)

    for i, (_, anchor) in enumerate(anchors.iterrows()):
        anchor_id = str(anchor["period_id"])
        anchor_time = pd.to_datetime(anchor["period_start"], errors="coerce")
        next_time = pd.to_datetime(anchor["next_period_start"], errors="coerce")

        row = {
            "anchor_row": i,
            "anchor_id": anchor_id,
            "anchor_period_start": str(anchor_time),
            "anchor_next_period_start": str(next_time),
            "split_suggested": anchor.get("split_suggested", ""),
        }
        for t in daily_targets:
            row[t] = anchor.get(t)
        if week_df is not None:
            week_ctx = join_week_context(anchor_time, week_df)
            if week_ctx is not None:
                row["week_context_period_id"] = str(week_ctx.get("period_id", ""))
                row["week_context_period_start"] = str(week_ctx.get("period_start", ""))
                for t in weekly_targets:
                    row[f"weekctx__{t}"] = week_ctx.get(t)
        anchor_rows.append(row)

        # day history up to current day inclusive
        day_hist_idx = daily_df["period_start"] <= anchor_time
        day_hist_rows = day_base.loc[day_hist_idx].copy()
        Xd, Md, Ad = pack_numeric_sequence(
            rows_df=day_hist_rows.assign(period_start=daily_df.loc[day_hist_idx, "period_start"].values),
            seq_len=args.day_seq_len,
            numeric_cols=day_num_cols,
            time_col="period_start",
            anchor_time=anchor_time,
        )
        day_tensor[i, :, :] = Xd
        day_mask[i, :] = Md
        day_age[i, :] = Ad
        keep_day_cols = ["period_id", "period_start"] + day_num_cols[:40] + day_cat_cols[:20]
        day_hist_long = daily_df.loc[day_hist_idx, keep_day_cols].copy()
        day_long_parts.append(longify_sequence(day_hist_long, args.day_seq_len, anchor_id, anchor_time, "period_start", keep_day_cols, "days"))

        # week history up to current day
        if week_df is not None:
            week_hist_idx = week_df["period_start"] <= anchor_time
            week_hist_rows = week_base.loc[week_hist_idx].copy()
            Xw, Mw, Aw = pack_numeric_sequence(
                rows_df=week_hist_rows.assign(period_start=week_df.loc[week_hist_idx, "period_start"].values),
                seq_len=args.week_seq_len,
                numeric_cols=week_num_cols,
                time_col="period_start",
                anchor_time=anchor_time,
            )
            week_tensor[i, :, :] = Xw
            week_mask[i, :] = Mw
            week_age[i, :] = Aw
            keep_week_cols = ["period_id", "period_start"] + week_num_cols[:40] + week_cat_cols[:20]
            week_hist_long = week_df.loc[week_hist_idx, keep_week_cols].copy()
            week_long_parts.append(longify_sequence(week_hist_long, args.week_seq_len, anchor_id, anchor_time, "period_start", keep_week_cols, "weeks"))

        # meal history up to next day boundary (includes current day meals)
        if meal_df is not None and meal_time_col is not None:
            meal_hist_idx = meal_df[meal_time_col] < next_time
            meal_hist_rows = meal_base.loc[meal_hist_idx].copy()
            Xm, Mm, Am = pack_numeric_sequence(
                rows_df=meal_hist_rows.assign(**{meal_time_col: meal_df.loc[meal_hist_idx, meal_time_col].values}),
                seq_len=args.meal_seq_len,
                numeric_cols=meal_num_cols,
                time_col=meal_time_col,
                anchor_time=anchor_time,
            )
            meal_tensor[i, :, :] = Xm
            meal_mask[i, :] = Mm
            meal_age[i, :] = Am
            keep_meal_cols = [meal_id_col, meal_time_col] + meal_num_cols[:40] + meal_cat_cols[:20]
            meal_hist_long = meal_df.loc[meal_hist_idx, keep_meal_cols].copy()
            meal_long_parts.append(longify_sequence(meal_hist_long, args.meal_seq_len, anchor_id, anchor_time, meal_time_col, keep_meal_cols, "meals"))

        modality_mask_rows.append({
            "anchor_id": anchor_id,
            "has_meals": bool(meal_df is not None and meal_mask[i].sum() > 0),
            "has_days": bool(day_mask[i].sum() > 0),
            "has_weeks": bool(week_df is not None and week_mask[i].sum() > 0),
            "n_meals_steps_observed": int(meal_mask[i].sum()) if meal_df is not None else 0,
            "n_days_steps_observed": int(day_mask[i].sum()),
            "n_weeks_steps_observed": int(week_mask[i].sum()) if week_df is not None else 0,
        })

    anchors_df = pd.DataFrame(anchor_rows)
    modality_mask_df = pd.DataFrame(modality_mask_rows)

    anchors_df.to_csv(out_dir / "anchors.csv", index=False)
    modality_mask_df.to_csv(out_dir / "modality_masks.csv", index=False)

    if meal_long_parts:
        pd.concat(meal_long_parts, ignore_index=True).to_csv(out_dir / "meals_sequence_long.csv.gz", index=False)
    if day_long_parts:
        pd.concat(day_long_parts, ignore_index=True).to_csv(out_dir / "days_sequence_long.csv.gz", index=False)
    if week_long_parts:
        pd.concat(week_long_parts, ignore_index=True).to_csv(out_dir / "weeks_sequence_long.csv.gz", index=False)

    write_npz(
        out_dir / "days_numeric_sequences.npz",
        X=day_tensor,
        mask=day_mask,
        age_days=day_age,
        anchor_ids=anchors_df["anchor_id"].astype(str).to_numpy(),
        feature_names=np.array(day_num_cols, dtype=object),
    )
    write_npz(
        out_dir / "meals_numeric_sequences.npz",
        X=meal_tensor,
        mask=meal_mask,
        age_days=meal_age,
        anchor_ids=anchors_df["anchor_id"].astype(str).to_numpy(),
        feature_names=np.array(meal_num_cols or [], dtype=object),
    )
    write_npz(
        out_dir / "weeks_numeric_sequences.npz",
        X=week_tensor,
        mask=week_mask,
        age_days=week_age,
        anchor_ids=anchors_df["anchor_id"].astype(str).to_numpy(),
        feature_names=np.array(week_num_cols or [], dtype=object),
    )

    source_manifest = {
        "daily_transition_csv": str(daily_path),
        "weekly_transition_csv": None if weekly_path is None else str(weekly_path),
        "meal_source_csv": None if meal_path is None else str(meal_path),
        "meal_source_detected": bool(meal_df is not None),
        "daily_targets": daily_targets,
        "weekly_targets": weekly_targets,
        "meal_id_col": meal_id_col,
        "meal_time_col": meal_time_col,
    }
    save_json(out_dir / "source_manifest.json", source_manifest)

    feature_manifest_rows = [
        {
            "modality": "days",
            "numeric_feature_count": len(day_num_cols),
            "categorical_feature_count": len(day_cat_cols),
            "sequence_length": args.day_seq_len,
        },
        {
            "modality": "meals",
            "numeric_feature_count": len(meal_num_cols or []),
            "categorical_feature_count": len(meal_cat_cols or []),
            "sequence_length": args.meal_seq_len,
        },
        {
            "modality": "weeks",
            "numeric_feature_count": len(week_num_cols or []),
            "categorical_feature_count": len(week_cat_cols or []),
            "sequence_length": args.week_seq_len,
        },
    ]
    pd.DataFrame(feature_manifest_rows).to_csv(out_dir / "feature_manifest.csv", index=False)

    overall_df = pd.DataFrame([{
        "anchors": int(len(anchors_df)),
        "daily_targets": len(daily_targets),
        "weekly_targets": len(weekly_targets),
        "day_numeric_features": len(day_num_cols),
        "meal_numeric_features": len(meal_num_cols or []),
        "week_numeric_features": len(week_num_cols or []),
        "meal_modality_detected": bool(meal_df is not None),
        "week_modality_detected": bool(week_df is not None),
        "pct_has_meals": float(modality_mask_df["has_meals"].mean()) if "has_meals" in modality_mask_df.columns else np.nan,
        "pct_has_days": float(modality_mask_df["has_days"].mean()) if "has_days" in modality_mask_df.columns else np.nan,
        "pct_has_weeks": float(modality_mask_df["has_weeks"].mean()) if "has_weeks" in modality_mask_df.columns else np.nan,
    }])
    overall_df.to_csv(out_dir / "overall_summary.csv", index=False)
    save_json(out_dir / "overall_summary.json", {
        "anchors": int(len(anchors_df)),
        "day_sequence_length": args.day_seq_len,
        "meal_sequence_length": args.meal_seq_len,
        "week_sequence_length": args.week_seq_len,
        "source_manifest": source_manifest,
    })
    (out_dir / "overall_report.md").write_text(
        build_report(overall_df, source_manifest),
        encoding="utf-8",
    )

    log("Done.")
    log(f"Wrote dataset under: {out_dir}")


if __name__ == "__main__":
    main()
