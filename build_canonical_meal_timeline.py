from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ScanConfig:
    roots: List[str]
    include_exts: Tuple[str, ...]
    exclude_keywords: Tuple[str, ...]
    id_candidates: List[str]
    time_candidates: List[str]
    day_candidates: List[str]
    kcal_candidates: List[str]
    restaurant_candidates: List[str]
    archetype_candidates: List[str]
    cuisine_candidates: List[str]
    protein_candidates: List[str]
    starch_candidates: List[str]
    meal_name_candidates: List[str]
    within_day_order_candidates: List[str]


CONFIG = ScanConfig(
    roots=["meal_db", "training", "canonical", "fused", "."],
    include_exts=(".csv",),
    exclude_keywords=(
        "days_transition_matrix",
        "weeks_transition_matrix",
        "weekends_transition_matrix",
        "overall_summary",
        "model_comparison",
        "test_predictions",
        "top_features",
        "calibration",
        "sequence_long",
        "scored_rows",
        "meal_sequence_source",
        "multires_sequence_dataset",
    ),
    id_candidates=["meal_id", "event_id", "entry_id", "log_id", "row_id", "id"],
    time_candidates=[
        "meal_time", "meal_datetime_local", "datetime_local", "event_time_local",
        "logged_at", "created_at", "timestamp_local", "timestamp",
        "datetime", "time_local", "meal_ts", "event_ts",
    ],
    day_candidates=["date", "date_local", "day_id", "day_date", "logged_date"],
    kcal_candidates=["calories_kcal", "noom_food_calories_kcal", "calories", "kcal", "energy_kcal"],
    restaurant_candidates=["restaurant_specific", "restaurant_flag", "is_restaurant", "restaurant_meal", "meal_restaurant_flag"],
    archetype_candidates=["meal_archetype", "meal_archetype_collapsed", "dominant_meal_archetype", "archetype"],
    cuisine_candidates=["cuisine", "dominant_cuisine", "meal_cuisine"],
    protein_candidates=["principal_protein", "dominant_principal_protein", "protein_anchor", "protein_type"],
    starch_candidates=["principal_starch", "dominant_principal_starch", "starch_base", "starch_type"],
    meal_name_candidates=["meal_name", "food_name", "title", "entry_name", "name"],
    within_day_order_candidates=[
        "meal_order", "event_order", "order_in_day", "row_num", "source_row_index",
        "sequence", "seq", "index_in_day", "meal_index", "position_in_day",
    ],
)


def log(msg: str) -> None:
    print(f"[canonical-meal] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def build_norm_map(columns: Sequence[str]) -> Dict[str, str]:
    out = {}
    for c in columns:
        n = normalize_name(c)
        if n not in out:
            out[n] = c
    return out


def first_present(norm_map: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        n = normalize_name(c)
        if n in norm_map:
            return norm_map[n]
    return None


def candidate_paths(project_root: Path) -> List[Path]:
    found = []
    seen = set()
    for rel_root in CONFIG.roots:
        root = (project_root / rel_root).resolve()
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in CONFIG.include_exts:
                continue
            low = str(p).lower()
            if any(k in low for k in CONFIG.exclude_keywords):
                continue
            if p in seen:
                continue
            seen.add(p)
            found.append(p)
    return sorted(found)


def parse_datetime_series(series: pd.Series) -> pd.Series:
    # First pass: generic parse
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() >= 0.85:
        return parsed
    # Second pass: common date-time patterns
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S"):
        parsed = pd.to_datetime(series, errors="coerce", format=fmt)
        if parsed.notna().mean() >= 0.85:
            return parsed
    return pd.to_datetime(series, errors="coerce")


def safe_str_cell(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    if isinstance(x, (list, tuple, set)):
        return "; ".join(str(v) for v in x)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    if isinstance(x, np.ndarray):
        return "; ".join(str(v) for v in x.tolist())
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def compute_rows_per_day(df: pd.DataFrame, day_col: Optional[str], time_col: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if day_col and day_col in df.columns:
        day_s = df[day_col]
        parsed = pd.to_datetime(day_s, errors="coerce")
        if parsed.notna().mean() > 0.8:
            day_key = parsed.dt.strftime("%Y-%m-%d")
        else:
            day_key = day_s.astype(str)
    elif time_col and time_col in df.columns:
        parsed = parse_datetime_series(df[time_col])
        day_key = parsed.dt.strftime("%Y-%m-%d")
    else:
        return None, None

    vc = day_key.dropna().value_counts()
    if vc.empty:
        return None, None
    return float(vc.mean()), float(vc.median())


def filename_semantic_boost(path: Path) -> Tuple[float, List[str]]:
    low = str(path).lower()
    score = 0.0
    reasons = []

    strong_positive = [
        ("meal_event_current", 7.0),
        ("meal_event_seed", 6.0),
        ("meal_semantic_features", 6.0),
        ("meal_db", 4.0),
        ("meal", 2.5),
    ]
    strong_negative = [
        ("15min", -12.0),
        ("telemetry", -12.0),
        ("master_15min", -12.0),
        ("master_event_ledger", -4.0),
        ("samsung_", -5.0),
        ("stress_raw", -5.0),
        ("hr_raw", -5.0),
        ("weight_events", -2.0),
    ]

    for key, val in strong_positive:
        if key in low:
            score += val
            reasons.append(f"filename_boost_{key}")
    for key, val in strong_negative:
        if key in low:
            score += val
            reasons.append(f"filename_penalty_{key}")

    return score, reasons


def score_candidate(path: Path, df: pd.DataFrame) -> Dict:
    norm_map = build_norm_map(df.columns)
    id_col = first_present(norm_map, CONFIG.id_candidates)
    time_col = first_present(norm_map, CONFIG.time_candidates)
    day_col = first_present(norm_map, CONFIG.day_candidates)
    order_col = first_present(norm_map, CONFIG.within_day_order_candidates)

    hits = {
        "kcal_col": first_present(norm_map, CONFIG.kcal_candidates),
        "restaurant_col": first_present(norm_map, CONFIG.restaurant_candidates),
        "archetype_col": first_present(norm_map, CONFIG.archetype_candidates),
        "cuisine_col": first_present(norm_map, CONFIG.cuisine_candidates),
        "protein_col": first_present(norm_map, CONFIG.protein_candidates),
        "starch_col": first_present(norm_map, CONFIG.starch_candidates),
        "meal_name_col": first_present(norm_map, CONFIG.meal_name_candidates),
    }

    sem_hits = sum(v is not None for v in hits.values())
    n_rows = len(df)
    score = 0.0
    reasons: List[str] = []

    fscore, freasons = filename_semantic_boost(path)
    score += fscore
    reasons.extend(freasons)

    if id_col is not None:
        score += 4.0
        reasons.append("has_id_col")
    if time_col is not None:
        score += 3.0
        reasons.append("has_time_col")
    if day_col is not None:
        score += 3.0
        reasons.append("has_day_col")
    if order_col is not None:
        score += 1.0
        reasons.append("has_within_day_order_col")

    score += sem_hits * 2.0
    if sem_hits:
        reasons.append(f"semantic_hits_{sem_hits}")

    if hits["protein_col"] is not None and hits["starch_col"] is not None:
        score += 4.0
        reasons.append("has_protein_and_starch")
    if hits["archetype_col"] is not None or hits["cuisine_col"] is not None:
        score += 2.0
        reasons.append("has_semantic_meal_labels")

    if 50 <= n_rows <= 500000:
        score += 1.0
        reasons.append("reasonable_row_count")

    parseable_frac = None
    if time_col is not None:
        t = parse_datetime_series(df[time_col])
        parseable_frac = float(t.notna().mean()) if len(df) else 0.0
        # Time helps, but not enough to let telemetry beat event tables.
        score += 2.0 * parseable_frac
        reasons.append(f"time_parseable_frac_{parseable_frac:.3f}")

    rows_per_day_mean, rows_per_day_median = compute_rows_per_day(df, day_col=day_col, time_col=time_col)
    if rows_per_day_mean is not None:
        reasons.append(f"rows_per_day_mean_{rows_per_day_mean:.2f}")
        # Strong preference for plausible meal density
        if rows_per_day_mean <= 1:
            score -= 3.0
            reasons.append("implausibly_sparse_events")
        elif rows_per_day_mean <= 12:
            score += 8.0
            reasons.append("plausible_meal_density")
        elif rows_per_day_mean <= 24:
            score += 1.0
            reasons.append("borderline_dense")
        else:
            # heavy penalty for telemetry-like grids
            score -= min(18.0, 0.35 * (rows_per_day_mean - 24.0))
            reasons.append("telemetry_like_density_penalty")

    return {
        "path": str(path),
        "rows": int(n_rows),
        "cols": int(df.shape[1]),
        "score": float(score),
        "reasons": reasons,
        "detected_id_col": id_col,
        "detected_time_col": time_col,
        "detected_day_col": day_col,
        "detected_order_col": order_col,
        **hits,
        "time_parseable_frac": parseable_frac,
        "rows_per_day_mean": rows_per_day_mean,
        "rows_per_day_median": rows_per_day_median,
    }


def make_day_id(df: pd.DataFrame, time_col: Optional[str], day_col: Optional[str]) -> pd.Series:
    if day_col is not None and day_col in df.columns:
        raw = df[day_col]
        parsed = pd.to_datetime(raw, errors="coerce")
        if parsed.notna().mean() > 0.8:
            return parsed.dt.strftime("%Y-%m-%d")
        return raw.astype(str)
    if time_col is not None and time_col in df.columns:
        t = parse_datetime_series(df[time_col])
        return t.dt.strftime("%Y-%m-%d")
    return pd.Series(np.arange(len(df)), index=df.index).astype(str)


def infer_within_day_order(df: pd.DataFrame, meta: Dict) -> pd.Series:
    order_col = meta.get("detected_order_col")
    if order_col is not None and order_col in df.columns:
        vals = pd.to_numeric(df[order_col], errors="coerce")
        if vals.notna().sum() > 0:
            return vals.rank(method="first").astype(int)

    meal_id_col = meta.get("detected_id_col")
    if meal_id_col is not None and meal_id_col in df.columns:
        # extract numeric suffix if present
        raw = df[meal_id_col].astype(str)
        num = raw.str.extract(r"(\d+)")[0]
        num = pd.to_numeric(num, errors="coerce")
        if num.notna().sum() > 0:
            return num.rank(method="first").astype(int)

    return pd.Series(np.arange(len(df)), index=df.index).astype(int)


def synthesize_meal_times(day_ids: pd.Series, order_in_day: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"day_id": day_ids.astype(str), "order_in_day": order_in_day.astype(int)})
    out = pd.Series(pd.NaT, index=tmp.index, dtype="datetime64[ns]")
    for day, g in tmp.groupby("day_id", sort=False):
        n = len(g)
        if n == 1:
            offsets = [12 * 60]
        else:
            # spread meals between 07:00 and 22:00 local
            offsets = np.linspace(7 * 60, 22 * 60, num=n)
        order = g.sort_values("order_in_day").index.tolist()
        for idx, mins in zip(order, offsets):
            base = pd.to_datetime(day, errors="coerce")
            if pd.notna(base):
                out.loc[idx] = base + pd.Timedelta(minutes=float(mins))
    return out


def standardize_meal_source(df: pd.DataFrame, scan_meta: Dict) -> Tuple[pd.DataFrame, Dict]:
    time_col = scan_meta["detected_time_col"]
    id_col = scan_meta["detected_id_col"]
    day_col = scan_meta["detected_day_col"]

    work = df.copy()
    if time_col is not None:
        work[time_col] = parse_datetime_series(work[time_col])

    if id_col is None or id_col not in work.columns:
        id_col = "__generated_meal_id"
        work[id_col] = np.arange(len(work)).astype(str)

    day_id = make_day_id(work, time_col=time_col, day_col=day_col)
    order_in_day = infer_within_day_order(work, scan_meta)

    out = pd.DataFrame(index=work.index)
    out["meal_event_id"] = work[id_col].astype(str)
    out["day_id"] = day_id
    out["order_in_day"] = order_in_day

    if time_col is not None and work[time_col].notna().sum() > 0:
        out["meal_time"] = pd.to_datetime(work[time_col], errors="coerce")
    else:
        out["meal_time"] = synthesize_meal_times(out["day_id"], out["order_in_day"])
    out["source_row_index"] = np.arange(len(work))
    out["source_table_path"] = scan_meta["path"]

    mapping = {}
    for std_name, src_name in [
        ("meal_kcal", scan_meta.get("kcal_col")),
        ("restaurant_flag", scan_meta.get("restaurant_col")),
        ("meal_archetype", scan_meta.get("archetype_col")),
        ("cuisine", scan_meta.get("cuisine_col")),
        ("principal_protein", scan_meta.get("protein_col")),
        ("principal_starch", scan_meta.get("starch_col")),
        ("meal_name", scan_meta.get("meal_name_col")),
    ]:
        if src_name is not None and src_name in work.columns:
            out[std_name] = work[src_name]
            mapping[std_name] = src_name

    # Keep all source columns too, prefixed, for richness.
    for c in work.columns:
        pref = f"src__{normalize_name(c)}"
        if pref not in out.columns:
            out[pref] = work[c]

    if "meal_kcal" in out.columns:
        out["meal_kcal"] = pd.to_numeric(out["meal_kcal"], errors="coerce")
    if "restaurant_flag" in out.columns:
        series = out["restaurant_flag"]
        if series.dtype == object:
            mapped = series.astype(str).str.lower().map({
                "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0,
                "yes": 1.0, "no": 0.0,
            })
            if mapped.notna().mean() > 0.5:
                out["restaurant_flag"] = mapped
        if str(out["restaurant_flag"].dtype) in ("bool", "boolean"):
            out["restaurant_flag"] = out["restaurant_flag"].astype("float")

    out = out[out["day_id"].notna()].copy()
    out = out.sort_values(["meal_time", "day_id", "order_in_day", "meal_event_id"], na_position="last").reset_index(drop=True)
    out["meal_seq_order"] = np.arange(len(out))

    manifest = {
        "chosen_source_path": scan_meta["path"],
        "chosen_source_score": scan_meta["score"],
        "rows_out": int(len(out)),
        "columns_out": int(out.shape[1]),
        "detected_id_col": scan_meta["detected_id_col"],
        "detected_time_col": scan_meta["detected_time_col"],
        "detected_day_col": scan_meta["detected_day_col"],
        "detected_order_col": scan_meta["detected_order_col"],
        "canonical_mapping": mapping,
        "time_min": None if out["meal_time"].dropna().empty else str(out["meal_time"].min()),
        "time_max": None if out["meal_time"].dropna().empty else str(out["meal_time"].max()),
        "distinct_day_count": int(out["day_id"].nunique(dropna=True)),
        "used_synthetic_time": bool(scan_meta["detected_time_col"] is None),
    }
    return out, manifest


def build_summary(std_df: pd.DataFrame, manifest: Dict) -> Dict:
    out = dict(manifest)
    out["rows_with_kcal"] = int(std_df["meal_kcal"].notna().sum()) if "meal_kcal" in std_df.columns else 0
    out["rows_with_restaurant_flag"] = int(std_df["restaurant_flag"].notna().sum()) if "restaurant_flag" in std_df.columns else 0
    out["rows_with_archetype"] = int(std_df["meal_archetype"].notna().sum()) if "meal_archetype" in std_df.columns else 0
    out["rows_with_cuisine"] = int(std_df["cuisine"].notna().sum()) if "cuisine" in std_df.columns else 0
    out["rows_with_protein"] = int(std_df["principal_protein"].notna().sum()) if "principal_protein" in std_df.columns else 0
    out["rows_with_starch"] = int(std_df["principal_starch"].notna().sum()) if "principal_starch" in std_df.columns else 0
    if len(std_df):
        meals_per_day = std_df.groupby("day_id").size()
        out["meals_per_day_mean"] = float(meals_per_day.mean())
        out["meals_per_day_median"] = float(meals_per_day.median())
        out["meals_per_day_p95"] = float(meals_per_day.quantile(0.95))
    else:
        out["meals_per_day_mean"] = 0.0
        out["meals_per_day_median"] = 0.0
        out["meals_per_day_p95"] = 0.0
    return out


def build_report(summary: Dict, top_candidates: pd.DataFrame) -> str:
    lines = []
    lines.append("# Canonical Meal Timeline Build")
    lines.append("")
    lines.append("## Chosen source")
    lines.append("")
    for k, v in summary.items():
        lines.append(f"- {k}: {safe_str_cell(v)}")
    lines.append("")
    lines.append("## Top scanned candidates")
    lines.append("")
    if top_candidates.empty:
        lines.append("_No candidate files found_")
    else:
        df = top_candidates.copy()
        df.columns = [str(c) for c in df.columns]
        for c in df.columns:
            df[c] = df[c].map(safe_str_cell)
        headers = df.columns.tolist()
        rows = df.values.tolist()
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
        body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
        lines.extend([header, sep] + body)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and standardize a canonical meal-event timeline.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--source-csv", default="", help="Optional explicit relative path to use as the meal source.")
    parser.add_argument("--out-dir", default="training/meal_timeline_canonical", help="Relative output directory.")
    parser.add_argument("--scan-limit", type=int, default=300, help="Max candidate files to score.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

    if args.source_csv:
        chosen_path = project_root / args.source_csv
        if not chosen_path.exists():
            raise FileNotFoundError(f"Missing explicit source CSV: {chosen_path}")
        df = safe_read_csv(chosen_path)
        if df is None:
            raise ValueError(f"Could not read explicit source CSV: {chosen_path}")
        scan_meta = score_candidate(chosen_path, df)
        top_candidates_df = pd.DataFrame([scan_meta])
    else:
        paths = candidate_paths(project_root)[: args.scan_limit]
        scored = []
        for p in paths:
            df = safe_read_csv(p)
            if df is None:
                continue
            meta = score_candidate(p, df)
            scored.append(meta)
        if not scored:
            raise FileNotFoundError("No readable candidate CSV files found for canonical meal-timeline discovery.")
        top_candidates_df = pd.DataFrame(scored).sort_values("score", ascending=False).reset_index(drop=True)
        chosen_meta = top_candidates_df.iloc[0].to_dict()
        chosen_path = Path(chosen_meta["path"])
        df = safe_read_csv(chosen_path)
        if df is None:
            raise ValueError(f"Top-scored candidate became unreadable: {chosen_path}")
        scan_meta = score_candidate(chosen_path, df)

    std_df, manifest = standardize_meal_source(df, scan_meta)
    summary = build_summary(std_df, manifest)

    std_path = out_dir / "canonical_meal_timeline.csv"
    preview_path = out_dir / "canonical_meal_timeline_preview.csv"
    manifest_path = out_dir / "canonical_meal_timeline_manifest.json"
    summary_path = out_dir / "canonical_meal_timeline_summary.json"
    candidates_path = out_dir / "meal_timeline_candidates_scored.csv"
    report_path = out_dir / "canonical_meal_timeline_report.md"

    std_df.to_csv(std_path, index=False)
    std_df.head(200).to_csv(preview_path, index=False)
    save_json(manifest_path, manifest)
    save_json(summary_path, summary)
    top_candidates_df.to_csv(candidates_path, index=False)
    report_path.write_text(build_report(summary, top_candidates_df.head(25)), encoding="utf-8")

    log("Done.")
    log(f"Wrote canonical meal timeline to: {std_path}")


if __name__ == "__main__":
    main()
