from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_DAY_SLOTS = ("lunch", "dinner", "snack")
SNACK_SLOTS = {"morning_snack", "afternoon_snack", "evening_snack"}
DEFAULT_HORIZONS = (3, 5, 7, 14, 30)


@dataclass(frozen=True)
class PlanningContext:
    start_date: pd.Timestamp
    latest_observed_date: str
    latest_weight_lb: float | None
    latest_weight_velocity_7d_lb: float | None
    recent_steps_mean: float | None
    recent_food_kcal_mean: float | None
    recent_restaurant_fraction: float | None
    recent_dominant_archetypes: Tuple[str, ...]
    season: str
    weekday: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def robust_bool(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes", "y"}:
        return 1.0
    if text in {"false", "0", "0.0", "no", "n"}:
        return 0.0
    return np.nan


def safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def minmax_score(value: float, low: float, high: float) -> float:
    if not np.isfinite(value) or not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def inverse_minmax_score(value: float, low: float, high: float) -> float:
    return 1.0 - minmax_score(value, low, high)


def season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "fall"


def slot_group(slot: str) -> str:
    return "snack" if str(slot) in SNACK_SLOTS else str(slot)


def observed_slot_summary(day_meals: pd.DataFrame) -> str:
    parts: List[str] = []
    for slot in ["breakfast", "lunch", "afternoon_snack", "dinner", "evening_snack"]:
        rows = day_meals.loc[day_meals["time_slot_label"].astype(str) == slot]
        if rows.empty:
            continue
        archetypes = rows["target_meal_archetype_primary"].astype(str).replace("nan", "unknown")
        calories = pd.to_numeric(rows["target_calories_kcal"], errors="coerce").fillna(0.0)
        parts.append(f"{slot}:{'/'.join(archetypes.head(2).tolist())} ({calories.sum():.0f} kcal)")
    return "; ".join(parts)


def load_source_tables(project_root: Path) -> Dict[str, pd.DataFrame]:
    meal_path = project_root / "training" / "meal_decision_points.csv"
    transition_path = project_root / "training" / "daily_transition" / "days_transition_matrix.csv"
    history_path = (
        project_root
        / "reports"
        / "backtests"
        / "temporal_multires"
        / "simple_loss_daysweeks_v2_operational_scoring_v1"
        / "history_scores.csv"
    )
    if not meal_path.exists():
        raise FileNotFoundError(f"Missing meal decision table: {meal_path}")
    if not transition_path.exists():
        raise FileNotFoundError(f"Missing daily transition table: {transition_path}")

    meals = pd.read_csv(meal_path, low_memory=False)
    transitions = pd.read_csv(transition_path, low_memory=False)
    history = pd.read_csv(history_path, low_memory=False) if history_path.exists() else pd.DataFrame()

    meals["date"] = pd.to_datetime(meals["date"], errors="coerce").dt.date.astype(str)
    transitions["date"] = pd.to_datetime(transitions["date"], errors="coerce").dt.date.astype(str)
    if not history.empty:
        history["anchor_id"] = history["anchor_id"].astype(str)
    return {"meals": meals, "transitions": transitions, "history_scores": history}


def build_planning_context(transitions: pd.DataFrame, start_date: str | None = None, recent_days: int = 14) -> PlanningContext:
    dated = transitions.copy()
    dated["_date_ts"] = pd.to_datetime(dated["date"], errors="coerce")
    dated = dated.dropna(subset=["_date_ts"]).sort_values("_date_ts")
    latest = dated.iloc[-1]
    recent = dated.tail(recent_days)
    if start_date:
        start_ts = pd.Timestamp(start_date)
    else:
        start_ts = latest["_date_ts"] + pd.Timedelta(days=1)
    dominant = tuple(
        recent["dominant_meal_archetype"].dropna().astype(str).value_counts().head(5).index.tolist()
    )
    return PlanningContext(
        start_date=start_ts.normalize(),
        latest_observed_date=str(latest["date"]),
        latest_weight_lb=None if pd.isna(latest.get("true_weight_lb")) else float(latest.get("true_weight_lb")),
        latest_weight_velocity_7d_lb=None
        if pd.isna(latest.get("weight_velocity_7d_lb"))
        else float(latest.get("weight_velocity_7d_lb")),
        recent_steps_mean=None
        if recent.get("steps_day", pd.Series(dtype=float)).dropna().empty
        else float(pd.to_numeric(recent["steps_day"], errors="coerce").mean()),
        recent_food_kcal_mean=None
        if recent.get("meal_calories_kcal_sum", pd.Series(dtype=float)).dropna().empty
        else float(pd.to_numeric(recent["meal_calories_kcal_sum"], errors="coerce").mean()),
        recent_restaurant_fraction=None
        if recent.get("restaurant_meal_fraction_day", pd.Series(dtype=float)).dropna().empty
        else float(pd.to_numeric(recent["restaurant_meal_fraction_day"], errors="coerce").mean()),
        recent_dominant_archetypes=dominant,
        season=season_for_month(int(start_ts.month)),
        weekday=str(start_ts.day_name()),
    )


def build_day_action_library(tables: Dict[str, pd.DataFrame], min_required_slots: bool = True) -> Tuple[pd.DataFrame, Dict]:
    meals = tables["meals"].copy()
    transitions = tables["transitions"].copy()
    history = tables["history_scores"].copy()

    numeric_cols = [
        "target_calories_kcal",
        "target_protein_g",
        "target_fiber_g",
        "target_sodium_mg",
        "target_comfort_food_score",
        "target_fresh_light_score",
        "target_indulgence_score",
    ]
    for col in numeric_cols:
        if col in meals.columns:
            meals[col] = pd.to_numeric(meals[col], errors="coerce")
    meals["slot_group"] = meals["time_slot_label"].astype(str).map(slot_group)
    meals["target_is_restaurant_meal_num"] = meals.get("target_is_restaurant_meal", False).map(robust_bool)

    transition_cols = [
        "date",
        "day_of_week",
        "day_of_week_num",
        "is_weekend",
        "season",
        "true_weight_lb",
        "weight_velocity_7d_lb",
        "steps_day",
        "calorie_budget_kcal",
        "budget_minus_noom_food_calories_kcal",
        "dominant_meal_archetype",
        "y_next_weight_loss_flag",
        "y_next_weight_gain_flag",
        "y_next_weight_delta_lb",
    ]
    transitions = transitions[[c for c in transition_cols if c in transitions.columns]].copy()
    if not history.empty and "score" in history.columns:
        transitions = transitions.merge(
            history[["anchor_id", "score", "policy_band"]].rename(columns={"anchor_id": "date", "score": "temporal_loss_score"}),
            on="date",
            how="left",
        )
    else:
        transitions["temporal_loss_score"] = np.nan
        transitions["policy_band"] = ""

    rows: List[Dict] = []
    for date, day_meals in meals.groupby("date", sort=True):
        slots = set(day_meals["slot_group"].dropna().astype(str).tolist())
        has_lunch = "lunch" in slots
        has_dinner = "dinner" in slots
        has_snack = "snack" in slots
        if min_required_slots and not (has_lunch and has_dinner and has_snack):
            continue
        transition_row = transitions.loc[transitions["date"] == date]
        t = transition_row.iloc[0].to_dict() if not transition_row.empty else {}
        calories = float(day_meals["target_calories_kcal"].fillna(0.0).sum())
        restaurant_count = float(day_meals["target_is_restaurant_meal_num"].fillna(0.0).sum())
        meal_count = int(len(day_meals))
        archetype_counts = day_meals["target_meal_archetype_primary"].dropna().astype(str).value_counts()
        signature = "|".join(
            f"{slot}:{','.join(day_meals.loc[day_meals['time_slot_label'].astype(str) == slot, 'target_meal_archetype_primary'].dropna().astype(str).head(2).tolist())}"
            for slot in ["lunch", "afternoon_snack", "dinner", "evening_snack"]
            if not day_meals.loc[day_meals["time_slot_label"].astype(str) == slot].empty
        )
        rows.append(
            {
                "template_id": f"day_{date}",
                "source_date": date,
                "day_of_week": t.get("day_of_week", ""),
                "day_of_week_num": safe_float(t.get("day_of_week_num"), np.nan),
                "is_weekend": bool(robust_bool(t.get("is_weekend")) == 1.0),
                "season": str(t.get("season", "")),
                "meal_count": meal_count,
                "has_lunch": has_lunch,
                "has_dinner": has_dinner,
                "has_snack": has_snack,
                "total_kcal": calories,
                "budget_gap_kcal": safe_float(t.get("budget_minus_noom_food_calories_kcal"), np.nan),
                "steps_day": safe_float(t.get("steps_day"), np.nan),
                "restaurant_fraction": restaurant_count / max(meal_count, 1),
                "dominant_archetype": str(t.get("dominant_meal_archetype") or archetype_counts.index[0]),
                "top_archetype": str(archetype_counts.index[0]) if len(archetype_counts) else "unknown",
                "archetype_signature": signature,
                "slot_summary": observed_slot_summary(day_meals),
                "protein_g": safe_float(day_meals["target_protein_g"].sum(skipna=True), 0.0),
                "fiber_g": safe_float(day_meals["target_fiber_g"].sum(skipna=True), 0.0),
                "sodium_mg": safe_float(day_meals["target_sodium_mg"].sum(skipna=True), 0.0),
                "comfort_mean": safe_float(day_meals["target_comfort_food_score"].mean(skipna=True), 0.4),
                "fresh_mean": safe_float(day_meals["target_fresh_light_score"].mean(skipna=True), 0.4),
                "indulgence_mean": safe_float(day_meals["target_indulgence_score"].mean(skipna=True), 0.4),
                "historical_loss_flag": robust_bool(t.get("y_next_weight_loss_flag")),
                "historical_gain_flag": robust_bool(t.get("y_next_weight_gain_flag")),
                "historical_weight_delta_lb": safe_float(t.get("y_next_weight_delta_lb"), np.nan),
                "temporal_loss_score": safe_float(t.get("temporal_loss_score"), np.nan),
                "temporal_policy_band": str(t.get("policy_band", "")),
            }
        )

    actions = pd.DataFrame(rows)
    if actions.empty:
        raise ValueError("No day templates passed required meal-slot constraints.")

    actions["loss_support_raw"] = actions["temporal_loss_score"]
    missing_loss_score = actions["loss_support_raw"].isna()
    actions.loc[missing_loss_score, "loss_support_raw"] = actions.loc[missing_loss_score, "historical_loss_flag"]
    actions["loss_support_raw"] = actions["loss_support_raw"].fillna(actions["loss_support_raw"].median()).fillna(0.5)
    actions["gain_risk_raw"] = actions["historical_gain_flag"].fillna(0.0)
    actions["weight_delta_raw"] = actions["historical_weight_delta_lb"].fillna(actions["historical_weight_delta_lb"].median()).fillna(0.0)

    bounds = {
        "total_kcal_q05": float(actions["total_kcal"].quantile(0.05)),
        "total_kcal_q95": float(actions["total_kcal"].quantile(0.95)),
        "steps_q10": float(actions["steps_day"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.10)),
        "steps_q90": float(actions["steps_day"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.90)),
        "budget_gap_q10": float(actions["budget_gap_kcal"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.10)),
        "budget_gap_q90": float(actions["budget_gap_kcal"].replace([np.inf, -np.inf], np.nan).dropna().quantile(0.90)),
        "n_templates": int(len(actions)),
    }
    actions["template_frequency"] = actions["archetype_signature"].map(actions["archetype_signature"].value_counts()) / len(actions)
    actions["within_core_kcal_band"] = actions["total_kcal"].between(bounds["total_kcal_q05"], bounds["total_kcal_q95"])
    metadata = {
        "required_slots": list(REQUIRED_DAY_SLOTS),
        "source": "observed full-day templates from training/meal_decision_points.csv joined to daily transitions",
        "bounds": bounds,
        "templates_after_required_slot_filter": int(len(actions)),
    }
    return actions.reset_index(drop=True), metadata


def build_meal_action_library(tables: Dict[str, pd.DataFrame], min_archetype_count: int = 3) -> Tuple[pd.DataFrame, Dict]:
    meals = tables["meals"].copy()
    meals["slot_group"] = meals["time_slot_label"].astype(str).map(slot_group)
    for col in [
        "target_calories_kcal",
        "target_protein_g",
        "target_fiber_g",
        "target_comfort_food_score",
        "target_fresh_light_score",
        "target_indulgence_score",
    ]:
        meals[col] = pd.to_numeric(meals.get(col), errors="coerce")
    meals["target_is_restaurant_meal_num"] = meals.get("target_is_restaurant_meal", False).map(robust_bool)
    counts = meals.groupby(["time_slot_label", "target_meal_archetype_primary"]).size().rename("archetype_slot_count")
    meals = meals.merge(counts.reset_index(), on=["time_slot_label", "target_meal_archetype_primary"], how="left")
    meals = meals.loc[meals["archetype_slot_count"] >= min_archetype_count].copy()
    meals["meal_action_id"] = ["meal_" + str(i).zfill(5) for i in range(len(meals))]
    metadata = {
        "source": "observed meal records from training/meal_decision_points.csv",
        "min_archetype_count": int(min_archetype_count),
        "meal_actions": int(len(meals)),
    }
    return meals.reset_index(drop=True), metadata


def candidate_template_pool(actions: pd.DataFrame, target_date: pd.Timestamp, context: PlanningContext) -> pd.DataFrame:
    pool = actions.copy()
    is_weekend = bool(target_date.dayofweek >= 5)
    season = season_for_month(int(target_date.month))
    day_match = pool.loc[pool["is_weekend"] == is_weekend]
    if len(day_match) >= 12:
        pool = day_match
    season_match = pool.loc[pool["season"].astype(str) == season]
    if len(season_match) >= 8:
        pool = season_match
    core = pool.loc[pool["within_core_kcal_band"]]
    if len(core) >= 8:
        pool = core
    return pool


def score_day_template(row: pd.Series, target_date: pd.Timestamp, context: PlanningContext, bounds: Dict, perturbation: Dict | None = None) -> Dict[str, float]:
    perturbation = perturbation or {}
    step_delta = safe_float(perturbation.get("step_delta"), 0.0)
    recent_shift = safe_float(perturbation.get("recent_intake_shift_kcal"), 0.0)
    force_daytype = perturbation.get("force_daytype")
    force_season = perturbation.get("force_season")

    projected_steps = safe_float(row.get("steps_day"), context.recent_steps_mean or 0.0) + step_delta
    projected_kcal = safe_float(row.get("total_kcal"), context.recent_food_kcal_mean or 0.0) + recent_shift
    kcal_food_reward = minmax_score(projected_kcal, bounds["total_kcal_q05"], bounds["total_kcal_q95"])
    budget_support = minmax_score(safe_float(row.get("budget_gap_kcal"), 0.0) - recent_shift, bounds["budget_gap_q10"], bounds["budget_gap_q90"])
    step_support = minmax_score(projected_steps, bounds["steps_q10"], bounds["steps_q90"])

    enjoyment = float(
        np.clip(
            0.40 * kcal_food_reward
            + 0.25 * safe_float(row.get("comfort_mean"), 0.4)
            + 0.20 * safe_float(row.get("indulgence_mean"), 0.4)
            + 0.15 * minmax_score(safe_float(row.get("template_frequency"), 0.0), 0.0, 0.08),
            0.0,
            1.0,
        )
    )
    protein_density = safe_float(row.get("protein_g"), 0.0) / max(projected_kcal, 1.0)
    fiber_density = safe_float(row.get("fiber_g"), 0.0) / max(projected_kcal, 1.0)
    health = float(
        np.clip(
            0.35 * safe_float(row.get("fresh_mean"), 0.4)
            + 0.25 * minmax_score(protein_density, 0.015, 0.08)
            + 0.20 * minmax_score(fiber_density, 0.002, 0.018)
            + 0.20 * inverse_minmax_score(safe_float(row.get("restaurant_fraction"), 0.0), 0.0, 0.75),
            0.0,
            1.0,
        )
    )

    planned_is_weekend = bool(target_date.dayofweek >= 5) if force_daytype is None else force_daytype == "weekend"
    planned_season = season_for_month(int(target_date.month)) if not force_season else str(force_season)
    daytype_match = 1.0 if bool(row.get("is_weekend")) == planned_is_weekend else 0.0
    season_match = 1.0 if str(row.get("season")) == planned_season else 0.0
    recent_match = 1.0 if str(row.get("dominant_archetype")) in context.recent_dominant_archetypes else 0.4
    consistency = float(
        np.clip(
            0.30 * daytype_match
            + 0.15 * season_match
            + 0.20 * recent_match
            + 0.20 * minmax_score(safe_float(row.get("template_frequency"), 0.0), 0.0, 0.08)
            + 0.15 * (1.0 if row.get("has_lunch") and row.get("has_dinner") and row.get("has_snack") else 0.0),
            0.0,
            1.0,
        )
    )

    weight_support = float(
        np.clip(
            0.35 * safe_float(row.get("loss_support_raw"), 0.5)
            + 0.25 * (1.0 - safe_float(row.get("gain_risk_raw"), 0.0))
            + 0.20 * budget_support
            + 0.15 * step_support
            + 0.05 * inverse_minmax_score(safe_float(row.get("weight_delta_raw"), 0.0), -1.0, 1.0),
            0.0,
            1.0,
        )
    )
    realism = float(
        np.clip(
            0.45 * (1.0 if row.get("within_core_kcal_band") else 0.6)
            + 0.25 * minmax_score(safe_float(row.get("template_frequency"), 0.0), 0.0, 0.08)
            + 0.15 * (1.0 if row.get("has_lunch") else 0.0)
            + 0.15 * (1.0 if row.get("has_dinner") and row.get("has_snack") else 0.0),
            0.0,
            1.0,
        )
    )
    combined = float(
        0.22 * enjoyment
        + 0.20 * health
        + 0.20 * consistency
        + 0.28 * weight_support
        + 0.10 * realism
    )
    return {
        "combined": combined,
        "enjoyment": enjoyment,
        "health": health,
        "consistency": consistency,
        "weight_support": weight_support,
        "realism": realism,
    }


def generate_candidate_plan(
    actions: pd.DataFrame,
    context: PlanningContext,
    horizon: int,
    rng: np.random.Generator,
    strategy: str,
    bounds: Dict,
) -> pd.DataFrame:
    chosen_rows: List[pd.Series] = []
    used_signatures: List[str] = []
    for offset in range(horizon):
        target_date = context.start_date + pd.Timedelta(days=offset)
        pool = candidate_template_pool(actions, target_date, context).copy()
        scores = []
        for _, row in pool.iterrows():
            day_score = score_day_template(row, target_date, context, bounds)
            if strategy == "weight":
                s = 0.65 * day_score["weight_support"] + 0.20 * day_score["health"] + 0.15 * day_score["realism"]
            elif strategy == "enjoyment":
                s = 0.55 * day_score["enjoyment"] + 0.25 * day_score["consistency"] + 0.20 * day_score["weight_support"]
            elif strategy == "routine":
                s = 0.55 * day_score["consistency"] + 0.25 * day_score["realism"] + 0.20 * day_score["weight_support"]
            elif strategy == "health":
                s = 0.55 * day_score["health"] + 0.25 * day_score["weight_support"] + 0.20 * day_score["enjoyment"]
            else:
                s = day_score["combined"]
            if used_signatures and str(row["archetype_signature"]) == used_signatures[-1]:
                s -= 0.08
            if len(used_signatures) >= 2 and str(row["archetype_signature"]) == used_signatures[-2]:
                s -= 0.04
            scores.append(s)
        pool["_selection_score"] = scores
        pool = pool.sort_values("_selection_score", ascending=False).head(25)
        if strategy in {"balanced", "routine", "weight", "health", "enjoyment"}:
            pick = pool.iloc[min(offset % max(min(len(pool), 5), 1), len(pool) - 1)]
        else:
            weights = np.exp((pool["_selection_score"].to_numpy() - pool["_selection_score"].max()) * 8.0)
            weights = weights / weights.sum()
            pick = pool.iloc[int(rng.choice(np.arange(len(pool)), p=weights))]
        chosen_rows.append(pick.drop(labels=["_selection_score"], errors="ignore"))
        used_signatures.append(str(pick["archetype_signature"]))
    plan = pd.DataFrame(chosen_rows).reset_index(drop=True)
    plan["planned_date"] = [(context.start_date + pd.Timedelta(days=i)).date().isoformat() for i in range(horizon)]
    plan["planned_day_of_week"] = [(context.start_date + pd.Timedelta(days=i)).day_name() for i in range(horizon)]
    return plan


def plan_rejection_reasons(plan: pd.DataFrame, horizon: int, bounds: Dict) -> List[str]:
    reasons: List[str] = []
    if len(plan) != horizon:
        reasons.append("incomplete_horizon")
    for slot in ["has_lunch", "has_dinner", "has_snack"]:
        if slot in plan.columns and not bool(plan[slot].all()):
            reasons.append(f"missing_{slot.replace('has_', '')}")
    if plan["total_kcal"].lt(bounds["total_kcal_q05"]).any() or plan["total_kcal"].gt(bounds["total_kcal_q95"]).any():
        reasons.append("outside_core_calorie_band")
    template_ids = plan["template_id"].astype(str).tolist()
    if any(template_ids[i] == template_ids[i - 1] for i in range(1, len(template_ids))):
        reasons.append("consecutive_duplicate_day_template")
    if horizon >= 5:
        max_signature_share = plan["archetype_signature"].value_counts(normalize=True).max()
        if max_signature_share > 0.45:
            reasons.append("repeat_frequency_too_high")
    return reasons


def score_plan(plan: pd.DataFrame, context: PlanningContext, bounds: Dict, perturbation: Dict | None = None) -> Dict[str, float]:
    day_scores: List[Dict[str, float]] = []
    for i, (_, row) in enumerate(plan.iterrows()):
        target_date = context.start_date + pd.Timedelta(days=i)
        day_scores.append(score_day_template(row, target_date, context, bounds, perturbation=perturbation))
    keys = ["combined", "enjoyment", "health", "consistency", "weight_support", "realism"]
    out = {k: float(np.mean([d[k] for d in day_scores])) for k in keys}
    out["mean_kcal"] = float(plan["total_kcal"].mean())
    out["mean_steps"] = float(plan["steps_day"].replace([np.inf, -np.inf], np.nan).mean())
    out["restaurant_fraction"] = float(plan["restaurant_fraction"].mean())
    out["mean_loss_support"] = float(plan["loss_support_raw"].mean())
    out["mean_gain_risk"] = float(plan["gain_risk_raw"].mean())
    return out


def robustness_perturbations(context: PlanningContext) -> List[Dict]:
    adjacent_season = {
        "winter": "spring",
        "spring": "summer",
        "summer": "fall",
        "fall": "winter",
    }.get(context.season, context.season)
    return [
        {"name": "base", "step_delta": 0, "recent_intake_shift_kcal": 0},
        {"name": "low_steps", "step_delta": -2000, "recent_intake_shift_kcal": 0},
        {"name": "high_steps", "step_delta": 2000, "recent_intake_shift_kcal": 0},
        {"name": "recent_heavier_intake", "step_delta": 0, "recent_intake_shift_kcal": 250},
        {"name": "recent_lighter_intake", "step_delta": 0, "recent_intake_shift_kcal": -200},
        {"name": "weekend_like", "force_daytype": "weekend", "step_delta": -750, "recent_intake_shift_kcal": 100},
        {"name": "weekday_like", "force_daytype": "weekday", "step_delta": 500, "recent_intake_shift_kcal": -50},
        {"name": "adjacent_season", "force_season": adjacent_season, "step_delta": -500, "recent_intake_shift_kcal": 100},
    ]


def robust_score_plan(plan: pd.DataFrame, context: PlanningContext, bounds: Dict) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows: List[Dict] = []
    for perturbation in robustness_perturbations(context):
        row = {"perturbation": perturbation["name"]}
        row.update(score_plan(plan, context, bounds, perturbation=perturbation))
        rows.append(row)
    stress = pd.DataFrame(rows)
    base = stress.loc[stress["perturbation"] == "base"].iloc[0].to_dict()
    combined_values = stress["combined"].to_numpy(dtype=float)
    weight_values = stress["weight_support"].to_numpy(dtype=float)
    robust_score = float(np.clip(0.80 * np.mean(combined_values) + 0.20 * np.min(combined_values) - 0.50 * np.std(combined_values), 0.0, 1.0))
    robust_weight_support = float(np.clip(np.mean(weight_values) - 0.50 * np.std(weight_values), 0.0, 1.0))
    fragility = float(np.max(combined_values) - np.min(combined_values))
    out = {
        **{k: safe_float(v) for k, v in base.items() if k != "perturbation"},
        "robust_score": robust_score,
        "robust_weight_support": robust_weight_support,
        "fragility": fragility,
        "stress_min_combined": float(np.min(combined_values)),
        "stress_max_combined": float(np.max(combined_values)),
    }
    return out, stress


def build_scenario_search(
    actions: pd.DataFrame,
    context: PlanningContext,
    horizons: Iterable[int],
    candidates_per_horizon: int,
    seed: int,
    metadata: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bounds = metadata["bounds"]
    strategy_cycle = ["balanced", "weight", "health", "routine", "enjoyment", "sampled"]
    ranking_rows: List[Dict] = []
    plan_rows: List[pd.DataFrame] = []
    stress_rows: List[pd.DataFrame] = []
    plan_counter = 0
    for horizon in horizons:
        seen_signatures = set()
        attempts = max(candidates_per_horizon * 3, 24)
        for attempt in range(attempts):
            strategy = strategy_cycle[attempt % len(strategy_cycle)]
            plan = generate_candidate_plan(actions, context, int(horizon), rng, strategy, bounds)
            signature = tuple(plan["template_id"].tolist())
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            rejection = plan_rejection_reasons(plan, int(horizon), bounds)
            score, stress = robust_score_plan(plan, context, bounds)
            if score["robust_score"] < 0.50:
                rejection.append("robust_score_below_floor")
            if score["robust_weight_support"] < 0.45:
                rejection.append("robust_weight_support_below_floor")
            if score["fragility"] > 0.22:
                rejection.append("fragility_above_ceiling")
            promoted = not rejection
            plan_id = f"scenario_h{horizon}_{plan_counter:04d}"
            ranking_rows.append(
                {
                    "plan_id": plan_id,
                    "horizon_days": int(horizon),
                    "strategy": strategy,
                    "promoted": bool(promoted),
                    "rejection_reasons": ";".join(rejection) if rejection else "",
                    **score,
                }
            )
            detailed = plan.copy()
            detailed.insert(0, "plan_id", plan_id)
            detailed.insert(1, "horizon_days", int(horizon))
            detailed.insert(2, "day_index", np.arange(1, len(detailed) + 1))
            plan_rows.append(detailed)
            stress_detail = stress.copy()
            stress_detail.insert(0, "plan_id", plan_id)
            stress_detail.insert(1, "horizon_days", int(horizon))
            stress_rows.append(stress_detail)
            plan_counter += 1
            if sum(r["horizon_days"] == int(horizon) for r in ranking_rows) >= candidates_per_horizon:
                break
    rankings = pd.DataFrame(ranking_rows)
    rankings = rankings.sort_values(["horizon_days", "promoted", "robust_score"], ascending=[True, False, False])
    return rankings, pd.concat(plan_rows, ignore_index=True), pd.concat(stress_rows, ignore_index=True)


def infer_current_slot(current_dt: datetime) -> str:
    hour = current_dt.hour
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 16:
        return "lunch"
    if 16 <= hour < 20:
        return "dinner"
    return "evening_snack"


def score_next_meal_candidates(
    meal_actions: pd.DataFrame,
    day_actions: pd.DataFrame,
    context: PlanningContext,
    current_dt: datetime,
    top_n: int,
    bounds: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    slot = infer_current_slot(current_dt)
    slot_pool = meal_actions.loc[meal_actions["time_slot_label"].astype(str) == slot].copy()
    if slot_pool.empty and slot in SNACK_SLOTS:
        slot_pool = meal_actions.loc[meal_actions["slot_group"] == "snack"].copy()
    if slot_pool.empty:
        slot_pool = meal_actions.copy()
    slot_counts = slot_pool["target_meal_archetype_primary"].astype(str).value_counts()
    slot_pool["slot_archetype_frequency"] = slot_pool["target_meal_archetype_primary"].astype(str).map(slot_counts) / len(slot_pool)
    slot_kcal = pd.to_numeric(slot_pool["target_calories_kcal"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    slot_q10 = float(slot_kcal.quantile(0.10)) if not slot_kcal.empty else bounds["total_kcal_q05"] / 4.0
    slot_q50 = float(slot_kcal.quantile(0.50)) if not slot_kcal.empty else bounds["total_kcal_q95"] / 3.0
    slot_q90 = float(slot_kcal.quantile(0.90)) if not slot_kcal.empty else bounds["total_kcal_q95"] / 2.0
    slot_iqr_like = max(slot_q90 - slot_q10, 1.0)
    slot_pool = slot_pool.sort_values(["slot_archetype_frequency", "target_calories_kcal"], ascending=[False, False]).head(150)

    rows: List[Dict] = []
    projections: List[pd.DataFrame] = []
    for i, (_, meal) in enumerate(slot_pool.iterrows()):
        archetype = str(meal.get("target_meal_archetype_primary", "unknown"))
        matching_days = day_actions.loc[day_actions["archetype_signature"].astype(str).str.contains(archetype, regex=False)].copy()
        if matching_days.empty:
            matching_days = day_actions.loc[day_actions["top_archetype"].astype(str) == archetype].copy()
        if matching_days.empty:
            matching_days = day_actions.copy()
        matching_days["_meal_projection_score"] = [
            score_day_template(row, context.start_date, context, bounds)["combined"] for _, row in matching_days.iterrows()
        ]
        projection_template = matching_days.sort_values("_meal_projection_score", ascending=False).head(1).drop(columns=["_meal_projection_score"]).iloc[0]
        projected_plan = pd.DataFrame([projection_template])
        projected_plan["planned_date"] = [context.start_date.date().isoformat()]
        projected_plan["planned_day_of_week"] = [context.start_date.day_name()]
        score, stress = robust_score_plan(projected_plan, context, bounds)
        kcal = safe_float(meal.get("target_calories_kcal"), 0.0)
        kcal_fit = float(np.clip(1.0 - abs(kcal - slot_q50) / slot_iqr_like, 0.0, 1.0))
        high_kcal_pressure = minmax_score(kcal, slot_q50, slot_q90)
        meal_enjoyment = float(
            np.clip(
                0.25 * kcal_fit
                + 0.25 * safe_float(meal.get("target_comfort_food_score"), 0.4)
                + 0.20 * safe_float(meal.get("target_indulgence_score"), 0.4)
                + 0.30 * minmax_score(safe_float(meal.get("slot_archetype_frequency"), 0.0), 0.0, 0.25),
                0.0,
                1.0,
            )
        )
        meal_health = float(
            np.clip(
                0.40 * safe_float(meal.get("target_fresh_light_score"), 0.4)
                + 0.25 * minmax_score(safe_float(meal.get("target_protein_g"), 0.0) / max(kcal, 1.0), 0.015, 0.08)
                + 0.20 * minmax_score(safe_float(meal.get("target_fiber_g"), 0.0) / max(kcal, 1.0), 0.002, 0.018)
                + 0.15 * (1.0 - safe_float(meal.get("target_is_restaurant_meal_num"), 0.0)),
                0.0,
                1.0,
            )
        )
        next_action_score = float(
            0.30 * meal_enjoyment
            + 0.20 * meal_health
            + 0.25 * score["robust_weight_support"]
            + 0.15 * minmax_score(safe_float(meal.get("slot_archetype_frequency"), 0.0), 0.0, 0.25)
            + 0.10 * score["realism"]
            - 0.10 * score["fragility"]
            - 0.08 * high_kcal_pressure
        )
        rows.append(
            {
                "meal_action_id": str(meal["meal_action_id"]),
                "current_slot": slot,
                "source_date": str(meal.get("date")),
                "meal_text": str(meal.get("target_meal_text", ""))[:240],
                "archetype": archetype,
                "calories_kcal": kcal,
                "protein_g": safe_float(meal.get("target_protein_g"), 0.0),
                "fresh_light_score": safe_float(meal.get("target_fresh_light_score"), 0.0),
                "comfort_food_score": safe_float(meal.get("target_comfort_food_score"), 0.0),
                "indulgence_score": safe_float(meal.get("target_indulgence_score"), 0.0),
                "slot_archetype_frequency": safe_float(meal.get("slot_archetype_frequency"), 0.0),
                "slot_kcal_fit": kcal_fit,
                "high_kcal_pressure": high_kcal_pressure,
                "projection_template_id": str(projection_template["template_id"]),
                "projected_day_slot_summary": str(projection_template["slot_summary"]),
                "projected_robust_weight_support": score["robust_weight_support"],
                "projected_fragility": score["fragility"],
                "meal_enjoyment": meal_enjoyment,
                "meal_health": meal_health,
                "next_action_score": next_action_score,
            }
        )
        stress.insert(0, "meal_action_id", str(meal["meal_action_id"]))
        projections.append(stress)
    scored = pd.DataFrame(rows).sort_values("next_action_score", ascending=False).head(top_n).reset_index(drop=True)
    return scored, pd.concat(projections, ignore_index=True)


def markdown_table(df: pd.DataFrame, columns: List[str], max_rows: int = 10) -> str:
    if df.empty:
        return "_No rows._"
    display = df[[c for c in columns if c in df.columns]].head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)
