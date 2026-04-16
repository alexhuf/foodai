from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from meal_scenario_planning_core_v1 import (
    PlanningContext,
    build_day_action_library,
    build_meal_action_library,
    build_planning_context,
    candidate_template_pool,
    ensure_dir,
    infer_current_slot,
    load_source_tables,
    markdown_table,
    minmax_score,
    robust_score_plan,
    safe_float,
    save_json,
    score_day_template,
)


DEFAULT_HORIZONS = (3, 5, 7, 14, 30)


def _clean_tokens(text: str) -> List[str]:
    raw = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()
    stop = {"and", "with", "the", "of", "style", "foods", "food", "trader", "joe", "meijer"}
    return sorted(t for t in raw if len(t) > 2 and t not in stop)


def _cluster_key(row: pd.Series) -> str:
    entities = str(row.get("target_target_top_canonical_entities", "")).strip().lower()
    if entities and entities != "nan":
        tokens = sorted(t.strip() for t in re.split(r"\s*\|\s*", entities) if t.strip())
    else:
        tokens = _clean_tokens(str(row.get("target_meal_text", "")))[:8]
    archetype = str(row.get("target_meal_archetype_primary", "unknown"))
    service = str(row.get("target_service_form_primary", "unknown"))
    protein = str(row.get("target_principal_protein", "unknown"))
    if service not in {"", "nan", "unknown"} and protein not in {"", "nan", "unknown", "none"}:
        # Service form + protein intentionally collapses optional side-item variants
        # such as tacos with rice/beans versus tacos with chips.
        return "|".join([archetype, service, protein])
    primary = tokens[0] if tokens else "unknown"
    return "|".join([archetype, service, protein, primary])


def _repeat_limits(horizon: int) -> Dict[str, int | float]:
    if horizon <= 3:
        return {"template_max": 1, "signature_max_share": 0.34, "recent_signature_window": 2}
    if horizon <= 7:
        return {"template_max": 1, "signature_max_share": 0.30, "recent_signature_window": 3}
    if horizon <= 14:
        return {"template_max": 2, "signature_max_share": 0.24, "recent_signature_window": 4}
    return {"template_max": 3, "signature_max_share": 0.20, "recent_signature_window": 5}


def _variant_row(row: pd.Series, label: str, multiplier: float, note: str) -> Dict:
    out = row.to_dict()
    source_template_id = str(row.get("source_template_id", row.get("template_id")))
    out["source_template_id"] = source_template_id
    out["template_id"] = f"{source_template_id}__{label}"
    out["portion_variant"] = label
    out["portion_multiplier"] = float(multiplier)
    out["bounded_portion_note"] = note
    for col in ["total_kcal", "protein_g", "fiber_g", "sodium_mg"]:
        out[col] = safe_float(row.get(col), 0.0) * multiplier
    if "budget_gap_kcal" in out:
        out["budget_gap_kcal"] = safe_float(row.get("budget_gap_kcal"), 0.0) + safe_float(row.get("total_kcal"), 0.0) * (1.0 - multiplier)
    return out


def add_bounded_day_variants(actions: pd.DataFrame, bounds: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Add conservative portion variants while preserving observed archetype signatures."""
    base = actions.copy()
    base["source_template_id"] = base["template_id"].astype(str)
    base["portion_variant"] = "observed"
    base["portion_multiplier"] = 1.0
    base["bounded_portion_note"] = "exact observed day template"

    rows: List[Dict] = []
    grouped = base.groupby("archetype_signature")
    for _, row in base.iterrows():
        cluster = grouped.get_group(row["archetype_signature"])
        if len(cluster) < 3:
            cluster = base.loc[base["top_archetype"].astype(str) == str(row.get("top_archetype"))]
        kcal = safe_float(row.get("total_kcal"), 0.0)
        if len(cluster) >= 3 and kcal > 0:
            q25 = float(cluster["total_kcal"].quantile(0.25))
            q75 = float(cluster["total_kcal"].quantile(0.75))
            lower_target = max(q25, kcal * 0.88)
            upper_target = min(q75, kcal * 1.12)
            for label, target, direction in [
                ("lighter_observed_portion", lower_target, "lower"),
                ("heartier_observed_portion", upper_target, "higher"),
            ]:
                multiplier = float(np.clip(target / kcal, 0.88, 1.12))
                if abs(multiplier - 1.0) < 0.035:
                    continue
                projected_kcal = kcal * multiplier
                if not (bounds["total_kcal_q05"] <= projected_kcal <= bounds["total_kcal_q95"]):
                    continue
                note = (
                    f"same observed archetype pattern with {direction} portion level "
                    f"bounded to observed cluster kcal range ({q25:.0f}-{q75:.0f})"
                )
                rows.append(_variant_row(row, label, multiplier, note))

    variants = pd.DataFrame(rows)
    combined = pd.concat([base, variants], ignore_index=True) if not variants.empty else base
    combined["template_frequency"] = combined["archetype_signature"].map(base["archetype_signature"].value_counts()) / len(base)
    metadata = {
        "base_templates": int(len(base)),
        "bounded_variants": int(len(variants)),
        "total_day_actions_v2": int(len(combined)),
        "portion_multiplier_min": float(combined["portion_multiplier"].min()),
        "portion_multiplier_max": float(combined["portion_multiplier"].max()),
        "constraint": "variants only scale observed full-day templates within observed archetype-signature calorie ranges",
    }
    return combined.reset_index(drop=True), metadata


def _strategy_score(day_score: Dict[str, float], strategy: str) -> float:
    if strategy == "weight":
        return 0.65 * day_score["weight_support"] + 0.20 * day_score["health"] + 0.15 * day_score["realism"]
    if strategy == "enjoyment":
        return 0.55 * day_score["enjoyment"] + 0.25 * day_score["consistency"] + 0.20 * day_score["weight_support"]
    if strategy == "routine":
        return 0.55 * day_score["consistency"] + 0.25 * day_score["realism"] + 0.20 * day_score["weight_support"]
    if strategy == "health":
        return 0.55 * day_score["health"] + 0.25 * day_score["weight_support"] + 0.20 * day_score["enjoyment"]
    if strategy == "variety":
        return 0.40 * day_score["combined"] + 0.25 * day_score["health"] + 0.20 * day_score["enjoyment"] + 0.15 * day_score["realism"]
    return day_score["combined"]


def generate_candidate_plan_v2(
    actions: pd.DataFrame,
    context: PlanningContext,
    horizon: int,
    rng: np.random.Generator,
    strategy: str,
    bounds: Dict,
) -> pd.DataFrame:
    chosen_rows: List[pd.Series] = []
    used_sources: List[str] = []
    used_signatures: List[str] = []
    limits = _repeat_limits(horizon)
    max_signature_count = max(1, int(math.ceil(horizon * float(limits["signature_max_share"]))))
    for offset in range(horizon):
        target_date = context.start_date + pd.Timedelta(days=offset)
        pool = candidate_template_pool(actions, target_date, context).copy()
        scored = []
        for _, row in pool.iterrows():
            source_id = str(row.get("source_template_id", row.get("template_id")))
            signature = str(row["archetype_signature"])
            day_score = score_day_template(row, target_date, context, bounds)
            s = _strategy_score(day_score, strategy)
            source_count = used_sources.count(source_id)
            signature_count = used_signatures.count(signature)
            recent_window = int(limits["recent_signature_window"])
            if source_count >= int(limits["template_max"]):
                s -= 0.45 + 0.10 * source_count
            if signature in used_signatures[-recent_window:]:
                s -= 0.16
            if signature_count >= max_signature_count:
                s -= 0.30 + 0.06 * signature_count
            if row.get("portion_variant") != "observed":
                s -= 0.025
            scored.append(s)
        pool["_selection_score"] = scored
        pool = pool.sort_values("_selection_score", ascending=False).head(35)
        if strategy == "sampled":
            weights = np.exp((pool["_selection_score"].to_numpy() - pool["_selection_score"].max()) * 7.0)
            weights = weights / weights.sum()
            pick = pool.iloc[int(rng.choice(np.arange(len(pool)), p=weights))]
        else:
            pick = pool.iloc[min(offset % max(min(len(pool), 7), 1), len(pool) - 1)]
        chosen_rows.append(pick.drop(labels=["_selection_score"], errors="ignore"))
        used_sources.append(str(pick.get("source_template_id", pick.get("template_id"))))
        used_signatures.append(str(pick["archetype_signature"]))
    plan = pd.DataFrame(chosen_rows).reset_index(drop=True)
    plan["planned_date"] = [(context.start_date + pd.Timedelta(days=i)).date().isoformat() for i in range(horizon)]
    plan["planned_day_of_week"] = [(context.start_date + pd.Timedelta(days=i)).day_name() for i in range(horizon)]
    plan["repeat_window_rule"] = f"source_template_max={limits['template_max']}; signature_recent_window={limits['recent_signature_window']}"
    return plan


def plan_rejection_reasons_v2(plan: pd.DataFrame, horizon: int, bounds: Dict) -> List[str]:
    reasons: List[str] = []
    if len(plan) != horizon:
        reasons.append("incomplete_horizon")
    for slot in ["has_lunch", "has_dinner", "has_snack"]:
        if slot in plan.columns and not bool(plan[slot].all()):
            reasons.append(f"missing_{slot.replace('has_', '')}")
    if plan["total_kcal"].lt(bounds["total_kcal_q05"]).any() or plan["total_kcal"].gt(bounds["total_kcal_q95"]).any():
        reasons.append("outside_core_calorie_band")
    limits = _repeat_limits(horizon)
    source_ids = plan.get("source_template_id", plan["template_id"]).astype(str)
    max_source_count = source_ids.value_counts().max()
    if max_source_count > int(limits["template_max"]):
        reasons.append("source_template_repeat_above_horizon_limit")
    max_signature_share = plan["archetype_signature"].value_counts(normalize=True).max()
    if max_signature_share > float(limits["signature_max_share"]) + 0.001:
        reasons.append("archetype_signature_repeat_above_horizon_limit")
    recent_window = int(limits["recent_signature_window"])
    signatures = plan["archetype_signature"].astype(str).tolist()
    for i, sig in enumerate(signatures):
        if sig in signatures[max(0, i - recent_window) : i]:
            reasons.append("archetype_signature_repeated_inside_recent_window")
            break
    return reasons


def plan_repeat_diagnostics(plan: pd.DataFrame) -> Dict[str, float | int]:
    source_ids = plan.get("source_template_id", plan["template_id"]).astype(str)
    return {
        "unique_source_templates": int(source_ids.nunique()),
        "unique_archetype_signatures": int(plan["archetype_signature"].nunique()),
        "max_source_template_count": int(source_ids.value_counts().max()),
        "max_signature_share": float(plan["archetype_signature"].value_counts(normalize=True).max()),
        "bounded_variant_days": int((plan.get("portion_variant", "observed") != "observed").sum()),
    }


def explain_plan(plan: pd.DataFrame, score: Dict[str, float]) -> str:
    top = []
    metrics = [
        ("weight support", score.get("robust_weight_support", 0.0)),
        ("health", score.get("health", 0.0)),
        ("enjoyment", score.get("enjoyment", 0.0)),
        ("consistency", score.get("consistency", 0.0)),
    ]
    for name, value in sorted(metrics, key=lambda x: x[1], reverse=True)[:2]:
        top.append(f"{name} {value:.2f}")
    repeat = plan_repeat_diagnostics(plan)
    archetypes = ", ".join(plan["top_archetype"].astype(str).value_counts().head(3).index.tolist())
    return (
        f"Scores well because {', '.join(top)} while staying inside observed calorie bands. "
        f"Uses {repeat['unique_source_templates']} source templates and {repeat['unique_archetype_signatures']} archetype patterns; "
        f"main observed archetypes: {archetypes}."
    )


def explain_day(row: pd.Series, target_date: pd.Timestamp, context: PlanningContext, bounds: Dict) -> str:
    score = score_day_template(row, target_date, context, bounds)
    reasons = []
    if score["weight_support"] >= 0.75:
        reasons.append("strong projected weight support")
    if score["health"] >= 0.70:
        reasons.append("good health score")
    if score["enjoyment"] >= 0.75:
        reasons.append("high familiarity/enjoyment")
    if row.get("portion_variant") != "observed":
        reasons.append(str(row.get("bounded_portion_note")))
    if not reasons:
        reasons.append("balanced observed template")
    return "; ".join(reasons)


def build_scenario_search_v2(
    actions: pd.DataFrame,
    context: PlanningContext,
    horizons: Iterable[int],
    candidates_per_horizon: int,
    seed: int,
    metadata: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bounds = metadata["bounds"]
    strategy_cycle = ["balanced", "weight", "health", "variety", "routine", "enjoyment", "sampled"]
    ranking_rows: List[Dict] = []
    plan_rows: List[pd.DataFrame] = []
    stress_rows: List[pd.DataFrame] = []
    plan_counter = 0
    for horizon in horizons:
        seen_signatures = set()
        attempts = max(candidates_per_horizon * 4, 32)
        for attempt in range(attempts):
            strategy = strategy_cycle[attempt % len(strategy_cycle)]
            plan = generate_candidate_plan_v2(actions, context, int(horizon), rng, strategy, bounds)
            signature = tuple(plan.get("source_template_id", plan["template_id"]).astype(str).tolist())
            variant_sig = tuple(plan.get("portion_variant", pd.Series(["observed"] * len(plan))).astype(str).tolist())
            if (signature, variant_sig) in seen_signatures:
                continue
            seen_signatures.add((signature, variant_sig))
            rejection = plan_rejection_reasons_v2(plan, int(horizon), bounds)
            score, stress = robust_score_plan(plan, context, bounds)
            if score["robust_score"] < 0.50:
                rejection.append("robust_score_below_floor")
            if score["robust_weight_support"] < 0.45:
                rejection.append("robust_weight_support_below_floor")
            if score["fragility"] > 0.22:
                rejection.append("fragility_above_ceiling")
            promoted = not rejection
            plan_id = f"scenario_v2_h{horizon}_{plan_counter:04d}"
            repeat = plan_repeat_diagnostics(plan)
            ranking_rows.append(
                {
                    "plan_id": plan_id,
                    "horizon_days": int(horizon),
                    "strategy": strategy,
                    "promoted": bool(promoted),
                    "rejection_reasons": ";".join(dict.fromkeys(rejection)) if rejection else "",
                    **score,
                    **repeat,
                    "plain_language_explanation": explain_plan(plan, score),
                }
            )
            detailed = plan.copy()
            detailed.insert(0, "plan_id", plan_id)
            detailed.insert(1, "horizon_days", int(horizon))
            detailed.insert(2, "day_index", np.arange(1, len(detailed) + 1))
            detailed["day_explanation"] = [
                explain_day(row, context.start_date + pd.Timedelta(days=i), context, bounds)
                for i, (_, row) in enumerate(detailed.iterrows())
            ]
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


def score_next_meal_candidates_v2(
    meal_actions: pd.DataFrame,
    day_actions: pd.DataFrame,
    context: PlanningContext,
    current_dt: datetime,
    top_n: int,
    bounds: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from meal_scenario_planning_core_v1 import score_next_meal_candidates

    slot = infer_current_slot(current_dt)
    raw_top_n = max(top_n * 8, 80)
    raw_scored, raw_stress = score_next_meal_candidates(
        meal_actions=meal_actions,
        day_actions=day_actions,
        context=context,
        current_dt=current_dt,
        top_n=raw_top_n,
        bounds=bounds,
    )
    if raw_scored.empty:
        return raw_scored, raw_stress

    enriched = raw_scored.merge(
        meal_actions[
            [
                "meal_action_id",
                "target_service_form_primary",
                "target_principal_protein",
                "target_principal_starch",
                "target_principal_veg",
                "target_target_top_canonical_entities",
            ]
        ],
        on="meal_action_id",
        how="left",
    )
    enriched["meal_cluster_key"] = enriched.apply(_cluster_key, axis=1)
    cluster_rows: List[Dict] = []
    for cluster_key, group in enriched.groupby("meal_cluster_key", sort=False):
        group = group.sort_values("next_action_score", ascending=False)
        best = group.iloc[0].to_dict()
        best["meal_cluster_id"] = f"cluster_{len(cluster_rows):03d}"
        best["meal_cluster_key"] = cluster_key
        best["cluster_observed_examples"] = int(len(group))
        best["cluster_kcal_min"] = float(group["calories_kcal"].min())
        best["cluster_kcal_median"] = float(group["calories_kcal"].median())
        best["cluster_kcal_max"] = float(group["calories_kcal"].max())
        best["alternate_observed_examples"] = " || ".join(group["meal_text"].dropna().astype(str).head(3).tolist())
        best["portion_guidance"] = (
            f"Observed {slot} examples in this archetype cluster range "
            f"{best['cluster_kcal_min']:.0f}-{best['cluster_kcal_max']:.0f} kcal; choose a portion near "
            f"{best['cluster_kcal_median']:.0f} kcal unless already over budget."
        )
        best["plain_language_explanation"] = _explain_meal_cluster(best)
        cluster_rows.append(best)
    scored = pd.DataFrame(cluster_rows).sort_values("next_action_score", ascending=False).head(top_n).reset_index(drop=True)
    keep_ids = set(scored["meal_action_id"].astype(str))
    stress = raw_stress.loc[raw_stress["meal_action_id"].astype(str).isin(keep_ids)].copy()
    return scored, stress


def _explain_meal_cluster(row: Dict) -> str:
    reasons = []
    if safe_float(row.get("meal_health"), 0.0) >= 0.65:
        reasons.append("solid protein/health profile")
    if safe_float(row.get("meal_enjoyment"), 0.0) >= 0.70:
        reasons.append("familiar high-fit lunch pattern")
    if safe_float(row.get("projected_robust_weight_support"), 0.0) >= 0.70:
        reasons.append("links to day templates with robust weight support")
    if safe_float(row.get("projected_fragility"), 1.0) <= 0.10:
        reasons.append("low stress-test fragility")
    if not reasons:
        reasons.append("balanced observed option")
    return "; ".join(reasons)
