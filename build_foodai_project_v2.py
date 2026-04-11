
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SAMSUNG_PATTERNS = {
    "sleep_stage": "com.samsung.health.sleep_stage",
    "weight": "com.samsung.health.weight",
    "activity_day_summary": "com.samsung.shealth.activity.day_summary",
    "calories_burned_details": "com.samsung.shealth.calories_burned.details",
    "exercise": "com.samsung.shealth.exercise",
    "sleep": "com.samsung.shealth.sleep",
    "stress": "com.samsung.shealth.stress",
    "heart_rate": "com.samsung.shealth.tracker.heart_rate",
    "oxygen_saturation": "com.samsung.shealth.tracker.oxygen_saturation",
    "pedometer_day_summary": "com.samsung.shealth.tracker.pedometer_day_summary",
    "pedometer_step_count": "com.samsung.shealth.tracker.pedometer_step_count",
}

NOOM_PATTERNS = {
    "actions": "actions",
    "android_food_entries": "android_food_entries",
    "application_opens": "application_opens",
    "assignments": "assignments",
    "curriculum_program_state": "curriculumProgramState",
    "daily_calorie_budgets": "daily_calorie_budgets",
    "finish_day": "finish_day",
    "goals": "goals",
    "goals_api_state": "goalsApiState",
    "user_events": "user_events",
    "user_model": "user_model",
}

TIME_SLOT_LABELS = {
    0: "breakfast",
    1: "morning_snack",
    2: "lunch",
    3: "afternoon_snack",
    4: "dinner",
    5: "evening_snack",
}

TIME_SLOT_HOUR = {
    0: 8,
    1: 10,
    2: 13,
    3: 16,
    4: 19,
    5: 21,
}


def log(msg: str) -> None:
    print(f"[foodai] {msg}")


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def find_first_csv(base_dir: Path, startswith: str) -> Optional[Path]:
    candidates = []
    if base_dir.exists():
        candidates.extend(sorted(base_dir.glob(f"{startswith}*.csv")))
    return candidates[0] if candidates else None


def find_source_file(project_root: Path, source_name: str, pattern: str) -> Optional[Path]:
    preferred_dir = project_root / source_name
    path = find_first_csv(preferred_dir, pattern)
    if path is not None:
        return path
    return find_first_csv(project_root, pattern)


def clean_wall_clock_string(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    s = re.sub(r"\[[^\]]+\]", "", s)
    return s


def to_datetime_local(value: Any) -> pd.Timestamp:
    s = clean_wall_clock_string(value)
    if s is None:
        return pd.NaT
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_localize(None)
    return ts


def to_date(value: Any) -> pd.Timestamp:
    ts = to_datetime_local(value)
    if pd.isna(ts):
        return pd.NaT
    return ts.floor("D")


def kg_to_lb(value: Any) -> float:
    try:
        return float(value) * 2.20462
    except Exception:
        return np.nan


def read_samsung_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1, index_col=False, low_memory=False)


def safe_json_loads(value: Any) -> Any:
    if pd.isna(value):
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def split_top_level(text: str) -> List[str]:
    text = str(text).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    # Noom sometimes serializes Java byte arrays like [B@123abc without a matching closing bracket.
    # Those are scalar values, not nested lists, so normalize them before splitting.
    text = text.replace("[B@", "B@")

    items: List[str] = []
    buf: List[str] = []
    depth_curly = 0
    depth_square = 0
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            buf.append(ch)
            continue

        if ch == "{":
            depth_curly += 1
            buf.append(ch)
            continue
        if ch == "}":
            depth_curly = max(0, depth_curly - 1)
            buf.append(ch)
            continue
        if ch == "[":
            depth_square += 1
            buf.append(ch)
            continue
        if ch == "]":
            depth_square = max(0, depth_square - 1)
            buf.append(ch)
            continue

        if ch == "," and depth_curly == 0 and depth_square == 0:
            items.append("".join(buf).strip())
            buf = []
            continue

        buf.append(ch)

    if buf:
        items.append("".join(buf).strip())

    return items


def parse_entries_row(row: pd.Series) -> Dict[str, Any]:
    text = row.get("entries")
    if pd.isna(text):
        return {}

    out: Dict[str, Any] = {}
    for token in split_top_level(text):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_values_keys_row(row: pd.Series) -> Dict[str, Any]:
    keys = split_top_level(row.get("keys", ""))
    values = split_top_level(row.get("values", ""))
    out: Dict[str, Any] = {}
    for i, key in enumerate(keys):
        out[str(key).strip()] = values[i].strip() if i < len(values) else None
    return out


def parse_noom_export(raw: pd.DataFrame) -> pd.DataFrame:
    if "entries" in raw.columns and raw["entries"].notna().any():
        parsed_rows = [parse_entries_row(row) for _, row in raw.iterrows()]
        return pd.DataFrame(parsed_rows)
    parsed_rows = [parse_values_keys_row(row) for _, row in raw.iterrows()]
    return pd.DataFrame(parsed_rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def dedupe_same_day_measurements(df: pd.DataFrame, dt_col: str, value_col: str, key_cols: List[str]) -> pd.DataFrame:
    temp = df.copy()
    temp["_day"] = temp[dt_col].dt.floor("D")
    temp["_rounded_value"] = pd.to_numeric(temp[value_col], errors="coerce").round(4)
    sort_cols = [dt_col] + [c for c in key_cols if c in temp.columns]
    temp = temp.sort_values(sort_cols)
    temp = temp.drop_duplicates(subset=["_day", "_rounded_value"] + [c for c in key_cols if c in temp.columns], keep="last")
    return temp.drop(columns=["_day", "_rounded_value"], errors="ignore")


def build_samsung_weight_events(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    out = pd.DataFrame({
        "source": "samsung",
        "event_type": "weight",
        "datetime_local": df["create_time"].map(to_datetime_local),
        "date": df["create_time"].map(to_date),
        "time_offset": df.get("time_offset"),
        "weight_kg": pd.to_numeric(df.get("weight"), errors="coerce"),
        "weight_lb": pd.to_numeric(df.get("weight"), errors="coerce").map(kg_to_lb),
        "basal_metabolic_rate": pd.to_numeric(df.get("basal_metabolic_rate"), errors="coerce"),
        "body_fat_percent": pd.to_numeric(df.get("body_fat"), errors="coerce"),
        "skeletal_muscle_mass_kg": pd.to_numeric(df.get("skeletal_muscle_mass"), errors="coerce"),
        "deviceuuid": df.get("deviceuuid"),
        "datauuid": df.get("datauuid"),
    })
    out = out.dropna(subset=["datetime_local", "weight_kg"])
    out = dedupe_same_day_measurements(out, "datetime_local", "weight_kg", ["deviceuuid"])
    out = out.drop_duplicates(subset=["datetime_local", "weight_kg", "deviceuuid"])
    return out.sort_values("datetime_local").reset_index(drop=True)


def build_samsung_sleep_sessions(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    out = pd.DataFrame({
        "source": "samsung",
        "event_type": "sleep_session",
        "datetime_start_local": df["com.samsung.health.sleep.start_time"].map(to_datetime_local),
        "datetime_end_local": df["com.samsung.health.sleep.end_time"].map(to_datetime_local),
        "date": df["com.samsung.health.sleep.start_time"].map(to_date),
        "time_offset": df.get("com.samsung.health.sleep.time_offset"),
        "sleep_duration_ms": pd.to_numeric(df.get("sleep_duration"), errors="coerce"),
        "sleep_score": pd.to_numeric(df.get("sleep_score"), errors="coerce"),
        "efficiency": pd.to_numeric(df.get("efficiency"), errors="coerce"),
        "quality": pd.to_numeric(df.get("quality"), errors="coerce"),
        "mental_recovery": pd.to_numeric(df.get("mental_recovery"), errors="coerce"),
        "physical_recovery": pd.to_numeric(df.get("physical_recovery"), errors="coerce"),
        "sleep_latency_ms": pd.to_numeric(df.get("sleep_latency"), errors="coerce"),
        "total_rem_duration_ms": pd.to_numeric(df.get("total_rem_duration"), errors="coerce"),
        "total_light_duration_ms": pd.to_numeric(df.get("total_light_duration"), errors="coerce"),
        "deviceuuid": df.get("com.samsung.health.sleep.deviceuuid"),
        "datauuid": df.get("com.samsung.health.sleep.datauuid"),
    })
    out = out.dropna(subset=["datetime_start_local"]).drop_duplicates(subset=["datetime_start_local", "datetime_end_local", "deviceuuid"])
    return out.sort_values("datetime_start_local").reset_index(drop=True)


def build_samsung_sleep_stages(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    stage_map = {40001: "awake", 40002: "light", 40003: "deep", 40004: "rem"}
    out = pd.DataFrame({
        "source": "samsung",
        "event_type": "sleep_stage",
        "sleep_id": df.get("sleep_id"),
        "datetime_start_local": df["start_time"].map(to_datetime_local),
        "datetime_end_local": df["end_time"].map(to_datetime_local),
        "date": df["start_time"].map(to_date),
        "time_offset": df.get("time_offset"),
        "stage_code": pd.to_numeric(df.get("stage"), errors="coerce"),
        "stage_label": pd.to_numeric(df.get("stage"), errors="coerce").map(stage_map),
        "deviceuuid": df.get("deviceuuid"),
        "datauuid": df.get("datauuid"),
    })
    out = out.dropna(subset=["datetime_start_local", "stage_code"]).drop_duplicates(subset=["sleep_id", "datetime_start_local", "stage_code"])
    return out.sort_values("datetime_start_local").reset_index(drop=True)


def build_samsung_exercise_sessions(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    out = pd.DataFrame({
        "source": "samsung",
        "event_type": "exercise",
        "datetime_start_local": df["com.samsung.health.exercise.start_time"].map(to_datetime_local),
        "datetime_end_local": df["com.samsung.health.exercise.end_time"].map(to_datetime_local),
        "date": df["com.samsung.health.exercise.start_time"].map(to_date),
        "time_offset": df.get("com.samsung.health.exercise.time_offset"),
        "exercise_type": pd.to_numeric(df.get("com.samsung.health.exercise.exercise_type"), errors="coerce"),
        "duration_ms": pd.to_numeric(df.get("com.samsung.health.exercise.duration"), errors="coerce"),
        "distance_m": pd.to_numeric(df.get("com.samsung.health.exercise.distance"), errors="coerce"),
        "calorie_kcal": pd.to_numeric(df.get("com.samsung.health.exercise.calorie"), errors="coerce"),
        "mean_hr": pd.to_numeric(df.get("com.samsung.health.exercise.mean_heart_rate"), errors="coerce"),
        "max_hr": pd.to_numeric(df.get("com.samsung.health.exercise.max_heart_rate"), errors="coerce"),
        "min_hr": pd.to_numeric(df.get("com.samsung.health.exercise.min_heart_rate"), errors="coerce"),
        "mean_speed_mps": pd.to_numeric(df.get("com.samsung.health.exercise.mean_speed"), errors="coerce"),
        "deviceuuid": df.get("com.samsung.health.exercise.deviceuuid"),
        "datauuid": df.get("com.samsung.health.exercise.datauuid"),
    })
    out = out.dropna(subset=["datetime_start_local"]).drop_duplicates(subset=["datetime_start_local", "datetime_end_local", "exercise_type", "deviceuuid"])
    return out.sort_values("datetime_start_local").reset_index(drop=True)


def build_samsung_hr_raw(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    out = pd.DataFrame({
        "datetime_local": df["com.samsung.health.heart_rate.start_time"].map(to_datetime_local),
        "date": df["com.samsung.health.heart_rate.start_time"].map(to_date),
        "time_offset": df.get("com.samsung.health.heart_rate.time_offset"),
        "heart_rate_bpm": pd.to_numeric(df.get("com.samsung.health.heart_rate.heart_rate"), errors="coerce"),
        "deviceuuid": df.get("com.samsung.health.heart_rate.deviceuuid"),
        "datauuid": df.get("com.samsung.health.heart_rate.datauuid"),
    })
    out["source"] = "samsung"
    out["event_type"] = "heart_rate"
    out = out.dropna(subset=["datetime_local", "heart_rate_bpm"]).drop_duplicates(subset=["datetime_local", "heart_rate_bpm", "deviceuuid"])
    return out.sort_values("datetime_local").reset_index(drop=True)


def build_samsung_stress_raw(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    out = pd.DataFrame({
        "datetime_local": df["start_time"].map(to_datetime_local),
        "date": df["start_time"].map(to_date),
        "time_offset": df.get("time_offset"),
        "stress_score": pd.to_numeric(df.get("score"), errors="coerce"),
        "deviceuuid": df.get("deviceuuid"),
        "datauuid": df.get("datauuid"),
    })
    out["source"] = "samsung"
    out["event_type"] = "stress"
    out = out.dropna(subset=["datetime_local", "stress_score"]).drop_duplicates(subset=["datetime_local", "stress_score", "deviceuuid"])
    return out.sort_values("datetime_local").reset_index(drop=True)


def build_samsung_oxygen_raw(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    out = pd.DataFrame({
        "datetime_local": df["com.samsung.health.oxygen_saturation.start_time"].map(to_datetime_local),
        "date": df["com.samsung.health.oxygen_saturation.start_time"].map(to_date),
        "time_offset": df.get("com.samsung.health.oxygen_saturation.time_offset"),
        "spo2_percent": pd.to_numeric(df.get("com.samsung.health.oxygen_saturation.spo2"), errors="coerce"),
        "heart_rate_bpm": pd.to_numeric(df.get("com.samsung.health.oxygen_saturation.heart_rate"), errors="coerce"),
        "deviceuuid": df.get("com.samsung.health.oxygen_saturation.deviceuuid"),
        "datauuid": df.get("com.samsung.health.oxygen_saturation.datauuid"),
    })
    out["source"] = "samsung"
    out["event_type"] = "oxygen_saturation"
    out = out.dropna(subset=["datetime_local", "spo2_percent"]).drop_duplicates(subset=["datetime_local", "spo2_percent", "deviceuuid"])
    return out.sort_values("datetime_local").reset_index(drop=True)


def build_samsung_steps_intraday(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    out = pd.DataFrame({
        "source": "samsung",
        "event_type": "steps_intraday",
        "datetime_start_local": df["com.samsung.health.step_count.start_time"].map(to_datetime_local),
        "datetime_end_local": df["com.samsung.health.step_count.end_time"].map(to_datetime_local),
        "date": df["com.samsung.health.step_count.start_time"].map(to_date),
        "time_offset": df.get("com.samsung.health.step_count.time_offset"),
        "step_count": pd.to_numeric(df.get("com.samsung.health.step_count.count"), errors="coerce"),
        "walk_step_count": pd.to_numeric(df.get("walk_step"), errors="coerce"),
        "run_step_count": pd.to_numeric(df.get("run_step"), errors="coerce"),
        "distance_m": pd.to_numeric(df.get("com.samsung.health.step_count.distance"), errors="coerce"),
        "step_calorie_kcal": pd.to_numeric(df.get("com.samsung.health.step_count.calorie"), errors="coerce"),
        "speed_mps": pd.to_numeric(df.get("com.samsung.health.step_count.speed"), errors="coerce"),
        "duration_ms": pd.to_numeric(df.get("duration"), errors="coerce"),
        "deviceuuid": df.get("com.samsung.health.step_count.deviceuuid"),
        "datauuid": df.get("com.samsung.health.step_count.datauuid"),
    })
    out = out.dropna(subset=["datetime_start_local"]).drop_duplicates(subset=["datetime_start_local", "step_count", "deviceuuid"])
    return out.sort_values("datetime_start_local").reset_index(drop=True)


def build_samsung_pedometer_daily(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    out = pd.DataFrame({
        "date": pd.to_datetime(pd.to_numeric(df.get("day_time"), errors="coerce"), unit="ms", errors="coerce").dt.floor("D"),
        "step_count": pd.to_numeric(df.get("step_count"), errors="coerce"),
        "walk_step_count": pd.to_numeric(df.get("walk_step_count"), errors="coerce"),
        "run_step_count": pd.to_numeric(df.get("run_step_count"), errors="coerce"),
        "distance_m": pd.to_numeric(df.get("distance"), errors="coerce"),
        "calorie_kcal": pd.to_numeric(df.get("calorie"), errors="coerce"),
        "active_time_ms": pd.to_numeric(df.get("active_time"), errors="coerce"),
        "speed_mps": pd.to_numeric(df.get("speed"), errors="coerce"),
        "healthy_step": pd.to_numeric(df.get("healthy_step"), errors="coerce"),
    }).dropna(subset=["date"])
    out = out.groupby("date", as_index=False).agg({
        "step_count": "max",
        "walk_step_count": "max",
        "run_step_count": "max",
        "distance_m": "max",
        "calorie_kcal": "max",
        "active_time_ms": "max",
        "speed_mps": "mean",
        "healthy_step": "max",
    })
    out["source"] = "samsung"
    return out.sort_values("date").reset_index(drop=True)


def build_samsung_activity_daily(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    out = pd.DataFrame({
        "date": df["day_time"].map(to_date),
        "step_count": pd.to_numeric(df.get("step_count"), errors="coerce"),
        "distance_m": pd.to_numeric(df.get("distance"), errors="coerce"),
        "calorie_kcal": pd.to_numeric(df.get("calorie"), errors="coerce"),
        "active_time_ms": pd.to_numeric(df.get("active_time"), errors="coerce"),
        "walk_time_ms": pd.to_numeric(df.get("walk_time"), errors="coerce"),
        "run_time_ms": pd.to_numeric(df.get("run_time"), errors="coerce"),
        "longest_active_time_ms": pd.to_numeric(df.get("longest_active_time"), errors="coerce"),
        "longest_idle_time_ms": pd.to_numeric(df.get("longest_idle_time"), errors="coerce"),
        "score": pd.to_numeric(df.get("score"), errors="coerce"),
        "floor_count": pd.to_numeric(df.get("floor_count"), errors="coerce"),
    }).dropna(subset=["date"])
    out = out.groupby("date", as_index=False).agg({
        "step_count": "max",
        "distance_m": "max",
        "calorie_kcal": "max",
        "active_time_ms": "max",
        "walk_time_ms": "max",
        "run_time_ms": "max",
        "longest_active_time_ms": "max",
        "longest_idle_time_ms": "max",
        "score": "mean",
        "floor_count": "max",
    })
    out["source"] = "samsung"
    return out.sort_values("date").reset_index(drop=True)


def build_samsung_energy_daily(path: Path) -> pd.DataFrame:
    df = read_samsung_csv(path)
    out = pd.DataFrame({
        "date": pd.to_datetime(pd.to_numeric(df.get("com.samsung.shealth.calories_burned.day_time"), errors="coerce"), unit="ms", errors="coerce").dt.floor("D"),
        "rest_calorie_kcal": pd.to_numeric(df.get("com.samsung.shealth.calories_burned.rest_calorie"), errors="coerce"),
        "active_calorie_kcal": pd.to_numeric(df.get("com.samsung.shealth.calories_burned.active_calorie"), errors="coerce"),
        "tef_calorie_kcal": pd.to_numeric(df.get("com.samsung.shealth.calories_burned.tef_calorie"), errors="coerce"),
        "active_time_ms": pd.to_numeric(df.get("com.samsung.shealth.calories_burned.active_time"), errors="coerce"),
        "exercise_calories": pd.to_numeric(df.get("exercise_calories"), errors="coerce"),
        "total_exercise_calories": pd.to_numeric(df.get("total_exercise_calories"), errors="coerce"),
    }).dropna(subset=["date"])
    out = out.groupby("date", as_index=False).agg({
        "rest_calorie_kcal": "max",
        "active_calorie_kcal": "max",
        "tef_calorie_kcal": "max",
        "active_time_ms": "max",
        "exercise_calories": "max",
        "total_exercise_calories": "max",
    })
    out["source"] = "samsung"
    return out.sort_values("date").reset_index(drop=True)


def build_noom_food_entries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    extra = df["extraDataJson"].map(safe_json_loads)
    nutrition = extra.map(lambda x: safe_json_loads(x.get("nutritionBreakdown")) if isinstance(x, dict) else None)

    out = pd.DataFrame({
        "source": "noom",
        "event_type": "food_entry",
        "uuid": df.get("uuid"),
        "user_id": df.get("userId"),
        "date": pd.to_datetime(df.get("dateConsumed"), errors="coerce").dt.floor("D"),
        "time_slot": pd.to_numeric(df.get("timeSlot"), errors="coerce"),
        "time_slot_label": pd.to_numeric(df.get("timeSlot"), errors="coerce").map(TIME_SLOT_LABELS),
        "datetime_local_approx": pd.to_datetime(df.get("dateConsumed"), errors="coerce").dt.floor("D")
            + pd.to_timedelta(pd.to_numeric(df.get("timeSlot"), errors="coerce").map(TIME_SLOT_HOUR).fillna(12), unit="h"),
        "client_time_inserted": df.get("clientTimeInserted").map(to_datetime_local),
        "food_type": df.get("foodType"),
        "food_category_code": df.get("foodCategoryCode"),
        "amount": df.get("amount"),
        "servings": pd.to_numeric(df.get("servings"), errors="coerce"),
        "calories_kcal": pd.to_numeric(df.get("calories"), errors="coerce"),
        "master_food_uuid": df.get("masterFoodUuid"),
        "custom_food_uuid": df.get("customFoodUuid"),
        "logged_name": extra.map(lambda x: x.get("name") if isinstance(x, dict) else None),
        "query_text": extra.map(lambda x: x.get("query") if isinstance(x, dict) else None),
        "unit_name": extra.map(lambda x: x.get("unitName") if isinstance(x, dict) else None),
        "protein_g": nutrition.map(lambda x: (x.get("protein") / 1000.0) if isinstance(x, dict) and x.get("protein") is not None else np.nan),
        "carbs_g": nutrition.map(lambda x: (x.get("carbohydrate") / 1000.0) if isinstance(x, dict) and x.get("carbohydrate") is not None else np.nan),
        "fat_g": nutrition.map(lambda x: (x.get("totalFat") / 1000.0) if isinstance(x, dict) and x.get("totalFat") is not None else np.nan),
        "fiber_g": nutrition.map(lambda x: (x.get("dietaryFiber") / 1000.0) if isinstance(x, dict) and x.get("dietaryFiber") is not None else np.nan),
        "sodium_mg": nutrition.map(lambda x: x.get("sodium") if isinstance(x, dict) else np.nan),
        "server_time_created": df.get("serverTimeCreated").map(to_datetime_local),
        "server_time_updated": df.get("serverTimeUpdated").map(to_datetime_local),
    })
    return out.dropna(subset=["date"]).sort_values(["date", "time_slot", "client_time_inserted", "server_time_created"]).reset_index(drop=True)


def build_noom_meal_events(food_entries: pd.DataFrame) -> pd.DataFrame:
    temp = food_entries.copy()
    temp["item_name"] = temp["logged_name"].fillna(temp["query_text"]).fillna("unknown_item")
    grp_cols = ["date", "time_slot", "time_slot_label", "datetime_local_approx"]
    meal = temp.groupby(grp_cols, dropna=False, as_index=False).agg({
        "calories_kcal": "sum",
        "protein_g": "sum",
        "carbs_g": "sum",
        "fat_g": "sum",
        "fiber_g": "sum",
        "sodium_mg": "sum",
        "uuid": "count",
        "item_name": lambda s: " + ".join(s.dropna().astype(str).tolist()),
    })
    meal = meal.rename(columns={"uuid": "item_count", "item_name": "meal_text"})
    meal["source"] = "noom"
    meal["event_type"] = "meal"
    return meal.sort_values(["date", "time_slot"]).reset_index(drop=True)


def build_noom_actions(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    payload = df["jsonString"].map(safe_json_loads)
    actions = pd.DataFrame({
        "uuid": df.get("uuid"),
        "action_type": df.get("actionType"),
        "date": pd.to_datetime(df.get("date"), errors="coerce").dt.floor("D"),
        "time_inserted": payload.map(lambda x: to_datetime_local(x.get("timeInserted")) if isinstance(x, dict) else pd.NaT),
        "time_updated": payload.map(lambda x: to_datetime_local(x.get("timeUpdated")) if isinstance(x, dict) else pd.NaT),
        "weight_kg": payload.map(lambda x: x.get("weightInKg") if isinstance(x, dict) else np.nan),
        "steps": payload.map(lambda x: x.get("steps") if isinstance(x, dict) else np.nan),
        "water_liters": payload.map(lambda x: x.get("amountDrankInLiters") if isinstance(x, dict) else np.nan),
        "source_platform": payload.map(lambda x: x.get("source", {}).get("platform") if isinstance(x, dict) else None),
        "source_product": payload.map(lambda x: x.get("source", {}).get("product") if isinstance(x, dict) else None),
        "attribution_type": payload.map(lambda x: x.get("attributionData", {}).get("type") if isinstance(x, dict) else None),
        "attribution_source_name": payload.map(lambda x: x.get("attributionData", {}).get("sourceName") if isinstance(x, dict) else None),
    })
    actions["source"] = "noom"

    weighins = actions[actions["action_type"] == "WEIGH_IN"].copy()
    weighins["weight_lb"] = weighins["weight_kg"].map(kg_to_lb)
    weighins["event_type"] = "weight"

    daily_steps = actions[actions["action_type"] == "DAILY_STEPS"].copy()
    daily_steps["event_type"] = "daily_steps"

    water = actions[actions["action_type"] == "WATER_LOGGING"].copy()
    water["event_type"] = "water"

    return actions, weighins, daily_steps, water


def build_noom_daily_budgets(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = parse_noom_export(raw)
    out = pd.DataFrame({
        "date": pd.to_datetime(df.get("date"), errors="coerce").dt.floor("D"),
        "calorie_budget_kcal": pd.to_numeric(df.get("calorieBudget"), errors="coerce"),
        "base_calorie_budget_kcal": pd.to_numeric(df.get("baseCalorieBudget"), errors="coerce"),
        "calories_to_lose_per_day_kcal": pd.to_numeric(df.get("caloriesToLosePerDay"), errors="coerce"),
        "calories_burned_kcal": pd.to_numeric(df.get("caloriesBurned"), errors="coerce"),
        "weight_loss_zone_lower_kcal": pd.to_numeric(df.get("weightLossZoneLowerBound"), errors="coerce"),
        "weight_loss_zone_upper_kcal": pd.to_numeric(df.get("weightLossZoneUpperBound"), errors="coerce"),
        "manual_calorie_adjustment_kcal": pd.to_numeric(df.get("manualCalorieAdjustment"), errors="coerce"),
        "client_time_inserted": df.get("clientTimeInserted").map(to_datetime_local),
        "client_time_updated": df.get("clientTimeUpdated").map(to_datetime_local),
        "client_timezone": df.get("clientTimeInsertedTimeZone"),
    })
    out["source"] = "noom"
    return out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)


def build_noom_finish_day(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = parse_noom_export(raw)
    out = pd.DataFrame({
        "date": pd.to_datetime(df.get("finishedDate"), errors="coerce").dt.floor("D"),
        "client_time_updated": df.get("clientTimeUpdated").map(to_datetime_local),
        "server_time_created": df.get("serverTimeCreated").map(to_datetime_local),
        "server_time_modified": df.get("serverTimeModified").map(to_datetime_local),
        "finished_day": 1,
    })
    out["source"] = "noom"
    return out.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)


def build_noom_assignments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    payload = df["jsonString"].map(safe_json_loads)
    out = pd.DataFrame({
        "uuid": df.get("uuid"),
        "assignment_type": df.get("assignmentType"),
        "start_date": pd.to_datetime(df.get("startDate"), errors="coerce").dt.floor("D"),
        "end_date": pd.to_datetime(df.get("endDate"), errors="coerce").dt.floor("D"),
        "score": pd.to_numeric(df.get("score"), errors="coerce"),
        "time_inserted": payload.map(lambda x: to_datetime_local(x.get("timeInserted")) if isinstance(x, dict) else pd.NaT),
        "time_updated": payload.map(lambda x: to_datetime_local(x.get("timeUpdated")) if isinstance(x, dict) else pd.NaT),
        "target_steps": payload.map(lambda x: x.get("targetSteps") if isinstance(x, dict) else np.nan),
        "target_water_liters": payload.map(lambda x: x.get("minWaterInLiters") if isinstance(x, dict) else np.nan),
    })
    out["source"] = "noom"
    return out.sort_values(["start_date", "assignment_type"]).reset_index(drop=True)


def build_noom_goals(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = parse_noom_export(raw)
    goal_json = df.get("goalJsonString").map(safe_json_loads) if "goalJsonString" in df.columns else pd.Series([None] * len(df))
    out = pd.DataFrame({
        "uuid": df.get("uuid"),
        "date": pd.to_datetime(df.get("date"), errors="coerce").dt.floor("D"),
        "time": df.get("time").map(to_datetime_local),
        "goal_type": df.get("goalType"),
        "score": pd.to_numeric(df.get("score"), errors="coerce"),
        "title": goal_json.map(lambda x: x.get("title") if isinstance(x, dict) else None),
        "content_type": goal_json.map(lambda x: x.get("contentType") if isinstance(x, dict) else None),
        "content_id": goal_json.map(lambda x: x.get("contentId") if isinstance(x, dict) else None),
        "goal_uri": goal_json.map(lambda x: x.get("goalUri") if isinstance(x, dict) else None),
        "completion_timestamp": goal_json.map(lambda x: to_datetime_local(x.get("goalStats", {}).get("completionTimestamp")) if isinstance(x, dict) else pd.NaT),
    })
    out["source"] = "noom"
    return out.sort_values(["date", "time"]).reset_index(drop=True)


def build_noom_app_opens(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "datetime_local": df.get("clientTime").map(to_datetime_local),
        "server_time_created": df.get("serverTimeCreated").map(to_datetime_local),
        "client_timezone": df.get("clientTimeZone"),
    })
    out["date"] = out["datetime_local"].dt.floor("D")
    out["source"] = "noom"
    out["event_type"] = "application_open"
    return out.dropna(subset=["datetime_local"]).sort_values("datetime_local").reset_index(drop=True)


def build_noom_user_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "uuid": df.get("uuid"),
        "type": df.get("type"),
        "payload": df.get("payload"),
        "datetime_local": df.get("timestamp").map(to_datetime_local),
        "server_time_created": df.get("serverTimeCreated").map(to_datetime_local),
    })
    out["date"] = out["datetime_local"].dt.floor("D")
    out["source"] = "noom"
    return out.dropna(subset=["datetime_local"]).sort_values("datetime_local").reset_index(drop=True)


def build_noom_curriculum_state(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["source"] = "noom"
    return df


def build_samsung_daily_features(
    samsung_weight_events: pd.DataFrame,
    samsung_sleep_sessions: pd.DataFrame,
    samsung_pedometer_daily: pd.DataFrame,
    samsung_activity_daily: pd.DataFrame,
    samsung_energy_daily: pd.DataFrame,
    samsung_exercise_sessions: pd.DataFrame,
) -> pd.DataFrame:
    weight_daily = samsung_weight_events.groupby("date", as_index=False).agg(
        samsung_weight_kg=("weight_kg", "mean"),
        samsung_weight_lb=("weight_lb", "mean"),
        samsung_weight_event_count=("weight_kg", "count"),
    )

    sleep_daily = samsung_sleep_sessions.groupby("date", as_index=False).agg(
        samsung_sleep_duration_ms=("sleep_duration_ms", "max"),
        samsung_sleep_score=("sleep_score", "max"),
        samsung_sleep_efficiency=("efficiency", "max"),
        samsung_mental_recovery=("mental_recovery", "max"),
        samsung_physical_recovery=("physical_recovery", "max"),
    )

    exercise_daily = samsung_exercise_sessions.groupby("date", as_index=False).agg(
        samsung_exercise_session_count=("event_type", "count"),
        samsung_exercise_duration_ms=("duration_ms", "sum"),
        samsung_exercise_distance_m=("distance_m", "sum"),
        samsung_exercise_calorie_kcal=("calorie_kcal", "sum"),
    )

    ped = samsung_pedometer_daily.rename(columns={
        "step_count": "samsung_pedometer_steps",
        "walk_step_count": "samsung_pedometer_walk_steps",
        "run_step_count": "samsung_pedometer_run_steps",
        "distance_m": "samsung_pedometer_distance_m",
        "calorie_kcal": "samsung_pedometer_calorie_kcal",
        "active_time_ms": "samsung_pedometer_active_time_ms",
        "speed_mps": "samsung_pedometer_speed_mps",
        "healthy_step": "samsung_pedometer_healthy_step",
    }).drop(columns=["source"], errors="ignore")

    act = samsung_activity_daily.rename(columns={
        "step_count": "samsung_activity_steps",
        "distance_m": "samsung_activity_distance_m",
        "calorie_kcal": "samsung_activity_calorie_kcal",
        "active_time_ms": "samsung_activity_active_time_ms",
        "walk_time_ms": "samsung_activity_walk_time_ms",
        "run_time_ms": "samsung_activity_run_time_ms",
        "longest_active_time_ms": "samsung_activity_longest_active_ms",
        "longest_idle_time_ms": "samsung_activity_longest_idle_ms",
        "score": "samsung_activity_score",
        "floor_count": "samsung_activity_floor_count",
    }).drop(columns=["source"], errors="ignore")

    energy = samsung_energy_daily.rename(columns={
        "rest_calorie_kcal": "samsung_rest_calorie_kcal",
        "active_calorie_kcal": "samsung_active_calorie_kcal",
        "tef_calorie_kcal": "samsung_tef_calorie_kcal",
        "active_time_ms": "samsung_energy_active_time_ms",
        "exercise_calories": "samsung_exercise_calories_field",
        "total_exercise_calories": "samsung_total_exercise_calories_field",
    }).drop(columns=["source"], errors="ignore")

    dfs = [weight_daily, sleep_daily, exercise_daily, ped, act, energy]
    out = None
    for df in dfs:
        out = df if out is None else out.merge(df, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True) if out is not None else pd.DataFrame(columns=["date"])


def build_noom_daily_features(
    noom_food_entries: pd.DataFrame,
    noom_meal_events: pd.DataFrame,
    noom_weighins: pd.DataFrame,
    noom_steps_daily: pd.DataFrame,
    noom_water_logs: pd.DataFrame,
    noom_daily_budgets: pd.DataFrame,
    noom_finish_day: pd.DataFrame,
    noom_app_opens: pd.DataFrame,
) -> pd.DataFrame:
    food_daily = noom_food_entries.groupby("date", as_index=False).agg(
        noom_food_entry_count=("uuid", "count"),
        noom_food_calories_kcal=("calories_kcal", "sum"),
        noom_food_protein_g=("protein_g", "sum"),
        noom_food_carbs_g=("carbs_g", "sum"),
        noom_food_fat_g=("fat_g", "sum"),
        noom_food_fiber_g=("fiber_g", "sum"),
        noom_food_sodium_mg=("sodium_mg", "sum"),
    )

    meal_daily = noom_meal_events.groupby("date", as_index=False).agg(
        noom_meal_event_count=("event_type", "count"),
        noom_mean_meal_calories_kcal=("calories_kcal", "mean"),
        noom_max_meal_calories_kcal=("calories_kcal", "max"),
    )

    weigh_daily = noom_weighins.groupby("date", as_index=False).agg(
        noom_weight_kg=("weight_kg", "mean"),
        noom_weight_lb=("weight_lb", "mean"),
        noom_weighin_count=("weight_kg", "count"),
    )

    steps_daily = noom_steps_daily.groupby("date", as_index=False).agg(noom_steps=("steps", "max"))
    water_daily = noom_water_logs.groupby("date", as_index=False).agg(noom_water_liters=("water_liters", "sum"))
    app_daily = noom_app_opens.groupby("date", as_index=False).agg(noom_app_open_count=("event_type", "count"))

    budget = noom_daily_budgets.drop_duplicates(subset=["date"], keep="last").copy().drop(columns=["source"], errors="ignore")
    finish = noom_finish_day.drop_duplicates(subset=["date"], keep="last")[["date", "finished_day"]].rename(columns={"finished_day": "noom_finished_day"})

    dfs = [food_daily, meal_daily, weigh_daily, steps_daily, water_daily, budget, finish, app_daily]
    out = None
    for df in dfs:
        out = df if out is None else out.merge(df, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True) if out is not None else pd.DataFrame(columns=["date"])


def add_weight_trends(master_daily: pd.DataFrame) -> pd.DataFrame:
    out = master_daily.sort_values("date").copy()
    out["true_weight_lb"] = out["noom_weight_lb"].combine_first(out["samsung_weight_lb"])
    for window in [3, 5, 7, 14, 30]:
        ema_col = f"weight_ema_{window}d_lb"
        vel_col = f"weight_velocity_{window}d_lb"
        out[ema_col] = out["true_weight_lb"].ewm(span=window, adjust=False).mean()
        out[vel_col] = out[ema_col].diff()
    return out


def build_master_daily_features_full(samsung_daily_features: pd.DataFrame, noom_daily_features: pd.DataFrame) -> pd.DataFrame:
    out = samsung_daily_features.merge(noom_daily_features, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    return add_weight_trends(out)


def apply_daily_spine(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    spine = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date, freq="D")})
    out = spine.merge(df, on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def build_master_event_ledger_active(
    noom_meal_events: pd.DataFrame,
    noom_weighins: pd.DataFrame,
    noom_app_opens: pd.DataFrame,
    samsung_weight_events: pd.DataFrame,
    samsung_exercise_sessions: pd.DataFrame,
    samsung_sleep_sessions: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    parts = []

    meal = noom_meal_events.copy()
    meal["datetime_local"] = meal["datetime_local_approx"]
    meal["metric_1"] = meal["calories_kcal"]
    meal["metric_1_name"] = "calories_kcal"
    meal["details"] = meal["meal_text"]
    parts.append(meal[["source", "event_type", "datetime_local", "date", "metric_1", "metric_1_name", "details"]])

    nw = noom_weighins.copy()
    nw["datetime_local"] = nw["time_inserted"].fillna(nw["date"])
    nw["metric_1"] = nw["weight_lb"]
    nw["metric_1_name"] = "weight_lb"
    nw["details"] = nw["attribution_source_name"]
    parts.append(nw[["source", "event_type", "datetime_local", "date", "metric_1", "metric_1_name", "details"]])

    app = noom_app_opens.copy()
    app["metric_1"] = 1
    app["metric_1_name"] = "app_open"
    app["details"] = None
    parts.append(app[["source", "event_type", "datetime_local", "date", "metric_1", "metric_1_name", "details"]])

    sw = samsung_weight_events.copy()
    sw["metric_1"] = sw["weight_lb"]
    sw["metric_1_name"] = "weight_lb"
    sw["details"] = sw["deviceuuid"]
    parts.append(sw[["source", "event_type", "datetime_local", "date", "metric_1", "metric_1_name", "details"]])

    ex = samsung_exercise_sessions.copy()
    ex["datetime_local"] = ex["datetime_start_local"]
    ex["metric_1"] = ex["calorie_kcal"]
    ex["metric_1_name"] = "exercise_calorie_kcal"
    ex["details"] = ex["exercise_type"].astype("Int64").astype(str)
    parts.append(ex[["source", "event_type", "datetime_local", "date", "metric_1", "metric_1_name", "details"]])

    sl = samsung_sleep_sessions.copy()
    sl["datetime_local"] = sl["datetime_start_local"]
    sl["metric_1"] = sl["sleep_duration_ms"]
    sl["metric_1_name"] = "sleep_duration_ms"
    sl["details"] = None
    parts.append(sl[["source", "event_type", "datetime_local", "date", "metric_1", "metric_1_name", "details"]])

    out = pd.concat(parts, ignore_index=True).sort_values("datetime_local").reset_index(drop=True)
    mask = (out["datetime_local"] >= start_dt) & (out["datetime_local"] < end_dt)
    return out.loc[mask].reset_index(drop=True)


def resample_intraday(df: pd.DataFrame, time_col: str, value_cols: Dict[str, str], freq: str = "15min") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    temp = df.dropna(subset=[time_col]).copy().set_index(time_col).sort_index()
    agg = temp[list(value_cols.keys())].resample(freq).agg(value_cols)
    agg = agg.reset_index().rename(columns={time_col: "datetime_local"})
    return agg


def build_master_15min_telemetry_active(
    noom_meal_events: pd.DataFrame,
    samsung_hr_raw: pd.DataFrame,
    samsung_stress_raw: pd.DataFrame,
    samsung_steps_intraday: pd.DataFrame,
    master_daily_features_active: pd.DataFrame,
) -> pd.DataFrame:
    first_meal_date = pd.to_datetime(noom_meal_events["date"]).min()
    last_meal_date = pd.to_datetime(noom_meal_events["date"]).max()
    start = pd.to_datetime(master_daily_features_active["date"]).min()
    end = pd.to_datetime(master_daily_features_active["date"]).max() + pd.Timedelta(days=1)

    grid = pd.DataFrame({"datetime_local": pd.date_range(start=start, end=end, freq="15min")})

    hr_15 = resample_intraday(samsung_hr_raw, "datetime_local", {"heart_rate_bpm": "mean"})
    stress_15 = resample_intraday(samsung_stress_raw, "datetime_local", {"stress_score": "mean"})
    steps_15 = resample_intraday(
        samsung_steps_intraday.rename(columns={"datetime_start_local": "datetime_local"}),
        "datetime_local",
        {"step_count": "sum", "distance_m": "sum", "step_calorie_kcal": "sum"},
    )

    meal_15 = noom_meal_events.copy()
    meal_15["datetime_local"] = meal_15["datetime_local_approx"].dt.round("15min")
    meal_15 = meal_15.groupby("datetime_local", as_index=False).agg({
        "calories_kcal": "sum",
        "protein_g": "sum",
        "carbs_g": "sum",
        "fat_g": "sum",
        "fiber_g": "sum",
        "sodium_mg": "sum",
        "meal_text": lambda s: " || ".join(s.dropna().astype(str)),
        "item_count": "sum",
    }).rename(columns={
        "calories_kcal": "meal_calories_kcal",
        "protein_g": "meal_protein_g",
        "carbs_g": "meal_carbs_g",
        "fat_g": "meal_fat_g",
        "fiber_g": "meal_fiber_g",
        "sodium_mg": "meal_sodium_mg",
        "item_count": "meal_item_count",
    })
    meal_15["is_meal_event"] = 1

    out = grid.merge(hr_15, on="datetime_local", how="left")
    out = out.merge(stress_15, on="datetime_local", how="left")
    out = out.merge(steps_15, on="datetime_local", how="left")
    out = out.merge(meal_15, on="datetime_local", how="left")

    out["date"] = out["datetime_local"].dt.floor("D")
    out = out.merge(master_daily_features_active, on="date", how="left")

    out["heart_rate_bpm"] = out["heart_rate_bpm"].interpolate(method="linear", limit=8)
    out["stress_score"] = out["stress_score"].interpolate(method="linear", limit=8)
    out["step_count"] = out["step_count"].fillna(0)
    out["distance_m"] = out["distance_m"].fillna(0)
    out["step_calorie_kcal"] = out["step_calorie_kcal"].fillna(0)
    out["is_meal_event"] = out["is_meal_event"].fillna(0).astype(int)
    out["bmr_15min_kcal"] = out["samsung_rest_calorie_kcal"] / 96.0
    out["total_burn_15min_kcal"] = out["bmr_15min_kcal"].fillna(0) + out["step_calorie_kcal"].fillna(0)
    out["cumulative_daily_burn_kcal"] = out.groupby("date")["total_burn_15min_kcal"].cumsum()

    mask = (out["datetime_local"] >= start) & (out["datetime_local"] < end)
    return out.loc[mask].reset_index(drop=True)


def archive_existing_output(project_root: Path, archive_name: str) -> None:
    archive_dir = project_root / archive_name
    ensure_dirs([archive_dir])
    for name in ["canonical", "fused"]:
        src = project_root / name
        if src.exists():
            dst = archive_dir / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))


def build_all(project_root: Path, warmup_days: int = 30, archive_existing: bool = False) -> None:
    if archive_existing:
        log("Archiving existing canonical/ and fused/ folders...")
        archive_existing_output(project_root, "_archive_previous_build")

    canonical_samsung = project_root / "canonical" / "samsung"
    canonical_noom = project_root / "canonical" / "noom"
    fused_dir = project_root / "fused"
    ensure_dirs([canonical_samsung, canonical_noom, fused_dir])

    samsung_paths = {name: find_source_file(project_root, "samsung", pattern) for name, pattern in SAMSUNG_PATTERNS.items()}
    noom_paths = {name: find_source_file(project_root, "noom", pattern) for name, pattern in NOOM_PATTERNS.items()}

    missing_critical = [
        ("samsung", "weight"),
        ("samsung", "sleep"),
        ("samsung", "sleep_stage"),
        ("samsung", "exercise"),
        ("samsung", "heart_rate"),
        ("samsung", "stress"),
        ("samsung", "pedometer_day_summary"),
        ("samsung", "calories_burned_details"),
        ("noom", "actions"),
        ("noom", "android_food_entries"),
        ("noom", "daily_calorie_budgets"),
        ("noom", "finish_day"),
    ]
    missing = []
    for side, name in missing_critical:
        path = samsung_paths.get(name) if side == "samsung" else noom_paths.get(name)
        if path is None:
            missing.append(f"{side}/{name}")
    if missing:
        raise FileNotFoundError("Missing required source files:\n- " + "\n- ".join(missing))

    log("Building Samsung canonical tables...")
    samsung_weight_events = build_samsung_weight_events(samsung_paths["weight"])
    samsung_sleep_sessions = build_samsung_sleep_sessions(samsung_paths["sleep"])
    samsung_sleep_stages = build_samsung_sleep_stages(samsung_paths["sleep_stage"])
    samsung_exercise_sessions = build_samsung_exercise_sessions(samsung_paths["exercise"])
    samsung_hr_raw = build_samsung_hr_raw(samsung_paths["heart_rate"])
    samsung_stress_raw = build_samsung_stress_raw(samsung_paths["stress"])
    samsung_steps_intraday = build_samsung_steps_intraday(samsung_paths["pedometer_step_count"]) if samsung_paths["pedometer_step_count"] else pd.DataFrame()
    samsung_pedometer_daily = build_samsung_pedometer_daily(samsung_paths["pedometer_day_summary"])
    samsung_activity_daily = build_samsung_activity_daily(samsung_paths["activity_day_summary"]) if samsung_paths["activity_day_summary"] else pd.DataFrame(columns=["date"])
    samsung_energy_daily = build_samsung_energy_daily(samsung_paths["calories_burned_details"])
    samsung_oxygen_raw = build_samsung_oxygen_raw(samsung_paths["oxygen_saturation"]) if samsung_paths["oxygen_saturation"] else pd.DataFrame()

    write_csv(samsung_weight_events, canonical_samsung / "samsung_weight_events.csv")
    write_csv(samsung_sleep_sessions, canonical_samsung / "samsung_sleep_sessions.csv")
    write_csv(samsung_sleep_stages, canonical_samsung / "samsung_sleep_stages.csv")
    write_csv(samsung_exercise_sessions, canonical_samsung / "samsung_exercise_sessions.csv")
    write_csv(samsung_hr_raw, canonical_samsung / "samsung_hr_raw.csv")
    write_csv(samsung_stress_raw, canonical_samsung / "samsung_stress_raw.csv")
    write_csv(samsung_steps_intraday, canonical_samsung / "samsung_steps_intraday.csv")
    write_csv(samsung_pedometer_daily, canonical_samsung / "samsung_pedometer_daily.csv")
    write_csv(samsung_activity_daily, canonical_samsung / "samsung_activity_daily.csv")
    write_csv(samsung_energy_daily, canonical_samsung / "samsung_energy_daily.csv")
    if not samsung_oxygen_raw.empty:
        write_csv(samsung_oxygen_raw, canonical_samsung / "samsung_oxygen_raw.csv")

    samsung_daily_features = build_samsung_daily_features(
        samsung_weight_events,
        samsung_sleep_sessions,
        samsung_pedometer_daily,
        samsung_activity_daily,
        samsung_energy_daily,
        samsung_exercise_sessions,
    )
    write_csv(samsung_daily_features, canonical_samsung / "samsung_daily_features.csv")

    log("Building Noom canonical tables...")
    noom_food_entries = build_noom_food_entries(noom_paths["android_food_entries"])
    noom_meal_events = build_noom_meal_events(noom_food_entries)
    noom_actions, noom_weighins, noom_steps_daily, noom_water_logs = build_noom_actions(noom_paths["actions"])
    noom_daily_budgets = build_noom_daily_budgets(noom_paths["daily_calorie_budgets"])
    noom_finish_day = build_noom_finish_day(noom_paths["finish_day"])
    noom_assignments = build_noom_assignments(noom_paths["assignments"]) if noom_paths["assignments"] else pd.DataFrame()
    noom_goals = build_noom_goals(noom_paths["goals"]) if noom_paths["goals"] else pd.DataFrame()
    noom_app_opens = build_noom_app_opens(noom_paths["application_opens"]) if noom_paths["application_opens"] else pd.DataFrame()
    noom_user_events = build_noom_user_events(noom_paths["user_events"]) if noom_paths["user_events"] else pd.DataFrame()
    noom_curriculum_state = build_noom_curriculum_state(noom_paths["curriculum_program_state"]) if noom_paths["curriculum_program_state"] else pd.DataFrame()

    write_csv(noom_food_entries, canonical_noom / "noom_food_entries.csv")
    write_csv(noom_meal_events, canonical_noom / "noom_meal_events.csv")
    write_csv(noom_actions, canonical_noom / "noom_actions.csv")
    write_csv(noom_weighins, canonical_noom / "noom_weighins.csv")
    write_csv(noom_steps_daily, canonical_noom / "noom_steps_daily.csv")
    write_csv(noom_water_logs, canonical_noom / "noom_water_logs.csv")
    write_csv(noom_daily_budgets, canonical_noom / "noom_daily_budgets.csv")
    write_csv(noom_finish_day, canonical_noom / "noom_finish_day.csv")
    if not noom_assignments.empty:
        write_csv(noom_assignments, canonical_noom / "noom_assignments.csv")
    if not noom_goals.empty:
        write_csv(noom_goals, canonical_noom / "noom_goals.csv")
    if not noom_app_opens.empty:
        write_csv(noom_app_opens, canonical_noom / "noom_application_opens.csv")
    if not noom_user_events.empty:
        write_csv(noom_user_events, canonical_noom / "noom_user_events.csv")
    if not noom_curriculum_state.empty:
        write_csv(noom_curriculum_state, canonical_noom / "noom_curriculum_program_state.csv")

    noom_daily_features = build_noom_daily_features(
        noom_food_entries,
        noom_meal_events,
        noom_weighins,
        noom_steps_daily,
        noom_water_logs,
        noom_daily_budgets,
        noom_finish_day,
        noom_app_opens,
    )
    write_csv(noom_daily_features, canonical_noom / "noom_daily_features.csv")

    first_meal_date = pd.to_datetime(noom_meal_events["date"]).min()
    last_meal_date = pd.to_datetime(noom_meal_events["date"]).max()
    active_start = first_meal_date - pd.Timedelta(days=warmup_days)
    active_end = last_meal_date + pd.Timedelta(days=1)

    log("Building fused outputs...")
    master_daily_features_full = build_master_daily_features_full(samsung_daily_features, noom_daily_features)
    master_daily_features = apply_daily_spine(master_daily_features_full, active_start, active_end)
    master_daily_features = add_weight_trends(master_daily_features)

    master_event_ledger = build_master_event_ledger_active(
        noom_meal_events,
        noom_weighins,
        noom_app_opens,
        samsung_weight_events,
        samsung_exercise_sessions,
        samsung_sleep_sessions,
        active_start,
        active_end + pd.Timedelta(days=1),
    )

    master_15min_telemetry_active = build_master_15min_telemetry_active(
        noom_meal_events,
        samsung_hr_raw,
        samsung_stress_raw,
        samsung_steps_intraday,
        master_daily_features,
    )

    write_csv(master_event_ledger, fused_dir / "master_event_ledger.csv")
    write_csv(master_daily_features, fused_dir / "master_daily_features.csv")
    write_csv(master_daily_features_full, fused_dir / "master_daily_features_full.csv")
    write_csv(master_15min_telemetry_active, fused_dir / "master_15min_telemetry_active.csv")

    log("Done.")
    log(f"Wrote Samsung canonical tables to: {canonical_samsung}")
    log(f"Wrote Noom canonical tables to: {canonical_noom}")
    log(f"Wrote fused outputs to: {fused_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical Samsung + Noom tables and fused FoodAI outputs.")
    parser.add_argument("--project-root", default=".", help="Path to the project root containing samsung/ and noom/ folders.")
    parser.add_argument("--warmup-days", type=int, default=30, help="Days of pre-meal warmup to keep before first Noom meal date.")
    parser.add_argument("--archive-existing", action="store_true", help="Move existing canonical/ and fused/ folders into _archive_previous_build/ before rebuilding.")
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    build_all(project_root, warmup_days=args.warmup_days, archive_existing=args.archive_existing)


if __name__ == "__main__":
    main()
