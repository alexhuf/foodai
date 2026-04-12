from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests


DEFAULT_LAT = 42.723
DEFAULT_LON = -84.400  # Okemos, MI


def log(msg: str) -> None:
    print(f"[weather-db] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_open_meteo_archive(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "America/New_York",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_url = "https://archive-api.open-meteo.com/v1/archive"

    hourly_vars = [
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "snow_depth",
        "weather_code",
        "cloud_cover",
        "wind_speed_10m",
        "wind_gusts_10m",
        "is_day",
        "sunshine_duration",
        "shortwave_radiation",
    ]

    daily_vars = [
        "weather_code",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_mean",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "sunrise",
        "sunset",
        "daylight_duration",
        "sunshine_duration",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "cloud_cover_mean",
        "cloud_cover_max",
        "cloud_cover_min",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "hourly": ",".join(hourly_vars),
        "daily": ",".join(daily_vars),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }

    log("Requesting Open-Meteo archive data...")
    resp = requests.get(base_url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    hourly = pd.DataFrame(data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"], errors="coerce")
    hourly = hourly.rename(columns={"time": "datetime_local"})

    daily = pd.DataFrame(data["daily"])
    daily["time"] = pd.to_datetime(daily["time"], errors="coerce").dt.floor("D")
    daily = daily.rename(columns={"time": "date"})

    return hourly, daily


def classify_temp_band_f(temp_f: pd.Series) -> pd.Series:
    bins = [-np.inf, 20, 32, 45, 60, 75, 85, np.inf]
    labels = [
        "extreme_cold",
        "freezing",
        "cold",
        "cool",
        "mild",
        "warm",
        "hot",
    ]
    return pd.cut(temp_f, bins=bins, labels=labels)


def classify_apparent_temp_band_f(temp_f: pd.Series) -> pd.Series:
    bins = [-np.inf, 15, 30, 45, 60, 75, 85, np.inf]
    labels = [
        "extreme_cold",
        "freezing",
        "cold",
        "cool",
        "mild",
        "warm",
        "hot",
    ]
    return pd.cut(temp_f, bins=bins, labels=labels)


def add_hourly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.copy()
    h["date"] = h["datetime_local"].dt.floor("D")
    h["temp_band_f"] = classify_temp_band_f(h["temperature_2m"])
    h["apparent_temp_band_f"] = classify_apparent_temp_band_f(h["apparent_temperature"])

    h["is_precip_hour"] = h["precipitation"].fillna(0) > 0
    h["is_rain_hour"] = h["rain"].fillna(0) > 0
    h["is_snow_hour"] = h["snowfall"].fillna(0) > 0
    h["is_cloudy_hour"] = h["cloud_cover"].fillna(0) >= 70
    h["is_gloomy_hour"] = (
        (h["cloud_cover"].fillna(0) >= 80)
        & (h["shortwave_radiation"].fillna(0) < 100)
    )
    h["is_hot_hour"] = h["apparent_temperature"].fillna(-999) >= 85
    h["is_cold_hour"] = h["apparent_temperature"].fillna(999) <= 32
    h["is_very_windy_hour"] = h["wind_speed_10m"].fillna(0) >= 20
    h["is_dark_hour"] = h["is_day"].fillna(1) == 0

    return h


def add_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()

    d["sunrise"] = pd.to_datetime(d["sunrise"], errors="coerce")
    d["sunset"] = pd.to_datetime(d["sunset"], errors="coerce")

    d["daylight_hours"] = d["daylight_duration"] / 3600.0
    d["sunshine_hours"] = d["sunshine_duration"] / 3600.0

    d["temp_band_f_mean"] = classify_temp_band_f(d["temperature_2m_mean"])
    d["apparent_temp_band_f_mean"] = classify_apparent_temp_band_f(d["apparent_temperature_mean"])

    d["is_precip_day"] = d["precipitation_sum"].fillna(0) > 0
    d["is_rain_day"] = d["rain_sum"].fillna(0) > 0
    d["is_snow_day"] = d["snowfall_sum"].fillna(0) > 0
    d["is_dark_early"] = d["sunset"].dt.hour.fillna(23) < 18
    d["is_very_dark_early"] = d["sunset"].dt.hour.fillna(23) < 17
    d["is_short_day"] = d["daylight_hours"].fillna(24) < 10
    d["is_long_day"] = d["daylight_hours"].fillna(0) > 14
    d["is_hot_day"] = d["apparent_temperature_max"].fillna(-999) >= 85
    d["is_very_hot_day"] = d["apparent_temperature_max"].fillna(-999) >= 95
    d["is_cold_day"] = d["apparent_temperature_min"].fillna(999) <= 32
    d["is_very_cold_day"] = d["apparent_temperature_min"].fillna(999) <= 15
    d["is_cloudy_day"] = d["cloud_cover_mean"].fillna(0) >= 70
    d["is_gloomy_day"] = (
        (d["cloud_cover_mean"].fillna(0) >= 75)
        & (d["sunshine_hours"].fillna(24) <= 2)
    )
    d["is_very_windy_day"] = d["wind_speed_10m_max"].fillna(0) >= 20

    return d


def add_streaks(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values("date").copy()

    def streak(series: pd.Series) -> pd.Series:
        out = []
        run = 0
        for val in series.fillna(False).astype(bool):
            run = run + 1 if val else 0
            out.append(run)
        return pd.Series(out, index=series.index)

    d["precip_streak_days"] = streak(d["is_precip_day"])
    d["rain_streak_days"] = streak(d["is_rain_day"])
    d["snow_streak_days"] = streak(d["is_snow_day"])
    d["gloomy_streak_days"] = streak(d["is_gloomy_day"])
    d["dark_early_streak_days"] = streak(d["is_dark_early"])
    d["hot_streak_days"] = streak(d["is_hot_day"])
    d["cold_streak_days"] = streak(d["is_cold_day"])

    d["snow_depth_gt_0"] = d.get("snowfall_sum", pd.Series([0] * len(d))).fillna(0) > 0
    return d


def build_weather_context(project_root: Path, latitude: float, longitude: float, timezone: str) -> None:
    fused_dir = project_root / "fused"
    weather_dir = project_root / "weather"
    ensure_dir(weather_dir)

    daily_path = fused_dir / "master_daily_features.csv"
    intraday_path = fused_dir / "master_15min_telemetry_active.csv"

    if not daily_path.exists():
        raise FileNotFoundError(f"Missing required fused file: {daily_path}")
    if not intraday_path.exists():
        raise FileNotFoundError(f"Missing required fused file: {intraday_path}")

    daily = pd.read_csv(daily_path, low_memory=False)
    intraday = pd.read_csv(intraday_path, low_memory=False)

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.floor("D")
    intraday["datetime_local"] = pd.to_datetime(intraday["datetime_local"], errors="coerce")

    start_date = daily["date"].min().strftime("%Y-%m-%d")
    end_date = daily["date"].max().strftime("%Y-%m-%d")

    log(f"Using date range {start_date} to {end_date}")
    hourly, daily_weather = fetch_open_meteo_archive(latitude, longitude, start_date, end_date, timezone)
    hourly = add_hourly_features(hourly)
    daily_weather = add_daily_features(daily_weather)
    daily_weather = add_streaks(daily_weather)

    # Align hourly weather to 15-minute fused grid by forward-fill from hourly observations.
    hourly_15 = hourly.copy()
    hourly_15 = hourly_15.set_index("datetime_local").sort_index()
    target_index = pd.date_range(
        start=intraday["datetime_local"].min(),
        end=intraday["datetime_local"].max(),
        freq="15min",
    )
    hourly_15 = hourly_15.reindex(target_index.union(hourly_15.index)).sort_index().ffill()
    hourly_15 = hourly_15.loc[target_index].reset_index().rename(columns={"index": "datetime_local"})
    hourly_15["date"] = hourly_15["datetime_local"].dt.floor("D")

    intraday_weather = intraday[["datetime_local"]].merge(hourly_15, on="datetime_local", how="left")
    intraday_weather = intraday_weather.merge(daily_weather, on="date", how="left", suffixes=("", "_daily"))

    daily_weather_aligned = daily[["date"]].merge(daily_weather, on="date", how="left")

    metadata = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date,
        "source": "Open-Meteo Historical Weather API",
        "hourly_rows": int(len(hourly)),
        "hourly_15_rows": int(len(intraday_weather)),
        "daily_rows": int(len(daily_weather_aligned)),
    }

    log("Writing weather datasets...")
    hourly.to_csv(weather_dir / "weather_hourly_raw.csv", index=False)
    daily_weather.to_csv(weather_dir / "weather_daily_raw.csv", index=False)
    intraday_weather.to_csv(weather_dir / "weather_context_15min.csv", index=False)
    daily_weather_aligned.to_csv(weather_dir / "weather_context_daily.csv", index=False)
    (weather_dir / "weather_context_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log("Done.")
    log(f"Wrote weather context files to: {weather_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical weather/daylight context aligned to the fused timeline.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--latitude", type=float, default=DEFAULT_LAT, help="Latitude for weather lookup.")
    parser.add_argument("--longitude", type=float, default=DEFAULT_LON, help="Longitude for weather lookup.")
    parser.add_argument("--timezone", default="America/New_York", help="IANA timezone string.")
    args = parser.parse_args()

    build_weather_context(
        project_root=Path(args.project_root).expanduser().resolve(),
        latitude=args.latitude,
        longitude=args.longitude,
        timezone=args.timezone,
    )


if __name__ == "__main__":
    main()
