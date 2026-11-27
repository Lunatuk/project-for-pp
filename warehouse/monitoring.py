import json

import numpy as np
import pandas as pd

from warehouse.config import Settings
from warehouse.mart import INDICATORS
from warehouse.models import TARGETS, metrics


def population_stability(reference: pd.Series, current: pd.Series) -> float | None:
    known = reference.dropna().to_numpy()
    if not len(known) or current.dropna().empty:
        return None
    edges = np.unique(np.quantile(known, (0.2, 0.4, 0.6, 0.8)))
    if len(edges) == 1:
        tolerance = max(1e-6, abs(edges[0]) * 1e-6)
        edges = edges[0] + np.array([-tolerance, tolerance])

    def frequencies(values: pd.Series) -> np.ndarray:
        bins = np.searchsorted(edges, values.dropna())
        counts = np.append(np.bincount(bins, minlength=len(edges) + 1), values.isna().sum())
        adjusted = counts + 0.5
        return adjusted / adjusted.sum()

    expected, observed = frequencies(reference), frequencies(current)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def monitor(daily: pd.DataFrame, cutoff: pd.Timestamp, settings: Settings, previous: dict | None, releases) -> dict:
    start = cutoff - pd.Timedelta(days=settings.monitor_days - 1)
    current = daily.loc[daily["date"] >= start]
    # Опорное окно предшествует наблюдаемому периоду.
    reference = daily.loc[daily["date"].between(
        start - pd.Timedelta(days=3 * settings.monitor_days), start, inclusive="left"
    )]
    drift = {field: population_stability(reference[field], current[field]) for field in INDICATORS}
    known = {(day.strftime("%Y-%m-%d"), zone): (shipments, occupancy)
             for day, zone, shipments, occupancy in current[["date", "zone", *TARGETS]].itertuples(index=False, name=None)}
    start_day = start.date().isoformat()
    forecasts = {target: {} for target in TARGETS}
    for release in releases:
        for target, result in json.loads(release["payload"])["targets"].items():
            for point in result["forecast"]:
                if point["date"] < start_day or (point["date"], point["zone"]) not in known:
                    continue
                key = point["origin"], point["date"], point["zone"]
                # При повторном выпуске на тот же день сохраняем первый доступный прогноз.
                forecasts[target].setdefault(key, point["forecast"])

    performance = {}
    for index, target in enumerate(TARGETS):
        actual, guess = [], []
        for (_, day, zone), value in forecasts[target].items():
            observation = known[day, zone][index]
            if pd.notna(observation):
                actual.append(observation)
                guess.append(value)
        older = previous["targets"][target] if previous else None
        reference_mae = next((row["mae"] for row in older["holdout"] if row["model"] == older["winner"]), None) if older else None
        mae = metrics(actual, guess)["mae"] if actual else None
        alert = bool(len(actual) >= settings.monitor_min_points and reference_mae is not None
                     and mae > reference_mae * (1 + settings.error_growth_threshold))
        performance[target] = {"mae": mae, "n": len(actual), "reference_mae": reference_mae, "alert": alert}

    return {
        "current_from": start_day, "current_until": cutoff.date().isoformat(),
        "reference_from": reference["date"].min().date().isoformat() if len(reference) else None,
        "reference_until": reference["date"].max().date().isoformat() if len(reference) else None,
        "psi": drift, "drift_threshold": settings.drift_threshold,
        "error_growth_threshold": settings.error_growth_threshold, "monitor_min_points": settings.monitor_min_points,
        "performance": performance,
        "recommend": any(value is not None and value >= settings.drift_threshold for value in drift.values())
        or any(item["alert"] for item in performance.values()),
    }
