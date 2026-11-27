import json
import sqlite3
from collections import defaultdict

import pandas as pd

from warehouse.config import Settings
from warehouse.ingest import identity

INDICATORS = ("shipments", "occupancy", "receiving")
EVENTS = {"inventory.snapshot": "occupancy", "receiving.closed": "receiving"}


def measurement(row: dict, metric: str, capacity: int) -> int | float:
    if metric == "shipments":
        value = int(row["pallets"])
    else:
        key = "occupied_cells" if metric == "occupancy" else "pallets"
        value = row["payload"]["details"][key]
        if type(value) is not int:
            raise ValueError("Требуется целое число.")
    if not 0 <= value <= (capacity if metric == "occupancy" else capacity * 5):
        raise ValueError("Значение вне допустимого диапазона.")
    return 100 * value / capacity if metric == "occupancy" else value


def prepare(db: sqlite3.Connection, settings: Settings) -> tuple[pd.DataFrame, pd.Timestamp]:
    observations = defaultdict(list)
    for record in db.execute("SELECT source, payload FROM raw_records ORDER BY source, id"):
        row = json.loads(record["payload"])
        try:
            day, zone = identity(record["source"], row, settings)
        except ValueError:
            # Исторические записи остаются в сыром слое при изменении конфигурации.
            continue
        metric = "shipments" if record["source"] == "carrier" else EVENTS.get(row["type"])
        if metric is None:
            continue
        try:
            value = measurement(row, metric, settings.zones[zone])
        except (KeyError, TypeError, ValueError):
            value = None
        observations[day, zone, metric].append(value)

    if not observations:
        raise ValueError("Нет складских наблюдений. Сначала подготовьте данные через demo или укажите источники.")

    latest = max(day for day, _, _ in observations)
    issues = []
    rows = []
    for stamp in pd.date_range(settings.start, latest):
        day = stamp.date().isoformat()
        for zone in sorted(settings.zones):
            values = []
            for metric in INDICATORS:
                records = observations.get((day, zone, metric), ())
                value = None
                if not records:
                    reason = "Нет суточной записи"
                elif len(records) > 1:
                    reason = "Несколько суточных записей"
                elif records[0] is None:
                    reason = "Некорректное значение"
                else:
                    reason = None
                    value = records[0]
                if reason:
                    issues.append((day, zone, metric, reason))
                values.append(value)
            rows.append((day, zone, *values))

    db.execute("DELETE FROM daily")
    db.execute("DELETE FROM quality_issues")
    db.executemany("INSERT INTO daily VALUES (?, ?, ?, ?, ?)", rows)
    db.executemany("INSERT INTO quality_issues VALUES (?, ?, ?, ?)", issues)

    daily = pd.DataFrame(rows, columns=("date", "zone", *INDICATORS))
    daily["date"] = pd.to_datetime(daily["date"])
    daily[list(INDICATORS)] = daily[list(INDICATORS)].astype(float)
    complete = daily[["shipments", "occupancy"]].notna().all(axis=1).groupby(daily["date"]).all()
    if not complete.any():
        raise ValueError("Ни за одну дату не получены обе метрики по всем зонам.")
    return daily, complete[complete].index[-1]
