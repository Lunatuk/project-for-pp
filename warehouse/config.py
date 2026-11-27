import os
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database: Path
    shipments: Path
    wms_events: Path
    start: date
    demo_end: date
    seed: int
    horizon: int
    validation_days: int
    holdout_days: int
    occupancy_alert: float
    zones: dict[str, int]
    search_budget: int = 3
    promotion_threshold: float = 0.01
    monitor_days: int = 28
    drift_threshold: float = 0.20
    error_growth_threshold: float = 0.25
    monitor_min_points: int = 12


def load_settings(path: str | Path | None = None) -> Settings:
    config = Path(path or os.environ.get("WAREHOUSE_CONFIG", "config.toml")).resolve()
    with config.open("rb") as source:
        values = tomllib.load(source)
    for name in ("database", "shipments", "wms_events"):
        values[name] = (config.parent / values[name]).resolve()
    for name in ("start", "demo_end"):
        values[name] = date.fromisoformat(values[name])
    settings = Settings(**values)
    if not 1 <= settings.horizon <= 7:
        raise ValueError("Горизонт должен быть от 1 до 7 дней.")
    if settings.validation_days < 7 or settings.holdout_days < 14:
        raise ValueError("Для валидации нужно ≥ 7 дней, для отложенной выборки ≥ 14 дней.")
    if not settings.zones or any(type(n) is not int or n <= 0 for n in settings.zones.values()):
        raise ValueError("У каждой зоны должна быть положительная вместимость в ячейках.")
    if settings.demo_end < settings.start or not 0 < settings.occupancy_alert <= 100:
        raise ValueError("Проверьте даты и порог загрузки в конфигурации.")
    if not 1 <= settings.search_budget <= 4 or not 0 <= settings.promotion_threshold < 1:
        raise ValueError("Проверьте бюджет поиска (1–4) и долю улучшения для смены конфигурации.")
    if (settings.monitor_days < 7 or settings.drift_threshold <= 0
            or settings.error_growth_threshold <= 0 or settings.monitor_min_points < 1):
        raise ValueError("Проверьте настройки окна и порогов мониторинга.")
    return settings
