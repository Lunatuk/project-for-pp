import csv
import io
import json
from collections import defaultdict
from datetime import date, timedelta
from math import cos, pi, sin
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile

from warehouse.config import Settings

MANIFEST_FIELDS = ("id", "date", "zone", "pallets")


def write_partition(path: Path, content: str) -> None:
    data = content.encode("utf-8")
    if path.exists():
        old = path.read_bytes()
        if old.startswith(data):
            return
        if not data.startswith(old):
            raise ValueError(f"Данные в {path} не совпадают с генератором. Для нового эксперимента укажите другой каталог.")
    write_bytes_atomic(path, data)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(data)
    pending = Path(temporary.name)
    try:
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def make_demo(settings: Settings, through: date) -> None:
    if through < settings.start:
        raise ValueError("Конец генерации раньше даты начала.")
    manifests = defaultdict(list)
    events = defaultdict(list)
    # История воспроизводима и при продлении диапазона сохраняет прежние строки.
    previous = {zone: 0.65 for zone in settings.zones}
    for offset in range((through - settings.start).days + 1):
        day = settings.start + timedelta(days=offset)
        partition = day.strftime("%Y-%m")
        weekday = day.weekday()
        annual = sin(2 * pi * (offset + 40) / 365.25)
        wave = cos(2 * pi * offset / 38)
        for zone, capacity in sorted(settings.zones.items()):
            rng = Random(f"{settings.seed}:{day}:{zone}")
            base = capacity * 0.44
            weekday_effect = (1.18, 1.09, 1.01, 1.03, 1.21, 0.59, 0.43)[weekday]
            demand = base * weekday_effect * (1 + 0.20 * annual + 0.13 * wave)
            pallets = max(0, round(demand + rng.gauss(0, base * 0.06)))
            receipts = max(0, round(base * (1 + 0.24 * wave) + rng.gauss(0, base * 0.11)))
            target = 0.70 + 0.12 * annual + 0.16 * wave + (receipts - pallets) / (3 * capacity)
            previous[zone] = min(1, max(0.25, 0.68 * previous[zone] + 0.32 * target + rng.gauss(0, 0.018)))
            occupied = round(previous[zone] * capacity)
            key = f"{day}:{zone}"
            # Редкие пропуски и выбросы позволяют проверить контроль качества.
            if not (offset % 131 == 41 and zone == "A"):
                manifests[partition].append((key, day, zone, pallets if offset % 179 != 64 else -1))
            for kind, hour, details in (
                ("receiving.closed", "19:00", {"pallets": receipts}),
                ("inventory.snapshot", "23:55", {"occupied_cells": occupied}),
            ):
                if kind == "inventory.snapshot" and offset % 151 == 57 and zone == "B":
                    continue
                event = {
                    "id": f"{key}:{kind}",
                    "timestamp": f"{day}T{hour}:00",
                    "type": kind,
                    "payload": {"location": {"zone": zone}, "details": details},
                }
                events[partition].append(json.dumps(event, ensure_ascii=False))
        if offset % 193 == 81:
            events[partition].append('{"type": "incomplete.scan"')

    for folder in (settings.shipments, settings.wms_events):
        folder.mkdir(parents=True, exist_ok=True)
    for partition, rows in manifests.items():
        manifest = io.StringIO()
        writer = csv.writer(manifest, lineterminator="\n")
        writer.writerow(MANIFEST_FIELDS)
        writer.writerows(rows)
        write_partition(settings.shipments / f"{partition}.csv", manifest.getvalue())
    for partition, rows in events.items():
        write_partition(settings.wms_events / f"{partition}.jsonl", "\n".join(rows) + "\n")
