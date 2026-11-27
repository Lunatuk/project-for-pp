import csv
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from warehouse.config import Settings
from warehouse.demo import MANIFEST_FIELDS, write_bytes_atomic


def save_feed(folder: Path, suffix: str, data: bytes) -> Path:
    if not data:
        raise ValueError("Передан пустой файл.")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"upload-{hashlib.sha256(data).hexdigest()}{suffix}"
    if not path.exists():
        write_bytes_atomic(path, data)
    return path


@dataclass
class LoadResult:
    source: str
    status: str = "ok"
    files_read: int = 0
    records_added: int = 0
    issues: int = 0
    message: str = ""


def read_with_retry(path: Path) -> bytes:
    for attempt in range(3):
        try:
            return path.read_bytes()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.15 * (attempt + 1))
    raise AssertionError("Не выполнено чтение файла.")


def parse_manifest(line: str) -> dict:
    fields = next(csv.reader([line], strict=True))
    if len(fields) != len(MANIFEST_FIELDS):
        raise ValueError("Ожидаются четыре поля CSV.")
    return dict(zip(MANIFEST_FIELDS, fields))


def identity(source: str, row: dict, settings: Settings) -> tuple[str, str]:
    if source == "carrier":
        day = date.fromisoformat(row["date"])
        zone = row["zone"]
    else:
        stamp = datetime.fromisoformat(row["timestamp"])
        if "T" not in row["timestamp"] or stamp.tzinfo is not None:
            raise ValueError("Ожидается локальное время склада без смещения.")
        day = stamp.date()
        zone = row["payload"]["location"]["zone"]
    if day < settings.start or zone not in settings.zones:
        raise ValueError("Дата до начала истории или неизвестная зона.")
    return day.isoformat(), zone


def import_file(
    db: sqlite3.Connection,
    settings: Settings,
    result: LoadResult,
    path: Path,
    parser: Callable[[str], dict],
) -> None:
    content = read_with_retry(path)
    digest = hashlib.sha256(content).hexdigest()
    stored = db.execute(
        "SELECT fingerprint FROM loaded_files WHERE source=? AND file=?",
        (result.source, path.name),
    ).fetchone()
    if stored and stored["fingerprint"] == digest:
        return

    result.files_read += 1
    lines = content.decode("utf-8-sig").splitlines()
    start = 0
    if result.source == "carrier":
        if not lines or next(csv.reader([lines[0]], strict=True)) != list(MANIFEST_FIELDS):
            raise ValueError("Неверный заголовок CSV.")
        start = 1
    for number, line in enumerate(lines[start:], start + 1):
        if not line.strip():
            continue
        try:
            row = parser(line)
            record_id = row["id"]
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError("Пустой или неверный идентификатор.")
            identity(result.source, row, settings)
            if result.source == "wms" and not isinstance(row.get("type"), str):
                raise ValueError("Отсутствует тип события WMS.")
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
            added = db.execute(
                "INSERT OR IGNORE INTO raw_records VALUES (?, ?, ?, ?)",
                (result.source, record_id, payload, path.name),
            ).rowcount
            if not added:
                old = db.execute(
                    "SELECT payload FROM raw_records WHERE source=? AND id=?", (result.source, record_id)
                ).fetchone()
                if old["payload"] != payload:
                    raise ValueError("Конфликт ID с уже принятой записью.")
            result.records_added += added
        except (ValueError, KeyError, TypeError, csv.Error) as exc:
            rejected = db.execute(
                "INSERT OR IGNORE INTO import_issues VALUES (?, ?, ?, ?, ?, ?)",
                (result.source, path.name, hashlib.sha256(line.encode("utf-8")).hexdigest(),
                 number, str(exc), line),
            )
            result.issues += rejected.rowcount
    db.execute(
        "INSERT OR REPLACE INTO loaded_files VALUES (?, ?, ?)",
        (result.source, path.name, digest),
    )


def ingest(db: sqlite3.Connection, settings: Settings) -> list[LoadResult]:
    results = []
    for source, folder, suffix, parser in (
        ("carrier", settings.shipments, ".csv", parse_manifest),
        ("wms", settings.wms_events, ".jsonl", json.loads),
    ):
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        result = LoadResult(source)
        errors = []
        with db:
            db.execute("BEGIN IMMEDIATE")
            try:
                files = sorted(path for path in folder.iterdir() if path.suffix.lower() == suffix)
                if not files:
                    errors.append(f"В каталоге {folder} нет файлов {suffix}.")
            except OSError as exc:
                files = []
                errors.append(str(exc))
            for path in files:
                try:
                    import_file(db, settings, result, path, parser)
                except (OSError, UnicodeError, ValueError, csv.Error) as exc:
                    errors.append(f"{path.name}: {exc}")
            if errors and (not files or len(errors) == len(files)):
                result.status = "error"
            elif errors or result.issues:
                result.status = "warning"
            result.message = " | ".join(errors)
            db.execute(
                "INSERT INTO load_runs (started_at, source, status, files_read, records_added, issues, message)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (started, source, result.status, result.files_read, result.records_added, result.issues, result.message),
            )
        results.append(result)
    return results
