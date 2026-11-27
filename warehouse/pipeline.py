import hashlib
import json
import pickle
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn

from warehouse.config import Settings
from warehouse.explain import importance, reference_row
from warehouse.features import build_examples
from warehouse.ingest import LoadResult, ingest
from warehouse.mart import prepare
from warehouse.models import Candidate, TARGETS, evaluate
from warehouse.monitoring import monitor
from warehouse.storage import connect

IMPLEMENTATION_VERSION = 2


def experiment_settings(settings: Settings) -> dict:
    return {
        "seed": settings.seed, "horizon": settings.horizon,
        "validation_days": settings.validation_days, "holdout_days": settings.holdout_days,
        "search_budget": settings.search_budget, "promotion_threshold": settings.promotion_threshold,
        "zones": settings.zones,
    }


def runtime_versions() -> dict:
    return {
        "implementation": IMPLEMENTATION_VERSION, "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
    }


def data_checksum(daily: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(daily, index=False).to_numpy().tobytes()).hexdigest()


def version_for(daily: pd.DataFrame, settings: Settings) -> str:
    config = {**runtime_versions(), **experiment_settings(settings), "data": data_checksum(daily)}
    config["monitor"] = [settings.monitor_days, settings.drift_threshold,
                         settings.error_growth_threshold, settings.monitor_min_points]
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def analyze(daily: pd.DataFrame, cutoff: pd.Timestamp, settings: Settings, incumbents: dict) -> tuple[dict, dict]:
    examples = build_examples(daily, cutoff, settings.horizon)
    history = daily.copy()
    history["date"] = history["date"].dt.strftime("%Y-%m-%d")
    payload = {
        "history": history.astype(object).where(history.notna(), None).to_dict("records"),
        "experiment": {"settings": experiment_settings(settings), "runtime": runtime_versions(),
                       "checksum": data_checksum(daily), "incumbents": incumbents},
        "targets": {},
    }
    artifacts = {}
    for target in TARGETS:
        incumbent = Candidate.from_specification(incumbents[target]) if target in incumbents else None
        result, models = evaluate(examples, cutoff, settings, target, incumbent)
        train = examples.dropna(subset=[f"actual_{target}"])
        holdout = pd.Timestamp(result["holdout_from"])
        stop = pd.Timestamp(result["holdout_until"])
        midpoint = holdout + (stop - holdout + pd.Timedelta(days=1)) // 2
        held_rows = train.loc[train["origin"].between(holdout, stop)]
        models["checked_reference"] = reference_row(train.loc[train["date"] < holdout], models["checked"])
        models["production_reference"] = reference_row(train, models["production"])
        models["held_rows"] = held_rows
        models["future_rows"] = examples.loc[examples["origin"] == cutoff]
        ranking = []
        for start, end in ((holdout, midpoint), (midpoint, stop + pd.Timedelta(days=1))):
            period = held_rows.loc[held_rows["origin"].between(start, end, inclusive="left")]
            if period.empty:
                continue
            scores = importance(models["checked"], period, models["checked_reference"], settings.seed)
            for name, score in scores.iterrows():
                ranking.append({
                    "period": f"{start:%d.%m.%Y} – {end - pd.Timedelta(days=1):%d.%m.%Y}",
                    "feature": name, **score.to_dict(),
                })
        result["importance"] = ranking
        payload["targets"][target] = result
        artifacts[target] = models
    return payload, artifacts


def run(settings: Settings) -> tuple[list[LoadResult], str | None]:
    with closing(connect(settings.database)) as db:
        loads = ingest(db, settings)
        if any(load.status == "error" for load in loads):
            return loads, None
        with db:
            db.execute("BEGIN IMMEDIATE")
            daily, cutoff = prepare(db, settings)
            daily = daily.loc[daily["date"] <= cutoff].reset_index(drop=True)
            version = version_for(daily, settings)
            existing = db.execute("SELECT 1 FROM analyses WHERE version=?", (version,)).fetchone()
            active = db.execute("""
                SELECT a.version, a.payload FROM analyses a JOIN active_analysis s USING(version)
            """).fetchone()
            if existing is None:
                previous = json.loads(active["payload"]) if active else None
                incumbents = {target: result["selection"] for target, result in previous["targets"].items()
                              if "selection" in result} if previous else {}
                payload, artifacts = analyze(daily, cutoff, settings, incumbents)
                stock_since = (cutoff - pd.Timedelta(days=settings.monitor_days + 6)).date().isoformat()
                releases = db.execute(
                    "SELECT payload FROM analyses WHERE cutoff>=? AND cutoff<? ORDER BY rowid", (stock_since, cutoff.date().isoformat())
                )
                record = monitor(daily, cutoff, settings, previous, releases)
                db.execute(
                    "INSERT INTO analyses VALUES (?, ?, ?, ?, ?)",
                    (version, now(), cutoff.date().isoformat(),
                     json.dumps(payload, ensure_ascii=False, allow_nan=False), pickle.dumps(artifacts)),
                )
                db.execute("INSERT INTO monitoring VALUES (?, ?)", (version, json.dumps(record, allow_nan=False)))
                activate(db, version, "publish")
            elif active:
                return loads, active["version"]
            else:
                activate(db, version, "publish")
    return loads, version


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_version(db: sqlite3.Connection, prefix: str) -> sqlite3.Row:
    if not prefix or any(character not in "0123456789abcdef" for character in prefix):
        raise ValueError("ID выпуска должен содержать шестнадцатеричные символы.")
    records = db.execute("SELECT version, cutoff, payload FROM analyses WHERE version LIKE ?", (prefix + "%",)).fetchall()
    if len(records) != 1:
        raise ValueError("Укажите полный или однозначный ID выпущенной версии.")
    return records[0]


def activate(db: sqlite3.Connection, version: str, action: str) -> None:
    current = db.execute("SELECT version FROM active_analysis WHERE slot=1").fetchone()
    if current is None or current["version"] != version:
        db.execute("INSERT OR REPLACE INTO active_analysis VALUES (1, ?)", (version,))
        db.execute("INSERT INTO activations (at, version, action) VALUES (?, ?, ?)", (now(), version, action))


def activate_version(settings: Settings, prefix: str) -> str:
    with closing(connect(settings.database)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        row = resolve_version(db, prefix)
        activate(db, row["version"], "manual")
        return row["version"]


def replay(settings: Settings, prefix: str) -> dict:
    with closing(connect(settings.database)) as db:
        row = resolve_version(db, prefix)
    original = json.loads(row["payload"])
    experiment = original.get("experiment")
    if not experiment or experiment["runtime"] != runtime_versions():
        raise ValueError("Для воспроизведения этого выпуска нужна версия кода и библиотек, использованная при обучении.")
    configured = replace(settings, **experiment["settings"])
    daily = pd.DataFrame(original["history"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily[["shipments", "occupancy", "receiving"]] = daily[["shipments", "occupancy", "receiving"]].astype(float)
    if data_checksum(daily) != experiment["checksum"]:
        raise ValueError("Контрольная сумма снимка данных не совпала.")
    cutoff = pd.Timestamp(row["cutoff"])
    examples = build_examples(daily, cutoff, configured.horizon)
    verified = []
    for target in TARGETS:
        spec = experiment["incumbents"].get(target)
        incumbent = Candidate.from_specification(spec) if spec else None
        reproduced, _ = evaluate(examples, cutoff, configured, target, incumbent)
        expected = original["targets"][target]
        for area in ("cv", "holdout"):
            if len(reproduced[area]) != len(expected[area]):
                raise ValueError("Число моделей при воспроизведении изменилось.")
            for generated, recorded in zip(reproduced[area], expected[area]):
                if generated["params"] != recorded["params"] or not np.allclose(
                    [generated[key] for key in ("mae", "rmse", "n")],
                    [recorded[key] for key in ("mae", "rmse", "n")], rtol=1e-9, atol=1e-9,
                ):
                    raise ValueError(f"Результаты для {target} не совпали с журналом.")
        if expected["selection"] != reproduced["selection"]:
            raise ValueError(f"Выбор модели для {target} изменился.")
        verified.append(target)
    return {"version": row["version"], "cutoff": row["cutoff"], "targets": verified}


def read_control(db: sqlite3.Connection) -> dict:
    versions = []
    for row in db.execute("""
        SELECT version, cutoff, created_at,
            json_extract(payload, '$.experiment') AS experiment,
            json_extract(payload, '$.targets.shipments.winner') AS shipments,
            json_extract(payload, '$.targets.occupancy.winner') AS occupancy,
            json_extract(payload, '$.targets.shipments.holdout') AS shipments_holdout,
            json_extract(payload, '$.targets.occupancy.holdout') AS occupancy_holdout
        FROM analyses ORDER BY cutoff DESC, rowid DESC
    """):
        holdout_mae = {
            target: next(score["mae"] for score in json.loads(row[f"{target}_holdout"])
                         if score["model"] == row[target])
            for target in TARGETS
        }
        versions.append({"version": row["version"], "cutoff": row["cutoff"], "created_at": row["created_at"],
                         "experiment": json.loads(row["experiment"]) if row["experiment"] else None,
                         "shipments": row["shipments"], "occupancy": row["occupancy"], "mae": holdout_mae})
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    checks = [{"version": row["version"], **json.loads(row["payload"])} for row in db.execute("""
        SELECT m.version, m.payload FROM monitoring m JOIN analyses a ON m.version=a.version
        ORDER BY a.cutoff DESC, a.rowid DESC LIMIT 20
    """)] if "monitoring" in tables else []
    activations = [dict(row) for row in db.execute(
        "SELECT at,version,action FROM activations ORDER BY id DESC LIMIT 20"
    )] if "activations" in tables else []
    return {"versions": versions, "checks": checks, "activations": activations}


def read_dashboard(db: sqlite3.Connection) -> tuple[sqlite3.Row | None, dict]:
    db.row_factory = sqlite3.Row
    active = db.execute("""
        SELECT a.version, a.created_at, a.cutoff, a.payload FROM analyses a
        JOIN active_analysis b ON b.version = a.version WHERE b.slot = 1
    """).fetchone()
    return active, {
        "runs": [dict(row) for row in db.execute("SELECT * FROM load_runs ORDER BY id DESC LIMIT 20")],
        "quality": [dict(row) for row in db.execute(
            "SELECT * FROM quality_issues ORDER BY date DESC, zone, indicator"
        )],
        "quarantined": db.execute("SELECT COUNT(*) FROM import_issues").fetchone()[0],
        "loaded_through": db.execute("SELECT MAX(date) FROM daily").fetchone()[0],
    }


def load_artifacts(db: sqlite3.Connection, version: str) -> dict:
    row = db.execute("SELECT models FROM analyses WHERE version=?", (version,)).fetchone()
    # Модели загружаются из локальной БД, которую создал и заполнил сам проект.
    return pickle.loads(row[0])
