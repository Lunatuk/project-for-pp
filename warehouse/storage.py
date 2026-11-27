import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS load_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL,
    files_read INTEGER NOT NULL, records_added INTEGER NOT NULL,
    issues INTEGER NOT NULL, message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS loaded_files (
    source TEXT NOT NULL, file TEXT NOT NULL, fingerprint TEXT NOT NULL,
    PRIMARY KEY (source, file)
);
CREATE TABLE IF NOT EXISTS raw_records (
    source TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
    file TEXT NOT NULL, PRIMARY KEY (source, id)
);
CREATE TABLE IF NOT EXISTS import_issues (
    source TEXT NOT NULL, file TEXT NOT NULL, fingerprint TEXT NOT NULL,
    line INTEGER NOT NULL, reason TEXT NOT NULL, raw TEXT NOT NULL,
    PRIMARY KEY (source, file, fingerprint, line)
);
CREATE TABLE IF NOT EXISTS daily (
    date TEXT NOT NULL, zone TEXT NOT NULL,
    shipments INTEGER, occupancy REAL, receiving INTEGER,
    PRIMARY KEY (date, zone)
);
CREATE TABLE IF NOT EXISTS quality_issues (
    date TEXT NOT NULL, zone TEXT NOT NULL, indicator TEXT NOT NULL,
    reason TEXT NOT NULL, PRIMARY KEY (date, zone, indicator)
);
CREATE TABLE IF NOT EXISTS analyses (
    version TEXT PRIMARY KEY, created_at TEXT NOT NULL, cutoff TEXT NOT NULL,
    payload TEXT NOT NULL, models BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS active_analysis (
    slot INTEGER PRIMARY KEY CHECK (slot = 1), version TEXT NOT NULL REFERENCES analyses(version)
);
CREATE TABLE IF NOT EXISTS activations (
    id INTEGER PRIMARY KEY, at TEXT NOT NULL, version TEXT NOT NULL REFERENCES analyses(version),
    action TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS monitoring (
    version TEXT PRIMARY KEY REFERENCES analyses(version), payload TEXT NOT NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db
