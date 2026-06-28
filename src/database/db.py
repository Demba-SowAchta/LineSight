"""
Database layer  ->  traceability + analytics.

WHY A DATABASE?
The project requirement is "traceability and analytics": every part the camera
inspects must leave a permanent, queryable record (when, which line, pass/fail,
defect type, confidence, link to the saved image). A database is what turns a
one-off script into an auditable industrial system.

WHY SQLITE?
We use Python's built-in `sqlite3` so the project runs with ZERO setup -- no
server to install, the whole database is a single file (artifacts/ivp.db).
For real multi-line production you swap this file for PostgreSQL; the table
schema and the function signatures below stay identical, so nothing else in the
codebase changes. See docs/04_mlops.md for the Postgres migration.

This module exposes a tiny, readable API:
    init_db()                         -> create tables if missing
    insert_inspection(record)         -> save one inspection, returns its id
    recent_inspections(limit)         -> latest N records (for the dashboard)
    summary_stats()                   -> counts / pass-rate / defect breakdown
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src import config


# ----------------------------------------------------------------------------
# Connection helper
# ----------------------------------------------------------------------------
@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """
    Open a database connection as a context manager so it is always closed.

    Usage:
        with get_connection() as conn:
            conn.execute(...)
    """
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    # Rows behave like dictionaries: row["verdict"] instead of row[3]. Much clearer.
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------
# One table is enough for the demo. Each column maps to one fact about an
# inspection. Comments explain the role of each field.
SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,   -- ISO timestamp (UTC) of the inspection
    line_id       TEXT    NOT NULL,   -- which production line
    station_id    TEXT    NOT NULL,   -- which station on that line
    category      TEXT    NOT NULL,   -- product family (e.g. screw_bag)
    part_id       TEXT,               -- optional serial / barcode of the part
    verdict       TEXT    NOT NULL,   -- 'PASS' or 'FAIL'
    defect_type   TEXT,               -- e.g. 'structural', 'logical', 'good'
    score         REAL    NOT NULL,   -- anomaly score from the model
    threshold     REAL    NOT NULL,   -- threshold used for the decision
    confidence    REAL,               -- 0..1 distance from the threshold
    model_name    TEXT    NOT NULL,   -- which model produced the verdict
    model_version TEXT,               -- model version for audit / rollback
    latency_ms    REAL,               -- inference latency, for monitoring
    image_path    TEXT,               -- saved evidence image
    heatmap_path  TEXT                -- saved anomaly heatmap (if any)
);

-- Indexes make the dashboard queries fast even with millions of rows.
CREATE INDEX IF NOT EXISTS idx_inspections_created ON inspections(created_at);
CREATE INDEX IF NOT EXISTS idx_inspections_verdict ON inspections(verdict);
CREATE INDEX IF NOT EXISTS idx_inspections_line    ON inspections(line_id);
"""


def init_db(db_path: Path | None = None) -> None:
    """Create the tables and indexes if they do not exist yet."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


# ----------------------------------------------------------------------------
# Writes
# ----------------------------------------------------------------------------
def insert_inspection(record: dict[str, Any], db_path: Path | None = None) -> int:
    """
    Save one inspection record and return its new database id.

    `record` is a plain dictionary. Missing optional keys default to None/now,
    so callers only have to provide what they actually have.
    """
    row = {
        "created_at": record.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "line_id": record.get("line_id", config.LINE_ID),
        "station_id": record.get("station_id", config.STATION_ID),
        "category": record.get("category", config.CATEGORY),
        "part_id": record.get("part_id"),
        "verdict": record["verdict"],
        "defect_type": record.get("defect_type"),
        "score": float(record["score"]),
        "threshold": float(record["threshold"]),
        "confidence": record.get("confidence"),
        "model_name": record.get("model_name", "unknown"),
        "model_version": record.get("model_version"),
        "latency_ms": record.get("latency_ms"),
        "image_path": record.get("image_path"),
        "heatmap_path": record.get("heatmap_path"),
    }
    columns = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    with get_connection(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO inspections ({columns}) VALUES ({placeholders})",
            tuple(row.values()),
        )
        return int(cur.lastrowid)


# ----------------------------------------------------------------------------
# Reads (used by the dashboard and the /stats API)
# ----------------------------------------------------------------------------
def recent_inspections(limit: int = 50, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return the most recent inspections as a list of dictionaries."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def summary_stats(db_path: Path | None = None) -> dict[str, Any]:
    """
    Aggregate KPIs for the dashboard:
      - total inspected
      - pass / fail counts and pass-rate
      - average latency
      - breakdown by defect_type
    """
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM inspections").fetchone()["c"]
        passed = conn.execute(
            "SELECT COUNT(*) AS c FROM inspections WHERE verdict = 'PASS'"
        ).fetchone()["c"]
        failed = total - passed
        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) AS a FROM inspections"
        ).fetchone()["a"]
        by_defect = conn.execute(
            """
            SELECT COALESCE(defect_type, 'unknown') AS defect_type, COUNT(*) AS c
            FROM inspections
            WHERE verdict = 'FAIL'
            GROUP BY defect_type
            ORDER BY c DESC
            """
        ).fetchall()

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total) if total else 0.0,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else 0.0,
        "defect_breakdown": {r["defect_type"]: r["c"] for r in by_defect},
    }


if __name__ == "__main__":
    # Smoke test: create the DB, insert a fake record, print the stats.
    init_db()
    insert_inspection(
        {
            "verdict": "FAIL",
            "defect_type": "structural",
            "score": 0.82,
            "threshold": 0.5,
            "confidence": 0.64,
            "model_name": "demo",
            "latency_ms": 12.3,
        }
    )
    print(summary_stats())
