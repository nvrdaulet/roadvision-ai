"""Persist maintenance decisions separately from detection and GPS outputs."""

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping


STATUSES = ("NEW", "ASSIGNED", "IN_PROGRESS", "FIXED")
PRIORITIES = ("LOW", "MEDIUM", "HIGH")
TEAMS = ("Road Team A", "Road Team B", "Emergency Crew")


def initial_priority(confidence: float, observation_count: int) -> str:
    """Suggest an initial workflow priority, separate from AI confidence."""
    if confidence >= 0.70 and observation_count >= 3:
        return "HIGH"
    if confidence >= 0.55:
        return "MEDIUM"
    return "LOW"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _defect_id(value: object) -> str:
    identifier = str(value).strip()
    if value is None or not identifier:
        raise ValueError("A defect ID is required.")
    return identifier


def load_maintenance(
    records: Iterable[Mapping[str, object]], db_path: str | Path
) -> list[dict]:
    """Create missing requests and return saved decisions without resetting edits.

    Each input record needs defect_id, max_confidence, and detection_count.
    All rows are returned so callers can join the current CSV by defect_id;
    saved decisions for IDs absent from a later CSV remain in the database.
    """
    database = Path(db_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS maintenance_requests (
                    defect_id TEXT PRIMARY KEY NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('NEW', 'ASSIGNED', 'IN_PROGRESS', 'FIXED')
                    ),
                    priority TEXT NOT NULL CHECK (
                        priority IN ('LOW', 'MEDIUM', 'HIGH')
                    ),
                    assigned_to TEXT NOT NULL DEFAULT '' CHECK (
                        assigned_to IN (
                            '', 'Road Team A', 'Road Team B', 'Emergency Crew'
                        )
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            now = _utc_now()
            for record in records:
                connection.execute(
                    """
                    INSERT INTO maintenance_requests (
                        defect_id, status, priority, assigned_to,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(defect_id) DO NOTHING
                    """,
                    (
                        _defect_id(record["defect_id"]),
                        "NEW",
                        initial_priority(
                            float(record["max_confidence"]),
                            int(record["detection_count"]),
                        ),
                        "",
                        now,
                        now,
                    ),
                )
            rows = connection.execute(
                "SELECT * FROM maintenance_requests ORDER BY defect_id"
            ).fetchall()
        return [dict(row) for row in rows]


def save_maintenance(
    defect_id: str,
    status: str,
    priority: str,
    assigned_to: str,
    db_path: str | Path,
) -> dict:
    """Update an existing request in one transaction, preserving created_at."""
    identifier = _defect_id(defect_id)
    if status not in STATUSES:
        raise ValueError("Select a valid maintenance status.")
    if priority not in PRIORITIES:
        raise ValueError("Select a valid maintenance priority.")
    if assigned_to not in ("", *TEAMS):
        raise ValueError("Select a listed responsible team or leave it unassigned.")
    database = Path(db_path)
    if not database.is_file():
        raise FileNotFoundError("Load defect records before saving maintenance changes.")

    with closing(sqlite3.connect(database, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            cursor = connection.execute(
                """
                UPDATE maintenance_requests
                SET status = ?, priority = ?, assigned_to = ?, updated_at = ?
                WHERE defect_id = ?
                """,
                (status, priority, assigned_to, _utc_now(), identifier),
            )
            if cursor.rowcount != 1:
                raise ValueError("This defect no longer exists in the maintenance database.")
            row = connection.execute(
                "SELECT * FROM maintenance_requests WHERE defect_id = ?",
                (identifier,),
            ).fetchone()
        return dict(row)
