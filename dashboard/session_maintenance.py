"""Keep the demo workflow usable when hosted SQLite storage is unavailable.

Fallback records belong to the caller's session only. Once a session falls back,
it keeps using its in-memory records so a later rerun cannot discard those edits.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, MutableMapping

if __package__:
    from .maintenance import (
        PRIORITIES,
        STATUSES,
        TEAMS,
        initial_priority,
        load_maintenance,
        save_maintenance,
    )
else:
    from maintenance import (
        PRIORITIES,
        STATUSES,
        TEAMS,
        initial_priority,
        load_maintenance,
        save_maintenance,
    )


SESSION_ONLY_KEY = "_roadvision_workflow_session_only"
_DATABASE_KEY = "_roadvision_workflow_database"
_RECORDS_KEY = "_roadvision_workflow_records"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _defect_id(value: object) -> str:
    identifier = str(value).strip()
    if value is None or not identifier:
        raise ValueError("A defect ID is required.")
    return identifier


def _prepare_state(db_path: str | Path, state: MutableMapping) -> dict:
    # Normalize without accessing a directory that may be unwritable/unavailable.
    database_key = os.path.normcase(os.path.abspath(os.fspath(db_path)))
    if state.get(_DATABASE_KEY) != database_key:
        state[_DATABASE_KEY] = database_key
        state[_RECORDS_KEY] = {}
        state[SESSION_ONLY_KEY] = False
    return state[_RECORDS_KEY]


def load_workflow(
    records: Iterable[Mapping[str, object]],
    db_path: str | Path,
    state: MutableMapping,
) -> list[dict]:
    """Load SQLite decisions or seed/preserve decisions in this session."""
    input_records = list(records)
    saved = _prepare_state(db_path, state)
    if not state[SESSION_ONLY_KEY]:
        try:
            stored_rows = load_maintenance(input_records, db_path)
        except (OSError, sqlite3.Error):
            state[SESSION_ONLY_KEY] = True
        else:
            saved = {row["defect_id"]: dict(row) for row in stored_rows}
            state[_RECORDS_KEY] = saved
            return [dict(row) for row in saved.values()]

    now = _utc_now()
    for record in input_records:
        identifier = _defect_id(record["defect_id"])
        if identifier not in saved:
            saved[identifier] = {
                "defect_id": identifier,
                "status": "NEW",
                "priority": initial_priority(
                    float(record["max_confidence"]), int(record["detection_count"])
                ),
                "assigned_to": "",
                "created_at": now,
                "updated_at": now,
            }
    state[_RECORDS_KEY] = saved
    return [dict(row) for row in saved.values()]


def save_workflow(
    defect_id: str,
    status: str,
    priority: str,
    assigned_to: str,
    db_path: str | Path,
    state: MutableMapping,
) -> dict:
    """Persist an edit, retaining it in the session if SQLite cannot save it."""
    saved = _prepare_state(db_path, state)
    identifier = _defect_id(defect_id)
    if status not in STATUSES:
        raise ValueError("Select a valid maintenance status.")
    if priority not in PRIORITIES:
        raise ValueError("Select a valid maintenance priority.")
    if assigned_to not in ("", *TEAMS):
        raise ValueError("Select a listed responsible team or leave it unassigned.")
    if identifier not in saved:
        raise ValueError("Load this defect before saving maintenance changes.")

    if not state[SESSION_ONLY_KEY]:
        try:
            updated = save_maintenance(
                identifier, status, priority, assigned_to, db_path
            )
        except (OSError, sqlite3.Error):
            state[SESSION_ONLY_KEY] = True
        else:
            saved[identifier] = dict(updated)
            state[_RECORDS_KEY] = saved
            return dict(updated)

    updated = {
        **saved[identifier],
        "status": status,
        "priority": priority,
        "assigned_to": assigned_to,
        "updated_at": _utc_now(),
    }
    saved[identifier] = updated
    state[_RECORDS_KEY] = saved
    return dict(updated)
