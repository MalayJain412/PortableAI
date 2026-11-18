from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


INTERRUPTIBLE_PLAN_STATUSES = ("pending_confirmation", "queued", "running")
INTERRUPTIBLE_ACTION_STATUSES = ("queued", "running")


@dataclass(frozen=True)
class StoredAction:
    sequence: int
    tool: str
    arguments: dict[str, Any]
    status: str
    requires_previous_success: bool


@dataclass(frozen=True)
class StoredPlan:
    plan_id: str
    user_text: str
    intent: str
    source: str
    status: str
    confirmation_required: bool
    actions: tuple[StoredAction, ...]


class PlanStore:
    """Persistent, host-owned execution checklist for validated tool plans."""

    def __init__(self, database_path: Path, *, session_id: str | None = None) -> None:
        self.database_path = database_path
        self.session_id = session_id or str(uuid.uuid4())
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.mark_interrupted_plans()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confirmation_required INTEGER NOT NULL,
                    confirmed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    summary TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS plan_actions (
                    plan_id TEXT NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requires_previous_success INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    PRIMARY KEY (plan_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status);
                CREATE INDEX IF NOT EXISTS idx_plans_session ON plans(session_id, created_at);
                """
            )

    def mark_interrupted_plans(self) -> None:
        now = utc_now()
        plan_placeholders = ", ".join("?" for _ in INTERRUPTIBLE_PLAN_STATUSES)
        action_placeholders = ", ".join("?" for _ in INTERRUPTIBLE_ACTION_STATUSES)
        with self._connect() as connection:
            interrupted_ids = [
                str(row["plan_id"])
                for row in connection.execute(
                    f"SELECT plan_id FROM plans WHERE status IN ({plan_placeholders})",
                    INTERRUPTIBLE_PLAN_STATUSES,
                )
            ]
            if not interrupted_ids:
                return
            id_placeholders = ", ".join("?" for _ in interrupted_ids)
            connection.execute(
                f"UPDATE plans SET status = 'interrupted', updated_at = ?, "
                f"error = COALESCE(error, 'Sentry stopped before this plan completed.') "
                f"WHERE plan_id IN ({id_placeholders})",
                (now, *interrupted_ids),
            )
            connection.execute(
                f"UPDATE plan_actions SET status = 'interrupted', finished_at = ?, "
                f"error = COALESCE(error, 'Execution was interrupted.') "
                f"WHERE plan_id IN ({id_placeholders}) AND status IN ({action_placeholders})",
                (now, *interrupted_ids, *INTERRUPTIBLE_ACTION_STATUSES),
            )

    def create_plan(
        self,
        *,
        user_text: str,
        intent: str,
        source: str,
        actions: list[dict[str, Any]],
        confirmation_required: bool,
    ) -> StoredPlan:
        plan_id = str(uuid.uuid4())
        now = utc_now()
        status = "pending_confirmation" if confirmation_required else "queued"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO plans (
                    plan_id, session_id, user_text, intent, source, status,
                    confirmation_required, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    self.session_id,
                    user_text,
                    intent,
                    source,
                    status,
                    int(confirmation_required),
                    now,
                    now,
                ),
            )
            for sequence, action in enumerate(actions):
                connection.execute(
                    """
                    INSERT INTO plan_actions (
                        plan_id, sequence, tool_name, arguments_json, status,
                        requires_previous_success
                    ) VALUES (?, ?, ?, ?, 'queued', ?)
                    """,
                    (
                        plan_id,
                        sequence,
                        str(action["tool"]),
                        json.dumps(action.get("arguments") or {}, sort_keys=True),
                        int(bool(action.get("requires_previous_success", False))),
                    ),
                )
        return self.get_plan(plan_id)

    def get_plan(self, plan_id: str) -> StoredPlan:
        with self._connect() as connection:
            plan = connection.execute(
                "SELECT * FROM plans WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
            if plan is None:
                raise KeyError(f"Unknown plan: {plan_id}")
            action_rows = connection.execute(
                "SELECT * FROM plan_actions WHERE plan_id = ? ORDER BY sequence",
                (plan_id,),
            ).fetchall()
        return StoredPlan(
            plan_id=str(plan["plan_id"]),
            user_text=str(plan["user_text"]),
            intent=str(plan["intent"]),
            source=str(plan["source"]),
            status=str(plan["status"]),
            confirmation_required=bool(plan["confirmation_required"]),
            actions=tuple(
                StoredAction(
                    sequence=int(row["sequence"]),
                    tool=str(row["tool_name"]),
                    arguments=json.loads(str(row["arguments_json"])),
                    status=str(row["status"]),
                    requires_previous_success=bool(row["requires_previous_success"]),
                )
                for row in action_rows
            ),
        )

    def confirm_plan(self, plan_id: str) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE plans
                SET status = 'queued', confirmed_at = ?, updated_at = ?
                WHERE plan_id = ? AND status = 'pending_confirmation'
                """,
                (now, now, plan_id),
            )

    def cancel_plan(self, plan_id: str) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE plans SET status = 'cancelled', updated_at = ? WHERE plan_id = ?",
                (now, plan_id),
            )
            connection.execute(
                """
                UPDATE plan_actions SET status = 'cancelled', finished_at = ?
                WHERE plan_id = ? AND status = 'queued'
                """,
                (now, plan_id),
            )

    def start_plan(self, plan_id: str) -> None:
        self._set_plan_status(plan_id, "running")

    def start_action(self, plan_id: str, sequence: int) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE plan_actions SET status = 'running', started_at = ?
                WHERE plan_id = ? AND sequence = ?
                """,
                (now, plan_id, sequence),
            )
            connection.execute(
                "UPDATE plans SET updated_at = ? WHERE plan_id = ?",
                (now, plan_id),
            )

    def finish_action(
        self,
        plan_id: str,
        sequence: int,
        *,
        status: str,
        result: Any | None = None,
        error: str | None = None,
    ) -> None:
        now = utc_now()
        result_json = json.dumps(result, sort_keys=True, default=str) if result is not None else None
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE plan_actions
                SET status = ?, result_json = ?, error = ?, finished_at = ?
                WHERE plan_id = ? AND sequence = ?
                """,
                (status, result_json, error, now, plan_id, sequence),
            )
            connection.execute(
                "UPDATE plans SET updated_at = ? WHERE plan_id = ?",
                (now, plan_id),
            )

    def finish_plan(
        self,
        plan_id: str,
        *,
        status: str,
        summary: str,
        error: str | None = None,
    ) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE plans SET status = ?, summary = ?, error = ?, updated_at = ?
                WHERE plan_id = ?
                """,
                (status, summary, error, now, plan_id),
            )

    def _set_plan_status(self, plan_id: str, status: str) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE plans SET status = ?, updated_at = ? WHERE plan_id = ?",
                (status, now, plan_id),
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
