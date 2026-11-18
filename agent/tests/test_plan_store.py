from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from plan_store import PlanStore


class PlanStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "agent_state.db"

    def test_records_ordered_plan_and_action_states(self) -> None:
        store = PlanStore(self.database_path, session_id="session-one")
        plan = store.create_plan(
            user_text="goodnight",
            intent="shutdown_cleanup",
            source="exact_alias",
            confirmation_required=True,
            actions=[
                {
                    "tool": "close_all_visible_applications",
                    "arguments": {},
                    "requires_previous_success": False,
                },
                {
                    "tool": "shutdown_computer",
                    "arguments": {},
                    "requires_previous_success": True,
                },
            ],
        )

        self.assertEqual(plan.status, "pending_confirmation")
        self.assertEqual([action.sequence for action in plan.actions], [0, 1])
        self.assertTrue(plan.actions[1].requires_previous_success)

        store.confirm_plan(plan.plan_id)
        store.start_plan(plan.plan_id)
        store.start_action(plan.plan_id, 0)
        store.finish_action(plan.plan_id, 0, status="succeeded", result={"ok": True})
        store.start_action(plan.plan_id, 1)
        store.finish_action(
            plan.plan_id,
            1,
            status="dispatched",
            result={"ok": True, "dispatched": True},
        )
        store.finish_plan(plan.plan_id, status="dispatched", summary="Shutdown scheduled.")

        completed = store.get_plan(plan.plan_id)
        self.assertEqual(completed.status, "dispatched")
        self.assertEqual([action.status for action in completed.actions], ["succeeded", "dispatched"])

    def test_new_session_marks_unfinished_plans_interrupted_without_resuming(self) -> None:
        first_store = PlanStore(self.database_path, session_id="session-one")
        plan = first_store.create_plan(
            user_text="deep sleep",
            intent="deep_sleep",
            source="exact_alias",
            confirmation_required=True,
            actions=[
                {
                    "tool": "close_all_visible_applications",
                    "arguments": {},
                    "requires_previous_success": False,
                },
                {
                    "tool": "sleep_computer",
                    "arguments": {},
                    "requires_previous_success": True,
                },
            ],
        )

        second_store = PlanStore(self.database_path, session_id="session-two")
        interrupted = second_store.get_plan(plan.plan_id)

        self.assertEqual(interrupted.status, "interrupted")
        self.assertEqual([action.status for action in interrupted.actions], ["interrupted", "interrupted"])


if __name__ == "__main__":
    unittest.main()
