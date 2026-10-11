from __future__ import annotations

import unittest
from unittest.mock import patch

import windows_tools


class CloseApplicationSettlingTests(unittest.TestCase):
    def test_close_catches_window_that_appears_again_during_settle_period(self) -> None:
        clock = [0.0]

        def monotonic() -> float:
            return clock[0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        def visible_windows(*args: object, **kwargs: object) -> list[dict[str, object]]:
            if clock[0] < 0.16:
                return []
            if clock[0] < 0.40:
                return [{"hwnd": 101, "pid": 1, "process": "whatsapp.exe", "title": "WhatsApp"}]
            if clock[0] < 0.72:
                return []
            if clock[0] < 0.96:
                return [{"hwnd": 202, "pid": 2, "process": "whatsapp.exe", "title": "WhatsApp"}]
            return []

        with (
            patch("windows_tools.time.monotonic", side_effect=monotonic),
            patch("windows_tools.time.sleep", side_effect=sleep),
            patch("windows_tools.visible_windows_for_process_names", side_effect=visible_windows),
            patch("windows_tools.win32gui.PostMessage") as post_message,
        ):
            result = windows_tools.close_application("whatsapp")

        targeted_handles = {call.args[0] for call in post_message.call_args_list}
        self.assertEqual(targeted_handles, {101, 202})
        self.assertTrue(result["ok"])
        self.assertEqual(result["closed"], 2)
        self.assertEqual(result["remaining"], [])


if __name__ == "__main__":
    unittest.main()
