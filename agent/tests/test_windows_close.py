from __future__ import annotations

import unittest
from types import SimpleNamespace
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


class CloseAllVisibleApplicationsTests(unittest.TestCase):
    def test_initialize_uses_visible_terminal_host_when_console_handle_is_hidden(self) -> None:
        kernel32 = SimpleNamespace(
            SetConsoleTitleW=lambda title: None,
            GetConsoleWindow=lambda: 600,
        )
        with (
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLE", None),
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLES", set()),
            patch.object(windows_tools.ctypes.windll, "kernel32", kernel32),
            patch("windows_tools.win32gui.GetForegroundWindow", return_value=700),
            patch("windows_tools.find_visible_window_by_title", return_value=None),
            patch("windows_tools.time.monotonic", side_effect=[0.0, 2.0]),
            patch("windows_tools.top_level_window_handle", side_effect=lambda hwnd: hwnd),
            patch("windows_tools.visible_top_level_window_handles", return_value={700}),
            patch("windows_tools.is_likely_terminal_host", return_value=True),
            patch("windows_tools.win32gui.IsWindow", return_value=True),
            patch("windows_tools.win32gui.IsWindowVisible", return_value=True),
            patch("windows_tools.related_window_handles", side_effect=lambda hwnd: {hwnd}),
            patch("windows_tools.window_process_id", return_value=70),
            patch("windows_tools.window_process_name", return_value="windowsterminal.exe"),
            patch("windows_tools.win32gui.GetWindowText", return_value="PowerShell - Sentry"),
        ):
            result = windows_tools.initialize_sentry_window()

        self.assertTrue(result["ok"])
        self.assertEqual(result["hwnd"], 700)
        self.assertEqual(result["protected_handles"], [600, 700])

    def test_close_all_fails_closed_when_sentry_terminal_is_unknown(self) -> None:
        with (
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLE", None),
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLES", set()),
            patch("windows_tools.win32gui.PostMessage") as post_message,
        ):
            result = windows_tools.close_all_visible_applications()

        self.assertFalse(result["ok"])
        self.assertIn("safely identify", result["message"])
        post_message.assert_not_called()

    def test_close_all_tracks_windows_that_appear_during_shutdown(self) -> None:
        clock = [0.0]
        first = {"hwnd": 10, "pid": 100, "process": "notepad++.exe", "title": "notes.txt"}
        second = {"hwnd": 20, "pid": 200, "process": "explorer.exe", "title": "Projects"}
        sentry = {
            "hwnd": 999,
            "pid": 999,
            "process": "windowsterminal.exe",
            "title": "SENTRY Portable AI",
            "reason": "sentry",
        }

        def monotonic() -> float:
            return clock[0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        visible_results = [
            ([first], [sentry]),
            ([second], [sentry]),
            ([], [sentry]),
            ([], [sentry]),
        ]
        with (
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLE", 999),
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLES", {999}),
            patch("windows_tools.win32gui.IsWindow", return_value=True),
            patch("windows_tools.visible_user_application_windows", side_effect=visible_results),
            patch("windows_tools.time.monotonic", side_effect=monotonic),
            patch("windows_tools.time.sleep", side_effect=sleep),
            patch("windows_tools.win32gui.PostMessage") as post_message,
        ):
            result = windows_tools.close_all_visible_applications()

        self.assertTrue(result["ok"])
        self.assertEqual(result["targeted"], 2)
        self.assertEqual(result["closed"], 2)
        self.assertEqual([call.args[0] for call in post_message.call_args_list], [10, 20])

    def test_close_all_refuses_to_run_if_visible_sentry_window_is_not_protected(self) -> None:
        other = {"hwnd": 10, "pid": 100, "process": "notepad.exe", "title": "notes.txt"}
        with (
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLE", 999),
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLES", {999}),
            patch("windows_tools.win32gui.IsWindow", return_value=True),
            patch("windows_tools.visible_user_application_windows", return_value=([other], [])),
            patch("windows_tools.win32gui.PostMessage") as post_message,
        ):
            result = windows_tools.close_all_visible_applications()

        self.assertFalse(result["ok"])
        self.assertEqual(result["targeted"], 0)
        self.assertIn("verify its visible terminal", result["message"])
        post_message.assert_not_called()

    def test_window_filter_keeps_explorer_notepad_plus_plus_and_other_terminal(self) -> None:
        handles = [1, 2, 3, 4, 5, 6]
        titles = {
            1: "SENTRY Portable AI [100]",
            2: "Projects",
            3: "notes.txt - Notepad++",
            4: "Administrator: PowerShell",
            5: "Taskbar",
            6: "Search",
        }
        pids = {1: 500, 2: 501, 3: 502, 4: 503, 5: 504, 6: 505}
        classes = {
            1: "ConsoleWindowClass",
            2: "CabinetWClass",
            3: "Notepad++",
            4: "ConsoleWindowClass",
            5: "Shell_TrayWnd",
            6: "Windows.UI.Core.CoreWindow",
        }
        process_names = {
            500: "WindowsTerminal.exe",
            501: "explorer.exe",
            502: "notepad++.exe",
            503: "powershell.exe",
            504: "explorer.exe",
            505: "SearchHost.exe",
        }

        def enum_windows(callback: object, extra: object) -> None:
            for hwnd in handles:
                callback(hwnd, extra)

        def process(pid: int) -> SimpleNamespace:
            return SimpleNamespace(name=lambda: process_names[pid])

        with (
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLE", 1),
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLES", {1}),
            patch("windows_tools.win32gui.EnumWindows", side_effect=enum_windows),
            patch("windows_tools.win32gui.IsWindowVisible", return_value=True),
            patch("windows_tools.win32gui.GetAncestor", side_effect=lambda hwnd, mode: hwnd),
            patch("windows_tools.win32gui.GetWindowText", side_effect=lambda hwnd: titles[hwnd]),
            patch("windows_tools.win32gui.GetClassName", side_effect=lambda hwnd: classes[hwnd]),
            patch(
                "windows_tools.win32process.GetWindowThreadProcessId",
                side_effect=lambda hwnd: (1, pids[hwnd]),
            ),
            patch("windows_tools.psutil.Process", side_effect=process),
        ):
            targets, skipped = windows_tools.visible_user_application_windows()

        self.assertEqual([window["hwnd"] for window in targets], [2, 3, 4])
        self.assertEqual([window["hwnd"] for window in skipped], [1, 5, 6])
        self.assertEqual(skipped[0]["reason"], "sentry")
        self.assertEqual(skipped[1]["reason"], "windows_shell")
        self.assertEqual(skipped[2]["reason"], "windows_shell")

    def test_window_filter_protects_top_level_host_of_console_handle(self) -> None:
        handles = [200, 300]
        titles = {200: "PowerShell - Sentry", 300: "Other PowerShell"}
        pids = {200: 20, 300: 30}

        def enum_windows(callback: object, extra: object) -> None:
            for hwnd in handles:
                callback(hwnd, extra)

        with (
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLE", 200),
            patch.object(windows_tools, "_SENTRY_WINDOW_HANDLES", {100}),
            patch.object(windows_tools, "_SENTRY_WINDOW_TITLE", "SENTRY Portable AI [123]"),
            patch("windows_tools.win32gui.EnumWindows", side_effect=enum_windows),
            patch("windows_tools.win32gui.IsWindowVisible", return_value=True),
            patch("windows_tools.win32gui.GetWindowText", side_effect=lambda hwnd: titles[hwnd]),
            patch("windows_tools.win32gui.GetClassName", return_value="ConsoleWindowClass"),
            patch(
                "windows_tools.win32gui.GetAncestor",
                side_effect=lambda hwnd, mode: 100 if hwnd == 200 else hwnd,
            ),
            patch(
                "windows_tools.win32process.GetWindowThreadProcessId",
                side_effect=lambda hwnd: (1, pids[hwnd]),
            ),
            patch(
                "windows_tools.psutil.Process",
                side_effect=lambda pid: SimpleNamespace(name=lambda: "windowsterminal.exe"),
            ),
        ):
            targets, skipped = windows_tools.visible_user_application_windows()

        self.assertEqual([window["hwnd"] for window in skipped], [200])
        self.assertEqual(skipped[0]["reason"], "sentry")
        self.assertEqual([window["hwnd"] for window in targets], [300])


class PowerToolTests(unittest.TestCase):
    def test_sleep_uses_windows_suspend_api(self) -> None:
        with patch.object(
            windows_tools.ctypes.windll.powrprof,
            "SetSuspendState",
            return_value=True,
        ) as set_suspend_state:
            result = windows_tools.sleep_computer()

        set_suspend_state.assert_called_once_with(False, False, False)
        self.assertTrue(result["ok"])

    def test_lock_uses_windows_workstation_api(self) -> None:
        with patch.object(
            windows_tools.ctypes.windll.user32,
            "LockWorkStation",
            return_value=True,
        ) as lock_workstation:
            result = windows_tools.lock_computer()

        lock_workstation.assert_called_once_with()
        self.assertTrue(result["ok"])

    def test_shutdown_is_graceful_and_does_not_imply_force(self) -> None:
        with patch(
            "windows_tools.subprocess.run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            result = windows_tools.shutdown_computer()

        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["shutdown.exe", "/s", "/t", "0"])
        self.assertNotIn("/f", command)
        self.assertTrue(result["dispatched"])


if __name__ == "__main__":
    unittest.main()
