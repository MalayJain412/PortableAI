from __future__ import annotations

import ctypes
import html
import json
import platform
import os
import re
import shutil
import subprocess
import time
import webbrowser
import winreg
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
import psutil
import win32clipboard
import win32con
import win32gui
import win32process
from mcp.server.mcpserver.exceptions import ToolError

from config import CACHE_DIR, CLIPBOARD_READ_LIMIT, DIRECTORY_LIST_LIMIT, DIRECTORY_SCAN_LIMIT, MAX_PROCESS_COUNT
from security import (
    managed_workspace_root,
    resolve_inspection_directory,
    resolve_managed_file,
    resolve_managed_file_destination,
    resolve_managed_directory,
    validate_allowed_application,
    validate_clipboard_text,
    validate_file_name,
    validate_notepad_filename,
    validate_notepad_text,
    validate_url,
)


_NOTEPAD_PIDS: set[int] = set()
_CURRENT_NOTEPAD_FILE = CACHE_DIR / "notepad" / "portable_ai_current_note.txt"
_ACTIVE_NOTEPAD_FILE = _CURRENT_NOTEPAD_FILE
_SENTRY_WINDOW_HANDLE: int | None = None
_SENTRY_WINDOW_TITLE: str | None = None
_SENTRY_WINDOW_HANDLES: set[int] = set()
_SENTRY_HOST_PID: int | None = None
_PROTECTED_WINDOW_CLASSES = {
    "progman",
    "workerw",
    "shell_traywnd",
    "shell_secondarytraywnd",
    "dv2controlhost",
}
_PROTECTED_SHELL_PROCESSES = {
    "searchapp.exe",
    "searchhost.exe",
    "shellexperiencehost.exe",
    "startmenuexperiencehost.exe",
    "textinputhost.exe",
}
_TERMINAL_HOST_PROCESSES = {
    "alacritty.exe",
    "code.exe",
    "cmd.exe",
    "conhost.exe",
    "cursor.exe",
    "idea64.exe",
    "mintty.exe",
    "openconsole.exe",
    "powershell.exe",
    "powershell_ise.exe",
    "pwsh.exe",
    "pycharm64.exe",
    "tabby.exe",
    "termius.exe",
    "wezterm-gui.exe",
    "windowsterminal.exe",
    "wt.exe",
}
_APPLICATION_SHELL_IDS: dict[str, str] = {
    "notion.exe": "com.electron.notion",
    "code.exe": "Microsoft.VisualStudioCode",
    "githubdesktop.exe": "com.squirrel.GitHubDesktop.GitHubDesktop",
    "outlook.exe": "Microsoft.Office.OUTLOOK.EXE.15",
    "mongodbcompass.exe": "com.squirrel.MongoDBCompass.MongoDBCompass",
    "termius.exe": "com.termius.mac",
}


def initialize_sentry_window() -> dict[str, Any]:
    """Give the agent terminal a unique title and capture its top-level window handle."""
    global _SENTRY_HOST_PID, _SENTRY_WINDOW_HANDLE, _SENTRY_WINDOW_HANDLES, _SENTRY_WINDOW_TITLE

    title = f"SENTRY Portable AI [{os.getpid()}]"
    kernel32 = ctypes.windll.kernel32
    foreground_handle = int(win32gui.GetForegroundWindow() or 0)
    kernel32.SetConsoleTitleW(title)

    console_handle = int(kernel32.GetConsoleWindow() or 0)
    console_root = top_level_window_handle(console_handle)
    titled_handle = find_visible_window_by_title(title)
    if titled_handle is None:
        deadline = time.monotonic() + 1.0
        while titled_handle is None and time.monotonic() < deadline:
            titled_handle = find_visible_window_by_title(title)
            if titled_handle is None:
                time.sleep(0.05)

    top_level_handles = visible_top_level_window_handles()
    console_candidate = console_root if console_root in top_level_handles else 0
    foreground_candidate = (
        foreground_handle
        if foreground_handle in top_level_handles and is_likely_terminal_host(foreground_handle, title)
        else 0
    )
    candidates = [titled_handle or 0, console_candidate, foreground_candidate]
    handle = next(
        (
            candidate
            for candidate in candidates
            if candidate and win32gui.IsWindow(candidate) and win32gui.IsWindowVisible(candidate)
        ),
        0,
    )

    _SENTRY_WINDOW_TITLE = title
    _SENTRY_WINDOW_HANDLE = handle or None
    _SENTRY_WINDOW_HANDLES = related_window_handles(handle)
    if console_handle:
        _SENTRY_WINDOW_HANDLES.update(related_window_handles(console_handle))
    _SENTRY_HOST_PID = window_process_id(handle) if handle else None
    host_title = win32gui.GetWindowText(handle).strip() if handle else ""
    host_process = window_process_name(_SENTRY_HOST_PID) if _SENTRY_HOST_PID else "unknown"
    return {
        "ok": _SENTRY_WINDOW_HANDLE is not None,
        "hwnd": _SENTRY_WINDOW_HANDLE,
        "title": title,
        "host_title": host_title,
        "host_process": host_process,
        "protected_handles": sorted(_SENTRY_WINDOW_HANDLES),
        "message": (
            f"Sentry terminal protected: {host_title or host_process}."
            if _SENTRY_WINDOW_HANDLE is not None
            else "The Sentry terminal window could not be identified."
        ),
    }


def find_visible_window_by_title(title: str) -> int | None:
    matches: list[int] = []

    def callback(hwnd: int, _: object) -> bool:
        window_title = win32gui.GetWindowText(hwnd).strip()
        if win32gui.IsWindowVisible(hwnd) and title in window_title:
            matches.append(hwnd)
        return True

    win32gui.EnumWindows(callback, None)
    return matches[0] if matches else None


def visible_top_level_window_handles() -> set[int]:
    handles: set[int] = set()

    def callback(hwnd: int, _: object) -> bool:
        if win32gui.IsWindowVisible(hwnd):
            handles.add(int(hwnd))
        return True

    win32gui.EnumWindows(callback, None)
    return handles


def is_likely_terminal_host(hwnd: int, sentry_title: str) -> bool:
    if not hwnd:
        return False
    title = win32gui.GetWindowText(hwnd).strip()
    if sentry_title and sentry_title in title:
        return True
    process_name = window_process_name(window_process_id(hwnd))
    if process_name in _TERMINAL_HOST_PROCESSES:
        return True
    try:
        class_name = win32gui.GetClassName(hwnd).lower()
    except Exception:
        class_name = ""
    return class_name in {"consolewindowclass", "cascadia_hosting_window_class"}


def top_level_window_handle(hwnd: int) -> int:
    if not hwnd:
        return 0
    try:
        root = int(win32gui.GetAncestor(hwnd, getattr(win32con, "GA_ROOT", 2)) or 0)
    except Exception:
        root = 0
    return root or hwnd


def related_window_handles(hwnd: int) -> set[int]:
    if not hwnd:
        return set()
    handles = {hwnd, top_level_window_handle(hwnd)}
    current = top_level_window_handle(hwnd)
    while current:
        try:
            owner = int(win32gui.GetWindow(current, win32con.GW_OWNER) or 0)
        except Exception:
            owner = 0
        if not owner or owner in handles:
            break
        handles.add(owner)
        handles.add(top_level_window_handle(owner))
        current = owner
    return {handle for handle in handles if handle}


def window_process_id(hwnd: int) -> int | None:
    if not hwnd:
        return None
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return int(pid) or None
    except Exception:
        return None


def window_process_name(pid: int | None) -> str:
    if pid is None:
        return "unknown"
    try:
        return psutil.Process(pid).name().lower()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return "unknown"


def is_sentry_window(hwnd: int, title: str = "") -> bool:
    if hwnd in _SENTRY_WINDOW_HANDLES or top_level_window_handle(hwnd) in _SENTRY_WINDOW_HANDLES:
        return True
    return bool(_SENTRY_WINDOW_TITLE and _SENTRY_WINDOW_TITLE in title)


def close_all_visible_applications() -> dict[str, Any]:
    """Gracefully close every visible user application except Sentry and shell windows."""
    if _SENTRY_WINDOW_HANDLE is None or not win32gui.IsWindow(_SENTRY_WINDOW_HANDLE):
        return {
            "ok": False,
            "targeted": 0,
            "closed": 0,
            "remaining": [],
            "skipped": [],
            "message": "Sentry could not safely identify its own terminal window, so nothing was closed.",
        }

    remaining, skipped = visible_user_application_windows()
    if not any(window.get("reason") == "sentry" for window in skipped):
        return {
            "ok": False,
            "targeted": 0,
            "closed": 0,
            "remaining": [],
            "skipped": [window_summary(window) for window in skipped],
            "message": (
                "Sentry could not verify its visible terminal in the protected window set, "
                "so nothing was closed."
            ),
        }

    targeted_by_handle: dict[int, dict[str, Any]] = {}
    skipped_by_handle: dict[int, dict[str, Any]] = {
        int(window["hwnd"]): window for window in skipped
    }
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        for window in remaining:
            hwnd = int(window["hwnd"])
            if hwnd in targeted_by_handle:
                continue
            targeted_by_handle[hwnd] = window
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        if not remaining:
            break
        time.sleep(0.10)
        remaining, skipped = visible_user_application_windows()
        for window in skipped:
            skipped_by_handle[int(window["hwnd"])] = window

    remaining, skipped = visible_user_application_windows()
    for window in skipped:
        skipped_by_handle[int(window["hwnd"])] = window
    remaining_handles = {int(window["hwnd"]) for window in remaining}
    closed = len(set(targeted_by_handle) - remaining_handles)
    safe_remaining = [window_summary(window) for window in remaining]
    return {
        "ok": not safe_remaining,
        "targeted": len(targeted_by_handle),
        "closed": closed,
        "remaining": safe_remaining,
        "skipped": [window_summary(window) for window in skipped_by_handle.values()],
        "message": (
            "All visible user applications closed."
            if not safe_remaining
            else "Some applications are still open, possibly because they need user attention."
        ),
    }


def visible_user_application_windows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def callback(hwnd: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return True
        pid = window_process_id(hwnd) or 0
        process_name = window_process_name(pid)
        try:
            class_name = win32gui.GetClassName(hwnd).lower()
        except Exception:
            class_name = ""
        window = {
            "hwnd": hwnd,
            "pid": pid,
            "process": process_name,
            "title": title,
            "class_name": class_name,
        }
        if is_sentry_window(hwnd, title) or pid == os.getpid():
            window["reason"] = "sentry"
            skipped.append(window)
        elif class_name in _PROTECTED_WINDOW_CLASSES or process_name in _PROTECTED_SHELL_PROCESSES:
            window["reason"] = "windows_shell"
            skipped.append(window)
        else:
            targets.append(window)
        return True

    win32gui.EnumWindows(callback, None)
    return targets, skipped


def window_summary(window: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "pid": int(window.get("pid") or 0),
        "process": str(window.get("process") or "unknown"),
        "title": str(window.get("title") or "untitled"),
    }
    if window.get("reason"):
        summary["reason"] = str(window["reason"])
    return summary


def sleep_computer() -> dict[str, Any]:
    """Suspend Windows without forcing applications to close."""
    set_suspend_state = ctypes.windll.powrprof.SetSuspendState
    set_suspend_state.argtypes = [ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
    set_suspend_state.restype = ctypes.c_bool
    success = bool(set_suspend_state(False, False, False))
    return {
        "ok": success,
        "state": "resumed" if success else "sleep_failed",
        "message": "The computer resumed from sleep." if success else "Windows did not enter sleep mode.",
    }


def lock_computer() -> dict[str, Any]:
    """Lock the current Windows workstation without closing applications."""
    success = bool(ctypes.windll.user32.LockWorkStation())
    return {
        "ok": success,
        "state": "locked" if success else "lock_failed",
        "message": "Windows was locked." if success else "Windows did not accept the lock request.",
    }


def shutdown_computer() -> dict[str, Any]:
    """Request a graceful Windows shutdown without an implied force flag."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        [
            "shutdown.exe",
            "/s",
            "/t",
            "0",
            "/c",
            "Sentry completed the approved shutdown routine.",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        check=False,
    )
    success = completed.returncode == 0
    return {
        "ok": success,
        "dispatched": success,
        "delay_seconds": 0,
        "message": (
            "Windows accepted the shutdown request."
            if success
            else f"Windows rejected the shutdown request with exit code {completed.returncode}."
        ),
    }


def open_application(application: str) -> dict[str, Any]:
    """Open one allowlisted Windows application."""
    allowed = validate_allowed_application(application)
    command = resolve_allowed_command(allowed.command)
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    return {
        "ok": True,
        "application": allowed.name,
        "message": f"{allowed.name} opened.",
    }


def close_application(application: str) -> dict[str, Any]:
    """Close visible windows for one allowlisted application."""
    allowed = validate_allowed_application(application)
    executable_names = allowed_process_names(allowed.command)
    title_keywords = allowed_title_keywords(allowed.name)
    windows = visible_windows_for_process_names(executable_names, title_keywords)

    appearance_deadline = time.monotonic() + 0.8
    while not windows and time.monotonic() < appearance_deadline:
        time.sleep(0.08)
        windows = visible_windows_for_process_names(executable_names, title_keywords)

    if not windows:
        return {
            "ok": True,
            "application": allowed.name,
            "closed": 0,
            "targeted": 0,
            "remaining": [],
            "message": f"No visible {allowed.name} windows were found.",
        }

    targeted: dict[int, dict[str, Any]] = {}
    last_signal: dict[int, float] = {}
    quiet_since: float | None = None
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        current = visible_windows_for_process_names(executable_names, title_keywords)
        if current:
            quiet_since = None
            for window in current:
                hwnd = int(window["hwnd"])
                targeted[hwnd] = window
                if hwnd not in last_signal or now - last_signal[hwnd] >= 0.4:
                    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                    last_signal[hwnd] = now
        elif quiet_since is None:
            quiet_since = now
        elif now - quiet_since >= 0.6:
            break
        time.sleep(0.08)

    remaining = visible_windows_for_process_names(executable_names, title_keywords)
    remaining_handles = {int(item["hwnd"]) for item in remaining}
    closed = len(set(targeted) - remaining_handles)
    return {
        "ok": not remaining,
        "application": allowed.name,
        "closed": closed,
        "targeted": len(targeted),
        "remaining": [{"pid": item["pid"], "title": item["title"]} for item in remaining],
        "message": f"{allowed.name} windows closed." if not remaining else f"Some {allowed.name} windows are still open.",
    }


def list_application_windows(application: str) -> dict[str, Any]:
    """List visible windows for one allowlisted application."""
    allowed = validate_allowed_application(application)
    executable_names = allowed_process_names(allowed.command)
    windows = visible_windows_for_process_names(executable_names, allowed_title_keywords(allowed.name))
    return {
        "ok": True,
        "application": allowed.name,
        "count": len(windows),
        "windows": [
            {"pid": item["pid"], "process": item["process"], "title": item["title"]}
            for item in windows
        ],
    }


def notepad_open(text: str = "") -> dict[str, Any]:
    """Open the agent-managed Notepad note, optionally replacing its text first."""
    safe_text = validate_notepad_text(text)
    note_path = ensure_current_notepad_file()
    if safe_text:
        note_path.write_text(safe_text, encoding="utf-8")
    process = launch_notepad_file(note_path)
    return {
        "ok": True,
        "path": str(note_path),
        "pid": process.pid,
        "characters": len(note_path.read_text(encoding="utf-8")),
        "message": "Managed Notepad note opened.",
    }


def open_notepad_with_text(text: str) -> dict[str, Any]:
    """Create a controlled note file and open it in Notepad."""
    safe_text = validate_notepad_text(text)
    note_dir = CACHE_DIR / "notepad"
    note_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    note_path = note_dir / f"portable_ai_note_{stamp}.txt"
    note_path.write_text(safe_text, encoding="utf-8")
    set_active_notepad_file(note_path)

    process = launch_notepad_file(note_path)
    return {
        "ok": True,
        "path": str(note_path),
        "pid": process.pid,
        "characters": len(safe_text),
        "message": "Notepad opened with the requested text.",
    }


def notepad_type(text: str) -> dict[str, Any]:
    """Append text to the agent-managed note and open it in Notepad."""
    safe_text = validate_notepad_text(text)
    note_path = ensure_current_notepad_file()
    close_info = close_application("notepad")
    if close_info["remaining"]:
        return {
            "ok": False,
            "path": str(note_path),
            "closed_before_reopen": close_info["closed"],
            "remaining_before_reopen": close_info["remaining"],
            "message": "Notepad did not close cleanly, so the note was not changed.",
        }

    existing = note_path.read_text(encoding="utf-8")
    separator = "" if not existing or existing.endswith(("\n", "\r")) else "\n"
    note_path.write_text(existing + separator + safe_text, encoding="utf-8")
    process = launch_notepad_file(note_path)
    return {
        "ok": True,
        "path": str(note_path),
        "pid": process.pid,
        "closed_before_reopen": close_info["closed"],
        "remaining_before_reopen": close_info["remaining"],
        "characters_added": len(safe_text),
        "total_characters": len(note_path.read_text(encoding="utf-8")),
        "message": "Text added to the managed Notepad note.",
    }


def notepad_save() -> dict[str, Any]:
    """Report the saved path for the agent-managed Notepad note."""
    note_path = ensure_current_notepad_file()
    return {
        "ok": True,
        "path": str(note_path),
        "characters": len(note_path.read_text(encoding="utf-8")),
        "message": "The managed Notepad note is already saved on disk.",
    }


def notepad_save_as(filename: str) -> dict[str, Any]:
    """Save the agent-managed Notepad note under a simple filename."""
    safe_filename = validate_notepad_filename(filename)
    note_path = ensure_current_notepad_file()
    target_path = note_path.parent / safe_filename
    target_path.write_text(note_path.read_text(encoding="utf-8"), encoding="utf-8")
    set_active_notepad_file(target_path)
    return {
        "ok": True,
        "path": str(target_path),
        "characters": len(target_path.read_text(encoding="utf-8")),
        "message": "The managed Notepad note was saved as the requested file.",
    }


def notepad_close() -> dict[str, Any]:
    """Close visible allowlisted Notepad windows."""
    return close_application("notepad")


def rename_file(path: str, new_name: str, overwrite: bool = False) -> dict[str, Any]:
    """Rename one file within a managed folder without changing its directory."""
    source = resolve_managed_file(path)
    safe_name = validate_file_name(new_name, default_suffix=source.suffix)
    destination = resolve_managed_file(str(source.with_name(safe_name)), must_exist=False)
    if source == destination:
        return {
            "ok": True,
            "source": str(source),
            "path": str(destination),
            "renamed": False,
            "message": "The file already has the requested name.",
        }
    prepare_file_destination(destination, overwrite)
    source.replace(destination)
    update_active_notepad_after_move(source, destination)
    return {
        "ok": True,
        "source": str(source),
        "path": str(destination),
        "renamed": True,
        "message": "File renamed.",
    }


def copy_file(source: str, destination: str, overwrite: bool = False) -> dict[str, Any]:
    """Copy one file between managed folders, preserving its metadata."""
    source_path = resolve_managed_file(source)
    destination_path = resolve_file_destination(destination, source_path)
    if source_path == destination_path:
        raise ToolError("Source and destination refer to the same file.")
    prepare_file_destination(destination_path, overwrite)
    shutil.copy2(source_path, destination_path)
    return {
        "ok": True,
        "source": str(source_path),
        "path": str(destination_path),
        "bytes": destination_path.stat().st_size,
        "message": "File copied.",
    }


def move_file(source: str, destination: str, overwrite: bool = False) -> dict[str, Any]:
    """Move or cut one file between managed folders."""
    source_path = resolve_managed_file(source)
    destination_path = resolve_file_destination(destination, source_path)
    if source_path == destination_path:
        raise ToolError("Source and destination refer to the same file.")
    prepare_file_destination(destination_path, overwrite)
    shutil.move(str(source_path), str(destination_path))
    update_active_notepad_after_move(source_path, destination_path)
    return {
        "ok": True,
        "source": str(source_path),
        "path": str(destination_path),
        "message": "File moved.",
    }


def delete_file(path: str) -> dict[str, Any]:
    """Permanently delete one file in a managed folder after host confirmation."""
    target = resolve_managed_file(path)
    size = target.stat().st_size
    target.unlink()
    reset_active_notepad_after_delete(target)
    return {
        "ok": True,
        "path": str(target),
        "deleted_bytes": size,
        "message": "File permanently deleted.",
    }


def create_directory(path: str) -> dict[str, Any]:
    """Create a directory and any missing parents inside the managed workspace."""
    target = resolve_managed_directory(path)
    existed = target.exists()
    target.mkdir(parents=True, exist_ok=True)
    return {
        "ok": True,
        "path": str(target),
        "relative_path": workspace_relative_path(target),
        "created": not existed,
        "message": "Directory created." if not existed else "Directory already exists.",
    }


def delete_directory(path: str) -> dict[str, Any]:
    """Permanently delete a directory tree inside the managed workspace."""
    target = resolve_managed_directory(path, must_exist=True, allow_root=False)
    summary = directory_summary_data(target)
    shutil.rmtree(target)
    return {
        "ok": True,
        "path": str(target),
        "relative_path": workspace_relative_path(target),
        "deleted_directories": summary["directories"] + 1,
        "deleted_files": summary["files"],
        "deleted_bytes": summary["total_bytes"],
        "scan_truncated": summary["truncated"],
        "message": "Directory and its contents were permanently deleted.",
    }


def list_directory(
    path: str = ".",
    recursive: bool = False,
    include: str = "all",
    output_format: str = "compact",
) -> dict[str, Any]:
    """List bounded directory metadata without reading file contents."""
    if include not in {"all", "directories", "files"}:
        raise ToolError("Directory listing include must be all, directories, or files.")
    if output_format not in {"compact", "tree"}:
        raise ToolError("Directory listing output format must be compact or tree.")
    target = resolve_inspection_directory(path)
    entries, truncated, inaccessible = scan_directory_entries(
        target,
        recursive=recursive,
        limit=DIRECTORY_LIST_LIMIT,
        include=include,
    )
    return {
        "ok": True,
        "path": str(target),
        "relative_path": directory_display_path(target),
        "recursive": recursive,
        "include": include,
        "output_format": output_format,
        "count": len(entries),
        "truncated": truncated,
        "inaccessible": inaccessible,
        "entries": entries,
        "message": "Directory contents listed.",
    }


def summarize_directory(path: str = ".") -> dict[str, Any]:
    """Summarize a directory tree from names and metadata without reading file contents."""
    target = resolve_inspection_directory(path)
    summary = directory_summary_data(target)
    return {
        "ok": True,
        "path": str(target),
        "relative_path": directory_display_path(target),
        **summary,
        "message": "Directory summary generated from file metadata.",
    }


def directory_summary_data(path: Path) -> dict[str, Any]:
    entries, truncated, inaccessible = scan_directory_entries(
        path,
        recursive=True,
        limit=DIRECTORY_SCAN_LIMIT,
    )
    files = [entry for entry in entries if entry["type"] == "file"]
    directories = [entry for entry in entries if entry["type"] == "directory"]
    links = [entry for entry in entries if entry["type"] == "link"]
    extensions = Counter(
        Path(str(entry["name"])).suffix.lower() or "[no extension]"
        for entry in files
    )
    largest_files = sorted(
        files,
        key=lambda item: int(item.get("size_bytes", 0)),
        reverse=True,
    )[:5]
    return {
        "directories": len(directories),
        "files": len(files),
        "links": len(links),
        "total_bytes": sum(int(entry.get("size_bytes", 0)) for entry in files),
        "extensions": dict(extensions.most_common(10)),
        "largest_files": [
            {"path": entry["path"], "size_bytes": entry.get("size_bytes", 0)}
            for entry in largest_files
        ],
        "scanned_entries": len(entries),
        "truncated": truncated,
        "inaccessible": inaccessible,
    }


def scan_directory_entries(
    path: Path,
    *,
    recursive: bool,
    limit: int,
    include: str = "all",
) -> tuple[list[dict[str, Any]], bool, int]:
    entries: list[dict[str, Any]] = []
    directories: deque[Path] = deque([path])
    inaccessible = 0
    scanned = 0
    while directories:
        current = directories.popleft()
        try:
            children = sorted(os.scandir(current), key=lambda entry: entry.name.lower())
        except OSError:
            inaccessible += 1
            continue

        for child in children:
            scanned += 1
            if scanned > DIRECTORY_SCAN_LIMIT:
                return entries, True, inaccessible
            child_path = Path(child.path)
            is_junction = getattr(child_path, "is_junction", lambda: False)
            try:
                if child.is_symlink() or is_junction():
                    entry_type = "link"
                elif child.is_dir(follow_symlinks=False):
                    entry_type = "directory"
                elif child.is_file(follow_symlinks=False):
                    entry_type = "file"
                else:
                    entry_type = "other"
                size = child.stat(follow_symlinks=False).st_size if entry_type == "file" else 0
            except OSError:
                inaccessible += 1
                continue

            relative = child_path.relative_to(path)
            should_include = (
                include == "all"
                or (include == "directories" and entry_type == "directory")
                or (include == "files" and entry_type == "file")
            )
            if should_include:
                if len(entries) >= limit:
                    return entries, True, inaccessible
                entries.append(
                    {
                        "name": child.name,
                        "path": str(relative),
                        "type": entry_type,
                        "size_bytes": size,
                    }
                )
            if recursive and entry_type == "directory":
                directories.append(child_path)
    return entries, False, inaccessible


def workspace_relative_path(path: Path) -> str:
    root = managed_workspace_root()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ToolError("Resolved path escaped the managed workspace.") from exc
    return str(relative) if relative.parts else "."


def directory_display_path(path: Path) -> str:
    root = managed_workspace_root()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return str(relative) if relative.parts else "."


def close_active_notepad_sessions() -> dict[str, Any]:
    pids = active_notepad_pids()
    if not pids:
        return {
            "closed": 0,
            "remaining": [],
            "message": "No agent-launched Notepad windows are open.",
        }

    close_notepad_windows_for_pids(pids)
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        if not any(process_is_running(pid) for pid in pids):
            break
        time.sleep(0.05)

    still_open = {pid for pid in pids if process_is_running(pid)}
    _NOTEPAD_PIDS.difference_update(pids - still_open)
    return {
        "closed": len(pids - still_open),
        "remaining": sorted(still_open),
        "message": "Agent-launched Notepad windows closed." if not still_open else "Some Notepad windows are still open.",
    }


def ensure_current_notepad_file() -> Path:
    note_path = _ACTIVE_NOTEPAD_FILE
    note_dir = note_path.parent
    note_dir.mkdir(parents=True, exist_ok=True)
    if not note_path.exists():
        note_path.write_text("", encoding="utf-8")
    return note_path


def set_active_notepad_file(path: Path) -> None:
    global _ACTIVE_NOTEPAD_FILE
    _ACTIVE_NOTEPAD_FILE = path.resolve(strict=False)


def update_active_notepad_after_move(source: Path, destination: Path) -> None:
    if source.resolve(strict=False) == _ACTIVE_NOTEPAD_FILE.resolve(strict=False):
        set_active_notepad_file(destination)


def reset_active_notepad_after_delete(path: Path) -> None:
    global _ACTIVE_NOTEPAD_FILE
    if path.resolve(strict=False) == _ACTIVE_NOTEPAD_FILE.resolve(strict=False):
        _ACTIVE_NOTEPAD_FILE = _CURRENT_NOTEPAD_FILE


def resolve_file_destination(destination: str, source: Path) -> Path:
    return resolve_managed_file_destination(destination, source_name=source.name)


def prepare_file_destination(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise ToolError(f"Destination file already exists: {path}")
    path.unlink()


def launch_notepad_file(path: Path) -> subprocess.Popen[bytes]:
    notepad = resolve_executable("notepad.exe")
    process = subprocess.Popen(
        [str(notepad), str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    _NOTEPAD_PIDS.add(process.pid)
    return process


def active_notepad_pids() -> set[int]:
    active: set[int] = set()
    for pid in set(_NOTEPAD_PIDS):
        if process_is_running(pid):
            active.add(pid)
    _NOTEPAD_PIDS.clear()
    _NOTEPAD_PIDS.update(active)
    return active


def process_is_running(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def close_notepad_windows_for_pids(pids: set[int]) -> None:
    def callback(hwnd: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        _, window_pid = win32process.GetWindowThreadProcessId(hwnd)
        if window_pid in pids:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        return True

    win32gui.EnumWindows(callback, None)


def allowed_process_names(command: tuple[str, ...]) -> set[str]:
    executable = Path(command[0]).name.lower()
    names = {executable}
    if executable == "calc.exe":
        names.add("calculatorapp.exe")
    if executable == "whatsapp.exe":
        names.update({"whatsapp.exe", "whatsappbeta.exe"})
    if executable == "outlook.exe":
        names.add("olk.exe")
    return names


def allowed_title_keywords(application: str) -> set[str]:
    if application == "whatsapp":
        return {"whatsapp"}
    return set()


def visible_windows_for_process_names(process_names: set[str], title_keywords: set[str] | None = None) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    title_keywords = title_keywords or set()

    def callback(hwnd: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return True
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            process_name = psutil.Process(pid).name().lower()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return True
        title_lower = title.lower()
        if process_name in process_names or any(keyword in title_lower for keyword in title_keywords):
            windows.append({"hwnd": hwnd, "pid": pid, "process": process_name, "title": title})
        return True

    win32gui.EnumWindows(callback, None)
    return windows


def resolve_allowed_command(command: tuple[str, ...]) -> list[str]:
    executable = command[0]
    resolved = resolve_executable(executable)
    if executable.lower() == "whatsapp.exe" and str(resolved).lower() == "whatsapp.exe":
        return [resolve_executable("explorer.exe"), r"shell:AppsFolder\5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"]
    app_id = _APPLICATION_SHELL_IDS.get(executable.lower())
    if app_id and str(resolved).lower() == executable.lower():
        return [resolve_executable("explorer.exe"), f"shell:AppsFolder\\{app_id}"]
    return [str(resolved), *command[1:]]


def resolve_executable(executable: str) -> str:
    found = shutil.which(executable)
    if found:
        return found

    app_path = resolve_app_path(executable)
    if app_path:
        return str(app_path)

    common_path = resolve_common_browser_path(executable)
    if common_path:
        return str(common_path)

    common_app_path = resolve_common_app_path(executable)
    if common_app_path:
        return str(common_app_path)

    # Return the original value so Windows can still resolve built-ins through
    # normal CreateProcess behavior; the caller will receive a clear OS error if not.
    return executable


def resolve_common_app_path(executable: str) -> Path | None:
    executable_lower = executable.lower()
    local_app_data = Path(os.environ.get("LocalAppData", ""))
    program_files = Path(os.environ.get("ProgramFiles", ""))
    program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", ""))

    candidates: dict[str, list[Path]] = {
        "whatsapp.exe": [
            local_app_data / "WhatsApp" / "WhatsApp.exe",
            program_files / "WindowsApps" / "5319275A.WhatsAppDesktop_cv1g1gvanyjgm" / "WhatsApp.exe",
        ],
        "notion.exe": [local_app_data / "Programs" / "Notion" / "Notion.exe"],
        "code.exe": [
            local_app_data / "Programs" / "Microsoft VS Code" / "Code.exe",
            program_files / "Microsoft VS Code" / "Code.exe",
        ],
        "githubdesktop.exe": [local_app_data / "GitHubDesktop" / "GitHubDesktop.exe"],
        "outlook.exe": [
            program_files / "Microsoft Office" / "root" / "Office16" / "OUTLOOK.EXE",
            program_files_x86 / "Microsoft Office" / "root" / "Office16" / "OUTLOOK.EXE",
        ],
        "mongodbcompass.exe": [
            local_app_data / "Programs" / "mongodb-compass" / "MongoDBCompass.exe",
            program_files / "MongoDB Compass" / "MongoDBCompass.exe",
        ],
        "termius.exe": [
            local_app_data / "Programs" / "Termius" / "Termius.exe",
            program_files / "Termius" / "Termius.exe",
        ],
    }
    existing = [path for path in candidates.get(executable_lower, []) if path.exists()]

    versioned_roots: dict[str, Path] = {
        "githubdesktop.exe": local_app_data / "GitHubDesktop",
        "mongodbcompass.exe": local_app_data / "MongoDBCompass",
    }
    versioned_root = versioned_roots.get(executable_lower)
    if versioned_root and versioned_root.exists():
        existing.extend(
            path
            for path in versioned_root.glob(f"app-*/{executable}")
            if path.exists()
        )

    if not existing:
        return None
    existing.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return existing[0]


def resolve_app_path(executable: str) -> Path | None:
    subkey = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{executable}"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "")
        except OSError:
            continue
        path = Path(str(value))
        if path.exists() and path.name.lower() == executable.lower():
            return path
    return None


def resolve_common_browser_path(executable: str) -> Path | None:
    if executable.lower() != "chrome.exe":
        return None

    candidates = [
        Path(os.environ.get("ProgramFiles", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("LocalAppData", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def open_url(url: str) -> dict[str, Any]:
    """Open an http or https URL with the default browser."""
    safe_url = validate_url(url)
    opened = webbrowser.open(safe_url, new=2, autoraise=True)
    return {
        "ok": bool(opened),
        "url": safe_url,
        "message": "URL opened." if opened else "Browser did not confirm the URL was opened.",
    }


def youtube_search_open(query: str) -> dict[str, Any]:
    """Search the web for a YouTube result and open the best safe YouTube URL."""
    cleaned_query = normalize_youtube_query(query)
    if not cleaned_query:
        return {"ok": False, "message": "YouTube search query cannot be empty."}

    known_result = known_youtube_result(cleaned_query)
    if known_result is not None:
        opened = webbrowser.open(known_result["url"], new=2, autoraise=True)
        return {
            "ok": bool(opened),
            "mode": "known_result",
            "query": cleaned_query,
            "title": known_result["title"],
            "url": known_result["url"],
            "message": "Known YouTube result opened." if opened else "Browser did not confirm the result was opened.",
        }

    candidates = search_youtube_candidates(cleaned_query)
    if candidates:
        best = candidates[0]
        opened = webbrowser.open(best["url"], new=2, autoraise=True)
        return {
            "ok": bool(opened),
            "mode": "best_result",
            "query": cleaned_query,
            "title": best["title"],
            "url": best["url"],
            "message": "Best YouTube result opened." if opened else "Browser did not confirm the result was opened.",
        }

    fallback_url = f"https://www.youtube.com/results?search_query={quote_plus(cleaned_query)}"
    opened = webbrowser.open(fallback_url, new=2, autoraise=True)
    return {
        "ok": bool(opened),
        "mode": "search_results",
        "query": cleaned_query,
        "url": fallback_url,
        "message": "YouTube search results opened because no direct result could be fetched.",
    }


def normalize_youtube_query(query: str) -> str:
    cleaned = " ".join(query.strip().split())
    cleaned = re.sub(r"\bkarapathy\b", "karpathy", cleaned, flags=re.IGNORECASE)
    return cleaned


def known_youtube_result(query: str) -> dict[str, str] | None:
    lower = query.lower()
    if "learnlinuxtv" in lower and "channel" in lower:
        return {
            "kind": "channel",
            "title": "Learn Linux TV - YouTube channel",
            "url": "https://www.youtube.com/@LearnLinuxTV",
        }
    if "makemore" in lower and "karpathy" in lower and ("playlist" in lower or "series" in lower):
        return {
            "kind": "playlist",
            "title": "karpathy building makemore - YouTube playlist",
            "url": "https://www.youtube.com/playlist?list=PLAV29EAhk_mX13BqhzdlgM8zkHwpcRajt",
        }
    return None


def search_youtube_candidates(query: str) -> list[dict[str, str]]:
    youtube_results = search_youtube_page_candidates(query)

    search_query = f"site:youtube.com {query}"
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
    headers = {"User-Agent": "Mozilla/5.0 PortableAI/0.1"}
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError:
        duck_results: list[dict[str, str]] = []
    else:
        duck_results = extract_youtube_results(response.text)

    combined: list[dict[str, str]] = []
    for candidate in [*duck_results, *youtube_results]:
        add_unique_result(combined, candidate)

    combined.sort(key=lambda item: youtube_rank(item, query))
    return combined


def search_youtube_page_candidates(query: str) -> list[dict[str, str]]:
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0 PortableAI/0.1"}
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError:
        return []

    data = extract_yt_initial_data(response.text)
    if data is None:
        return []

    candidates: list[dict[str, str]] = []
    collect_youtube_renderers(data, candidates)
    candidates.sort(key=lambda item: youtube_rank(item, query))
    return candidates


def youtube_rank(item: dict[str, str], query: str) -> tuple[int, str]:
    lower = query.lower()
    kind = item.get("kind", "")
    if "channel" in lower:
        priority = {"channel": 0, "playlist": 1, "video": 2}.get(kind, 3)
    elif "playlist" in lower or "series" in lower:
        priority = {"playlist": 0, "channel": 1, "video": 2}.get(kind, 3)
    else:
        priority = {"video": 0, "playlist": 1, "channel": 2}.get(kind, 3)
    return (priority, item.get("title", "").lower())


def extract_yt_initial_data(page_html: str) -> dict[str, Any] | None:
    marker = "ytInitialData"
    index = page_html.find(marker)
    while index != -1:
        equals = page_html.find("=", index)
        brace = page_html.find("{", equals)
        if equals != -1 and brace != -1:
            blob = extract_balanced_json(page_html, brace)
            if blob:
                try:
                    parsed = json.loads(blob)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
        index = page_html.find(marker, index + len(marker))
    return None


def extract_balanced_json(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def collect_youtube_renderers(value: Any, results: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        if "playlistRenderer" in value:
            renderer = value["playlistRenderer"]
            playlist_id = renderer.get("playlistId")
            title = get_youtube_title(renderer)
            if playlist_id:
                add_unique_result(
                    results,
                    {
                        "kind": "playlist",
                        "title": title or "YouTube playlist",
                        "url": f"https://www.youtube.com/playlist?list={playlist_id}",
                    },
                )
        if "channelRenderer" in value:
            renderer = value["channelRenderer"]
            title = get_youtube_title(renderer)
            channel_url = get_youtube_channel_url(renderer)
            if channel_url:
                add_unique_result(
                    results,
                    {
                        "kind": "channel",
                        "title": title or "YouTube channel",
                        "url": channel_url,
                    },
                )
        if "videoRenderer" in value:
            renderer = value["videoRenderer"]
            video_id = renderer.get("videoId")
            title = get_youtube_title(renderer)
            if video_id:
                add_unique_result(
                    results,
                    {
                        "kind": "video",
                        "title": title or "YouTube video",
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                    },
                )
        for child in value.values():
            collect_youtube_renderers(child, results)
    elif isinstance(value, list):
        for child in value:
            collect_youtube_renderers(child, results)


def get_youtube_title(renderer: dict[str, Any]) -> str:
    title = renderer.get("title")
    if isinstance(title, dict):
        simple = title.get("simpleText")
        if isinstance(simple, str):
            return simple
        runs = title.get("runs")
        if isinstance(runs, list):
            return "".join(run.get("text", "") for run in runs if isinstance(run, dict)).strip()
    return ""


def get_youtube_channel_url(renderer: dict[str, Any]) -> str:
    endpoint = renderer.get("navigationEndpoint")
    if isinstance(endpoint, dict):
        metadata = endpoint.get("commandMetadata")
        if isinstance(metadata, dict):
            web = metadata.get("webCommandMetadata")
            if isinstance(web, dict):
                url = web.get("url")
                if isinstance(url, str) and url.startswith("/"):
                    return "https://www.youtube.com" + url
        browse = endpoint.get("browseEndpoint")
        if isinstance(browse, dict):
            canonical = browse.get("canonicalBaseUrl")
            if isinstance(canonical, str) and canonical.startswith("/"):
                return "https://www.youtube.com" + canonical
            browse_id = browse.get("browseId")
            if isinstance(browse_id, str):
                return "https://www.youtube.com/channel/" + browse_id

    subscriber = renderer.get("subscriberCountText")
    if isinstance(subscriber, dict):
        handle = subscriber.get("simpleText")
        if isinstance(handle, str) and handle.startswith("@"):
            return "https://www.youtube.com/" + handle
    return ""


def add_unique_result(results: list[dict[str, str]], candidate: dict[str, str]) -> None:
    if any(item["url"] == candidate["url"] for item in results):
        return
    results.append(candidate)


def extract_youtube_results(page_html: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for match in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page_html, re.I | re.S):
        raw_url = html.unescape(match.group(1))
        title = clean_html(match.group(2))
        youtube_url = normalize_search_result_url(raw_url)
        if youtube_url is None:
            continue
        if youtube_url in {item["url"] for item in results}:
            continue
        if "/playlist?" in youtube_url:
            kind = "playlist"
        elif "/@" in youtube_url or "/channel/" in youtube_url or "/c/" in youtube_url:
            kind = "channel"
        else:
            kind = "video"
        results.append({"kind": kind, "title": title or youtube_url, "url": youtube_url})
    return results


def normalize_search_result_url(raw_url: str) -> str | None:
    parsed = urlparse(raw_url)
    if parsed.netloc.endswith("duckduckgo.com"):
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            raw_url = unquote(uddg[0])
            parsed = urlparse(raw_url)

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in {"youtube.com", "youtu.be"}:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None

    safe_url = parsed._replace(scheme="https", fragment="").geturl()
    if "/redirect" in parsed.path or parsed.path.startswith("/account") or parsed.path.startswith("/signin"):
        return None
    return safe_url


def clean_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", value)
    return html.unescape(without_tags).strip()


def list_processes() -> dict[str, Any]:
    """Return a concise, non-sensitive list of running processes."""
    processes: list[dict[str, Any]] = []
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            info = proc.info
            name = info.get("name")
            pid = info.get("pid")
            if name and pid is not None:
                processes.append({"pid": pid, "name": name})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

    processes.sort(key=lambda item: (str(item["name"]).lower(), int(item["pid"])))
    return {
        "count": len(processes),
        "returned": min(len(processes), MAX_PROCESS_COUNT),
        "processes": processes[:MAX_PROCESS_COUNT],
    }


def system_information() -> dict[str, Any]:
    """Return basic local system information without collecting personal data."""
    virtual_memory = psutil.virtual_memory()
    cpu_name = platform.processor() or platform.machine() or "unknown"
    return {
        "windows": platform.platform(),
        "cpu": cpu_name,
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(virtual_memory.total / (1024**3), 2),
        "ram_available_gb": round(virtual_memory.available / (1024**3), 2),
        "gpu": "not detected by basic local probe",
    }


def active_window() -> dict[str, Any]:
    """Return the current foreground window title and process name when available."""
    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        return {"ok": False, "title": "", "process": "", "message": "No active window found."}

    title = win32gui.GetWindowText(hwnd)
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    process_name = ""
    try:
        process_name = psutil.Process(pid).name()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        process_name = "unknown"

    return {
        "ok": True,
        "title": title,
        "process": process_name,
    }


def clipboard_read() -> dict[str, Any]:
    """Read text from the Windows clipboard, truncated to a safe size."""
    text = ""
    win32clipboard.OpenClipboard()
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
            text = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()

    truncated = len(text) > CLIPBOARD_READ_LIMIT
    if truncated:
        text = text[:CLIPBOARD_READ_LIMIT]

    return {
        "ok": True,
        "text": text,
        "truncated": truncated,
        "limit": CLIPBOARD_READ_LIMIT,
    }


def clipboard_write(text: str) -> dict[str, Any]:
    """Write text to the Windows clipboard."""
    safe_text = validate_clipboard_text(text)
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, safe_text)
    finally:
        win32clipboard.CloseClipboard()

    return {
        "ok": True,
        "characters": len(safe_text),
        "message": "Clipboard updated.",
    }
