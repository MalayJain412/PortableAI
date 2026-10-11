from __future__ import annotations

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

from config import CACHE_DIR, CLIPBOARD_READ_LIMIT, MAX_PROCESS_COUNT
from security import validate_allowed_application, validate_clipboard_text, validate_notepad_filename, validate_notepad_text, validate_url


_NOTEPAD_PIDS: set[int] = set()
_CURRENT_NOTEPAD_FILE = CACHE_DIR / "notepad" / "portable_ai_current_note.txt"


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
    if not windows:
        return {
            "ok": True,
            "application": allowed.name,
            "closed": 0,
            "targeted": 0,
            "remaining": [],
            "message": f"No visible {allowed.name} windows were found.",
        }

    for window in windows:
        win32gui.PostMessage(window["hwnd"], win32con.WM_CLOSE, 0, 0)

    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline:
        remaining = visible_windows_for_process_names(executable_names, title_keywords)
        if not remaining:
            break
        time.sleep(0.05)

    remaining = visible_windows_for_process_names(executable_names, title_keywords)
    return {
        "ok": not remaining,
        "application": allowed.name,
        "closed": len(windows) - len(remaining),
        "targeted": len(windows),
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
    return {
        "ok": True,
        "path": str(target_path),
        "characters": len(target_path.read_text(encoding="utf-8")),
        "message": "The managed Notepad note was saved as the requested file.",
    }


def notepad_close() -> dict[str, Any]:
    """Close visible allowlisted Notepad windows."""
    return close_application("notepad")


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
    note_dir = _CURRENT_NOTEPAD_FILE.parent
    note_dir.mkdir(parents=True, exist_ok=True)
    if not _CURRENT_NOTEPAD_FILE.exists():
        _CURRENT_NOTEPAD_FILE.write_text("", encoding="utf-8")
    return _CURRENT_NOTEPAD_FILE


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
    if executable.lower() != "whatsapp.exe":
        return None

    candidates = [
        Path(os.environ.get("LocalAppData", "")) / "WhatsApp" / "WhatsApp.exe",
        Path(os.environ.get("ProgramFiles", "")) / "WindowsApps" / "5319275A.WhatsAppDesktop_cv1g1gvanyjgm" / "WhatsApp.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


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
