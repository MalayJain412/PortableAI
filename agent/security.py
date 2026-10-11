from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from mcp.server.mcpserver.exceptions import ToolError

from config import ALLOWED_APPLICATIONS, NOTEPAD_TEXT_LIMIT


@dataclass(frozen=True)
class AllowedApplication:
    name: str
    command: tuple[str, ...]


def normalize_name(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ToolError("Value cannot be empty.")
    return normalized


def validate_allowed_application(application: str) -> AllowedApplication:
    normalized = normalize_name(application)
    command = ALLOWED_APPLICATIONS.get(normalized)
    if command is None:
        allowed = ", ".join(sorted(ALLOWED_APPLICATIONS))
        raise ToolError(f"Application not allowed. Allowed applications: {allowed}.")
    return AllowedApplication(name=normalized, command=command)


def validate_url(url: str) -> str:
    cleaned = url.strip()
    if not cleaned:
        raise ToolError("URL cannot be empty.")

    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError("Only http:// and https:// URLs are allowed.")
    if not parsed.netloc:
        raise ToolError("URL must include a host.")
    if any(ch.isspace() for ch in cleaned):
        raise ToolError("URL cannot contain whitespace.")

    return cleaned


def validate_clipboard_text(text: str) -> str:
    if not isinstance(text, str):
        raise ToolError("Clipboard text must be a string.")
    return text


def validate_notepad_text(text: str) -> str:
    if not isinstance(text, str):
        raise ToolError("Notepad text must be a string.")
    if len(text) > NOTEPAD_TEXT_LIMIT:
        raise ToolError(f"Notepad text is too long. Limit: {NOTEPAD_TEXT_LIMIT} characters.")
    return text


def validate_notepad_filename(filename: str) -> str:
    if not isinstance(filename, str):
        raise ToolError("Notepad filename must be a string.")
    cleaned = filename.strip().strip("\"'")
    if not cleaned:
        raise ToolError("Notepad filename cannot be empty.")
    if any(separator in cleaned for separator in ("\\", "/")) or cleaned in {".", ".."}:
        raise ToolError("Notepad filename must be a simple filename, not a path.")
    if any(ch in cleaned for ch in '<>:"/\\|?*'):
        raise ToolError("Notepad filename contains characters that Windows does not allow.")
    if not cleaned.lower().endswith(".txt"):
        cleaned += ".txt"
    return cleaned
