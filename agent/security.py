from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from mcp.server.mcpserver.exceptions import ToolError

from config import ALLOWED_APPLICATIONS, NOTEPAD_TEXT_LIMIT, WORKSPACE_DIR


APPLICATION_ALIASES = {
    "note pad": "notepad",
    "vs code": "vscode",
    "visual studio code": "vscode",
    "github desktop": "github_desktop",
    "mongo compass": "mongodb_compass",
    "mongodb compass": "mongodb_compass",
    "mongodb": "mongodb_compass",
}


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
    requested = normalize_name(application)
    normalized = APPLICATION_ALIASES.get(requested, requested)
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


def validate_directory_path(path: str) -> str:
    if not isinstance(path, str):
        raise ToolError("Directory path must be a string.")
    cleaned = path.strip().strip("\"'")
    if not cleaned:
        raise ToolError("Directory path cannot be empty. Use . for the workspace root.")
    if len(cleaned) > 500:
        raise ToolError("Directory path is too long.")
    if "\x00" in cleaned or any(character in cleaned for character in "*?"):
        raise ToolError("Directory paths cannot contain null bytes or wildcard characters.")
    drive_alias = re.fullmatch(
        r"([a-zA-Z])\s+(?:drive|disk)(?:[\\/](.*))?",
        cleaned,
        flags=re.IGNORECASE,
    )
    if drive_alias:
        suffix = drive_alias.group(2) or ""
        cleaned = f"{drive_alias.group(1).upper()}:\\{suffix}"
    elif re.fullmatch(r"[a-zA-Z]:", cleaned):
        cleaned += "\\"
    return cleaned


def managed_workspace_root() -> Path:
    root = WORKSPACE_DIR.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_managed_directory(
    path: str,
    *,
    must_exist: bool = False,
    allow_root: bool = True,
) -> Path:
    """Resolve a directory path without allowing access outside the managed workspace."""
    cleaned = validate_directory_path(path)
    root = managed_workspace_root()
    requested = Path(cleaned)
    if not requested.is_absolute():
        requested = root / requested

    lexical = Path(os.path.abspath(requested))
    _require_within_workspace(lexical, root)
    _reject_linked_path(lexical, root)

    resolved = lexical.resolve(strict=False)
    _require_within_workspace(resolved, root)
    if not allow_root and resolved == root:
        raise ToolError("The managed workspace root cannot be deleted.")
    if must_exist and not resolved.exists():
        raise ToolError(f"Directory does not exist: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ToolError(f"Path is not a directory: {resolved}")
    return resolved


def resolve_inspection_directory(path: str) -> Path:
    """Resolve a readable local directory, with relative paths based in the managed workspace."""
    cleaned = validate_directory_path(path)
    if cleaned.startswith(("\\\\", "//")):
        raise ToolError("Network and device paths are not allowed.")

    requested = Path(cleaned)
    if re.match(r"^[a-zA-Z]:", cleaned) and not requested.is_absolute():
        raise ToolError("Drive paths must be absolute, for example E:\\ or E:\\Projects.")
    if not requested.is_absolute():
        requested = managed_workspace_root() / requested

    resolved = requested.resolve(strict=False)
    if not resolved.exists():
        raise ToolError(f"Directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ToolError(f"Path is not a directory: {resolved}")
    return resolved


def _require_within_workspace(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ToolError(f"Directory access is limited to the managed workspace: {root}") from exc


def _reject_linked_path(path: Path, root: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if not current.exists():
            break
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            raise ToolError("Symbolic links and directory junctions are not allowed in managed paths.")
