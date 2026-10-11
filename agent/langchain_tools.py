from __future__ import annotations

from typing import Any, Literal

from langchain.tools import tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

import windows_tools
from security import (
    resolve_managed_directory,
    resolve_inspection_directory,
    validate_clipboard_text,
    validate_directory_path,
    validate_notepad_filename,
    validate_notepad_text,
    validate_url,
)


ApplicationName = Literal[
    "notepad",
    "calculator",
    "calc",
    "chrome",
    "whatsapp",
    "notion",
    "vscode",
    "github_desktop",
    "outlook",
    "mongodb_compass",
    "termius",
]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyInput(ToolInput):
    pass


class ApplicationInput(ToolInput):
    application: ApplicationName = Field(
        description=(
            "Canonical allowlisted desktop application: notepad, calculator, calc, chrome, whatsapp, notion, "
            "vscode, github_desktop, outlook, mongodb_compass, or termius."
        )
    )


class OptionalNotepadTextInput(ToolInput):
    text: str = Field(default="", description="Optional starting text for the managed Notepad note.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _as_value_error(validate_notepad_text, value)


class NotepadTextInput(ToolInput):
    text: str = Field(description="Exact text to write to the managed Notepad note.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _as_value_error(validate_notepad_text, value)


class NotepadFilenameInput(ToolInput):
    filename: str = Field(description="Simple filename only, without a directory path. A .txt suffix is optional.")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        return _as_value_error(validate_notepad_filename, value)


class UrlInput(ToolInput):
    url: str = Field(description="Complete http:// or https:// URL to open in the default browser.")

    @field_validator("url")
    @classmethod
    def validate_url_value(cls, value: str) -> str:
        return _as_value_error(validate_url, value)


class YoutubeSearchInput(ToolInput):
    query: str = Field(
        min_length=1,
        description="YouTube content, playlist, video, or channel search query. Do not use for opening YouTube home.",
    )


class ClipboardTextInput(ToolInput):
    text: str = Field(description="Exact text to place on the Windows clipboard.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _as_value_error(validate_clipboard_text, value)


class DirectoryPathInput(ToolInput):
    path: str = Field(
        description=(
            "Exact directory path named by the user, inside the managed PortableAI workspace. Preserve multi-word "
            "names and nested separators. If the user names projects, send projects, never .; use . only when the "
            "user explicitly requests the workspace root."
        )
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        cleaned = _as_value_error(validate_directory_path, value)
        _as_value_error(lambda path: str(resolve_managed_directory(path)), cleaned)
        return cleaned


class DeleteDirectoryInput(DirectoryPathInput):
    @field_validator("path")
    @classmethod
    def reject_workspace_root(cls, value: str) -> str:
        _as_value_error(
            lambda path: str(resolve_managed_directory(path, allow_root=False)),
            value,
        )
        return value


class InspectionDirectoryPathInput(ToolInput):
    path: str = Field(
        description=(
            "Exact existing local directory to inspect. Use an absolute drive path such as E:\\ or E:\\Projects; "
            "relative paths are based in the PortableAI workspace. Convert phrases like E drive to E:\\."
        )
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        cleaned = _as_value_error(validate_directory_path, value)
        _as_value_error(lambda path: str(resolve_inspection_directory(path)), cleaned)
        return cleaned


class ListDirectoryInput(InspectionDirectoryPathInput):
    recursive: bool = Field(
        default=False,
        description=(
            "Set true only for explicit recursive, nested, entire-tree, or all-levels requests. Asking for all "
            "items in one directory is not recursive."
        ),
    )
    include: Literal["all", "directories", "files"] = Field(
        default="all",
        description=(
            "Use directories when the user asks for folders, files when the user asks only for files, and all "
            "when both are requested."
        ),
    )
    output_format: Literal["compact", "tree"] = Field(
        default="compact",
        description="Use tree only when the user explicitly requests tree format; otherwise use compact.",
    )


class SummarizeDirectoryInput(InspectionDirectoryPathInput):
    pass


def _as_value_error(validator: Any, value: str) -> str:
    try:
        return validator(value)
    except Exception as exc:
        raise ValueError(str(exc)) from exc


@tool(args_schema=ApplicationInput)
def open_application(application: ApplicationName) -> dict[str, Any]:
    """Open exactly one allowlisted Windows application. Call once for each requested application."""
    return windows_tools.open_application(application)


@tool(args_schema=ApplicationInput)
def close_application(application: ApplicationName) -> dict[str, Any]:
    """Close visible windows for exactly one allowlisted application. Call once per requested application."""
    return windows_tools.close_application(application)


@tool(args_schema=ApplicationInput)
def list_application_windows(application: ApplicationName) -> dict[str, Any]:
    """List visible windows for exactly one allowlisted application."""
    return windows_tools.list_application_windows(application)


@tool(args_schema=OptionalNotepadTextInput)
def notepad_open(text: str = "") -> dict[str, Any]:
    """Open the managed Notepad note, optionally replacing its contents with starting text."""
    return windows_tools.notepad_open(text)


@tool(args_schema=NotepadTextInput)
def open_notepad_with_text(text: str) -> dict[str, Any]:
    """Create a one-shot note containing the exact requested text and open it in Notepad."""
    return windows_tools.open_notepad_with_text(text)


@tool(args_schema=NotepadTextInput)
def notepad_type(text: str) -> dict[str, Any]:
    """Append exact text to the managed Notepad note and reopen the note."""
    return windows_tools.notepad_type(text)


@tool(args_schema=EmptyInput)
def notepad_save() -> dict[str, Any]:
    """Report the current saved path of the managed Notepad note."""
    return windows_tools.notepad_save()


@tool(args_schema=NotepadFilenameInput)
def notepad_save_as(filename: str) -> dict[str, Any]:
    """Save the managed Notepad note under a simple filename in its controlled note directory."""
    return windows_tools.notepad_save_as(filename)


@tool(args_schema=EmptyInput)
def notepad_close() -> dict[str, Any]:
    """Close visible allowlisted Notepad windows."""
    return windows_tools.notepad_close()


@tool(args_schema=UrlInput)
def open_url(url: str) -> dict[str, Any]:
    """Open an explicitly requested HTTP or HTTPS destination; never use as an app-opening fallback."""
    return windows_tools.open_url(url)


@tool(args_schema=YoutubeSearchInput)
def youtube_search_open(query: str) -> dict[str, Any]:
    """Search for a specific YouTube video, playlist, or channel and open the best matching safe result."""
    return windows_tools.youtube_search_open(query)


@tool(args_schema=EmptyInput)
def list_processes() -> dict[str, Any]:
    """Return a concise non-sensitive list of running processes."""
    return windows_tools.list_processes()


@tool(args_schema=EmptyInput)
def system_information() -> dict[str, Any]:
    """Return basic local Windows, processor, core-count, and memory information."""
    return windows_tools.system_information()


@tool(args_schema=EmptyInput)
def active_window() -> dict[str, Any]:
    """Return the title and process name of the current foreground window."""
    return windows_tools.active_window()


@tool(args_schema=EmptyInput)
def clipboard_read() -> dict[str, Any]:
    """Read text from the Windows clipboard up to the configured safety limit."""
    return windows_tools.clipboard_read()


@tool(args_schema=ClipboardTextInput)
def clipboard_write(text: str) -> dict[str, Any]:
    """Write exact text to the Windows clipboard."""
    return windows_tools.clipboard_write(text)


@tool(args_schema=DirectoryPathInput)
def create_directory(path: str) -> dict[str, Any]:
    """Create one directory, including missing parents, inside the managed PortableAI workspace."""
    return windows_tools.create_directory(path)


@tool(args_schema=DeleteDirectoryInput)
def delete_directory(path: str) -> dict[str, Any]:
    """Permanently delete one directory and all its contents. The host must obtain user confirmation first."""
    return windows_tools.delete_directory(path)


@tool(args_schema=ListDirectoryInput)
def list_directory(
    path: str = ".",
    recursive: bool = False,
    include: Literal["all", "directories", "files"] = "all",
    output_format: Literal["compact", "tree"] = "compact",
) -> dict[str, Any]:
    """List one directory's direct children, or its tree when recursion is explicitly requested; never reads contents."""
    return windows_tools.list_directory(path, recursive, include, output_format)


@tool(args_schema=SummarizeDirectoryInput)
def summarize_directory(path: str = ".") -> dict[str, Any]:
    """Summarize folder counts, file counts, sizes, and types without reading file contents."""
    return windows_tools.summarize_directory(path)


LANGCHAIN_TOOLS: list[BaseTool] = [
    open_application,
    close_application,
    list_application_windows,
    notepad_open,
    open_notepad_with_text,
    notepad_type,
    notepad_save,
    notepad_save_as,
    notepad_close,
    open_url,
    youtube_search_open,
    list_processes,
    system_information,
    active_window,
    clipboard_read,
    clipboard_write,
    create_directory,
    delete_directory,
    list_directory,
    summarize_directory,
]

TOOL_REGISTRY: dict[str, BaseTool] = {item.name: item for item in LANGCHAIN_TOOLS}

