from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Callable

import windows_tools
from mcp_server import create_server


def print_json(label: str, value: Any) -> None:
    print(f"\n## {label}")
    print(json.dumps(value, indent=2, ensure_ascii=False))


def expect_tool_error(label: str, fn: Callable[..., Any], *args: Any) -> None:
    try:
        fn(*args)
    except Exception as exc:
        print_json(label, {"ok": True, "blocked": True, "error": str(exc)})
        return
    raise AssertionError(f"{label} should have failed validation")


async def run_safe_tests() -> None:
    print_json("system_information", windows_tools.system_information())
    print_json("list_processes", windows_tools.list_processes())
    print_json("active_window", windows_tools.active_window())
    print_json("clipboard_write", windows_tools.clipboard_write("Hello from Portable AI"))
    print_json("clipboard_read", windows_tools.clipboard_read())
    expect_tool_error("blocked_application", windows_tools.open_application, "powershell")
    expect_tool_error("blocked_url_scheme", windows_tools.open_url, "javascript:alert(1)")

    server = create_server()
    tools = [tool.name for tool in await server.list_tools()]
    print_json("mcp_registered_tools", tools)


def run_interactive_tests() -> None:
    print_json("open_application_notepad", windows_tools.open_application("notepad"))
    print_json("open_url_example", windows_tools.open_url("https://www.google.com"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PortableAI Phase 2 Windows tools.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Also open Notepad and a browser tab. Safe tests do not launch applications.",
    )
    args = parser.parse_args()

    asyncio.run(run_safe_tests())
    if args.interactive:
        run_interactive_tests()


if __name__ == "__main__":
    main()
