from __future__ import annotations

from mcp.server.mcpserver import MCPServer

import windows_tools


def create_server() -> MCPServer:
    server = MCPServer(
        name="portable-ai-windows-tools",
        title="Portable AI Windows Tools",
        description="Safe allowlisted Windows tools for the portable offline AI agent.",
        version="0.1.0",
        log_level="INFO",
    )

    server.add_tool(windows_tools.open_application)
    server.add_tool(windows_tools.close_application)
    server.add_tool(windows_tools.list_application_windows)
    server.add_tool(windows_tools.notepad_open)
    server.add_tool(windows_tools.open_notepad_with_text)
    server.add_tool(windows_tools.notepad_type)
    server.add_tool(windows_tools.notepad_save)
    server.add_tool(windows_tools.notepad_save_as)
    server.add_tool(windows_tools.notepad_close)
    server.add_tool(windows_tools.open_url)
    server.add_tool(windows_tools.youtube_search_open)
    server.add_tool(windows_tools.list_processes)
    server.add_tool(windows_tools.system_information)
    server.add_tool(windows_tools.active_window)
    server.add_tool(windows_tools.clipboard_read)
    server.add_tool(windows_tools.clipboard_write)
    server.add_tool(windows_tools.create_directory)
    server.add_tool(windows_tools.list_directory)
    server.add_tool(windows_tools.summarize_directory)
    # Raw MCP clients have no trusted confirmation handshake, so deletion remains chat-agent only.

    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
