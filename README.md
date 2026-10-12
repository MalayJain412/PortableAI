# PortableAI

PortableAI is a local Windows terminal assistant designed to run from this folder, including from a portable drive. It uses a local `llamafile` model server and a controlled set of Windows tools so the assistant can answer questions, open apps, manage simple notes, inspect system state, and work with directories without running arbitrary shell commands.

The project is built around local-first operation: the chat model runs on `127.0.0.1`, the terminal UI runs from the portable Python runtime, and Windows actions are limited to validated allowlisted tools.

## Quick Start

Double-click:

```text
START.bat
```

`START.bat` checks the local model server at:

```text
http://127.0.0.1:8080
```

If the model server is not running, it starts:

```text
models\Qwen3.5-0.8B-Q8_0.exe
```

It then waits for `/health` and launches the terminal chat UI:

```text
agent\main.py
```

## Stop

Exit the chat with:

```text
/exit
```

To stop the local model server afterward, run:

```text
STOP.bat
```

## Chat Commands

```text
/help
/tools
/clear
/exit
```

- `/help` shows terminal commands.
- `/tools` lists the available Windows tools.
- `/clear` clears the current chat memory and pending confirmations.
- `/exit` exits the terminal assistant.

## Core Capabilities

### Local Chat

PortableAI can answer normal chat questions using the local model server. It streams responses in the terminal and prints basic generation metrics after each response:

- Time to first token
- Total response time
- Approximate tokens per second

### Application Control

PortableAI can open, close, and inspect visible windows for allowlisted Windows applications.

Supported applications:

```text
notepad
calculator
chrome
whatsapp
notion
vscode
github_desktop
outlook
mongodb_compass
termius
```

Examples:

```text
Open Notepad
```

```text
Open Chrome and WhatsApp
```

```text
Close Chrome and WhatsApp
```

```text
List all Chrome windows
```

```text
Open VS Code and GitHub Desktop
```

```text
Open Outlook, MongoDB Compass, and Termius
```

### Notepad Notes

PortableAI includes special handling for Notepad. It can create and manage controlled text files, then open them in Notepad.

Capabilities:

- Open a managed Notepad note
- Open Notepad with exact text
- Append text to the managed note
- Report the saved path
- Save the managed note under a simple `.txt` filename
- Close Notepad windows

Examples:

```text
Open note pad and write in it "Hello this is a test text."
```

```text
Type "Add this as the next line" in Notepad
```

```text
Save Notepad
```

```text
Open Notepad and type "hello this is Testing", then save the file as "testing" and close it
```

### Browser and Web Opening

PortableAI can open explicitly requested `http://` and `https://` URLs in the default browser.

It can also understand simple Google and YouTube requests:

```text
Open Chrome and go to Google
```

```text
Open chatgpt.com on Chrome
```

```text
Search for LangChain on Google
```

```text
Open YouTube on Chrome
```

```text
Open the Makemore playlist by Andrej Karpathy on YouTube
```

For specific YouTube content, it searches for a matching video, playlist, or channel and opens the best safe YouTube result. If no direct result is found, it opens YouTube search results.

### System Inspection

PortableAI can inspect basic local system state without collecting file contents.

Examples:

```text
What applications are currently running?
```

```text
What window am I currently using?
```

```text
What are my basic system specs?
```

Available system tools include:

- Running process list
- Active foreground window title/process
- Basic Windows, CPU, core count, and RAM information

### Clipboard

PortableAI can read and write text clipboard content.

Examples:

```text
Copy "Hello from my portable AI" to the clipboard
```

```text
Read the clipboard
```

Clipboard reads are limited by a configured safety limit.

## Managed Directories

Directory creation and deletion are confined to the managed workspace:

```text
workspace
```

Relative paths are resolved from that folder.

PortableAI can:

- Create directories inside the managed workspace
- Delete directories inside the managed workspace after confirmation
- List existing local directories by metadata
- Summarize existing local directories by metadata
- Show compact or tree-style directory listings

Directory listing and summarization use names, sizes, file types, and metadata only. They do not read file contents.

Examples:

```text
Create a directory named projects\demo
```

```text
List the projects directory
```

```text
What are the folders in E drive?
```

```text
Recursively list projects\demo
```

```text
Summarize the projects directory
```

```text
Delete the projects\demo directory
```

Deletion is recursive and always requires confirmation in chat before the delete action runs.

## Tool Decision Flow

For each user message, the agent decides whether the request is normal chat or a tool action.

The flow is:

1. The assistant asks the local model to interpret the user's intent.
2. The model compares the request against available tools.
3. If the request clearly maps to tools, the agent builds an ordered action plan.
4. The host validates the plan and executes actions sequentially.
5. If a request is ambiguous, the assistant asks for confirmation.
6. If the request has no tool intent, the assistant answers normally.

For typo-like or ambiguous commands, the agent can store the proposed action. If the user replies `yes`, the stored action is executed directly instead of asking the model to reinterpret the word `yes`.

For destructive directory deletion, the complete action plan is stored and nothing executes until the user confirms.

## Safety Model

PortableAI does not run arbitrary model-generated shell commands.

The model can only request allowlisted tools. Each tool validates its arguments before interacting with Windows APIs.

Important safety constraints:

- Only allowlisted desktop applications can be opened or closed.
- URLs must use `http://` or `https://`.
- Directory creation and deletion are limited to the managed workspace.
- The managed workspace root cannot be deleted.
- Symbolic links and directory junctions are rejected in managed paths.
- Network and device paths are blocked for directory inspection.
- Directory listing and summarization do not read file contents.
- Notepad filenames must be simple filenames, not paths.
- Notepad text length is capped.
- Clipboard reads are truncated to a configured limit.
- Destructive directory deletion requires confirmation.

## Project Structure

```text
PortableAI
|-- README.md
|-- START.bat
|-- STOP.bat
|-- tools
|   `-- uv.exe
`-- agent
    |-- main.py
    |-- agent.py
    |-- llm.py
    |-- config.py
    |-- security.py
    |-- windows_tools.py
    |-- langchain_tools.py
    |-- mcp_server.py
    |-- requirements.txt
    |-- test_tools.py
    |-- test_directory_tools.py
    |-- test_langchain_agent.py
    `-- test_windows_close.py
```

Main files:

- `START.bat` starts the local model server if needed and launches the chat UI.
- `STOP.bat` stops the local model server process.
- `agent/main.py` contains the terminal chat loop.
- `agent/agent.py` contains the planner, confirmation flow, memory handling, action reconciliation, and result summaries.
- `agent/llm.py` talks to the local `llamafile` OpenAI-compatible API.
- `agent/windows_tools.py` contains the actual Windows automation functions.
- `agent/langchain_tools.py` wraps Windows functions as LangChain tools with schemas.
- `agent/security.py` validates application names, URLs, paths, Notepad input, and workspace boundaries.
- `agent/mcp_server.py` exposes the safe Windows tools through MCP.
- `agent/config.py` defines paths, model settings, limits, and allowlisted applications.

## Requirements

Python packages are listed in:

```text
agent\requirements.txt
```

The main dependencies include:

- `mcp`
- `httpx`
- `langchain`
- `langchain-openai`
- `langgraph`
- `psutil`
- `pywin32`

The launcher expects a portable Python virtual environment at:

```text
runtime\venv\Scripts\python.exe
```

If required packages are missing, `START.bat` prints an install command using the bundled `tools\uv.exe`.

## Testing

The repo includes tests for:

- Safe tool behavior
- App allowlist enforcement
- Directory path validation
- Directory create/list/summarize/delete behavior
- Planner action repair and confirmation flow
- Compound command ordering
- Window close settling behavior

Test files:

```text
agent\test_tools.py
agent\test_directory_tools.py
agent\test_langchain_agent.py
agent\test_windows_close.py
```

## Notes

PortableAI is Windows-specific. It depends on Windows APIs through `pywin32`, uses `.bat` launchers, and includes application discovery logic for common Windows install locations.

Some web-related actions, such as YouTube result lookup, may require internet access. The core chat model and Windows tool execution are designed to run locally.
