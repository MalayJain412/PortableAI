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
/voice
/text
/listen
/voice-status
/clear
/cancel
/exit
```

- `/help` shows terminal commands.
- `/tools` lists the available Windows tools.
- `/voice` switches to hands-free Alexa wake-word input.
- `/text` returns to keyboard input.
- `/listen` waits for Alexa and captures one command without changing the current input mode.
- `/voice-status` checks voice dependencies, model files, and the local audio server.
- `/clear` clears the current chat memory and pending confirmations.
- `/cancel` cancels a pending safety-sensitive plan.
- `/exit` exits the terminal assistant.

## Core Capabilities

### Local Chat

PortableAI can answer normal chat questions using the local model server. It streams responses in the terminal and prints basic generation metrics after each response:

- Time to first token
- Total response time
- Approximate tokens per second

### Alexa Voice Input

Type `/voice`, say **Alexa**, and then speak one command. Sentry records until two seconds of silence, transcribes the command locally with LFM3-Omni-2B through llama.cpp on port `8081`, and sends the resulting text through the same agent and tool-calling flow used by keyboard input.

The wake word is currently fixed to `alexa`. Press `Ctrl+C` while Sentry is waiting for Alexa, or say **Alexa, switch to text mode**, to return to keyboard input. The audio server starts only when voice input is first used and is stopped when Sentry exits if Sentry launched it.

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
- Reopen the same active managed note
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

## Managed Files

PortableAI can rename, copy, move/cut, and delete files inside these controlled locations:

- `workspace`
- `cache\notepad`

File changes outside those folders are rejected. Existing destination files are not overwritten unless the user explicitly asks to overwrite or replace them. File deletion always requires confirmation.

Examples:

```text
Rename "F:\PortableAI\cache\notepad\portable_ai_current_note.txt" as nagpur-expenses
```

```text
Copy expenses.txt to backup\expenses.txt
```

```text
Cut expenses.txt to archive
```

```text
Delete the file old-expenses.txt
```

`cut` and `move` use the same move operation. Renaming a file without specifying a new extension preserves its current extension.

## Power And Session Commands

Sentry provides four guarded power routines:

```text
Good Night
Tata
Shutdown
Deep Sleep
Sleep
Lock
```

- `Good Night`, `GoodNight`, `Tata`, and exact shutdown commands ask for confirmation, gracefully close every visible user application except the Sentry terminal, then request a non-forced shutdown.
- `Deep Sleep` asks for confirmation, gracefully closes the same visible applications, then sleeps the laptop.
- Exact `Sleep` or `Normal Sleep` commands sleep immediately without closing applications.
- Exact `Lock` or `Hide` commands lock Windows immediately without closing applications.

Close-all uses normal window close messages and never force-terminates a process. If an application remains open, for example because it is showing an unsaved-work prompt, the dependent sleep or shutdown action is cancelled. Explorer folder windows and other terminal windows are included; the exact Sentry terminal and essential Windows shell windows are protected.

At startup, Sentry resolves the visible top-level terminal host rather than trusting the hidden pseudo-console handle used by Windows Terminal. Before close-all sends any window message, it verifies that this visible host is present in the protected set. If verification fails, nothing is closed and the dependent power action is blocked.

After Windows accepts shutdown, the agent exits with a dedicated launcher code. `START.bat` detects it and exits immediately without reaching its normal `pause`, so the Sentry console does not remain waiting for input.

Validated plans and their action-by-action statuses are audited in:

```text
cache\agent_state.db
```

Interrupted plans are marked as interrupted on the next launch and are never resumed automatically.

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
- File rename, copy, move, and delete are limited to the workspace and managed Notepad folder.
- Existing files cannot be overwritten unless overwrite or replacement is explicitly requested.
- The managed workspace root cannot be deleted.
- Symbolic links and directory junctions are rejected in managed paths.
- Network and device paths are blocked for directory inspection.
- Directory listing and summarization do not read file contents.
- Notepad filenames must be simple filenames, not paths.
- Notepad text length is capped.
- Clipboard reads are truncated to a configured limit.
- Destructive file and directory deletion requires confirmation.
- Close-all and shutdown routines require confirmation.
- Model-inferred sleep requests require confirmation; exact `Sleep` and `Normal Sleep` aliases do not.
- Shutdown is never forced, and it is cancelled if the close-all prerequisite fails.
- Lock, sleep, and shutdown tools are intentionally not exposed through the optional raw MCP server because that path has no trusted confirmation UI.

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
    |-- prompts.py
    |-- prompts
    |   |-- identity.md
    |   |-- tool_catalog.md
    |   |-- tool_execution_guide.md
    |   |-- system_tool_agent.md
    |   |-- system_structured_planner.md
    |   |-- system_natural_answer.md
    |   |-- tool_calling_probe_system.md
    |   |-- tool_calling_probe_user.md
    |   |-- native_tool_call_retry.md
    |   `-- structured_plan_retry.md
    |-- plan_store.py
    |-- llm.py
    |-- config.py
    |-- security.py
    |-- windows_tools.py
    |-- langchain_tools.py
    |-- mcp_server.py
    |-- requirements.txt
    `-- tests
        |-- __init__.py
        |-- test_tools.py
        |-- test_directory_tools.py
        |-- test_file_tools.py
        |-- test_langchain_agent.py
        |-- test_plan_store.py
        `-- test_windows_close.py
```

Main files:

- `START.bat` starts the local model server if needed and launches the chat UI.
- `STOP.bat` stops the local model server process.
- `agent/main.py` contains the terminal chat loop.
- `agent/agent.py` contains the planner, confirmation flow, memory handling, action reconciliation, and result summaries.
- `agent/prompts/` holds every prompt sent to the model as a separate Markdown file. `identity.md` sets the assistant's name (Sentry), maker (Portable AI), and developer (Malay Jain) for all system prompts. `{{name}}` marks a value filled in at runtime, such as `{{tool_execution_guide}}` or `{{validation_errors}}`. Edits take effect the next time the agent starts.
- `agent/prompts.py` loads those files and fills in their placeholders.
- `agent/plan_store.py` stores the persistent SQLite execution checklist and audit states.
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

The venv records its base interpreter (`runtime\python\cpython-3.12.*`) as an absolute path in `runtime\venv\pyvenv.cfg`, including the drive letter. Windows can assign the USB drive a different letter on each PC, so `START.bat` rewrites that `home =` line to the current drive on every launch. It also clears `PYTHONHOME`/`PYTHONPATH` so a Python installed on the host PC cannot interfere.

If required packages are missing, `START.bat` installs them into the venv with the bundled `tools\uv.exe` (internet required), keeping uv's cache in `cache\uv` on the drive.

## Testing

The repo includes tests for:

- Safe tool behavior
- App allowlist enforcement
- Directory path validation
- Directory create/list/summarize/delete behavior
- Managed file rename/copy/move/delete behavior
- Planner action repair and confirmation flow
- Compound command ordering
- Window close settling behavior

Test files live in `agent\tests\`:

```text
agent\tests\test_tools.py
agent\tests\test_directory_tools.py
agent\tests\test_file_tools.py
agent\tests\test_langchain_agent.py
agent\tests\test_plan_store.py
agent\tests\test_windows_close.py
```

`agent\tests\__init__.py` adds the `agent` folder to the import path, so run tests as modules from the `agent` folder:

```bat
cd agent
..\runtime\venv\Scripts\python.exe -m unittest
..\runtime\venv\Scripts\python.exe -m unittest tests.test_plan_store
```

`test_tools.py` is a manual check rather than a unittest suite. It touches the real clipboard and can open applications. Run it with `..\runtime\venv\Scripts\python.exe -m tests.test_tools` from the `agent` folder.

## Notes

PortableAI is Windows-specific. It depends on Windows APIs through `pywin32`, uses `.bat` launchers, and includes application discovery logic for common Windows install locations.

Some web-related actions, such as YouTube result lookup, may require internet access. The core chat model and Windows tool execution are designed to run locally.
