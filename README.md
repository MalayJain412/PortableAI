# PortableAI

PortableAI is a local Windows terminal assistant designed to run from this USB folder.

## Start

Double-click:

```text
START.bat
```

It checks the local llamafile server at `http://127.0.0.1:8080`, starts it if needed, waits for `/health`, and then launches the terminal chat UI.

## Stop

Exit the chat with:

```text
/exit
```

To stop the USB model server afterward, run:

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

## Tool Decision Flow

The agent first asks the local model to identify the user's intent and compare it with the available MCP tools. A complete match runs the tool, a slight match asks for confirmation, and no clear tool intent becomes a normal chat answer.

If a typo-like command is confirmed, the stored tool call is executed directly instead of asking the model to reinterpret `yes`.

## Voice Input

Voice input is disabled for now because the offline recognizers were not accurate enough for comfortable use. The voice files and dependencies are left in place so this can be revisited later, but text mode is the supported input mode.

## Things To Ask

```text
Open Notepad
```

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

```text
Close Notepad
```

```text
Close Chrome
```

```text
List all Chrome windows
```

```text
Open WhatsApp
```

```text
Close WhatsApp
```

```text
Close Chrome and WhatsApp
```

```text
Open Chrome and go to Google
```

```text
What applications are currently running?
```

```text
What window am I currently using?
```

```text
Copy "Hello from my portable AI" to the clipboard
```

```text
What are my basic system specs?
```

## Safety

The agent does not run arbitrary model-generated shell commands. The model can only request allowlisted MCP tools, and each tool validates its arguments before touching Windows APIs.
