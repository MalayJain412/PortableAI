from __future__ import annotations

import asyncio
import difflib
import json
import re
from urllib.parse import quote_plus
from dataclasses import dataclass
from typing import Any

from llm import GenerationMetrics, LLMResult, LlamafileClient, Message, TokenCallback
from mcp_server import create_server


SYSTEM_CHAT = """You are Portable AI, a local offline Windows assistant.
Be concise, useful, and honest. You can chat normally.
When tool results are provided, summarize only what the results support."""

SYSTEM_ROUTER = """You are the tool router for Portable AI.
Return ONLY one JSON object. No markdown. No explanation.

Use this schema:
{"type":"answer","content":"short answer"}
or:
{"type":"tools","actions":[{"tool":"tool_name","arguments":{}}]}
or:
{"type":"confirm","content":"Did you mean: close WhatsApp? Reply yes to run it, or no to cancel.","decision":{"type":"tools","actions":[{"tool":"tool_name","arguments":{}}]}}

Available tools:
- open_application: {"application":"notepad|calculator|calc|chrome|whatsapp"}
- close_application: {"application":"notepad|calculator|calc|chrome|whatsapp"}
- list_application_windows: {"application":"notepad|calculator|calc|chrome|whatsapp"}
- notepad_open: {"text":"optional starting text"}
- open_notepad_with_text: {"text":"text to write"}
- notepad_type: {"text":"text to append"}
- notepad_save: {}
- notepad_save_as: {"filename":"simple filename"}
- notepad_close: {}
- open_url: {"url":"http://... or https://..."}
- youtube_search_open: {"query":"YouTube search query"}
- list_processes: {}
- system_information: {}
- active_window: {}
- clipboard_read: {}
- clipboard_write: {"text":"text to copy"}

Rules:
- First identify the user's intent.
- If there is no clear intent to use a tool, answer naturally.
- If there is a clear tool intent, compare it with the available tools.
- If an available tool completely matches the intent, return tools.
- If an available tool slightly matches the intent, return confirm with the proposed tool call.
- Split compound requests into action buckets. Each requested app, URL, or operation should become its own action.
- For "close Chrome and WhatsApp", return two close_application actions: one for chrome and one for whatsapp.
- For "list Chrome and WhatsApp windows", return two list_application_windows actions.
- Preserve the user's requested order when creating actions.
- Use tools for requests to open apps, open URLs, YouTube searches, inspect processes, inspect system info, inspect active window, list app windows, or use clipboard.
- Use close_application for requests to close allowlisted applications or their visible windows.
- Use list_application_windows for requests to list, show, check, or inspect visible windows for an allowlisted application.
- For one-shot requests to open Notepad and write text, use open_notepad_with_text.
- For multi-step Notepad work, use notepad_open, notepad_type, notepad_save, and notepad_close.
- notepad_type appends text to the agent-managed Notepad note.
- notepad_save reports the saved file path because the managed note is already disk-backed.
- notepad_save_as saves the managed note under a simple filename.
- notepad_close only closes Notepad windows launched by this agent.
- If the user has a minor typo but the intent is clear, infer the intended available tool.
- Never invent tool names.
- Never request cmd, powershell, python, shell, keyboard, mouse, files, registry, services, or startup changes.
- For "Open Chrome and go to Google", request open_application then open_url.
- Use at most 5 actions."""


@dataclass(frozen=True)
class ChatTurnResult:
    content: str
    metrics: LLMResult | None


class PortableAgent:
    def __init__(self, llm: LlamafileClient | None = None) -> None:
        self.llm = llm or LlamafileClient()
        self.server = create_server()
        self.history: list[Message] = []
        self.last_youtube_query: str | None = None
        self.pending_confirmation: dict[str, Any] | None = None

    async def tool_names(self) -> list[str]:
        tools = await self.server.list_tools()
        return [tool.name for tool in tools]

    async def handle(self, user_text: str, on_token: TokenCallback | None = None) -> LLMResult:
        pending_result = await self._handle_pending_confirmation(user_text, on_token)
        if pending_result is not None:
            return pending_result

        obvious = obvious_safe_tool_intent(user_text, self.last_youtube_query)
        if obvious is not None and obvious.get("type") != "confirm":
            if obvious.get("type") == "blocked":
                return static_result(str(obvious["content"]))
            remember_youtube_query = get_youtube_query_from_decision(obvious)
            if remember_youtube_query:
                self.last_youtube_query = remember_youtube_query
            return await self._handle_tools(user_text, obvious, on_token)

        decision = self._decide(user_text)
        repair = notepad_repair_decision(user_text, decision)
        if repair is not None:
            decision = repair
        if decision.get("type") == "confirm":
            return self._set_confirmation(user_text, decision)

        if decision.get("type") == "tools":
            remember_youtube_query = get_youtube_query_from_decision(decision)
            if remember_youtube_query:
                self.last_youtube_query = remember_youtube_query
            return await self._handle_tools(user_text, decision, on_token)

        fallback = obvious
        if fallback is None:
            fallback = obvious_safe_tool_intent(user_text, self.last_youtube_query)
        if fallback is not None:
            if fallback.get("type") == "blocked":
                return static_result(str(fallback["content"]))
            if fallback.get("type") == "confirm":
                return self._set_confirmation(user_text, fallback)
            remember_youtube_query = get_youtube_query_from_decision(fallback)
            if remember_youtube_query:
                self.last_youtube_query = remember_youtube_query
            return await self._handle_tools(user_text, fallback, on_token)

        if decision.get("type") == "answer" and decision.get("content"):
            return self._chat(user_text, on_token)

        return self._chat(user_text, on_token)

    def _decide(self, user_text: str) -> dict[str, Any]:
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_ROUTER},
            {"role": "user", "content": user_text},
        ]
        for _ in range(2):
            result = self.llm.chat(messages, max_tokens=220, temperature=0.0, stream=False)
            parsed = parse_json_object(result.content)
            if isinstance(parsed, dict) and parsed.get("type") in {"answer", "tools", "confirm"}:
                return parsed
            messages.append({"role": "assistant", "content": result.content})
            messages.append({"role": "user", "content": "Return only valid JSON using the required schema."})
        return {"type": "answer", "content": ""}

    async def _handle_pending_confirmation(
        self,
        user_text: str,
        on_token: TokenCallback | None,
    ) -> LLMResult | None:
        pending = self.pending_confirmation
        if pending is None:
            return None

        answer = parse_confirmation_answer(user_text)
        if answer is True:
            self.pending_confirmation = None
            decision = pending.get("decision")
            original_text = str(pending.get("user_text") or user_text)
            if isinstance(decision, dict):
                return await self._handle_tools(original_text, decision, on_token)
            return static_result("I lost the pending action, so I did not run anything.")

        if answer is False:
            self.pending_confirmation = None
            return static_result("No problem. What would you like help with instead?")

        self.pending_confirmation = None
        return None

    def _set_confirmation(self, user_text: str, decision: dict[str, Any]) -> LLMResult:
        tool_decision = decision.get("decision")
        content = decision.get("content")
        if not isinstance(tool_decision, dict) or tool_decision.get("type") != "tools":
            return static_result("I need a clearer command before I use a tool.")
        if not isinstance(content, str) or not content.strip() or content.strip().lower() == "yes/no question":
            content = confirmation_prompt_for_decision(tool_decision)
        self.pending_confirmation = {"user_text": user_text, "decision": tool_decision}
        return static_result(content.strip())

    def _chat(self, user_text: str, on_token: TokenCallback | None) -> LLMResult:
        messages = [{"role": "system", "content": SYSTEM_CHAT}, *self.history[-8:], {"role": "user", "content": user_text}]
        result = self.llm.chat(messages, max_tokens=320, temperature=0.3, stream=True, on_token=on_token)
        self._remember(user_text, result.content)
        return result

    async def _handle_tools(
        self,
        user_text: str,
        decision: dict[str, Any],
        on_token: TokenCallback | None,
    ) -> LLMResult:
        actions = decision.get("actions")
        if not isinstance(actions, list):
            return self._chat(user_text, on_token)

        available = set(await self.tool_names())
        results: list[dict[str, Any]] = []
        for action in actions[:5]:
            if not isinstance(action, dict):
                continue
            tool_name = action.get("tool")
            arguments = action.get("arguments") or {}
            if tool_name not in available or not isinstance(arguments, dict):
                results.append({"tool": tool_name, "ok": False, "error": "Tool is not available or arguments are invalid."})
                continue

            try:
                call_result = await self.server.call_tool(tool_name, arguments)
                results.append(
                    {
                        "tool": tool_name,
                        "ok": not call_result.is_error,
                        "result": call_result.structured_content,
                    }
                )
            except Exception as exc:
                results.append({"tool": tool_name, "ok": False, "error": str(exc)})

        deterministic = deterministic_tool_reply(results)
        if deterministic is not None:
            self._remember(user_text, deterministic)
            return static_result(deterministic)

        final_messages: list[Message] = [
            {"role": "system", "content": SYSTEM_CHAT},
            {
                "role": "user",
                "content": (
                    "The user asked:\n"
                    f"{user_text}\n\n"
                    "Tool results as JSON:\n"
                    f"{json.dumps(results, ensure_ascii=False)}\n\n"
                    "If a tool result says ok=true, say that the action completed. "
                    "Do not claim you lack permission for an action that already completed. "
                    "Do not merely say data was retrieved; include the important details. "
                    "For system_information, include Windows, CPU cores, total RAM, and available RAM. "
                    "For list_processes, summarize the count and name a few visible processes. "
                    "For clipboard_read, include the clipboard text unless it is empty. "
                    "Reply to the user in one or two concise sentences."
                ),
            },
        ]
        result = self.llm.chat(final_messages, max_tokens=220, temperature=0.2, stream=True, on_token=on_token)
        self._remember(user_text, result.content)
        return result

    def _remember(self, user_text: str, assistant_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        self.history = self.history[-12:]


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def obvious_safe_tool_intent(user_text: str, last_youtube_query: str | None = None) -> dict[str, Any] | None:
    text = user_text.strip()
    lower = text.lower()
    actions: list[dict[str, Any]] = []

    notepad_command = parse_notepad_command(text)
    if notepad_command is not None:
        return notepad_command

    notepad_text = parse_notepad_write(text)
    if notepad_text is not None:
        return {"type": "tools", "actions": [{"tool": "open_notepad_with_text", "arguments": {"text": notepad_text}}]}

    fuzzy_notepad_write = parse_fuzzy_notepad_write_confirmation(text)
    if fuzzy_notepad_write is not None:
        return fuzzy_notepad_write

    clipboard_write = parse_clipboard_write(text)
    if clipboard_write is not None:
        return {"type": "tools", "actions": [{"tool": "clipboard_write", "arguments": {"text": clipboard_write}}]}

    if asks_capabilities(lower):
        return {
            "type": "blocked",
            "content": (
                "I can chat with the local Qwen model and use safe Windows tools: open Notepad, Calculator, "
                "Chrome, or WhatsApp; list and close allowlisted app windows; open, type into, save, and close an agent-managed Notepad note; open http/https URLs; list running processes; show basic system information; "
                "report the active window; and read or write clipboard text. I cannot run arbitrary shell, "
                "PowerShell, Python, mouse, keyboard, file, registry, or browser automation commands yet."
            ),
        }

    if "clipboard" in lower and any(word in lower for word in ("read", "show", "what", "paste")):
        return {"type": "tools", "actions": [{"tool": "clipboard_read", "arguments": {}}]}

    list_windows_apps = parse_list_application_windows(lower)
    if list_windows_apps:
        return {
            "type": "tools",
            "actions": [{"tool": "list_application_windows", "arguments": {"application": app}} for app in list_windows_apps],
        }

    close_apps = parse_close_application(lower)
    if close_apps:
        return {
            "type": "tools",
            "actions": [{"tool": "close_application", "arguments": {"application": app}} for app in close_apps],
        }

    youtube_query = parse_youtube_query(text, last_youtube_query)
    if youtube_query is not None:
        return {"type": "tools", "actions": [{"tool": "youtube_search_open", "arguments": {"query": youtube_query}}]}

    app = parse_open_application(lower)
    if app is not None:
        actions.append({"tool": "open_application", "arguments": {"application": app}})
    else:
        fuzzy_open = parse_fuzzy_open_application_confirmation(text)
        if fuzzy_open is not None:
            return fuzzy_open

    url = parse_open_url(text)
    if url is not None:
        actions.append({"tool": "open_url", "arguments": {"url": url}})

    if actions:
        return {"type": "tools", "actions": actions[:3]}

    if looks_like_unsupported_open_request(text):
        return {
            "type": "blocked",
            "content": (
                "I can only open allowlisted apps right now: notepad, calculator, calc, chrome, and whatsapp. "
                "That requested app is not in the allowlist yet."
            ),
        }

    if any(phrase in lower for phrase in ("running processes", "running applications", "applications are currently running")):
        return {"type": "tools", "actions": [{"tool": "list_processes", "arguments": {}}]}

    if any(phrase in lower for phrase in ("system specs", "system information", "basic system", "computer specs")):
        return {"type": "tools", "actions": [{"tool": "system_information", "arguments": {}}]}

    if any(phrase in lower for phrase in ("active window", "current window", "window am i")):
        return {"type": "tools", "actions": [{"tool": "active_window", "arguments": {}}]}

    return None


def notepad_repair_decision(user_text: str, decision: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(decision, dict) or decision.get("type") != "tools":
        return None

    fallback = parse_notepad_command(user_text)
    if fallback is None or fallback.get("type") != "tools":
        return None

    fallback_tools = tool_names_from_decision(fallback)
    if not fallback_tools or len(fallback_tools) <= 1:
        return None

    decision_tools = tool_names_from_decision(decision)
    if not decision_tools:
        return fallback

    notepad_tools = {"notepad_open", "open_notepad_with_text", "notepad_type", "notepad_save", "notepad_save_as", "notepad_close"}
    if any(tool in notepad_tools for tool in decision_tools) and not set(fallback_tools).issubset(set(decision_tools)):
        return fallback
    return None


def tool_names_from_decision(decision: dict[str, Any]) -> list[str]:
    actions = decision.get("actions")
    if not isinstance(actions, list):
        return []
    return [str(action.get("tool")) for action in actions if isinstance(action, dict) and action.get("tool")]


OPEN_APPLICATION_ALIASES: dict[str, tuple[str, ...]] = {
    "notepad": ("notepad", "note pad"),
    "calculator": ("calculator", "calc"),
    "chrome": ("chrome", "google chrome"),
    "whatsapp": ("whatsapp", "whats app"),
}


def parse_confirmation_answer(text: str) -> bool | None:
    normalized = " ".join(text.strip().lower().split())
    if normalized in {"y", "yes", "yeah", "yep", "ok", "okay", "correct", "confirm", "do it", "go ahead", "run it"}:
        return True
    if normalized in {"n", "no", "nope", "cancel", "stop", "wrong", "do not", "don't", "dont"}:
        return False
    return None


def confirmation_prompt_for_decision(decision: dict[str, Any]) -> str:
    actions = decision.get("actions")
    if isinstance(actions, list) and len(actions) == 1 and isinstance(actions[0], dict):
        action = actions[0]
        tool = action.get("tool")
        arguments = action.get("arguments") or {}
        if isinstance(arguments, dict):
            app = arguments.get("application")
            if isinstance(app, str) and app:
                label = format_application_name(app)
                if tool == "close_application":
                    return f"Did you mean: close {label}? Reply yes to run it, or no to cancel."
                if tool == "open_application":
                    return f"Did you mean: open {label}? Reply yes to run it, or no to cancel."
                if tool == "list_application_windows":
                    return f"Did you mean: list visible {label} windows? Reply yes to run it, or no to cancel."
    return "Did you mean this action? Reply yes to run it, or no to cancel."


def deterministic_tool_reply(results: list[dict[str, Any]]) -> str | None:
    if not results:
        return None

    if all(item.get("tool") in {"open_application", "open_url"} for item in results):
        failed = [item for item in results if not item.get("ok")]
        if failed:
            return "I could not complete that open request with the available tools."

        apps = [
            format_application_name(str((item.get("result") or {}).get("application")))
            for item in results
            if item.get("tool") == "open_application" and (item.get("result") or {}).get("application")
        ]
        urls = [
            str((item.get("result") or {}).get("url"))
            for item in results
            if item.get("tool") == "open_url" and (item.get("result") or {}).get("url")
        ]
        if apps and urls:
            return f"I opened {', '.join(apps)} and navigated to {', '.join(urls)}."
        if apps:
            return f"I opened {', '.join(apps)}."
        if urls:
            return f"I opened {', '.join(urls)}."

    if all(item.get("tool") == "close_application" for item in results):
        return deterministic_multi_close_application_reply(results)

    if all(item.get("tool") == "list_application_windows" for item in results):
        return "\n\n".join(deterministic_list_application_windows_reply(item) for item in results)

    if len(results) == 1 and results[0].get("tool") == "close_application":
        return deterministic_close_application_reply(results[0])

    if len(results) == 1 and results[0].get("tool") == "list_application_windows":
        return deterministic_list_application_windows_reply(results[0])

    notepad_tools = {"notepad_open", "open_notepad_with_text", "notepad_type", "notepad_save", "notepad_save_as", "notepad_close"}
    if all(item.get("tool") in notepad_tools for item in results):
        return deterministic_notepad_reply(results)

    if len(results) != 1:
        return None
    item = results[0]
    tool = item.get("tool")

    if tool == "notepad_open":
        if not item.get("ok"):
            return "I could not open the managed Notepad note."
        result = item.get("result") or {}
        path = result.get("path")
        if path:
            return f"I opened the managed Notepad note. File: {path}"
        return "I opened the managed Notepad note."

    if tool == "open_notepad_with_text":
        if not item.get("ok"):
            return "I could not open Notepad with the requested text."
        result = item.get("result") or {}
        path = result.get("path")
        characters = result.get("characters")
        if path and isinstance(characters, int):
            return f"I opened Notepad with the requested text ({characters} characters). File: {path}"
        return "I opened Notepad with the requested text."

    if tool == "notepad_type":
        if not item.get("ok"):
            return "I could not add text to the managed Notepad note."
        result = item.get("result") or {}
        path = result.get("path")
        characters = result.get("characters_added")
        if path and isinstance(characters, int):
            return f"I added the text to the managed Notepad note ({characters} characters). File: {path}"
        return "I added the text to the managed Notepad note."

    if tool == "notepad_save":
        if not item.get("ok"):
            return "I could not save the managed Notepad note."
        result = item.get("result") or {}
        path = result.get("path")
        characters = result.get("characters")
        if path and isinstance(characters, int):
            return f"The managed Notepad note is saved ({characters} characters). File: {path}"
        return "The managed Notepad note is saved."

    if tool == "notepad_close":
        if not item.get("ok"):
            result = item.get("result") or {}
            remaining = result.get("remaining")
            if remaining:
                return f"I asked the agent-launched Notepad windows to close, but some are still open: {remaining}."
            return "I could not close the agent-launched Notepad windows."
        result = item.get("result") or {}
        closed = result.get("closed")
        if isinstance(closed, int):
            return f"I closed {closed} agent-launched Notepad window{'s' if closed != 1 else ''}."
        return "I closed the agent-launched Notepad windows."

    if tool == "clipboard_write":
        if not item.get("ok"):
            return "I could not update the clipboard."
        result = item.get("result") or {}
        characters = result.get("characters")
        if isinstance(characters, int):
            return f"I copied the text to the clipboard ({characters} characters)."
        return "I copied the text to the clipboard."

    if tool != "youtube_search_open":
        return None

    result = item.get("result") or {}
    if not item.get("ok"):
        return "I could not fetch a direct YouTube result, and the browser did not confirm opening fallback search results."

    mode = result.get("mode")
    title = result.get("title")
    url = result.get("url")
    query = result.get("query")
    if mode in {"best_result", "known_result"} and url:
        if title:
            return f"I opened the best YouTube result I found: {title} ({url})."
        return f"I opened the best YouTube result I found: {url}."
    if mode == "search_results" and url:
        return f"I opened YouTube search results for {query!r}: {url}. I could not confirm a single best result."
    return None


def format_application_name(application: str) -> str:
    return {"notepad": "Notepad", "calculator": "Calculator", "calc": "Calculator", "chrome": "Chrome", "whatsapp": "WhatsApp"}.get(
        application.lower(),
        application,
    )


def deterministic_notepad_reply(results: list[dict[str, Any]]) -> str:
    failed = [item for item in results if not item.get("ok")]
    if failed:
        tool = failed[0].get("tool")
        if tool == "notepad_type":
            return "I could not add text because the managed Notepad window did not close cleanly."
        if tool == "notepad_close":
            result = failed[0].get("result") or {}
            remaining = result.get("remaining")
            return f"I updated the note, but some Notepad windows are still open: {remaining}."
        return "I could not complete the full Notepad request."

    opened_item = next((item for item in results if item.get("tool") in {"notepad_open", "open_notepad_with_text"}), None)
    typed = next((item for item in results if item.get("tool") == "notepad_type"), None)
    saved = next((item for item in results if item.get("tool") in {"notepad_save", "notepad_save_as"}), None)
    closed = next((item for item in results if item.get("tool") == "notepad_close"), None)

    pieces: list[str] = []
    if opened_item:
        result = opened_item.get("result") or {}
        characters = result.get("characters")
        if isinstance(characters, int) and characters > 0:
            pieces.append("opened Notepad with the requested text")
        else:
            pieces.append("opened Notepad")
    if typed:
        result = typed.get("result") or {}
        characters = result.get("characters_added")
        if isinstance(characters, int):
            pieces.append(f"added {characters} characters")
        else:
            pieces.append("added the text")
    if saved:
        result = saved.get("result") or {}
        path = result.get("path")
        if path:
            pieces.append(f"saved it as {path}")
        else:
            pieces.append("saved it")
    if closed:
        result = closed.get("result") or {}
        count = result.get("closed")
        targeted = result.get("targeted")
        if isinstance(count, int):
            if isinstance(targeted, int) and targeted == 0:
                pieces.append("found no visible Notepad windows to close")
            else:
                pieces.append(f"closed {count} Notepad window{'s' if count != 1 else ''}")
        else:
            pieces.append("closed Notepad")

    if pieces:
        return "I " + ", ".join(pieces) + "."
    return "I completed the Notepad request."


def deterministic_close_application_reply(item: dict[str, Any]) -> str:
    error = item.get("error")
    if not item.get("ok") and error:
        return f"I could not close the application: {error}"

    result = item.get("result") or {}
    application = format_application_name(str(result.get("application") or "application"))
    closed = result.get("closed")
    targeted = result.get("targeted")
    remaining = result.get("remaining")

    if item.get("ok"):
        if isinstance(closed, int):
            if isinstance(targeted, int) and targeted == 0:
                return f"No visible {application} windows were found."
            return f"I closed {closed} visible {application} window{'s' if closed != 1 else ''}."
        return f"I closed the visible {application} windows."

    if remaining:
        return f"I tried to close {application}, but some windows are still open: {remaining}."
    return f"I could not close {application}."


def deterministic_multi_close_application_reply(results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in results:
        sentence = deterministic_close_application_reply(item)
        if sentence.endswith("."):
            sentence = sentence[:-1]
        parts.append(sentence)
    return "; ".join(parts) + "."


def deterministic_list_application_windows_reply(item: dict[str, Any]) -> str:
    result = item.get("result") or {}
    application = format_application_name(str(result.get("application") or "application"))
    windows = result.get("windows") or []
    if not item.get("ok"):
        return f"I could not check visible {application} windows."
    if not windows:
        return f"No visible {application} windows are open."

    lines = [f"Visible {application} windows ({len(windows)}):"]
    for index, window in enumerate(windows[:12], start=1):
        title = str(window.get("title") or "(untitled)")
        pid = window.get("pid")
        lines.append(f"{index}. {title} (pid {pid})")
    if len(windows) > 12:
        lines.append(f"...and {len(windows) - 12} more.")
    return "\n".join(lines)


def static_result(content: str) -> LLMResult:
    return LLMResult(
        content=content,
        metrics=GenerationMetrics(
            ttft_seconds=0.0,
            total_seconds=0.0,
            output_characters=len(content),
            approx_tokens=0,
        ),
    )


def parse_clipboard_write(text: str) -> str | None:
    lower = text.lower()
    if "clipboard" not in lower or not any(word in lower for word in ("copy", "write", "put", "set")):
        return None

    quoted = re.search(r'"([^"]+)"', text)
    if quoted:
        return quoted.group(1)

    match = re.search(r"(?:copy|write|put|set)\s+(.+?)\s+(?:to|on|into)\s+(?:the\s+)?clipboard", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def parse_notepad_command(text: str) -> dict[str, Any] | None:
    lower = text.lower()
    if not re.search(r"\bnote\s*pad\b|\bnotepad\b", lower):
        return None

    actions: list[dict[str, Any]] = []
    text_to_type = extract_text_to_write(text)
    save_as_filename = extract_save_as_filename(text)
    wants_open = any(word in lower for word in ("open", "launch", "start"))
    wants_type = text_to_type is not None and any(word in lower for word in ("write", "type", "put", "add", "enter", "paste"))
    wants_save = "save" in lower
    wants_close = any(word in lower for word in ("close", "quit", "exit"))

    if wants_open and wants_type:
        actions.append({"tool": "notepad_open", "arguments": {"text": text_to_type}})
    elif wants_open:
        quoted = extract_quoted_text(text)
        arguments = {"text": quoted} if quoted is not None else {}
        actions.append({"tool": "notepad_open", "arguments": arguments})
    elif wants_type:
        actions.append({"tool": "notepad_type", "arguments": {"text": text_to_type}})

    if save_as_filename is not None:
        actions.append({"tool": "notepad_save_as", "arguments": {"filename": save_as_filename}})
    elif wants_save:
        actions.append({"tool": "notepad_save", "arguments": {}})

    if wants_close:
        actions.append({"tool": "notepad_close", "arguments": {}})

    if actions:
        return {"type": "tools", "actions": actions[:5]}

    if any(word in lower for word in ("close", "quit", "exit")) and not any(
        word in lower for word in ("write", "type", "put", "add", "enter", "paste")
    ):
        return {"type": "tools", "actions": [{"tool": "notepad_close", "arguments": {}}]}

    if "save" in lower and not any(word in lower for word in ("write", "type", "put", "add", "enter", "paste")):
        return {"type": "tools", "actions": [{"tool": "notepad_save", "arguments": {}}]}

    if any(word in lower for word in ("type", "append", "add")):
        text_to_type = extract_text_to_write(text)
        if text_to_type is not None:
            return {"type": "tools", "actions": [{"tool": "notepad_type", "arguments": {"text": text_to_type}}]}

    if any(word in lower for word in ("open", "launch", "start")) and not any(
        word in lower for word in ("write", "type", "put", "add", "enter", "paste")
    ):
        quoted = extract_quoted_text(text)
        arguments = {"text": quoted} if quoted is not None else {}
        return {"type": "tools", "actions": [{"tool": "notepad_open", "arguments": arguments}]}

    return None


def parse_notepad_write(text: str) -> str | None:
    lower = text.lower()
    if not re.search(r"\bnote\s*pad\b|\bnotepad\b", lower):
        return None
    if not any(word in lower for word in ("write", "type", "put", "add", "enter", "paste")):
        return None

    quoted = extract_quoted_text(text)
    if quoted is not None:
        return quoted

    patterns = (
        r"\b(?:write|type|put|add|enter|paste)\s+(?:in\s+it\s+|into\s+it\s+|to\s+it\s+)?(.+)$",
        r"\b(?:write|type|put|add|enter|paste)\s+(?:this\s+)?(.+?)\s+(?:in|into|to)\s+(?:note\s*pad|notepad)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        candidate = match.group(1)
        candidate = re.sub(r"\s+(?:in|into|to)\s+(?:note\s*pad|notepad)\b.*$", "", candidate, flags=re.IGNORECASE | re.DOTALL)
        candidate = candidate.strip(" \t\r\n\"'.")
        if candidate:
            return candidate
    return None


def parse_fuzzy_notepad_write_confirmation(text: str) -> dict[str, Any] | None:
    lower = text.lower()
    if not any(word in lower for word in ("write", "type", "put", "add", "enter", "paste")):
        return None

    requested_app = extract_open_target(text)
    if requested_app is None:
        return None

    app, score = fuzzy_application_match(requested_app)
    if app != "notepad" or score < 0.55:
        return None

    text_to_write = extract_text_to_write(text)
    if text_to_write is None:
        return None

    return confirm_decision(
        f'Did you mean: open Notepad and write "{text_to_write}"? Reply yes to run it, or no to cancel.',
        [{"tool": "open_notepad_with_text", "arguments": {"text": text_to_write}}],
    )


def parse_fuzzy_open_application_confirmation(text: str) -> dict[str, Any] | None:
    requested_app = extract_open_target(text)
    if requested_app is None:
        return None

    app, score = fuzzy_application_match(requested_app)
    if app is None or score < 0.55:
        return None

    actions = [{"tool": "open_application", "arguments": {"application": app}}]
    url = parse_open_url(text)
    if url is not None:
        actions.append({"tool": "open_url", "arguments": {"url": url}})

    label = {"notepad": "Notepad", "calculator": "Calculator", "chrome": "Chrome"}[app]
    if url is not None:
        content = f"Did you mean: open {label} and go to {url}? Reply yes to run it, or no to cancel."
    else:
        content = f"Did you mean: open {label}? Reply yes to run it, or no to cancel."
    return confirm_decision(content, actions)


def confirm_decision(content: str, actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "confirm", "content": content, "decision": {"type": "tools", "actions": actions[:3]}}


def extract_text_to_write(text: str) -> str | None:
    quoted = extract_quoted_text(text)
    if quoted is not None:
        return quoted

    match = re.search(r"\b(?:write|type|put|add|enter|paste)\s+(?:in\s+it\s+|into\s+it\s+|to\s+it\s+)?(.+)$", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    candidate = match.group(1)
    candidate = re.sub(r"\s+(?:in|into|to)\s+\S+.*$", "", candidate, flags=re.IGNORECASE | re.DOTALL)
    candidate = candidate.strip(" \t\r\n\"'.")
    return candidate or None


def extract_save_as_filename(text: str) -> str | None:
    patterns = (
        r"\bsave\s+(?:the\s+)?(?:file\s+)?as\s+\"([^\"]+)\"",
        r"\bsave\s+(?:the\s+)?(?:file\s+)?as\s+'([^']+)'",
        r"\bsave\s+(?:the\s+)?(?:file\s+)?as\s+([A-Za-z0-9 _.-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        filename = match.group(1).strip(" \t\r\n\"'.")
        filename = re.split(r"\s+\b(?:and|then)\b", filename, maxsplit=1, flags=re.IGNORECASE)[0].strip(" \t\r\n\"'.")
        if filename:
            return filename
    return None


def extract_quoted_text(text: str) -> str | None:
    quoted = re.search(r'"([^"]*)"', text, re.DOTALL)
    if quoted:
        return quoted.group(1)

    single_quoted = re.search(r"'([^']*)'", text, re.DOTALL)
    if single_quoted:
        return single_quoted.group(1)
    return None


def extract_open_target(text: str) -> str | None:
    match = re.search(r"\b(?:open|launch|start)\s+(.+)$", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    requested = match.group(1)
    requested = re.split(r"\s+\b(?:and|then|with|to|on|please)\b", requested, maxsplit=1, flags=re.IGNORECASE)[0]
    requested = requested.strip(" \t\r\n\"'.,")
    return requested or None


def fuzzy_application_match(value: str) -> tuple[str | None, float]:
    normalized = normalize_match_text(value)
    if not normalized:
        return None, 0.0

    best_app: str | None = None
    best_score = 0.0
    for app, aliases in OPEN_APPLICATION_ALIASES.items():
        for alias in aliases:
            score = difflib.SequenceMatcher(None, normalized, normalize_match_text(alias)).ratio()
            if score > best_score:
                best_app = app
                best_score = score
    return best_app, best_score


def normalize_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def asks_capabilities(lower: str) -> bool:
    phrases = (
        "what can you do",
        "what are your capabilities",
        "what tools do you have",
        "what can i ask",
        "help me use this",
    )
    return any(phrase in lower for phrase in phrases)


def parse_youtube_query(text: str, last_youtube_query: str | None) -> str | None:
    lower = text.lower()
    if "youtube" not in lower and "yt" not in lower:
        return None
    if re.search(r"\b(?:search\s+for|open|go to|visit)\s+(?:https?://)?(?:www\.)?youtube\.com\b", lower):
        return None
    if any(word in lower for word in ("again", "same", "it")) and last_youtube_query:
        return last_youtube_query
    if not any(word in lower for word in ("open", "search", "find", "play", "playlist", "video", "series")):
        return None

    channel_query = parse_youtube_channel_query(text)
    if channel_query is not None:
        return f"{channel_query} channel"

    query = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    replacements = (
        r"\bon youtube\b",
        r"\byoutube\b",
        r"\byt\b",
        r"\bplease\b",
        r"\bnow\b",
        r"\bok\b",
        r"\bopen\b",
        r"\bsearch\s+for\b",
        r"\bsearch\b",
        r"\bfind\b",
        r"\bplay\b",
        r"\bthe\b",
    )
    for pattern in replacements:
        query = re.sub(pattern, " ", query, flags=re.IGNORECASE)
    query = " ".join(query.strip(" .,!?:;\"'").split())
    if query.lower() in {"com", ".com", "youtube.com", "www.youtube.com"}:
        return None
    if not query and last_youtube_query:
        return last_youtube_query
    return query or None


def parse_youtube_channel_query(text: str) -> str | None:
    lower = text.lower()
    if "channel" not in lower:
        return None
    patterns = (
        r"(?:youtube\s+)?channel\s+(?:of|for|called|named)?\s+(.+)$",
        r"(.+?)\s+(?:youtube\s+)?channel\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1)
        candidate = re.sub(r"^.*?\b(?:open|find|search for|show me|want you to open)\b", " ", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\b(?:the|youtube|channel|of|for|called|named|please|again|video|opened|you|i|want|to|open)\b", " ", candidate, flags=re.IGNORECASE)
        candidate = " ".join(candidate.strip(" .,!?:;\"'").split())
        if candidate:
            return candidate
    return None


def get_youtube_query_from_decision(decision: dict[str, Any]) -> str | None:
    actions = decision.get("actions")
    if not isinstance(actions, list):
        return None
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("tool") == "youtube_search_open":
            arguments = action.get("arguments") or {}
            query = arguments.get("query")
            if isinstance(query, str) and query.strip():
                return query.strip()
    return None


def parse_open_application(lower: str) -> str | None:
    if "open" not in lower and "launch" not in lower and "start" not in lower:
        return None
    if re.search(r"\bnote\s+pad\b|\bnotepad\b", lower):
        return "notepad"
    if re.search(r"\bwhats\s*app\b|\bwhatsapp\b", lower):
        return "whatsapp"
    for name in ("notepad", "calculator", "calc", "chrome", "whatsapp"):
        if re.search(rf"\b{name}\b", lower):
            return name
    return None


def parse_list_application_windows(lower: str) -> list[str]:
    if not any(word in lower for word in ("list", "show", "check", "inspect", "see")):
        return []
    if not any(word in lower for word in ("window", "windows", "open")):
        return []
    return parse_application_names_from_text(lower)


def parse_close_application(lower: str) -> list[str]:
    if not any(word in lower for word in ("close", "quit", "exit")):
        return []
    return parse_application_names_from_text(lower)


def parse_application_name_from_text(lower: str) -> str | None:
    names = parse_application_names_from_text(lower)
    return names[0] if names else None


def parse_application_names_from_text(lower: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    if re.search(r"\bnote\s+pad\b|\bnotepad\b", lower):
        match = re.search(r"\bnote\s+pad\b|\bnotepad\b", lower)
        if match:
            matches.append((match.start(), "notepad"))
    whatsapp_match = re.search(r"\bwhats\s*app\b|\bwhatsapp\b", lower)
    if whatsapp_match:
        matches.append((whatsapp_match.start(), "whatsapp"))
    for name, app in (("calculator", "calculator"), ("calc", "calc"), ("chrome", "chrome")):
        match = re.search(rf"\b{name}\b", lower)
        if match:
            matches.append((match.start(), app))

    ordered: list[str] = []
    for _, app in sorted(matches, key=lambda item: item[0]):
        if app not in ordered:
            ordered.append(app)
    return ordered


def looks_like_unsupported_open_request(text: str) -> bool:
    lower = text.lower().strip()
    if re.search(r"https?://", text, re.IGNORECASE):
        return False
    if "google" in lower:
        return False
    match = re.match(r"^(open|launch|start)\s+(.+)$", lower)
    if not match:
        return False
    requested = match.group(2).strip()
    if not requested:
        return False
    if any(word in requested for word in ("process", "system", "clipboard", "window")):
        return False
    return True


def parse_open_url(text: str) -> str | None:
    match = re.search(r"https?://[^\s]+", text, re.IGNORECASE)
    if match:
        return match.group(0).rstrip(".,)")

    domain = find_bare_domain(text)
    if domain is not None:
        return "https://" + domain.removeprefix("www.")

    lower = text.lower()
    if "google" in lower and any(phrase in lower for phrase in ("go to", "open", "browse", "visit")):
        return "https://www.google.com"

    target = parse_navigation_target(text)
    if target:
        if looks_like_domain(target):
            return "https://" + target.removeprefix("www.")
        if "search" in lower:
            return "https://www.google.com/search?q=" + quote_plus(target)
    return None


def find_bare_domain(text: str) -> str | None:
    match = re.search(r"\b(?:www\.)?[A-Za-z0-9-]+\.(?:com|org|net|in|io|dev|ai|co|edu|gov)\b/?", text, re.IGNORECASE)
    if not match:
        return None
    return match.group(0).strip().rstrip(".,)")


def parse_navigation_target(text: str) -> str | None:
    patterns = (
        r"\b(?:go to|visit|open)\s+([A-Za-z0-9][^\s,]*)",
        r"\bsearch\s+(?:for\s+)?(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            target = match.group(1).strip().strip("\"'.,)")
            if target.lower() in {"chrome", "notepad", "calculator", "calc"}:
                continue
            return target
    return None


def looks_like_domain(value: str) -> bool:
    return bool(re.fullmatch(r"(?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+/?", value.strip()))
