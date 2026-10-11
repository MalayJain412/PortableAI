from __future__ import annotations

import json
import time
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
- A tool handles one application, URL, or operation. Call it once for every requested target.
- Preserve the user's requested order. The host executes your calls sequentially.
- "open WhatsApp and Chrome" requires open_application for whatsapp and then chrome.
- "open Chrome and Notepad" requires open_application for chrome and then notepad.
- "close Chrome and Notepad" requires close_application for chrome and then notepad.
- "open YouTube on Chrome" requires open_application for chrome and open_url for https://youtube.com.
- "open youtube.com on chrome" has the same two actions.
- Opening YouTube home is URL navigation, not a YouTube content search.
- Use youtube_search_open only for a specific video, playlist, channel, or content search.
- For a one-shot request to open Notepad with text, use open_notepad_with_text.
- For Notepad requests that also save or close, use notepad_open with the text, then notepad_save_as or notepad_save, then notepad_close.
- Treat "note pad" as the clear spelling variant "notepad".
- Only call tools that are provided. Never invent a tool or an argument.
- If an application is unsupported, say which applications are allowed. Do not map it to an unrelated application.
- If the meaning is genuinely ambiguous, ask one short confirmation question and do not call any tool yet.
- For ordinary conversation with no tool intent, answer naturally and concisely.
- Never claim an action succeeded before receiving its tool result.
"""

SYSTEM_STRUCTURED_PLANNER = """You are the action planner for Portable AI, a local offline Windows assistant.
Return only data matching the supplied JSON schema.

Build one ordered action for every task in the user's message. Never silently drop a coordinated item.
Available tool names are:
open_application, close_application, list_application_windows, notepad_open,
open_notepad_with_text, notepad_type, notepad_save, notepad_save_as,
notepad_close, open_url, youtube_search_open, list_processes,
system_information, active_window, clipboard_read, clipboard_write.

Canonical applications are: notepad, calculator, calc, chrome, whatsapp.

Required examples:
- open WhatsApp and Chrome -> two open_application actions in that order.
- open Chrome and Notepad -> two open_application actions in that order.
- close Chrome and Notepad -> two close_application actions in that order.
- open YouTube on Chrome -> open_application(chrome), then open_url(https://youtube.com).
- open youtube.com on chrome -> open_application(chrome), then open_url(https://youtube.com).
- find the Makemore playlist on YouTube -> youtube_search_open with a specific search query.

Opening YouTube home is navigation and must not use youtube_search_open. A specific video, playlist, or channel request must use youtube_search_open.
For a one-shot Notepad text request use open_notepad_with_text. If save or close is also requested, use notepad_open with the text followed by the requested save and close actions.
Treat "note pad" as the clear spelling variant "notepad". For a genuinely uncertain match, return type "confirm", include the proposed actions, and ask a short question. For unsupported applications, return type "answer" and explain the allowlist. For normal conversation, return type "answer" with a concise natural response.
Never invent tools or arguments. Use no more than eight actions.
"""


class PlannedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(description="Exact registered LangChain tool name.")
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["answer", "tools", "confirm"]
    content: str = ""
    actions: list[PlannedAction] = Field(default_factory=list, max_length=MAX_ACTIONS)

    @model_validator(mode="after")
    def validate_plan_shape(self) -> "ToolPlan":
        if self.type in {"tools", "confirm"} and not self.actions:
            raise ValueError(f"A {self.type} plan must contain at least one action.")
        if self.type == "answer" and self.actions:
            raise ValueError("An answer plan cannot contain tool actions.")
        return self


class ProbeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Literal["ready"]


@tool(args_schema=ProbeInput)
def portable_ai_tool_probe(value: Literal["ready"]) -> str:
    """Capability probe. Call this tool with the exact value ready when explicitly requested."""
    return value


@dataclass(frozen=True)
class PendingConfirmation:
    user_text: str
    actions: list[PlannedAction]


class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class PortableAgent:
    def __init__(
        self,
        llm: LlamafileClient | None = None,
        *,
        chat_model: Any | None = None,
        tools: list[BaseTool] | None = None,
        tool_calling_supported: bool | None = None,
    ) -> None:
        self.llm = llm or LlamafileClient()
        self.chat_model = chat_model or ChatOpenAI(
            model=self.llm.model,
            base_url=f"{self.llm.base_url}/v1",
            api_key="portable-ai-local",
            temperature=0,
            max_tokens=480,
            timeout=120,
            max_retries=1,
        )
        self.tools = list(tools or LANGCHAIN_TOOLS)
        self.tool_registry = {item.name: item for item in self.tools}
        self._memory = InMemorySaver()
        self._memory_graph = build_memory_graph(self._memory)
        self._memory_config = {"configurable": {"thread_id": "portable-ai-session"}}
        self.pending_confirmation: PendingConfirmation | None = None
        self._tool_calling_supported = tool_calling_supported
        self._bound_model: Any | None = None
        self._planner_parser = PydanticOutputParser(pydantic_object=ToolPlan)

    async def initialize(self) -> bool:
        """Probe the local model once and select native tool calling or structured planning."""
        return await self._supports_native_tool_calls()

    async def tool_names(self) -> list[str]:
        return list(self.tool_registry)

    def clear_history(self) -> None:
        self._memory.delete_thread("portable-ai-session")
        self.pending_confirmation = None

    async def handle(self, user_text: str, on_token: TokenCallback | None = None) -> LLMResult:
        started_at = time.perf_counter()
        pending_result = await self._handle_pending_confirmation(user_text, started_at)
        if pending_result is not None:
            return pending_result

        if await self._supports_native_tool_calls():
            native_result = await self._handle_native(user_text, on_token, started_at)
            if native_result is not None:
                return native_result

        return await self._handle_structured(user_text, on_token, started_at)

    async def _supports_native_tool_calls(self) -> bool:
        if self._tool_calling_supported is not None:
            return self._tool_calling_supported

        try:
            probe_model = self.chat_model.bind_tools([portable_ai_tool_probe])
            response = await probe_model.ainvoke(
                [
                    SystemMessage(content="You are testing OpenAI-compatible tool calling."),
                    HumanMessage(content="Call portable_ai_tool_probe now with value ready. Do not answer in text."),
                ]
            )
            calls = getattr(response, "tool_calls", None) or []
            self._tool_calling_supported = any(
                call.get("name") == portable_ai_tool_probe.name and call.get("args", {}).get("value") == "ready"
                for call in calls
                if isinstance(call, dict)
            )
        except Exception:
            self._tool_calling_supported = False

        if self._tool_calling_supported:
            self._bound_model = self.chat_model.bind_tools(self.tools)
        return bool(self._tool_calling_supported)

    async def _handle_native(
        self,
        user_text: str,
        on_token: TokenCallback | None,
        started_at: float,
    ) -> LLMResult | None:
        model = self._bound_model or self.chat_model.bind_tools(self.tools)
        messages = [SystemMessage(content=SYSTEM_TOOL_AGENT), *self._recent_messages(), HumanMessage(content=user_text)]
        validation_feedback = ""

        for _ in range(2):
            attempt_messages = list(messages)
            if validation_feedback:
                attempt_messages.append(
                    HumanMessage(
                        content=(
                            "Your proposed tool calls were invalid. Correct every issue and return a complete replacement "
                            f"set of tool calls. Validation errors: {validation_feedback}"
                        )
                    )
                )
            response = await model.ainvoke(attempt_messages)
            calls = getattr(response, "tool_calls", None) or []
            if calls:
                actions = [
                    PlannedAction(tool=str(call.get("name", "")), arguments=call.get("args") or {})
                    for call in calls
                    if isinstance(call, dict)
                ]
                prepared, validation_feedback = self._validate_actions(actions)
                if prepared is None:
                    continue
                return await self._execute_and_report(user_text, prepared, started_at)

            content = message_text(response)
            if content:
                self._remember(user_text, content)
                if on_token is not None:
                    on_token(content)
                return measured_result(content, started_at)
            validation_feedback = "The response contained neither text nor valid tool calls."

        return None

    async def _handle_structured(
        self,
        user_text: str,
        on_token: TokenCallback | None,
        started_at: float,
    ) -> LLMResult:
        validation_feedback = ""
        for _ in range(2):
            plan = await self._create_structured_plan(user_text, validation_feedback)
            if plan is None:
                validation_feedback = "Return exactly one valid JSON object matching the supplied schema."
                continue

            if plan.type == "answer":
                content = plan.content.strip()
                if not content:
                    break
                self._remember(user_text, content)
                if on_token is not None:
                    on_token(content)
                return measured_result(content, started_at)

            prepared, validation_feedback = self._validate_actions(plan.actions)
            if prepared is None:
                continue

            if plan.type == "confirm":
                self.pending_confirmation = PendingConfirmation(user_text=user_text, actions=prepared)
                content = plan.content.strip() or confirmation_prompt(prepared)
                return measured_result(content, started_at)

            return await self._execute_and_report(user_text, prepared, started_at)

        return await self._stream_natural_answer(user_text, on_token, started_at)

    async def _create_structured_plan(self, user_text: str, feedback: str) -> ToolPlan | None:
        history_text = render_history(self._recent_messages())
        messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    f"{SYSTEM_STRUCTURED_PLANNER}\n\n"
                    f"JSON schema instructions:\n{self._planner_parser.get_format_instructions()}"
                )
            ),
        ]
        if history_text:
            messages.append(SystemMessage(content=f"Recent conversation:\n{history_text}"))
        if feedback:
            messages.append(SystemMessage(content=f"Correct these validation errors: {feedback}"))
        messages.append(HumanMessage(content=user_text))

        try:
            response = await self.chat_model.ainvoke(messages)
            return self._planner_parser.parse(message_text(response))
        except (ValidationError, ValueError, json.JSONDecodeError, TypeError):
            return None

    def _validate_actions(
        self,
        actions: list[PlannedAction],
    ) -> tuple[list[PlannedAction] | None, str]:
        if not actions:
            return None, "At least one tool action is required."
        if len(actions) > MAX_ACTIONS:
            return None, f"No more than {MAX_ACTIONS} actions are allowed."

        prepared: list[PlannedAction] = []
        errors: list[str] = []
        for index, action in enumerate(actions, start=1):
            selected_tool = self.tool_registry.get(action.tool)
            if selected_tool is None:
                errors.append(f"Action {index} uses unavailable tool {action.tool!r}.")
                continue
            try:
                schema = selected_tool.get_input_schema()
                validated = schema.model_validate(action.arguments)
                arguments = validated.model_dump(exclude_none=False)
            except Exception as exc:
                errors.append(f"Action {index} ({action.tool}) has invalid arguments: {exc}")
                continue
            prepared.append(PlannedAction(tool=action.tool, arguments=arguments))

        if errors:
            return None, " ".join(errors)
        return prepared, ""

    async def _execute_and_report(
        self,
        user_text: str,
        actions: list[PlannedAction],
        started_at: float,
    ) -> LLMResult:
        results: list[dict[str, Any]] = []
        for action in actions:
            selected_tool = self.tool_registry[action.tool]
            try:
                output = await selected_tool.ainvoke(action.arguments)
                output_ok = output.get("ok", True) if isinstance(output, dict) else True
                results.append({"tool": action.tool, "ok": bool(output_ok), "result": output})
            except Exception as exc:
                results.append({"tool": action.tool, "ok": False, "error": str(exc)})

        content = summarize_tool_results(results)
        self._remember(user_text, content)
        return measured_result(content, started_at)

    async def _handle_pending_confirmation(
        self,
        user_text: str,
        started_at: float,
    ) -> LLMResult | None:
        pending = self.pending_confirmation
        if pending is None:
            return None

        answer = confirmation_answer(user_text)
        if answer is True:
            self.pending_confirmation = None
            return await self._execute_and_report(pending.user_text, pending.actions, started_at)
        if answer is False:
            self.pending_confirmation = None
            return measured_result("No problem. I cancelled that action.", started_at)

        self.pending_confirmation = None
        return None

    async def _stream_natural_answer(
        self,
        user_text: str,
        on_token: TokenCallback | None,
        started_at: float,
    ) -> LLMResult:
        messages = [
            SystemMessage(
                content="You are Portable AI, a concise and honest local Windows assistant. Answer naturally."
            ),
            *self._recent_messages(),
            HumanMessage(content=user_text),
        ]
        chunks: list[str] = []
        first_token_at: float | None = None
        async for chunk in self.chat_model.astream(messages):
            text = message_text(chunk)
            if not text:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(text)
            if on_token is not None:
                on_token(text)

        content = "".join(chunks)
        self._remember(user_text, content)
        return measured_result(content, started_at, first_token_at)

    def _recent_messages(self) -> list[BaseMessage]:
        snapshot = self._memory_graph.get_state(self._memory_config)
        messages = snapshot.values.get("messages", [])
        return list(messages[-MAX_HISTORY_MESSAGES:])

    def _remember(self, user_text: str, assistant_text: str) -> None:
        self._memory_graph.invoke(
            {"messages": [HumanMessage(content=user_text), AIMessage(content=assistant_text)]},
            self._memory_config,
        )


def build_memory_graph(checkpointer: InMemorySaver) -> Any:
    builder = StateGraph(ConversationState)
    builder.add_node("remember", lambda state: {})
    builder.add_edge(START, "remember")
    builder.add_edge("remember", END)
    return builder.compile(checkpointer=checkpointer)


def confirmation_answer(text: str) -> bool | None:
    normalized = " ".join(text.strip().lower().split())
    if normalized in {"y", "yes", "yes please", "confirm", "proceed", "do it"}:
        return True
    if normalized in {"n", "no", "no thanks", "cancel", "stop", "do not"}:
        return False
    return None


def confirmation_prompt(actions: list[PlannedAction]) -> str:
    descriptions = [describe_action(action) for action in actions]
    joined = ", then ".join(descriptions)
    return f"Did you mean: {joined}? Reply yes to run it, or no to cancel."


def describe_action(action: PlannedAction) -> str:
    target = action.arguments.get("application") or action.arguments.get("url") or action.arguments.get("filename")
    if target:
        return f"{action.tool.replace('_', ' ')} {target}"
    return action.tool.replace("_", " ")


def summarize_tool_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No tool action was completed."
    return " ".join(summarize_tool_result(item) for item in results)


def summarize_tool_result(item: dict[str, Any]) -> str:
    tool_name = str(item.get("tool") or "tool")
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    error = item.get("error")
    if error:
        return f"I could not run {tool_name}: {error}."

    if tool_name == "open_application":
        app = display_application(str(result.get("application") or "application"))
        return f"I opened {app}." if item.get("ok") else f"I could not open {app}."

    if tool_name in {"close_application", "notepad_close"}:
        default_app = "notepad" if tool_name == "notepad_close" else "application"
        app = display_application(str(result.get("application") or default_app))
        closed = result.get("closed")
        targeted = result.get("targeted")
        if item.get("ok") and targeted == 0:
            return f"No visible {app} windows were found."
        if item.get("ok") and isinstance(closed, int):
            return f"I closed {closed} visible {app} window{'s' if closed != 1 else ''}."
        remaining = result.get("remaining")
        return f"I could not close every {app} window; remaining: {remaining}."

    if tool_name == "list_application_windows":
        app = display_application(str(result.get("application") or "application"))
        windows = result.get("windows") or []
        if not item.get("ok"):
            return f"I could not inspect {app} windows."
        if not windows:
            return f"No visible {app} windows are open."
        titles = [str(window.get("title") or "untitled") for window in windows[:5]]
        return f"Visible {app} windows ({len(windows)}): {', '.join(titles)}."

    if tool_name == "open_url":
        url = str(result.get("url") or "the requested URL")
        return f"I opened {url}." if item.get("ok") else f"The browser did not confirm opening {url}."

    if tool_name == "youtube_search_open":
        url = result.get("url")
        title = result.get("title")
        if not item.get("ok"):
            return "I could not open a YouTube result."
        if title and url:
            return f"I opened the best YouTube result: {title} ({url})."
        if url:
            return f"I opened YouTube results at {url}."

    if tool_name in {"notepad_open", "open_notepad_with_text", "notepad_type", "notepad_save", "notepad_save_as"}:
        message = str(result.get("message") or "Notepad action completed.")
        path = result.get("path")
        if not item.get("ok"):
            return f"Notepad action failed: {message}"
        return f"{message} File: {path}." if path else message

    if tool_name == "clipboard_write":
        count = result.get("characters")
        return f"I copied {count} characters to the clipboard." if isinstance(count, int) else "I updated the clipboard."

    if tool_name == "clipboard_read":
        text = str(result.get("text") or "")
        return f"Clipboard text: {text}" if text else "The clipboard does not contain text."

    if tool_name == "system_information":
        return (
            f"System: {result.get('windows')}; CPU: {result.get('cpu')} "
            f"({result.get('cpu_physical_cores')} physical, {result.get('cpu_logical_cores')} logical cores); "
            f"RAM: {result.get('ram_total_gb')} GB total, {result.get('ram_available_gb')} GB available."
        )

    if tool_name == "list_processes":
        processes = result.get("processes") or []
        names = [str(process.get("name")) for process in processes[:5]]
        return f"Running processes: {result.get('count', len(processes))}; examples: {', '.join(names)}."

    if tool_name == "active_window":
        if not item.get("ok"):
            return "I could not identify the active window."
        return f"Active window: {result.get('title') or '(untitled)'} ({result.get('process') or 'unknown'})."

    message = result.get("message")
    if message:
        return str(message).rstrip(".") + "."
    return f"{tool_name.replace('_', ' ').capitalize()} {'completed' if item.get('ok') else 'failed'}."


def display_application(application: str) -> str:
    return {
        "notepad": "Notepad",
        "calculator": "Calculator",
        "calc": "Calculator",
        "chrome": "Chrome",
        "whatsapp": "WhatsApp",
    }.get(application.lower(), application)


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def render_history(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        text = message_text(message).strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def measured_result(
    content: str,
    started_at: float,
    first_token_at: float | None = None,
) -> LLMResult:
    total = max(0.0, time.perf_counter() - started_at)
    ttft = total if first_token_at is None else max(0.0, first_token_at - started_at)
    return LLMResult(
        content=content,
        metrics=GenerationMetrics(
            ttft_seconds=ttft,
            total_seconds=total,
            output_characters=len(content),
            approx_tokens=estimate_tokens(content),
        ),
    )
