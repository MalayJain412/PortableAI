from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict
from urllib.parse import parse_qs, quote_plus, urlparse

from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from config import AGENT_STATE_DB
from langchain_tools import LANGCHAIN_TOOLS
from llm import GenerationMetrics, LLMResult, LlamafileClient, TokenCallback, estimate_tokens
from plan_store import PlanStore
from prompts import load_prompt


MAX_ACTIONS = 8
MAX_HISTORY_MESSAGES = 12
CONFIRMATION_REQUIRED_TOOLS = frozenset(
    {"delete_directory", "delete_file", "close_all_visible_applications", "shutdown_computer"}
)
POWER_TOOLS = frozenset({"sleep_computer", "shutdown_computer", "lock_computer"})
TERMINAL_POWER_TOOLS = frozenset({"sleep_computer", "shutdown_computer"})

ASSISTANT_IDENTITY = load_prompt("identity")
TOOL_CATALOG = load_prompt("tool_catalog")
TOOL_EXECUTION_GUIDE = load_prompt("tool_execution_guide")
SYSTEM_TOOL_AGENT = load_prompt(
    "system_tool_agent",
    assistant_identity=ASSISTANT_IDENTITY,
    tool_execution_guide=TOOL_EXECUTION_GUIDE,
)
SYSTEM_STRUCTURED_PLANNER = load_prompt(
    "system_structured_planner",
    assistant_identity=ASSISTANT_IDENTITY,
    tool_catalog=TOOL_CATALOG,
    tool_execution_guide=TOOL_EXECUTION_GUIDE,
)
SYSTEM_NATURAL_ANSWER = load_prompt("system_natural_answer", assistant_identity=ASSISTANT_IDENTITY)
TOOL_CALLING_PROBE_SYSTEM = load_prompt("tool_calling_probe_system")
TOOL_CALLING_PROBE_USER = load_prompt("tool_calling_probe_user")

APPLICATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgithub\s*desktop\b", re.IGNORECASE), "github_desktop"),
    (re.compile(r"\b(?:mongo\s*db|mongo)\s*compass\b", re.IGNORECASE), "mongodb_compass"),
    (re.compile(r"\b(?:visual\s*studio\s*code|vs\s*code|vscode)\b", re.IGNORECASE), "vscode"),
    (re.compile(r"\bwhats\s*app\b", re.IGNORECASE), "whatsapp"),
    (re.compile(r"\bnotion\b", re.IGNORECASE), "notion"),
    (re.compile(r"\boutlook\b", re.IGNORECASE), "outlook"),
    (re.compile(r"\btermius\b", re.IGNORECASE), "termius"),
    (re.compile(r"\bchrome\b", re.IGNORECASE), "chrome"),
    (re.compile(r"\bnote\s*pad\b", re.IGNORECASE), "notepad"),
    (re.compile(r"\bcalculator\b", re.IGNORECASE), "calculator"),
    (re.compile(r"\bcalc\b", re.IGNORECASE), "calc"),
)
APPLICATION_ARGUMENT_ALIASES = {
    "notepad": "notepad",
    "calculator": "calculator",
    "calc": "calculator",
    "chrome": "chrome",
    "whatsapp": "whatsapp",
    "notion": "notion",
    "vscode": "vscode",
    "visualstudiocode": "vscode",
    "githubdesktop": "github_desktop",
    "outlook": "outlook",
    "mongodb": "mongodb_compass",
    "mongodbcompass": "mongodb_compass",
    "mongocompass": "mongodb_compass",
    "termius": "termius",
}
DOMAIN_PATTERN = re.compile(
    r"(?i)\b(?:https?://)?(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?:/[^\s,;]*)?"
)
CLOSE_WORDS = re.compile(r"\b(close|quit|shut|terminate)\b", re.IGNORECASE)
OPEN_WORDS = re.compile(r"\b(open|launch|start)\b", re.IGNORECASE)
WINDOW_STATE_WORDS = re.compile(
    r"(?:\b(?:is|are)\b.*\bopen\b|\bstill\s+open\b|\bcheck\b.*\bopen\b)",
    re.IGNORECASE,
)
WEB_CONTEXT_WORDS = re.compile(
    r"\b(?:website|web\s*site|browser|online|webpage|page|link|url|chrome|google)\b|https?://|\.[a-z]{2,}\b",
    re.IGNORECASE,
)
WEB_RESOURCE_WORDS = re.compile(
    r"\b(?:website|web\s*site|webpage|page|link|url|online)\b",
    re.IGNORECASE,
)
WEB_SEARCH_WORDS = re.compile(r"\b(?:search|look\s*up|google)\b", re.IGNORECASE)
BROWSER_DESTINATION_WORDS = re.compile(r"\b(?:chrome|google|browser|web)\b", re.IGNORECASE)
WEB_NAVIGATION_WORDS = re.compile(r"\b(?:open|navigate|visit|go)\b", re.IGNORECASE)
BROWSER_QUALIFIER = re.compile(
    r"\b(?:on|in|using|with)\s+(?:google\s+chrome|chrome|google|the\s+browser|browser|the\s+web|web)\b",
    re.IGNORECASE,
)
WEB_SEARCH_REQUEST_PATTERNS = (
    re.compile(
        r"\b(?:search(?:\s+for)?|look\s+up|find)\s+(?P<query>.+?)\s+"
        r"(?:on|in|using|with)\s+(?:google(?:\s+chrome)?|chrome|the\s+browser|browser|the\s+web|web)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bopen\s+(?P<query>.+?)\s+(?:on|in|using|with)\s+google\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsearch\s+google\s+for\s+(?P<query>.+?)(?:[?.!]|$)", re.IGNORECASE),
)
KNOWN_WEB_DESTINATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bnotion\b", re.IGNORECASE), "https://notion.so"),
    (re.compile(r"\byou\s*tube\b", re.IGNORECASE), "https://youtube.com"),
    (re.compile(r"\bchat\s*gpt\b", re.IGNORECASE), "https://chatgpt.com"),
)
ACTIVE_WINDOW_REQUEST_WORDS = re.compile(
    r"\b(?:active|foreground|current)\s+window\b|\bwhat\s+window\b",
    re.IGNORECASE,
)
DIRECTORY_CREATE_WORDS = re.compile(
    r"\b(?:create|make|mkdir)\b|\bnew\s+(?:folder|directory)\b|\badd\s+(?:a\s+)?(?:folder|directory)\b",
    re.IGNORECASE,
)
DIRECTORY_DELETE_WORDS = re.compile(r"\b(?:delete|remove|erase)\b", re.IGNORECASE)
DIRECTORY_READ_WORDS = re.compile(
    r"\b(?:list|show|look|inspect|browse|contents?|inside|folders?|directories|files?|tree)\b",
    re.IGNORECASE,
)
DIRECTORY_SUMMARY_WORDS = re.compile(r"\b(?:summarize|summary|overview|analyze)\b", re.IGNORECASE)
DIRECTORY_TREE_WORDS = re.compile(r"\btree(?:\s+format)?\b", re.IGNORECASE)
DIRECTORY_RECURSIVE_WORDS = re.compile(
    r"\b(?:recursive|recursively|nested)\b|\b(?:all|every)\s+levels?\b|\b(?:entire|whole)\s+tree\b",
    re.IGNORECASE,
)
DIRECTORY_FOLLOWUP_WORDS = re.compile(
    r"\b(?:it|its|them|those|these|same|actual|current)\b",
    re.IGNORECASE,
)
FILE_RENAME_WORDS = re.compile(
    r"\b(?:rename|change\s+(?:the\s+)?file\s*name|edit\s+(?:the\s+)?file\s*name)\b",
    re.IGNORECASE,
)
FILE_COPY_WORDS = re.compile(r"\b(?:copy|duplicate)\b", re.IGNORECASE)
FILE_MOVE_WORDS = re.compile(r"\b(?:move|cut)\b", re.IGNORECASE)
FILE_DELETE_WORDS = re.compile(r"\b(?:delete|remove|erase)\b", re.IGNORECASE)
FILE_OVERWRITE_WORDS = re.compile(r"\b(?:overwrite|replace)\b", re.IGNORECASE)
NOTEPAD_WRITE_WORDS = re.compile(r"\b(?:write|type|add|insert|put)\b", re.IGNORECASE)
NOTEPAD_SAVE_WORDS = re.compile(r"\bsave\b", re.IGNORECASE)
POWER_NEGATION_WORDS = re.compile(
    r"\b(?:do\s+not|don't|dont|never)\b.{0,40}"
    r"\b(?:shutdown|shut\s+down|sleep|suspend|lock|close)\b",
    re.IGNORECASE,
)
POWER_DISCUSSION_WORDS = re.compile(
    r"^\s*(?:how|why|what|when|where|should\s+i|can\s+i|could\s+i|"
    r"does|is\s+it)\b|\bhow\s+to\b|\b(?:explain|describe)\b",
    re.IGNORECASE,
)
CLOSE_ALL_REQUEST = re.compile(
    r"\bclose\s+(?:all|every)\s+(?:open\s+)?(?:applications?|apps?|windows?)\b",
    re.IGNORECASE,
)
SHUTDOWN_REQUEST = re.compile(
    r"\b(?:shutdown|shut\s+down|turn\s+off|power\s+off)\s+"
    r"(?:my\s+|the\s+|this\s+)?(?:laptop|computer|pc|machine|windows)\b",
    re.IGNORECASE,
)
SLEEP_REQUEST = re.compile(
    r"\b(?:(?:put|send)\s+(?:my\s+|the\s+|this\s+)?(?:laptop|computer|pc|machine|windows)\s+to\s+sleep|"
    r"(?:sleep|suspend)\s+(?:my\s+|the\s+|this\s+)?(?:laptop|computer|pc|machine|windows))\b",
    re.IGNORECASE,
)
LOCK_REQUEST = re.compile(
    r"\block\s+(?:my\s+|the\s+|this\s+)?(?:laptop|computer|pc|machine|windows|screen|workstation)\b",
    re.IGNORECASE,
)
CLAUSE_SEPARATOR = re.compile(
    r"\s*(?:,\s*(?:and\s+)?|;\s*|\b(?:and\s+then|then|and)\b)\s*",
    re.IGNORECASE,
)


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
    plan_id: str
    user_text: str
    actions: list[PlannedAction]
    intent: str
    source: str


@dataclass(frozen=True)
class DirectoryContext:
    path: str
    directories: tuple[str, ...]
    include: Literal["all", "directories", "files"]
    recursive: bool
    output_format: Literal["compact", "tree"]


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
        plan_store: PlanStore | None = None,
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
        self.plan_store = plan_store or PlanStore(AGENT_STATE_DB)
        self.pending_confirmation: PendingConfirmation | None = None
        self.directory_context: DirectoryContext | None = None
        self.exit_requested = False
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
        if self.pending_confirmation is not None:
            self.plan_store.cancel_plan(self.pending_confirmation.plan_id)
        self.pending_confirmation = None
        self.directory_context = None

    async def handle(self, user_text: str, on_token: TokenCallback | None = None) -> LLMResult:
        started_at = time.perf_counter()
        pending_result = await self._handle_pending_confirmation(user_text, started_at)
        if pending_result is not None:
            return pending_result

        alias_plan = exact_power_alias_plan(user_text)
        if alias_plan is not None:
            intent, actions = alias_plan
            prepared, error = self._validate_actions(actions)
            if prepared is None:
                return measured_result(f"I could not prepare that power routine: {error}", started_at)
            return await self._execute_or_confirm(
                user_text,
                prepared,
                started_at,
                intent=intent,
                source="exact_alias",
            )

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
                    SystemMessage(content=TOOL_CALLING_PROBE_SYSTEM),
                    HumanMessage(content=TOOL_CALLING_PROBE_USER),
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
                        content=load_prompt("native_tool_call_retry", validation_errors=validation_feedback)
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
                actions = self._reconcile_actions(user_text, actions)
                prepared, validation_feedback = self._validate_actions(actions)
                if prepared is None:
                    continue
                return await self._execute_or_confirm(user_text, prepared, started_at)

            content = message_text(response)
            required_actions = self._reconcile_actions(user_text, [])
            if required_actions:
                prepared, validation_feedback = self._validate_actions(required_actions)
                if prepared is None:
                    continue
                return await self._execute_or_confirm(user_text, prepared, started_at)
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

            reconciled_actions = self._reconcile_actions(user_text, plan.actions)
            if plan.type == "answer" and reconciled_actions:
                prepared, validation_feedback = self._validate_actions(reconciled_actions)
                if prepared is None:
                    continue
                return await self._execute_or_confirm(user_text, prepared, started_at)

            if plan.type == "answer":
                content = plan.content.strip()
                if not content:
                    break
                self._remember(user_text, content)
                if on_token is not None:
                    on_token(content)
                return measured_result(content, started_at)

            prepared, validation_feedback = self._validate_actions(reconciled_actions)
            if prepared is None:
                continue

            if plan.type == "confirm":
                return await self._execute_or_confirm(
                    user_text,
                    prepared,
                    started_at,
                    force_confirmation=True,
                    confirmation_text=plan.content.strip() or None,
                )

            return await self._execute_or_confirm(user_text, prepared, started_at)

        return await self._stream_natural_answer(user_text, on_token, started_at)

    async def _create_structured_plan(self, user_text: str, feedback: str) -> ToolPlan | None:
        history_text = render_history(self._recent_messages())
        system_sections = [
            SYSTEM_STRUCTURED_PLANNER,
            f"JSON schema instructions:\n{self._planner_parser.get_format_instructions()}",
        ]
        if history_text:
            system_sections.append(f"Recent conversation:\n{history_text}")
        if feedback:
            system_sections.append(load_prompt("structured_plan_retry", validation_errors=feedback))
        messages: list[BaseMessage] = [
            SystemMessage(content="\n\n".join(system_sections)),
            HumanMessage(content=user_text),
        ]

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

        terminal_indexes = [
            index for index, action in enumerate(prepared) if action.tool in TERMINAL_POWER_TOOLS
        ]
        if len(terminal_indexes) > 1:
            return None, "A plan may contain only one sleep or shutdown action."
        if terminal_indexes and terminal_indexes[0] != len(prepared) - 1:
            return None, "Sleep or shutdown must be the final action in the plan."
        return prepared, ""

    def _reconcile_actions(
        self,
        user_text: str,
        actions: list[PlannedAction],
    ) -> list[PlannedAction]:
        explicit_actions = reconcile_explicit_actions(user_text, actions)
        notepad_actions = reconcile_notepad_actions(user_text, explicit_actions)
        directory_actions = reconcile_directory_actions(
            user_text,
            notepad_actions,
            self.directory_context,
        )
        file_actions = reconcile_file_actions(user_text, directory_actions)
        return reconcile_power_actions(user_text, file_actions)

    async def _execute_and_report(
        self,
        user_text: str,
        actions: list[PlannedAction],
        started_at: float,
        plan_id: str,
    ) -> LLMResult:
        results: list[dict[str, Any]] = []
        close_all_failed = False
        shutdown_dispatched = False
        self.plan_store.start_plan(plan_id)
        for sequence, action in enumerate(actions):
            requires_previous_success = (
                action.tool in TERMINAL_POWER_TOOLS
                and any(previous.tool == "close_all_visible_applications" for previous in actions[:sequence])
            )
            if requires_previous_success and close_all_failed:
                error = "Skipped because an earlier close-all action did not complete successfully"
                results.append({"tool": action.tool, "ok": False, "error": error})
                self.plan_store.finish_action(
                    plan_id,
                    sequence,
                    status="blocked",
                    error=error,
                )
                continue

            selected_tool = self.tool_registry[action.tool]
            self.plan_store.start_action(plan_id, sequence)
            try:
                output = await selected_tool.ainvoke(action.arguments)
                output_ok = output.get("ok", True) if isinstance(output, dict) else True
                results.append({"tool": action.tool, "ok": bool(output_ok), "result": output})
                if output_ok and isinstance(output, dict):
                    self._remember_directory_result(action, output)
                action_status = "succeeded" if output_ok else "failed"
                if (
                    action.tool == "shutdown_computer"
                    and output_ok
                    and isinstance(output, dict)
                    and output.get("dispatched")
                ):
                    action_status = "dispatched"
                    shutdown_dispatched = True
                    self.exit_requested = True
                self.plan_store.finish_action(
                    plan_id,
                    sequence,
                    status=action_status,
                    result=output,
                    error=None if output_ok else str(output.get("message") or "Tool reported failure"),
                )
                if action.tool == "close_all_visible_applications" and not output_ok:
                    close_all_failed = True
            except Exception as exc:
                error = str(exc)
                results.append({"tool": action.tool, "ok": False, "error": error})
                self.plan_store.finish_action(
                    plan_id,
                    sequence,
                    status="failed",
                    error=error,
                )
                if action.tool == "close_all_visible_applications":
                    close_all_failed = True

        content = summarize_tool_results(results)
        failed = any(not bool(item.get("ok")) for item in results)
        final_status = "failed" if failed else "dispatched" if shutdown_dispatched else "completed"
        self.plan_store.finish_plan(
            plan_id,
            status=final_status,
            summary=content,
            error="One or more planned actions failed." if failed else None,
        )
        self._remember(user_text, content)
        return measured_result(content, started_at)

    def _remember_directory_result(self, action: PlannedAction, output: dict[str, Any]) -> None:
        if action.tool == "list_directory":
            entries = output.get("entries") if isinstance(output.get("entries"), list) else []
            directories = tuple(
                str(entry.get("path") or entry.get("name"))
                for entry in entries
                if isinstance(entry, dict) and entry.get("type") == "directory"
            )
            self.directory_context = DirectoryContext(
                path=str(output.get("path") or action.arguments.get("path") or "."),
                directories=directories,
                include=normalized_directory_include(output.get("include")),
                recursive=bool(output.get("recursive", False)),
                output_format=normalized_directory_output_format(output.get("output_format")),
            )
        elif action.tool == "summarize_directory":
            self.directory_context = DirectoryContext(
                path=str(output.get("path") or action.arguments.get("path") or "."),
                directories=(),
                include="all",
                recursive=True,
                output_format="compact",
            )

    async def _execute_or_confirm(
        self,
        user_text: str,
        actions: list[PlannedAction],
        started_at: float,
        *,
        intent: str = "tool_request",
        source: str = "llm",
        force_confirmation: bool = False,
        confirmation_text: str | None = None,
    ) -> LLMResult:
        confirmation_required = force_confirmation or requires_confirmation(actions, source, intent)
        stored_plan = self.plan_store.create_plan(
            user_text=user_text,
            intent=intent,
            source=source,
            actions=[
                {
                    "tool": action.tool,
                    "arguments": action.arguments,
                    "requires_previous_success": (
                        action.tool in TERMINAL_POWER_TOOLS
                        and any(
                            previous.tool == "close_all_visible_applications"
                            for previous in actions[:index]
                        )
                    ),
                }
                for index, action in enumerate(actions)
            ],
            confirmation_required=confirmation_required,
        )
        if confirmation_required:
            self.pending_confirmation = PendingConfirmation(
                plan_id=stored_plan.plan_id,
                user_text=user_text,
                actions=actions,
                intent=intent,
                source=source,
            )
            content = confirmation_text or safety_confirmation_prompt(actions, intent)
            return measured_result(content, started_at)
        return await self._execute_and_report(user_text, actions, started_at, stored_plan.plan_id)

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
            self.plan_store.confirm_plan(pending.plan_id)
            return await self._execute_and_report(
                pending.user_text,
                pending.actions,
                started_at,
                pending.plan_id,
            )
        if answer is False:
            self.pending_confirmation = None
            self.plan_store.cancel_plan(pending.plan_id)
            return measured_result("No problem. I cancelled that action.", started_at)

        return measured_result(
            "A safety-sensitive plan is waiting. Reply yes to run it, or no to cancel.",
            started_at,
        )

    async def _stream_natural_answer(
        self,
        user_text: str,
        on_token: TokenCallback | None,
        started_at: float,
    ) -> LLMResult:
        messages = [
            SystemMessage(content=SYSTEM_NATURAL_ANSWER),
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


def exact_power_alias_plan(user_text: str) -> tuple[str, list[PlannedAction]] | None:
    """Map intentionally short power aliases before the text reaches the model."""
    command = normalized_exact_command(user_text)
    if command in {"goodnight", "good night", "tata", "shutdown", "shut down", "turn off", "power off"}:
        return (
            "shutdown_cleanup",
            [
                PlannedAction(tool="close_all_visible_applications", arguments={}),
                PlannedAction(tool="shutdown_computer", arguments={}),
            ],
        )
    if command == "deep sleep":
        return (
            "deep_sleep",
            [
                PlannedAction(tool="close_all_visible_applications", arguments={}),
                PlannedAction(tool="sleep_computer", arguments={}),
            ],
        )
    if command in {"sleep", "normal sleep"}:
        return "sleep_only", [PlannedAction(tool="sleep_computer", arguments={})]
    if command in {"lock", "hide"}:
        return "lock_only", [PlannedAction(tool="lock_computer", arguments={})]
    return None


def normalized_exact_command(user_text: str) -> str:
    lowered = user_text.strip().lower()
    without_punctuation = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return " ".join(without_punctuation.split())


def explicit_power_intent(user_text: str) -> Literal[
    "shutdown", "sleep", "lock", "close_all"
] | None:
    """Recognize explicit commands only; questions, explanations, and negations are denied."""
    if POWER_NEGATION_WORDS.search(user_text) or POWER_DISCUSSION_WORDS.search(user_text):
        return None
    if SHUTDOWN_REQUEST.search(user_text):
        return "shutdown"
    if SLEEP_REQUEST.search(user_text):
        return "sleep"
    if LOCK_REQUEST.search(user_text):
        return "lock"
    if CLOSE_ALL_REQUEST.search(user_text):
        return "close_all"
    return None


def reconcile_power_actions(
    user_text: str,
    proposed: list[PlannedAction],
) -> list[PlannedAction]:
    """Authorize power tools from the user's text and repair their required ordering."""
    intent = explicit_power_intent(user_text)
    ordinary_actions = [
        action
        for action in proposed
        if action.tool not in POWER_TOOLS and action.tool != "close_all_visible_applications"
    ]
    if intent is None:
        return ordinary_actions

    if intent == "shutdown":
        return [
            *ordinary_actions,
            PlannedAction(tool="close_all_visible_applications", arguments={}),
            PlannedAction(tool="shutdown_computer", arguments={}),
        ]
    if intent == "sleep":
        close_first = bool(CLOSE_ALL_REQUEST.search(user_text))
        return [
            *ordinary_actions,
            *(
                [PlannedAction(tool="close_all_visible_applications", arguments={})]
                if close_first
                else []
            ),
            PlannedAction(tool="sleep_computer", arguments={}),
        ]
    if intent == "lock":
        return [*ordinary_actions, PlannedAction(tool="lock_computer", arguments={})]
    return [PlannedAction(tool="close_all_visible_applications", arguments={}), *ordinary_actions]


def reconcile_notepad_actions(
    user_text: str,
    proposed: list[PlannedAction],
) -> list[PlannedAction]:
    actions = [normalize_action_arguments(action) for action in proposed]
    mentions_notepad = bool(re.search(r"\bnote\s*pad\b", user_text, re.IGNORECASE))
    opens_notepad = mentions_notepad and bool(OPEN_WORDS.search(user_text))
    requested_text = extract_notepad_text(user_text)

    if opens_notepad and requested_text is not None:
        replacement = PlannedAction(tool="notepad_open", arguments={"text": requested_text})
        replaceable = {
            "open_application",
            "notepad_open",
            "open_notepad_with_text",
            "notepad_type",
        }
        insertion = next(
            (
                index
                for index, action in enumerate(actions)
                if action.tool in replaceable and action_application(action) == "notepad"
            ),
            0,
        )
        actions = [
            action
            for action in actions
            if not (action.tool in replaceable and action_application(action) == "notepad")
        ]
        actions.insert(min(insertion, len(actions)), replacement)

    has_notepad_action = any(action_application(action) == "notepad" for action in actions)
    save_requested = bool(NOTEPAD_SAVE_WORDS.search(user_text))
    close_requested = bool(CLOSE_WORDS.search(user_text))
    if (mentions_notepad or has_notepad_action) and save_requested and close_requested:
        save_tools = {"notepad_save", "notepad_save_as"}
        close_tools = {"notepad_close", "close_application"}
        save_actions = [action for action in actions if action.tool in save_tools]
        close_actions = [
            action
            for action in actions
            if action.tool in close_tools and action_application(action) == "notepad"
        ]
        if not save_actions:
            save_actions = [PlannedAction(tool="notepad_save", arguments={})]
        if not close_actions:
            close_actions = [PlannedAction(tool="notepad_close", arguments={})]
        affected_indexes = [
            index
            for index, action in enumerate(actions)
            if action in save_actions or action in close_actions
        ]
        insertion = min(affected_indexes) if affected_indexes else len(actions)
        actions = [
            action
            for action in actions
            if action not in save_actions and action not in close_actions
        ]
        actions[insertion:insertion] = [*save_actions, *close_actions]
    return actions


def extract_notepad_text(user_text: str) -> str | None:
    write_match = NOTEPAD_WRITE_WORDS.search(user_text)
    if write_match is None:
        return None

    quoted = re.search(r"([\"'])(?P<text>.+?)\1", user_text[write_match.end() :])
    if quoted is not None:
        text = quoted.group("text").strip()
        return text or None

    tail = user_text[write_match.end() :]
    tail = re.sub(
        r"^\s*(?:the\s+text\s*)?"
        r"(?:(?:in|into|to)\s+(?:it|that|the\s+note|notepad|note\s*pad)\s*)?"
        r"[,;:]?\s*",
        "",
        tail,
        flags=re.IGNORECASE,
    )
    tail = re.split(
        r"\s*(?:,|;)?\s*\b(?:and\s+then|then|and)\s+(?:save|close)\b",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = tail.strip().strip("\"'").rstrip(".,; ")
    return text or None


def reconcile_file_actions(user_text: str, proposed: list[PlannedAction]) -> list[PlannedAction]:
    rename_intent = bool(FILE_RENAME_WORDS.search(user_text))
    copy_intent = bool(FILE_COPY_WORDS.search(user_text))
    move_intent = bool(FILE_MOVE_WORDS.search(user_text))
    delete_intent = bool(FILE_DELETE_WORDS.search(user_text) and explicit_file_reference(user_text))
    allowed = {
        "rename_file": rename_intent,
        "copy_file": copy_intent,
        "move_file": move_intent,
        "delete_file": delete_intent,
    }
    actions = [
        action
        for action in proposed
        if action.tool not in allowed or allowed[action.tool]
    ]
    overwrite_allowed = bool(FILE_OVERWRITE_WORDS.search(user_text))
    actions = [enforce_file_overwrite_intent(action, overwrite_allowed) for action in actions]

    required_rename = explicit_rename_file_action(user_text) if rename_intent else None
    if required_rename is not None:
        first_rename = next(
            (index for index, action in enumerate(actions) if action.tool == "rename_file"),
            len(actions),
        )
        actions = [action for action in actions if action.tool != "rename_file"]
        actions.insert(first_rename, required_rename)
    return actions


def explicit_file_reference(user_text: str) -> bool:
    names_directory_only = bool(
        re.search(r"\b(?:folder|directory)\b", user_text, re.IGNORECASE)
        and not re.search(r"\bfile\b", user_text, re.IGNORECASE)
    )
    if names_directory_only:
        return False
    return bool(
        re.search(r"\bfiles?\b", user_text, re.IGNORECASE)
        or re.search(r"[a-zA-Z]:[\\/]", user_text)
        or re.search(r"\b[^\s\\/]+\.[a-zA-Z0-9]{1,10}\b", user_text)
    )


def enforce_file_overwrite_intent(
    action: PlannedAction,
    overwrite_allowed: bool,
) -> PlannedAction:
    if action.tool not in {"rename_file", "copy_file", "move_file"}:
        return action
    arguments = dict(action.arguments)
    arguments["overwrite"] = bool(arguments.get("overwrite")) and overwrite_allowed
    return PlannedAction(tool=action.tool, arguments=arguments)


def explicit_rename_file_action(user_text: str) -> PlannedAction | None:
    quoted_path = re.search(r"[\"'](?P<path>[a-zA-Z]:[\\/][^\"']+)[\"']", user_text)
    if quoted_path is None:
        return None
    target = re.search(
        r"\b(?:as|to)\s+(?:the\s+)?(?P<name>[\"']?[^\"']+?[\"']?)"
        r"(?:\s+(?:and|then)\b.*)?[.!]?\s*$",
        user_text[quoted_path.end() :],
        re.IGNORECASE,
    )
    if target is None:
        return None
    new_name = target.group("name").strip().strip("\"'").rstrip(".,; ")
    if not new_name:
        return None
    return PlannedAction(
        tool="rename_file",
        arguments={"path": quoted_path.group("path"), "new_name": new_name, "overwrite": False},
    )


def reconcile_directory_actions(
    user_text: str,
    proposed: list[PlannedAction],
    context: DirectoryContext | None,
) -> list[PlannedAction]:
    create_intent = bool(DIRECTORY_CREATE_WORDS.search(user_text))
    delete_intent = bool(DIRECTORY_DELETE_WORDS.search(user_text))
    summarize_intent = bool(DIRECTORY_SUMMARY_WORDS.search(user_text))
    read_intent = summarize_intent or bool(DIRECTORY_READ_WORDS.search(user_text))
    followup = bool(DIRECTORY_FOLLOWUP_WORDS.search(user_text) or DIRECTORY_TREE_WORDS.search(user_text))

    reconciled: list[PlannedAction] = []
    for action in proposed:
        current = normalize_action_arguments(action)
        if current.tool == "create_directory" and not create_intent:
            if not read_intent:
                continue
            current = directory_read_action(current, summarize_intent)
        elif current.tool == "delete_directory" and not delete_intent:
            if not read_intent:
                continue
            current = directory_read_action(current, summarize_intent)

        if current.tool in {"list_directory", "summarize_directory"}:
            arguments = dict(current.arguments)
            raw_path = str(arguments.get("path") or ".")
            arguments["path"] = resolve_contextual_directory_path(raw_path, context, followup)
            if current.tool == "list_directory":
                arguments["include"] = infer_directory_include(user_text, arguments, context, followup)
                arguments["recursive"] = infer_directory_recursion(user_text, context, followup)
                arguments["output_format"] = (
                    "tree" if DIRECTORY_TREE_WORDS.search(user_text) else "compact"
                )
            current = PlannedAction(tool=current.tool, arguments=arguments)
        reconciled.append(current)

    required_creates = explicit_nested_directory_create_actions(user_text)
    if required_creates:
        reconciled = required_creates + [action for action in reconciled if action.tool != "create_directory"]
    return reconciled


def explicit_nested_directory_create_actions(user_text: str) -> list[PlannedAction]:
    if not DIRECTORY_CREATE_WORDS.search(user_text):
        return []

    children_match = re.search(
        r"\b(?:inside\s+it|inside\s+(?:that|the)\s+(?:directory|folder)|with)\s+"
        r"(?:create\s+|make\s+)?(?:\d+\s+)?"
        r"(?:sub[\s-]*(?:directories|folders)|directories|folders)"
        r"(?:\s+(?:named|called))?\s*[:,]?\s*(?P<children>.+?)\s*[.!?]*$",
        user_text,
        re.IGNORECASE,
    )
    if children_match is None:
        return []

    parent_match = re.search(
        r"\b(?:create|make)\s+(?:a\s+)?(?:directory|folder)\s+"
        r"(?:named\s+|called\s+)?(?P<parent>.+?)"
        r"(?:\s+(?:in|inside)\s+(?:the\s+)?workspace)?(?:\s+and)?\s*$",
        user_text[: children_match.start()],
        re.IGNORECASE,
    )
    if parent_match is None:
        return []

    parent = clean_directory_segment(parent_match.group("parent"))
    children = [
        clean_directory_segment(value)
        for value in re.split(r"\s*(?:,|\band\b)\s*", children_match.group("children"), flags=re.IGNORECASE)
    ]
    children = [child for child in children if child]
    if not parent or len(children) < 2:
        return []

    return [
        PlannedAction(tool="create_directory", arguments={"path": parent}),
        *[
            PlannedAction(tool="create_directory", arguments={"path": str(Path(parent) / child)})
            for child in children
        ],
    ]


def clean_directory_segment(value: str) -> str:
    cleaned = value.strip().strip("\"'").rstrip(".,;:!? ")
    if not cleaned or cleaned in {".", ".."} or any(separator in cleaned for separator in ("/", "\\")):
        return ""
    return cleaned


def directory_read_action(action: PlannedAction, summarize: bool) -> PlannedAction:
    path = action.arguments.get("path") or "."
    if summarize:
        return PlannedAction(tool="summarize_directory", arguments={"path": path})
    return PlannedAction(
        tool="list_directory",
        arguments={
            "path": path,
            "recursive": False,
            "include": "all",
            "output_format": "compact",
        },
    )


def resolve_contextual_directory_path(
    value: str,
    context: DirectoryContext | None,
    followup: bool,
) -> str:
    cleaned = value.strip().strip("\"'") or "."
    if context is None:
        return cleaned
    if cleaned == "." and followup:
        return context.path
    if Path(cleaned).is_absolute() or re.match(r"^[a-zA-Z]:[\\/]", cleaned):
        return cleaned

    matched = closest_context_directory(cleaned, context.directories)
    if matched is None:
        return cleaned
    return str(Path(context.path) / matched)


def closest_context_directory(value: str, directories: tuple[str, ...]) -> str | None:
    requested = normalized_directory_name(Path(value).name)
    if not requested:
        return None
    best_name: str | None = None
    best_score = 0.0
    for directory in directories:
        candidate_name = Path(directory).name
        candidate = normalized_directory_name(candidate_name)
        if candidate == requested:
            return directory
        score = SequenceMatcher(None, requested, candidate).ratio()
        if score > best_score:
            best_name = directory
            best_score = score
    return best_name if best_score >= 0.78 else None


def normalized_directory_name(value: str) -> str:
    without_label = re.sub(r"\s+(?:folder|directory)$", "", value.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", without_label.lower())


def infer_directory_include(
    user_text: str,
    arguments: dict[str, Any],
    context: DirectoryContext | None,
    followup: bool,
) -> Literal["all", "directories", "files"]:
    has_folders = bool(re.search(r"\b(?:folders|directories)\b", user_text, re.IGNORECASE))
    has_files = bool(re.search(r"\bfiles\b", user_text, re.IGNORECASE))
    if has_folders and not has_files:
        return "directories"
    if has_files and not has_folders:
        return "files"
    if has_files and has_folders:
        return "all"
    if followup and context is not None:
        return context.include
    return normalized_directory_include(arguments.get("include"))


def infer_directory_recursion(
    user_text: str,
    context: DirectoryContext | None,
    followup: bool,
) -> bool:
    if DIRECTORY_RECURSIVE_WORDS.search(user_text):
        return True
    if followup and context is not None:
        return context.recursive
    return False


def normalized_directory_include(value: Any) -> Literal["all", "directories", "files"]:
    return value if value in {"all", "directories", "files"} else "all"


def normalized_directory_output_format(value: Any) -> Literal["compact", "tree"]:
    return "tree" if value == "tree" else "compact"


def reconcile_explicit_actions(user_text: str, proposed: list[PlannedAction]) -> list[PlannedAction]:
    required = explicit_required_actions(user_text)
    normalized = [normalize_action_arguments(action) for action in proposed]
    if not explicit_web_navigation_requested(user_text):
        normalized = [action for action in normalized if action.tool != "open_url"]
    if not required:
        return normalized

    required_by_application: dict[str, list[PlannedAction]] = {}
    for action in required:
        application = action_application(action)
        if application is not None:
            required_by_application.setdefault(application, []).append(action)
    required_urls = [action for action in required if action.tool == "open_url"]
    remaining: list[PlannedAction] = []
    for action in normalized:
        application = action_application(action)
        expected_actions = required_by_application.get(application, [])
        if application is not None and is_application_control_action(action) and not expected_actions:
            continue
        if expected_actions and is_application_control_action(action):
            if not any(actions_equivalent(action, expected) for expected in expected_actions):
                continue
        if action.tool == "open_url" and required_urls:
            if not any(actions_equivalent(action, required_url) for required_url in required_urls):
                continue
        if action.tool == "active_window" and required_by_application and not ACTIVE_WINDOW_REQUEST_WORDS.search(user_text):
            continue
        if action.tool == "youtube_search_open" and required_urls:
            if any("youtube.com" in normalized_url(item.arguments.get("url")) for item in required_urls):
                continue
        remaining.append(action)

    reconciled: list[PlannedAction] = []
    for required_action in required:
        match_index = next(
            (index for index, action in enumerate(remaining) if actions_equivalent(action, required_action)),
            None,
        )
        if match_index is None:
            reconciled.append(required_action)
        else:
            reconciled.append(remaining.pop(match_index))
    reconciled.extend(remaining)

    unique: list[PlannedAction] = []
    for action in reconciled:
        if any(actions_equivalent(action, existing) for existing in unique):
            continue
        unique.append(action)
    return unique


def explicit_web_navigation_requested(user_text: str) -> bool:
    for clause in command_clauses(user_text):
        if DOMAIN_PATTERN.search(clause):
            return True
        if WEB_NAVIGATION_WORDS.search(clause) and WEB_RESOURCE_WORDS.search(clause):
            return True
        if WEB_SEARCH_WORDS.search(clause) and BROWSER_DESTINATION_WORDS.search(clause):
            return True
        if WEB_NAVIGATION_WORDS.search(clause) and BROWSER_QUALIFIER.search(clause):
            return True
        if re.search(r"\b(?:open|visit|navigate\s+to|go\s+to)\s+(?:google|youtube)\b", clause, re.IGNORECASE):
            return True
    return False


def explicit_required_actions(user_text: str) -> list[PlannedAction]:
    actions: list[PlannedAction] = []
    inherited_operation: Literal["open", "close", "inspect"] | None = None
    for clause in command_clauses(user_text):
        search_query = explicit_web_search_query(clause)
        if search_query is not None:
            if any(application == "chrome" for _, application in explicit_application_mentions(clause)):
                actions.append(PlannedAction(tool="open_application", arguments={"application": "chrome"}))
            actions.append(
                PlannedAction(
                    tool="open_url",
                    arguments={"url": google_search_url(search_query)},
                )
            )
            continue

        operation = explicit_operation(clause) or inherited_operation
        if operation is None:
            continue
        inherited_operation = operation

        applications = explicit_application_mentions(clause)
        urls = explicit_url_mentions(clause) if operation == "open" else []
        known_url = explicit_known_web_destination(clause) if operation == "open" else None
        if known_url is not None and not urls:
            urls.append((len(clause), known_url))
        if operation == "open" and (urls or WEB_CONTEXT_WORDS.search(clause)):
            applications = [item for item in applications if item[1] == "chrome"]
        if operation == "close":
            actions.extend(
                PlannedAction(tool="close_application", arguments={"application": application})
                for _, application in applications
            )
            continue
        if operation == "inspect":
            actions.extend(
                PlannedAction(tool="list_application_windows", arguments={"application": application})
                for _, application in applications
            )
            continue

        if urls and any(application == "chrome" for _, application in applications):
            for _, application in applications:
                actions.append(PlannedAction(tool="open_application", arguments={"application": application}))
                if application == "chrome":
                    actions.extend(
                        PlannedAction(tool="open_url", arguments={"url": url}) for _, url in urls
                    )
            continue

        events: list[tuple[int, PlannedAction]] = [
            (position, PlannedAction(tool="open_application", arguments={"application": application}))
            for position, application in applications
        ]
        events.extend(
            (position, PlannedAction(tool="open_url", arguments={"url": url}))
            for position, url in urls
        )
        events.sort(key=lambda item: item[0])
        actions.extend(action for _, action in events)
    return actions


def explicit_web_search_query(user_text: str) -> str | None:
    for pattern in WEB_SEARCH_REQUEST_PATTERNS:
        match = pattern.search(user_text)
        if match is None:
            continue
        query = match.group("query").strip().strip("\"'").rstrip("?.!").strip()
        if query:
            return query
    return None


def google_search_url(query: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(query)}"


def explicit_known_web_destination(user_text: str) -> str | None:
    if not BROWSER_QUALIFIER.search(user_text):
        return None
    for pattern, url in KNOWN_WEB_DESTINATIONS:
        if pattern.search(user_text):
            return url
    return None


def command_clauses(user_text: str) -> list[str]:
    return [clause.strip() for clause in CLAUSE_SEPARATOR.split(user_text) if clause.strip()]


def explicit_operation(user_text: str) -> Literal["open", "close", "inspect"] | None:
    if CLOSE_WORDS.search(user_text):
        return "close"
    if WINDOW_STATE_WORDS.search(user_text):
        return "inspect"
    if OPEN_WORDS.search(user_text):
        return "open"
    return None


def explicit_application_mentions(user_text: str) -> list[tuple[int, str]]:
    mentions: list[tuple[int, str]] = []
    seen: set[str] = set()
    for pattern, application in APPLICATION_PATTERNS:
        match = pattern.search(user_text)
        canonical = "calculator" if application == "calc" else application
        if match is None or canonical in seen:
            continue
        mentions.append((match.start(), canonical))
        seen.add(canonical)
    mentions.sort(key=lambda item: item[0])
    return mentions


def explicit_url_mentions(user_text: str) -> list[tuple[int, str]]:
    urls: list[tuple[int, str]] = []
    seen: set[str] = set()
    for match in DOMAIN_PATTERN.finditer(user_text):
        raw_url = match.group(0).rstrip(".?!)]}'\"")
        url = raw_url if raw_url.lower().startswith(("http://", "https://")) else f"https://{raw_url}"
        key = url.lower().rstrip("/")
        if key in seen:
            continue
        urls.append((match.start(), url))
        seen.add(key)
    return urls


def normalize_action_arguments(action: PlannedAction) -> PlannedAction:
    arguments = dict(action.arguments)
    if action.tool in {"open_application", "close_application", "list_application_windows"}:
        if "application" not in arguments:
            alternate = arguments.pop("name", None) or arguments.pop("app", None)
            if isinstance(alternate, str):
                arguments["application"] = canonical_application_name(alternate)
        elif isinstance(arguments.get("application"), str):
            arguments["application"] = canonical_application_name(str(arguments["application"]))
    if action.tool == "open_url" and "url" not in arguments:
        alternate_url = arguments.pop("website", None)
        if isinstance(alternate_url, str):
            arguments["url"] = alternate_url
    return PlannedAction(tool=action.tool, arguments=arguments)


def actions_equivalent(left: PlannedAction, right: PlannedAction) -> bool:
    left_app = action_application(left)
    right_app = action_application(right)
    if right.tool == "open_application" and right_app == "notepad":
        return left.tool in {"open_application", "notepad_open", "open_notepad_with_text"} and left_app in {
            None,
            "notepad",
        }
    if right.tool == "close_application" and right_app == "notepad":
        return left.tool in {"close_application", "notepad_close"} and left_app in {None, "notepad"}
    if left.tool != right.tool:
        return False
    if left.tool in {"open_application", "close_application", "list_application_windows"}:
        return left_app == right_app
    if left.tool == "open_url":
        return normalized_url(left.arguments.get("url")) == normalized_url(right.arguments.get("url"))
    return left.arguments == right.arguments


def action_application(action: PlannedAction) -> str | None:
    value = action.arguments.get("application")
    if isinstance(value, str):
        return canonical_application_name(value)
    if action.tool in {"notepad_open", "open_notepad_with_text", "notepad_type", "notepad_save", "notepad_save_as", "notepad_close"}:
        return "notepad"
    return None


def canonical_application_name(value: str) -> str:
    normalized = re.sub(r"[\s_-]+", "", value.strip().lower())
    return APPLICATION_ARGUMENT_ALIASES.get(normalized, value.strip().lower())


def is_application_control_action(action: PlannedAction) -> bool:
    return action.tool in {
        "open_application",
        "close_application",
        "list_application_windows",
        "notepad_open",
        "open_notepad_with_text",
        "notepad_close",
    }


def normalized_url(value: Any) -> str:
    return str(value or "").strip().lower().rstrip("/")


def confirmation_answer(text: str) -> bool | None:
    normalized = " ".join(text.strip().lower().split())
    if normalized in {"y", "yes", "yes please", "confirm", "proceed", "do it"}:
        return True
    if normalized in {"n", "no", "no thanks", "cancel", "/cancel", "stop", "do not"}:
        return False
    return None


def confirmation_prompt(actions: list[PlannedAction]) -> str:
    descriptions = [describe_action(action) for action in actions]
    joined = ", then ".join(descriptions)
    return f"Did you mean: {joined}? Reply yes to run it, or no to cancel."


def requires_confirmation(
    actions: list[PlannedAction],
    source: str = "llm",
    intent: str = "tool_request",
) -> bool:
    if any(action.tool in CONFIRMATION_REQUIRED_TOOLS for action in actions):
        return True
    if any(action.tool == "sleep_computer" for action in actions):
        return not (source == "exact_alias" and intent == "sleep_only")
    return False


def destructive_confirmation_prompt(actions: list[PlannedAction]) -> str:
    deletion_actions = [
        action for action in actions if action.tool in {"delete_directory", "delete_file"}
    ]
    deletion_paths = [
        str(action.arguments.get("path") or "the requested item") for action in deletion_actions
    ]
    targets = ", ".join(deletion_paths)
    plan_note = (
        f" The complete {len(actions)}-action plan will run only after confirmation."
        if len(actions) > 1
        else ""
    )
    includes_directory = any(action.tool == "delete_directory" for action in deletion_actions)
    directory_note = " Directory deletion also removes everything inside." if includes_directory else ""
    return f"This will permanently delete {targets}.{directory_note}{plan_note} Reply yes to continue, or no to cancel."


def safety_confirmation_prompt(actions: list[PlannedAction], intent: str) -> str:
    tools = {action.tool for action in actions}
    if "shutdown_computer" in tools:
        return (
            "This will gracefully close all visible applications and then shut down the laptop. "
            "If any application remains open, shutdown will be cancelled. "
            "Reply yes to continue, or no to cancel."
        )
    if "sleep_computer" in tools and "close_all_visible_applications" in tools:
        return (
            "This will gracefully close all visible applications and then put the laptop to sleep. "
            "If any application remains open, sleep will be cancelled. "
            "Reply yes to continue, or no to cancel."
        )
    if "close_all_visible_applications" in tools:
        return (
            "This will ask every visible application to close. Unsaved applications may remain open "
            "and will not be forced. Reply yes to continue, or no to cancel."
        )
    if "sleep_computer" in tools:
        return "This will put the laptop to sleep. Reply yes to continue, or no to cancel."
    if tools.intersection({"delete_directory", "delete_file"}):
        return destructive_confirmation_prompt(actions)
    return confirmation_prompt(actions)


def describe_action(action: PlannedAction) -> str:
    target = (
        action.arguments.get("application")
        or action.arguments.get("url")
        or action.arguments.get("filename")
        or action.arguments.get("path")
        or action.arguments.get("source")
    )
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

    if tool_name == "close_all_visible_applications":
        closed = int(result.get("closed") or 0)
        targeted = int(result.get("targeted") or 0)
        remaining = result.get("remaining") if isinstance(result.get("remaining"), list) else []
        if item.get("ok"):
            if targeted == 0:
                return "No other visible applications were open."
            return f"I closed {closed} visible application window{'s' if closed != 1 else ''}."
        titles = [str(window.get("title") or "untitled") for window in remaining if isinstance(window, dict)]
        detail = ", ".join(titles[:5]) or str(result.get("message") or "unknown windows")
        return f"I could not close every visible application; still open: {detail}."

    if tool_name == "sleep_computer":
        if item.get("ok"):
            return "The computer resumed from sleep."
        return "Windows did not enter sleep mode."

    if tool_name == "lock_computer":
        return "I locked Windows." if item.get("ok") else "Windows did not accept the lock request."

    if tool_name == "shutdown_computer":
        if item.get("ok") and result.get("dispatched"):
            delay = int(result.get("delay_seconds") or 0)
            if delay > 0:
                return f"Shutdown is scheduled in {delay} seconds."
            return "Windows accepted the shutdown request."
        return "Windows did not accept the shutdown request."

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
        search_query = google_query_from_url(url)
        if search_query is not None:
            if item.get("ok"):
                return f'I searched Google for "{search_query}" and opened the results.'
            return f'I could not open Google search results for "{search_query}".'
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

    if tool_name == "create_directory":
        path = str(result.get("relative_path") or result.get("path") or "directory")
        if not item.get("ok"):
            return f"I could not create directory {path}."
        if result.get("created"):
            return f"I created directory {path}."
        return f"Directory {path} already exists."

    if tool_name == "delete_directory":
        path = str(result.get("relative_path") or result.get("path") or "directory")
        if not item.get("ok"):
            return f"I could not delete directory {path}."
        files = int(result.get("deleted_files") or 0)
        directories = int(result.get("deleted_directories") or 0)
        return f"I permanently deleted {path} ({directories} directories and {files} files)."

    if tool_name == "rename_file":
        source = str(result.get("source") or "the requested file")
        path = str(result.get("path") or "the requested filename")
        if not item.get("ok"):
            return f"I could not rename {source}."
        if result.get("renamed") is False:
            return f"The file already has the requested name: {path}."
        return f"I renamed {source} to {path}."

    if tool_name == "copy_file":
        source = str(result.get("source") or "the requested file")
        path = str(result.get("path") or "the destination")
        return f"I copied {source} to {path}." if item.get("ok") else f"I could not copy {source}."

    if tool_name == "move_file":
        source = str(result.get("source") or "the requested file")
        path = str(result.get("path") or "the destination")
        return f"I moved {source} to {path}." if item.get("ok") else f"I could not move {source}."

    if tool_name == "delete_file":
        path = str(result.get("path") or "the requested file")
        return f"I permanently deleted {path}." if item.get("ok") else f"I could not delete {path}."

    if tool_name == "list_directory":
        path = str(result.get("relative_path") or result.get("path") or ".")
        entries = result.get("entries") if isinstance(result.get("entries"), list) else []
        if not item.get("ok"):
            return f"I could not list directory {path}."
        if not entries:
            return f"Directory {path} is empty."
        if result.get("output_format") == "tree":
            tree = format_directory_tree(path, entries)
            if result.get("truncated"):
                tree += "\n[scan limit reached; listing is partial]"
            return tree
        labels = [directory_entry_label(entry) for entry in entries[:20] if isinstance(entry, dict)]
        if result.get("truncated"):
            suffix = " The scan limit was reached, so the listing is partial."
        elif len(entries) > 20:
            suffix = f" Showing the first 20 of {len(entries)} entries."
        else:
            suffix = ""
        return f"Directory {path} contains: {', '.join(labels)}.{suffix}"

    if tool_name == "summarize_directory":
        path = str(result.get("relative_path") or result.get("path") or ".")
        if not item.get("ok"):
            return f"I could not summarize directory {path}."
        directories = int(result.get("directories") or 0)
        files = int(result.get("files") or 0)
        total_bytes = int(result.get("total_bytes") or 0)
        extensions = result.get("extensions") if isinstance(result.get("extensions"), dict) else {}
        extension_text = ", ".join(f"{name}: {count}" for name, count in list(extensions.items())[:5])
        details = f" Common types: {extension_text}." if extension_text else ""
        suffix = " Scan limit reached, so totals are partial." if result.get("truncated") else ""
        return (
            f"Directory {path} has {directories} subdirectories and {files} files "
            f"using {format_byte_count(total_bytes)}.{details}{suffix}"
        )

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


def directory_entry_label(entry: dict[str, Any]) -> str:
    path = str(entry.get("path") or entry.get("name") or "unnamed")
    entry_type = str(entry.get("type") or "item")
    return f"{path} [{entry_type}]"


def google_query_from_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host not in {"google.com", "www.google.com"} or parsed.path.rstrip("/") != "/search":
        return None
    query = parse_qs(parsed.query).get("q", [""])[0].strip()
    return query or None


def format_directory_tree(root: str, entries: list[Any]) -> str:
    valid_entries = [entry for entry in entries if isinstance(entry, dict)]
    valid_entries.sort(
        key=lambda entry: tuple(part.lower() for part in Path(str(entry.get("path") or "")).parts)
    )
    lines = [root]
    for entry in valid_entries:
        parts = Path(str(entry.get("path") or entry.get("name") or "unnamed")).parts
        if not parts:
            continue
        suffix = "/" if entry.get("type") == "directory" else ""
        lines.append(f"{'    ' * (len(parts) - 1)}|-- {parts[-1]}{suffix}")
    return "\n".join(lines)


def format_byte_count(size: int) -> str:
    value = float(max(size, 0))
    units = ("bytes", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "bytes":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} bytes"


def display_application(application: str) -> str:
    return {
        "notepad": "Notepad",
        "calculator": "Calculator",
        "calc": "Calculator",
        "chrome": "Chrome",
        "whatsapp": "WhatsApp",
        "notion": "Notion",
        "vscode": "VS Code",
        "github_desktop": "GitHub Desktop",
        "outlook": "Outlook",
        "mongodb_compass": "MongoDB Compass",
        "mongodb": "MongoDB Compass",
        "termius": "Termius",
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
