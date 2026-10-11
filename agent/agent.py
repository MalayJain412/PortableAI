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

from langchain_tools import LANGCHAIN_TOOLS
from llm import GenerationMetrics, LLMResult, LlamafileClient, TokenCallback, estimate_tokens


MAX_ACTIONS = 8
MAX_HISTORY_MESSAGES = 12
CONFIRMATION_REQUIRED_TOOLS = frozenset({"delete_directory"})

TOOL_CATALOG = """Available tools (each call performs one atomic action):
- open_application(application): open one allowlisted application.
- close_application(application): close all visible windows of one allowlisted application.
- list_application_windows(application): inspect current visible windows for one application.
- open_url(url): open one explicitly requested HTTP or HTTPS destination in the default browser.
- youtube_search_open(query): find and open one specific YouTube video, playlist, or channel.
- open_notepad_with_text(text): create and open a one-shot Notepad note with exact text.
- notepad_open(text): open the managed note, optionally replacing it with starting text.
- notepad_type(text): append exact text to the managed note.
- notepad_save(): report the managed note's current saved path.
- notepad_save_as(filename): save the managed note under a simple filename.
- notepad_close(): close visible Notepad windows.
- list_processes(): list running processes.
- system_information(): read basic local system information.
- active_window(): inspect the current foreground window.
- clipboard_read(): read text from the clipboard.
- clipboard_write(text): write exact text to the clipboard.
- create_directory(path): create a directory inside the managed PortableAI workspace.
- delete_directory(path): permanently delete a directory tree inside the managed workspace.
- list_directory(path, recursive, include, output_format): list entries in any existing local directory from names and metadata only.
- summarize_directory(path): summarize any existing local directory from counts, sizes, and file types only.
"""

TOOL_EXECUTION_GUIDE = """Plan Windows tasks with this exact workflow:

1. SPLIT: Divide the request into ordered clauses at words such as "and", "then", and commas.
2. INTERPRET: For each clause, identify its own action, target, and arguments. A later explicit verb replaces the earlier verb. A clause with no verb inherits the previous clause's verb.
3. MAP: Select one tool for every requested target. Keep all actions in user order.
4. DEPENDENCIES: Open a named browser before its URL. For Notepad, open/write before save and save before close.
5. AUDIT: Before responding, verify that every clause and every named target has exactly one appropriate action. Do not drop, duplicate, reverse, or change an action.

Mixed verbs are independent:
- "close Chrome and open WhatsApp" -> close_application(chrome), open_application(whatsapp)
- "open Chrome and close WhatsApp" -> open_application(chrome), close_application(whatsapp)
- "close Chrome and WhatsApp" -> close_application(chrome), close_application(whatsapp), because the second clause inherits "close"
- "close Chrome, then open Chrome" -> close_application(chrome), open_application(chrome)

Navigation:
- "open chatgpt.com on Chrome, and WhatsApp" -> open_application(chrome), open_url(https://chatgpt.com), open_application(whatsapp)
- "open YouTube on Chrome" -> open_application(chrome), open_url(https://youtube.com)
- "search for LangChain on Google" -> open_url(https://www.google.com/search?q=LangChain)
- "open LangChain on Google" means search Google for LangChain and uses the same search URL.
- A literal website or YouTube home is navigation with open_url.
- A specific YouTube video, playlist, channel, or content request uses youtube_search_open.
- Browser navigation is opt-in. Use open_url only when the user explicitly supplies a URL/domain, asks to open a link/website/page, names a web destination followed by "on/in Chrome" or "in the browser", or asks to search on Google/Chrome/the web.
- Never open Google, a website, or a search page as a fallback for an application request. "open Chrome" means open_application(chrome) only. "open Notion and Chrome" means two open_application calls and no URL.
- Do not invent a URL, search query, website, or browser action that the user did not request.
- Never say that a page was opened or a search was performed without emitting the corresponding tool call and receiving its result.

Desktop applications:
- A named allowlisted app means the installed desktop application by default. "open Notion" and "open Notion on laptop" both use open_application(notion).
- Use a website only when the user explicitly requests web navigation. "open the Notion website" and "open Notion on Chrome" may use open_url; plain "open Notion" must not.
- Never replace an open-app request with active_window, list_processes, or a similarly unrelated inspection tool.
- Interpret VS Code or Visual Studio Code as vscode, GitHub Desktop as github_desktop, MongoDB Compass as mongodb_compass, and Termius as termius.

Notepad:
- A one-shot open-and-write request uses open_notepad_with_text.
- A request that also saves or closes uses notepad_open with text, then save, then close.
- Treat "note pad" as "notepad" when the meaning is clear.

Live state:
- "is WhatsApp still open?" uses list_application_windows(whatsapp).
- Never infer current window state from conversation history or an earlier result.

Directories:
- "folder" and "directory" mean the same thing.
- Intent verbs are strict. create_directory requires an explicit create, make, mkdir, new-folder, or add-folder request. Never create for list, show, look, inspect, browse, contents, inside, folders, files, summarize, or tree requests.
- delete_directory requires an explicit delete, remove, or erase verb. Never infer a destructive action from a read-only request.
- create_directory and delete_directory are confined to the managed PortableAI workspace. Relative paths are based there.
- list_directory and summarize_directory may inspect any existing local drive directory. They never read file contents and never accept network paths.
- Convert a drive phrase to an absolute root path: "E drive" means path="E:\\". Do not treat it as a folder named "E drive" inside the workspace.
- Extract the complete requested path and preserve its words, spaces, hyphens, and nested separators. Remove only command wording, a leading "the", and a trailing "folder" or "directory" label.
- Never use path . when the user names a folder. Use . only when the user explicitly means the workspace root or gives no narrower path.
- "create a projects folder" -> create_directory(path="projects").
- "create a folder named client work" -> create_directory(path="client work").
- "show/list what is in projects" -> list_directory(path="projects", recursive=false, include="all", output_format="compact").
- "look inside the projects directory and list all its folders and files" -> list_directory(path="projects", recursive=false, include="all").
- "what are the folders in E drive?" -> list_directory(path="E:\\", recursive=false, include="directories", output_format="compact").
- After that result, "list them in a tree format" keeps path="E:\\", recursive=false, include="directories" and changes only output_format="tree".
- "what files are in E drive?" -> list_directory(path="E:\\", recursive=false, include="files", output_format="compact").
- "show every nested item in projects" -> list_directory(path="projects", recursive=true, include="all").
- Resolve a named child against the most recently listed directory. If E:\\GPT-mini listed "Makemoer Series", then "list contents of Makemore Series" means list_directory(path="E:\\GPT-mini\\Makemoer Series", recursive=false, include="all", output_format="compact"), never create_directory.
- "summarize the projects directory" -> summarize_directory(path="projects").
- "delete the old archive directory" -> delete_directory(path="old archive").
- "summarize the projects directory and delete the old archive directory" -> summarize_directory(path="projects"), delete_directory(path="old archive").
- Emit delete_directory normally; the host requests confirmation before running the complete schedule.
- Use one directory tool call for every requested path. Never read file contents to summarize a directory.

Safety and output:
- Canonical applications: notepad, calculator, calc, chrome, whatsapp, notion, vscode, github_desktop, outlook, mongodb_compass, termius.
- Use only provided tools and their exact argument names. Never invent an application, URL, tool, or argument.
- If a target is unsupported, explain the allowlist instead of substituting another target.
- If intent is genuinely ambiguous, ask one short confirmation question and call no tool.
- For ordinary conversation with no tool intent, answer naturally and concisely.
- Never claim success before receiving tool results.
- Maximum eight actions.
"""

SYSTEM_TOOL_AGENT = f"""You are Portable AI, a local offline Windows assistant with allowlisted tools.
The host provides each available tool's description and exact argument schema. When a request needs tools, silently build the complete schedule using the workflow below, then emit all required tool calls in one response. The host validates the full schedule and executes it sequentially.

{TOOL_EXECUTION_GUIDE}
"""

SYSTEM_STRUCTURED_PLANNER = f"""You are the structured action planner for Portable AI.
Return only data matching the supplied JSON schema. Use type "tools" for an executable schedule, "confirm" for a proposed schedule that needs confirmation, or "answer" when no tool should run.

{TOOL_CATALOG}

{TOOL_EXECUTION_GUIDE}
"""

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
DIRECTORY_FOLLOWUP_WORDS = re.compile(r"\b(?:it|its|them|those|these|same)\b", re.IGNORECASE)
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
    user_text: str
    actions: list[PlannedAction]


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
        self.directory_context: DirectoryContext | None = None
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
        self.directory_context = None

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
                self.pending_confirmation = PendingConfirmation(user_text=user_text, actions=prepared)
                if requires_confirmation(prepared):
                    content = destructive_confirmation_prompt(prepared)
                else:
                    content = plan.content.strip() or confirmation_prompt(prepared)
                return measured_result(content, started_at)

            return await self._execute_or_confirm(user_text, prepared, started_at)

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

    def _reconcile_actions(
        self,
        user_text: str,
        actions: list[PlannedAction],
    ) -> list[PlannedAction]:
        explicit_actions = reconcile_explicit_actions(user_text, actions)
        return reconcile_directory_actions(user_text, explicit_actions, self.directory_context)

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
                if output_ok and isinstance(output, dict):
                    self._remember_directory_result(action, output)
            except Exception as exc:
                results.append({"tool": action.tool, "ok": False, "error": str(exc)})

        content = summarize_tool_results(results)
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
    ) -> LLMResult:
        if requires_confirmation(actions):
            self.pending_confirmation = PendingConfirmation(user_text=user_text, actions=actions)
            return measured_result(destructive_confirmation_prompt(actions), started_at)
        return await self._execute_and_report(user_text, actions, started_at)

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
    return reconciled


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
    if normalized in {"n", "no", "no thanks", "cancel", "stop", "do not"}:
        return False
    return None


def confirmation_prompt(actions: list[PlannedAction]) -> str:
    descriptions = [describe_action(action) for action in actions]
    joined = ", then ".join(descriptions)
    return f"Did you mean: {joined}? Reply yes to run it, or no to cancel."


def requires_confirmation(actions: list[PlannedAction]) -> bool:
    return any(action.tool in CONFIRMATION_REQUIRED_TOOLS for action in actions)


def destructive_confirmation_prompt(actions: list[PlannedAction]) -> str:
    deletion_paths = [
        str(action.arguments.get("path") or "the requested directory")
        for action in actions
        if action.tool == "delete_directory"
    ]
    targets = ", ".join(deletion_paths)
    plan_note = (
        f" The complete {len(actions)}-action plan will run only after confirmation."
        if len(actions) > 1
        else ""
    )
    noun = "them" if len(deletion_paths) > 1 else "it"
    return (
        f"This will permanently delete {targets} and everything inside {noun}."
        f"{plan_note} Reply yes to continue, or no to cancel."
    )


def describe_action(action: PlannedAction) -> str:
    target = (
        action.arguments.get("application")
        or action.arguments.get("url")
        or action.arguments.get("filename")
        or action.arguments.get("path")
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
