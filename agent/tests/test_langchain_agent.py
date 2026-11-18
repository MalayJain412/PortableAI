from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage

from agent import PortableAgent
from langchain_tools import TOOL_REGISTRY
from plan_store import PlanStore


class FakeChatModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.bound_tool_names: list[str] = []
        self.seen_messages: list[list[Any]] = []

    def bind_tools(self, tools: list[Any]) -> "FakeChatModel":
        self.bound_tool_names = [item.name for item in tools]
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.seen_messages.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeChatModel has no response left.")
        return self.responses.pop(0)

    async def astream(self, messages: list[Any]) -> AsyncIterator[AIMessageChunk]:
        response = await self.ainvoke(messages)
        yield AIMessageChunk(content=response.content)


def tool_response(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": arguments, "id": f"call-{index}", "type": "tool_call"}
            for index, (name, arguments) in enumerate(calls, start=1)
        ],
    )


class PortableAgentToolCallingTests(unittest.IsolatedAsyncioTestCase):
    async def test_completeness_guard_adds_literal_url_between_apps(self) -> None:
        model = FakeChatModel(
            [
                tool_response(
                    ("open_application", {"application": "chrome"}),
                    ("open_url", {"url": "https://example.com"}),
                    ("open_application", {"application": "whatsapp"}),
                    ("open_application", {"application": "calculator"}),
                )
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[tuple[str, str]] = []

        def fake_open(application: str) -> dict[str, Any]:
            calls.append(("application", application))
            return {"ok": True, "application": application}

        def fake_url(url: str) -> dict[str, Any]:
            calls.append(("url", url))
            return {"ok": True, "url": url}

        with (
            patch("langchain_tools.windows_tools.open_application", side_effect=fake_open),
            patch("langchain_tools.windows_tools.open_url", side_effect=fake_url),
        ):
            await agent.handle("open chatgpt.com on chrome, and whatsapp on laptop")

        self.assertEqual(
            calls,
            [
                ("application", "chrome"),
                ("url", "https://chatgpt.com"),
                ("application", "whatsapp"),
            ],
        )

    async def test_completeness_guard_adds_missing_close_target(self) -> None:
        model = FakeChatModel([tool_response(("close_application", {"application": "chrome"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[str] = []

        def fake_close(application: str) -> dict[str, Any]:
            calls.append(application)
            return {
                "ok": True,
                "application": application,
                "closed": 1,
                "targeted": 1,
                "remaining": [],
            }

        with patch("langchain_tools.windows_tools.close_application", side_effect=fake_close):
            await agent.handle("ok close chrome AND WHATSAPP")

        self.assertEqual(calls, ["chrome", "whatsapp"])

    async def test_mixed_verbs_are_scheduled_per_clause(self) -> None:
        model = FakeChatModel(
            [
                tool_response(
                    ("close_application", {"application": "chrome"}),
                    ("close_application", {"application": "whatsapp"}),
                )
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[tuple[str, str]] = []

        def fake_close(application: str) -> dict[str, Any]:
            calls.append(("close", application))
            return {
                "ok": True,
                "application": application,
                "closed": 1,
                "targeted": 1,
                "remaining": [],
            }

        def fake_open(application: str) -> dict[str, Any]:
            calls.append(("open", application))
            return {"ok": True, "application": application}

        with (
            patch("langchain_tools.windows_tools.close_application", side_effect=fake_close),
            patch("langchain_tools.windows_tools.open_application", side_effect=fake_open),
        ):
            await agent.handle("close all chrome applications and open whatsapp")

        self.assertEqual(calls, [("close", "chrome"), ("open", "whatsapp")])

    async def test_clause_without_verb_inherits_previous_operation(self) -> None:
        model = FakeChatModel([tool_response(("close_application", {"application": "chrome"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[str] = []

        def fake_close(application: str) -> dict[str, Any]:
            calls.append(application)
            return {
                "ok": True,
                "application": application,
                "closed": 1,
                "targeted": 1,
                "remaining": [],
            }

        with patch("langchain_tools.windows_tools.close_application", side_effect=fake_close):
            await agent.handle("close chrome and whatsapp")

        self.assertEqual(calls, ["chrome", "whatsapp"])

    async def test_same_application_can_have_ordered_distinct_actions(self) -> None:
        model = FakeChatModel([tool_response(("close_application", {"application": "chrome"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[tuple[str, str]] = []

        def fake_close(application: str) -> dict[str, Any]:
            calls.append(("close", application))
            return {
                "ok": True,
                "application": application,
                "closed": 1,
                "targeted": 1,
                "remaining": [],
            }

        def fake_open(application: str) -> dict[str, Any]:
            calls.append(("open", application))
            return {"ok": True, "application": application}

        with (
            patch("langchain_tools.windows_tools.close_application", side_effect=fake_close),
            patch("langchain_tools.windows_tools.open_application", side_effect=fake_open),
        ):
            await agent.handle("close chrome, then open chrome")

        self.assertEqual(calls, [("close", "chrome"), ("open", "chrome")])

    async def test_window_state_question_is_checked_instead_of_guessed(self) -> None:
        model = FakeChatModel([AIMessage(content="Yes, WhatsApp is still open.")])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with patch(
            "langchain_tools.windows_tools.list_application_windows",
            return_value={"ok": True, "application": "whatsapp", "count": 0, "windows": []},
        ) as list_windows:
            result = await agent.handle("whatsapp is still open?")

        list_windows.assert_called_once_with("whatsapp")
        self.assertEqual(result.content, "No visible WhatsApp windows are open.")

    async def test_opens_whatsapp_and_chrome_in_order(self) -> None:
        model = FakeChatModel(
            [
                tool_response(
                    ("open_application", {"application": "whatsapp"}),
                    ("open_application", {"application": "chrome"}),
                )
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[str] = []

        def fake_open(application: str) -> dict[str, Any]:
            calls.append(application)
            return {"ok": True, "application": application, "message": f"{application} opened."}

        with patch("langchain_tools.windows_tools.open_application", side_effect=fake_open):
            result = await agent.handle("hey, can you open up whatsapp and chrome for me")

        self.assertEqual(calls, ["whatsapp", "chrome"])
        self.assertIn("WhatsApp", result.content)
        self.assertIn("Chrome", result.content)

    async def test_opens_chrome_and_notepad_in_order(self) -> None:
        model = FakeChatModel(
            [
                tool_response(
                    ("open_application", {"application": "chrome"}),
                    ("open_application", {"application": "notepad"}),
                )
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[str] = []

        def fake_open(application: str) -> dict[str, Any]:
            calls.append(application)
            return {"ok": True, "application": application}

        with patch("langchain_tools.windows_tools.open_application", side_effect=fake_open):
            await agent.handle("open chrome and note pad")

        self.assertEqual(calls, ["chrome", "notepad"])

    async def test_plain_notion_request_opens_desktop_app_not_website(self) -> None:
        model = FakeChatModel([tool_response(("open_url", {"url": "https://notion.so"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with (
            patch(
                "langchain_tools.windows_tools.open_application",
                return_value={"ok": True, "application": "notion"},
            ) as open_application,
            patch("langchain_tools.windows_tools.open_url") as open_url,
        ):
            result = await agent.handle("open notion")

        open_application.assert_called_once_with("notion")
        open_url.assert_not_called()
        self.assertEqual(result.content, "I opened Notion.")

    async def test_open_chrome_does_not_authorize_an_invented_url(self) -> None:
        model = FakeChatModel([tool_response(("open_url", {"url": "https://google.com"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with (
            patch(
                "langchain_tools.windows_tools.open_application",
                return_value={"ok": True, "application": "chrome"},
            ) as open_application,
            patch("langchain_tools.windows_tools.open_url") as open_url,
        ):
            await agent.handle("open chrome")

        open_application.assert_called_once_with("chrome")
        open_url.assert_not_called()

    async def test_compound_desktop_request_does_not_authorize_a_website(self) -> None:
        model = FakeChatModel(
            [
                tool_response(
                    ("open_url", {"url": "https://notion.so"}),
                    ("open_application", {"application": "chrome"}),
                )
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        opened: list[str] = []

        def fake_open(application: str) -> dict[str, Any]:
            opened.append(application)
            return {"ok": True, "application": application}

        with (
            patch("langchain_tools.windows_tools.open_application", side_effect=fake_open),
            patch("langchain_tools.windows_tools.open_url") as open_url,
        ):
            await agent.handle("open Notion and Chrome")

        self.assertEqual(opened, ["notion", "chrome"])
        open_url.assert_not_called()

    async def test_explicit_search_on_google_authorizes_its_url(self) -> None:
        search_url = "https://www.google.com/search?q=LangChain"
        model = FakeChatModel([AIMessage(content='I searched for "LangChain" on Google.')])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with patch(
            "langchain_tools.windows_tools.open_url",
            return_value={"ok": True, "url": search_url},
        ) as open_url:
            result = await agent.handle("search for LangChain on Google")

        open_url.assert_called_once_with(search_url)
        self.assertEqual(result.content, 'I searched Google for "LangChain" and opened the results.')

    async def test_open_subject_on_google_is_a_required_search_action(self) -> None:
        search_url = "https://www.google.com/search?q=LangChain"
        model = FakeChatModel([AIMessage(content="I opened the search results page.")])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with patch(
            "langchain_tools.windows_tools.open_url",
            return_value={"ok": True, "url": search_url},
        ) as open_url:
            await agent.handle("open LangChain on Google")

        open_url.assert_called_once_with(search_url)

    async def test_explicit_on_chrome_navigation_authorizes_its_url(self) -> None:
        model = FakeChatModel([tool_response(("open_application", {"application": "chrome"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[tuple[str, str]] = []

        def fake_open(application: str) -> dict[str, Any]:
            calls.append(("application", application))
            return {"ok": True, "application": application}

        def fake_url(url: str) -> dict[str, Any]:
            calls.append(("url", url))
            return {"ok": True, "url": url}

        with (
            patch("langchain_tools.windows_tools.open_application", side_effect=fake_open),
            patch("langchain_tools.windows_tools.open_url", side_effect=fake_url),
        ):
            await agent.handle("open Notion on Chrome")

        self.assertEqual(calls, [("application", "chrome"), ("url", "https://notion.so")])

    async def test_on_laptop_request_cannot_become_active_window_query(self) -> None:
        model = FakeChatModel([tool_response(("active_window", {}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with (
            patch(
                "langchain_tools.windows_tools.open_application",
                return_value={"ok": True, "application": "notion"},
            ) as open_application,
            patch("langchain_tools.windows_tools.active_window") as active_window,
        ):
            await agent.handle("open notion on laptop")

        open_application.assert_called_once_with("notion")
        active_window.assert_not_called()

    async def test_all_new_desktop_apps_are_preserved_in_compound_request(self) -> None:
        model = FakeChatModel([tool_response(("open_application", {"application": "notion"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[str] = []

        def fake_open(application: str) -> dict[str, Any]:
            calls.append(application)
            return {"ok": True, "application": application}

        with patch("langchain_tools.windows_tools.open_application", side_effect=fake_open):
            await agent.handle(
                "open Notion, VS Code, GitHub Desktop, Outlook, MongoDB Compass, and Termius"
            )

        self.assertEqual(
            calls,
            ["notion", "vscode", "github_desktop", "outlook", "mongodb_compass", "termius"],
        )

    async def test_close_failure_does_not_skip_later_application(self) -> None:
        model = FakeChatModel(
            [
                tool_response(
                    ("close_application", {"application": "chrome"}),
                    ("close_application", {"application": "notepad"}),
                )
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[str] = []

        def fake_close(application: str) -> dict[str, Any]:
            calls.append(application)
            if application == "chrome":
                raise RuntimeError("Chrome refused to close")
            return {
                "ok": True,
                "application": application,
                "closed": 1,
                "targeted": 1,
                "remaining": [],
            }

        with patch("langchain_tools.windows_tools.close_application", side_effect=fake_close):
            result = await agent.handle("close chrome and notepad")

        self.assertEqual(calls, ["chrome", "notepad"])
        self.assertIn("Chrome refused to close", result.content)
        self.assertIn("closed 1 visible Notepad window", result.content)

    async def test_youtube_home_is_navigation_not_search(self) -> None:
        model = FakeChatModel(
            [
                tool_response(
                    ("open_application", {"application": "chrome"}),
                    ("open_url", {"url": "https://youtube.com"}),
                )
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[tuple[str, str]] = []

        def fake_open(application: str) -> dict[str, Any]:
            calls.append(("application", application))
            return {"ok": True, "application": application}

        def fake_url(url: str) -> dict[str, Any]:
            calls.append(("url", url))
            return {"ok": True, "url": url}

        with (
            patch("langchain_tools.windows_tools.open_application", side_effect=fake_open),
            patch("langchain_tools.windows_tools.open_url", side_effect=fake_url),
            patch("langchain_tools.windows_tools.youtube_search_open") as youtube_search,
        ):
            result = await agent.handle("open youtube on chrome")

        self.assertEqual(calls, [("application", "chrome"), ("url", "https://youtube.com")])
        youtube_search.assert_not_called()
        self.assertIn("https://youtube.com", result.content)

    async def test_specific_youtube_content_uses_search_tool(self) -> None:
        model = FakeChatModel(
            [tool_response(("youtube_search_open", {"query": "Makemore playlist Andrej Karpathy"}))]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with patch(
            "langchain_tools.windows_tools.youtube_search_open",
            return_value={
                "ok": True,
                "mode": "best_result",
                "title": "Makemore",
                "url": "https://www.youtube.com/playlist?list=test",
            },
        ) as youtube_search:
            result = await agent.handle("open the Makemore playlist by Andrej Karpathy on YouTube")

        youtube_search.assert_called_once_with("Makemore playlist Andrej Karpathy")
        self.assertIn("Makemore", result.content)

    async def test_structured_fallback_executes_all_actions(self) -> None:
        plan = {
            "type": "tools",
            "content": "",
            "actions": [
                {"tool": "open_application", "arguments": {"application": "whatsapp"}},
                {"tool": "open_application", "arguments": {"application": "chrome"}},
            ],
        }
        model = FakeChatModel([AIMessage(content=json.dumps(plan))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=False)
        calls: list[str] = []

        def fake_open(application: str) -> dict[str, Any]:
            calls.append(application)
            return {"ok": True, "application": application}

        with patch("langchain_tools.windows_tools.open_application", side_effect=fake_open):
            await agent.handle("open whatsapp and chrome")

        self.assertEqual(calls, ["whatsapp", "chrome"])

    async def test_structured_fallback_repairs_app_keys_and_missing_url(self) -> None:
        plan = {
            "type": "tools",
            "content": "",
            "actions": [
                {"tool": "open_application", "arguments": {"name": "chrome"}},
                {"tool": "open_application", "arguments": {"name": "whatsapp"}},
            ],
        }
        model = FakeChatModel([AIMessage(content=json.dumps(plan))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=False)
        calls: list[tuple[str, str]] = []

        def fake_open(application: str) -> dict[str, Any]:
            calls.append(("application", application))
            return {"ok": True, "application": application}

        def fake_url(url: str) -> dict[str, Any]:
            calls.append(("url", url))
            return {"ok": True, "url": url}

        with (
            patch("langchain_tools.windows_tools.open_application", side_effect=fake_open),
            patch("langchain_tools.windows_tools.open_url", side_effect=fake_url),
        ):
            await agent.handle("open chatgpt.com on chrome, and whatsapp on laptop")

        self.assertEqual(
            calls,
            [
                ("application", "chrome"),
                ("url", "https://chatgpt.com"),
                ("application", "whatsapp"),
            ],
        )

    async def test_confirmation_stores_then_executes_complete_plan(self) -> None:
        plan = {
            "type": "confirm",
            "content": "Did you mean open Notepad? Reply yes or no.",
            "actions": [{"tool": "open_application", "arguments": {"application": "notepad"}}],
        }
        model = FakeChatModel([AIMessage(content=json.dumps(plan))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=False)

        with patch(
            "langchain_tools.windows_tools.open_application",
            return_value={"ok": True, "application": "notepad"},
        ) as open_application:
            first = await agent.handle("open note thing")
            open_application.assert_not_called()
            second = await agent.handle("yes")

        self.assertIn("Did you mean", first.content)
        open_application.assert_called_once_with("notepad")
        self.assertIn("Notepad", second.content)

    async def test_directory_delete_requires_host_confirmation(self) -> None:
        model = FakeChatModel([tool_response(("delete_directory", {"path": "old-project"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with patch(
            "langchain_tools.windows_tools.delete_directory",
            return_value={
                "ok": True,
                "path": "F:\\PortableAI\\workspace\\old-project",
                "relative_path": "old-project",
                "deleted_directories": 2,
                "deleted_files": 3,
            },
        ) as delete_directory:
            first = await agent.handle("delete the old-project directory")
            delete_directory.assert_not_called()
            second = await agent.handle("yes")

        self.assertIn("permanently delete old-project", first.content)
        delete_directory.assert_called_once_with("old-project")
        self.assertIn("permanently deleted old-project", second.content)

    async def test_destructive_plan_executes_nothing_before_confirmation(self) -> None:
        model = FakeChatModel(
            [
                tool_response(
                    ("create_directory", {"path": "archive"}),
                    ("delete_directory", {"path": "old-project"}),
                )
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[tuple[str, str]] = []

        def fake_create(path: str) -> dict[str, Any]:
            calls.append(("create", path))
            return {"ok": True, "relative_path": path, "created": True}

        def fake_delete(path: str) -> dict[str, Any]:
            calls.append(("delete", path))
            return {
                "ok": True,
                "relative_path": path,
                "deleted_directories": 1,
                "deleted_files": 0,
            }

        with (
            patch("langchain_tools.windows_tools.create_directory", side_effect=fake_create),
            patch("langchain_tools.windows_tools.delete_directory", side_effect=fake_delete),
        ):
            first = await agent.handle("create archive and delete old-project")
            self.assertEqual(calls, [])
            self.assertIn("complete 2-action plan", first.content)
            await agent.handle("yes")

        self.assertEqual(calls, [("create", "archive"), ("delete", "old-project")])

    async def test_read_request_cannot_accidentally_create_directory(self) -> None:
        model = FakeChatModel(
            [
                tool_response(
                    (
                        "list_directory",
                        {
                            "path": "E:\\GPT-mini",
                            "recursive": False,
                            "include": "all",
                            "output_format": "compact",
                        },
                    )
                ),
                tool_response(("create_directory", {"path": "Makemore Series"})),
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[tuple[str, bool, str, str]] = []

        def fake_list(path: str, recursive: bool, include: str, output_format: str) -> dict[str, Any]:
            calls.append((path, recursive, include, output_format))
            if path == "E:\\GPT-mini":
                entries = [
                    {"name": ".git", "path": ".git", "type": "directory", "size_bytes": 0},
                    {
                        "name": "Makemoer Series",
                        "path": "Makemoer Series",
                        "type": "directory",
                        "size_bytes": 0,
                    },
                ]
            else:
                entries = [{"name": "lesson.txt", "path": "lesson.txt", "type": "file", "size_bytes": 10}]
            return {
                "ok": True,
                "path": path,
                "relative_path": path,
                "recursive": recursive,
                "include": include,
                "output_format": output_format,
                "entries": entries,
                "truncated": False,
            }

        with (
            patch("langchain_tools.resolve_inspection_directory", side_effect=lambda path: Path(path)),
            patch("langchain_tools.windows_tools.list_directory", side_effect=fake_list),
            patch("langchain_tools.windows_tools.create_directory") as create_directory,
        ):
            await agent.handle("what are the contents of the GPT-mini folder?")
            result = await agent.handle("list contents of Makemore Series")

        create_directory.assert_not_called()
        self.assertEqual(
            calls[-1],
            ("E:\\GPT-mini\\Makemoer Series", False, "all", "compact"),
        )
        self.assertIn("lesson.txt", result.content)

    async def test_tree_followup_preserves_previous_listing_scope(self) -> None:
        model = FakeChatModel(
            [
                tool_response(
                    (
                        "list_directory",
                        {
                            "path": "E:\\",
                            "recursive": False,
                            "include": "directories",
                            "output_format": "compact",
                        },
                    )
                ),
                tool_response(
                    (
                        "list_directory",
                        {
                            "path": "E:\\",
                            "recursive": True,
                            "include": "all",
                            "output_format": "compact",
                        },
                    )
                ),
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[tuple[str, bool, str, str]] = []

        def fake_list(path: str, recursive: bool, include: str, output_format: str) -> dict[str, Any]:
            calls.append((path, recursive, include, output_format))
            return {
                "ok": True,
                "path": path,
                "relative_path": path,
                "recursive": recursive,
                "include": include,
                "output_format": output_format,
                "entries": [
                    {"name": "GPT-mini", "path": "GPT-mini", "type": "directory", "size_bytes": 0},
                    {"name": "Projects", "path": "Projects", "type": "directory", "size_bytes": 0},
                ],
                "truncated": False,
            }

        with (
            patch("langchain_tools.resolve_inspection_directory", side_effect=lambda path: Path(path)),
            patch("langchain_tools.windows_tools.list_directory", side_effect=fake_list),
        ):
            await agent.handle("what are the folders in E drive?")
            result = await agent.handle("please list them in a tree format")

        self.assertEqual(calls[-1], ("E:\\", False, "directories", "tree"))
        self.assertEqual(result.content, "E:\\\n|-- GPT-mini/\n|-- Projects/")

    async def test_explicit_create_request_still_creates_directory(self) -> None:
        model = FakeChatModel([tool_response(("create_directory", {"path": "new-project"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with patch(
            "langchain_tools.windows_tools.create_directory",
            return_value={"ok": True, "relative_path": "new-project", "created": True},
        ) as create_directory:
            result = await agent.handle("create a directory named new-project")

        create_directory.assert_called_once_with("new-project")
        self.assertIn("created directory new-project", result.content)

    async def test_nested_create_request_restores_every_missing_subdirectory(self) -> None:
        model = FakeChatModel([tool_response(("create_directory", {"path": "test2"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        created: list[str] = []

        def fake_create(path: str) -> dict[str, Any]:
            created.append(path)
            return {"ok": True, "relative_path": path, "created": True}

        with patch("langchain_tools.windows_tools.create_directory", side_effect=fake_create):
            result = await agent.handle(
                "create a directory test2 in workspace and inside it 2 sub directories, dir1 and dir2"
            )

        self.assertEqual(created, ["test2", "test2\\dir1", "test2\\dir2"])
        self.assertIn("created directory test2\\dir1", result.content)
        self.assertIn("created directory test2\\dir2", result.content)

    async def test_structured_fallback_keeps_one_leading_system_message(self) -> None:
        first_plan = {"type": "answer", "content": "First answer.", "actions": []}
        second_plan = {"type": "answer", "content": "Follow-up answer.", "actions": []}
        model = FakeChatModel(
            [AIMessage(content=json.dumps(first_plan)), AIMessage(content=json.dumps(second_plan))]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=False)

        await agent.handle("first question")
        await agent.handle("follow-up question")

        followup_messages = model.seen_messages[-1]
        system_messages = [message for message in followup_messages if isinstance(message, SystemMessage)]
        self.assertEqual(len(system_messages), 1)
        self.assertIs(followup_messages[0], system_messages[0])
        self.assertIn("Recent conversation:", str(system_messages[0].content))

    async def test_invalid_native_call_is_retried_before_execution(self) -> None:
        model = FakeChatModel(
            [
                tool_response(("open_application", {"application": "powershell"})),
                tool_response(("open_application", {"application": "chrome"})),
            ]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with patch(
            "langchain_tools.windows_tools.open_application",
            return_value={"ok": True, "application": "chrome"},
        ) as open_application:
            await agent.handle("open chrome")

        open_application.assert_called_once_with("chrome")

    async def test_natural_answer_does_not_call_a_tool(self) -> None:
        model = FakeChatModel([AIMessage(content="I can help with local Windows tasks and general questions.")])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        streamed: list[str] = []

        result = await agent.handle("what can you do?", on_token=streamed.append)

        self.assertEqual(streamed, [result.content])
        self.assertIn("Windows tasks", result.content)

    def test_application_schema_blocks_non_allowlisted_target(self) -> None:
        with self.assertRaises(Exception):
            TOOL_REGISTRY["open_application"].invoke({"application": "powershell"})

    def test_application_schema_accepts_new_desktop_apps(self) -> None:
        schema = TOOL_REGISTRY["open_application"].get_input_schema()
        for application in (
            "notion",
            "vscode",
            "github_desktop",
            "outlook",
            "mongodb_compass",
            "termius",
        ):
            with self.subTest(application=application):
                validated = schema.model_validate({"application": application})
                self.assertEqual(validated.application, application)

    def test_directory_schema_blocks_path_outside_workspace(self) -> None:
        with self.assertRaises(Exception):
            TOOL_REGISTRY["create_directory"].invoke({"path": "C:\\Windows"})

    def test_directory_schema_blocks_workspace_root_deletion(self) -> None:
        with self.assertRaises(Exception):
            TOOL_REGISTRY["delete_directory"].invoke({"path": "."})


    async def test_open_notepad_with_unquoted_text_repairs_open_only_plan(self) -> None:
        model = FakeChatModel([tool_response(("open_application", {"application": "notepad"}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        requested = "255 spent on the swiggy order in the stryki"

        with (
            patch(
                "langchain_tools.windows_tools.notepad_open",
                return_value={
                    "ok": True,
                    "path": "F:\\PortableAI\\cache\\notepad\\portable_ai_current_note.txt",
                    "message": "Managed Notepad note opened.",
                },
            ) as notepad_open,
            patch("langchain_tools.windows_tools.open_application") as open_application,
        ):
            result = await agent.handle(f"open notepad and add the text in that, {requested}")

        notepad_open.assert_called_once_with(requested)
        open_application.assert_not_called()
        self.assertIn("portable_ai_current_note.txt", result.content)

    async def test_notepad_save_is_executed_before_close(self) -> None:
        model = FakeChatModel(
            [tool_response(("notepad_close", {}), ("notepad_save", {}))]
        )
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        calls: list[str] = []

        def fake_save() -> dict[str, Any]:
            calls.append("save")
            return {"ok": True, "path": "note.txt", "message": "Saved."}

        def fake_close() -> dict[str, Any]:
            calls.append("close")
            return {
                "ok": True,
                "application": "notepad",
                "closed": 1,
                "targeted": 1,
                "remaining": [],
            }

        with (
            patch("langchain_tools.windows_tools.notepad_save", side_effect=fake_save),
            patch("langchain_tools.windows_tools.notepad_close", side_effect=fake_close),
        ):
            await agent.handle("save the file and close notepad")

        self.assertEqual(calls, ["save", "close"])

    async def test_explicit_quoted_path_rename_is_repaired_when_model_refuses(self) -> None:
        model = FakeChatModel([AIMessage(content="I cannot rename files.")])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)
        source = "F:\\PortableAI\\cache\\notepad\\portable_ai_current_note.txt"
        destination = "F:\\PortableAI\\cache\\notepad\\nagpur-expenses.txt"

        with (
            patch("langchain_tools.resolve_managed_file", side_effect=lambda path: Path(path)),
            patch(
                "langchain_tools.windows_tools.rename_file",
                return_value={
                    "ok": True,
                    "source": source,
                    "path": destination,
                    "renamed": True,
                },
            ) as rename_file,
        ):
            result = await agent.handle(f'"{source}" rename this as the nagpur-expenses')

        rename_file.assert_called_once_with(source, "nagpur-expenses", False)
        self.assertIn("nagpur-expenses.txt", result.content)

    async def test_delete_file_requires_confirmation(self) -> None:
        source = "F:\\PortableAI\\workspace\\old.txt"
        model = FakeChatModel([tool_response(("delete_file", {"path": source}))])
        agent = PortableAgent(chat_model=model, tool_calling_supported=True)

        with (
            patch("langchain_tools.resolve_managed_file", side_effect=lambda path: Path(path)),
            patch(
                "langchain_tools.windows_tools.delete_file",
                return_value={"ok": True, "path": source, "deleted_bytes": 3},
            ) as delete_file,
        ):
            first = await agent.handle(f'delete the file "{source}"')
            delete_file.assert_not_called()
            second = await agent.handle("yes")

        self.assertIn("permanently delete", first.content)
        delete_file.assert_called_once_with(source)
        self.assertIn("permanently deleted", second.content)


class PortableAgentPowerFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.plan_store = PlanStore(Path(self.temporary_directory.name) / "plans.db")

    def make_agent(self, responses: list[AIMessage] | None = None) -> PortableAgent:
        return PortableAgent(
            chat_model=FakeChatModel(responses or []),
            tool_calling_supported=True,
            plan_store=self.plan_store,
        )

    async def test_goodnight_waits_for_confirmation_then_closes_and_shuts_down(self) -> None:
        agent = self.make_agent()
        calls: list[str] = []

        def close_all() -> dict[str, Any]:
            calls.append("close_all")
            return {"ok": True, "targeted": 2, "closed": 2, "remaining": []}

        def shutdown() -> dict[str, Any]:
            calls.append("shutdown")
            return {"ok": True, "dispatched": True, "delay_seconds": 0}

        with (
            patch("langchain_tools.windows_tools.close_all_visible_applications", side_effect=close_all),
            patch("langchain_tools.windows_tools.shutdown_computer", side_effect=shutdown),
        ):
            first = await agent.handle("Good Night")
            self.assertEqual(calls, [])
            self.assertIn("shut down", first.content)
            plan_id = agent.pending_confirmation.plan_id
            second = await agent.handle("yes")

        self.assertEqual(calls, ["close_all", "shutdown"])
        self.assertTrue(agent.exit_requested)
        self.assertIn("accepted the shutdown request", second.content)
        stored = self.plan_store.get_plan(plan_id)
        self.assertEqual(stored.status, "dispatched")
        self.assertEqual([action.status for action in stored.actions], ["succeeded", "dispatched"])

    async def test_tata_uses_the_same_confirmed_shutdown_routine(self) -> None:
        agent = self.make_agent()

        with (
            patch("langchain_tools.windows_tools.close_all_visible_applications") as close_all,
            patch("langchain_tools.windows_tools.shutdown_computer") as shutdown,
        ):
            result = await agent.handle("Tata!")

        self.assertIn("shut down", result.content)
        close_all.assert_not_called()
        shutdown.assert_not_called()

    async def test_deep_sleep_is_blocked_when_an_application_remains_open(self) -> None:
        agent = self.make_agent()
        close_result = {
            "ok": False,
            "targeted": 2,
            "closed": 1,
            "remaining": [{"title": "Unsaved - Notepad", "process": "notepad.exe", "pid": 10}],
            "message": "Some applications are still open.",
        }

        with (
            patch(
                "langchain_tools.windows_tools.close_all_visible_applications",
                return_value=close_result,
            ),
            patch("langchain_tools.windows_tools.sleep_computer") as sleep_computer,
        ):
            await agent.handle("deep sleep")
            plan_id = agent.pending_confirmation.plan_id
            result = await agent.handle("yes")

        sleep_computer.assert_not_called()
        self.assertIn("Unsaved - Notepad", result.content)
        self.assertIn("Skipped", result.content)
        stored = self.plan_store.get_plan(plan_id)
        self.assertEqual(stored.status, "failed")
        self.assertEqual([action.status for action in stored.actions], ["failed", "blocked"])

    async def test_exact_sleep_runs_immediately_without_closing_applications(self) -> None:
        agent = self.make_agent()

        with (
            patch("langchain_tools.windows_tools.close_all_visible_applications") as close_all,
            patch(
                "langchain_tools.windows_tools.sleep_computer",
                return_value={"ok": True, "state": "resumed"},
            ) as sleep_computer,
        ):
            result = await agent.handle("normal sleep")

        close_all.assert_not_called()
        sleep_computer.assert_called_once_with()
        self.assertEqual(result.content, "The computer resumed from sleep.")

    async def test_exact_hide_locks_immediately(self) -> None:
        agent = self.make_agent()

        with patch(
            "langchain_tools.windows_tools.lock_computer",
            return_value={"ok": True, "state": "locked"},
        ) as lock_computer:
            result = await agent.handle("hide")

        lock_computer.assert_called_once_with()
        self.assertEqual(result.content, "I locked Windows.")

    async def test_pending_confirmation_is_not_lost_on_unrelated_reply(self) -> None:
        agent = self.make_agent()

        await agent.handle("goodnight")
        plan_id = agent.pending_confirmation.plan_id
        result = await agent.handle("maybe later")

        self.assertIn("Reply yes", result.content)
        self.assertEqual(agent.pending_confirmation.plan_id, plan_id)
        self.assertEqual(self.plan_store.get_plan(plan_id).status, "pending_confirmation")

    async def test_cancel_discards_the_complete_pending_plan(self) -> None:
        agent = self.make_agent()

        await agent.handle("deep sleep")
        plan_id = agent.pending_confirmation.plan_id
        result = await agent.handle("/cancel")

        self.assertIn("cancelled", result.content)
        self.assertIsNone(agent.pending_confirmation)
        stored = self.plan_store.get_plan(plan_id)
        self.assertEqual(stored.status, "cancelled")
        self.assertEqual([action.status for action in stored.actions], ["cancelled", "cancelled"])

    async def test_natural_shutdown_is_repaired_to_close_all_then_shutdown(self) -> None:
        agent = self.make_agent([AIMessage(content="I will shut it down.")])
        calls: list[str] = []

        with (
            patch(
                "langchain_tools.windows_tools.close_all_visible_applications",
                side_effect=lambda: calls.append("close_all")
                or {"ok": True, "targeted": 0, "closed": 0, "remaining": []},
            ),
            patch(
                "langchain_tools.windows_tools.shutdown_computer",
                side_effect=lambda: calls.append("shutdown")
                or {"ok": True, "dispatched": True, "delay_seconds": 0},
            ),
        ):
            await agent.handle("please shutdown my laptop")
            self.assertEqual(calls, [])
            await agent.handle("yes")

        self.assertEqual(calls, ["close_all", "shutdown"])

    async def test_exact_shutdown_does_not_depend_on_model_planning(self) -> None:
        agent = self.make_agent()

        with (
            patch("langchain_tools.windows_tools.close_all_visible_applications") as close_all,
            patch("langchain_tools.windows_tools.shutdown_computer") as shutdown,
        ):
            result = await agent.handle("shutdown")

        self.assertIn("shut down", result.content)
        close_all.assert_not_called()
        shutdown.assert_not_called()

    async def test_negated_sleep_does_not_authorize_power_tool(self) -> None:
        agent = self.make_agent(
            [
                tool_response(("sleep_computer", {})),
                AIMessage(content="Okay, I will leave the computer awake."),
            ]
        )

        with patch("langchain_tools.windows_tools.sleep_computer") as sleep_computer:
            result = await agent.handle("do not sleep the computer")

        sleep_computer.assert_not_called()
        self.assertIn("awake", result.content)

    async def test_power_question_does_not_authorize_power_tool(self) -> None:
        agent = self.make_agent(
            [
                tool_response(("shutdown_computer", {})),
                AIMessage(content="It depends on whether you have saved your work."),
            ]
        )

        with patch("langchain_tools.windows_tools.shutdown_computer") as shutdown_computer:
            result = await agent.handle("should I shutdown my laptop?")

        shutdown_computer.assert_not_called()
        self.assertIn("saved your work", result.content)


if __name__ == "__main__":
    unittest.main()

