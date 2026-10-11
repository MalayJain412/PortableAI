from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk

from agent import PortableAgent
from langchain_tools import TOOL_REGISTRY


class FakeChatModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools: list[Any]) -> "FakeChatModel":
        self.bound_tool_names = [item.name for item in tools]
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
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


if __name__ == "__main__":
    unittest.main()

