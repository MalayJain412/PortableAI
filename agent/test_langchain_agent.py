from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator
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


if __name__ == "__main__":
    unittest.main()

