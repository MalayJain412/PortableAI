from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from config import LLAMAFILE_BASE_URL, LLAMAFILE_HOST, LLAMAFILE_PORT, LOGS_DIR, MODEL_EXE, MODEL_ID, PROJECT_ROOT


logging.getLogger("httpx").setLevel(logging.WARNING)

Message = dict[str, str]
TokenCallback = Callable[[str], None]


@dataclass(frozen=True)
class GenerationMetrics:
    ttft_seconds: float | None
    total_seconds: float
    output_characters: int
    approx_tokens: int

    @property
    def approx_tokens_per_second(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return self.approx_tokens / self.total_seconds


@dataclass(frozen=True)
class LLMResult:
    content: str
    metrics: GenerationMetrics


class LlamafileClient:
    def __init__(self, base_url: str = LLAMAFILE_BASE_URL, model: str = MODEL_ID) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json().get("status") == "ok"
        except Exception:
            return False

    def ensure_ready(self, *, restart: bool = True, timeout_seconds: float = 90.0) -> None:
        if self.health():
            return
        if restart:
            self.start_server()

        deadline = time.perf_counter() + timeout_seconds
        while time.perf_counter() < deadline:
            if self.health():
                return
            time.sleep(0.5)
        raise RuntimeError(f"Model server is not ready at {self.base_url}.")

    def start_server(self) -> None:
        if not MODEL_EXE.exists():
            raise RuntimeError(f"Model executable not found: {MODEL_EXE}")

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stdout_path = LOGS_DIR / "llamafile.out.log"
        stderr_path = LOGS_DIR / "llamafile.err.log"
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.Popen(
                [
                    str(MODEL_EXE),
                    "--server",
                    "--host",
                    LLAMAFILE_HOST,
                    "--port",
                    str(LLAMAFILE_PORT),
                    "--nologo",
                    "--no-webui",
                ],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                creationflags=creationflags,
            )
        finally:
            stdout.close()
            stderr.close()

    def models(self) -> list[str]:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            data = response.json()
        return [item["id"] for item in data.get("data", []) if "id" in item]

    def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        stream: bool = True,
        on_token: TokenCallback | None = None,
    ) -> LLMResult:
        self.ensure_ready()
        try:
            if stream:
                return self._stream_chat(messages, max_tokens=max_tokens, temperature=temperature, on_token=on_token)
            return self._chat_once(messages, max_tokens=max_tokens, temperature=temperature)
        except httpx.TransportError:
            self.ensure_ready(restart=True)
            if stream:
                return self._stream_chat(messages, max_tokens=max_tokens, temperature=temperature, on_token=on_token)
            return self._chat_once(messages, max_tokens=max_tokens, temperature=temperature)

    def _chat_once(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        start = time.perf_counter()
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        total = time.perf_counter() - start
        content = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage") or {}
        approx_tokens = int(usage.get("completion_tokens") or estimate_tokens(content))
        return LLMResult(
            content=content,
            metrics=GenerationMetrics(
                ttft_seconds=total,
                total_seconds=total,
                output_characters=len(content),
                approx_tokens=approx_tokens,
            ),
        )

    def _stream_chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        temperature: float,
        on_token: TokenCallback | None,
    ) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        start = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []

        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    try:
                        event: dict[str, Any] = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = event.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if not delta:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(delta)
                    if on_token is not None:
                        on_token(delta)

        total = time.perf_counter() - start
        content = "".join(chunks)
        return LLMResult(
            content=content,
            metrics=GenerationMetrics(
                ttft_seconds=None if first_token_at is None else first_token_at - start,
                total_seconds=total,
                output_characters=len(content),
                approx_tokens=estimate_tokens(content),
            ),
        )


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / 4))
