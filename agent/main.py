from __future__ import annotations

import asyncio
import sys
import threading
import time

from agent import PortableAgent
from llm import GenerationMetrics, LlamafileClient
from wake_voice import AlexaVoiceInput, VoiceUnavailableError, voice_runtime_status
from windows_tools import initialize_sentry_window


HELP_TEXT = """Commands:
  /help     Show this help
  /tools    List available Windows tools
  /voice    Switch to Alexa wake-word input
  /text     Switch to keyboard input
  /listen   Wait for Alexa once, then remain in the current mode
  /voice-status  Show Alexa voice runtime status
  /clear    Clear chat memory
  /cancel   Cancel a pending safety-sensitive plan
  /exit     Quit
"""


class ResponseLoader:
    def __init__(self, label: str = "Thinking") -> None:
        self.label = label
        self._stop = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._clear()

    def _run(self) -> None:
        frames = "|/-\\"
        index = 0
        while not self._stop.is_set():
            frame = frames[index % len(frames)]
            sys.stdout.write(f"\rAI: {self.label} {frame}")
            sys.stdout.flush()
            index += 1
            time.sleep(0.12)

    def _clear(self) -> None:
        sys.stdout.write("\r" + " " * 80 + "\rAI: ")
        sys.stdout.flush()


# def print_banner() -> None:
#     print("=" * 56)
#     print(" Portable AI")
#     print(" Local Qwen3.5 0.8B")
#     print(" MCP Windows Agent")
#     print(" Developed by: Malay Jain")
#     print("=" * 56)

def print_banner() -> None:
    banner = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ███████╗███████╗███╗   ██╗████████╗██████╗ ██╗   ██╗           ║
║   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔══██╗╚██╗ ██╔╝           ║
║   ███████╗█████╗  ██╔██╗ ██║   ██║   ██████╔╝ ╚████╔╝            ║
║   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗  ╚██╔╝             ║
║   ███████║███████╗██║ ╚████║   ██║   ██║  ██║   ██║              ║
║   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝              ║
║                                                                  ║
║                   LOCAL INTELLIGENCE SYSTEM                      ║
║              YOUR AI. YOUR DEVICE. YOUR CONTROL.                 ║
║                                                                  ║
║   MODEL      : Qwen3.5 0.8B                                      ║
║   ENGINE     : llama.cpp                                         ║
║   AGENT      : Windows MCP Agent                                 ║
║   MODE       : LOCAL  |  PORTABLE  |  PRIVATE                    ║
║                                                                  ║
║                    DEVELOPED BY MALAY JAIN                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    try:
        print(banner)
    except UnicodeEncodeError:
        print("=" * 68)
        print(" SENTRY - LOCAL INTELLIGENCE SYSTEM")
        print(" Local Qwen3.5 0.8B | Windows Agent | Portable | Private")
        print(" Developed by Malay Jain")
        print("=" * 68)


def print_metrics(metrics: GenerationMetrics) -> None:
    ttft = "n/a" if metrics.ttft_seconds is None else f"{metrics.ttft_seconds:.2f}s"
    print(
        f"\n[metrics] TTFT: {ttft} | Total: {metrics.total_seconds:.2f}s | "
        f"Approx: {metrics.approx_tokens_per_second:.1f} tok/s"
    )


async def run_chat() -> int:
    terminal = initialize_sentry_window()
    if not terminal["ok"]:
        print("Warning: Sentry could not protect this terminal, so close-all routines will be blocked.")
    else:
        print(
            "Terminal guard: "
            f"{terminal.get('host_process') or 'unknown'} | "
            f"{terminal.get('host_title') or terminal.get('title')}"
        )
    print_banner()
    llm = LlamafileClient()
    if not llm.health():
        print("Model server is not ready at http://127.0.0.1:8080.")
        print("Start the llamafile server first or use START.bat.")
        return 1

    models = llm.models()
    if models:
        print(f"Model: {models[0]}")
    print(HELP_TEXT)

    agent = PortableAgent(llm)
    await agent.initialize()
    input_mode = "text"
    voice_input: AlexaVoiceInput | None = None

    def close_voice_input() -> None:
        nonlocal voice_input
        if voice_input is not None:
            voice_input.close()
            voice_input = None

    while True:
        if input_mode == "voice":
            try:
                if voice_input is None:
                    voice_input = AlexaVoiceInput()
                user_text = voice_input.listen_once().strip()
            except KeyboardInterrupt:
                input_mode = "text"
                print("\nText mode enabled. Type /voice to listen for Alexa again.")
                continue
            except VoiceUnavailableError as exc:
                print(f"\nVoice input error: {exc}")
                close_voice_input()
                input_mode = "text"
                print("Returning to text mode.")
                continue
        else:
            try:
                user_text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                close_voice_input()
                print("\nGoodbye.")
                return 0

        if not user_text:
            continue
        command = user_text.casefold()
        if input_mode == "voice" and command in {
            "text mode",
            "switch to text mode",
            "use text mode",
            "keyboard mode",
        }:
            input_mode = "text"
            print("Text mode enabled.")
            continue
        if command in {"/exit", "exit", "quit","by","bye","ok by", "ok bye","by by"}:
            close_voice_input()
            print("Goodbye.")
            return 0
        if command == "/help":
            print(HELP_TEXT)
            continue
        if command == "/tools":
            tools = await agent.tool_names()
            print("Tools: " + ", ".join(tools))
            continue
        if command == "/voice":
            try:
                voice_input = voice_input or AlexaVoiceInput()
            except VoiceUnavailableError as exc:
                print(f"Voice input unavailable: {exc}")
                continue
            input_mode = "voice"
            print("Alexa voice mode enabled. Say 'Alexa', then speak your command.")
            print("Press Ctrl+C while waiting to return to text mode.")
            continue
        if command == "/text":
            input_mode = "text"
            print("Text mode enabled.")
            continue
        if command == "/listen":
            try:
                voice_input = voice_input or AlexaVoiceInput()
                user_text = voice_input.listen_once().strip()
            except KeyboardInterrupt:
                print("\nVoice listening cancelled.")
                continue
            except VoiceUnavailableError as exc:
                print(f"Voice input error: {exc}")
                continue
            if not user_text:
                continue
            command = user_text.casefold()
        if command == "/voice-status":
            status = voice_runtime_status()
            print(
                "Voice dependencies: "
                + ", ".join(
                    f"{name}={'ok' if installed else 'missing'}"
                    for name, installed in status["dependencies"].items()
                )
            )
            print(f"Wake word: {status['wake_word']}")
            print(f"Audio server: {'ready' if status['audio_server_ready'] else 'stopped'}")
            missing_files = [path for path, present in status["files"].items() if not present]
            print("Voice files: " + ("ready" if not missing_files else "missing"))
            for path in missing_files:
                print(f"  {path}")
            continue
        if command == "/clear":
            agent.clear_history()
            print("Chat memory cleared.")
            continue

        print("\nAI: ", end="", flush=True)
        streamed = False
        loader = ResponseLoader()
        loader.start()

        def print_token(token: str) -> None:
            nonlocal streamed
            if not streamed:
                streamed = True
                loader.stop()
            print(token, end="", flush=True)

        try:
            result = await agent.handle(user_text, on_token=print_token)
        except Exception as exc:
            loader.stop()
            print(f"\nError: {exc}")
            continue
        loader.stop()
        if result.content and not streamed:
            print(result.content, end="", flush=True)
        print_metrics(result.metrics)
        if agent.exit_requested:
            close_voice_input()
            print("Sentry is exiting so it cannot delay shutdown.")
            return 20


def main() -> None:
    try:
        raise SystemExit(asyncio.run(run_chat()))
    except KeyboardInterrupt:
        print("\nGoodbye.")
        raise SystemExit(0)


if __name__ == "__main__":
    sys.exit(main())
