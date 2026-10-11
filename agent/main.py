from __future__ import annotations

import asyncio
import sys
import threading
import time

from agent import PortableAgent
from llm import GenerationMetrics, LlamafileClient
from voice import OfflineVoiceInput, VoiceConfig, VoiceUnavailableError, available_vosk_models, normalize_voice_command, resolve_vosk_model_path, voice_dependencies_status


VOICE_ENABLED = False

HELP_TEXT = """Commands:
  /help     Show this help
  /tools    List available Windows tools
  /clear    Clear chat memory
  /exit     Quit
"""

VOICE_DISABLED_MESSAGE = "Voice input is disabled for now. Text mode is the supported input mode."


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
    voice_engine = "auto"
    voice_confirm = True
    voice_input: OfflineVoiceInput | None = None
    while True:
        try:
            if input_mode == "voice":
                print("\nYou [voice]: ", end="", flush=True)
                if voice_input is None:
                    voice_input = OfflineVoiceInput(VoiceConfig(engine=voice_engine))
                try:
                    user_text = voice_input.listen_once().strip()
                except VoiceUnavailableError as exc:
                    print(f"\nVoice input error: {exc}")
                    print("Still in voice mode. Say /text to switch back, or try again.")
                    continue
                user_text = prepare_voice_text(user_text, voice_confirm)
                if not user_text:
                    continue
                print(user_text)
            else:
                user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not user_text:
            continue
        command = user_text.lower()
        if command in {"/exit", "exit", "quit","by","bye","ok by", "ok bye","by by"}:
            print("Goodbye.")
            return 0
        if command == "/help":
            print(HELP_TEXT)
            continue
        if command == "/tools":
            tools = await agent.tool_names()
            print("Tools: " + ", ".join(tools))
            continue
        if is_voice_command(command) and not VOICE_ENABLED:
            print(VOICE_DISABLED_MESSAGE)
            continue
        if command == "/voice":
            try:
                voice_input = OfflineVoiceInput(VoiceConfig(engine=voice_engine))
            except VoiceUnavailableError as exc:
                print(f"Voice input unavailable: {exc}")
                print("Try /voice-engine sphinx, or add a Vosk model under F:\\PortableAI\\models.")
                continue
            input_mode = "voice"
            print(f"Voice mode enabled using {voice_input.engine}. Say /text to switch back after a recognized utterance.")
            continue
        if command.startswith("/voice-engine"):
            parts = command.split()
            if len(parts) != 2:
                print("Usage: /voice-engine auto|vosk|sphinx")
                continue
            voice_engine = parts[1]
            voice_input = None
            try:
                probe = OfflineVoiceInput(VoiceConfig(engine=voice_engine))
            except VoiceUnavailableError as exc:
                print(f"Voice engine unavailable: {exc}")
                continue
            print(f"Voice engine set to {probe.engine}.")
            voice_input = probe
            continue
        if command == "/voice-models":
            models = available_vosk_models()
            selected = resolve_vosk_model_path()
            if not models:
                print("No Vosk model folders found. Expected a folder like F:\\PortableAI\\models\\vosk-model-small-en-us-0.15")
            else:
                print("Vosk models:")
                for model in models:
                    marker = " [selected]" if selected and str(selected) == model else ""
                    print(f"  {model}{marker}")
            continue
        if command == "/text":
            input_mode = "text"
            print("Text mode enabled.")
            continue
        if command == "/listen":
            try:
                voice_input = voice_input or OfflineVoiceInput(VoiceConfig(engine=voice_engine))
                spoken = voice_input.listen_once().strip()
            except VoiceUnavailableError as exc:
                print(f"Voice input error: {exc}")
                continue
            user_text = prepare_voice_text(spoken, voice_confirm)
            if not user_text:
                continue
            print(f"You [voice]: {user_text}")
            command = user_text.lower()
            if not user_text:
                continue
        if command == "/confirm":
            voice_confirm = not voice_confirm
            print(f"Voice confirmation {'enabled' if voice_confirm else 'disabled'}.")
            continue
        if command == "/voice-status":
            status = voice_dependencies_status()
            print(
                "Voice dependencies: "
                + ", ".join(f"{name}={'ok' if installed else 'missing'}" for name, installed in status.items())
            )
            selected_model = resolve_vosk_model_path()
            print(f"Voice engine preference: {voice_engine}")
            print(f"Voice confirmation: {'on' if voice_confirm else 'off'}")
            print(f"Detected Vosk model: {selected_model if selected_model else 'none'}")
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


def prepare_voice_text(spoken: str, confirm: bool) -> str:
    normalized, confidence, source = normalize_voice_command(spoken)
    print(f"\nHeard: {spoken}")
    if normalized != spoken.strip():
        print(f"Interpreted: {normalized} ({source}, confidence {confidence:.2f})")
    if not confirm:
        return normalized

    answer = input("Send this command? [Y/n/edit]: ").strip()
    if not answer or answer.lower() in {"y", "yes"}:
        return normalized
    if answer.lower() in {"n", "no"}:
        print("Cancelled.")
        return ""
    return answer


def is_voice_command(command: str) -> bool:
    return command in {"/voice", "/text", "/listen", "/confirm", "/voice-status", "/voice-models"} or command.startswith(
        "/voice-engine"
    )


def main() -> None:
    try:
        raise SystemExit(asyncio.run(run_chat()))
    except KeyboardInterrupt:
        print("\nGoodbye.")
        raise SystemExit(0)


if __name__ == "__main__":
    sys.exit(main())
