from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import MODELS_DIR

os.environ.setdefault("VOSK_LOG_LEVEL", "-1")

_VOSK_MODELS: dict[str, Any] = {}


class VoiceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceConfig:
    timeout_seconds: float = 6.0
    phrase_time_limit_seconds: float = 10.0
    ambient_duration_seconds: float = 0.6
    engine: str = "auto"
    vosk_model_path: Path | None = None


class OfflineVoiceInput:
    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig()
        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise VoiceUnavailableError(
                "Voice input packages are not installed. Install SpeechRecognition, pocketsphinx, and PyAudio."
            ) from exc

        self.sr: Any = sr
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.engine = select_engine(self.config.engine, self.config.vosk_model_path)

    def listen_once(self) -> str:
        try:
            microphone = self.sr.Microphone()
        except Exception as exc:
            raise VoiceUnavailableError(
                "Could not access a microphone. Check PyAudio installation and Windows microphone permissions."
            ) from exc

        with microphone as source:
            print("Listening... speak now.")
            self.recognizer.adjust_for_ambient_noise(source, duration=self.config.ambient_duration_seconds)
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=self.config.timeout_seconds,
                    phrase_time_limit=self.config.phrase_time_limit_seconds,
                )
            except self.sr.WaitTimeoutError as exc:
                raise VoiceUnavailableError("No speech detected before timeout.") from exc

        if self.engine == "vosk":
            return self._recognize_vosk(audio)
        return self._recognize_sphinx(audio)

    def _recognize_sphinx(self, audio: Any) -> str:
        try:
            return self.recognizer.recognize_sphinx(audio).strip()
        except self.sr.UnknownValueError as exc:
            raise VoiceUnavailableError("I could not understand that audio.") from exc
        except self.sr.RequestError as exc:
            raise VoiceUnavailableError(f"Offline speech recognizer failed: {exc}") from exc

    def _recognize_vosk(self, audio: Any) -> str:
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as exc:
            raise VoiceUnavailableError("Vosk is not installed. Use /voice-engine sphinx or install vosk.") from exc
        SetLogLevel(-1)

        model_path = resolve_vosk_model_path(self.config.vosk_model_path)
        if model_path is None:
            raise VoiceUnavailableError(
                "Vosk is installed, but no Vosk model folder was found under F:\\PortableAI\\models. "
                "Download an English Vosk model and extract it there, or use /voice-engine sphinx."
            )

        sample_rate = 16000
        wav_data = audio.get_raw_data(convert_rate=sample_rate, convert_width=2)
        model_key = str(model_path)
        model = _VOSK_MODELS.get(model_key)
        if model is None:
            model = Model(model_key)
            _VOSK_MODELS[model_key] = model
        recognizer = KaldiRecognizer(model, sample_rate)
        recognizer.AcceptWaveform(wav_data)
        result = json.loads(recognizer.FinalResult())
        text = str(result.get("text", "")).strip()
        if not text:
            raise VoiceUnavailableError("I could not understand that audio.")
        return text


def select_engine(requested: str, vosk_model_path: Path | None = None) -> str:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "vosk", "sphinx"}:
        raise VoiceUnavailableError("Voice engine must be one of: auto, vosk, sphinx.")
    if normalized == "sphinx":
        return "sphinx"
    if normalized == "vosk":
        if resolve_vosk_model_path(vosk_model_path) is None:
            raise VoiceUnavailableError("Vosk model folder not found. Use /voice-models for the expected location.")
        return "vosk"
    if voice_dependencies_status().get("vosk") and resolve_vosk_model_path(vosk_model_path) is not None:
        return "vosk"
    return "sphinx"


def resolve_vosk_model_path(explicit: Path | None = None) -> Path | None:
    if explicit is not None and explicit.exists():
        return explicit
    if not MODELS_DIR.exists():
        return None
    candidates = [path for path in MODELS_DIR.iterdir() if path.is_dir() and path.name.lower().startswith("vosk-model")]
    if not candidates:
        return None
    # Prefer small English models when more than one is present.
    candidates.sort(key=lambda path: (0 if "en" in path.name.lower() else 1, len(path.name), path.name.lower()))
    return candidates[0]


def available_vosk_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return [str(path) for path in MODELS_DIR.iterdir() if path.is_dir() and path.name.lower().startswith("vosk-model")]


def voice_dependencies_status() -> dict[str, bool]:
    import importlib.util

    return {
        "speech_recognition": importlib.util.find_spec("speech_recognition") is not None,
        "pocketsphinx": importlib.util.find_spec("pocketsphinx") is not None,
        "pyaudio": importlib.util.find_spec("pyaudio") is not None,
        "vosk": importlib.util.find_spec("vosk") is not None,
    }


def normalize_voice_command(text: str) -> tuple[str, float, str]:
    cleaned = clean_spoken_text(text)
    if not cleaned:
        return "", 0.0, "empty"

    if is_exit_phrase(cleaned):
        return "bye", 1.0, "exit"

    if is_youtube_chrome_phrase(cleaned):
        return "open youtube.com on chrome", 0.95, "rule"

    if is_chrome_phrase(cleaned):
        return "open chrome", 0.9, "rule"

    if is_notepad_phrase(cleaned):
        return "open notepad", 0.9, "rule"

    if is_calculator_phrase(cleaned):
        return "open calculator", 0.9, "rule"

    candidates = [
        "open youtube.com on chrome",
        "open chrome",
        "open notepad",
        "open calculator",
        "what can you do",
        "what are my basic system specs",
        "what applications are currently running",
        "read clipboard",
    ]
    best = max(candidates, key=lambda candidate: SequenceMatcher(None, cleaned, candidate).ratio())
    score = SequenceMatcher(None, cleaned, best).ratio()
    if score >= 0.78:
        return best, score, "fuzzy"
    return cleaned, score, "raw"


def clean_spoken_text(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9.@/ ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def is_exit_phrase(text: str) -> bool:
    return text in {"bye", "bye bye", "goodbye", "exit", "quit", "by", "by by", "ok bye", "okay bye"}


def is_youtube_chrome_phrase(text: str) -> bool:
    has_open = any(word in text for word in ("open", "go", "launch", "start"))
    has_chrome = "chrome" in text or "grown" in text or "groan" in text or "gold" in text
    has_youtube = "youtube" in text or "you tube" in text or "u tube" in text or "u. tube" in text or "tube" in text
    has_site = ".com" in text or "dot com" in text or "total" in text
    return has_open and has_youtube and (has_chrome or has_site)


def is_chrome_phrase(text: str) -> bool:
    return any(word in text for word in ("open chrome", "launch chrome", "start chrome", "open grown", "open groan"))


def is_notepad_phrase(text: str) -> bool:
    return "notepad" in text or "note pad" in text


def is_calculator_phrase(text: str) -> bool:
    return "calculator" in text or "calculate" in text
