"""Alexa wake-word input backed by the portable LFM3-Omni llama.cpp server."""

from __future__ import annotations

import atexit
import importlib.util
import os
import queue
import subprocess
import tempfile
import time
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, BinaryIO

from config import (
    ALEXA_WAKE_MODEL_PATH,
    AUDIO_ASR_PROMPT,
    AUDIO_MMPROJ_PATH,
    AUDIO_MODEL_ID,
    AUDIO_MODEL_PATH,
    AUDIO_SERVER_BASE_URL,
    AUDIO_SERVER_EXE,
    AUDIO_SERVER_HOST,
    AUDIO_SERVER_PORT,
    AUDIO_TRANSCRIPTIONS_URL,
    LOGS_DIR,
    ROOT_WAKE_MODELS_DIR,
    VOICE_MAX_RECORDING_SECONDS,
    VOICE_SILENCE_RMS,
    VOICE_SILENCE_SECONDS,
    WAKE_EMBEDDING_PATH,
    WAKE_MELSPECTROGRAM_PATH,
    WAKE_THRESHOLD,
    WAKE_WORD,
)


REQUIRED_MODULES = ("numpy", "scipy", "sounddevice", "httpx", "openwakeword", "onnxruntime")
AUDIO_QUEUE_SIZE = 200
MODEL_START_TIMEOUT_SECONDS = 180.0


class VoiceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class MicrophoneInput:
    index: int
    name: str
    host_api: str
    sample_rate: int
    channels: int


def voice_dependencies_status() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}


def audio_server_command() -> list[str]:
    return [
        str(AUDIO_SERVER_EXE),
        "--model",
        str(AUDIO_MODEL_PATH),
        "--mmproj",
        str(AUDIO_MMPROJ_PATH),
        "--host",
        AUDIO_SERVER_HOST,
        "--port",
        str(AUDIO_SERVER_PORT),
        "--reasoning",
        "off",
        "--temp",
        "0",
    ]


def required_voice_files() -> tuple[Path, ...]:
    return (
        AUDIO_SERVER_EXE,
        AUDIO_MODEL_PATH,
        AUDIO_MMPROJ_PATH,
        ALEXA_WAKE_MODEL_PATH,
        WAKE_MELSPECTROGRAM_PATH,
        WAKE_EMBEDDING_PATH,
    )


def missing_voice_files() -> list[Path]:
    return [path for path in required_voice_files() if not path.is_file()]


def audio_server_ready(timeout: float = 1.5) -> bool:
    try:
        import httpx

        response = httpx.get(f"{AUDIO_SERVER_BASE_URL}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def extract_transcription(response: Any) -> str:
    text = response.text.strip()
    try:
        data = response.json()
    except ValueError:
        return text
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        for key in ("text", "transcript", "transcription"):
            value = data.get(key)
            if isinstance(value, str):
                return value.strip()
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                value = choice.get("text")
                if isinstance(value, str):
                    return value.strip()
    return text


class AudioServer:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.log_file: BinaryIO | None = None
        self.owns_process = False

    def ensure_ready(self) -> None:
        if audio_server_ready():
            return
        missing = [path for path in (AUDIO_SERVER_EXE, AUDIO_MODEL_PATH, AUDIO_MMPROJ_PATH) if not path.is_file()]
        if missing:
            paths = "\n".join(f"  - {path}" for path in missing)
            raise VoiceUnavailableError(f"The portable audio runtime is incomplete:\n{paths}")

        if self.process is not None and self.process.poll() is not None:
            self.close()

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / "llamacpp-audio.log"
        self.log_file = log_path.open("ab")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        print(f"Starting Alexa transcription server at {AUDIO_SERVER_BASE_URL}...", flush=True)
        self.process = subprocess.Popen(
            audio_server_command(),
            cwd=AUDIO_SERVER_EXE.parent,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        self.owns_process = True
        atexit.register(self.close)

        deadline = time.monotonic() + MODEL_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                code = self.process.returncode
                self.close()
                raise VoiceUnavailableError(
                    f"The Alexa transcription server exited with code {code}. Check {log_path}."
                )
            if audio_server_ready():
                print("Alexa transcription server is ready.", flush=True)
                return
            time.sleep(0.5)

        self.close()
        raise VoiceUnavailableError(f"The Alexa transcription server did not become ready. Check {log_path}.")

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None and self.owns_process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        self.owns_process = False
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None


class AlexaVoiceInput:
    """Wait for Alexa, record one command, and return its transcription."""

    def __init__(self) -> None:
        missing_dependencies = [name for name, installed in voice_dependencies_status().items() if not installed]
        if missing_dependencies:
            raise VoiceUnavailableError(
                "Missing voice dependencies: " + ", ".join(missing_dependencies)
                + ". Install agent\\requirements.txt into the portable runtime."
            )

        self._ensure_wake_models()
        missing = missing_voice_files()
        if missing:
            paths = "\n".join(f"  - {path}" for path in missing)
            raise VoiceUnavailableError(f"The Alexa voice runtime is incomplete:\n{paths}")

        import numpy as np
        import sounddevice as sd
        from openwakeword.model import Model
        from scipy.io.wavfile import write
        from scipy.signal import resample_poly

        self.np = np
        self.sd = sd
        self.write_wav = write
        self.resample_poly = resample_poly
        self.server = AudioServer()
        try:
            self.server.ensure_ready()
            self.microphone = self._select_microphone()
            print(
                f"Using {self.microphone.name} ({self.microphone.host_api}) at "
                f"{self.microphone.sample_rate} Hz, {self.microphone.channels} capture channel(s).",
                flush=True,
            )
            print("Loading Alexa wake-word model...", flush=True)
            self.wake_model = Model(
                wakeword_models=[str(ALEXA_WAKE_MODEL_PATH)],
                inference_framework="onnx",
                melspec_model_path=str(WAKE_MELSPECTROGRAM_PATH),
                embedding_model_path=str(WAKE_EMBEDDING_PATH),
            )
            print("Alexa wake-word model loaded.", flush=True)
        except Exception as error:
            self.server.close()
            if isinstance(error, VoiceUnavailableError):
                raise
            raise VoiceUnavailableError(str(error)) from error

        self.audio_queue: queue.Queue[Any] = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
        self.prediction_key = ALEXA_WAKE_MODEL_PATH.stem

    def _ensure_wake_models(self) -> None:
        wake_files = (ALEXA_WAKE_MODEL_PATH, WAKE_MELSPECTROGRAM_PATH, WAKE_EMBEDDING_PATH)
        if all(path.is_file() for path in wake_files):
            return
        try:
            from openwakeword.utils import download_models

            print("Downloading Alexa wake-word files for first-time setup...", flush=True)
            ROOT_WAKE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            download_models(model_names=[WAKE_WORD], target_directory=str(ROOT_WAKE_MODELS_DIR))
        except Exception as error:
            raise VoiceUnavailableError(
                "Alexa wake-word files are missing and could not be downloaded. "
                f"Connect to the internet and retry. Details: {error}"
            ) from error

    def _select_microphone(self) -> MicrophoneInput:
        host_apis = self.sd.query_hostapis()
        devices = self.sd.query_devices()
        candidates = [
            (index, device)
            for index, device in enumerate(devices)
            if int(device["max_input_channels"]) > 0
        ]
        if not candidates:
            raise VoiceUnavailableError("No microphone input devices were found.")

        wasapi = next((api for api in host_apis if api["name"] == "Windows WASAPI"), None)
        preferred_index = int(wasapi["default_input_device"]) if wasapi else int(self.sd.default.device[0])
        candidates.sort(
            key=lambda item: (
                item[0] != preferred_index,
                host_apis[item[1]["hostapi"]]["name"] != "Windows WASAPI",
            )
        )

        for index, device in candidates:
            native_rate = round(float(device["default_samplerate"]))
            for rate in dict.fromkeys((native_rate, 48000, 44100, 16000)):
                if rate <= 0 or rate % 25:
                    continue
                try:
                    self.sd.check_input_settings(device=index, channels=1, dtype="int16", samplerate=rate)
                except (ValueError, self.sd.PortAudioError):
                    continue
                return MicrophoneInput(
                    index=index,
                    name=str(device["name"]),
                    host_api=str(host_apis[device["hostapi"]]["name"]),
                    sample_rate=rate,
                    channels=1,
                )
        raise VoiceUnavailableError("No compatible microphone format could be opened.")

    def _audio_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(f"Audio status: {status}", flush=True)
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _clear_queue(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                return

    def _mono(self, audio: Any) -> Any:
        if audio.ndim == 1:
            return audio
        if audio.shape[1] == 1:
            return audio[:, 0]
        return self.np.rint(audio.mean(axis=1)).astype(self.np.int16)

    def _wake_audio(self, audio: Any) -> Any:
        if self.microphone.sample_rate == 16000:
            return audio
        divisor = gcd(16000, self.microphone.sample_rate)
        resampled = self.resample_poly(
            audio.astype(self.np.float32),
            16000 // divisor,
            self.microphone.sample_rate // divisor,
        )
        return self.np.clip(resampled, -32768, 32767).astype(self.np.int16)

    def _record_command(self) -> Any | None:
        print("[RECORDING] Speak your command...", flush=True)
        chunks: list[Any] = []
        speech_started = False
        silence_start: float | None = None
        started = time.monotonic()

        while time.monotonic() - started < VOICE_MAX_RECORDING_SECONDS:
            audio = self._mono(self.audio_queue.get())
            chunks.append(audio.copy())
            samples = audio.astype(self.np.float32)
            level = float(self.np.sqrt(self.np.mean(samples ** 2)))
            now = time.monotonic()
            if level >= VOICE_SILENCE_RMS:
                speech_started = True
                silence_start = None
            elif speech_started:
                if silence_start is None:
                    silence_start = now
                elif now - silence_start >= VOICE_SILENCE_SECONDS:
                    print(f"[RECORDING STOPPED] {VOICE_SILENCE_SECONDS:g} seconds of silence detected.", flush=True)
                    break

        if not speech_started:
            print("[NO SPEECH] Waiting for Alexa again.", flush=True)
            return None
        return self.np.concatenate(chunks, axis=0)

    def _transcribe(self, audio: Any) -> str:
        import httpx

        self.server.ensure_ready()
        temporary_path: Path | None = None
        try:
            samples = self.np.clip(audio, -32768, 32767).astype(self.np.int16)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
                temporary_path = Path(temporary_file.name)
            self.write_wav(temporary_path, self.microphone.sample_rate, samples)
            print("[LLAMA.CPP] Transcribing command...", flush=True)
            with temporary_path.open("rb") as audio_file:
                response = httpx.post(
                    AUDIO_TRANSCRIPTIONS_URL,
                    files={"file": (temporary_path.name, audio_file, "audio/wav")},
                    data={
                        "model": AUDIO_MODEL_ID,
                        "prompt": AUDIO_ASR_PROMPT,
                        "response_format": "json",
                    },
                    timeout=180.0,
                )
            response.raise_for_status()
            transcription = extract_transcription(response).strip()
            if not transcription:
                raise VoiceUnavailableError("The audio server returned an empty transcription.")
            return transcription
        except VoiceUnavailableError:
            raise
        except Exception as error:
            raise VoiceUnavailableError(f"Audio transcription failed: {error}") from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def listen_once(self) -> str:
        self._clear_queue()
        self.wake_model.reset()
        blocksize = self.microphone.sample_rate * 2 // 25
        print("Waiting for wake word: Alexa", flush=True)
        with self.sd.InputStream(
            device=self.microphone.index,
            samplerate=self.microphone.sample_rate,
            channels=self.microphone.channels,
            dtype="int16",
            blocksize=blocksize,
            callback=self._audio_callback,
        ):
            while True:
                audio = self._mono(self.audio_queue.get())
                predictions = self.wake_model.predict(self._wake_audio(audio))
                score = float(predictions.get(self.prediction_key, 0.0))
                if score < WAKE_THRESHOLD:
                    continue
                print(f"[WAKE DETECTED] Alexa | Score: {score:.4f}", flush=True)
                self.wake_model.reset()
                command_audio = self._record_command()
                if command_audio is None:
                    self._clear_queue()
                    self.wake_model.reset()
                    continue
                transcription = self._transcribe(command_audio)
                print(f"You [voice]: {transcription}", flush=True)
                return transcription

    def close(self) -> None:
        self.server.close()


def voice_runtime_status() -> dict[str, Any]:
    return {
        "wake_word": WAKE_WORD,
        "dependencies": voice_dependencies_status(),
        "files": {str(path): path.is_file() for path in required_voice_files()},
        "audio_server_ready": audio_server_ready(),
    }
