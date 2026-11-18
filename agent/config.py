from __future__ import annotations

from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent
PROMPTS_DIR = AGENT_DIR / "prompts"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
AGENT_STATE_DB = CACHE_DIR / "agent_state.db"
MODELS_DIR = PROJECT_ROOT / "models"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
MODEL_EXE = MODELS_DIR / "Qwen3.5-0.8B-Q8_0.exe"

# Alexa voice pipeline. Root-level locations are preferred for the final
# portable layout; the validated prototype locations remain compatible while
# the integration is being migrated.
VOICE_PIPELINE_DIR = PROJECT_ROOT / "open-wake-word-with-audio-pipeline"


def preferred_file(primary: Path, fallback: Path) -> Path:
    return primary if primary.is_file() else fallback


AUDIO_SERVER_EXE = preferred_file(
    PROJECT_ROOT / "runtime" / "llama.cpp" / "llama-server.exe",
    VOICE_PIPELINE_DIR / "llama.cpp" / "llama-server.exe",
)
AUDIO_MODEL_PATH = preferred_file(
    MODELS_DIR / "LFM3-Omni-2B-Q4_K_M.gguf",
    VOICE_PIPELINE_DIR / "models" / "LFM3-Omni-2B-Q4_K_M.gguf",
)
AUDIO_MMPROJ_PATH = preferred_file(
    MODELS_DIR / "mmproj-LFM3-Omni-2B-Q4_K_M.gguf",
    VOICE_PIPELINE_DIR / "models" / "mmproj-LFM3-Omni-2B-Q4_K_M.gguf",
)

ROOT_WAKE_MODELS_DIR = MODELS_DIR / "openwakeword"
PROTOTYPE_WAKE_MODELS_DIR = VOICE_PIPELINE_DIR / "models" / "openwakeword"
WAKE_MODEL_FILENAMES = (
    "alexa_v0.1.onnx",
    "melspectrogram.onnx",
    "embedding_model.onnx",
)
ROOT_WAKE_MODELS_READY = all((ROOT_WAKE_MODELS_DIR / name).is_file() for name in WAKE_MODEL_FILENAMES)
PROTOTYPE_WAKE_MODELS_READY = all(
    (PROTOTYPE_WAKE_MODELS_DIR / name).is_file() for name in WAKE_MODEL_FILENAMES
)
WAKE_MODELS_DIR = (
    ROOT_WAKE_MODELS_DIR
    if ROOT_WAKE_MODELS_READY or not PROTOTYPE_WAKE_MODELS_READY
    else PROTOTYPE_WAKE_MODELS_DIR
)
ALEXA_WAKE_MODEL_PATH = WAKE_MODELS_DIR / "alexa_v0.1.onnx"
WAKE_MELSPECTROGRAM_PATH = WAKE_MODELS_DIR / "melspectrogram.onnx"
WAKE_EMBEDDING_PATH = WAKE_MODELS_DIR / "embedding_model.onnx"

AUDIO_SERVER_HOST = "127.0.0.1"
AUDIO_SERVER_PORT = 8081
AUDIO_SERVER_BASE_URL = f"http://{AUDIO_SERVER_HOST}:{AUDIO_SERVER_PORT}"
AUDIO_TRANSCRIPTIONS_URL = f"{AUDIO_SERVER_BASE_URL}/v1/audio/transcriptions"
AUDIO_MODEL_ID = "LFM3-Omni-2B"
AUDIO_ASR_PROMPT = "Perform ASR in English."

WAKE_WORD = "alexa"
WAKE_THRESHOLD = 0.5
VOICE_SILENCE_SECONDS = 2.0
VOICE_SILENCE_RMS = 450.0
VOICE_MAX_RECORDING_SECONDS = 30.0

LLAMAFILE_HOST = "127.0.0.1"
LLAMAFILE_PORT = 8080
LLAMAFILE_BASE_URL = f"http://{LLAMAFILE_HOST}:{LLAMAFILE_PORT}"
MODEL_ID = "/zip/Qwen3.5-0.8B-Q8_0.gguf"

CLIPBOARD_READ_LIMIT = 4096
NOTEPAD_TEXT_LIMIT = 32768
MAX_PROCESS_COUNT = 80
DIRECTORY_LIST_LIMIT = 200
DIRECTORY_SCAN_LIMIT = 5000

ALLOWED_APPLICATIONS: dict[str, tuple[str, ...]] = {
    "notepad": ("notepad.exe",),
    "calculator": ("calc.exe",),
    "calc": ("calc.exe",),
    "chrome": ("chrome.exe",),
    "whatsapp": ("whatsapp.exe",),
    "notion": ("notion.exe",),
    "vscode": ("Code.exe",),
    "github_desktop": ("GitHubDesktop.exe",),
    "outlook": ("OUTLOOK.EXE",),
    "mongodb_compass": ("MongoDBCompass.exe",),
    "mongodb": ("MongoDBCompass.exe",),
    "termius": ("Termius.exe",),
}
