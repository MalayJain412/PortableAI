from __future__ import annotations

from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_EXE = MODELS_DIR / "Qwen3.5-0.8B-Q8_0.exe"

LLAMAFILE_HOST = "127.0.0.1"
LLAMAFILE_PORT = 8080
LLAMAFILE_BASE_URL = f"http://{LLAMAFILE_HOST}:{LLAMAFILE_PORT}"
MODEL_ID = "/zip/Qwen3.5-0.8B-Q8_0.gguf"

CLIPBOARD_READ_LIMIT = 4096
NOTEPAD_TEXT_LIMIT = 32768
MAX_PROCESS_COUNT = 80

ALLOWED_APPLICATIONS: dict[str, tuple[str, ...]] = {
    "notepad": ("notepad.exe",),
    "calculator": ("calc.exe",),
    "calc": ("calc.exe",),
    "chrome": ("chrome.exe",),
    "whatsapp": ("whatsapp.exe",),
}
