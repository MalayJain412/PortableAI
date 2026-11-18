from __future__ import annotations

import sys
from pathlib import Path


# The tests import agent modules (agent, windows_tools, security, ...) by their top-level names.
AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
