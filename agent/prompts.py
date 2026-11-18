from __future__ import annotations

import re
from functools import lru_cache

from config import PROMPTS_DIR


PLACEHOLDER_PATTERN = re.compile(r"\{\{(\w+)\}\}")


@lru_cache(maxsize=None)
def read_prompt(name: str) -> str:
    """Read prompts/<name>.md, tolerating a BOM and Windows line endings."""
    path = PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8-sig").rstrip()


def load_prompt(name: str, **values: str) -> str:
    """Load a prompt file and fill each {{placeholder}} with the matching keyword value."""
    text = read_prompt(name)

    def fill(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"Prompt {name!r} requires a value for {{{{{key}}}}}.")
        return values[key]

    return PLACEHOLDER_PATTERN.sub(fill, text)
