"""Zero-dependency `.env` loader.

We deliberately avoid pulling in `python-dotenv`: the format we support is
the trivial subset (`KEY=value`, `#` comments, optional surrounding quotes).
Existing process environment ALWAYS wins — a `.env` file never clobbers a
variable the user exported in their shell or CI.

Call `load_dotenv()` once at an entry point (CLI, Streamlit launcher). It is
intentionally NOT called at import time of library modules so the test suite
stays hermetic (a stray `.env` must not silently configure a provider during
`pytest`).
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_dotenv(start: Path) -> Path | None:
    """Walk up from `start` looking for a `.env` file (repo-root convention)."""

    for parent in [start, *start.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: str | os.PathLike[str] | None = None, *, override: bool = False) -> dict[str, str]:
    """Load `KEY=value` pairs from a `.env` file into `os.environ`.

    Returns the dict of keys it actually set (for logging / tests). Missing
    file is a no-op. Lines that don't parse are skipped silently — a malformed
    `.env` must never crash the app at startup.
    """

    if path is not None:
        dotenv_path: Path | None = Path(path)
        if not dotenv_path.is_file():
            dotenv_path = None
    else:
        dotenv_path = _find_dotenv(Path.cwd())

    if dotenv_path is None:
        return {}

    applied: dict[str, str] = {}
    try:
        raw = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip a single layer of matching quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
