"""Console-script entry point: `optagent-ui`.

Spawns ``streamlit run path/to/app.py`` so users don't need to remember
the streamlit invocation. We delegate to streamlit's CLI rather than
calling its Bootstrap.run() internals so we stay forward-compatible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


APP_PATH = Path(__file__).parent / "app.py"


def main() -> int:
    try:
        from streamlit.web import cli as stcli  # noqa: WPS433
    except ImportError:
        sys.stderr.write(
            "ERROR: streamlit is not installed. Install with `pip install streamlit plotly`\n"
            "or with the project's ui extra: `pip install -e .[ui]`\n"
        )
        return 2

    argv = ["streamlit", "run", str(APP_PATH), "--server.headless=false"]
    # Pass through any extra args, e.g. --server.port 8765
    if len(sys.argv) > 1:
        argv.extend(sys.argv[1:])
    sys.argv = argv
    return int(stcli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
