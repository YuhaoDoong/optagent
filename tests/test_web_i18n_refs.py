"""AC-8 negative test: every i18n key referenced in the web UI source must
exist in the translation table (no silent key-echo in either language)."""

from __future__ import annotations

import re
from pathlib import Path

from optagent.web.i18n import _TABLE

_WEB_DIR = Path(__file__).resolve().parents[1] / "src" / "optagent" / "web"
_KEY_RE = re.compile(r"""\bt\(\s*["']([a-zA-Z0-9_.]+)["']""")


def test_all_referenced_i18n_keys_exist():
    missing: list[str] = []
    for py in _WEB_DIR.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for key in _KEY_RE.findall(text):
            if key not in _TABLE:
                missing.append(f"{py.name}:{key}")
    assert not missing, f"referenced i18n keys absent from table: {missing}"


def test_every_table_key_has_both_languages():
    for key, row in _TABLE.items():
        assert row.get("en"), f"{key} missing en"
        assert row.get("zh"), f"{key} missing zh"
