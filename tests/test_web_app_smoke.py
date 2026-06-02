"""Headless render smoke test for the Streamlit app.

Catches the class of bug where a tab body raises at render time (import
errors, missing i18n keys, `st.columns(n)` unpack mismatches, etc.). It does
NOT click the run buttons — those trigger live network fetches — so it stays
hermetic. Skipped if streamlit's AppTest harness is unavailable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "src/optagent/web/app.py"


def test_app_renders_default_chinese_without_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.exception == [] or not at.exception, f"render raised: {at.exception}"
    labels = [b.label for b in at.button]
    # Default language is Chinese -> the run buttons are localised.
    assert "分析" in labels
    assert "运行筛选" in labels


def test_app_renders_english_without_exception():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["lang"] = "en"
    at.run()
    assert not at.exception, f"render raised: {at.exception}"
    labels = [b.label for b in at.button]
    assert "Analyze" in labels
    assert "Run screen" in labels
