"""Headless render smoke test for the Streamlit app.

Catches render-time bugs (import errors, missing i18n keys, st.columns unpack
mismatches, widget-key misuse) without clicking run buttons (which would hit
the network). Skipped if streamlit's AppTest harness is unavailable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "src/optagent/web/app.py"


def test_app_renders_default_chinese_screen_view():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, f"render raised: {at.exception}"
    labels = [b.label for b in at.button]
    # Default view is Market screen -> its run button is localised to zh.
    assert "运行筛选" in labels
    # Chat panel is always present (persistent), regardless of the active view.
    assert len(at.chat_input) >= 1


def test_app_renders_analyze_view_without_exception():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["active_view"] = "analyze"
    at.session_state["view_radio"] = "analyze"
    at.run()
    assert not at.exception, f"render raised: {at.exception}"
    assert "分析" in [b.label for b in at.button]


def test_app_renders_english_without_exception():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["lang"] = "en"
    at.run()
    assert not at.exception, f"render raised: {at.exception}"
    assert "Run screen" in [b.label for b in at.button]


@pytest.mark.parametrize("view", ["screen", "analyze", "ml", "ledger"])
def test_chat_panel_persists_across_all_views(view):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["active_view"] = view
    at.session_state["view_radio"] = view
    at.run()
    assert not at.exception, f"render raised on view={view}: {at.exception}"
    # The persistent chat panel (its chat_input) is present on every view.
    assert len(at.chat_input) >= 1


def test_analyze_plain_render_does_not_call_provider(monkeypatch):
    called = {"analyze": 0}

    def _spy(*a, **k):
        called["analyze"] += 1
        raise AssertionError("analyze must not be auto-called on a plain render")

    monkeypatch.setattr("optagent.orchestrator.analyze", _spy)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["active_view"] = "analyze"
    at.session_state["view_radio"] = "analyze"
    # No pending_drilldown and the Analyze button is not pressed.
    at.run()
    assert not at.exception, f"render raised: {at.exception}"
    assert called["analyze"] == 0
