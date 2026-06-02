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


def test_explain_button_visible_with_empty_synthesis():
    at = AppTest.from_file(APP, default_timeout=60)
    # A completed screen with per-strategy results but NO synthesis rows.
    at.session_state["last_screen"] = {
        "results": {"s1": {"error": None, "signals": [], "n_triggered": 0, "n_evaluated": 5}},
        "synthesis": [],
    }
    at.run()
    assert not at.exception, f"render raised: {at.exception}"
    assert "🤖 让 AI 解释结果" in [b.label for b in at.button]  # default zh label


def test_failed_screen_attempt_overwrites_grounding():
    at = AppTest.from_file(APP, default_timeout=60)
    # Pre-seed a stale successful screen snapshot in the store.
    at.session_state["last_screen"] = {"results": {}, "synthesis": []}
    at.run()
    # Empty the strategy multiselect, then click "Run screen".
    at.multiselect[0].set_value([]).run()
    run_btn = [b for b in at.button if b.label == "运行筛选"][0]
    run_btn.click().run()
    assert not at.exception, f"render raised: {at.exception}"
    screen_snap = at.session_state["research_store"]["screen"]
    assert screen_snap is not None and screen_snap["available"] is False


def test_ledger_view_writes_snapshot():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["active_view"] = "ledger"
    at.session_state["view_radio"] = "ledger"
    at.run()
    assert not at.exception, f"render raised: {at.exception}"
    led = at.session_state["research_store"]["ledger"]
    assert led is not None and "available" in led  # the view always records a snapshot


def test_empty_universe_failed_screen_overwrites_grounding(monkeypatch):
    # An empty sector-filtered universe must overwrite prior successful grounding.
    # _build_universe lazily imports filter_to_sector from the source module, so
    # patching the source there takes effect inside the AppTest-loaded script.
    monkeypatch.setattr("optagent.strategies.filter_to_sector", lambda u, s: [])
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["last_screen"] = {"results": {}, "synthesis": []}  # stale success
    at.run()
    # Pick the sector selectbox (its options include the sector_any marker) and
    # choose a real sector so the (patched) sector filter runs.
    sector_sb = [sb for sb in at.selectbox if any("全部" in str(o) for o in sb.options)][0]
    real_sector = [o for o in sector_sb.options if "全部" not in str(o)][0]
    sector_sb.set_value(real_sector).run()
    run_btn = [b for b in at.button if b.label == "运行筛选"][0]
    run_btn.click().run()
    assert not at.exception, f"render raised: {at.exception}"
    screen_snap = at.session_state["research_store"]["screen"]
    assert screen_snap is not None and screen_snap["available"] is False


def test_non_empty_ledger_writes_available_snapshot(monkeypatch):
    import pandas as pd

    df = pd.DataFrame({"action": ["SKIP", "SKIP", "LONG_CALL"], "ticker": ["A", "B", "C"]})
    # Patch the SOURCE (components) since the AppTest script re-imports it there.
    monkeypatch.setattr("optagent.web.components.ledger_index", lambda *a, **k: df)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["active_view"] = "ledger"
    at.session_state["view_radio"] = "ledger"
    at.run()
    assert not at.exception, f"render raised: {at.exception}"
    led = at.session_state["research_store"]["ledger"]
    assert led["available"] is True
    assert led["n_rows"] == 3
    assert led["action_counts"].get("SKIP") == 2
    assert led["inputs"].get("days_back") is not None


def test_explain_handler_renders_error_on_provider_exception(monkeypatch):
    # A non-RuntimeError provider failure (timeout/rate-limit/APIError) must be
    # caught and rendered, not crash the Streamlit interaction.
    def _boom(*a, **k):
        raise ValueError("simulated provider timeout")

    monkeypatch.setattr("optagent.web.screen_llm.explain_screen", _boom)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["last_screen"] = {
        "results": {"s1": {"error": None, "signals": [], "n_triggered": 0}},
        "synthesis": [],
    }
    at.run()
    explain_btn = [b for b in at.button if "解释" in b.label][0]
    explain_btn.click().run()
    assert not at.exception, f"explain handler did not catch provider error: {at.exception}"


def test_run_multi_strategy_shares_memoized_fetcher(monkeypatch):
    # The same fetcher must be reused across strategies and memoized per ticker
    # so market data is downloaded once, not once per strategy.
    base_calls = []

    def fake_base(ticker):
        base_calls.append(ticker)
        return (None, None)

    monkeypatch.setattr("optagent.strategies.screen.make_default_fetcher", lambda *a, **k: fake_base)
    monkeypatch.setattr("optagent.strategies.get_strategy", lambda sid: object())

    fetchers = []

    class _Res:
        top_signals = []
        n_triggered = 0
        n_evaluated = 0
        stale_bars = []
        top_near_misses = []

    def fake_screen(strategy, universe, *, fetcher, top_n):
        fetchers.append(fetcher)
        for tk in universe:
            fetcher(tk)  # each strategy "fetches" every ticker
        return _Res()

    monkeypatch.setattr("optagent.strategies.screen_universe", fake_screen)

    from optagent.web.app import _run_multi_strategy

    _run_multi_strategy(["s1", "s2", "s3"], ["AAPL", "MSFT"])
    assert fetchers[0] is fetchers[1] is fetchers[2]      # one shared fetcher
    assert sorted(base_calls) == ["AAPL", "MSFT"]          # each ticker fetched ONCE


def test_run_multi_strategy_does_not_pretruncate_before_synthesis(monkeypatch):
    # Regression for the review [P1]: each strategy must be screened with an
    # UNBOUNDED top-N (the full universe) so synthesis sees every triggered row.
    captured = {}

    class _Res:
        top_signals = []
        n_triggered = 0
        n_evaluated = 0
        stale_bars = []
        top_near_misses = []

    def _fake_screen(strategy, universe, *, fetcher=None, top_n):
        captured["top_n"] = top_n
        return _Res()

    monkeypatch.setattr("optagent.strategies.get_strategy", lambda sid: object())
    monkeypatch.setattr("optagent.strategies.screen_universe", _fake_screen)

    from optagent.web.app import _run_multi_strategy

    universe = [f"T{i}" for i in range(85)]
    _run_multi_strategy(["s1"], universe)
    assert captured["top_n"] == len(universe)  # not the user's small Top-N


def test_ml_plain_render_does_not_call_provider(monkeypatch):
    called = {"ml": 0}

    class _SpyAdapter:
        def __init__(self, *a, **k):
            called["ml"] += 1

        def signal(self, *a, **k):
            called["ml"] += 1
            return None

    monkeypatch.setattr("optagent.ml.MLDirectionAdapter", _SpyAdapter)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["active_view"] = "ml"
    at.session_state["view_radio"] = "ml"
    # No pending_drilldown and the Compute-signal button is not pressed.
    at.run()
    assert not at.exception, f"render raised: {at.exception}"
    assert called["ml"] == 0
