"""Streamlit entry point: ``optagent-ui`` (or ``streamlit run -m optagent.web.app``).

Two-column layout: a main content area with a session-state-backed view
selector (replacing st.tabs so drill-down can navigate programmatically) and a
persistent right-side chat panel grounded on a session research store.

Main views (Market screen first):
  1. Market screen — multi-strategy run with per-strategy diagnostics, a
     deterministic cross-strategy synthesis, opt-in LLM explanation, and
     one-click drill-down into the Analyze / ML views.
  2. Analyze a single ticker (the CLI `analyze` subcommand visualised).
  3. ML direction signal (per-ticker model with Wilson CI gauge).
  4. Audit ledger viewer.

Every view respects the project's safety invariants:
  - Disclaimer banner at the top of the page.
  - Bounded verdict enum (template-only by default; LLM is opt-in).
  - Fail-closed validator runs unchanged behind the scenes.
  - LLM commentary (chat + screen explanation) is research-only and grounded
    in escaped, bounded <analysis_context> blocks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Pull provider keys (e.g. OPENROUTER_API_KEY) from a gitignored .env so the
# UI can synthesise / chat without the user exporting them by hand. Shell env
# always wins over .env (see env_loader.load_dotenv).
from optagent.env_loader import load_dotenv as _load_dotenv

_load_dotenv()

from optagent import DISCLAIMER, __version__
from optagent.web.chat import (
    ChatMessage,
    chat_complete,
)
from optagent.web.components import (
    candidate_table,
    candle_chart,
    envelope_summary,
    feature_radar,
    iv_smile_frame,
    ledger_index,
    ml_signal_gauge,
    strategy_signal_table,
    verdict_badge,
)
from optagent.web import research_store as rs
from optagent.web.i18n import supported_languages, t


# Ordered main views (replaces st.tabs so drill-down can navigate
# programmatically — st.tabs cannot be switched from code). Market screen
# is intentionally first.
_VIEWS: tuple[str, ...] = ("screen", "analyze", "ml", "ledger")
_VIEW_LABEL_KEY = {
    "screen": "tab.screen",
    "analyze": "tab.analyze",
    "ml": "tab.ml",
    "ledger": "tab.ledger",
}


def _store() -> dict[str, Any]:
    """Get-or-init the per-session research store."""

    if "research_store" not in st.session_state:
        st.session_state["research_store"] = rs.init_store()
    return st.session_state["research_store"]


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _consume_drilldown(target: str) -> str | None:
    """If a one-shot drill-down is pending for `target`, claim its ticker once.

    Returns the ticker (and clears the marker) on the first render after a
    drill-down click; returns None on subsequent reruns so the auto-run fires
    exactly once.
    """

    return rs.consume_drilldown(st.session_state, target)


# ---------------------------------------------------------------------------
# Page-level config (run once)

st.set_page_config(
    page_title="optagent — US equity options research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _current_lang() -> str:
    return st.session_state.get("lang", "zh")


def _disclaimer_banner() -> None:
    lang = _current_lang()
    msg = t("disclaimer.banner", lang, disclaimer=DISCLAIMER, version=__version__)
    import html as _html
    safe = _html.escape(msg)
    st.markdown(
        f"<div style='background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;"
        f"padding:8px 12px;font-weight:600;color:#78350f;margin-bottom:12px;'>"
        f"{safe}</div>",
        unsafe_allow_html=True,
    )


def _verdict_card(verdict_dict: dict[str, Any]) -> None:
    """Render the verdict badge.

    SECURITY: every field that flows into HTML below is either (a) a member
    of a closed enum (action, label) whose values are hard-coded in the
    project, or (b) a #RRGGBB colour string drawn from _DIRECTION_STYLE.
    No user-supplied or upstream string reaches the unsafe_allow_html
    sink — guards against the Codex web-UI audit's XSS concern.
    """

    import html as _html

    label = _html.escape(str(verdict_dict["label"]))[:64]
    color = str(verdict_dict["color"])
    if not (
        len(color) == 7
        and color.startswith("#")
        and all(c in "0123456789abcdefABCDEF" for c in color[1:])
    ):
        color = "#374151"  # fallback to neutral grey if upstream sends junk
    badge_html = (
        f"<div style='display:inline-block;padding:8px 16px;border-radius:6px;"
        f"background:{color};color:white;font-size:18px;font-weight:700;'>"
        f"{label}</div>"
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    info_bits = []
    conv = verdict_dict.get("conviction")
    if conv is not None:
        info_bits.append(f"Conviction: {conv:.2f}")
    skip_reason = verdict_dict.get("skip_reason")
    if skip_reason:
        # skip_reason comes from the SkipReason enum (closed set); still
        # render via st.markdown's text path, not the HTML path.
        st.markdown(f"Reason: `{skip_reason}`")
    if info_bits:
        st.markdown("  ".join(info_bits))


# ---------------------------------------------------------------------------
# Sidebar


def _sidebar() -> dict[str, Any]:
    lang_options = supported_languages()
    lang_keys = [k for k, _ in lang_options]
    current_lang = st.session_state.get("lang", "zh")
    try:
        idx = lang_keys.index(current_lang)
    except ValueError:
        idx = 0
    chosen_label = st.sidebar.selectbox(
        "🌐 Language / 语言",
        options=[label for _, label in lang_options],
        index=idx,
    )
    chosen_key = next(k for k, lbl in lang_options if lbl == chosen_label)
    st.session_state["lang"] = chosen_key
    lang = chosen_key

    st.sidebar.title(t("sidebar.title", lang))
    st.sidebar.caption(t("sidebar.version", lang, version=__version__))

    st.sidebar.subheader(t("sidebar.adapters", lang))
    fred_key = st.sidebar.text_input(
        t("sidebar.fred_label", lang),
        type="password",
        value=os.environ.get("FRED_API_KEY", ""),
        help=t("sidebar.fred_help", lang),
    )
    user_agent = st.sidebar.text_input(
        t("sidebar.sec_label", lang),
        value=os.environ.get("OPTAGENT_USER_AGENT", ""),
        placeholder="optagent/0.x (you@example.com)",
        help=t("sidebar.sec_help", lang),
    )

    st.sidebar.subheader(t("sidebar.llm_header", lang))
    enable_llm = st.sidebar.checkbox(t("sidebar.enable_llm", lang), value=True)
    auto_label = t("sidebar.auto_detect", lang)
    provider = st.sidebar.selectbox(
        t("sidebar.provider", lang),
        options=(auto_label, "openrouter", "anthropic", "openai", "gemini"),
        index=0,
        disabled=not enable_llm,
    )
    enable_ml = st.sidebar.checkbox(t("sidebar.enable_ml", lang), value=False)
    use_moomoo = st.sidebar.checkbox(
        t("sidebar.use_moomoo", lang), value=True, help=t("sidebar.moomoo_help", lang)
    )

    return {
        "lang": lang,
        "fred_key": fred_key,
        "user_agent": user_agent,
        "enable_llm": enable_llm,
        "provider": (None if provider == auto_label else provider),
        "enable_ml": enable_ml,
        "use_moomoo": use_moomoo,
    }


# ---------------------------------------------------------------------------
# Tab 1: Single-ticker analyze


def _tab_analyze(sidebar_opts: dict[str, Any]) -> None:
    lang = sidebar_opts.get("lang", "en")
    st.header(t("analyze.header", lang))
    st.caption(t("analyze.caption", lang))

    # A drill-down from the screen view prefills + auto-runs once.
    auto_ticker = _consume_drilldown("analyze")
    if auto_ticker:
        st.session_state["analyze_ticker"] = auto_ticker
        st.info(t("drill.autorun_note", lang, ticker=auto_ticker))
    elif "analyze_ticker" not in st.session_state:
        st.session_state["analyze_ticker"] = st.session_state.get("active_ticker") or "AAPL"

    col_l, col_r = st.columns([2, 1])
    with col_l:
        ticker = st.text_input(
            t("analyze.ticker_label", lang), max_chars=10, key="analyze_ticker"
        ).upper().strip()
    with col_r:
        horizon = st.number_input(t("analyze.horizon_label", lang), min_value=1, max_value=120, value=14)

    max_loss = st.number_input(
        t("analyze.max_loss_label", lang), min_value=0.0, value=0.0, step=100.0,
    )
    max_loss_v = max_loss if max_loss > 0 else None

    if not (st.button(t("analyze.run_btn", lang), type="primary") or auto_ticker):
        return

    # Codex web-audit fix: DO NOT write sidebar secrets to os.environ.
    # In a multi-user Streamlit deployment process-global env mutation leaks
    # across sessions. Pass credentials directly to adapter constructors.

    # Lazy imports so the page renders before yfinance / anthropic import cost.
    from optagent.orchestrator import analyze
    from optagent.profiles import ensure_default_profiles
    from optagent.registry import ProviderRegistry

    registry = ProviderRegistry()
    ensure_default_profiles(registry)

    fred_adapter = None
    if sidebar_opts["fred_key"]:
        try:
            from optagent.adapters import FREDAdapter
            fred_adapter = FREDAdapter(registry, api_key=sidebar_opts["fred_key"])
        except Exception as e:  # noqa: BLE001
            st.warning(f"FRED adapter unavailable: {e}")
    sec_adapter = None
    if sidebar_opts["user_agent"]:
        try:
            from optagent.adapters import SECEdgarAdapter
            sec_adapter = SECEdgarAdapter(registry, user_agent=sidebar_opts["user_agent"])
        except Exception as e:  # noqa: BLE001
            st.warning(f"SEC EDGAR adapter unavailable: {e}")
    news_adapter = None
    try:
        from optagent.adapters import YahooNewsAdapter
        news_adapter = YahooNewsAdapter(registry)
    except Exception:  # noqa: BLE001
        pass
    ml_adapter = None
    if sidebar_opts["enable_ml"]:
        from optagent.ml import MLDirectionAdapter
        ml_adapter = MLDirectionAdapter()

    moomoo_adapter = None
    if sidebar_opts.get("use_moomoo"):
        try:
            from optagent.adapters import MoomooAdapter
            moomoo_adapter = MoomooAdapter(registry)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Moomoo adapter unavailable: {e}")

    llm_client = model = price_table = ttl_table = None
    if sidebar_opts["enable_llm"]:
        try:
            from optagent.config_loader import load_bundle
            from optagent.llm import make_client_from_env

            bundle = load_bundle()
            price_table = bundle.price_table
            ttl_table = bundle.ttl_table
            llm_client, _provider, model = make_client_from_env(
                provider=sidebar_opts["provider"]
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"LLM unavailable: {e}")
            st.stop()

    with st.spinner(t("analyze.spinner", lang, ticker=ticker)):
        result = analyze(
            ticker,
            registry=registry,
            moomoo_adapter=moomoo_adapter,
            fred_adapter=fred_adapter,
            sec_edgar_adapter=sec_adapter,
            news_adapter=news_adapter,
            ml_direction_adapter=ml_adapter,
            horizon_days=int(horizon),
            max_loss_usd=max_loss_v,
            enable_llm=sidebar_opts["enable_llm"],
            llm_client=llm_client,
            model_version=model,
            price_table=price_table,
            ttl_table=ttl_table,
            lang=lang,
        )
    if moomoo_adapter is not None:
        moomoo_adapter.close()

    # Write a normalized analysis snapshot into the research store so the chat
    # panel can ground on it (compact, JSON-serializable — not rendered markup).
    verdict_dump = (
        result.verdict.model_dump(mode="json") if result.verdict is not None else None
    )
    cand_dumps = [
        c.model_dump(mode="json") for c in (result.screener_candidates or [])
    ]
    env_summary = [
        {
            "tool_call_id": getattr(e, "tool_call_id", None),
            "source": e.source,
            "confidence": e.confidence.value,
            "delay_assumption": getattr(e, "delay_assumption", None),
            "as_of": e.as_of.isoformat() if e.as_of else None,
            "warnings": list(getattr(e, "warnings", []) or []),
        }
        for e in (result.envelopes or [])
    ]
    store = _store()
    store["analysis"][ticker] = rs.analysis_snapshot(
        ticker,
        verdict_dump,
        cand_dumps,
        _now_iso(),
        inputs={"ticker": ticker, "horizon_days": int(horizon), "max_loss_usd": max_loss_v},
        envelopes=env_summary,
    )
    if result.ml_signal:
        store["ml"][ticker] = rs.ml_snapshot(ticker, dict(result.ml_signal), _now_iso())
    store["active_ticker"] = ticker

    _verdict_card(verdict_badge(result.verdict))
    if result.verdict.primary_reasons:
        st.markdown(t("analyze.primary_reasons", lang))
        for r in result.verdict.primary_reasons:
            # st.markdown auto-escapes by default; reasons are rendered as
            # plain text rather than HTML.
            st.markdown(f"- {r}")

    # 60-day candle chart with EMA20/EMA50 overlays. Best-effort: silently
    # skip if yfinance history fetch fails.
    try:
        import yfinance as yf  # noqa: WPS433
        import plotly.graph_objects as go

        hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=False)
        candle_df = candle_chart(hist, max_rows=60)
        if not candle_df.empty:
            st.subheader(t("analyze.candle_title", lang))
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=candle_df.index,
                        open=candle_df["Open"],
                        high=candle_df["High"],
                        low=candle_df["Low"],
                        close=candle_df["Close"],
                        name=ticker,
                        increasing_line_color="#16a34a",
                        decreasing_line_color="#dc2626",
                    )
                ]
            )
            if "EMA20" in candle_df.columns:
                fig.add_scatter(
                    x=candle_df.index, y=candle_df["EMA20"],
                    mode="lines", name="EMA20", line=dict(width=1.5, color="#0ea5e9"),
                )
                fig.add_scatter(
                    x=candle_df.index, y=candle_df["EMA50"],
                    mode="lines", name="EMA50", line=dict(width=1.5, color="#a78bfa"),
                )
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_rangeslider_visible=False,
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:  # noqa: BLE001
        pass

    # Codex web-audit fix: read envelopes / candidates / ml_signal directly
    # from the AnalyzeResult instead of re-opening the shared ledger file
    # (which suffered a multi-user race condition where another user's
    # concurrent run could land between our write and our read).
    if result.envelopes:
        st.subheader(t("analyze.envelopes_title", lang))
        st.dataframe(envelope_summary(result.envelopes), use_container_width=True, hide_index=True)

    if result.screener_candidates:
        st.subheader(t("analyze.candidates_title", lang, n=len(result.screener_candidates)))
        st.dataframe(
            candidate_table(result.screener_candidates),
            use_container_width=True,
            hide_index=True,
        )

        # IV smile chart from the chain envelope rows.
        chain_env = next(
            (
                e
                for e in result.envelopes
                if e.source == "yfinance"
                and isinstance(e.value, dict)
                and (e.value.get("rows") if e.value else None)
            ),
            None,
        )
        if chain_env:
            smile_df = iv_smile_frame(chain_env.value["rows"])
            if not smile_df.empty:
                import plotly.express as px

                st.subheader(t("analyze.smile_title", lang))
                fig = px.line(
                    smile_df,
                    x="strike",
                    y="iv",
                    color="right",
                    markers=True,
                    height=320,
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis_tickformat=".0%",
                )
                st.plotly_chart(fig, use_container_width=True)

    if result.ml_signal:
        _render_ml_gauge(result.ml_signal)

    st.subheader(t("analyze.memo_title", lang))
    st.code(result.memo, language="text")


def _render_ml_gauge(ml_signal: dict[str, Any]) -> None:
    lang = _current_lang()
    info = ml_signal_gauge(ml_signal)
    if not info.get("available"):
        return
    st.subheader(t("ml.gauge_title", lang))
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=info["prob_up"] * 100,
            number={"suffix": " %", "valueformat": ".1f"},
            title={"text": f"prob_up ({info['class_label']})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0ea5e9"},
                "steps": [
                    {"range": [0, 45], "color": "#fee2e2"},
                    {"range": [45, 55], "color": "#e5e7eb"},
                    {"range": [55, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.85,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=30, r=30, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(info["subtitle"])

    snap_df = feature_radar(info.get("feature_snapshot") or {})
    if not snap_df.empty:
        with st.expander(t("ml.feature_snapshot", lang)):
            st.dataframe(snap_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: Cross-ticker screen


def _signal_to_dict(sig: Any) -> dict[str, Any]:
    return sig.to_dict() if hasattr(sig, "to_dict") else dict(sig)


def _build_universe(sector: str, sector_any: str) -> list[str]:
    from optagent.strategies import builtin_us_large_cap, filter_to_sector

    universe = builtin_us_large_cap()
    if sector != sector_any:
        universe = filter_to_sector(universe, sector)
    return universe


def _run_multi_strategy(strategy_ids: list[str], universe: list[str]):
    """Run each strategy sequentially with per-strategy error isolation.

    Per-strategy execution is isolated by `research_store.run_strategies`, so
    one strategy raising never aborts the others. Each strategy is run with an
    UNBOUNDED top-N (the full universe) so EVERY triggered row is preserved for
    cross-strategy synthesis — the user's Top-N is applied only after
    aggregation (and to the per-strategy display tables), so a ticker ranked
    N+1 within a strategy can still win on resonance. Returns
    results_by_strategy: {sid: {signals, n_triggered, n_evaluated,
    stale_tickers, stale_bars, near_misses, error}}.
    """

    from optagent.strategies import get_strategy, screen_universe
    from optagent.strategies.screen import make_default_fetcher

    full_n = max(len(universe), 1)  # never truncate triggered rows pre-synthesis

    # Build ONE fetcher, memoized per ticker, and share it across every
    # strategy so the same Yahoo history/chain is downloaded once — not once
    # per strategy (which tripled traffic + rate-limit risk for 3 built-ins).
    _base_fetcher = make_default_fetcher()
    _cache: dict[str, Any] = {}

    def shared_fetcher(ticker: str):
        if ticker not in _cache:
            _cache[ticker] = _base_fetcher(ticker)
        return _cache[ticker]

    def run_one(sid: str) -> dict[str, Any]:
        if not universe:
            return {"error": "empty_universe", "signals": []}
        res = screen_universe(get_strategy(sid), universe, fetcher=shared_fetcher, top_n=full_n)
        return {
            "error": None,
            "signals": [_signal_to_dict(s) for s in res.top_signals],
            "n_triggered": res.n_triggered,
            "n_evaluated": res.n_evaluated,
            "stale_tickers": [tk for (tk, _d, _n) in res.stale_bars],
            "stale_bars": [[tk, d, n] for (tk, d, n) in res.stale_bars],
            "near_misses": [_signal_to_dict(s) for s in res.top_near_misses],
        }

    return rs.run_strategies(strategy_ids, run_one)


def _tab_screen(lang: str = "en") -> None:
    st.header(t("screen.header", lang))
    st.caption(t("screen.caption", lang))

    from optagent.strategies import list_sectors, list_strategy_ids

    sector_any = t("screen.sector_any", lang)
    all_strategies = list_strategy_ids()
    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_ids = st.multiselect(
            t("screen.multiselect_label", lang),
            options=all_strategies,
            default=all_strategies[:1],
        )
    with col2:
        sector = st.selectbox(t("screen.sector_label", lang), options=[sector_any] + list_sectors())
    with col3:
        limit = st.number_input(t("screen.limit_label", lang), min_value=1, max_value=20, value=5)

    def _fail_screen(inputs: dict[str, Any]) -> None:
        # A failed/empty screen attempt must overwrite prior successful grounding
        # so the chat panel never treats stale results as the latest screen.
        st.session_state["last_screen"] = None
        _store()["screen"] = rs.screen_snapshot({}, None, _now_iso(), inputs=inputs)

    if st.button(t("screen.run_btn", lang), type="primary"):
        if not strategy_ids:
            _fail_screen({"strategies": [], "sector": sector, "limit": int(limit)})
            st.warning(t("screen.select_prompt", lang))
        else:
            universe = _build_universe(sector, sector_any)
            if not universe:
                _fail_screen({"strategies": strategy_ids, "sector": sector, "limit": int(limit)})
                st.warning(t("screen.sector_empty_warning", lang, sector=sector))
            else:
                with st.spinner(
                    t("screen.spinner", lang, strategy=", ".join(strategy_ids), n=len(universe))
                ):
                    results = _run_multi_strategy(strategy_ids, universe)
                # Synthesis sees the FULL triggered set; Top-N applies after.
                synthesis = rs.synthesise_cross_strategy(results, top_n=int(limit))
                st.session_state["screen_display_limit"] = int(limit)
                # Persist so reruns (e.g. the Explain button) keep the data, and
                # the chat panel can ground on it.
                st.session_state["last_screen"] = {"results": results, "synthesis": synthesis}
                _store()["screen"] = rs.screen_snapshot(
                    results,
                    synthesis,
                    _now_iso(),
                    inputs={"strategies": strategy_ids, "sector": sector, "limit": int(limit)},
                )

    snap = st.session_state.get("last_screen")
    if not snap:
        return
    results = snap["results"]
    synthesis = snap["synthesis"]

    # --- Cross-strategy synthesis (deterministic ranking) ---
    st.subheader(t("screen.synthesis_title", lang))
    st.caption(t("screen.synthesis_caption", lang))
    if not synthesis:
        st.info(t("screen.no_synthesis", lang))
    else:
        syn_df = pd.DataFrame(
            [
                {
                    "Ticker": p.get("ticker"),
                    t("screen.col_resonance", lang): p.get("resonance"),
                    t("screen.col_score", lang): p.get("combined_score"),
                    t("screen.col_support", lang): ", ".join(p.get("supporting") or []),
                }
                for p in synthesis
            ]
        )
        st.dataframe(syn_df, use_container_width=True, hide_index=True)

        # One-click drill-down per pick.
        for p in synthesis:
            tk = p.get("ticker")
            if not tk:
                continue
            c0, c1, c2 = st.columns([2, 1, 1])
            c0.markdown(f"**{tk}** · {t('screen.col_resonance', lang)}={p.get('resonance')}")
            if c1.button(t("screen.drill_analyze", lang), key=f"drill_an_{tk}"):
                _drilldown(tk, "analyze")
            if c2.button(t("screen.drill_ml", lang), key=f"drill_ml_{tk}"):
                _drilldown(tk, "ml")

    # Opt-in LLM explanation — available after ANY completed screen, even when
    # the deterministic synthesis is empty but per-strategy results exist.
    if st.button(t("screen.explain_btn", lang), key="explain_screen_btn"):
        try:
            from optagent.web.screen_llm import explain_screen

            with st.spinner(t("screen.explain_spinner", lang)):
                prose = explain_screen(
                    _store().get("screen"),
                    lang=lang,
                    provider=st.session_state.get("provider_override"),
                )
            st.subheader(t("screen.explain_title", lang))
            st.markdown(prose)
        except Exception as e:  # noqa: BLE001 — provider SDKs raise many error types
            # Timeouts, rate limits, and API errors are NOT RuntimeError; render
            # the localized error instead of crashing the interaction.
            if isinstance(e, RuntimeError) and "No LLM provider configured" in str(e):
                st.error(t("chat.no_llm", lang))
            else:
                st.error(t("chat.error", lang, err=str(e)))

    # --- Per-strategy result tables + diagnostics ---
    # Synthesis ran on the full triggered set above; the per-strategy tables
    # show only the user's Top-N for readability.
    display_limit = int(st.session_state.get("screen_display_limit", 5))
    st.subheader(t("screen.per_strategy_title", lang))
    import plotly.express as px

    for sid, res in results.items():
        with st.expander(f"{sid}", expanded=len(results) == 1):
            if res.get("error"):
                st.error(t("screen.strategy_error", lang, sid=sid, err=res["error"]))
                continue
            c_l, c_r = st.columns(2)
            c_l.metric(t("screen.metric_evaluated", lang), res.get("n_evaluated"))
            c_r.metric(t("screen.metric_triggered", lang), res.get("n_triggered"))

            stale_bars = res.get("stale_bars") or []
            if stale_bars:
                st.warning(t("screen.stale_warning", lang, n=len(stale_bars)))
                with st.expander(t("screen.stale_details", lang)):
                    st.dataframe(
                        pd.DataFrame(stale_bars, columns=["ticker", "last_bar", "trading_days_behind"]),
                        use_container_width=True, hide_index=True,
                    )

            sigs = (res.get("signals") or [])[:display_limit]
            if sigs:
                df = strategy_signal_table(sigs)
                st.dataframe(df, use_container_width=True, hide_index=True)
                if "Ticker" in df and "Score" in df and not df.empty:
                    fig = px.bar(df, x="Ticker", y="Score", color="Direction", height=260)
                    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(t("screen.no_trigger", lang))

            nm = (res.get("near_misses") or [])[:display_limit]
            if nm:
                with st.expander(t("screen.near_misses_expander", lang, n=len(nm))):
                    st.dataframe(
                        strategy_signal_table(nm), use_container_width=True, hide_index=True
                    )


def _drilldown(ticker: str, target: str) -> None:
    """Queue a one-shot drill-down and navigate to the target view.

    Sets only plain session vars (not the radio's widget key) plus a `nav_to`
    marker that `_view_selector` applies before the radio instantiates next
    run. Avoids the "modify widget value after instantiation" error.
    """

    st.session_state["active_ticker"] = ticker.upper()
    st.session_state["pending_drilldown"] = {"ticker": ticker.upper(), "target": target}
    st.session_state["nav_to"] = target
    st.rerun()


# ---------------------------------------------------------------------------
# Tab 3: ML direction model


def _tab_ml(lang: str = "en") -> None:
    st.header(t("ml.header", lang))
    st.caption(t("ml.caption", lang))

    auto_ticker = _consume_drilldown("ml")
    if auto_ticker:
        st.session_state["ml_ticker"] = auto_ticker
        st.info(t("drill.autorun_note", lang, ticker=auto_ticker))
    elif "ml_ticker" not in st.session_state:
        st.session_state["ml_ticker"] = st.session_state.get("active_ticker") or "AAPL"

    ticker = st.text_input(
        t("ml.ticker_label", lang), max_chars=10, key="ml_ticker"
    ).upper().strip()
    if not (st.button(t("ml.run_btn", lang), type="primary", key="ml_run") or auto_ticker):
        return

    from optagent.ml import MLDirectionAdapter

    with st.spinner(t("ml.spinner", lang, ticker=ticker)):
        adapter = MLDirectionAdapter()
        sig = adapter.signal(ticker)
    if sig is None:
        # Overwrite any prior successful snapshot so the chat panel never grounds
        # on stale success as if it were the latest result.
        _store()["ml"][ticker] = rs.ml_snapshot(ticker, None, _now_iso())
        _store()["active_ticker"] = ticker
        st.error(t("ml.unavailable", lang))
        return
    sig_dict = sig.to_dict()
    _store()["ml"][ticker] = rs.ml_snapshot(ticker, sig_dict, _now_iso())
    _store()["active_ticker"] = ticker
    _render_ml_gauge(sig_dict)


# ---------------------------------------------------------------------------
# Entry point


def _tab_ledger(lang: str = "en") -> None:
    st.header(t("ledger.header", lang))
    st.caption(t("ledger.caption", lang))
    col1, col2 = st.columns([1, 1])
    with col1:
        days_back = st.slider(t("ledger.days_back_label", lang), min_value=1, max_value=30, value=7)
    with col2:
        ledger_dir = st.text_input(
            t("ledger.dir_label", lang), value="data/ledger",
            help="Override if you ran with --ledger-dir.",
        )

    from pathlib import Path as _P

    ledger_inputs = {"days_back": int(days_back), "ledger_dir": ledger_dir}
    df = ledger_index(_P(ledger_dir), days_back=int(days_back))
    if df.empty:
        _store()["ledger"] = rs.ledger_summary_snapshot(None, 0, _now_iso(), inputs=ledger_inputs)
        st.info(t("ledger.empty", lang))
        return

    st.markdown(t("ledger.count", lang, n=len(df)))
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Verdict distribution pie chart.
    import plotly.express as px

    counts = df["action"].value_counts().rename_axis("action").reset_index(name="count")
    if not counts.empty:
        fig = px.pie(counts, names="action", values="count", height=320)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.subheader(t("ledger.pie_title", lang))
        st.plotly_chart(fig, use_container_width=True)

    # Record an aggregate ledger summary for chat grounding (counts only).
    _store()["ledger"] = rs.ledger_summary_snapshot(
        {str(r["action"]): int(r["count"]) for _i, r in counts.iterrows()},
        len(df),
        _now_iso(),
        inputs=ledger_inputs,
    )


def _chat_panel(sidebar_opts: dict[str, Any]) -> None:
    """Persistent right-side chat panel grounded on ALL view results.

    The grounding context is assembled from the session research store (screen
    / analyze / ML / ledger snapshots), so the chat reflects the latest results
    across every view — not just the single-ticker analysis.
    """

    lang = sidebar_opts.get("lang", "en")
    st.subheader(t("chat.header", lang))
    st.caption(t("chat.panel_grounding", lang))

    store = _store()
    context_block = rs.build_context(store, lang, now_iso=_now_iso())

    history: list[ChatMessage] = st.session_state.setdefault("chat_history", [])
    if st.button(t("chat.clear_btn", lang), use_container_width=True, key="chat_clear"):
        st.session_state["chat_history"] = []
        history = st.session_state["chat_history"]

    for m in history:
        with st.chat_message(m.role):
            st.markdown(m.content)

    user_input = st.chat_input(t("chat.placeholder", lang))
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)
    history.append(ChatMessage(role="user", content=user_input))

    try:
        with st.chat_message("assistant"):
            with st.spinner(t("chat.spinner", lang)):
                reply = chat_complete(
                    history=history[:-1],
                    user_message=user_input,
                    context_bundle=None,
                    context_block=context_block,
                    lang=lang,
                    provider=sidebar_opts.get("provider"),
                    disclaimer=DISCLAIMER,
                )
            st.markdown(reply)
        history.append(ChatMessage(role="assistant", content=reply))
        st.session_state["chat_history"] = history
    except Exception as e:  # noqa: BLE001 — provider SDKs raise many error types
        msg = str(e)
        if isinstance(e, RuntimeError) and "No LLM provider configured" in msg:
            st.error(t("chat.no_llm", lang))
        else:
            st.error(t("chat.error", lang, err=msg))
        history.pop()
        st.session_state["chat_history"] = history


def _view_selector(lang: str) -> str:
    """Session-state-backed view selector (replaces st.tabs so drill-down can
    navigate programmatically). Market screen is first.

    Programmatic navigation uses a `nav_to` marker consumed HERE, before the
    radio widget is instantiated — Streamlit forbids mutating a widget-keyed
    session value after the widget renders, so the drill-down handler only sets
    `nav_to` + reruns, and this function applies it pre-instantiation.
    """

    nav_to = st.session_state.pop("nav_to", None)
    if nav_to in _VIEWS:
        st.session_state["view_radio"] = nav_to
    chosen = st.radio(
        t("nav.label", lang),
        options=_VIEWS,
        horizontal=True,
        format_func=lambda v: t(_VIEW_LABEL_KEY[v], lang),
        key="view_radio",
    )
    st.session_state["active_view"] = chosen
    return chosen


def main() -> None:
    opts = _sidebar()
    lang = opts["lang"]
    # Stash provider override so the screen "Explain" button can reach it.
    st.session_state["provider_override"] = opts.get("provider")

    main_col, chat_col = st.columns([3, 1])
    with main_col:
        _disclaimer_banner()
        view = _view_selector(lang)
        if view == "screen":
            _tab_screen(lang)
        elif view == "analyze":
            _tab_analyze(opts)
        elif view == "ml":
            _tab_ml(lang)
        elif view == "ledger":
            _tab_ledger(lang)
    with chat_col:
        _chat_panel(opts)


# Streamlit invokes the top-level module — pandas imports need to be local
# to avoid heavy startup cost in tests that just import optagent.web.
import pandas as pd  # noqa: E402  (used by tab handlers via strategy_signal_table)


if __name__ == "__main__":
    main()
else:
    # When Streamlit imports this module as the script body, calling main()
    # here is what actually draws the page. The if-name guard above keeps
    # `python -m optagent.web.app` ergonomic for debugging.
    main()
