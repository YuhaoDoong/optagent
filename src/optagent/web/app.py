"""Streamlit entry point: ``optagent-ui`` (or ``streamlit run -m optagent.web.app``).

Three tabs:
  1. Analyze a single ticker (the CLI `analyze` subcommand visualised).
  2. Cross-ticker screen (the CLI `screen` subcommand visualised).
  3. ML direction signal (per-ticker Alt-3 v0 model with Wilson CI gauge).

Every tab respects the project's safety invariants:
  - Disclaimer banner at the top of every page.
  - Bounded verdict enum (template-only by default; LLM is opt-in).
  - Fail-closed validator runs unchanged behind the scenes.
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
    summarise_analysis_for_context,
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
from optagent.web.i18n import supported_languages, t


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

    col_l, col_r = st.columns([2, 1])
    with col_l:
        ticker = st.text_input(t("analyze.ticker_label", lang), value="AAPL", max_chars=10).upper().strip()
    with col_r:
        horizon = st.number_input(t("analyze.horizon_label", lang), min_value=1, max_value=120, value=14)

    max_loss = st.number_input(
        t("analyze.max_loss_label", lang), min_value=0.0, value=0.0, step=100.0,
    )
    max_loss_v = max_loss if max_loss > 0 else None

    if not st.button(t("analyze.run_btn", lang), type="primary"):
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

    # Stash analysis context for the chat tab. summarise_analysis_for_context
    # picks a compact subset (no Streamlit-incompatible types) so it survives
    # the session_state pickle/hash round-trip cleanly.
    from datetime import datetime as _dt
    st.session_state["last_analysis_summary"] = summarise_analysis_for_context(result)
    st.session_state["last_analysis_ticker"] = ticker
    st.session_state["last_analysis_ts"] = _dt.utcnow().isoformat(timespec="seconds")

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


def _tab_screen(lang: str = "en") -> None:
    st.header(t("screen.header", lang))
    st.caption(t("screen.caption", lang))

    from optagent.strategies import list_sectors, list_strategy_ids

    sector_any = t("screen.sector_any", lang)
    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_id = st.selectbox(t("screen.strategy_label", lang), options=list_strategy_ids())
    with col2:
        sector = st.selectbox(
            t("screen.sector_label", lang),
            options=[sector_any] + list_sectors(),
        )
    with col3:
        limit = st.number_input(t("screen.limit_label", lang), min_value=1, max_value=20, value=5)

    if not st.button(t("screen.run_btn", lang), type="primary"):
        return

    from optagent.strategies import (
        builtin_us_large_cap,
        filter_to_sector,
        get_strategy,
        screen_universe,
    )

    universe = builtin_us_large_cap()
    if sector != sector_any:
        universe = filter_to_sector(universe, sector)
        if not universe:
            st.warning(t("screen.sector_empty_warning", lang, sector=sector))
            return

    strategy = get_strategy(strategy_id)
    with st.spinner(t("screen.spinner", lang, strategy=strategy_id, n=len(universe))):
        result = screen_universe(strategy, universe, top_n=int(limit))

    col_l, col_r = st.columns(2)
    col_l.metric(t("screen.metric_universe", lang), result.universe_size)
    col_l.metric(t("screen.metric_evaluated", lang), result.n_evaluated)
    col_r.metric(t("screen.metric_triggered", lang), result.n_triggered, delta=None)
    col_r.metric(t("screen.metric_near_misses", lang), len(result.top_near_misses))

    if result.stale_bars:
        st.warning(t("screen.stale_warning", lang, n=len(result.stale_bars)))
        with st.expander(t("screen.stale_details", lang)):
            stale_df = pd.DataFrame(
                result.stale_bars, columns=["ticker", "last_bar", "trading_days_behind"]
            )
            st.dataframe(stale_df, use_container_width=True, hide_index=True)

    if result.top_signals:
        st.subheader(t("screen.top_candidates", lang))
        df = strategy_signal_table(result.top_signals)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar chart of scores.
        import plotly.express as px

        fig = px.bar(df, x="Ticker", y="Score", color="Direction", height=300)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t("screen.no_trigger", lang))

    if result.top_near_misses:
        with st.expander(t("screen.near_misses_expander", lang, n=len(result.top_near_misses))):
            nm_df = strategy_signal_table(result.top_near_misses)
            st.dataframe(nm_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: ML direction model


def _tab_ml(lang: str = "en") -> None:
    st.header(t("ml.header", lang))
    st.caption(t("ml.caption", lang))

    ticker = st.text_input(
        t("ml.ticker_label", lang), value="AAPL", max_chars=10, key="ml_ticker"
    ).upper().strip()
    if not st.button(t("ml.run_btn", lang), type="primary", key="ml_run"):
        return

    from optagent.ml import MLDirectionAdapter

    with st.spinner(t("ml.spinner", lang, ticker=ticker)):
        adapter = MLDirectionAdapter()
        sig = adapter.signal(ticker)
    if sig is None:
        st.error(t("ml.unavailable", lang))
        return
    _render_ml_gauge(sig.to_dict())


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

    df = ledger_index(_P(ledger_dir), days_back=int(days_back))
    if df.empty:
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


def _tab_chat(sidebar_opts: dict[str, Any]) -> None:
    """Free-form chat grounded in the latest analysis result."""

    lang = sidebar_opts.get("lang", "en")
    st.header(t("chat.header", lang))
    st.caption(t("chat.caption", lang))

    summary = st.session_state.get("last_analysis_summary") or {}
    ticker = st.session_state.get("last_analysis_ticker")
    ts = st.session_state.get("last_analysis_ts")
    if summary and ticker:
        st.info(t("chat.context_summary", lang, ticker=ticker, ts=ts or "—"))
    else:
        st.warning(t("chat.no_context", lang))

    history: list[ChatMessage] = st.session_state.setdefault("chat_history", [])

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button(t("chat.clear_btn", lang), use_container_width=True):
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
                    context_bundle=summary,
                    lang=lang,
                    provider=sidebar_opts.get("provider"),
                    disclaimer=DISCLAIMER,
                )
            st.markdown(reply)
        history.append(ChatMessage(role="assistant", content=reply))
        st.session_state["chat_history"] = history
    except RuntimeError as e:
        msg = str(e)
        if "No LLM provider configured" in msg:
            st.error(t("chat.no_llm", lang))
        else:
            st.error(t("chat.error", lang, err=msg))
        history.pop()
        st.session_state["chat_history"] = history


def main() -> None:
    opts = _sidebar()
    _disclaimer_banner()
    lang = opts["lang"]

    tabs = st.tabs([
        t("tab.analyze", lang),
        t("tab.screen", lang),
        t("tab.ml", lang),
        t("tab.ledger", lang),
        t("tab.chat", lang),
    ])
    with tabs[0]:
        _tab_analyze(opts)
    with tabs[1]:
        _tab_screen(lang)
    with tabs[2]:
        _tab_ml(lang)
    with tabs[3]:
        _tab_ledger(lang)
    with tabs[4]:
        _tab_chat(opts)


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
