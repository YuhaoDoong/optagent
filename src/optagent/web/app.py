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

from .. import DISCLAIMER, __version__
from .components import (
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


# ---------------------------------------------------------------------------
# Page-level config (run once)

st.set_page_config(
    page_title="optagent — US equity options research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _disclaimer_banner() -> None:
    st.markdown(
        f"<div style='background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;"
        f"padding:8px 12px;font-weight:600;color:#78350f;margin-bottom:12px;'>"
        f"⚠️ {DISCLAIMER} optagent v{__version__} is a research tool only.</div>",
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
    st.sidebar.title("optagent")
    st.sidebar.caption(f"v{__version__}")

    st.sidebar.subheader("Optional adapters")
    fred_key = st.sidebar.text_input(
        "FRED_API_KEY",
        type="password",
        value=os.environ.get("FRED_API_KEY", ""),
        help="Set to enable macro context (10y/2y yields, VIX, CPI, Fed funds, USD index).",
    )
    user_agent = st.sidebar.text_input(
        "SEC EDGAR User-Agent",
        value=os.environ.get("OPTAGENT_USER_AGENT", ""),
        placeholder="optagent/0.x (you@example.com)",
        help="SEC EDGAR REQUIRES a contact email. Adapter fails closed without one.",
    )

    st.sidebar.subheader("LLM (optional)")
    enable_llm = st.sidebar.checkbox("Enable LLM synthesis", value=False)
    provider = st.sidebar.selectbox(
        "Provider",
        options=("auto-detect", "anthropic", "openai", "gemini"),
        index=0,
        disabled=not enable_llm,
    )

    enable_ml = st.sidebar.checkbox(
        "Enable ML direction signal (Alt-3 v0)", value=False
    )

    return {
        "fred_key": fred_key,
        "user_agent": user_agent,
        "enable_llm": enable_llm,
        "provider": (None if provider == "auto-detect" else provider),
        "enable_ml": enable_ml,
    }


# ---------------------------------------------------------------------------
# Tab 1: Single-ticker analyze


def _tab_analyze(sidebar_opts: dict[str, Any]) -> None:
    st.header("📊 Single-ticker analysis")
    st.caption(
        "Equivalent to running `optagent analyze <ticker>` from the CLI. Outputs a "
        "structured research memo with the same disclaimer-first contract."
    )

    col_l, col_r = st.columns([2, 1])
    with col_l:
        ticker = st.text_input("Ticker", value="AAPL", max_chars=10).upper().strip()
    with col_r:
        horizon = st.number_input("Horizon (days)", min_value=1, max_value=120, value=14)

    max_loss = st.number_input(
        "Max-loss budget (USD, optional)", min_value=0.0, value=0.0, step=100.0,
    )
    max_loss_v = max_loss if max_loss > 0 else None

    if not st.button("Analyze", type="primary"):
        return

    # Codex web-audit fix: DO NOT write sidebar secrets to os.environ.
    # In a multi-user Streamlit deployment process-global env mutation leaks
    # across sessions. Pass credentials directly to adapter constructors.

    # Lazy imports so the page renders before yfinance / anthropic import cost.
    from ..orchestrator import analyze
    from ..profiles import ensure_default_profiles
    from ..registry import ProviderRegistry

    registry = ProviderRegistry()
    ensure_default_profiles(registry)

    fred_adapter = None
    if sidebar_opts["fred_key"]:
        try:
            from ..adapters import FREDAdapter
            fred_adapter = FREDAdapter(registry, api_key=sidebar_opts["fred_key"])
        except Exception as e:  # noqa: BLE001
            st.warning(f"FRED adapter unavailable: {e}")
    sec_adapter = None
    if sidebar_opts["user_agent"]:
        try:
            from ..adapters import SECEdgarAdapter
            sec_adapter = SECEdgarAdapter(registry, user_agent=sidebar_opts["user_agent"])
        except Exception as e:  # noqa: BLE001
            st.warning(f"SEC EDGAR adapter unavailable: {e}")
    news_adapter = None
    try:
        from ..adapters import YahooNewsAdapter
        news_adapter = YahooNewsAdapter(registry)
    except Exception:  # noqa: BLE001
        pass
    ml_adapter = None
    if sidebar_opts["enable_ml"]:
        from ..ml import MLDirectionAdapter
        ml_adapter = MLDirectionAdapter()

    llm_client = model = price_table = ttl_table = None
    if sidebar_opts["enable_llm"]:
        try:
            from ..config_loader import load_bundle
            from ..llm import make_client_from_env

            bundle = load_bundle()
            price_table = bundle.price_table
            ttl_table = bundle.ttl_table
            llm_client, _provider, model = make_client_from_env(
                provider=sidebar_opts["provider"]
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"LLM unavailable: {e}")
            st.stop()

    with st.spinner(f"Running analysis on {ticker}..."):
        result = analyze(
            ticker,
            registry=registry,
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
        )

    _verdict_card(verdict_badge(result.verdict))
    if result.verdict.primary_reasons:
        st.markdown("**Primary reasons:**")
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
            st.subheader("Recent price action (60d)")
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
        st.subheader("Upstream envelopes")
        st.dataframe(envelope_summary(result.envelopes), use_container_width=True, hide_index=True)

    if result.screener_candidates:
        st.subheader(f"Screener candidates ({len(result.screener_candidates)})")
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

                st.subheader("Options chain IV smile")
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

    st.subheader("Memo (text)")
    st.code(result.memo, language="text")


def _render_ml_gauge(ml_signal: dict[str, Any]) -> None:
    info = ml_signal_gauge(ml_signal)
    if not info.get("available"):
        return
    st.subheader("ML direction signal")
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
        with st.expander("Feature snapshot"):
            st.dataframe(snap_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: Cross-ticker screen


def _tab_screen() -> None:
    st.header("🔭 Cross-ticker screen")
    st.caption(
        "Equivalent to `optagent screen --strategy ... --sector ...`. The "
        "framework runs a quant strategy across the universe and ranks the "
        "top candidates by score."
    )

    from ..strategies import list_sectors, list_strategy_ids

    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_id = st.selectbox("Strategy", options=list_strategy_ids())
    with col2:
        sector = st.selectbox(
            "Sector (optional)",
            options=["(any)"] + list_sectors(),
        )
    with col3:
        limit = st.number_input("Top N", min_value=1, max_value=20, value=5)

    if not st.button("Run screen", type="primary"):
        return

    from ..strategies import (
        builtin_us_large_cap,
        filter_to_sector,
        get_strategy,
        screen_universe,
    )

    universe = builtin_us_large_cap()
    if sector != "(any)":
        universe = filter_to_sector(universe, sector)
        if not universe:
            st.warning(f"Sector '{sector}' has no overlap with the built-in universe.")
            return

    strategy = get_strategy(strategy_id)
    with st.spinner(f"Running {strategy_id} across {len(universe)} tickers..."):
        result = screen_universe(strategy, universe, top_n=int(limit))

    col_l, col_r = st.columns(3)
    col_l.metric("Universe size", result.universe_size)
    col_l.metric("Evaluated", result.n_evaluated)
    col_r.metric("Triggered", result.n_triggered, delta=None)
    col_r.metric("Top near-misses", len(result.top_near_misses))

    if result.stale_bars:
        st.warning(
            f"{len(result.stale_bars)} ticker(s) had stale OHLCV bars (US market "
            "holidays / long weekends). They were still evaluated; verify before action."
        )
        with st.expander("Stale-bar details"):
            stale_df = pd.DataFrame(
                result.stale_bars, columns=["ticker", "last_bar", "trading_days_behind"]
            )
            st.dataframe(stale_df, use_container_width=True, hide_index=True)

    if result.top_signals:
        st.subheader("Top candidates")
        df = strategy_signal_table(result.top_signals)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar chart of scores.
        import plotly.express as px

        fig = px.bar(df, x="Ticker", y="Score", color="Direction", height=300)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No tickers triggered the strategy.")

    if result.top_near_misses:
        with st.expander(f"Near-misses ({len(result.top_near_misses)})"):
            nm_df = strategy_signal_table(result.top_near_misses)
            st.dataframe(nm_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: ML direction model


def _tab_ml() -> None:
    st.header("🧠 ML direction model")
    st.caption(
        "Per-ticker `GradientBoostingClassifier` with walk-forward OOS validation, "
        "Wilson 95% CI, and class-baseline comparison. Output is INFORMATIONAL — the "
        "fail-closed validator does not let it become the sole reason for a non-SKIP "
        "verdict."
    )

    ticker = st.text_input("Ticker", value="AAPL", max_chars=10, key="ml_ticker").upper().strip()
    if not st.button("Compute signal", type="primary", key="ml_run"):
        return

    from ..ml import MLDirectionAdapter

    with st.spinner(f"Training / loading model for {ticker}..."):
        adapter = MLDirectionAdapter()
        sig = adapter.signal(ticker)
    if sig is None:
        st.error("ML signal unavailable (no yfinance history or invalid ticker).")
        return
    _render_ml_gauge(sig.to_dict())


# ---------------------------------------------------------------------------
# Entry point


def _tab_ledger() -> None:
    st.header("📒 Audit ledger viewer")
    st.caption(
        "Browse recent runs persisted to `data/ledger/YYYY-MM-DD.jsonl`. Useful "
        "for post-hoc inspection without re-running."
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        days_back = st.slider("Days back", min_value=1, max_value=30, value=7)
    with col2:
        ledger_dir = st.text_input(
            "Ledger directory", value="data/ledger",
            help="Override if you ran with --ledger-dir.",
        )

    from pathlib import Path as _P

    df = ledger_index(_P(ledger_dir), days_back=int(days_back))
    if df.empty:
        st.info(
            "No ledger rows found. Run `optagent analyze <ticker>` (CLI or this UI) "
            "to populate the ledger."
        )
        return

    st.markdown(f"**{len(df)}** recent runs found.")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Verdict distribution pie chart.
    import plotly.express as px

    counts = df["action"].value_counts().rename_axis("action").reset_index(name="count")
    if not counts.empty:
        fig = px.pie(counts, names="action", values="count", height=320)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.subheader("Verdict distribution")
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    opts = _sidebar()
    _disclaimer_banner()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Analyze ticker", "🔭 Screen market", "🧠 ML signal", "📒 Ledger"]
    )
    with tab1:
        _tab_analyze(opts)
    with tab2:
        _tab_screen()
    with tab3:
        _tab_ml()
    with tab4:
        _tab_ledger()


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
