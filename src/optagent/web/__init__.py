"""Streamlit-based web UI for optagent.

Local-first: `optagent-ui` starts a Streamlit server on localhost.
Designed to port cleanly to a hosted Streamlit Cloud or any container
runtime without code changes.

The UI is a thin presentation layer over the existing `analyze()` /
`screen_universe()` / `MLDirectionAdapter` APIs — it does NOT add any
verdict logic, and the same fail-closed validator runs every call.

> RESEARCH ONLY — NOT FINANCIAL ADVICE.
"""

from .components import (
    DISCLAIMER_BANNER,
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

__all__ = [
    "DISCLAIMER_BANNER",
    "candidate_table",
    "candle_chart",
    "envelope_summary",
    "feature_radar",
    "iv_smile_frame",
    "ledger_index",
    "ml_signal_gauge",
    "strategy_signal_table",
    "verdict_badge",
]
