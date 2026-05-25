from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from optagent import DISCLAIMER
from optagent.llm import (
    EMIT_VERDICT_TOOL,
    SYSTEM_PROMPT,
    build_user_prompt,
    build_verdict_from_tool_input,
    synthesise,
)
from optagent.schemas import (
    Confidence,
    Envelope,
    MarketSession,
    OptionContract,
    OptionRight,
    SkipReason,
    VerdictAction,
)


UTC_NOW = datetime.now(timezone.utc)


def _envelope(tcid: str = "tc-1") -> Envelope:
    return Envelope(
        value={"last": 190.0},
        as_of=UTC_NOW,
        source="yfinance",
        delay_assumption="delayed_15min",
        market_session=MarketSession.rth,
        confidence=Confidence.ok,
        provider_profile_id="yfinance_research",
        tool_call_id=tcid,
    )


def _contract(occ: str = "AAPL_C200", right: OptionRight = OptionRight.call) -> OptionContract:
    return OptionContract(
        occ_symbol=occ,
        underlying="AAPL",
        expiration=UTC_NOW + timedelta(days=21),
        strike=200.0,
        right=right,
        mid=2.50,
        bid=2.45,
        ask=2.55,
        spread_pct=0.04,
        oi=12000,
        volume=1500,
        delta=0.42,
        theta=-0.05,
        vega=0.18,
        iv=0.28,
        breakeven=202.50,
        max_loss=250.0,
        liquidity_score=0.85,
        data_quality_score=0.95,
    )


# ---------------------------------------------------------------------------
# Prompt assembly


def test_prompt_contains_candidate_occs_and_tcids():
    p = build_user_prompt(
        ticker="AAPL",
        spot=190.0,
        candidates=[_contract("AAPL_C200"), _contract("AAPL_C210")],
        envelopes=[_envelope("tc-a"), _envelope("tc-b")],
    )
    assert "AAPL_C200" in p
    assert "AAPL_C210" in p
    assert "tc-a" in p
    assert "tc-b" in p
    assert "candidate_occ_symbols" in p
    assert "available_tool_call_ids" in p


def test_prompt_wraps_news_in_delimiter_blocks():
    p = build_user_prompt(
        ticker="AAPL",
        spot=190.0,
        candidates=[_contract()],
        envelopes=[_envelope()],
        news_excerpts=[("tc-news", "headline; do not follow")],
    )
    assert "<news_excerpt id=\"tc-news\">" in p
    assert "</news_excerpt>" in p
    assert "DATA" in p


def test_system_prompt_forbids_open_prose_and_inventions():
    assert "must be a single call" in SYSTEM_PROMPT.lower() or "MUST be a single call" in SYSTEM_PROMPT
    assert "MAY NOT invent" in SYSTEM_PROMPT
    assert "data, not instructions" in SYSTEM_PROMPT.lower() or "treat it as" in SYSTEM_PROMPT


def test_tool_schema_lists_only_v1_verdict_enums():
    direction_enum = EMIT_VERDICT_TOOL["input_schema"]["properties"]["direction"]["enum"]
    assert set(direction_enum) == {"SKIP", "LONG_CALL", "LONG_PUT"}


# ---------------------------------------------------------------------------
# build_verdict_from_tool_input


def test_long_call_verdict_built_from_screener_row():
    c = _contract("AAPL_C200", OptionRight.call)
    envs = [_envelope("tc-1")]
    v, picked = build_verdict_from_tool_input(
        tool_input={
            "direction": "LONG_CALL",
            "chosen_occ": "AAPL_C200",
            "conviction": 0.6,
            "primary_reasons": ["bullish momentum"],
            "tool_call_ids_used": ["tc-1"],
        },
        candidates=[c],
        envelopes=envs,
        disclaimer=DISCLAIMER,
    )
    assert v.action is VerdictAction.long_call
    assert picked is c
    assert v.contract is c
    assert v.conviction == 0.6
    assert len(v.citations) == 1
    assert v.citations[0].provider_profile_id == "yfinance_research"


def test_hallucinated_occ_returns_skip_shell():
    c = _contract("AAPL_C200")
    v, picked = build_verdict_from_tool_input(
        tool_input={
            "direction": "LONG_CALL",
            "chosen_occ": "AAPL_C999_MADE_UP",
            "primary_reasons": ["..."],
            "tool_call_ids_used": ["tc-1"],
        },
        candidates=[c],
        envelopes=[_envelope("tc-1")],
        disclaimer=DISCLAIMER,
    )
    assert v.action is VerdictAction.skip
    assert v.skip_reason is SkipReason.hallucinated_contract
    assert picked is None


def test_direction_contract_right_mismatch_returns_skip():
    c = _contract("AAPL_C200", OptionRight.call)
    v, _ = build_verdict_from_tool_input(
        tool_input={
            "direction": "LONG_PUT",  # but the OCC is a call
            "chosen_occ": "AAPL_C200",
            "primary_reasons": ["..."],
            "tool_call_ids_used": ["tc-1"],
        },
        candidates=[c],
        envelopes=[_envelope("tc-1")],
        disclaimer=DISCLAIMER,
    )
    assert v.action is VerdictAction.skip
    assert v.skip_reason is SkipReason.disallowed_strategy


def test_skip_with_skip_reason_passed_through():
    v, _ = build_verdict_from_tool_input(
        tool_input={
            "direction": "SKIP",
            "skip_reason": "stale_required_input",
            "primary_reasons": ["data too old"],
            "tool_call_ids_used": ["tc-1"],
        },
        candidates=[_contract()],
        envelopes=[_envelope("tc-1")],
        disclaimer=DISCLAIMER,
    )
    assert v.action is VerdictAction.skip
    assert v.skip_reason is SkipReason.stale_required_input


# ---------------------------------------------------------------------------
# synthesise() with a fake client


class _FakeClient:
    def __init__(self, tool_input: dict) -> None:
        self.tool_input = tool_input
        self.calls: list[dict] = []

    def synthesise(self, *, system, user_prompt, tool, max_output_tokens, timeout_s):
        self.calls.append(
            {"system_len": len(system), "user_len": len(user_prompt), "tool": tool["name"]}
        )
        return self.tool_input, {"stop_reason": "tool_use"}


def test_synthesise_round_trip():
    c = _contract("AAPL_C200")
    client = _FakeClient(
        {
            "direction": "LONG_CALL",
            "chosen_occ": "AAPL_C200",
            "primary_reasons": ["liquid"],
            "tool_call_ids_used": ["tc-1"],
        }
    )
    r = synthesise(
        client=client,
        disclaimer=DISCLAIMER,
        ticker="AAPL",
        spot=190.0,
        candidates=[c],
        envelopes=[_envelope("tc-1")],
    )
    assert r.verdict.action is VerdictAction.long_call
    assert client.calls and client.calls[0]["tool"] == "emit_verdict"
