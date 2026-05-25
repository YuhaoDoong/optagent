from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from optagent.schemas import (
    AuditRecord,
    Citation,
    Confidence,
    Envelope,
    MarketSession,
    OptionContract,
    OptionRight,
    PermittedUse,
    ProviderProfile,
    Redistribution,
    RunConfig,
    RunMode,
    SkipReason,
    ValidatorDecision,
    Verdict,
    VerdictAction,
)


UTC_NOW = datetime.now(timezone.utc)


def _sample_profile() -> ProviderProfile:
    return ProviderProfile(
        id="yfinance_research",
        permitted_use=PermittedUse.research_only,
        redistribution=Redistribution.none,
        entitlement_required=False,
        terms_url="https://pypi.org/project/yfinance/",
        profile_version="2026-05-25",
    )


def _sample_envelope(profile_id: str = "yfinance_research", value: float = 180.5) -> Envelope:
    return Envelope(
        value=value,
        as_of=UTC_NOW,
        source="yfinance",
        delay_assumption="delayed_15min",
        market_session=MarketSession.rth,
        confidence=Confidence.ok,
        provider_profile_id=profile_id,
    )


def _sample_contract() -> OptionContract:
    return OptionContract(
        occ_symbol="AAPL260619C00200000",
        underlying="AAPL",
        expiration=UTC_NOW + timedelta(days=21),
        strike=200.0,
        right=OptionRight.call,
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
# Envelope


def test_envelope_round_trip_ok():
    env = _sample_envelope()
    assert env.confidence is Confidence.ok
    assert env.tool_call_id.startswith("tc-")
    assert env.fetched_at.tzinfo is not None


def test_envelope_rejects_naive_timestamp():
    naive = datetime.now()
    with pytest.raises(ValidationError):
        Envelope(
            value=1.0,
            as_of=naive,
            source="x",
            delay_assumption="realtime",
            market_session=MarketSession.rth,
            provider_profile_id="any",
        )


def test_envelope_unavailable_must_have_null_value():
    with pytest.raises(ValidationError):
        Envelope(
            value=1.0,
            as_of=UTC_NOW,
            source="x",
            delay_assumption="realtime",
            market_session=MarketSession.closed,
            confidence=Confidence.unavailable,
            provider_profile_id="any",
        )


def test_envelope_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Envelope(
            value=1.0,
            as_of=UTC_NOW,
            source="x",
            delay_assumption="realtime",
            market_session=MarketSession.rth,
            provider_profile_id="any",
            mystery=42,
        )


# ---------------------------------------------------------------------------
# ProviderProfile


def test_profile_attribution_required_when_redistribution_attribution():
    with pytest.raises(ValidationError):
        ProviderProfile(
            id="fred_bad",
            permitted_use=PermittedUse.production_safe,
            redistribution=Redistribution.attribution,
            terms_url="https://example",
            profile_version="v1",
        )


def test_profile_rejects_id_with_space():
    with pytest.raises(ValidationError):
        ProviderProfile(
            id="bad id",
            permitted_use=PermittedUse.research_only,
            redistribution=Redistribution.none,
            terms_url="https://example",
            profile_version="v1",
        )


def test_profile_is_frozen():
    p = _sample_profile()
    with pytest.raises(ValidationError):
        p.id = "different"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Verdict


def test_verdict_skip_requires_skip_reason_and_no_contract():
    with pytest.raises(ValidationError):
        Verdict(disclaimer="RESEARCH ONLY", action=VerdictAction.skip)
    v = Verdict(
        disclaimer="RESEARCH ONLY",
        action=VerdictAction.skip,
        skip_reason=SkipReason.no_candidates_after_screen,
    )
    assert v.contract is None


def test_verdict_long_call_requires_contract():
    with pytest.raises(ValidationError):
        Verdict(disclaimer="RESEARCH ONLY", action=VerdictAction.long_call)
    contract = _sample_contract()
    v = Verdict(
        disclaimer="RESEARCH ONLY",
        action=VerdictAction.long_call,
        contract=contract,
        conviction=0.55,
        citations=[Citation(tool_call_id="tc-abc", provider_profile_id="yfinance_research")],
    )
    assert v.contract is contract
    assert v.conviction == 0.55


def test_verdict_rejects_unknown_action_string():
    with pytest.raises(ValidationError):
        Verdict.model_validate(
            {
                "disclaimer": "x",
                "action": "SHORT_CALL",
            }
        )


# ---------------------------------------------------------------------------
# AuditRecord


def test_audit_record_minimal_round_trip():
    contract = _sample_contract()
    verdict = Verdict(
        disclaimer="RESEARCH ONLY",
        action=VerdictAction.long_call,
        contract=contract,
        conviction=0.6,
    )
    rec = AuditRecord(
        run_id="run-abc",
        ticker="AAPL",
        user_prefs={"horizon_days": 14},
        run_mode=RunMode.personal_research,
        envelopes=[_sample_envelope()],
        screener_input={"min_oi": 500},
        screener_output=[contract],
        prompt_version="v0",
        final_verdict=verdict,
        validator_decisions=[ValidatorDecision(check_id="a", passed=True)],
        started_at=UTC_NOW,
        finished_at=UTC_NOW + timedelta(seconds=2),
    )
    dumped = rec.model_dump_json()
    back = AuditRecord.model_validate_json(dumped)
    assert back.run_id == "run-abc"
    assert back.final_verdict.action is VerdictAction.long_call


# ---------------------------------------------------------------------------
# RunConfig


def test_runconfig_is_frozen():
    rc = RunConfig(ticker="AAPL")
    with pytest.raises(ValidationError):
        rc.ticker = "TSLA"  # type: ignore[misc]


def test_runconfig_defaults():
    rc = RunConfig(ticker="SPY")
    assert rc.run_mode is RunMode.personal_research
    assert rc.enable_llm is False
    assert rc.prompt_version == "v0"
    assert rc.run_id.startswith("run-")
