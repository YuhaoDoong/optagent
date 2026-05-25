from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from optagent import DISCLAIMER
from optagent.registry import ProviderRegistry
from optagent.render import render_template
from optagent.schemas import (
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
    SkipReason,
    Verdict,
    VerdictAction,
)
from optagent.validator import validate


UTC_NOW = datetime.now(timezone.utc)


TTL_TABLE = {
    "price": {"rth": 10, "after_hours": 300, "critical": True},
    "options_chain": {
        "rth_low_vol": 30,
        "rth_high_vol_or_near_expiry": 15,
        "after_hours": 300,
        "critical": True,
    },
    "macro": {"baseline": 86400, "critical": False},
}


def _registry(run_mode="personal_research", moomoo_entitled=False) -> ProviderRegistry:
    r = ProviderRegistry()
    r.register(
        ProviderProfile(
            id="yfinance_research",
            permitted_use=PermittedUse.research_only,
            redistribution=Redistribution.none,
            terms_url="https://pypi.org/project/yfinance/",
            profile_version="2026-05-25",
        )
    )
    r.register(
        ProviderProfile(
            id="fred_default",
            permitted_use=PermittedUse.production_safe,
            redistribution=Redistribution.attribution,
            attribution_string="Data sourced from FRED.",
            terms_url="https://fred.stlouisfed.org/docs/api/terms_of_use.html",
            profile_version="2026-05-25",
        )
    )
    rc = RunConfig(ticker="AAPL", run_mode=run_mode, moomoo_entitled=moomoo_entitled)  # type: ignore[arg-type]
    r.bind(rc)
    return r


def _price_env(tcid="tc-price", session=MarketSession.rth, age_seconds=1) -> Envelope:
    return Envelope(
        value={"last": 190.0},
        as_of=UTC_NOW - timedelta(seconds=age_seconds),
        source="yfinance",
        delay_assumption="delayed_15min",
        market_session=session,
        confidence=Confidence.ok,
        provider_profile_id="yfinance_research",
        tool_call_id=tcid,
    )


def _chain_env(tcid="tc-chain", session=MarketSession.rth, age_seconds=2) -> Envelope:
    return Envelope(
        value={"ticker": "AAPL", "expiration": "2026-06-19", "dte": 21, "rows": []},
        as_of=UTC_NOW - timedelta(seconds=age_seconds),
        source="yfinance",
        delay_assumption="delayed_15min",
        market_session=session,
        confidence=Confidence.ok,
        provider_profile_id="yfinance_research",
        tool_call_id=tcid,
    )


def _contract(**overrides) -> OptionContract:
    base = dict(
        occ_symbol="AAPL_C200",
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
    base.update(overrides)
    return OptionContract(**base)


def _long_verdict(contract: OptionContract, citations: list[Citation]) -> Verdict:
    return Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.long_call if contract.right is OptionRight.call else VerdictAction.long_put,
        contract=contract,
        conviction=0.5,
        primary_reasons=["test"],
        citations=citations,
    )


def test_clean_long_call_passes_all_checks():
    contract = _contract()
    chain = _chain_env()
    price = _price_env()
    v = _long_verdict(
        contract,
        [
            Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id),
            Citation(tool_call_id=price.tool_call_id, provider_profile_id=price.provider_profile_id),
        ],
    )
    memo = render_template(v, [price, chain])
    outcome = validate(
        verdict=v,
        candidates=[contract],
        envelopes=[price, chain],
        llm_tool_input={"tool_call_ids_used": ["tc-price", "tc-chain"]},
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=memo,
        now=UTC_NOW,
    )
    assert outcome.skip_reason is None
    assert outcome.final_verdict.action is VerdictAction.long_call
    assert all(d.passed for d in outcome.decisions)


def test_hallucinated_contract_downgrades_to_skip():
    contract = _contract(occ_symbol="AAPL_C999_PHANTOM")  # not in candidates
    real = _contract(occ_symbol="AAPL_C200")
    chain = _chain_env()
    v = _long_verdict(contract, [Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id)])
    outcome = validate(
        verdict=v,
        candidates=[real],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [chain]),
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.hallucinated_contract
    assert outcome.final_verdict.action is VerdictAction.skip


def test_phantom_citation_downgrades_to_skip():
    contract = _contract()
    chain = _chain_env()
    v = _long_verdict(
        contract,
        [Citation(tool_call_id="tc-MADE-UP", provider_profile_id="yfinance_research")],
    )
    outcome = validate(
        verdict=v,
        candidates=[contract],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [chain]),
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.phantom_citation


def test_numeric_mismatch_downgrades_to_skip():
    real = _contract(mid=2.50)
    tampered = _contract(mid=99.0)  # LLM tried to change the price
    chain = _chain_env()
    v = _long_verdict(
        tampered,
        [Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id)],
    )
    outcome = validate(
        verdict=v,
        candidates=[real],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [chain]),
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.numeric_grounding_mismatch


def test_stale_price_envelope_downgrades_to_skip():
    contract = _contract()
    stale_price = _price_env(age_seconds=3600)  # 1 hour old → outside RTH price TTL of 10s
    chain = _chain_env(age_seconds=2)
    v = _long_verdict(
        contract,
        [
            Citation(tool_call_id=stale_price.tool_call_id, provider_profile_id=stale_price.provider_profile_id),
            Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id),
        ],
    )
    outcome = validate(
        verdict=v,
        candidates=[contract],
        envelopes=[stale_price, chain],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [stale_price, chain]),
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.stale_required_input


def test_compliance_gate_block_downgrades_to_skip():
    contract = _contract()
    chain = _chain_env()
    v = _long_verdict(
        contract,
        [Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id)],
    )
    # yfinance is research_only; distributed run_mode blocks it.
    outcome = validate(
        verdict=v,
        candidates=[contract],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(run_mode="distributed"),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [chain, _price_env()]),
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.compliance_gate_failed


def test_disclaimer_missing_fails_presence_check():
    contract = _contract()
    chain = _chain_env()
    v = _long_verdict(
        contract,
        [Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id)],
    )
    bad_render = "no disclaimer here\nVerdict: LONG_CALL\n"
    outcome = validate(
        verdict=v,
        candidates=[contract],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=bad_render,
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.presence_check_failed


def test_fred_citation_without_attribution_fails_presence():
    contract = _contract()
    chain = _chain_env()
    fred_env = Envelope(
        value={"cpi": 3.0},
        as_of=UTC_NOW,
        source="fred",
        delay_assumption="eod",
        market_session=MarketSession.rth,
        confidence=Confidence.ok,
        provider_profile_id="fred_default",
        tool_call_id="tc-fred",
    )
    v = _long_verdict(
        contract,
        [
            Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id),
            Citation(tool_call_id=fred_env.tool_call_id, provider_profile_id=fred_env.provider_profile_id),
        ],
    )
    # render_template will include the attribution when cited_fred=True; here we omit it.
    bad_render = render_template(v, [chain, fred_env, _price_env()], cited_fred=False)
    outcome = validate(
        verdict=v,
        candidates=[contract],
        envelopes=[chain, fred_env, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=bad_render,
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.presence_check_failed


def test_future_as_of_envelope_is_rejected():
    """Codex finding (f): future as_of must fail-closed, not pass silently."""

    contract = _contract()
    chain = _chain_env(age_seconds=-3600)  # 1h in the future
    v = _long_verdict(
        contract,
        [Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id)],
    )
    outcome = validate(
        verdict=v,
        candidates=[contract],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [chain, _price_env()]),
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.stale_required_input
    assert "future" in (outcome.decisions[-3].detail or "").lower() or "future" in (outcome.decisions[5].detail or "").lower()


def test_nan_in_numeric_field_is_rejected():
    """Codex finding (d): NaN/Inf must fail numeric grounding."""

    real = _contract()
    bad = _contract()
    bad_dict = bad.model_dump()
    bad_dict["mid"] = float("nan")
    bad = OptionContract(**bad_dict)
    chain = _chain_env()
    v = _long_verdict(
        bad,
        [Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id)],
    )
    outcome = validate(
        verdict=v,
        candidates=[real],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [chain]),
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.numeric_grounding_mismatch


def test_strike_tampering_is_rejected():
    """Codex finding (d): strike must be numerically grounded too."""

    real = _contract(strike=200.0)
    tampered = _contract()
    tampered_dict = tampered.model_dump()
    tampered_dict["strike"] = 201.5  # different strike value
    tampered = OptionContract(**tampered_dict)
    chain = _chain_env()
    v = _long_verdict(
        tampered,
        [Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id)],
    )
    outcome = validate(
        verdict=v,
        candidates=[real],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [chain]),
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.numeric_grounding_mismatch


def test_duplicate_candidate_occs_fail_contract_match():
    """Codex finding (b): exactly-one canonical row is required."""

    real_a = _contract(occ_symbol="AAPL_C200")
    real_b = _contract(occ_symbol="AAPL_C200", bid=2.40)  # duplicate OCC
    chain = _chain_env()
    v = _long_verdict(
        real_a,
        [Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id)],
    )
    outcome = validate(
        verdict=v,
        candidates=[real_a, real_b],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [chain]),
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.hallucinated_contract


def test_citation_provider_id_spoof_detected():
    """Codex finding (c): citation provider_profile_id must match envelope's."""

    contract = _contract()
    chain = _chain_env()  # provider_profile_id = yfinance_research
    # The LLM cites the SAME tcid but lies about the provider.
    v = _long_verdict(
        contract,
        [Citation(tool_call_id=chain.tool_call_id, provider_profile_id="moomoo_user_entitled")],
    )
    outcome = validate(
        verdict=v,
        candidates=[contract],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [chain]),
        now=UTC_NOW,
    )
    assert outcome.skip_reason is SkipReason.phantom_citation


def test_validator_downgrade_always_uses_canonical_disclaimer():
    """Codex finding: validator must not echo back the LLM's `verdict.disclaimer`."""

    from optagent import DISCLAIMER as CANON

    contract = _contract(occ_symbol="AAPL_C200")
    chain = _chain_env()
    v = Verdict(
        disclaimer="LIES — THIS IS LEGAL ADVICE",  # adversarial disclaimer
        action=VerdictAction.long_call,
        contract=_contract(occ_symbol="AAPL_C_HALLUCINATED"),
        primary_reasons=["..."],
        citations=[Citation(tool_call_id=chain.tool_call_id, provider_profile_id=chain.provider_profile_id)],
    )
    outcome = validate(
        verdict=v,
        candidates=[contract],
        envelopes=[chain, _price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [chain]),
        now=UTC_NOW,
    )
    assert outcome.final_verdict.disclaimer == CANON


def test_volume_oi_caveat_anchor_matches_renderer():
    """Codex finding (h): renderer caveat MUST contain the validator's anchor."""

    from optagent.adapters.volume_oi_context import VolumeOIContextAdapter
    from optagent.validator import VOLUME_OI_REQUIRED_PHRASE

    # The phrase the validator searches for must appear in the adapter's caveat
    # AND in the renderer's caveat block (case-insensitive substring).
    assert VOLUME_OI_REQUIRED_PHRASE.lower() in VolumeOIContextAdapter.CAVEAT.lower()


def test_validator_records_all_checks_on_skip():
    """SKIP verdicts must produce a full decisions list — never silently skipped."""

    v = Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.skip,
        skip_reason=SkipReason.no_candidates_after_screen,
    )
    outcome = validate(
        verdict=v,
        candidates=[],
        envelopes=[_price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [_price_env()]),
        now=UTC_NOW,
    )
    check_ids = {d.check_id for d in outcome.decisions}
    # All nine independent checks must appear in the decision list.
    assert {"a_verdict_enum", "b_contract_match", "c_citation_existence", "d_numeric_grounding",
            "e_compliance_gate", "f_staleness", "g_strategy_scope", "h_presence",
            "i_positive_path_gating"} <= check_ids


def test_skip_path_phantom_citation_recorded():
    """SKIP verdict citing a phantom tcid must not silently mark (c) as passed."""

    v = Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.skip,
        skip_reason=SkipReason.no_candidates_after_screen,
        citations=[Citation(tool_call_id="tc-PHANTOM", provider_profile_id="yfinance_research")],
    )
    outcome = validate(
        verdict=v,
        candidates=[],
        envelopes=[_price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [_price_env()]),
        now=UTC_NOW,
    )
    citation_decision = next(d for d in outcome.decisions if d.check_id == "c_citation_existence")
    assert citation_decision.passed is False


def test_skip_verdict_passes_minimal_checks():
    v = Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.skip,
        skip_reason=SkipReason.no_candidates_after_screen,
        primary_reasons=["empty"],
    )
    outcome = validate(
        verdict=v,
        candidates=[],
        envelopes=[_price_env()],
        llm_tool_input=None,
        registry=_registry(),
        ttl_table=TTL_TABLE,
        rendered_output=render_template(v, [_price_env()]),
        now=UTC_NOW,
    )
    assert outcome.final_verdict.action is VerdictAction.skip
    assert outcome.skip_reason is SkipReason.no_candidates_after_screen
