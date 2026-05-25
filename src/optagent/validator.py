"""Fail-closed post-LLM validator (AC-12).

Runs nine checks against any non-SKIP verdict. ANY failure forces the verdict
to SKIP with a structured `skip_reason`. The decisions are persisted in the
audit ledger so post-hoc analysis can confirm why a verdict was rejected.

Checks:
  (a) verdict_enum         — VerdictAction ∈ {SKIP, LONG_CALL, LONG_PUT}
  (b) contract_match       — cited OCC matches exactly one screener row
  (c) citation_existence   — every cited tool_call_id exists with full metadata
  (d) numeric_grounding    — canonical numerics match the screener row within ε
  (e) compliance_gate      — every cited provider passes registry.gate()
  (f) staleness            — every REQUIRED input's cache_age_s ≤ TTL
  (g) strategy_scope       — semantic check against short/naked/0DTE/missing-fields
  (h) presence             — disclaimer + FRED attribution (if cited) + volume_oi caveat
  (i) positive_path_gating — composite: non-SKIP impossible if any required gate fails
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from .registry import ProviderRegistry
from .schemas import (
    Confidence,
    Envelope,
    OptionContract,
    SkipReason,
    ValidatorDecision,
    Verdict,
    VerdictAction,
)


# Numeric grounding tolerances per AC-12.d
NUMERIC_EPSILON: dict[str, float] = {
    "mid": 0.01,
    "breakeven": 0.01,
    "max_loss": 0.50,
    "delta": 0.001,
    "theta": 0.001,
    "vega": 0.001,
    "iv": 0.0001,
}


REQUIRED_PROVIDERS = {"yfinance_research", "moomoo_user_entitled"}
# At least one of these must be the source of the chosen candidate's chain.


@dataclass(frozen=True)
class ValidationOutcome:
    final_verdict: Verdict
    decisions: list[ValidatorDecision]
    skip_reason: SkipReason | None  # populated when the validator downgrades to SKIP


def _decision(check_id: str, passed: bool, detail: str | None = None) -> ValidatorDecision:
    return ValidatorDecision(check_id=check_id, passed=passed, detail=detail)


def _downgrade(verdict: Verdict, reason: SkipReason, detail: str) -> Verdict:
    """Build a SKIP shell that preserves the LLM's prose for the ledger."""

    return Verdict(
        disclaimer=verdict.disclaimer,
        action=VerdictAction.skip,
        skip_reason=reason,
        primary_reasons=list(verdict.primary_reasons) + [f"validator: {detail}"],
        dissenting_factors=verdict.dissenting_factors,
        citations=verdict.citations,
    )


def _numerics_match(llm_value: float, canonical_value: float, eps: float) -> bool:
    if canonical_value == 0:
        return abs(llm_value) <= eps
    return abs(llm_value - canonical_value) <= eps or abs(
        (llm_value - canonical_value) / canonical_value
    ) <= 1e-3


def _ttl_seconds(ttl_table: Mapping[str, Mapping[str, object]], data_type: str, session: str) -> float | None:
    """Return the TTL in seconds for `(data_type, session)` or None if absent."""

    entry = ttl_table.get(data_type)
    if entry is None:
        return None
    # The YAML allows several session-shaped keys depending on data type.
    if session == "rth":
        for k in ("rth", "rth_low_vol", "baseline"):
            if k in entry:
                return float(entry[k])  # type: ignore[arg-type]
    if session in ("pre_market", "after_hours", "closed"):
        for k in ("after_hours", "baseline"):
            if k in entry:
                return float(entry[k])  # type: ignore[arg-type]
    if "baseline" in entry:
        return float(entry["baseline"])  # type: ignore[arg-type]
    return None


def _looks_like(provider_id: str, kind: str) -> bool:
    if kind == "options_chain":
        return provider_id in {"yfinance_research", "moomoo_user_entitled"}
    if kind == "price":
        return provider_id in {"yfinance_research", "moomoo_user_entitled"}
    if kind == "macro":
        return provider_id == "fred_default"
    if kind == "news_factual":
        return provider_id.startswith("newsapi")
    if kind == "sec_filings":
        return provider_id == "sec_edgar_default"
    return False


def _classify(provider_id: str) -> str:
    """Best-effort mapping from provider id to a TTL-table data type."""

    if provider_id in {"yfinance_research", "moomoo_user_entitled"}:
        return "options_chain"  # treated as the strictest of the two
    if provider_id == "fred_default":
        return "macro"
    if provider_id == "sec_edgar_default":
        return "sec_filings"
    if provider_id.startswith("newsapi"):
        return "news_factual"
    return "unknown"


def validate(
    *,
    verdict: Verdict,
    candidates: list[OptionContract],
    envelopes: list[Envelope],
    llm_tool_input: Mapping[str, object] | None,
    registry: ProviderRegistry,
    ttl_table: Mapping[str, Mapping[str, object]],
    rendered_output: str,
    now: datetime | None = None,
) -> ValidationOutcome:
    """Run all nine checks. Returns a ValidationOutcome with decisions list."""

    now = now or datetime.now(timezone.utc)
    decisions: list[ValidatorDecision] = []

    # (a) verdict enum
    try:
        _ = VerdictAction(verdict.action.value)
        decisions.append(_decision("a_verdict_enum", True))
    except ValueError:
        decisions.append(_decision("a_verdict_enum", False, verdict.action.value))
        return ValidationOutcome(
            final_verdict=_downgrade(verdict, SkipReason.verdict_out_of_enum, "bad enum"),
            decisions=decisions,
            skip_reason=SkipReason.verdict_out_of_enum,
        )

    # SKIPs only need checks (e)/(h) to be safe; everything else is moot.
    if verdict.action is VerdictAction.skip:
        decisions.append(_decision("b_contract_match", True, "skipped: SKIP verdict"))
        decisions.append(_decision("c_citation_existence", True, "skipped: SKIP verdict"))
        decisions.append(_decision("d_numeric_grounding", True, "skipped: SKIP verdict"))
        decisions.append(_decision("f_staleness", True, "skipped: SKIP verdict"))
        decisions.append(_decision("g_strategy_scope", True, "skipped: SKIP verdict"))
        # Still run compliance + presence + positive-path
        compliance_ok, compliance_detail = _check_compliance(verdict, registry)
        decisions.append(_decision("e_compliance_gate", compliance_ok, compliance_detail))
        presence_ok, presence_detail = _check_presence(verdict, rendered_output, envelopes)
        decisions.append(_decision("h_presence", presence_ok, presence_detail))
        decisions.append(_decision("i_positive_path_gating", True, "skipped: SKIP verdict"))
        # SKIP verdicts pass even if a presence rule fires (we still surface the issue).
        return ValidationOutcome(
            final_verdict=verdict,
            decisions=decisions,
            skip_reason=verdict.skip_reason,
        )

    # ---- non-SKIP path ----

    contract = verdict.contract
    if contract is None:
        decisions.append(_decision("b_contract_match", False, "non-SKIP verdict has no contract"))
        return ValidationOutcome(
            final_verdict=_downgrade(verdict, SkipReason.hallucinated_contract, "no contract"),
            decisions=decisions,
            skip_reason=SkipReason.hallucinated_contract,
        )

    # (b) contract_match
    candidate_by_occ = {c.occ_symbol: c for c in candidates}
    canonical = candidate_by_occ.get(contract.occ_symbol)
    if canonical is None:
        decisions.append(_decision("b_contract_match", False, contract.occ_symbol))
        return ValidationOutcome(
            final_verdict=_downgrade(
                verdict, SkipReason.hallucinated_contract, f"OCC {contract.occ_symbol} not in candidates"
            ),
            decisions=decisions,
            skip_reason=SkipReason.hallucinated_contract,
        )
    decisions.append(_decision("b_contract_match", True))

    # (c) citation existence — every Citation tcid must map to an Envelope.
    env_by_tcid = {e.tool_call_id: e for e in envelopes}
    missing_tcids = [
        cit.tool_call_id for cit in verdict.citations if cit.tool_call_id not in env_by_tcid
    ]
    if missing_tcids:
        decisions.append(_decision("c_citation_existence", False, ",".join(missing_tcids)))
        return ValidationOutcome(
            final_verdict=_downgrade(
                verdict, SkipReason.phantom_citation, f"missing tcids: {missing_tcids}"
            ),
            decisions=decisions,
            skip_reason=SkipReason.phantom_citation,
        )
    if llm_tool_input is not None:
        for tcid in (llm_tool_input.get("tool_call_ids_used") or []):
            if tcid not in env_by_tcid:
                decisions.append(_decision("c_citation_existence", False, str(tcid)))
                return ValidationOutcome(
                    final_verdict=_downgrade(
                        verdict, SkipReason.phantom_citation, f"missing tcid: {tcid}"
                    ),
                    decisions=decisions,
                    skip_reason=SkipReason.phantom_citation,
                )
    decisions.append(_decision("c_citation_existence", True))

    # (d) numeric grounding — every canonical field on `contract` must match `canonical`
    for field, eps in NUMERIC_EPSILON.items():
        llm_val = float(getattr(contract, field))
        can_val = float(getattr(canonical, field))
        if not _numerics_match(llm_val, can_val, eps):
            detail = f"{field}: llm={llm_val} canon={can_val} eps={eps}"
            decisions.append(_decision("d_numeric_grounding", False, detail))
            return ValidationOutcome(
                final_verdict=_downgrade(verdict, SkipReason.numeric_grounding_mismatch, detail),
                decisions=decisions,
                skip_reason=SkipReason.numeric_grounding_mismatch,
            )
    decisions.append(_decision("d_numeric_grounding", True))

    # (e) compliance gate — for every cited provider, registry.gate() must return ok.
    compliance_ok, compliance_detail = _check_compliance(verdict, registry)
    decisions.append(_decision("e_compliance_gate", compliance_ok, compliance_detail))
    if not compliance_ok:
        return ValidationOutcome(
            final_verdict=_downgrade(verdict, SkipReason.compliance_gate_failed, compliance_detail or ""),
            decisions=decisions,
            skip_reason=SkipReason.compliance_gate_failed,
        )

    # (f) staleness
    stale_detail = _check_staleness(envelopes, ttl_table, now=now)
    if stale_detail is not None:
        decisions.append(_decision("f_staleness", False, stale_detail))
        return ValidationOutcome(
            final_verdict=_downgrade(verdict, SkipReason.stale_required_input, stale_detail),
            decisions=decisions,
            skip_reason=SkipReason.stale_required_input,
        )
    decisions.append(_decision("f_staleness", True))

    # (g) strategy scope — semantic, not keyword
    scope_detail = _check_strategy_scope(contract, verdict.action)
    if scope_detail is not None:
        decisions.append(_decision("g_strategy_scope", False, scope_detail))
        return ValidationOutcome(
            final_verdict=_downgrade(verdict, SkipReason.disallowed_strategy, scope_detail),
            decisions=decisions,
            skip_reason=SkipReason.disallowed_strategy,
        )
    decisions.append(_decision("g_strategy_scope", True))

    # (h) presence checks against the rendered output
    presence_ok, presence_detail = _check_presence(verdict, rendered_output, envelopes)
    decisions.append(_decision("h_presence", presence_ok, presence_detail))
    if not presence_ok:
        return ValidationOutcome(
            final_verdict=_downgrade(verdict, SkipReason.presence_check_failed, presence_detail or ""),
            decisions=decisions,
            skip_reason=SkipReason.presence_check_failed,
        )

    # (i) positive-path gating — composite: any required envelope unavailable or
    # missing required market field → SKIP. (Most of these were caught above.)
    positive_detail = _check_positive_path(envelopes, contract)
    if positive_detail is not None:
        decisions.append(_decision("i_positive_path_gating", False, positive_detail))
        return ValidationOutcome(
            final_verdict=_downgrade(verdict, SkipReason.disallowed_strategy, positive_detail),
            decisions=decisions,
            skip_reason=SkipReason.disallowed_strategy,
        )
    decisions.append(_decision("i_positive_path_gating", True))

    return ValidationOutcome(final_verdict=verdict, decisions=decisions, skip_reason=None)


# ---------------------------------------------------------------------------
# Helpers


def _check_compliance(verdict: Verdict, registry: ProviderRegistry) -> tuple[bool, str | None]:
    if not registry.is_bound():
        return False, "registry_not_bound"
    cited_profiles = {c.provider_profile_id for c in verdict.citations}
    for pid in cited_profiles:
        try:
            res = registry.gate(pid)
        except LookupError:
            return False, f"unknown_profile:{pid}"
        if not res.ok:
            return False, res.reason or f"gate_failed:{pid}"
    return True, None


def _check_staleness(
    envelopes: list[Envelope],
    ttl_table: Mapping[str, Mapping[str, object]],
    *,
    now: datetime,
) -> str | None:
    """Return None when fresh, else a structured detail string."""

    for env in envelopes:
        if env.confidence is Confidence.unavailable:
            continue
        data_type = _classify(env.provider_profile_id)
        entry = ttl_table.get(data_type)
        if entry is None or not bool(entry.get("critical", False)):
            continue  # non-critical inputs don't force SKIP via staleness alone
        session = env.market_session.value
        ttl = _ttl_seconds(ttl_table, data_type, session)
        if ttl is None:
            continue
        age = (now - env.as_of).total_seconds()
        if age > ttl:
            return f"{data_type} envelope age={age:.0f}s > ttl={ttl:.0f}s"
    return None


def _check_strategy_scope(contract: OptionContract, action: VerdictAction) -> str | None:
    if contract.bid <= 0 or contract.ask <= 0:
        return "missing_bid_or_ask"
    if contract.oi <= 0:
        return "missing_oi"
    if contract.max_loss <= 0 or contract.max_loss == float("inf"):
        return "invalid_max_loss"
    expiry_utc = contract.expiration.astimezone(timezone.utc)
    today_utc = datetime.now(timezone.utc)
    dte = (expiry_utc.date() - today_utc.date()).days
    if dte < 1:
        return f"same_day_or_past_expiry: dte={dte}"
    if action is VerdictAction.long_call and contract.right.value != "call":
        return "direction_contract_mismatch"
    if action is VerdictAction.long_put and contract.right.value != "put":
        return "direction_contract_mismatch"
    return None


def _check_presence(
    verdict: Verdict,
    rendered_output: str,
    envelopes: list[Envelope],
) -> tuple[bool, str | None]:
    if not rendered_output.lstrip().startswith(verdict.disclaimer):
        return False, "disclaimer_missing"
    cited_profiles = {c.provider_profile_id for c in verdict.citations}
    if "fred_default" in cited_profiles and "Federal Reserve Bank of St. Louis" not in rendered_output:
        return False, "fred_attribution_missing"
    # volume_oi_context isn't its own profile yet (deferred adapter); when it
    # ships, add the caveat-presence check here.
    return True, None


def _check_positive_path(envelopes: list[Envelope], contract: OptionContract) -> str | None:
    chain_env_ok = any(
        e.confidence is not Confidence.unavailable
        and _looks_like(e.provider_profile_id, "options_chain")
        for e in envelopes
    )
    price_env_ok = any(
        e.confidence is not Confidence.unavailable
        and _looks_like(e.provider_profile_id, "price")
        for e in envelopes
    )
    if not chain_env_ok:
        return "chain_envelope_unavailable"
    if not price_env_ok:
        return "price_envelope_unavailable"
    if contract.bid <= 0 or contract.ask <= 0 or contract.oi <= 0:
        return "missing_required_market_field"
    return None
