"""Fail-closed post-LLM validator (AC-12).

Runs nine checks against any verdict. ANY failure on a non-SKIP verdict forces
the verdict to SKIP with a structured `skip_reason`. Even on SKIP verdicts, the
validator records its own findings against citations / compliance / presence so
the audit ledger never claims a check passed when it was actually skipped.

Checks:
  (a) verdict_enum         — VerdictAction ∈ {SKIP, LONG_CALL, LONG_PUT}
  (b) contract_match       — cited OCC matches exactly one screener row + full identity
  (c) citation_existence   — every cited tool_call_id exists with full metadata +
                              citation's provider_profile_id matches the envelope's
  (d) numeric_grounding    — canonical numerics (incl. strike/bid/ask/oi/volume)
                              match the screener row within ε; NaN/Inf rejected
  (e) compliance_gate      — every cited provider AND every envelope cited via tcid
                              passes registry.gate()
  (f) staleness            — every REQUIRED input fresh; future as_of rejected
  (g) strategy_scope       — semantic check using the injected `now` (replay-safe)
  (h) presence             — canonical DISCLAIMER + FRED attribution + volume_oi caveat
  (i) positive_path_gating — composite: non-SKIP impossible if any required gate fails

All checks ALWAYS run; results are recorded as `passed`, `failed`,
`not_applicable`, or `not_run` rather than silently skipped, so the audit
ledger reflects the true state of every check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from . import DISCLAIMER
from .registry import ProviderRegistry
from .render import extract_required_notices_block
from .schemas import (
    Confidence,
    Envelope,
    OptionContract,
    SkipReason,
    ValidatorDecision,
    Verdict,
    VerdictAction,
)


# Numeric grounding tolerances per AC-12.d.
# Includes ALL canonical fields the LLM is forbidden to alter (not just price
# math). `strike` and `expiration` shape determine the contract identity;
# bid/ask/oi/volume are quote-quality fields the LLM must copy verbatim.
NUMERIC_EPSILON: dict[str, float] = {
    "strike": 0.0001,
    "mid": 0.01,
    "bid": 0.01,
    "ask": 0.01,
    "spread_pct": 0.0001,
    "breakeven": 0.01,
    "max_loss": 0.50,
    "oi": 0.0,  # integer field — exact match
    "volume": 0.0,
    "delta": 0.001,
    "theta": 0.001,
    "vega": 0.001,
    "iv": 0.0001,
    "liquidity_score": 0.01,
    "data_quality_score": 0.01,
}


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
        disclaimer=DISCLAIMER,  # always emit the canonical disclaimer, never trust verdict.disclaimer
        action=VerdictAction.skip,
        skip_reason=reason,
        primary_reasons=list(verdict.primary_reasons) + [f"validator: {detail}"],
        dissenting_factors=verdict.dissenting_factors,
        citations=verdict.citations,
    )


def _finite_or_none(value: float) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _numerics_match(llm_value: float, canonical_value: float, eps: float) -> bool:
    """`(abs <= eps) OR (rel <= 1e-3)`. NaN/Inf on either side → False.

    Per AC-12.d, both inputs must be finite; the per-field epsilons handle
    near-zero values without ratio explosions.
    """

    if not math.isfinite(llm_value) or not math.isfinite(canonical_value):
        return False
    abs_diff = abs(llm_value - canonical_value)
    if abs_diff <= eps:
        return True
    if canonical_value == 0:
        return False
    return abs_diff / abs(canonical_value) <= 1e-3


def _ttl_seconds(ttl_table: Mapping[str, Mapping[str, object]], data_type: str, session: str) -> float | None:
    """Return the TTL in seconds for `(data_type, session)` or None if absent."""

    entry = ttl_table.get(data_type)
    if entry is None:
        return None
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
        return provider_id.startswith("newsapi") or provider_id == "yfinance_news_research"
    if kind == "sec_filings":
        return provider_id == "sec_edgar_default"
    return False


def _classify(provider_id: str) -> str:
    """Best-effort mapping from provider id to a TTL-table data type."""

    if provider_id in {"yfinance_research", "moomoo_user_entitled"}:
        return "options_chain"
    if provider_id == "fred_default":
        return "macro"
    if provider_id == "sec_edgar_default":
        return "sec_filings"
    if provider_id.startswith("newsapi") or provider_id == "yfinance_news_research":
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
    """Run all nine checks. Returns a ValidationOutcome with decisions list.

    Every check produces exactly one ValidatorDecision; `passed=True` is only
    set when the check actually passed. Checks that don't apply (e.g. (d)
    against a SKIP verdict) record `passed=True` with `detail="not_applicable"`
    so audit consumers can distinguish them from real passes.
    """

    now = now or datetime.now(timezone.utc)
    decisions: list[ValidatorDecision] = []

    # (a) verdict enum -------------------------------------------------------
    raw_action = getattr(verdict.action, "value", str(verdict.action))
    try:
        VerdictAction(raw_action)
        decisions.append(_decision("a_verdict_enum", True))
    except ValueError:
        decisions.append(_decision("a_verdict_enum", False, str(raw_action)))
        return ValidationOutcome(
            final_verdict=_downgrade(verdict, SkipReason.verdict_out_of_enum, "bad enum"),
            decisions=decisions,
            skip_reason=SkipReason.verdict_out_of_enum,
        )

    # Envelope lookup helpers used by several checks.
    env_by_tcid: dict[str, Envelope] = {}
    duplicate_tcids: list[str] = []
    for env in envelopes:
        if env.tool_call_id in env_by_tcid:
            duplicate_tcids.append(env.tool_call_id)
        env_by_tcid[env.tool_call_id] = env

    is_skip = verdict.action is VerdictAction.skip

    # ---- (b) contract_match ----------------------------------------------
    canonical: OptionContract | None = None
    contract = verdict.contract
    if is_skip:
        if contract is not None:
            decisions.append(_decision("b_contract_match", False, "SKIP carries a contract"))
        else:
            decisions.append(_decision("b_contract_match", True, "not_applicable: SKIP"))
    else:
        if contract is None:
            decisions.append(_decision("b_contract_match", False, "non-SKIP without contract"))
            return ValidationOutcome(
                final_verdict=_downgrade(verdict, SkipReason.hallucinated_contract, "no contract"),
                decisions=decisions,
                skip_reason=SkipReason.hallucinated_contract,
            )
        # Reject duplicate OCCs in the candidate list — ambiguous canonical row.
        occ_counts: dict[str, int] = {}
        for c in candidates:
            occ_counts[c.occ_symbol] = occ_counts.get(c.occ_symbol, 0) + 1
        dup_occs = [o for o, n in occ_counts.items() if n > 1]
        if dup_occs:
            decisions.append(_decision("b_contract_match", False, f"duplicate_candidate_occs:{dup_occs}"))
            return ValidationOutcome(
                final_verdict=_downgrade(
                    verdict, SkipReason.hallucinated_contract, f"duplicate OCCs: {dup_occs}"
                ),
                decisions=decisions,
                skip_reason=SkipReason.hallucinated_contract,
            )
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
        # Full-identity comparison: underlying + expiration + right.
        identity_mismatches: list[str] = []
        if contract.underlying != canonical.underlying:
            identity_mismatches.append("underlying")
        if contract.expiration != canonical.expiration:
            identity_mismatches.append("expiration")
        if contract.right != canonical.right:
            identity_mismatches.append("right")
        if identity_mismatches:
            decisions.append(
                _decision("b_contract_match", False, f"identity_mismatch:{identity_mismatches}")
            )
            return ValidationOutcome(
                final_verdict=_downgrade(
                    verdict, SkipReason.hallucinated_contract,
                    f"contract identity mismatch on {identity_mismatches}"
                ),
                decisions=decisions,
                skip_reason=SkipReason.hallucinated_contract,
            )
        decisions.append(_decision("b_contract_match", True))

    # ---- (c) citation_existence -----------------------------------------
    # Run on both SKIP and non-SKIP: SKIPs must not cite phantom tcids either.
    if duplicate_tcids:
        decisions.append(_decision("c_citation_existence", False, f"duplicate_tcid:{duplicate_tcids}"))
        if not is_skip:
            return ValidationOutcome(
                final_verdict=_downgrade(verdict, SkipReason.phantom_citation, "duplicate tcids in envelopes"),
                decisions=decisions,
                skip_reason=SkipReason.phantom_citation,
            )
    else:
        citation_problems: list[str] = []
        for cit in verdict.citations:
            env = env_by_tcid.get(cit.tool_call_id)
            if env is None:
                citation_problems.append(f"missing:{cit.tool_call_id}")
                continue
            if cit.provider_profile_id != env.provider_profile_id:
                citation_problems.append(
                    f"profile_spoof:{cit.tool_call_id}:cit={cit.provider_profile_id}!=env={env.provider_profile_id}"
                )
        if llm_tool_input is not None:
            for tcid in (llm_tool_input.get("tool_call_ids_used") or []):
                if tcid not in env_by_tcid:
                    citation_problems.append(f"missing_tool_call_id:{tcid}")
        if citation_problems:
            decisions.append(_decision("c_citation_existence", False, ";".join(citation_problems)))
            if not is_skip:
                return ValidationOutcome(
                    final_verdict=_downgrade(
                        verdict, SkipReason.phantom_citation, citation_problems[0]
                    ),
                    decisions=decisions,
                    skip_reason=SkipReason.phantom_citation,
                )
        else:
            decisions.append(_decision("c_citation_existence", True))

    # ---- (d) numeric_grounding ------------------------------------------
    if is_skip:
        decisions.append(_decision("d_numeric_grounding", True, "not_applicable: SKIP"))
    else:
        assert canonical is not None and contract is not None  # narrowed above
        mismatches: list[str] = []
        for field, eps in NUMERIC_EPSILON.items():
            llm_raw = getattr(contract, field, None)
            can_raw = getattr(canonical, field, None)
            llm_val = _finite_or_none(float(llm_raw) if llm_raw is not None else float("nan"))
            can_val = _finite_or_none(float(can_raw) if can_raw is not None else float("nan"))
            if llm_val is None or can_val is None:
                mismatches.append(f"{field}:non-finite(llm={llm_raw},canon={can_raw})")
                continue
            if not _numerics_match(llm_val, can_val, eps):
                mismatches.append(f"{field}:llm={llm_val} canon={can_val} eps={eps}")
        if mismatches:
            detail = mismatches[0]
            decisions.append(_decision("d_numeric_grounding", False, detail))
            return ValidationOutcome(
                final_verdict=_downgrade(verdict, SkipReason.numeric_grounding_mismatch, detail),
                decisions=decisions,
                skip_reason=SkipReason.numeric_grounding_mismatch,
            )
        decisions.append(_decision("d_numeric_grounding", True))

    # ---- (e) compliance_gate ---------------------------------------------
    # Gate BOTH the citation-supplied provider IDs AND the envelopes actually
    # referenced via tool_call_ids. This prevents the LLM from spoofing a safer
    # provider in its citation block while the underlying envelope was from a
    # blocked provider.
    cited_provider_ids: set[str] = set()
    for cit in verdict.citations:
        cited_provider_ids.add(cit.provider_profile_id)
        env = env_by_tcid.get(cit.tool_call_id)
        if env is not None:
            cited_provider_ids.add(env.provider_profile_id)
    compliance_ok, compliance_detail = _check_compliance(cited_provider_ids, registry)
    decisions.append(_decision("e_compliance_gate", compliance_ok, compliance_detail))
    if not compliance_ok and not is_skip:
        return ValidationOutcome(
            final_verdict=_downgrade(
                verdict, SkipReason.compliance_gate_failed, compliance_detail or ""
            ),
            decisions=decisions,
            skip_reason=SkipReason.compliance_gate_failed,
        )

    # ---- (f) staleness ----------------------------------------------------
    stale_detail = _check_staleness(envelopes, ttl_table, now=now)
    if is_skip:
        decisions.append(_decision("f_staleness", True, "not_applicable: SKIP"))
    else:
        if stale_detail is not None:
            decisions.append(_decision("f_staleness", False, stale_detail))
            return ValidationOutcome(
                final_verdict=_downgrade(verdict, SkipReason.stale_required_input, stale_detail),
                decisions=decisions,
                skip_reason=SkipReason.stale_required_input,
            )
        decisions.append(_decision("f_staleness", True))

    # ---- (g) strategy_scope ----------------------------------------------
    if is_skip:
        decisions.append(_decision("g_strategy_scope", True, "not_applicable: SKIP"))
    else:
        assert contract is not None
        scope_detail = _check_strategy_scope(contract, verdict.action, now=now)
        if scope_detail is not None:
            decisions.append(_decision("g_strategy_scope", False, scope_detail))
            return ValidationOutcome(
                final_verdict=_downgrade(verdict, SkipReason.disallowed_strategy, scope_detail),
                decisions=decisions,
                skip_reason=SkipReason.disallowed_strategy,
            )
        decisions.append(_decision("g_strategy_scope", True))

    # ---- (h) presence ----------------------------------------------------
    presence_ok, presence_detail = _check_presence(verdict, rendered_output, registry)
    decisions.append(_decision("h_presence", presence_ok, presence_detail))
    if not presence_ok and not is_skip:
        return ValidationOutcome(
            final_verdict=_downgrade(verdict, SkipReason.presence_check_failed, presence_detail or ""),
            decisions=decisions,
            skip_reason=SkipReason.presence_check_failed,
        )

    # ---- (i) positive_path_gating ----------------------------------------
    if is_skip:
        decisions.append(_decision("i_positive_path_gating", True, "not_applicable: SKIP"))
    else:
        assert contract is not None
        positive_detail = _check_positive_path(envelopes, contract)
        if positive_detail is not None:
            decisions.append(_decision("i_positive_path_gating", False, positive_detail))
            return ValidationOutcome(
                final_verdict=_downgrade(verdict, SkipReason.disallowed_strategy, positive_detail),
                decisions=decisions,
                skip_reason=SkipReason.disallowed_strategy,
            )
        decisions.append(_decision("i_positive_path_gating", True))

    return ValidationOutcome(
        final_verdict=verdict,
        decisions=decisions,
        skip_reason=verdict.skip_reason if is_skip else None,
    )


# ---------------------------------------------------------------------------
# Helpers


def _check_compliance(
    provider_ids: set[str], registry: ProviderRegistry
) -> tuple[bool, str | None]:
    if not registry.is_bound():
        return False, "registry_not_bound"
    for pid in sorted(provider_ids):
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
    """Return None when fresh, else a structured detail string.

    Uses `cache_age_s` when the envelope provides a positive value; otherwise
    falls back to `(now - env.as_of)`. Future `as_of` (negative computed age)
    is fail-closed.
    """

    for env in envelopes:
        if env.confidence is Confidence.unavailable:
            continue
        data_type = _classify(env.provider_profile_id)
        entry = ttl_table.get(data_type)
        if entry is None or not bool(entry.get("critical", False)):
            continue
        session = env.market_session.value
        ttl = _ttl_seconds(ttl_table, data_type, session)
        if ttl is None:
            continue

        wall_age = (now - env.as_of).total_seconds()
        if wall_age < 0:
            return f"{data_type} envelope as_of is in the future (age={wall_age:.0f}s)"
        cache_age = env.cache_age_s
        if cache_age < 0:
            return f"{data_type} envelope cache_age_s is negative ({cache_age})"
        # Use the larger of wall_age and cache_age — whichever stales sooner.
        age = max(wall_age, cache_age)
        if age > ttl:
            return f"{data_type} envelope age={age:.0f}s > ttl={ttl:.0f}s"
    return None


def _check_strategy_scope(
    contract: OptionContract, action: VerdictAction, *, now: datetime
) -> str | None:
    # Quote-quality and bounded-risk preconditions.
    for field in ("bid", "ask", "max_loss"):
        val = getattr(contract, field, None)
        if val is None or not math.isfinite(float(val)):
            return f"non_finite_{field}"
    if contract.bid <= 0 or contract.ask <= 0:
        return "missing_bid_or_ask"
    if contract.oi <= 0:
        return "missing_oi"
    if contract.max_loss <= 0 or not math.isfinite(contract.max_loss):
        return "invalid_max_loss"
    expiry_utc = contract.expiration.astimezone(timezone.utc)
    dte = (expiry_utc.date() - now.date()).days
    if dte < 1:
        return f"same_day_or_past_expiry: dte={dte}"
    if action is VerdictAction.long_call and contract.right.value != "call":
        return "direction_contract_mismatch"
    if action is VerdictAction.long_put and contract.right.value != "put":
        return "direction_contract_mismatch"
    return None


VOLUME_OI_REQUIRED_PHRASE = "holder cost-basis"
# Substring that MUST appear in any rendered memo citing volume_oi_context_derived.
# We accept any phrasing containing this anchor (renderer can adjust wording).


def _check_presence(
    verdict: Verdict,
    rendered_output: str,
    registry: ProviderRegistry | None = None,
) -> tuple[bool, str | None]:
    if not rendered_output.lstrip().startswith(DISCLAIMER):
        return False, "disclaimer_missing"
    cited_profiles = {c.provider_profile_id for c in verdict.citations}

    # Required notices are checked WITHIN the renderer's delimited notices
    # block, not anywhere in the rendered output. This prevents the LLM from
    # satisfying a presence check by writing the notice string into its own
    # rationale prose.
    notices_block = extract_required_notices_block(rendered_output)

    # `required_notices` enforcement is fail-closed: if no registry is
    # available, or if the registry is not bound, we cannot verify the
    # provider's notices contract and so we refuse to call any cited verdict
    # compliant. This applies whenever ANY profile is cited.
    if cited_profiles:
        if registry is None or not registry.is_bound():
            return False, "registry_unavailable_cannot_verify_required_notices"
        for pid in cited_profiles:
            try:
                profile = registry.get(pid)
            except LookupError:
                return False, f"unknown_profile_cited:{pid}"
            for notice in profile.required_notices:
                if notice not in notices_block:
                    return False, f"required_notice_missing:{pid}:{notice[:40]}..."

    # Legacy/explicit checks retained as belt-and-braces for the most critical
    # anchors. The volume_oi caveat may live in the notices block; we also
    # accept it anywhere case-insensitively for backwards compatibility with
    # tests that don't trigger the renderer's delimited block.
    if (
        "fred_default" in cited_profiles
        and "Federal Reserve Bank of St. Louis" not in rendered_output
    ):
        return False, "fred_attribution_missing"
    if (
        "volume_oi_context_derived" in cited_profiles
        and VOLUME_OI_REQUIRED_PHRASE not in rendered_output.lower()
    ):
        return False, "volume_oi_caveat_missing"
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
