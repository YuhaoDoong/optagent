"""Template-only renderer + disclaimer wrapper.

DEFAULT memo mode. No LLM call. The output is fully determined by the
inputs (run_config + screener output + envelopes), so byte-stable replay
holds as long as the same fixtures are used.
"""

from __future__ import annotations

from . import DISCLAIMER
from .schemas import (
    Envelope,
    OptionContract,
    SkipReason,
    Verdict,
    VerdictAction,
)


_BANNER = "=" * 72
_DECLINED_VERDICT_PREFIXES = ("SHORT_", "IRON_", "STRADDLE", "STRANGLE", "NAKED_")

# Delimiter strings the validator (h) uses to scope required-notice presence
# checks. Notices outside this delimited block do NOT count toward presence —
# this prevents an LLM from satisfying the check by writing the notice string
# into its own rationale prose.
NOTICES_BEGIN_MARKER = "--- REQUIRED PROVIDER NOTICES (renderer-emitted) ---"
NOTICES_END_MARKER = "--- END REQUIRED PROVIDER NOTICES ---"


def assert_supported_action(value: str | VerdictAction) -> VerdictAction:
    """Bounded-verdict guard. Raises on anything other than v1 enum members."""

    raw = value.value if isinstance(value, VerdictAction) else str(value).upper()
    if any(raw.startswith(p) for p in _DECLINED_VERDICT_PREFIXES):
        raise UnsupportedVerdictError(f"verdict {raw!r} is outside the v1 scope")
    try:
        return VerdictAction(raw)
    except ValueError as e:
        raise UnsupportedVerdictError(f"verdict {raw!r} is outside the v1 scope") from e


class UnsupportedVerdictError(ValueError):
    """Raised when a caller requests a verdict outside `VerdictAction`."""


def render_with_disclaimer(body: str) -> str:
    """Prepend the disclaimer banner to any rendered body."""

    return f"{DISCLAIMER}\n{_BANNER}\n{body.rstrip()}\n"


def render_template(
    verdict: Verdict,
    envelopes: list[Envelope],
    cited_fred: bool = False,
    cited_volume_oi_context: bool = False,
) -> str:
    """Render a structured memo without invoking an LLM."""

    assert_supported_action(verdict.action)

    lines: list[str] = []
    if verdict.action is VerdictAction.skip:
        lines.append(f"Verdict: SKIP ({_skip_reason_label(verdict.skip_reason)})")
        lines.append("")
        lines.append("Rationale:")
        if verdict.primary_reasons:
            for r in verdict.primary_reasons:
                lines.append(f"  - {r}")
        else:
            lines.append("  - The pre-LLM screener did not produce an actionable candidate.")
        lines.append("")
        # Surface the market context we DID gather, even though no options
        # trade is recommended. A bare SKIP with no data reads like a failure;
        # this shows the user what the agent actually saw.
        snapshot = _market_snapshot_lines(envelopes)
        if snapshot:
            lines.append("Market snapshot (data gathered before the SKIP):")
            lines.extend(f"  - {s}" for s in snapshot)
            lines.append("")
    else:
        c: OptionContract = verdict.contract  # type: ignore[assignment]
        lines.append(f"Verdict: {verdict.action.value}")
        lines.append(f"Contract: {c.occ_symbol}")
        lines.append(f"  Underlying: {c.underlying}")
        lines.append(f"  Expiration: {c.expiration.date().isoformat()}")
        lines.append(f"  Strike:     ${c.strike:.2f}")
        lines.append(f"  Right:      {c.right.value}")
        lines.append(f"  Mid:        ${c.mid:.2f}  (bid ${c.bid:.2f} / ask ${c.ask:.2f})")
        lines.append(f"  Breakeven:  ${c.breakeven:.2f}")
        lines.append(f"  Max-loss:   ${c.max_loss:.2f}")
        lines.append(
            f"  Greeks:     delta={c.delta:.3f}  theta={c.theta:.4f}  "
            f"vega={c.vega:.3f}  iv={c.iv:.3f}"
        )
        lines.append(f"  OI:         {c.oi}    Volume: {c.volume}    Spread: {c.spread_pct:.2%}")
        if c.days_to_event is not None:
            lines.append(f"  Days-to-event: {c.days_to_event}")
        lines.append(f"  Liquidity score: {c.liquidity_score:.2f}")
        lines.append(f"  Data-quality score: {c.data_quality_score:.2f}")
        if verdict.conviction is not None:
            lines.append(f"  Conviction: {verdict.conviction:.2f}")
        lines.append("")
        if verdict.primary_reasons:
            lines.append("Primary reasons:")
            for r in verdict.primary_reasons:
                lines.append(f"  - {r}")
        if verdict.dissenting_factors:
            lines.append("Dissenting factors:")
            for r in verdict.dissenting_factors:
                lines.append(f"  - {r}")
        lines.append("")

    lines.append("Sources:")
    if not envelopes:
        lines.append("  (no upstream tool data)")
    else:
        for env in envelopes:
            lines.append(
                f"  - [{env.tool_call_id}] {env.source} "
                f"({env.delay_assumption}, confidence={env.confidence.value}, "
                f"profile={env.provider_profile_id}, as_of={env.as_of.isoformat()})"
            )

    # Required notices are emitted INSIDE a delimited block so the validator
    # can verify the renderer (not the LLM rationale) produced them.
    if cited_fred or cited_volume_oi_context:
        lines.append("")
        lines.append(NOTICES_BEGIN_MARKER)
        if cited_fred:
            lines.append("Required FRED notices:")
            lines.append("  - Data sourced from the Federal Reserve Bank of St. Louis (FRED).")
            lines.append(
                "  - This product uses the FRED® API but is not endorsed or certified "
                "by the Federal Reserve Bank of St. Louis."
            )
            series_attribs: list[str] = []
            for env in envelopes:
                if env.provider_profile_id != "fred_default":
                    continue
                value = env.value or {}
                for attrib in (value.get("series_attributions") or []):
                    if attrib not in series_attribs:
                        series_attribs.append(attrib)
            if series_attribs:
                lines.append("  Per-series sources:")
                for attrib in series_attribs:
                    lines.append(f"    - {attrib}")
        if cited_volume_oi_context:
            lines.append(
                "Caveat: volume_oi_context is a derived supply/demand proxy and is NOT a "
                "true holder cost-basis ('chip distribution') measure."
            )
        lines.append(NOTICES_END_MARKER)

    return render_with_disclaimer("\n".join(lines))


def extract_required_notices_block(rendered_output: str) -> str:
    """Return only the substring between the notice-block delimiters.

    Used by the validator's presence check so LLM rationale text cannot
    satisfy the required-notice substring search.
    """

    start = rendered_output.find(NOTICES_BEGIN_MARKER)
    if start < 0:
        return ""
    end = rendered_output.find(NOTICES_END_MARKER, start)
    if end < 0:
        return ""
    return rendered_output[start:end]


def _market_snapshot_lines(envelopes: list[Envelope]) -> list[str]:
    """Extract a human-readable market snapshot from upstream envelopes.

    Pure function of the envelope values (no timestamps), so it stays
    byte-stable under replay. Best-effort: any field that isn't present is
    silently skipped.
    """

    spot = None
    last_close = high = low = hv20 = None
    next_event = None
    macro_bits: list[str] = []

    for env in envelopes:
        v = env.value if isinstance(env.value, dict) else None
        if not v:
            continue
        # Spot/price envelope: {'ticker','last'} with no chain rows.
        if "last" in v and "rows" not in v and spot is None:
            spot = v.get("last")
        # OHLCV-derived envelope.
        if "recent_high_60d" in v:
            high = v.get("recent_high_60d")
            low = v.get("recent_low_60d")
            hv20 = v.get("hv20_annual")
            if last_close is None:
                last_close = v.get("last_close")
        # Econ-calendar envelope.
        if v.get("next_event"):
            ne = v["next_event"]
            label = ne.get("label") or ne.get("kind")
            next_event = (label, ne.get("date"), v.get("days_to_next_event"))
        # FRED macro series (one envelope may carry several series values).
        if env.provider_profile_id == "fred_default":
            series = v.get("series") or v.get("observations") or {}
            if isinstance(series, dict):
                for name, val in list(series.items())[:6]:
                    macro_bits.append(f"{name}={val}")

    out: list[str] = []
    px = spot if spot is not None else last_close
    if px is not None:
        out.append(f"Spot / last close: ${float(px):.2f}")
    if high is not None and low is not None:
        extra = ""
        if px is not None and float(high) != float(low):
            pos = (float(px) - float(low)) / (float(high) - float(low))
            extra = f"  (at {pos:.0%} of the 60d range)"
        out.append(f"60-day range: ${float(low):.2f} – ${float(high):.2f}{extra}")
    if hv20 is not None:
        out.append(f"20-day realised volatility (annualised): {float(hv20):.1%}")
    if next_event is not None:
        label, date, days = next_event
        when = f"in {days}d" if isinstance(days, int) else "upcoming"
        out.append(f"Next macro event: {label} on {date} ({when})")
    if macro_bits:
        out.append("Macro (FRED): " + ", ".join(macro_bits))
    return out


def _skip_reason_label(reason: SkipReason | None) -> str:
    if reason is None:
        return "unspecified"
    return reason.value
