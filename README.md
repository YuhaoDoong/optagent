# optagent — US Equity Options Research Agent (v1)

> **RESEARCH ONLY — NOT FINANCIAL ADVICE.**
> This tool produces structured research memos about US equity options.
> It does not place orders, recommend trades, or claim suitability.

Given a US equity ticker, `optagent` returns an auditable memo with a verdict in `{SKIP, LONG_CALL, LONG_PUT}` (long-premium strategies only in v1) plus a concrete candidate contract, payoff math (`breakeven`, `max_loss`), and a rationale grounded only in tool-returned facts.

Default mode is `template_only` (no LLM call) for deterministic, replayable behaviour. LLM synthesis is opt-in and gated by provider-compliance, a deterministic worst-case budget pre-check, and a passing replay smoke test.

## Status

Pre-alpha. Round-0 lays the Foundation milestone:
- pydantic v2 schemas (`Envelope`, `OptionContract`, `Verdict`, `AuditRecord`, `ProviderProfile`, `RunConfig`)
- `ProviderRegistry` with init-time + call-time gating
- YAML config loaders for provider profiles, TTL policy, and LLM price table

Adapters, contract screener, fail-closed validator, audit ledger, and CLI follow in subsequent rounds.

## Quickstart (after v1 ships)

```bash
optagent analyze AAPL --horizon 14d --max-loss 500
```

## Plan

See `.humanize/plans/age-20260523-150441-plan.md` for the full implementation plan, 12 acceptance criteria, and Claude-Codex deliberation record.
