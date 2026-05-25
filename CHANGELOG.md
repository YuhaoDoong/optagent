# Changelog

All notable changes to `optagent` are documented here.

> **RESEARCH ONLY — NOT FINANCIAL ADVICE.** Every release of this tool produces
> structured research memos about US equity options. It does not place orders,
> recommend trades, or claim suitability for any user.

## [0.1.0] - 2026-05-26

First public release. v1 is bounded to long-premium strategies (`SKIP`,
`LONG_CALL`, `LONG_PUT`); short premium, spreads, and 0-DTE are out of scope
by design.

### Highlights

- **Twelve acceptance criteria** locked by tests (179 passing, runs in ~1s).
- **End-to-end pipeline**: CLI → orchestrator → 6 data adapters → contract
  screener → optional LLM synthesis → fail-closed validator (9 checks) →
  audit ledger.
- **Multi-model LLM support**: Anthropic Claude, OpenAI, Google Gemini via a
  single `LLMClient` protocol.
- **Replay harness**: 5-ticker fixture batch (SPY/QQQ/AAPL/NVDA/TSLA) with
  byte-stable template-mode output.
- **Two parallel Codex review passes** integrated (task28/29/30 × 2).

### Adapters

- `yfinance` — research-tier price + 60d OHLCV (→ HV20) + options chain
- `FRED` — six macro series with per-series source attributions
- Economic calendar — offline FOMC/CPI/NFP/PPI/GDP schedule
- SEC EDGAR — recent 8-K metadata, polite User-Agent + ~9 rps rate limit
- `volume_oi_context` — derived Max Pain / OI walls / PCR / centre-of-mass
- (deferred to v0.2) — news_factual via NewsAPI; per-ticker DL model cache;
  Moomoo live adapter

### Safety invariants

1. **Canonical numerics come from the screener**, never from the LLM. The LLM
   may only pick `chosen_occ`, direction, conviction, and rationale prose;
   every numeric field (strike, bid, ask, mid, breakeven, max_loss, Greeks,
   IV, scores) is copied verbatim from the screener row.
2. **Fail-closed post-LLM validator**. Nine independent checks
   (`a` verdict_enum, `b` contract_match, `c` citation_existence,
   `d` numeric_grounding, `e` compliance_gate, `f` staleness, `g`
   strategy_scope, `h` presence, `i` positive_path_gating); ANY failure
   downgrades the verdict to `SKIP` with a structured `skip_reason`.
3. **Compile-time absence of order-placement code.** A CI grep test asserts
   `place_order|submit_order|new_order` never appears under `src/`.
4. **Mandatory disclaimer header** on every output. The validator always uses
   the canonical `DISCLAIMER` constant, never the LLM-supplied
   `verdict.disclaimer`, so an adversarial LLM cannot supply its own.

### Compliance posture

- Per-provider compliance profiles with license-tier split
  (`yfinance_research`, `fred_default`, `sec_edgar_default`,
  `moomoo_user_entitled`, `newsapi_free_dev`, `newsapi_paid_production`,
  `econ_calendar_builtin`, `volume_oi_context_derived`).
- `ProviderProfile.required_notices: tuple[str, ...]` substrings enforced
  by the validator's presence check (h). FRED ships with both the canonical
  attribution and the non-endorsement notice.
- `ProviderRegistry.gate(profile_id)` invoked **at every adapter call**
  against an immutable `RunConfig` snapshot (so the gate cannot be bypassed
  mid-run).
- Cache TTL policy (`config/ttl_table.yaml`); out-of-TTL critical inputs
  force SKIP via validator check (f).
- LLM budget guardrail (`config/price_table.yaml`); worst-case cost computed
  before any LLM call; unknown model → fallback to template-only.

### Known limitations (carry-overs to v0.2)

- `news_factual` adapter not yet implemented — the prompt-builder defence
  layer is in place (delimiter wrapping + system instruction), but no live
  news source is wired.
- IV-rank gating requires accumulated per-stock IV history; the current
  `iv_richness_summary` (median IV ÷ HV20) is informational only.
- Per-ticker DL model cache (Alt-3 from the design phase) and multi-agent
  debate mode (Alt-5) are deferred.
- Replay fixtures cover 5 tickers × 1 trading day; AC-6 calls for 5 × 5.
- FRED per-series citations are emitted in the rendered footer when FRED is
  cited but are not enforced individually by the validator yet.
- NewsAPI source/proprietary-notice preservation requirements (Codex
  task29) will land with the `news_factual` adapter.

### Iteration history

The project shipped in seven iterative rounds. See `git log --oneline` for
the round-summary commits.

| Round | Commit (suffix) | What it added |
|---|---|---|
| 0 | `024d271` | Foundation: schemas, registry, config loaders (30 tests) |
| 1 | `7852968` | Template-only pipeline end-to-end (78 tests) |
| 2 | `1c8eb6d` | LLM synthesis + budget pre-check + fail-closed validator (106 tests) |
| 3 | `27e5d54` | FRED / econ-calendar / SEC EDGAR adapters (120 tests) |
| 4 | `909c891` | volume_oi_context + IV richness + multi-model LLM + replay harness (134 tests) |
| 5 | `f8bfdbd` | Prompt-injection regression + 5-ticker fixture batch + CLAUDE.md (163 tests) |
| 6 | `7e116b2` | First Codex review integration (validator/profiles/screener fixes; 178 tests) |
| 7 | this release | FRED non-endorsement + per-series sources + second Codex review (179 tests) |

### Acknowledgements

This project was developed iteratively via the [humanize](https://github.com/PolyArch/humanize)
plan-then-implement workflow. Three Codex `analyze` tasks (task28/29/30)
audited the screener thresholds, provider compliance, and validator
coverage matrix; their findings shaped Rounds 6 and 7.

The data-connector idioms borrow patterns from a sibling gold/silver
project (`~/Gold`, not imported at runtime; only the method signatures and
adapter shapes were copied into ticker-parameterised forms in
`src/optagent/`).
