# CLAUDE.md — optagent

> **RESEARCH ONLY — NOT FINANCIAL ADVICE.** This is a US-equity options
> research agent. It does not place orders, recommend trades, or claim
> suitability for any user.

This file is auto-loaded when Claude Code opens this repo. Use it as a
project map; the detailed implementation plan lives at
`.humanize/plans/age-20260523-150441-plan.md` (local-only, gitignored).

---

## What this project does

Given a US equity ticker, `optagent` returns a STRUCTURED, AUDITABLE
research memo with:

- A verdict in `{SKIP, LONG_CALL, LONG_PUT}` (long-premium only in v1)
- A concrete OCC contract candidate (strike, expiry, breakeven, max-loss)
- A rationale grounded only in tool-returned facts (every claim cites a `tool_call_id`)
- A full audit ledger row capturing inputs, all envelopes, and validator decisions
- A mandatory disclaimer header

Default mode is `template_only` (no LLM call). LLM synthesis is opt-in via
`--enable-llm`, gated by a deterministic worst-case budget pre-check and a
fail-closed post-LLM validator with nine independent checks.

## Architecture (top-down)

```
CLI (`optagent analyze <ticker>`)        src/optagent/cli.py
        │
        ▼
Orchestrator (analyze)                   src/optagent/orchestrator.py
        │
        ├── ProviderRegistry (gate at every adapter call)   registry.py + profiles.py
        ├── Adapters (envelope-wrapped, never raise)        adapters/
        │     ├── YFinanceAdapter         price / chain / 60d OHLCV → HV20
        │     ├── EconCalendarAdapter     FOMC/CPI/NFP/PPI/GDP days-to-next
        │     ├── FREDAdapter             6 curated macro series
        │     ├── SECEdgarAdapter         recent 8-K metadata
        │     └── VolumeOIContextAdapter  Max Pain / walls / PCR (derived)
        ├── Contract screener            screener.py
        │     liquidity + DTE + event-proximity + Greeks + IV-richness
        ├── (--enable-llm only) Budget pre-check            budget.py
        ├── (--enable-llm only) LLM synthesis               llm.py
        │     Anthropic / OpenAI / Gemini via LLMClient protocol
        ├── (--enable-llm only) Fail-closed validator       validator.py
        │     AC-12 checks (a)–(i); any failure → SKIP
        ├── Renderer + disclaimer                          render.py
        └── Audit ledger (JSONL per day)                   ledger.py
```

Two key safety invariants:

1. **Canonical numerics come from the screener**, never from the LLM.
   The LLM only picks `chosen_occ` + direction + rationale prose; every
   numeric field is copied verbatim from the screener row.
2. **Post-LLM validator is authoritative**. Hallucinated OCC, phantom
   citations, stale inputs, compliance gate fail, missing attribution,
   wrong contract right — any of them downgrades the verdict to SKIP.

## Acceptance criteria

12 ACs total. See `.humanize/plans/age-20260523-150441-plan.md` for full
TDD-style positive + negative tests per AC. Quick map:

| AC | What it locks down | Owner module |
|---|---|---|
| AC-1 | Uniform `Envelope` shape on every adapter call | `schemas.py`, all adapters |
| AC-2 | Deterministic contract screener (BEFORE the LLM) | `screener.py`, `payoff.py`, `pricing.py` |
| AC-3 | LLM synthesis with anti-hallucination grounding | `llm.py` |
| AC-4 | Audit ledger JSONL | `ledger.py` |
| AC-5 | Disclaimer + bounded scope (no order endpoints) | `render.py`, `cli.py` |
| AC-6 | Cross-ticker generality + replay reproducibility | `replay.py`, `tests/fixtures/` |
| AC-7 | Graceful degradation | `adapters/*.py`, `orchestrator.py` |
| AC-8 | News/SEC factual-only + prompt-injection guards | `llm.py`, `tests/test_prompt_injection.py` |
| AC-9 | Provider compliance profile (license-tier aware) | `profiles.py`, `registry.py` |
| AC-10 | Cache/TTL policy with stale-data SKIP | `validator.py`, `config/ttl_table.yaml` |
| AC-11 | LLM budget pre-check (deterministic worst-case) | `budget.py`, `config/price_table.yaml` |
| AC-12 | Fail-closed post-LLM validator (9 checks a–i) | `validator.py` |

## Run

```bash
# template-only (no LLM, no API key needed)
optagent analyze AAPL --horizon 14 --max-loss 500

# with LLM synthesis — provider auto-detected from env vars
ANTHROPIC_API_KEY=sk-...  optagent analyze AAPL --enable-llm
OPENAI_API_KEY=sk-...     optagent analyze AAPL --enable-llm --provider openai
GEMINI_API_KEY=...        optagent analyze AAPL --enable-llm --provider gemini

# enable optional adapters
export FRED_API_KEY=...                                   # FRED macro
export OPTAGENT_USER_AGENT="me/0.0.1 (me@example.com)"    # SEC EDGAR
optagent analyze AAPL
```

Dependencies:

- Required: `pydantic>=2`, `pyyaml`
- `[adapters]` extra: `yfinance`, `fredapi`, `requests`
- `[llm]` extra: `anthropic`, `tiktoken` (OpenAI / Gemini SDKs lazy-imported)
- `[test]` extra: `pytest`

The `gold` conda env on this machine already has `yfinance` 1.2 installed.

## Dev workflow

```bash
# all tests (currently ~163, runs in <2s)
PYTHONPATH=src python -m pytest tests/

# capture / regenerate replay fixtures
python scripts/capture_fixtures.py --include-sec

# replay a single fixture
PYTHONPATH=src python -c "
from pathlib import Path
from optagent.replay import replay
r = replay(Path('tests/fixtures/AAPL.json'), write_ledger=False)
print(r.memo)
"
```

## Conventions

- **No `place_order` / `submit_order` / `new_order` strings anywhere under
  `src/`.** This is asserted by `tests/test_cli.py::test_cli_does_not_expose_order_placement_verb`.
- **`Verdict` action enum is closed to `{SKIP, LONG_CALL, LONG_PUT}`.**
  Adding a variant requires a new pydantic model AND a new validator path.
- **Every adapter returns `Envelope`, never a bare value.** Tests enforce
  the shape; the validator depends on `tool_call_id`, `as_of`,
  `provider_profile_id`, `cache_age_s` being present.
- **No `from random import seed`, no hardcoded risk-free rate.** Use
  `RunConfig.random_seed` and the FRED envelope respectively.
- **Implementation code must NOT contain plan-specific terminology** like
  `AC-`, `Milestone`, `Step`, `Phase`. Those belong in the plan document.

## What's deferred to v2

- `news_factual` adapter (NewsAPI integration; AC-8's news side currently
  has the LLM prompt-builder defence but no live news adapter yet)
- IV-history-based IV rank filter (currently we only compute IV/HV20 as
  informational; gating filter requires accumulated per-stock IV history)
- Per-ticker DL model cache (Alt-3 in the design)
- Multi-agent debate mode (Alt-5)
- LEAPS / spreads (the bounded-verdict invariant blocks them by design)

## Round-by-round history

The project ships in iterative rounds. Each round is a single commit
pushed to `main`. To see the round-summary commits, run `git log --oneline`.

| Round | Surface |
|---|---|
| 0 | Foundation: schemas, registry, config loaders |
| 1 | template_only pipeline end-to-end (math, adapter, screener, ledger, CLI) |
| 2 | LLM synthesis + budget pre-check + fail-closed validator |
| 3 | FRED / econ calendar / SEC EDGAR adapters |
| 4 | volume_oi_context + IV richness + multi-model LLM + replay harness |
| 5 | prompt-injection regression tests + 5-ticker fixture batch + this CLAUDE.md |

## Reference repo (data-source patterns only)

`~/Gold` is the gold/silver-focused sibling project this design borrows
its connector idioms from. It is **not** imported at runtime. Reuse pattern,
copy code by hand into ticker-parameterised versions in `src/optagent/`.

## When in doubt — defer to SKIP

If you encounter an ambiguous situation while modifying this code (stale
data, missing field, compliance question, broken provider), the safe choice
is always: **emit SKIP with a structured `skip_reason`**. The v1 design
explicitly prefers abstention over a guess.
