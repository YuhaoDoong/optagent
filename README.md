# optagent

**US equity options research agent.** Given a ticker, produces a structured,
auditable research memo with a long-premium verdict, payoff math, and a
rationale grounded only in tool-returned facts.

> **RESEARCH ONLY — NOT FINANCIAL ADVICE.**
> Does not place orders, recommend trades, or claim suitability. The verdict
> enum is bounded to `{SKIP, LONG_CALL, LONG_PUT}` by design.

[![tests](https://img.shields.io/badge/tests-179%20passing-brightgreen)](https://github.com/YuhaoDoong/optagent/actions)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-blue)](#license)

---

## What it does

```
$ optagent analyze AAPL --horizon 14 --max-loss 500

RESEARCH ONLY — NOT FINANCIAL ADVICE.
========================================================================
Verdict: SKIP (no_candidates_after_screen)

Rationale:
  - Template-only mode (no LLM) produced 'neutral' direction; the agent
    defaults to SKIP rather than guessing.
  - 4 candidate(s) survived the screener; pass --enable-llm to let the LLM
    synthesise a verdict.

Sources:
  - [tc-...] yfinance     (delayed_15min, profile=yfinance_research)
  - [tc-...] yfinance     (delayed_15min, profile=yfinance_research)
  - [tc-...] yfinance     (delayed_15min, profile=yfinance_research)
  - [tc-...] econ_calendar (static_schedule, profile=econ_calendar_builtin)
  - [tc-...] volume_oi_context (derived_from_chain, profile=volume_oi_context_derived)
  - [tc-...] sec_edgar    (eod, profile=sec_edgar_default)
```

Every run writes one JSON-Lines row to `data/ledger/YYYY-MM-DD.jsonl`
containing: ticker, user prefs, all upstream envelopes (with timestamps and
provider profile IDs), the screener output, validator decisions per check,
and the final verdict. Replay any historical run from disk.

## Architecture

```
CLI (`optagent analyze <ticker>`)
        │
        ▼
Orchestrator
        ├── ProviderRegistry (gate at every adapter call)
        ├── Adapters (envelope-wrapped, never raise)
        │     ├── yfinance              price / chain / 60d OHLCV → HV20
        │     ├── econ_calendar         FOMC/CPI/NFP/PPI/GDP days-to-next
        │     ├── FRED                  6 macro series + per-series sources
        │     ├── SEC EDGAR             recent 8-K metadata
        │     └── volume_oi_context     Max Pain / OI walls / PCR (derived)
        ├── Contract screener           liquidity + DTE + event + Greeks + IV richness
        ├── (--enable-llm) Budget pre-check (deterministic worst-case)
        ├── (--enable-llm) LLM synthesis (Anthropic / OpenAI / Gemini)
        ├── (--enable-llm) Fail-closed validator (9 checks; ANY fail → SKIP)
        ├── Renderer + canonical disclaimer
        └── Audit ledger (JSONL per day)
```

**Default mode is `template_only`** (no LLM call, deterministic). LLM
synthesis is opt-in via `--enable-llm` and is gated by provider compliance,
a worst-case budget pre-check, and the 9-check fail-closed validator.

See [CLAUDE.md](CLAUDE.md) for the full architecture map and acceptance
criteria.

## Install

Python 3.11+ required.

```bash
git clone https://github.com/YuhaoDoong/optagent.git
cd optagent
pip install -e .[adapters]   # yfinance + fredapi + requests
pip install -e .[llm]        # anthropic + tiktoken (optional)
```

`yfinance` is required for the lower-bound deployment. `fredapi`, SEC EDGAR
(no extra dep — uses stdlib), and Moomoo are optional.

## Quickstart

```bash
# template-only (no LLM, no API key needed)
optagent analyze AAPL --horizon 14 --max-loss 500

# enable optional adapters
export FRED_API_KEY=...
export OPTAGENT_USER_AGENT="me/0.0.1 (me@example.com)"  # SEC EDGAR requires this
optagent analyze AAPL

# LLM mode — provider auto-detected from env vars
ANTHROPIC_API_KEY=sk-... optagent analyze AAPL --enable-llm
OPENAI_API_KEY=sk-...    optagent analyze AAPL --enable-llm --provider openai
GEMINI_API_KEY=...       optagent analyze AAPL --enable-llm --provider gemini
```

## Features

### Bounded scope by design
- Verdict enum is closed to `{SKIP, LONG_CALL, LONG_PUT}` — adding a variant
  requires a new pydantic model AND a new validator path.
- `grep -rE "place_order|submit_order|new_order" src/` returns 0 matches.
  Compile-time absence is asserted by a CI test.
- Refuses verdicts outside the v1 enum at the CLI / API surface.

### Fail-closed validator (AC-12)
Nine independent checks; ANY failure forces the verdict to SKIP with a
structured `skip_reason`. The audit ledger records every check decision.

| Check | What it catches |
|---|---|
| (a) verdict_enum | LLM outputs `SHORT_CALL` / `IRON_CONDOR` / unknown enum |
| (b) contract_match | LLM picks an OCC not in the screener list; duplicate OCCs; identity mismatch on underlying / expiration / right |
| (c) citation_existence | LLM cites a phantom `tool_call_id`; cites the wrong provider |
| (d) numeric_grounding | LLM tampers with mid / strike / bid / ask / Greeks / IV; NaN/Inf input |
| (e) compliance_gate | Cited provider blocked under active `run_mode` (research-only data in distributed mode) |
| (f) staleness | Required input outside TTL; future `as_of`; negative `cache_age_s` |
| (g) strategy_scope | 0-DTE, missing OI / bid / ask, direction-vs-right mismatch |
| (h) presence | Disclaimer missing; FRED attribution + non-endorsement missing; volume_oi caveat missing |
| (i) positive_path_gating | Composite: non-SKIP impossible if any required gate fails |

### Multi-model LLM
Three providers ship; the `LLMClient` protocol is provider-agnostic.

- `make_anthropic_client()` — Claude tool_use (default for Claude API keys)
- `make_openai_client()` — Chat Completions function-calling
- `make_gemini_client()` — Tool function declarations (auto-strips `null`
  enums from the schema)

Provider auto-detection cascade: `--provider` flag → `OPTAGENT_LLM_PROVIDER`
env → `ANTHROPIC_API_KEY` → `OPENAI_API_KEY` → `GEMINI_API_KEY`.

### Deterministic budget pre-check (AC-11)
Before any LLM call:

```
estimated_cost = (input_tokens + max_output_tokens)
                 × (max_retries + 1)
                 × price_per_token[model_version]
                 × (1 + safety_margin)
```

Defaults: `max_input_tokens=60_000`, `max_output_tokens=2_000`,
`max_retries=2`, `safety_margin=0.20`. Unknown model → template-only fall
back. Estimator over-counts by design (2.5 chars/token) so the budget gate
is never accidentally permissive.

### Replay harness (AC-6)
Capture upstream adapter outputs once; replay deterministically from disk.

```bash
# capture fresh fixtures (network required)
python scripts/capture_fixtures.py --include-sec

# replay any committed fixture (no network)
PYTHONPATH=src python -c "
from pathlib import Path
from optagent.replay import replay
print(replay(Path('tests/fixtures/AAPL.json'), write_ledger=False).memo)
"
```

5 tickers × 4 tests = 20 fixture-replay tests run on every commit.

## Configuration

Three YAML files under `config/`:

- `providers.yaml` — provider compliance profiles
- `ttl_table.yaml` — cache TTL policy per data type
- `price_table.yaml` — LLM pricing per model

See the comments in each for the field definitions. The fields are loaded
into pydantic models so YAML typos surface as validation errors.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/
# 179 passed in ~1s
```

Highlights:

- `tests/test_validator.py` — 19 tests covering all 9 AC-12 checks with
  adversarial cases (hallucinated OCC, phantom citations, numeric
  tampering, future `as_of`, stale price, distributed-mode block, FRED
  non-endorsement enforcement).
- `tests/test_prompt_injection.py` — 9 tests covering AC-8 prompt
  injection defence: prompt-builder wrapping, system-prompt framing, and
  validator-as-safety-net.
- `tests/test_replay_fixtures.py` — 20 parametrised tests across the
  5-ticker fixture batch.

## When in doubt — defer to SKIP

The v1 design explicitly prefers abstention over a guess. If you encounter
an ambiguous situation while modifying this code (stale data, missing
field, compliance question, broken provider), the safe choice is always:
**emit SKIP with a structured `skip_reason`**.

## License

MIT. See [LICENSE](LICENSE) (if present) or `pyproject.toml`.

## Acknowledgements

Developed iteratively via the [humanize](https://github.com/PolyArch/humanize)
plan-then-implement workflow. Two parallel rounds of Codex `analyze` audits
shaped the safety surface.

The data-connector idioms borrow patterns from a sibling gold/silver
project (not imported at runtime — only method signatures and adapter
shapes were copied into ticker-parameterised forms).
