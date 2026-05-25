# strategies/ — pluggable cross-ticker screening

This module turns optagent from a single-ticker analyzer into a market
screener. Given a universe of tickers and a strategy, it ranks candidates
by a strategy-defined score and surfaces the top N as **observation
signals** (not entry orders).

> **RESEARCH ONLY — NOT FINANCIAL ADVICE.** Every strategy ships with the
> same disclaimer header as the rest of optagent. Triggered signals are
> for human consideration with their own IV / event confirmation.

## Design — one schema per concern

The framework is intentionally modular: each piece is a separate file
and they communicate via one stable data shape (`StrategySignal`).

```
strategies/
├── base.py           StrategySignal dataclass + BaseStrategy ABC
├── oversold_rebound.py     One concrete strategy (the first one)
├── universe.py       Hand-curated tickers + soft market-cap / volume filters
├── screen.py         Cross-ticker orchestrator (`screen_universe`)
├── registry.py       Central strategy_id -> class lookup
├── __init__.py       Public surface
└── README.md         This file
```

| Module | Responsibility | Talks to |
|---|---|---|
| `base.py` | Define `StrategySignal` + `BaseStrategy` | nothing (pure schema) |
| `universe.py` | Build/filter the candidate ticker list | yfinance fast_info (optional) |
| `oversold_rebound.py` | Per-ticker condition check | OHLCV (pandas), options chain (dict) |
| `screen.py` | Fan strategy across universe, sort by score | strategy + universe + yfinance |
| `registry.py` | Look up strategy class by id | strategy modules |

## StrategySignal — the canonical output

Inspired by the buy-side observation template (multi-timeframe diagnosis →
pricing context → execution friction → reward space → human caveats):

```python
StrategySignal(
    strategy_id="oversold_rebound",
    ticker="XYZ",
    timestamp=datetime(...),
    spot=24.10,
    direction=SignalDirection.long_call_observation,   # | long_put_observation | skip
    score=0.812,                                       # strategy-defined ranking
    daily=DiagnosticBlock(label="daily", conditions={...}),
    hourly=DiagnosticBlock(...) | None,
    intraday=DiagnosticBlock(...) | None,
    pricing=PricingContextBlock(iv_rank=None, ...),    # explicit None when unknown
    friction=FrictionBlock(spread_pct=0.04, liquidity_note="good", ...),
    reward=RewardBlock(target_price=26.40, repair_space_pct=0.0954, ...),
    notes=["v0.3 strategy: observation only; needs human + IV + event confirmation"],
    disclaimer="RESEARCH ONLY — NOT FINANCIAL ADVICE.",
)
```

`to_dict()` makes it JSON-serialisable (numpy scalars coerced via
`.item()`); the audit ledger eats it directly.

## Adding a new strategy

1. Create `src/optagent/strategies/<your_strategy>.py`.
2. Subclass `BaseStrategy`, set class-level `id`, implement
   `evaluate(...)` returning a `StrategySignal` (or `None` for "did not
   apply"). Strategies MUST be pure functions of the inputs (no network).
3. Register in `registry.py`:

   ```python
   from .your_strategy import YourStrategy

   STRATEGY_REGISTRY[YourStrategy.id] = YourStrategy
   ```

4. Optionally write a test under `tests/test_strategies_<name>.py` that
   exercises the trigger condition on synthetic OHLCV.

5. The CLI auto-picks it up:

   ```bash
   optagent screen --strategy your_strategy
   ```

## The first strategy: `oversold_rebound`

US-equity port of the **极值修复买方观察** model (the user's PTA template).

Daily layer (all conditions required):
- `RSI(14) < 35` — momentum oscillator deep in oversold
- `Williams %R(14) < -85` — confirmation by a second oscillator
- `close < lower Bollinger Band(20, 2σ)` — statistical extreme
- `(close / EMA20 - 1) < -4.5%` — meaningful deviation below trend
- `consecutive_down_days >= 3` — sustained selling pressure

Hourly proxy (60m feed not wired in v0.3):
- `ATR(14) <= median ATR over last 14 days × 1.05` — downside momentum
  exhausting (no acceleration)

Intraday proxy (15m feed not wired):
- `latest_close > prior_close` — bleeding has stopped on the latest bar

Pricing block: surfaces broker IV from the options chain if available,
otherwise marks `iv_rank=None` (explicit unknown, not zero).

Friction block: median chain spread; flag as `wide_spread_risk` when ≥5%.

Reward block: target = EMA20 mean-reversion; valid only when potential
repair lies in `[3%, 20%]` (rejects "already-recovered" and "untradeable
distance" setups).

Direction is `long_call_observation` when ALL layers pass + reward is
in the band; otherwise `skip` (still scored at 25% of full-trigger
score so almost-triggers can be inspected post-hoc).

Score = `0.45 × RSI-depth + 0.30 × WR-depth + 0.25 × repair-room`
(each term clipped to [0, 1]).

## Universe filtering

```python
from optagent.strategies import builtin_us_large_cap, UniverseFilter, load_universe

# 60-ticker default (~all liquid US options markets)
tickers = builtin_us_large_cap()

# custom file: one ticker per line, `#` for comments
tickers = load_universe("my_watchlist.txt")

# soft caps via live fast_info (cached 24h under data/cache/universe_cache.json)
tickers = UniverseFilter(min_market_cap_usd=10e9, min_avg_volume=1e6).apply(tickers)
```

When yfinance is unavailable or fast_info lookup fails, the filter
**keeps the ticker** (we don't have evidence to reject it). The strategy
downstream still has its own conditions.

## Why "observation" not "verdict"

Every strategy output ships with the literal note `"observation only;
needs human + IV + event confirmation"`. The framework deliberately does
NOT promote a triggered signal into a `LONG_CALL` verdict — that path
still requires the full `optagent analyze <ticker> --enable-llm`
pipeline with the fail-closed validator. This mirrors the
**SKIP-by-default** invariant from the rest of the project.
