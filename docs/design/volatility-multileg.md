# Design: volatility (multi-leg long-premium) verdicts

> Status: **PROPOSAL — not yet implemented.** Author-reviewed against the
> codebase at commit `fd5a6b9`. This extends the v1 single-leg, long-premium
> agent to two-leg long-volatility structures while preserving every existing
> safety invariant.

## 1. Goal & scope

Add the ability to recommend **long-volatility, defined-risk, debit-only**
structures when the data supports a "big move, direction unclear" thesis:

- `LONG_STRADDLE` — buy 1 call + 1 put at the **same** strike & expiry (ATM).
- `LONG_STRANGLE` — buy 1 OTM call + 1 OTM put, **different** strikes, same expiry.

**Explicitly still out of scope** (preserves the bounded-risk philosophy):

- Anything you can be a net *seller* of premium on (short straddle/strangle,
  credit spreads, naked legs, iron condors). Max-loss must stay = debit paid.
- Calendar / diagonal spreads (two expiries) — defers to a later round.
- Ratio structures (unequal leg counts) — net delta/vega ambiguity.

Why only these two: both are **pure long premium**, so the v1 invariant
"max-loss is bounded and known up front = total debit" still holds exactly.
That keeps the safety story identical to LONG_CALL / LONG_PUT — we are only
generalising *one leg* to *two legs of the same sign*.

## 2. The invariant this breaks, and how we keep it safe

`render.assert_supported_action` hard-blocks `STRADDLE` / `STRANGLE` via
`_DECLINED_VERDICT_PREFIXES`, and `VerdictAction` is a closed 3-member enum
(`schemas.py:41`). CLAUDE.md states adding a variant "requires a new pydantic
model AND a new validator path." This design honours that:

| Invariant (today) | How it stays true with multi-leg |
|---|---|
| Canonical numerics come from the screener, never the LLM | Screener emits the **pair** (both legs + combined math); LLM only picks the `position_id` + direction label. |
| Max-loss is bounded & pre-known | Long-only ⇒ max-loss = sum of both leg debits. No short leg ⇒ no unbounded risk. |
| Post-LLM validator is authoritative | New leg-aware checks; any mismatch ⇒ SKIP (unchanged philosophy). |
| Bounded verdict enum | Enum grows to 5 members, each with an explicit validator path. Selling structures remain permanently blocked. |
| Defer to SKIP on ambiguity | If only one viable leg is found, or combined liquidity/spread fails, ⇒ SKIP. |

## 3. Data model changes (`schemas.py`)

Keep `OptionContract` exactly as-is (it stays the single-leg row). Add a
**composite** that holds 1–2 legs so single-leg and multi-leg flow through one
type:

```python
class OptionLeg(BaseModel):           # thin wrapper, references a screener row
    model_config = ConfigDict(extra="forbid")
    contract: OptionContract
    quantity: int = 1                 # always +1 in v2 (long). No negatives ⇒ no shorts.

class OptionPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position_id: str                  # deterministic: f"{occ_call}+{occ_put}" sorted
    structure: PositionStructure      # enum: SINGLE | STRADDLE | STRANGLE
    legs: tuple[OptionLeg, ...]       # 1 leg for SINGLE, 2 for straddle/strangle
    # combined canonical math (screener-computed, LLM-immutable):
    net_debit: float                  # = sum(leg.contract.mid * 100 * qty)
    max_loss: float                   # = net_debit (long-only)
    lower_breakeven: float            # put strike - net_debit/100 (strangle/straddle)
    upper_breakeven: float            # call strike + net_debit/100
    net_delta: float
    net_theta: float
    net_vega: float
    combined_iv: float                # debit-weighted mean leg IV
    combined_liquidity_score: float   # min(leg scores) — weakest leg dominates
    combined_spread_pct: float        # debit-weighted
    days_to_event: int | None = None

    @model_validator(mode="after")
    def _long_only(self):
        if any(l.quantity <= 0 for l in self.legs):
            raise ValueError("v2 positions are long-only; quantity must be > 0")
        if self.structure is not PositionStructure.single and len(self.legs) != 2:
            raise ValueError("straddle/strangle require exactly 2 legs")
        return self
```

`VerdictAction` grows:

```python
class VerdictAction(str, Enum):
    skip = "SKIP"
    long_call = "LONG_CALL"
    long_put = "LONG_PUT"
    long_straddle = "LONG_STRADDLE"     # NEW
    long_strangle = "LONG_STRANGLE"     # NEW
```

`Verdict` carries `position: OptionPosition | None` **in addition to** the
existing `contract` field (kept for backward compat: LONG_CALL/LONG_PUT keep
populating `contract`; a `@model_validator` maps single-leg ⇄ position so old
ledger readers and tests don't break). SKIP still carries neither.

## 4. Payoff math (`payoff.py`)

New pure functions (unit-tested in isolation, no I/O):

- `straddle_breakevens(strike, net_debit_per_share) -> (lo, hi)`
  = `(K - d, K + d)`.
- `strangle_breakevens(put_K, call_K, d) -> (put_K - d, call_K + d)`.
- `combined_max_loss(legs)` = `sum(mid * 100)` — long debit only.
- `net_greeks(legs)` = per-greek sum (call delta + put delta naturally nets
  toward ~0 for an ATM straddle — that's the signal).

These mirror the existing `payoff.breakeven` / `payoff.max_loss` style so the
determinism + rounding conventions match.

## 5. Screener pairing (`screener.py`)

After the existing per-row screen produces single-leg `candidates`, add a
**pairing pass** (only when a `--allow-vol` / config flag is on, default off
in v2.0 so existing behaviour is untouched):

1. Partition surviving candidates by `right`.
2. **Straddle**: for each strike present in BOTH calls and puts, pair them.
   Prefer the strike nearest spot (ATM). Combined gates:
   - both legs already passed single-leg liquidity ⇒ pair inherits it;
   - `combined_spread_pct <= max_spread_pct` (stricter: paying two spreads);
   - `net_debit <= max_loss_budget` if the user set one.
3. **Strangle**: pair the nearest OTM call with the nearest OTM put bracketing
   spot (e.g. 0.20–0.35 |delta| band each). One pair per expiry (the most
   liquid), to keep the candidate set bounded.
4. Emit `OptionPosition` rows with deterministic `position_id` and the
   combined math. Sort key extends the existing one:
   `(-combined_liquidity_score, days_to_event, position_id)`.

Single-leg candidates remain in the output unchanged; positions are an
additive list (`ScreenerOutput.positions`). When the flag is off, `positions`
is empty and nothing else changes.

## 6. LLM grounding (`llm.py`)

- `EMIT_VERDICT_TOOL` enum extends to the 5 actions; add an optional
  `chosen_position_id` (mutually exclusive with `chosen_occ`).
- The user prompt lists `candidate_position_ids` + the combined math block,
  same "canonical, do-NOT-alter" framing as today.
- `build_verdict_from_tool_input` copies BOTH legs' numerics verbatim from the
  screener position matched by `position_id`; the LLM supplies only the label
  + reasons. (Same anti-hallucination guarantee, now over a pair.)
- Direction↔structure consistency: `LONG_STRADDLE` ⇒ matched position must be
  `STRADDLE`, both legs present, same strike; else SKIP(`disallowed_strategy`).

## 7. Validator (`validator.py`) — the authoritative gate

Each existing check generalises leg-wise; a position passes only if **every
leg** passes. New/edited paths:

- (a) verdict_enum: now accepts 5 members; selling prefixes still rejected.
- (b) contract_match → **position_match**: `position_id` must exist in the
  screener `positions`; every leg OCC must match a canonical row by full
  identity (underlying + expiration + right + strike). Any phantom leg ⇒
  SKIP(`hallucinated_contract`).
- (d) numeric grounding: compare net_debit / breakevens / net greeks to the
  screener position within float tolerance.
- (e) strategy_scope: assert long-only (all quantities > 0) and structure ∈
  {SINGLE, STRADDLE, STRANGLE}. **A net credit ⇒ SKIP** (defence in depth: even
  if some future bug produced a short leg, negative debit fails here).
- staleness/compliance/presence/positive-path: run per leg; union the cited
  envelopes.

New `SkipReason` members: `incomplete_vol_structure` (only one viable leg),
`net_credit_rejected` (debit ≤ 0 — should be impossible, fail-closed anyway).

## 8. Render + UI

- `render.py`: drop STRADDLE/STRANGLE from `_DECLINED_VERDICT_PREFIXES`; add a
  position block (both legs, two breakevens, net debit/greeks). Keep the
  required-notices block logic unchanged.
- `components.py` / `app.py`: a two-leg payoff diagram (Plotly) showing the
  V-shaped long-vol payoff with both breakevens marked; verdict badge gains
  the two new colours.
- i18n: add zh/en strings for the new labels.

## 9. Test plan (TDD)

- `payoff` unit tests: breakevens, max-loss = debit, net greeks for a known
  ATM straddle (net delta ≈ 0).
- screener pairing: straddle picks ATM same-strike; strangle picks bracketing
  OTM; flag-off ⇒ zero positions (regression guard).
- validator: phantom leg ⇒ SKIP; net-credit ⇒ SKIP; numeric mismatch ⇒ SKIP;
  one-leg-only ⇒ SKIP(`incomplete_vol_structure`).
- orchestrator: end-to-end straddle verdict on a fixture; fail-closed paths.
- backward-compat: every existing single-leg test still green; ledger schema
  superset (old readers ignore `position`).
- the no-order-verb grep test stays green.

## 10. Rollout

1. Land schemas + payoff + tests (no behaviour change; flag off).
2. Land screener pairing behind `ScreenerThresholds.allow_vol=False`.
3. Land validator + render + LLM, still flag-gated.
4. Flip the CLI `--allow-vol` / UI checkbox on; document in README + CLAUDE.md
   (move "vol structures" from the deferred list to the round table).

Estimated surface: ~6 files core + ~5 test files. Single round, reviewable as
one commit, fully behind a default-off flag until the test matrix is green.

## 11. Open questions for sign-off

1. **Strangle delta band** — symmetric 0.20–0.35 each leg, or ATM-relative %
   OTM? (Affects which pair is "the" candidate.)
2. **Entry trigger** — should vol structures require an *upcoming catalyst*
   inside the expiry window (earnings/FOMC/CPI) to even be offered? (Strong
   argument: long vol without a catalyst just bleeds theta — exactly what the
   SPY example showed.)
3. **IV-rich guard** — block long straddles when IV/HV20 is already rich
   (you'd be buying expensive vol)? Ties into the deferred IV-rank filter.
