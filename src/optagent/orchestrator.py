"""End-to-end run orchestration.

Wires the data adapters into the screener, optionally calls the LLM for
synthesis, runs the fail-closed validator, writes the audit ledger, and
returns the rendered memo.

Default run mode is `template_only` (no LLM call). When `enable_llm=True`
AND the deterministic budget pre-check passes, the orchestrator constructs
an LLM prompt from the screener output, calls the supplied `LLMClient`,
and runs the AC-12 validator over the result. Any validator failure
downgrades the verdict to SKIP with a structured reason.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import DISCLAIMER
from .adapters import (
    EconCalendarAdapter,
    FREDAdapter,
    SECEdgarAdapter,
    VolumeOIContextAdapter,
    YahooNewsAdapter,
    YFinanceAdapter,
    news_excerpts_from_envelope,
)
from .budget import BudgetResult, precheck as budget_precheck
from .iv_history import (
    append_snapshot as iv_history_append,
    compute_iv_rank,
    median_iv_from_chain_rows,
)
from .ledger import append as ledger_append
from .ml import MLDirectionAdapter, MLDirectionSignal
from .llm import LLMClient, SYNTHESIS_PROMPT_VERSION, synthesise
from .profiles import ensure_default_profiles
from .registry import ProviderRegistry
from .render import render_template
from .schemas import (
    AuditRecord,
    Citation,
    Confidence,
    Envelope,
    OptionContract,
    RunConfig,
    SkipReason,
    ValidatorDecision,
    Verdict,
    VerdictAction,
)
from .screener import (
    ScreenerInputs,
    ScreenerOutput,
    ScreenerThresholds,
    screen,
    split_by_bias,
)
from .validator import validate


CRITICAL_PROVIDERS = ("yfinance_research",)
DEFAULT_RISK_FREE_RATE = 0.045


class AnalyzeResult:
    """Structured result returned from `analyze()`.

    Carries the envelopes / screener candidates / ml_signal in-process so
    callers (UI, downstream tooling) never need to re-read the shared
    ledger file and risk picking up another concurrent run's row.
    """

    def __init__(
        self,
        run_config: RunConfig,
        verdict: Verdict,
        memo: str,
        ledger_path: Path | None,
        envelopes: list[Envelope] | None = None,
        screener_candidates: list[OptionContract] | None = None,
        ml_signal: dict | None = None,
    ) -> None:
        self.run_config = run_config
        self.verdict = verdict
        self.memo = memo
        self.ledger_path = ledger_path
        self.envelopes = envelopes or []
        self.screener_candidates = screener_candidates or []
        self.ml_signal = ml_signal

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_config.run_id,
            "ticker": self.run_config.ticker,
            "verdict": self.verdict.model_dump(mode="json"),
            "ledger_path": str(self.ledger_path) if self.ledger_path else None,
            "memo_lines": self.memo.splitlines(),
        }


def _cites_fred(verdict: Verdict) -> bool:
    return any(c.provider_profile_id == "fred_default" for c in verdict.citations)


def _cites_volume_oi(verdict: Verdict) -> bool:
    return any(
        c.provider_profile_id == "volume_oi_context_derived" for c in verdict.citations
    )


# profile registration is delegated to optagent.profiles.ensure_default_profiles


def _build_skip_verdict(reason: SkipReason, primary_reasons: list[str]) -> Verdict:
    return Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.skip,
        skip_reason=reason,
        primary_reasons=primary_reasons,
    )


# Rejection reasons that mean "the data provider gave us no usable quote",
# as opposed to "we had real quotes but none cleared the quality bar". When
# the empty screen is dominated by these, the honest skip_reason is
# `stale_required_input` (a data problem), NOT `no_candidates_after_screen`
# (which reads like the strategy found nothing actionable).
_QUOTELESS_REJECTIONS = frozenset(
    {
        "zero_bid",
        "invalid_quote",
        "invalid_iv",
        "invalid_greeks",
        "crossed_book",
        "locked_book",
        "missing_field",
        "invalid_spot",
    }
)


def _diagnose_empty_screen(screener_output: ScreenerOutput) -> tuple[SkipReason, list[str]]:
    """Explain WHY the screener produced zero candidates.

    Distinguishes a data-provider problem (no live bid/ask, market closed,
    Yahoo returned a quote-less chain) from a legitimate "quotes were fine but
    nothing met the liquidity / delta bar". The user-facing message is the
    single most-actionable thing we can say, so we lead with the dominant
    cause and the concrete next step.
    """

    from collections import Counter

    rejected = screener_output.rejected
    summary = screener_output.inputs_summary or {}
    n_rows = int(summary.get("n_rows_in", len(rejected)) or 0)

    if n_rows == 0:
        return (
            SkipReason.stale_required_input,
            [
                "The data provider returned an EMPTY options chain (zero rows) for "
                "the chosen expiry. This is a data availability problem, not a "
                "strategy result. Re-run during US market hours (09:30–16:00 ET); "
                "if it persists the provider (yfinance/Yahoo) is degraded."
            ],
        )

    counts = Counter(reason for _occ, reason in rejected)
    quoteless = sum(n for reason, n in counts.items() if reason in _QUOTELESS_REJECTIONS)
    total = sum(counts.values()) or 1
    dominant, dominant_n = counts.most_common(1)[0]

    if quoteless / total >= 0.8:
        pct = round(100 * quoteless / total)
        exp = summary.get("expiration")
        return (
            SkipReason.stale_required_input,
            [
                f"The options chain came back WITHOUT live bid/ask quotes — "
                f"{quoteless}/{total} rows ({pct}%) had no usable two-sided quote "
                f"(dominant reason: {dominant}). This almost always means the data "
                f"provider had no live market snapshot: the US market is closed, or "
                f"yfinance/Yahoo returned only last-trade prices with zeroed "
                f"bid/ask and open-interest"
                + (f" (expiry {exp})." if exp else ".") ,
                "This is a DATA problem, not a verdict against the stock. Re-run "
                "during US regular trading hours (09:30–16:00 ET) to get real "
                "quotes; the screener intentionally refuses to act on quote-less "
                "rows rather than guess prices.",
            ],
        )

    # Genuine: quotes were present, nothing met the bar.
    bits = ", ".join(f"{reason}×{n}" for reason, n in counts.most_common(4))
    return (
        SkipReason.no_candidates_after_screen,
        [
            f"Quotes were available but no contract met the liquidity / DTE / "
            f"delta bar ({total} rows screened out; breakdown: {bits}).",
        ],
    )


def _build_template_long_verdict(
    contract: OptionContract,
    chain_env: Envelope,
    price_env: Envelope,
    dte: int,
) -> Verdict:
    return Verdict(
        disclaimer=DISCLAIMER,
        action=(
            VerdictAction.long_call
            if contract.right.value == "call"
            else VerdictAction.long_put
        ),
        contract=contract,
        conviction=None,
        primary_reasons=[f"Most-liquid {contract.right.value} in the {dte}-DTE window."],
        citations=[
            Citation(
                tool_call_id=chain_env.tool_call_id,
                provider_profile_id=chain_env.provider_profile_id,
            ),
            Citation(
                tool_call_id=price_env.tool_call_id,
                provider_profile_id=price_env.provider_profile_id,
            ),
        ],
    )


def analyze(
    ticker: str,
    *,
    registry: ProviderRegistry | None = None,
    yfinance_adapter: YFinanceAdapter | None = None,
    fred_adapter: FREDAdapter | None = None,
    econ_calendar_adapter: EconCalendarAdapter | None = None,
    sec_edgar_adapter: SECEdgarAdapter | None = None,
    volume_oi_adapter: VolumeOIContextAdapter | None = None,
    news_adapter: YahooNewsAdapter | None = None,
    ml_direction_adapter: MLDirectionAdapter | None = None,
    horizon_days: int = 14,
    max_loss_usd: float | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ledger_dir: Path | None = None,
    write_ledger: bool = True,
    enable_llm: bool = False,
    llm_client: LLMClient | None = None,
    model_version: str | None = None,
    price_table: Mapping[str, Any] | None = None,
    ttl_table: Mapping[str, Mapping[str, Any]] | None = None,
) -> AnalyzeResult:
    """Run an end-to-end analysis for `ticker`.

    `enable_llm=True` requires `llm_client`, `model_version`, `price_table`,
    and `ttl_table` to be supplied (loaded from `config/`). The LLM path:
      1. budget pre-check (AC-11) — fall back to template_only on overrun.
      2. synthesis via Claude tool_use.
      3. fail-closed validator (AC-12.a-i).
      4. ledger write + render.
    """

    started_at = datetime.now(timezone.utc)

    if registry is None:
        registry = ProviderRegistry()
    ensure_default_profiles(registry)

    run_config = RunConfig(
        ticker=ticker,
        horizon_days=horizon_days,
        max_loss_usd=max_loss_usd,
        started_at=started_at,
        enable_llm=enable_llm,
        model_version=model_version,
        prompt_version=SYNTHESIS_PROMPT_VERSION if enable_llm else "v0",
    )
    if not registry.is_bound():
        registry.bind(run_config)

    if yfinance_adapter is None:
        yfinance_adapter = YFinanceAdapter(registry)
    if econ_calendar_adapter is None:
        econ_calendar_adapter = EconCalendarAdapter(registry)
    if volume_oi_adapter is None:
        volume_oi_adapter = VolumeOIContextAdapter(registry)

    price_env = yfinance_adapter.get_price(ticker)
    chain_env = yfinance_adapter.get_options_chain(
        ticker, min_dte=max(1, horizon_days // 2), max_dte=max(horizon_days * 3, 45)
    )
    history_env = yfinance_adapter.get_history(ticker)
    econ_env = econ_calendar_adapter.get_calendar()
    spot_for_voi = None
    if price_env.confidence is not Confidence.unavailable and price_env.value:
        spot_for_voi = float(price_env.value.get("last", 0)) or None
    voi_env = volume_oi_adapter.compute(chain_env.value, spot=spot_for_voi)

    fred_env: Envelope | None = None
    if fred_adapter is not None:
        fred_env = fred_adapter.get_macro()

    sec_env: Envelope | None = None
    if sec_edgar_adapter is not None:
        sec_env = sec_edgar_adapter.get_recent_8k(ticker)

    news_env: Envelope | None = None
    if news_adapter is not None:
        news_env = news_adapter.get_news(ticker)

    envelopes: list[Envelope] = [price_env, chain_env, history_env, econ_env, voi_env]
    if fred_env is not None:
        envelopes.append(fred_env)
    if sec_env is not None:
        envelopes.append(sec_env)
    if news_env is not None:
        envelopes.append(news_env)

    unavailable_warnings = [
        env.source for env in envelopes if env.confidence is Confidence.unavailable
    ]

    # Critical-provider check
    if (
        price_env.confidence is Confidence.unavailable
        or chain_env.confidence is Confidence.unavailable
    ):
        verdict = _build_skip_verdict(
            SkipReason.critical_provider_unavailable,
            [
                f"{env.source}: {env.warnings[0] if env.warnings else 'unavailable'}"
                for env in envelopes
                if env.confidence is Confidence.unavailable
            ],
        )
        memo = render_template(verdict, envelopes, cited_fred=_cites_fred(verdict), cited_volume_oi_context=_cites_volume_oi(verdict))
        return _finalize(
            run_config,
            verdict,
            memo,
            envelopes,
            screener_output=None,
            unavailable_warnings=unavailable_warnings,
            registry=registry,
            started_at=started_at,
            ledger_dir=ledger_dir,
            write_ledger=write_ledger,
            validator_decisions=[
                ValidatorDecision(
                    check_id="critical_provider_check",
                    passed=False,
                    detail="price_or_chain_unavailable",
                )
            ],
        )

    chain_value = chain_env.value
    rows = chain_value["rows"]
    expiration_str = chain_value["expiration"]
    dte = int(chain_value["dte"])
    spot = float(price_env.value["last"])

    # Pull `days_to_event` from the calendar when available so the screener
    # can drop contracts that bracket a FOMC/CPI/NFP print.
    days_to_event: int | None = None
    if econ_env.value is not None:
        days_by_kind = econ_env.value.get("days_by_kind") or {}
        if days_by_kind:
            days_to_event = min(days_by_kind.values())

    hv20_annual: float | None = None
    if history_env.confidence is not Confidence.unavailable and history_env.value:
        hv20_annual = float(history_env.value.get("hv20_annual") or 0.0) or None

    # Persist + look up IV history so the audit ledger / screener can surface
    # an IV-rank percentile when enough observations exist.
    atm_iv_today = median_iv_from_chain_rows(rows)
    iv_rank_summary: dict | None = None
    if atm_iv_today is not None:
        iv_history_append(
            ticker,
            atm_iv_median=atm_iv_today,
            hv20_annual=hv20_annual,
            as_of=started_at,
        )
        iv_rank_summary = compute_iv_rank(ticker, current_iv=atm_iv_today)

    # Optional per-ticker ML direction signal (Alt-3 v0). Informational only —
    # the verdict path remains template_only / LLM-driven and the fail-closed
    # validator is unchanged.
    ml_signal: MLDirectionSignal | None = None
    if ml_direction_adapter is not None:
        ml_signal = ml_direction_adapter.signal(ticker)

    screener_inputs = ScreenerInputs(
        ticker=ticker,
        spot=spot,
        rows=rows,
        expiration_str=expiration_str,
        dte=dte,
        risk_free_rate=risk_free_rate,
        days_to_event=days_to_event,
        hv20_annual=hv20_annual,
        iv_rank_summary=iv_rank_summary,
        ml_signal=(_ml_signal_dict := ml_signal.to_dict() if ml_signal is not None else None),
        thresholds=ScreenerThresholds(
            min_dte=max(1, horizon_days // 2),
            max_dte=max(horizon_days * 3, 45),
        ),
    )
    screener_output = screen(screener_inputs)

    if not screener_output.candidates:
        skip_reason, primary_reasons = _diagnose_empty_screen(screener_output)
        verdict = _build_skip_verdict(skip_reason, primary_reasons)
        memo = render_template(verdict, envelopes, cited_fred=_cites_fred(verdict), cited_volume_oi_context=_cites_volume_oi(verdict))
        return _finalize(
            run_config,
            verdict,
            memo,
            envelopes,
            screener_output=screener_output,
            unavailable_warnings=unavailable_warnings,
            registry=registry,
            started_at=started_at,
            ledger_dir=ledger_dir,
            write_ledger=write_ledger,
            validator_decisions=[
                ValidatorDecision(
                    check_id="no_candidates",
                    passed=False,
                    detail=f"0 candidates; skip_reason={skip_reason.value}",
                ),
            ],
        )

    # ---- LLM path ----
    if enable_llm and llm_client is not None and price_table is not None and ttl_table is not None:
        chosen_model = model_version or price_table.get("default_model")
        return _run_llm_path(
            run_config=run_config,
            chosen_model=chosen_model,
            price_table=price_table,
            ttl_table=ttl_table,
            llm_client=llm_client,
            spot=spot,
            ticker=ticker,
            chain_env=chain_env,
            price_env=price_env,
            envelopes=envelopes,
            screener_output=screener_output,
            unavailable_warnings=unavailable_warnings,
            registry=registry,
            started_at=started_at,
            ledger_dir=ledger_dir,
            write_ledger=write_ledger,
            dte=dte,
            ml_signal=ml_signal,
        )

    # ---- template_only fall-through ----
    return _run_template_only_path(
        run_config=run_config,
        ticker=ticker,
        chain_env=chain_env,
        price_env=price_env,
        envelopes=envelopes,
        screener_output=screener_output,
        unavailable_warnings=unavailable_warnings,
        registry=registry,
        started_at=started_at,
        ledger_dir=ledger_dir,
        write_ledger=write_ledger,
        dte=dte,
        ml_signal_dict=_ml_signal_dict,
    )


def _run_template_only_path(
    *,
    run_config: RunConfig,
    ticker: str,
    chain_env: Envelope,
    price_env: Envelope,
    envelopes: list[Envelope],
    screener_output: ScreenerOutput,
    unavailable_warnings: list[str],
    registry: ProviderRegistry,
    started_at: datetime,
    ledger_dir: Path | None,
    write_ledger: bool,
    dte: int,
    ml_signal_dict: dict | None = None,
) -> AnalyzeResult:
    # template_only mode: no LLM → neutral bias → SKIP (safe default).
    verdict = _build_skip_verdict(
        SkipReason.no_candidates_after_screen,
        [
            "Template-only mode (no LLM) produced 'neutral' direction; "
            "the agent defaults to SKIP rather than guessing.",
            f"{len(screener_output.candidates)} candidate(s) survived the screener; "
            "pass --enable-llm to let the LLM synthesise a verdict.",
        ],
    )
    memo = render_template(verdict, envelopes, cited_fred=_cites_fred(verdict), cited_volume_oi_context=_cites_volume_oi(verdict))
    return _finalize(
        run_config,
        verdict,
        memo,
        envelopes,
        screener_output=screener_output,
        unavailable_warnings=unavailable_warnings,
        registry=registry,
        started_at=started_at,
        ledger_dir=ledger_dir,
        write_ledger=write_ledger,
        validator_decisions=[
            ValidatorDecision(check_id="template_only_default", passed=True, detail="no-LLM mode"),
        ],
        ml_signal_dict=ml_signal_dict,
    )


def _run_llm_path(
    *,
    run_config: RunConfig,
    chosen_model: str | None,
    price_table: Mapping[str, Any],
    ttl_table: Mapping[str, Mapping[str, Any]],
    llm_client: LLMClient,
    spot: float,
    ticker: str,
    chain_env: Envelope,
    price_env: Envelope,
    envelopes: list[Envelope],
    screener_output: ScreenerOutput,
    unavailable_warnings: list[str],
    registry: ProviderRegistry,
    started_at: datetime,
    ledger_dir: Path | None,
    write_ledger: bool,
    dte: int,
    ml_signal: MLDirectionSignal | None = None,
) -> AnalyzeResult:
    from .llm import build_user_prompt

    if not chosen_model:
        verdict = _build_skip_verdict(
            SkipReason.unknown_model_pricing,
            ["No model_version supplied and price_table has no default_model."],
        )
        memo = render_template(verdict, envelopes, cited_fred=_cites_fred(verdict), cited_volume_oi_context=_cites_volume_oi(verdict))
        return _finalize(
            run_config,
            verdict,
            memo,
            envelopes,
            screener_output=screener_output,
            unavailable_warnings=unavailable_warnings,
            registry=registry,
            started_at=started_at,
            ledger_dir=ledger_dir,
            write_ledger=write_ledger,
            validator_decisions=[
                ValidatorDecision(check_id="model_version_missing", passed=False),
            ],
        )

    # Budget pre-check (AC-11)
    preview_prompt = build_user_prompt(
        ticker=ticker,
        spot=spot,
        candidates=screener_output.candidates,
        envelopes=envelopes,
    )
    budget = budget_precheck(
        prompt_text=preview_prompt,
        model_version=chosen_model,
        price_table=price_table,
    )

    budget_decision = ValidatorDecision(
        check_id="budget_precheck",
        passed=budget.proceed,
        detail=(
            f"est=${budget.estimated_usd:.4f} tokens={budget.input_tokens} "
            f"model={chosen_model}"
            if budget.proceed
            else (budget.fallback_reason or "budget_failed")
        ),
    )

    if not budget.proceed:
        skip_reason = (
            SkipReason.unknown_model_pricing
            if budget.fallback_reason == "unknown_model_pricing"
            else SkipReason.budget_exceeded
        )
        verdict = _build_skip_verdict(
            skip_reason,
            [
                f"LLM mode requested but budget pre-check failed: {budget.fallback_reason}",
                "Falling back to template_only behaviour for this run.",
            ],
        )
        memo = render_template(verdict, envelopes, cited_fred=_cites_fred(verdict), cited_volume_oi_context=_cites_volume_oi(verdict))
        return _finalize(
            run_config,
            verdict,
            memo,
            envelopes,
            screener_output=screener_output,
            unavailable_warnings=unavailable_warnings,
            registry=registry,
            started_at=started_at,
            ledger_dir=ledger_dir,
            write_ledger=write_ledger,
            validator_decisions=[budget_decision],
            budget_estimate_usd=budget.estimated_usd,
            model_version=chosen_model,
            price_table_version=budget.price_table_version,
            tokenizer_version=budget.tokenizer_version,
            fallback_reason=budget.fallback_reason,
        )

    # Build news / SEC excerpts under prompt-injection delimiters.
    news_excerpts: list[tuple[str, str]] = []
    for env in envelopes:
        if env.source == "yahoo_news":
            news_excerpts.extend(news_excerpts_from_envelope(env))

    # Synthesis. The LLM call is the one place a third-party can crash the
    # pipeline (network error, truncated tool JSON, SDK exception). Per the
    # defer-to-SKIP invariant we catch ANY exception and emit a structured
    # SKIP rather than letting it propagate out of analyze().
    try:
        synthesis = synthesise(
            client=llm_client,
            disclaimer=DISCLAIMER,
            ticker=ticker,
            spot=spot,
            candidates=screener_output.candidates,
            envelopes=envelopes,
            news_excerpts=news_excerpts or None,
            ml_signal=ml_signal.to_dict() if ml_signal is not None else None,
            max_output_tokens=budget.max_output_tokens,
        )
    except Exception as e:  # noqa: BLE001 — fail closed on any LLM error
        verdict = _build_skip_verdict(
            SkipReason.critical_provider_unavailable,
            [
                f"LLM synthesis failed ({type(e).__name__}: {str(e)[:160]}). The "
                "agent fails closed to SKIP rather than guess a verdict.",
                "Re-run; if it persists, check the provider/model, network, or "
                "raise the output-token budget.",
            ],
        )
        memo = render_template(
            verdict, envelopes, cited_fred=_cites_fred(verdict), cited_volume_oi_context=_cites_volume_oi(verdict)
        )
        return _finalize(
            run_config,
            verdict,
            memo,
            envelopes,
            screener_output=screener_output,
            unavailable_warnings=unavailable_warnings,
            registry=registry,
            started_at=started_at,
            ledger_dir=ledger_dir,
            write_ledger=write_ledger,
            validator_decisions=[
                budget_decision,
                ValidatorDecision(
                    check_id="llm_synthesis", passed=False, detail=str(e)[:200]
                ),
            ],
            budget_estimate_usd=budget.estimated_usd,
            model_version=chosen_model,
            price_table_version=budget.price_table_version,
            tokenizer_version=budget.tokenizer_version,
        )

    pre_render = render_template(
        synthesis.verdict, envelopes, cited_fred=_cites_fred(synthesis.verdict), cited_volume_oi_context=_cites_volume_oi(synthesis.verdict)
    )

    outcome = validate(
        verdict=synthesis.verdict,
        candidates=screener_output.candidates,
        envelopes=envelopes,
        llm_tool_input=synthesis.tool_input,
        registry=registry,
        ttl_table=ttl_table,
        rendered_output=pre_render,
    )

    final_verdict = outcome.final_verdict
    final_memo = render_template(
        final_verdict, envelopes, cited_fred=_cites_fred(final_verdict), cited_volume_oi_context=_cites_volume_oi(final_verdict)
    )

    return _finalize(
        run_config,
        final_verdict,
        final_memo,
        envelopes,
        screener_output=screener_output,
        unavailable_warnings=unavailable_warnings,
        registry=registry,
        started_at=started_at,
        ledger_dir=ledger_dir,
        write_ledger=write_ledger,
        validator_decisions=[budget_decision] + outcome.decisions,
        budget_estimate_usd=budget.estimated_usd,
        model_version=chosen_model,
        price_table_version=budget.price_table_version,
        tokenizer_version=budget.tokenizer_version,
    )


def _finalize(
    run_config: RunConfig,
    verdict: Verdict,
    memo: str,
    envelopes: list[Envelope],
    *,
    screener_output: ScreenerOutput | None,
    unavailable_warnings: list[str],
    registry: ProviderRegistry,
    started_at: datetime,
    ledger_dir: Path | None,
    write_ledger: bool,
    validator_decisions: list[ValidatorDecision] | None = None,
    budget_estimate_usd: float | None = None,
    model_version: str | None = None,
    price_table_version: str | None = None,
    tokenizer_version: str | None = None,
    fallback_reason: str | None = None,
    ml_signal_dict: dict | None = None,
) -> AnalyzeResult:
    """Persist the audit row and return the final result object."""

    finished_at = datetime.now(timezone.utc)

    ledger_path: Path | None = None
    if write_ledger:
        record = AuditRecord(
            run_id=run_config.run_id,
            ticker=run_config.ticker,
            user_prefs={
                "horizon_days": run_config.horizon_days,
                "max_loss_usd": run_config.max_loss_usd,
            },
            run_mode=run_config.run_mode,
            envelopes=envelopes,
            screener_input=(screener_output.inputs_summary if screener_output else {}),
            screener_output=(screener_output.candidates if screener_output else []),
            prompt_version=run_config.prompt_version,
            tokenizer_version=tokenizer_version or run_config.tokenizer_version,
            model_version=model_version or run_config.model_version,
            price_table_version=price_table_version or run_config.price_table_version,
            budget_estimate_usd=budget_estimate_usd,
            final_verdict=verdict,
            validator_decisions=validator_decisions
            or [ValidatorDecision(check_id="default", passed=True)],
            unavailable_data_warnings=unavailable_warnings,
            profile_versions=registry.profile_versions(),
            started_at=started_at,
            finished_at=finished_at,
            fallback_reason=fallback_reason,
        )
        ledger_path = ledger_append(record, base=ledger_dir)

    return AnalyzeResult(
        run_config=run_config,
        verdict=verdict,
        memo=memo,
        ledger_path=ledger_path,
        envelopes=list(envelopes),
        screener_candidates=(list(screener_output.candidates) if screener_output else []),
        ml_signal=ml_signal_dict,
    )
