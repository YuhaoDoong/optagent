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
from .adapters import YFinanceAdapter
from .budget import BudgetResult, precheck as budget_precheck
from .ledger import append as ledger_append
from .llm import LLMClient, SYNTHESIS_PROMPT_VERSION, synthesise
from .registry import ProviderRegistry
from .render import render_template
from .schemas import (
    AuditRecord,
    Citation,
    Confidence,
    Envelope,
    OptionContract,
    ProviderProfile,
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
    """Structured result returned from `analyze()`."""

    def __init__(
        self,
        run_config: RunConfig,
        verdict: Verdict,
        memo: str,
        ledger_path: Path | None,
    ) -> None:
        self.run_config = run_config
        self.verdict = verdict
        self.memo = memo
        self.ledger_path = ledger_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_config.run_id,
            "ticker": self.run_config.ticker,
            "verdict": self.verdict.model_dump(mode="json"),
            "ledger_path": str(self.ledger_path) if self.ledger_path else None,
            "memo_lines": self.memo.splitlines(),
        }


def _build_yfinance_profile_if_missing(registry: ProviderRegistry) -> None:
    try:
        registry.get("yfinance_research")
    except LookupError:
        registry.register(
            ProviderProfile(
                id="yfinance_research",
                permitted_use="research_only",  # type: ignore[arg-type]
                redistribution="none",  # type: ignore[arg-type]
                terms_url="https://pypi.org/project/yfinance/",
                profile_version="2026-05-25",
            )
        )


def _build_skip_verdict(reason: SkipReason, primary_reasons: list[str]) -> Verdict:
    return Verdict(
        disclaimer=DISCLAIMER,
        action=VerdictAction.skip,
        skip_reason=reason,
        primary_reasons=primary_reasons,
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
    _build_yfinance_profile_if_missing(registry)

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

    price_env = yfinance_adapter.get_price(ticker)
    chain_env = yfinance_adapter.get_options_chain(
        ticker, min_dte=max(1, horizon_days // 2), max_dte=max(horizon_days * 3, 45)
    )
    envelopes: list[Envelope] = [price_env, chain_env]
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
        memo = render_template(verdict, envelopes)
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

    screener_inputs = ScreenerInputs(
        ticker=ticker,
        spot=spot,
        rows=rows,
        expiration_str=expiration_str,
        dte=dte,
        risk_free_rate=risk_free_rate,
        thresholds=ScreenerThresholds(
            min_dte=max(1, horizon_days // 2),
            max_dte=max(horizon_days * 3, 45),
        ),
    )
    screener_output = screen(screener_inputs)

    if not screener_output.candidates:
        verdict = _build_skip_verdict(
            SkipReason.no_candidates_after_screen,
            ["Screener produced zero candidates after liquidity + DTE filters."],
        )
        memo = render_template(verdict, envelopes)
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
                ValidatorDecision(check_id="no_candidates", passed=False, detail="0 candidates"),
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
    memo = render_template(verdict, envelopes)
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
) -> AnalyzeResult:
    from .llm import build_user_prompt

    if not chosen_model:
        verdict = _build_skip_verdict(
            SkipReason.unknown_model_pricing,
            ["No model_version supplied and price_table has no default_model."],
        )
        memo = render_template(verdict, envelopes)
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
        memo = render_template(verdict, envelopes)
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

    # Synthesis
    synthesis = synthesise(
        client=llm_client,
        disclaimer=DISCLAIMER,
        ticker=ticker,
        spot=spot,
        candidates=screener_output.candidates,
        envelopes=envelopes,
        max_output_tokens=budget.max_output_tokens,
    )

    pre_render = render_template(synthesis.verdict, envelopes)

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
    final_memo = render_template(final_verdict, envelopes)

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
    )
