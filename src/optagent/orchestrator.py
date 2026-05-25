"""End-to-end run orchestration.

Wires the data adapters into the screener, picks a verdict (template_only
mode for v1 release), writes the audit ledger, and returns the rendered memo.

No LLM call is made in v1 release; the verdict is derived deterministically
from the screener output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import DISCLAIMER
from .adapters import YFinanceAdapter
from .ledger import append as ledger_append
from .registry import ProviderRegistry
from .render import render_template
from .schemas import (
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


def _bias_from_price_history(_prices: list[float]) -> str:
    """Conservative MVP bias: always 'neutral' in template_only mode.

    Direction inference requires either OHLCV momentum logic (task5) or
    LLM synthesis (task16), neither of which ships in this round. Returning
    'neutral' enforces SKIP from the template renderer — the safest default
    for the v1 release.
    """

    return "neutral"


def _build_yfinance_profile_if_missing(registry: ProviderRegistry) -> None:
    """Register the yfinance profile if the caller did not preload it.

    Keeps `analyze()` callable from a script without forcing the user to wire
    up the YAML loader first.
    """

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


def _build_long_verdict(
    contract: OptionContract,
    citations: list[Citation],
    primary_reasons: list[str],
) -> Verdict:
    return Verdict(
        disclaimer=DISCLAIMER,
        action=(
            VerdictAction.long_call
            if contract.right.value == "call"
            else VerdictAction.long_put
        ),
        contract=contract,
        conviction=None,  # template_only mode does not assign conviction
        primary_reasons=primary_reasons,
        citations=citations,
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
) -> AnalyzeResult:
    """Run an end-to-end analysis for `ticker`.

    Parameters mirror the CLI flags. `registry` and `yfinance_adapter` are
    injectable so tests can pass fakes; production callers leave them None
    and the orchestrator builds defaults.
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

    if price_env.confidence is Confidence.unavailable or chain_env.confidence is Confidence.unavailable:
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
        )

    # Template-only mode: no LLM call → conservative bias → SKIP unless
    # the caller has opted in to a direction. We keep the contract picks
    # available for the LLM round to take over later.
    bias = _bias_from_price_history([spot])
    if bias == "neutral":
        verdict = _build_skip_verdict(
            SkipReason.no_candidates_after_screen,  # closest enum match for "no direction"
            [
                "Template-only mode (no LLM) produced 'neutral' direction; "
                "the agent defaults to SKIP rather than guessing.",
                f"{len(screener_output.candidates)} candidate(s) survived the screener; "
                "enable --enable-llm in a future round to let the LLM synthesise a verdict.",
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
        )

    # Non-neutral branches (reserved for the LLM round):
    best_call, best_put = split_by_bias(screener_output.candidates, bias)
    pick = best_call if bias == "bullish" else best_put
    if pick is None:
        verdict = _build_skip_verdict(
            SkipReason.no_candidates_after_screen,
            [f"No {bias} candidate survived the screener."],
        )
    else:
        verdict = _build_long_verdict(
            pick,
            citations=[
                Citation(tool_call_id=chain_env.tool_call_id, provider_profile_id=chain_env.provider_profile_id),
                Citation(tool_call_id=price_env.tool_call_id, provider_profile_id=price_env.provider_profile_id),
            ],
            primary_reasons=[f"Most-liquid {pick.right.value} in the {dte}-DTE window."],
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
) -> AnalyzeResult:
    """Persist the audit row and return the final result object."""

    finished_at = datetime.now(timezone.utc)

    ledger_path: Path | None = None
    if write_ledger:
        from .schemas import AuditRecord

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
            tokenizer_version=run_config.tokenizer_version,
            model_version=run_config.model_version,
            price_table_version=run_config.price_table_version,
            final_verdict=verdict,
            validator_decisions=[
                ValidatorDecision(check_id="template_only_default", passed=True),
            ],
            unavailable_data_warnings=unavailable_warnings,
            profile_versions=registry.profile_versions(),
            started_at=started_at,
            finished_at=finished_at,
        )
        ledger_path = ledger_append(record, base=ledger_dir)

    return AnalyzeResult(
        run_config=run_config,
        verdict=verdict,
        memo=memo,
        ledger_path=ledger_path,
    )
