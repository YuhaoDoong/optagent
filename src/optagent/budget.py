"""Deterministic worst-case LLM budget pre-check.

Before any LLM call we compute:

    estimated_cost = (input_tokens + max_output_tokens)
                     × (max_retries + 1)
                     × price_per_token[model_version]
                     × (1 + safety_margin)

If the estimate exceeds the configured cap OR the model version is missing
from the price table, the run falls back to `template_only` mode and records
a structured `fallback_reason` in the audit ledger.

The token estimator is pluggable so tests can inject a deterministic counter
without taking a `tiktoken` import dependency. Production callers can pass a
real tokenizer; default `_byte_pair_estimate` is a cheap upper-bound estimate
based on byte length, suitable for budget gating (never under-counts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


class TokenEstimator(Protocol):
    """Pluggable token-count function. Must NEVER under-count (budget safety)."""

    def __call__(self, text: str, *, model_version: str) -> int:
        ...


def _byte_pair_estimate(text: str, *, model_version: str) -> int:
    """Conservative tokens-per-character upper bound.

    Real Claude tokenization averages ~3.5 chars/token for English. We use
    2.5 chars/token to guarantee we OVER-count for budget purposes — better
    to skip an LLM call we could have afforded than to overshoot the cap.
    """

    if not text:
        return 0
    return math.ceil(len(text) / 2.5)


@dataclass(frozen=True)
class BudgetLimits:
    max_input_tokens: int
    max_output_tokens: int
    max_retries: int
    timeout_s: int
    safety_margin: float
    cap_usd: float


@dataclass(frozen=True)
class BudgetResult:
    """Outcome of `precheck()` consumed by the orchestrator.

    `proceed` is True only when the run should actually call the LLM. When
    False, `fallback_reason` carries a stable string the ledger persists.
    """

    proceed: bool
    estimated_usd: float
    fallback_reason: str | None
    input_tokens: int
    max_output_tokens: int
    model_version: str
    tokenizer_version: str | None
    price_table_version: str


def limits_from_price_table(price_table: Mapping[str, Any]) -> BudgetLimits:
    """Translate the YAML `limits:` block into a `BudgetLimits` dataclass.

    `cap_usd` defaults to 5.00 if not present in the YAML — most single
    research-memo runs cost well under one cent at current Claude pricing,
    so a $5 cap is a comfortable ceiling that still catches runaway inputs.
    """

    lim = price_table.get("limits", {})
    return BudgetLimits(
        max_input_tokens=int(lim.get("max_input_tokens", 60_000)),
        max_output_tokens=int(lim.get("max_output_tokens", 2_000)),
        max_retries=int(lim.get("max_retries", 2)),
        timeout_s=int(lim.get("timeout_s", 45)),
        safety_margin=float(lim.get("safety_margin", 0.20)),
        cap_usd=float(lim.get("cap_usd", 5.00)),
    )


def precheck(
    *,
    prompt_text: str,
    model_version: str,
    price_table: Mapping[str, Any],
    estimator: TokenEstimator | None = None,
) -> BudgetResult:
    """Compute the worst-case cost and decide whether to proceed.

    `price_table` is the dict loaded from `config/price_table.yaml`; the
    keys we care about are `models[model_version]` (input/output USD per
    Mtok + tokenizer_version) and `limits` (see `BudgetLimits`).
    """

    estimator = estimator or _byte_pair_estimate
    limits = limits_from_price_table(price_table)
    price_table_version = str(price_table.get("price_table_version", "unknown"))

    models = price_table.get("models", {})
    model_entry = models.get(model_version)
    if model_entry is None:
        return BudgetResult(
            proceed=False,
            estimated_usd=0.0,
            fallback_reason="unknown_model_pricing",
            input_tokens=0,
            max_output_tokens=limits.max_output_tokens,
            model_version=model_version,
            tokenizer_version=None,
            price_table_version=price_table_version,
        )

    tokenizer_version = model_entry.get("tokenizer_version")
    input_usd_per_mtok = float(model_entry["input_usd_per_mtok"])
    output_usd_per_mtok = float(model_entry["output_usd_per_mtok"])

    input_tokens = estimator(prompt_text, model_version=model_version)

    if input_tokens > limits.max_input_tokens:
        return BudgetResult(
            proceed=False,
            estimated_usd=0.0,
            fallback_reason=f"input_tokens_exceeded:{input_tokens}>{limits.max_input_tokens}",
            input_tokens=input_tokens,
            max_output_tokens=limits.max_output_tokens,
            model_version=model_version,
            tokenizer_version=tokenizer_version,
            price_table_version=price_table_version,
        )

    attempts = limits.max_retries + 1
    raw_input_cost = (input_tokens / 1_000_000.0) * input_usd_per_mtok
    raw_output_cost = (limits.max_output_tokens / 1_000_000.0) * output_usd_per_mtok
    estimated_usd = (raw_input_cost + raw_output_cost) * attempts * (1.0 + limits.safety_margin)

    if estimated_usd > limits.cap_usd:
        return BudgetResult(
            proceed=False,
            estimated_usd=estimated_usd,
            fallback_reason=f"budget_exceeded:est=${estimated_usd:.4f}>cap=${limits.cap_usd:.2f}",
            input_tokens=input_tokens,
            max_output_tokens=limits.max_output_tokens,
            model_version=model_version,
            tokenizer_version=tokenizer_version,
            price_table_version=price_table_version,
        )

    return BudgetResult(
        proceed=True,
        estimated_usd=estimated_usd,
        fallback_reason=None,
        input_tokens=input_tokens,
        max_output_tokens=limits.max_output_tokens,
        model_version=model_version,
        tokenizer_version=tokenizer_version,
        price_table_version=price_table_version,
    )
