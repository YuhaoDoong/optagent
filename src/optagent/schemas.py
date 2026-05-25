from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RunMode(str, Enum):
    personal_research = "personal_research"
    distributed = "distributed"


class Confidence(str, Enum):
    ok = "ok"
    degraded = "degraded"
    unavailable = "unavailable"


class MarketSession(str, Enum):
    pre_market = "pre_market"
    rth = "rth"
    after_hours = "after_hours"
    closed = "closed"


class PermittedUse(str, Enum):
    research_only = "research_only"
    dev_only = "dev_only"
    production_safe = "production_safe"


class Redistribution(str, Enum):
    none = "none"
    attribution = "attribution"
    paid_tier_required = "paid_tier_required"


class VerdictAction(str, Enum):
    skip = "SKIP"
    long_call = "LONG_CALL"
    long_put = "LONG_PUT"


class OptionRight(str, Enum):
    call = "call"
    put = "put"


class SkipReason(str, Enum):
    verdict_out_of_enum = "verdict_out_of_enum"
    hallucinated_contract = "hallucinated_contract"
    phantom_citation = "phantom_citation"
    numeric_grounding_mismatch = "numeric_grounding_mismatch"
    compliance_gate_failed = "compliance_gate_failed"
    stale_required_input = "stale_required_input"
    disallowed_strategy = "disallowed_strategy"
    presence_check_failed = "presence_check_failed"
    critical_provider_unavailable = "critical_provider_unavailable"
    budget_exceeded = "budget_exceeded"
    injected_instruction_followed = "injected_instruction_followed"
    no_candidates_after_screen = "no_candidates_after_screen"
    unknown_model_pricing = "unknown_model_pricing"


class ProviderProfile(BaseModel):
    """Compliance profile attached to every adapter, split BY LICENSE TIER.

    Identity is `id`; two tiers of the same provider are two distinct profiles.

    `required_notices` is a list of plain-text substrings that MUST appear in
    the rendered memo whenever any envelope from this profile is cited. The
    renderer is responsible for emitting them; the validator's presence check
    enforces them fail-closed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    permitted_use: PermittedUse
    redistribution: Redistribution
    entitlement_required: bool = False
    rate_limit_qpm: int | None = None
    attribution_string: str | None = None
    terms_url: str
    profile_version: str
    required_notices: tuple[str, ...] = ()

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not v or " " in v:
            raise ValueError("ProviderProfile.id must be non-empty and contain no spaces")
        return v

    @field_validator("required_notices", mode="before")
    @classmethod
    def _coerce_required_notices(cls, v):
        if v is None:
            return ()
        if isinstance(v, str):
            return (v,)
        return tuple(v)

    @model_validator(mode="after")
    def _attribution_required_when_redistribution_attribution(self) -> "ProviderProfile":
        if self.redistribution is Redistribution.attribution and not self.attribution_string:
            raise ValueError(
                f"provider profile {self.id!r} declares redistribution=attribution but "
                "attribution_string is empty"
            )
        return self


class Envelope(BaseModel):
    """Uniform wrapper around every adapter call output.

    Adapters never return bare values; they wrap them in an `Envelope` so the
    downstream registry, screener, validator, and ledger can reason about
    freshness, provenance, and compliance uniformly.
    """

    model_config = ConfigDict(extra="forbid")

    value: Any | None = None
    as_of: datetime
    source: str
    delay_assumption: str = Field(
        description="Plain text e.g. 'realtime', 'delayed_15min', 'eod_only'."
    )
    market_session: MarketSession
    confidence: Confidence = Confidence.ok
    provider_profile_id: str
    cache_age_s: float = 0.0
    tool_call_id: str = Field(default_factory=lambda: f"tc-{uuid.uuid4().hex}")
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    warnings: list[str] = Field(default_factory=list)

    @field_validator("as_of", "fetched_at")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Envelope timestamps must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _unavailable_implies_null(self) -> "Envelope":
        if self.confidence is Confidence.unavailable and self.value is not None:
            raise ValueError("Envelope confidence=unavailable requires value=None")
        return self


class OptionContract(BaseModel):
    """A single option contract row produced by the screener.

    Numeric fields here are the canonical source of truth — the LLM (when used)
    must copy these verbatim into its structured output.
    """

    model_config = ConfigDict(extra="forbid")

    occ_symbol: str
    underlying: str
    expiration: datetime
    strike: float
    right: OptionRight
    mid: float
    bid: float
    ask: float
    spread_pct: float
    oi: int
    volume: int
    delta: float
    theta: float
    vega: float
    iv: float
    breakeven: float
    max_loss: float
    days_to_event: int | None = None
    liquidity_score: float
    data_quality_score: float
    rejection_reason: str | None = None

    @field_validator("expiration")
    @classmethod
    def _exp_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("expiration must be timezone-aware")
        return v


class Citation(BaseModel):
    """A reference from the LLM output back to a tool call in the same run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_call_id: str
    provider_profile_id: str


class Verdict(BaseModel):
    """The structured research output returned to the user.

    `action` is bounded to {SKIP, LONG_CALL, LONG_PUT} for v1.
    """

    model_config = ConfigDict(extra="forbid")

    disclaimer: str
    action: VerdictAction
    contract: OptionContract | None = None
    conviction: float | None = Field(default=None, ge=0.0, le=1.0)
    primary_reasons: list[str] = Field(default_factory=list)
    dissenting_factors: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    skip_reason: SkipReason | None = None
    rationale_prose: str | None = None

    @model_validator(mode="after")
    def _skip_implies_no_contract(self) -> "Verdict":
        if self.action is VerdictAction.skip and self.contract is not None:
            raise ValueError("SKIP verdict must not carry a contract")
        if self.action is not VerdictAction.skip and self.contract is None:
            raise ValueError("non-SKIP verdict must carry a contract")
        if self.action is VerdictAction.skip and self.skip_reason is None:
            raise ValueError("SKIP verdict must carry a skip_reason")
        return self


class ValidatorDecision(BaseModel):
    """Per-check decision from the fail-closed post-LLM validator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    passed: bool
    detail: str | None = None


class AuditRecord(BaseModel):
    """One JSONL row in the audit ledger.

    Captures everything needed to replay or evaluate a run after the fact.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    ticker: str
    user_prefs: dict[str, Any]
    run_mode: RunMode
    envelopes: list[Envelope]
    screener_input: dict[str, Any]
    screener_output: list[OptionContract]
    prompt_version: str
    tokenizer_version: str | None = None
    model_version: str | None = None
    price_table_version: str | None = None
    budget_estimate_usd: float | None = None
    budget_actual_usd: float | None = None
    final_verdict: Verdict
    validator_decisions: list[ValidatorDecision] = Field(default_factory=list)
    unavailable_data_warnings: list[str] = Field(default_factory=list)
    profile_versions: dict[str, str] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime
    fallback_reason: str | None = None

    @field_validator("started_at", "finished_at")
    @classmethod
    def _tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return v


class RunConfig(BaseModel):
    """Immutable snapshot of run-time choices, taken at the start of a run.

    Every adapter call and every validator check resolves against this object;
    it is not mutated for the lifetime of the run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex}")
    ticker: str
    run_mode: RunMode = RunMode.personal_research
    enable_llm: bool = False
    model_version: str | None = None
    tokenizer_version: str | None = None
    prompt_version: str = "v0"
    price_table_version: str | None = None
    random_seed: int = 0
    horizon_days: int = 14
    max_loss_usd: float | None = None
    newsapi_tier: str = "free"
    moomoo_entitled: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
