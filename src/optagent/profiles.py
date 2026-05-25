"""Centralised provider-profile registration.

Calling `ensure_default_profiles(registry)` registers every v1 profile the
orchestrator might cite. Each adapter is responsible only for declaring its
`profile_id`; this module owns the mapping from id to `ProviderProfile`.

In a future round these constants will be loaded directly from
`config/providers.yaml` (the schema is already wired in `config_loader.py`).
Keeping them in code lets the orchestrator work in environments that have
not yet shipped the YAML files (notably the test suite).
"""

from __future__ import annotations

from .registry import ProviderRegistry
from .schemas import PermittedUse, ProviderProfile, Redistribution


_FRED_ATTRIBUTION = "Data sourced from the Federal Reserve Bank of St. Louis (FRED)."
_FRED_NON_ENDORSEMENT = (
    "This product uses the FRED® API but is not endorsed or certified by "
    "the Federal Reserve Bank of St. Louis."
)


_DEFAULT_PROFILES: list[ProviderProfile] = [
    ProviderProfile(
        id="yfinance_research",
        permitted_use=PermittedUse.research_only,
        redistribution=Redistribution.none,
        terms_url="https://pypi.org/project/yfinance/",
        profile_version="2026-05-25",
    ),
    ProviderProfile(
        id="fred_default",
        permitted_use=PermittedUse.production_safe,
        redistribution=Redistribution.attribution,
        rate_limit_qpm=120,
        attribution_string=_FRED_ATTRIBUTION,
        terms_url="https://fred.stlouisfed.org/docs/api/terms_of_use.html",
        profile_version="2026-05-26",
        required_notices=(_FRED_ATTRIBUTION, _FRED_NON_ENDORSEMENT),
    ),
    ProviderProfile(
        id="sec_edgar_default",
        permitted_use=PermittedUse.production_safe,
        redistribution=Redistribution.none,
        rate_limit_qpm=600,
        terms_url="https://www.sec.gov/about/developer-resources",
        profile_version="2026-05-25",
    ),
    ProviderProfile(
        id="econ_calendar_builtin",
        permitted_use=PermittedUse.production_safe,
        redistribution=Redistribution.none,
        terms_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        profile_version="2026-05-25",
    ),
    ProviderProfile(
        id="volume_oi_context_derived",
        permitted_use=PermittedUse.production_safe,
        redistribution=Redistribution.none,
        terms_url="https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document",
        profile_version="2026-05-25",
    ),
    ProviderProfile(
        id="moomoo_user_entitled",
        permitted_use=PermittedUse.research_only,
        redistribution=Redistribution.none,
        entitlement_required=True,
        terms_url="https://openapi.moomoo.com/moomoo-api-doc/en/intro/authority.html",
        profile_version="2026-05-25",
    ),
    ProviderProfile(
        id="newsapi_free_dev",
        permitted_use=PermittedUse.dev_only,
        redistribution=Redistribution.none,
        terms_url="https://newsapi.org/terms",
        profile_version="2026-05-25",
    ),
    ProviderProfile(
        id="newsapi_paid_production",
        permitted_use=PermittedUse.production_safe,
        redistribution=Redistribution.paid_tier_required,
        entitlement_required=True,
        terms_url="https://newsapi.org/terms",
        profile_version="2026-05-25",
    ),
]


def ensure_default_profiles(registry: ProviderRegistry) -> None:
    """Idempotently register every v1 profile that isn't already present."""

    existing = set(registry.ids())
    for p in _DEFAULT_PROFILES:
        if p.id in existing:
            continue
        registry.register(p)
