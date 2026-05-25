from __future__ import annotations

import pytest

from optagent.registry import (
    MissingProviderProfileError,
    ProviderRegistry,
    RegistryNotBoundError,
)
from optagent.schemas import (
    PermittedUse,
    ProviderProfile,
    Redistribution,
    RunConfig,
    RunMode,
)


def _yfinance_research() -> ProviderProfile:
    return ProviderProfile(
        id="yfinance_research",
        permitted_use=PermittedUse.research_only,
        redistribution=Redistribution.none,
        terms_url="https://pypi.org/project/yfinance/",
        profile_version="2026-05-25",
    )


def _newsapi_free_dev() -> ProviderProfile:
    return ProviderProfile(
        id="newsapi_free_dev",
        permitted_use=PermittedUse.dev_only,
        redistribution=Redistribution.none,
        terms_url="https://newsapi.org/terms",
        profile_version="2026-05-25",
    )


def _newsapi_paid_production() -> ProviderProfile:
    return ProviderProfile(
        id="newsapi_paid_production",
        permitted_use=PermittedUse.production_safe,
        redistribution=Redistribution.paid_tier_required,
        terms_url="https://newsapi.org/terms",
        profile_version="2026-05-25",
    )


def _fred_default() -> ProviderProfile:
    return ProviderProfile(
        id="fred_default",
        permitted_use=PermittedUse.production_safe,
        redistribution=Redistribution.attribution,
        attribution_string="Data sourced from FRED.",
        terms_url="https://fred.stlouisfed.org/docs/api/terms_of_use.html",
        profile_version="2026-05-25",
    )


def _moomoo_user_entitled() -> ProviderProfile:
    return ProviderProfile(
        id="moomoo_user_entitled",
        permitted_use=PermittedUse.research_only,
        redistribution=Redistribution.none,
        entitlement_required=True,
        terms_url="https://openapi.moomoo.com/futu-api-doc/en/intro/authority.html",
        profile_version="2026-05-25",
    )


def test_register_then_get():
    r = ProviderRegistry()
    p = _yfinance_research()
    r.register(p)
    assert r.get("yfinance_research") is p
    assert r.ids() == ["yfinance_research"]


def test_register_rejects_duplicate_id():
    r = ProviderRegistry()
    r.register(_yfinance_research())
    with pytest.raises(ValueError):
        r.register(_yfinance_research())


def test_get_missing_profile_raises():
    r = ProviderRegistry()
    with pytest.raises(MissingProviderProfileError):
        r.get("does_not_exist")


def test_gate_before_bind_raises():
    r = ProviderRegistry()
    r.register(_yfinance_research())
    with pytest.raises(RegistryNotBoundError):
        r.gate("yfinance_research")


def test_gate_research_only_permitted_in_personal():
    r = ProviderRegistry()
    r.register(_yfinance_research())
    r.bind(RunConfig(ticker="AAPL", run_mode=RunMode.personal_research))
    result = r.gate("yfinance_research")
    assert result.ok is True
    assert result.reason is None


def test_gate_research_only_blocked_in_distributed():
    r = ProviderRegistry()
    r.register(_yfinance_research())
    r.bind(RunConfig(ticker="AAPL", run_mode=RunMode.distributed))
    result = r.gate("yfinance_research")
    assert result.ok is False
    assert result.reason is not None
    assert "research_only" in result.reason


def test_gate_dev_only_blocked_in_distributed_but_permitted_in_personal():
    r = ProviderRegistry()
    r.register_many([_newsapi_free_dev()])
    r.bind(RunConfig(ticker="AAPL", run_mode=RunMode.distributed))
    assert r.gate("newsapi_free_dev").ok is False

    r2 = ProviderRegistry()
    r2.register_many([_newsapi_free_dev()])
    r2.bind(RunConfig(ticker="AAPL", run_mode=RunMode.personal_research))
    assert r2.gate("newsapi_free_dev").ok is True


def test_gate_production_safe_passes_in_distributed():
    r = ProviderRegistry()
    r.register_many([_newsapi_paid_production(), _fred_default()])
    r.bind(RunConfig(ticker="AAPL", run_mode=RunMode.distributed))
    assert r.gate("newsapi_paid_production").ok is True
    assert r.gate("fred_default").ok is True


def test_gate_blocks_moomoo_without_entitlement_even_in_personal():
    r = ProviderRegistry()
    r.register(_moomoo_user_entitled())
    r.bind(RunConfig(ticker="AAPL", run_mode=RunMode.personal_research, moomoo_entitled=False))
    result = r.gate("moomoo_user_entitled")
    assert result.ok is False
    assert "entitlement" in (result.reason or "")


def test_gate_passes_moomoo_when_user_entitled():
    r = ProviderRegistry()
    r.register(_moomoo_user_entitled())
    r.bind(RunConfig(ticker="AAPL", run_mode=RunMode.personal_research, moomoo_entitled=True))
    assert r.gate("moomoo_user_entitled").ok is True


def test_double_bind_raises():
    r = ProviderRegistry()
    r.bind(RunConfig(ticker="AAPL"))
    with pytest.raises(RuntimeError):
        r.bind(RunConfig(ticker="TSLA"))


def test_profile_versions_snapshot():
    r = ProviderRegistry()
    r.register_many([_yfinance_research(), _fred_default()])
    versions = r.profile_versions()
    assert versions == {
        "yfinance_research": "2026-05-25",
        "fred_default": "2026-05-25",
    }
