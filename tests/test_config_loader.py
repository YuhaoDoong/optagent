from __future__ import annotations

from pathlib import Path

import pytest

from optagent.config_loader import find_config_dir, load_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_finds_config_dir_from_repo_root():
    base = find_config_dir(REPO_ROOT / "config")
    assert base == REPO_ROOT / "config"
    for name in ("providers.yaml", "ttl_table.yaml", "price_table.yaml"):
        assert (base / name).exists()


def test_load_bundle_provides_six_built_in_profiles():
    bundle = load_bundle(REPO_ROOT / "config")
    ids = sorted(p.id for p in bundle.providers)
    assert ids == [
        "fred_default",
        "moomoo_user_entitled",
        "newsapi_free_dev",
        "newsapi_paid_production",
        "sec_edgar_default",
        "yfinance_research",
    ]


def test_ttl_table_marks_price_and_chain_as_critical():
    bundle = load_bundle(REPO_ROOT / "config")
    assert bundle.ttl_table["price"]["critical"] is True
    assert bundle.ttl_table["options_chain"]["critical"] is True
    assert bundle.ttl_table["macro"]["critical"] is False


def test_price_table_includes_default_model_and_limits():
    bundle = load_bundle(REPO_ROOT / "config")
    pt = bundle.price_table
    assert pt["default_model"] == "claude-opus-4-7"
    assert "claude-opus-4-7" in pt["models"]
    assert pt["limits"]["max_input_tokens"] == 60000
    assert pt["limits"]["safety_margin"] == 0.20


def test_missing_config_dir_raises():
    with pytest.raises(FileNotFoundError):
        find_config_dir(REPO_ROOT / "nonexistent_directory_for_test")
