from __future__ import annotations

import pytest

from optagent.strategies.sectors import (
    SECTOR_TICKERS,
    filter_to_sector,
    list_sectors,
    tickers_for_sector,
)


def test_list_sectors_returns_sorted_keys():
    sectors = list_sectors()
    assert sectors == sorted(sectors)
    assert "energy" in sectors
    assert "tech_chips" in sectors


def test_tickers_for_sector_known():
    energy = tickers_for_sector("energy")
    assert "XOM" in energy
    assert "CVX" in energy
    # Order is preserved from SECTOR_TICKERS
    assert energy == list(SECTOR_TICKERS["energy"])


def test_tickers_for_sector_case_insensitive():
    assert tickers_for_sector("ENERGY") == list(SECTOR_TICKERS["energy"])


def test_tickers_for_unknown_sector_raises():
    with pytest.raises(KeyError) as ei:
        tickers_for_sector("does_not_exist")
    assert "available" in str(ei.value)


def test_filter_to_sector_intersects():
    universe = ["AAPL", "XOM", "TSLA", "CVX", "ZZZ"]
    energy_only = filter_to_sector(universe, "energy")
    assert "XOM" in energy_only
    assert "CVX" in energy_only
    assert "AAPL" not in energy_only
    assert "ZZZ" not in energy_only


def test_filter_to_sector_preserves_sector_order():
    universe = ["CVX", "XOM"]  # input order
    result = filter_to_sector(universe, "energy")
    expected = [t for t in SECTOR_TICKERS["energy"] if t in {"CVX", "XOM"}]
    assert result == expected


def test_every_sector_has_tickers():
    for sector_key, tickers in SECTOR_TICKERS.items():
        assert len(tickers) > 0, f"sector {sector_key} has no tickers"
        assert all(t.isupper() for t in tickers), f"sector {sector_key} has lowercase tickers"
