from __future__ import annotations

from datetime import datetime, timezone

import pytest

from optagent.adapters.fred_adapter import FREDAdapter, SERIES_IDS
from optagent.profiles import ensure_default_profiles
from optagent.registry import ProviderRegistry
from optagent.schemas import Confidence, RunConfig, RunMode


def _registry(run_mode=RunMode.personal_research) -> ProviderRegistry:
    r = ProviderRegistry()
    ensure_default_profiles(r)
    r.bind(RunConfig(ticker="AAPL", run_mode=run_mode))
    return r


class _FakeFredOK:
    """Stand-in for fredapi.Fred returning canned series for every id."""

    def __init__(self, mapping: dict[str, float]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def get_series_latest_release(self, series_id: str):
        self.calls.append(series_id)
        try:
            import pandas as pd

            idx = pd.date_range("2026-05-01", periods=1, tz="UTC")
            return pd.Series([self.mapping[series_id]], index=idx)
        except ImportError:
            return _FakePandasSeries([self.mapping[series_id]], ["2026-05-01T00:00:00+00:00"])


class _FakePandasSeries:
    """Tiny duck type for when pandas isn't around (it is, but be defensive)."""

    def __init__(self, values, iso_dates):
        self.values = values
        self.iso = iso_dates

    def __len__(self):
        return len(self.values)

    @property
    def iloc(self):
        class _ILoc:
            def __init__(self, parent):
                self._parent = parent

            def __getitem__(self, idx):
                return self._parent.values[idx]

        return _ILoc(self)

    @property
    def index(self):
        class _Idx:
            def __init__(self, iso_dates):
                self._iso = iso_dates

            def __getitem__(self, idx):
                class _T:
                    tz = "UTC"
                    iso = self._iso[idx]

                    def isoformat(self_inner):
                        return _T.iso

                    def tz_convert(self_inner, tz):
                        return _T

                return _T

        return _Idx(self.iso)


def test_macro_returns_ok_envelope_with_curated_series():
    fake = _FakeFredOK(
        {
            "DGS10": 4.30,
            "DGS2": 4.70,
            "VIXCLS": 14.5,
            "CPIAUCSL": 312.0,
            "FEDFUNDS": 5.25,
            "DTWEXBGS": 120.0,
        }
    )
    adapter = FREDAdapter(_registry(), fred_client=fake)
    env = adapter.get_macro()
    assert env.confidence is Confidence.ok
    assert set(env.value["readings"].keys()) == set(SERIES_IDS.keys())
    assert env.value["readings"]["ten_year_yield"]["value"] == 4.30


def test_macro_missing_key_returns_unavailable():
    adapter = FREDAdapter(_registry(), fred_client=None, api_key=None)
    env = adapter.get_macro()
    assert env.confidence is Confidence.unavailable
    assert "no_FRED_API_KEY" in (env.warnings[0] if env.warnings else "")


def test_macro_partial_failure_marks_degraded():
    class _Partial(_FakeFredOK):
        def get_series_latest_release(self, series_id):
            if series_id == "VIXCLS":
                raise RuntimeError("network")
            return super().get_series_latest_release(series_id)

    adapter = FREDAdapter(_registry(), fred_client=_Partial({k: 1.0 for k in SERIES_IDS.values()}))
    env = adapter.get_macro()
    assert env.confidence is Confidence.degraded
    assert any("VIXCLS_fetch_failed" in w for w in env.warnings)


def test_macro_blocked_when_research_only_in_distributed():
    # fred_default is production_safe so distributed is permitted; assert OK.
    adapter = FREDAdapter(_registry(run_mode=RunMode.distributed), fred_client=_FakeFredOK({k: 1.0 for k in SERIES_IDS.values()}))
    env = adapter.get_macro()
    assert env.confidence is Confidence.ok
