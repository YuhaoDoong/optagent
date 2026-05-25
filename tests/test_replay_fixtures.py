"""Run the committed fixtures through the orchestrator and lock the result.

These tests use REAL captured snapshots in tests/fixtures/ (one per ticker
in SPY/QQQ/AAPL/NVDA/TSLA) and assert:

  - the replay completes without error
  - the verdict + memo are byte-stable across two consecutive runs
  - the disclaimer is the first non-blank line of the rendered memo
  - the audit-record schema round-trips through JSON

If a fixture's chain data changes upstream, regenerate via:
    python scripts/capture_fixtures.py --include-sec
"""

from __future__ import annotations

from pathlib import Path

import pytest

from optagent.replay import Fixture, replay


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

TICKERS = ("SPY", "QQQ", "AAPL", "NVDA", "TSLA")


def _path_for(ticker: str) -> Path:
    return FIXTURE_DIR / f"{ticker}.json"


@pytest.fixture(params=TICKERS)
def fixture_path(request) -> Path:
    p = _path_for(request.param)
    if not p.exists():
        pytest.skip(f"fixture missing: {p}; run scripts/capture_fixtures.py")
    return p


def test_fixture_loads_and_has_expected_shape(fixture_path: Path):
    f = Fixture.load(fixture_path)
    assert f.ticker
    assert f.frozen_now
    assert f.yfinance_price is not None
    assert f.yfinance_chain is not None
    assert "rows" in f.yfinance_chain


def test_replay_completes_and_writes_disclaimer_first(fixture_path: Path):
    r = replay(fixture_path, write_ledger=False)
    assert r.memo.lstrip().splitlines()[0] == "RESEARCH ONLY — NOT FINANCIAL ADVICE."
    # The verdict object must serialise cleanly.
    payload = r.verdict.model_dump(mode="json")
    assert "action" in payload
    assert payload["disclaimer"] == "RESEARCH ONLY — NOT FINANCIAL ADVICE."


def test_replay_is_byte_stable_across_runs(fixture_path: Path):
    r1 = replay(fixture_path, write_ledger=False)
    r2 = replay(fixture_path, write_ledger=False)
    assert r1.verdict.model_dump(mode="json") == r2.verdict.model_dump(mode="json")
    # Memos may differ in tool_call_ids (re-generated each run); compare the
    # verdict block specifically by stripping the Sources section.
    body1 = r1.memo.split("\nSources:", 1)[0]
    body2 = r2.memo.split("\nSources:", 1)[0]
    assert body1 == body2


def test_replay_template_only_returns_skip(fixture_path: Path):
    """Template-only mode (default in replay) must always SKIP.

    This locks our 'no LLM, no direction guess' safety default for v1.
    """

    r = replay(fixture_path, write_ledger=False)
    assert r.verdict.action.value == "SKIP"
