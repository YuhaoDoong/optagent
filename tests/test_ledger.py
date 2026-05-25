from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from optagent.ledger import LedgerError, append, ledger_path_for, read_all
from optagent.schemas import (
    AuditRecord,
    Confidence,
    Envelope,
    MarketSession,
    OptionContract,
    OptionRight,
    RunMode,
    SkipReason,
    ValidatorDecision,
    Verdict,
    VerdictAction,
)


UTC_NOW = datetime.now(timezone.utc)


def _envelope() -> Envelope:
    return Envelope(
        value={"last": 190.5},
        as_of=UTC_NOW,
        source="yfinance",
        delay_assumption="delayed_15min",
        market_session=MarketSession.rth,
        confidence=Confidence.ok,
        provider_profile_id="yfinance_research",
    )


def _contract() -> OptionContract:
    return OptionContract(
        occ_symbol="AAPL260619C00200000",
        underlying="AAPL",
        expiration=UTC_NOW + timedelta(days=21),
        strike=200.0,
        right=OptionRight.call,
        mid=2.50,
        bid=2.45,
        ask=2.55,
        spread_pct=0.04,
        oi=12000,
        volume=1500,
        delta=0.42,
        theta=-0.05,
        vega=0.18,
        iv=0.28,
        breakeven=202.50,
        max_loss=250.0,
        liquidity_score=0.85,
        data_quality_score=0.95,
    )


def _record(disclaimer: str = "RESEARCH ONLY") -> AuditRecord:
    v = Verdict(
        disclaimer=disclaimer,
        action=VerdictAction.skip,
        skip_reason=SkipReason.no_candidates_after_screen,
    )
    return AuditRecord(
        run_id="run-abc",
        ticker="AAPL",
        user_prefs={"horizon_days": 14},
        run_mode=RunMode.personal_research,
        envelopes=[_envelope()],
        screener_input={"min_oi": 500},
        screener_output=[],
        prompt_version="v0",
        final_verdict=v,
        validator_decisions=[ValidatorDecision(check_id="x", passed=True)],
        started_at=UTC_NOW,
        finished_at=UTC_NOW + timedelta(seconds=1),
    )


def test_ledger_path_for_uses_iso_date():
    p = ledger_path_for(day=None, base=Path("/tmp/x"))
    assert p.parent == Path("/tmp/x")
    assert p.name.endswith(".jsonl")


def test_append_and_read_back(tmp_path: Path):
    rec = _record()
    written = append(rec, base=tmp_path)
    assert written.exists()
    assert written.parent == tmp_path

    back = read_all(written)
    assert len(back) == 1
    assert back[0].run_id == "run-abc"
    assert back[0].final_verdict.action.value == "SKIP"


def test_append_two_runs_appends_jsonl_lines(tmp_path: Path):
    append(_record(), base=tmp_path)
    append(_record(), base=tmp_path)
    written = ledger_path_for(UTC_NOW.date(), base=tmp_path)
    lines = written.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_append_raises_ledger_error_on_unwritable_base(tmp_path: Path):
    bad = tmp_path / "ro"
    bad.mkdir()
    bad.chmod(0o400)  # read-only directory
    try:
        with pytest.raises(LedgerError):
            append(_record(), base=bad)
    finally:
        bad.chmod(0o700)


def test_read_all_returns_empty_for_missing_file(tmp_path: Path):
    out = read_all(tmp_path / "does_not_exist.jsonl")
    assert out == []
