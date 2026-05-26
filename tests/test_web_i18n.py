"""Tests for the bilingual i18n table."""

from __future__ import annotations

import pytest

from optagent.web.i18n import all_keys, supported_languages, t


def test_supported_languages_includes_en_and_zh():
    keys = [k for k, _ in supported_languages()]
    assert keys == ["en", "zh"]


def test_every_key_has_both_languages():
    from optagent.web.i18n import _TABLE

    for key, row in _TABLE.items():
        assert "en" in row, f"missing en for {key}"
        assert "zh" in row, f"missing zh for {key}"


def test_missing_key_returns_key_itself():
    # Surface missing-key bugs in the UI instead of crashing.
    assert t("does.not.exist", "en") == "does.not.exist"


def test_format_kwargs_interpolate():
    msg = t("disclaimer.banner", "en", disclaimer="RESEARCH ONLY.", version="0.4.0")
    assert "0.4.0" in msg
    assert "RESEARCH ONLY." in msg


def test_zh_returns_chinese_text():
    msg = t("tab.analyze", "zh")
    # Anything containing 单股票分析 is good enough; avoids brittle exact-match.
    assert "单股票" in msg


def test_falls_back_to_english_on_unknown_lang():
    msg = t("tab.analyze", "fr")
    assert msg == t("tab.analyze", "en")


def test_format_failure_returns_unformatted_value():
    # If the caller forgets to supply a kwarg, we don't crash — we return
    # the raw template so the page still renders.
    out = t("disclaimer.banner", "en")
    assert "{disclaimer}" in out or "{version}" in out


def test_all_keys_returns_sorted_list():
    keys = all_keys()
    assert keys == sorted(keys)
    assert len(keys) >= 30  # cover sidebar + 5 tabs minimum
