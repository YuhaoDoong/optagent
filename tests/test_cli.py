from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_version():
    r = subprocess.run(
        [sys.executable, "-m", "optagent.cli", "--version"],
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert r.returncode == 0
    assert "optagent" in r.stdout


def test_cli_help_lists_analyze():
    r = subprocess.run(
        [sys.executable, "-m", "optagent.cli", "--help"],
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert r.returncode == 0
    assert "analyze" in r.stdout
    assert "NOT FINANCIAL ADVICE" in r.stdout


def test_cli_does_not_expose_order_placement_verb():
    """Compile-time absence of order-placement code (AC-5 grep test)."""

    src_dir = REPO_ROOT / "src"
    hits: list[Path] = []
    for path in src_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("place_order", "submit_order", "new_order"):
            if forbidden in text:
                hits.append(path)
                break
    assert hits == [], f"order-placement verb found in: {hits}"
