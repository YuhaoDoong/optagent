"""Tests for the zero-dependency .env loader."""

from __future__ import annotations

from optagent.env_loader import load_dotenv


def test_loads_simple_pairs(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO=bar\nBAZ = qux\n")
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)
    applied = load_dotenv(env)
    assert applied == {"FOO": "bar", "BAZ": "qux"}


def test_shell_env_wins_by_default(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO=from_dotenv\n")
    monkeypatch.setenv("FOO", "from_shell")
    applied = load_dotenv(env)
    assert "FOO" not in applied  # not overridden
    import os

    assert os.environ["FOO"] == "from_shell"


def test_override_true_replaces(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO=from_dotenv\n")
    monkeypatch.setenv("FOO", "from_shell")
    load_dotenv(env, override=True)
    import os

    assert os.environ["FOO"] == "from_dotenv"


def test_comments_blank_lines_and_export_prefix(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nexport KEY=value\n  # indented comment\n")
    monkeypatch.delenv("KEY", raising=False)
    applied = load_dotenv(env)
    assert applied == {"KEY": "value"}


def test_strips_matching_quotes(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('A="quoted"\nB=\'single\'\nC=un"matched\n')
    for k in ("A", "B", "C"):
        monkeypatch.delenv(k, raising=False)
    applied = load_dotenv(env)
    assert applied["A"] == "quoted"
    assert applied["B"] == "single"
    assert applied["C"] == 'un"matched'  # only a full matching pair is stripped


def test_missing_file_is_noop(tmp_path):
    assert load_dotenv(tmp_path / "does-not-exist") == {}


def test_malformed_lines_skipped(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("no_equals_here\n=novalue\nGOOD=1\n")
    monkeypatch.delenv("GOOD", raising=False)
    applied = load_dotenv(env)
    assert applied == {"GOOD": "1"}
