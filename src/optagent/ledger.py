"""Audit ledger writer.

One JSONL row per run. The writer validates the row against `AuditRecord`
BEFORE appending; if validation fails OR the file cannot be written the run
aborts so we never emit a verdict that has no persistent trace.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

from .schemas import AuditRecord


DEFAULT_LEDGER_DIR = Path("data/ledger")


class LedgerError(IOError):
    """Raised when the ledger cannot be appended to (disk full / read-only)."""


def ledger_path_for(day: date | None = None, base: Path | None = None) -> Path:
    base = base or DEFAULT_LEDGER_DIR
    day = day or datetime.now(timezone.utc).date()
    return base / f"{day.isoformat()}.jsonl"


def append(record: AuditRecord, base: Path | None = None) -> Path:
    """Validate and append a single record. Returns the path written.

    Pydantic re-validation forces every required field to be present and the
    correct type at write time, independent of how the record was constructed.
    """

    # Re-validate to catch any post-construction mutation drift.
    record = AuditRecord.model_validate(record.model_dump())

    base = base or DEFAULT_LEDGER_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = ledger_path_for(record.started_at.date(), base)

    payload = record.model_dump(mode="json")
    line = json.dumps(payload, separators=(",", ":"), sort_keys=False, ensure_ascii=False)

    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        raise LedgerError(f"failed to append ledger row to {path}: {e}") from e

    return path


def read_all(path: Path) -> list[AuditRecord]:
    """Read a JSONL ledger file back into AuditRecord objects.

    Used by the evaluator (future task26) and by tests.
    """

    out: list[AuditRecord] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(AuditRecord.model_validate_json(line))
    return out
