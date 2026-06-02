"""Session research store + grounding-context builder + cross-strategy synthesis.

Pure, Streamlit-free, JSON-serializable. The Streamlit layer (`app.py`) keeps
one of these dicts in `st.session_state` and feeds normalized snapshots in as
each view runs; the chat panel turns the store into a single grounding block.

Design rules enforced here:
  - Snapshots store COMPACT normalized data (not rendered markdown) plus a
    `computed_at` ISO timestamp, an availability flag, and a `stale` marker.
  - `build_context` merges only AVAILABLE snapshots, explicitly names missing
    sections, escapes untrusted strings so they cannot break out of the
    `<analysis_context>` delimiter, and deterministically truncates the whole
    block to a character cap.
  - Cross-strategy ranking is DETERMINISTIC and LLM-free: resonance (how many
    strategies triggered a ticker) first, then summed within-strategy
    normalized score, then ticker alphabetical. The LLM may only explain this
    ordering — it never produces it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1

# Whole-context character cap (deterministic truncation guard for the prompt).
MAX_CONTEXT_CHARS = 8000
# Per-field cap so one giant note can't dominate / blow the budget.
_FIELD_CAP = 400

_OPEN = "<analysis_context>"
_CLOSE = "</analysis_context>"


# ---------------------------------------------------------------------------
# Serialization + injection safety


def json_safe(obj: Any) -> Any:
    """Recursively coerce `obj` to something `json.dumps` can handle.

    Unknown / non-serializable values become their `str()`. This guarantees a
    snapshot never raises at `json.dumps` time even if an upstream object
    sneaks in.
    """

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return str(obj)


# Instruction-like phrases an attacker might smuggle through untrusted fields
# (ticker names, notes, provider text). We don't try to be exhaustive — we
# defang the common prompt-injection openers by inserting a zero-width space
# after the first character so the literal phrase no longer reads as a command.
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:the\s+)?previous",
    r"disregard\s+(?:all\s+)?(?:the\s+)?previous",
    r"forget\s+(?:all\s+)?(?:the\s+)?previous",
    r"system\s+prompt",
    r"you\s+are\s+now",
    r"new\s+instructions?",
    r"override\s+(?:all\s+)?instructions?",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_ZWSP = "​"


def _defang_injection(text: str) -> str:
    def _ins(m: "re.Match[str]") -> str:
        tok = m.group(0)
        return tok[:1] + _ZWSP + tok[1:]

    return _INJECTION_RE.sub(_ins, text)


def neutralize_text(text: Any) -> str:
    """Make a possibly-untrusted string safe to embed in a context block.

    Strips control characters, escapes angle brackets (so embedded text can
    never open/close the `<analysis_context>` delimiter or any pseudo-tag), and
    defangs common instruction-injection phrases. No length cap — callers apply
    their own. Non-strings are stringified first.
    """

    s = text if isinstance(text, str) else str(text)
    s = "".join(c for c in s if c >= " " or c in "\n\t")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return _defang_injection(s)


def escape_untrusted(text: Any, *, cap: int = _FIELD_CAP) -> str:
    """`neutralize_text` plus a per-field length cap (for line-oriented use)."""

    s = neutralize_text(text)
    if len(s) > cap:
        s = s[:cap] + "…"
    return s


def _wrap_bounded(body: str, max_chars: int) -> str:
    """Wrap `body` in the `<analysis_context>` delimiters, guaranteeing the
    WHOLE result is <= max_chars AND the wrapper stays well-formed.

    Truncates the body (never the delimiters). Raises ValueError if `max_chars`
    is too small to hold even an empty wrapped block — we refuse to emit a
    partial/closing-tag-less delimiter rather than ship malformed prompt
    framing.
    """

    overhead = len(_OPEN) + len(_CLOSE) + 2  # the two newlines around the body
    if max_chars < overhead:
        raise ValueError(
            f"max_chars={max_chars} too small for a valid wrapped block "
            f"(need >= {overhead})"
        )
    budget = max_chars - overhead
    if len(body) > budget:
        marker = "\n…(truncated)"
        body = body[: budget - len(marker)] + marker if budget > len(marker) else body[:budget]
    return f"{_OPEN}\n{body}\n{_CLOSE}"


def serialize_bundle(bundle: Mapping[str, Any] | None, *, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """One bounded, sanitized, single-wrapper serializer for an arbitrary dict.

    Reused on every LLM grounding path (chat legacy bundle fallback + screen
    explanation) so untrusted text is always neutralized, the block is always
    bounded, and there is always exactly one well-formed delimiter pair.
    """

    payload = json.dumps(json_safe(dict(bundle or {})), ensure_ascii=False, indent=2)
    return _wrap_bounded(neutralize_text(payload), max_chars)


# ---------------------------------------------------------------------------
# Store


def init_store() -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "screen": None,
        "analysis": {},   # ticker -> snapshot
        "ml": {},         # ticker -> snapshot
        "ledger": None,
        "active_ticker": None,
        "pending_drilldown": None,  # {"ticker": str, "target": "ml"|"analyze"}
    }


def _base_snapshot(kind: str, computed_at: str, available: bool, stale: bool) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "kind": kind,
        "computed_at": computed_at,
        "available": available,
        "stale": bool(stale),
    }


def screen_snapshot(
    results_by_strategy: Mapping[str, Mapping[str, Any]],
    synthesis: Sequence[Mapping[str, Any]] | None,
    computed_at: str,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a multi-strategy screen run into a compact snapshot.

    `results_by_strategy[strategy_id]` is expected to carry `signals` (list of
    compact signal dicts), `error` (str|None), `n_triggered`, and optionally a
    `stale_tickers` list. Empty / falsy input yields an unavailable snapshot.
    `inputs` records the run parameters (strategies, sector, limit).
    """

    if not results_by_strategy:
        snap = _base_snapshot("screen", computed_at, available=False, stale=False)
        snap["inputs"] = json_safe(dict(inputs or {}))
        return snap
    strategies = {}
    any_stale = False
    for sid, res in results_by_strategy.items():
        res = res or {}
        stale_tickers = list(res.get("stale_tickers") or [])
        any_stale = any_stale or bool(stale_tickers)
        strategies[str(sid)] = json_safe(
            {
                "error": res.get("error"),
                "n_triggered": res.get("n_triggered"),
                "n_evaluated": res.get("n_evaluated"),
                "signals": [
                    {
                        "ticker": s.get("ticker"),
                        "direction": s.get("direction"),
                        "score": s.get("score"),
                        "notes": (s.get("notes") or [])[:3],
                    }
                    for s in (res.get("signals") or [])[:10]
                ],
                "stale_tickers": stale_tickers,
            }
        )
    snap = _base_snapshot("screen", computed_at, available=True, stale=any_stale)
    snap["inputs"] = json_safe(dict(inputs or {}))
    snap["strategies"] = strategies
    snap["synthesis"] = json_safe(list(synthesis or []))
    return snap


def analysis_snapshot(
    ticker: str,
    verdict: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]] | None,
    computed_at: str,
    inputs: Mapping[str, Any] | None = None,
    envelopes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not ticker:
        snap = _base_snapshot("analysis", computed_at, available=False, stale=False)
        snap["inputs"] = json_safe(dict(inputs or {}))
        return snap
    snap = _base_snapshot("analysis", computed_at, available=verdict is not None, stale=False)
    snap["ticker"] = ticker
    snap["inputs"] = json_safe(dict(inputs or {}))
    snap["verdict"] = json_safe(verdict) if verdict is not None else None
    snap["candidates"] = json_safe(
        [
            {
                "occ_symbol": c.get("occ_symbol"),
                "right": c.get("right"),
                "strike": c.get("strike"),
                "mid": c.get("mid"),
                "delta": c.get("delta"),
                "iv": c.get("iv"),
                "breakeven": c.get("breakeven"),
                "max_loss": c.get("max_loss"),
            }
            for c in (candidates or [])[:8]
        ]
    )
    # Compact upstream-status summary so chat can cite data provenance/freshness.
    snap["sources"] = json_safe(
        [
            {
                "source": e.get("source"),
                "confidence": e.get("confidence"),
                "as_of": e.get("as_of"),
            }
            for e in (envelopes or [])[:10]
        ]
    )
    return snap


def ml_snapshot(
    ticker: str,
    ml: Mapping[str, Any] | None,
    computed_at: str,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not ticker:
        snap = _base_snapshot("ml", computed_at, available=False, stale=False)
        snap["inputs"] = json_safe(dict(inputs or {}))
        return snap
    snap = _base_snapshot("ml", computed_at, available=ml is not None, stale=False)
    snap["ticker"] = ticker
    snap["inputs"] = json_safe(dict(inputs or {"ticker": ticker}))
    snap["signal"] = json_safe(ml) if ml is not None else None
    return snap


def ledger_summary_snapshot(
    action_counts: Mapping[str, int] | None,
    n_rows: int,
    computed_at: str,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snap = _base_snapshot("ledger", computed_at, available=bool(action_counts), stale=False)
    snap["inputs"] = json_safe(dict(inputs or {}))
    snap["n_rows"] = int(n_rows or 0)
    snap["action_counts"] = json_safe(dict(action_counts or {}))
    return snap


# ---------------------------------------------------------------------------
# Cross-strategy synthesis (deterministic, LLM-free)


def run_strategies(strategy_ids: Sequence[str], run_one) -> dict[str, dict[str, Any]]:
    """Run `run_one(strategy_id)` per id with per-strategy error isolation.

    Pure orchestration (no Streamlit / network of its own): `run_one` is a
    caller-supplied callable returning the per-strategy result dict. A raising
    `run_one` is captured as an `error` entry so one strategy's failure never
    aborts the others. Order follows `strategy_ids` (deterministic).
    """

    results: dict[str, dict[str, Any]] = {}
    for sid in strategy_ids:
        try:
            results[sid] = run_one(sid)
        except Exception as e:  # noqa: BLE001 — isolate one strategy's failure
            results[sid] = {"error": f"{type(e).__name__}: {e}", "signals": []}
    return results


def consume_drilldown(state: dict[str, Any], target: str) -> str | None:
    """Claim a one-shot drill-down for `target` from a mutable state dict.

    Returns the ticker exactly once (clearing the marker) when a pending
    drill-down targets this view; returns None otherwise. Pure: operates on a
    plain dict so it is unit-testable without Streamlit.
    """

    pending = state.get("pending_drilldown")
    if pending and pending.get("target") == target and pending.get("ticker"):
        ticker = str(pending["ticker"]).upper()
        state["active_ticker"] = ticker
        state["pending_drilldown"] = None
        return ticker
    return None


def synthesise_cross_strategy(
    results_by_strategy: Mapping[str, Mapping[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Rank the best overall picks across strategies. Pure & deterministic.

    Resonance-first: a ticker triggered by more strategies ranks higher; ties
    broken by summed within-strategy normalized score, then ticker name.
    Tickers that only appear as stale bars for a strategy are excluded from
    that strategy's contribution. The LLM is NOT involved.
    """

    agg: dict[str, dict[str, Any]] = {}
    for sid in sorted(results_by_strategy.keys()):
        res = results_by_strategy.get(sid) or {}
        if res.get("error"):
            continue
        signals = list(res.get("signals") or [])
        stale = set(res.get("stale_tickers") or [])
        # Deduplicate each strategy's signals by ticker BEFORE scoring so one
        # strategy contributes at most once per ticker (keep the strongest row).
        best_by_ticker: dict[str, tuple[float, Any]] = {}
        for s in signals:
            tk = s.get("ticker")
            if not tk or tk in stale:
                continue
            score = float(s.get("score") or 0.0)
            if tk not in best_by_ticker or abs(score) > abs(best_by_ticker[tk][0]):
                best_by_ticker[tk] = (score, s.get("direction"))
        scored = [(tk, sc, d) for tk, (sc, d) in best_by_ticker.items()]
        if not scored:
            continue
        max_score = max((abs(sc) for _t, sc, _d in scored), default=0.0)
        for ticker, score, direction in scored:
            norm = (score / max_score) if max_score > 0 else 0.0
            entry = agg.setdefault(
                ticker,
                {"ticker": ticker, "resonance": 0, "combined_score": 0.0,
                 "supporting": [], "directions": {}},
            )
            if sid not in entry["supporting"]:
                entry["supporting"].append(sid)
                entry["resonance"] += 1
            entry["combined_score"] += norm
            entry["directions"][sid] = direction

    ranked = sorted(
        agg.values(),
        key=lambda e: (-e["resonance"], -round(e["combined_score"], 6), e["ticker"]),
    )
    for e in ranked:
        e["combined_score"] = round(e["combined_score"], 4)
    return ranked[: max(0, int(top_n))]


# ---------------------------------------------------------------------------
# Grounding context


def _newest_first(d: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    """Per-ticker snapshots sorted by computed_at DESC, ticker ASC tie-break.

    Ensures the chat grounds on the LATEST attempt per view (available or not),
    and that the newest of more-than-N tickers is never dropped by a cap.
    """

    items = sorted((d or {}).items(), key=lambda kv: kv[0])  # ticker asc
    items.sort(key=lambda kv: str(kv[1].get("computed_at") or ""), reverse=True)  # stable
    return items


def _is_stale_for_grounding(snap: Mapping[str, Any], now_iso: str | None, max_age_s: int) -> bool:
    if snap.get("stale"):
        return True
    if now_iso is None:
        return False
    ca = snap.get("computed_at")
    if not isinstance(ca, str):
        return False
    try:
        from datetime import datetime

        age = (datetime.fromisoformat(now_iso) - datetime.fromisoformat(ca)).total_seconds()
    except (ValueError, TypeError):
        return False
    return age > max_age_s


def build_context(
    store: Mapping[str, Any] | None,
    lang: str = "en",
    *,
    now_iso: str | None = None,
    max_age_s: int = 1800,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Render the research store as one `<analysis_context>` grounding block.

    Available sections are summarized; missing/stale sections are explicitly
    labeled (never fabricated). All embedded strings are escaped so they cannot
    close the delimiter, and the whole block is truncated to `max_chars`.
    """

    zh = lang == "zh"
    missing = "(无数据)" if zh else "(not available)"
    stale_tag = " [stale]"

    lines: list[str] = []
    store = store or {}

    def section(title_en: str, title_zh: str) -> str:
        return (title_zh if zh else title_en)

    # --- Screen ---
    screen = store.get("screen")
    lines.append(section("## Market screen", "## 市场筛选"))
    if not screen or not screen.get("available"):
        lines.append(missing)
    else:
        tag = stale_tag if _is_stale_for_grounding(screen, now_iso, max_age_s) else ""
        lines.append(f"computed_at: {escape_untrusted(screen.get('computed_at'))}{tag}")
        syn = screen.get("synthesis") or []
        if syn:
            lines.append(section("Top cross-strategy picks:", "跨策略最佳候选:"))
            for p in syn[:5]:
                sup = ",".join(escape_untrusted(s) for s in (p.get("supporting") or []))
                lines.append(
                    f"  - {escape_untrusted(p.get('ticker'))} "
                    f"resonance={escape_untrusted(p.get('resonance'))} "
                    f"score={escape_untrusted(p.get('combined_score'))} "
                    f"strategies=[{sup}]"
                )
        for sid, sres in (screen.get("strategies") or {}).items():
            if sres.get("error"):
                lines.append(f"  [{escape_untrusted(sid)}] error: {escape_untrusted(sres.get('error'))}")
                continue
            lines.append(
                f"  [{escape_untrusted(sid)}] triggered={escape_untrusted(sres.get('n_triggered'))}:"
            )
            for s in (sres.get("signals") or [])[:6]:
                notes = "; ".join(escape_untrusted(n) for n in (s.get("notes") or [])[:2])
                lines.append(
                    f"    - {escape_untrusted(s.get('ticker'))} "
                    f"dir={escape_untrusted(s.get('direction'))} "
                    f"score={escape_untrusted(s.get('score'))}"
                    + (f" notes=[{notes}]" if notes else "")
                )

    na = "not available" if not zh else "无数据"

    # --- Analysis (per ticker) — newest-first; unavailable shown explicitly ---
    lines.append(section("## Single-stock analysis", "## 单股票分析"))
    analysis = _newest_first(store.get("analysis") or {})
    if not analysis:
        lines.append(missing)
    else:
        for ticker, snap in analysis[:5]:
            tag = stale_tag if _is_stale_for_grounding(snap, now_iso, max_age_s) else ""
            ca = escape_untrusted(snap.get("computed_at"))
            if not snap.get("available"):
                lines.append(f"  {escape_untrusted(ticker)}: {na} (computed_at={ca}{tag})")
                continue
            v = snap.get("verdict") or {}
            srcs = ",".join(
                escape_untrusted(s.get("source")) for s in (snap.get("sources") or [])[:6]
            )
            cands = ",".join(
                escape_untrusted(c.get("occ_symbol")) for c in (snap.get("candidates") or [])[:5]
            )
            lines.append(
                f"  {escape_untrusted(ticker)}: verdict={escape_untrusted(v.get('action'))} "
                f"skip_reason={escape_untrusted(v.get('skip_reason'))} computed_at={ca}{tag}"
            )
            if cands:
                lines.append(f"      candidates=[{cands}]")
            if srcs:
                lines.append(f"      sources=[{srcs}]")

    # --- ML (per ticker) — newest-first; unavailable shown explicitly ---
    lines.append(section("## ML direction signal", "## ML 方向信号"))
    ml = _newest_first(store.get("ml") or {})
    if not ml:
        lines.append(missing)
    else:
        for ticker, snap in ml[:5]:
            tag = stale_tag if _is_stale_for_grounding(snap, now_iso, max_age_s) else ""
            ca = escape_untrusted(snap.get("computed_at"))
            if not snap.get("available"):
                lines.append(f"  {escape_untrusted(ticker)}: {na} (computed_at={ca}{tag})")
                continue
            sig = snap.get("signal") or {}
            lines.append(
                f"  {escape_untrusted(ticker)}: prob_up={escape_untrusted(sig.get('prob_up'))} "
                f"credibility={escape_untrusted(sig.get('credibility'))} computed_at={ca}{tag}"
            )

    # --- Ledger ---
    lines.append(section("## Audit ledger", "## 审计账本"))
    ledger = store.get("ledger")
    if not ledger or not ledger.get("available"):
        lines.append(missing)
    else:
        tag = stale_tag if _is_stale_for_grounding(ledger, now_iso, max_age_s) else ""
        counts = ", ".join(
            f"{escape_untrusted(k)}={escape_untrusted(v)}"
            for k, v in (ledger.get("action_counts") or {}).items()
        )
        lines.append(
            f"rows={escape_untrusted(ledger.get('n_rows'))} "
            f"computed_at={escape_untrusted(ledger.get('computed_at'))}{tag} verdicts: {counts}"
        )

    # Bound the whole block while keeping the wrapper well-formed (raises if the
    # cap is too small to hold even an empty wrapped block).
    return _wrap_bounded("\n".join(lines), max_chars)
