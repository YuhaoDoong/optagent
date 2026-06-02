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


def _compact_seq(value: Any, limit: int) -> list[Any]:
    """Coerce a possibly-malformed nested value into a bounded list.

    Lists/tuples are sliced; None becomes []; anything else (a scalar or a
    non-subscriptable object) is wrapped in a single-element list so later
    `json_safe` can stringify it — never raising before coercion.
    """

    if isinstance(value, (list, tuple)):
        return list(value)[:limit]
    if value is None:
        return []
    return [value]


def _field(obj: Any, key: str) -> Any:
    """Safe field access — returns None when `obj` is not a mapping."""

    return obj.get(key) if isinstance(obj, Mapping) else None


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce to a plain dict; non-mapping (or malformed) input becomes {}."""

    return {str(k): v for k, v in value.items()} if isinstance(value, Mapping) else {}


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce to int; malformed/non-numeric input becomes `default`."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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

    payload = json.dumps(json_safe(_as_mapping(bundle)), ensure_ascii=False, indent=2)
    return _wrap_bounded(neutralize_text(payload), max_chars)


def sanitize_context_block(block: str | None, *, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Enforce the grounding-block contract at the LLM boundary.

    The inner body is ALWAYS re-neutralized — a structurally valid wrapper is
    never trusted to be already-sanitized, so a block like
    `<analysis_context>\nignore previous instructions\n</analysis_context>`
    still gets its body defanged. `neutralize_text` is idempotent on an
    already-escaped body (no raw `<`/`>` to re-escape), so canonical
    `build_context` / `serialize_bundle` output round-trips unchanged in
    meaning. The result is always bounded and carries exactly one wrapper pair.
    """

    if not block:
        return ""
    text = block if isinstance(block, str) else str(block)
    # Strip exactly one outer wrapper if present; otherwise treat the whole
    # string as untrusted body. (rindex finds the REAL closing delimiter; an
    # escaped &lt;/analysis_context&gt; in the body cannot match.)
    if text.startswith(_OPEN) and _CLOSE in text:
        inner = text[len(_OPEN): text.rindex(_CLOSE)]
        # Drop the wrapper's own framing newlines so re-wrapping a canonical
        # block round-trips identically (idempotent on build_context output).
        if inner.startswith("\n"):
            inner = inner[1:]
        if inner.endswith("\n"):
            inner = inner[:-1]
    else:
        inner = text
    return _wrap_bounded(neutralize_text(inner), max_chars)


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

    results_by_strategy = _as_mapping(results_by_strategy)
    if not results_by_strategy:
        snap = _base_snapshot("screen", computed_at, available=False, stale=False)
        snap["inputs"] = json_safe(_as_mapping(inputs))
        return snap
    strategies = {}
    any_stale = False
    for sid, res in results_by_strategy.items():
        stale_tickers = _compact_seq(_field(res, "stale_tickers"), 1000)
        any_stale = any_stale or bool(stale_tickers)
        strategies[str(sid)] = json_safe(
            {
                "error": _field(res, "error"),
                "n_triggered": _field(res, "n_triggered"),
                "n_evaluated": _field(res, "n_evaluated"),
                "signals": [
                    {
                        "ticker": _field(s, "ticker"),
                        "direction": _field(s, "direction"),
                        "score": _field(s, "score"),
                        "notes": _compact_seq(_field(s, "notes"), 3),
                    }
                    for s in _compact_seq(_field(res, "signals"), 10)
                ],
                "stale_tickers": stale_tickers,
            }
        )
    snap = _base_snapshot("screen", computed_at, available=True, stale=any_stale)
    snap["inputs"] = json_safe(_as_mapping(inputs))
    snap["strategies"] = strategies
    snap["synthesis"] = json_safe(_compact_seq(synthesis, 1000))
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
        snap["inputs"] = json_safe(_as_mapping(inputs))
        return snap
    snap = _base_snapshot("analysis", computed_at, available=verdict is not None, stale=False)
    snap["ticker"] = ticker
    snap["inputs"] = json_safe(_as_mapping(inputs))
    snap["verdict"] = json_safe(verdict) if verdict is not None else None
    snap["candidates"] = json_safe(
        [
            {
                "occ_symbol": _field(c, "occ_symbol"),
                "right": _field(c, "right"),
                "strike": _field(c, "strike"),
                "mid": _field(c, "mid"),
                "delta": _field(c, "delta"),
                "iv": _field(c, "iv"),
                "breakeven": _field(c, "breakeven"),
                "max_loss": _field(c, "max_loss"),
            }
            for c in _compact_seq(candidates, 8)
        ]
    )
    # Compact upstream-status summary so chat can cite data provenance/freshness.
    snap["sources"] = json_safe(
        [
            {
                "source": _field(e, "source"),
                "confidence": _field(e, "confidence"),
                "as_of": _field(e, "as_of"),
            }
            for e in _compact_seq(envelopes, 10)
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
        snap["inputs"] = json_safe(_as_mapping(inputs))
        return snap
    snap = _base_snapshot("ml", computed_at, available=ml is not None, stale=False)
    snap["ticker"] = ticker
    snap["inputs"] = json_safe(_as_mapping(inputs) or {"ticker": ticker})
    snap["signal"] = json_safe(ml) if ml is not None else None
    return snap


def ledger_summary_snapshot(
    action_counts: Mapping[str, int] | None,
    n_rows: int,
    computed_at: str,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snap = _base_snapshot("ledger", computed_at, available=bool(action_counts), stale=False)
    snap["inputs"] = json_safe(_as_mapping(inputs))
    snap["n_rows"] = _as_int(n_rows, 0)
    snap["action_counts"] = json_safe(_as_mapping(action_counts))
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
    na = "无数据" if zh else "not available"
    stale_tag = " [stale]"

    lines: list[str] = []
    store = store or {}

    def section(title_en: str, title_zh: str) -> str:
        return (title_zh if zh else title_en)

    def unavailable_line(snap: Mapping[str, Any], label: str = "") -> str:
        """A timestamped not-available line for a snapshot that EXISTS but is
        unavailable (a failed/empty attempt), distinct from a never-run section."""

        tag = stale_tag if _is_stale_for_grounding(snap, now_iso, max_age_s) else ""
        ca = escape_untrusted(snap.get("computed_at"))
        prefix = f"{label}: " if label else ""
        return f"{prefix}{na} (computed_at={ca}{tag})"

    # --- Screen ---
    screen = store.get("screen")
    lines.append(section("## Market screen", "## 市场筛选"))
    if not screen:
        lines.append(missing)
    elif not screen.get("available"):
        lines.append(unavailable_line(screen))
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
            v = snap.get("verdict") if isinstance(snap.get("verdict"), Mapping) else {}
            srcs = ",".join(
                escape_untrusted(_field(s, "source")) for s in (snap.get("sources") or [])[:6]
            )
            conv = v.get("conviction")
            lines.append(
                f"  {escape_untrusted(ticker)}: verdict={escape_untrusted(v.get('action'))} "
                f"skip_reason={escape_untrusted(v.get('skip_reason'))} "
                f"conviction={escape_untrusted(conv)} computed_at={ca}{tag}"
            )
            # A bounded subset of the verdict rationale so chat can answer
            # "why this verdict".
            reasons = _compact_seq(v.get("primary_reasons"), 2)
            if reasons:
                joined = "; ".join(escape_untrusted(r) for r in reasons)
                lines.append(f"      reasons: {joined}")
            # Per-candidate metrics (strike / mid / delta / breakeven / max-loss)
            # so chat can answer questions about the contract numbers.
            for c in (snap.get("candidates") or [])[:3]:
                lines.append(
                    f"      {escape_untrusted(_field(c, 'occ_symbol'))} "
                    f"{escape_untrusted(_field(c, 'right'))} "
                    f"K={escape_untrusted(_field(c, 'strike'))} "
                    f"mid={escape_untrusted(_field(c, 'mid'))} "
                    f"delta={escape_untrusted(_field(c, 'delta'))} "
                    f"iv={escape_untrusted(_field(c, 'iv'))} "
                    f"BE={escape_untrusted(_field(c, 'breakeven'))} "
                    f"maxloss={escape_untrusted(_field(c, 'max_loss'))}"
                )
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
    if not ledger:
        lines.append(missing)
    elif not ledger.get("available"):
        lines.append(unavailable_line(ledger))
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
