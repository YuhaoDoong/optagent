"""LLM research-commentary over a market-screen snapshot.

Feature: after a (multi-strategy) screen runs, explain in plain language why
the surfaced tickers were selected, what each strategy's logic is, and a
credibility/confidence read on the picks — INCLUDING the deterministic
cross-strategy synthesis ordering (which the LLM may explain but never
produce).

This reuses `chat.chat_complete`, so it inherits the research-only system
prompt (no trade advice, no new verdicts, no sizing/shorts/naked/0DTE, bounded
to {SKIP, LONG_CALL, LONG_PUT}) and the `<analysis_context>` DATA-not-
instructions framing. The screen snapshot is passed as the grounding context;
the LLM returns prose only and is never asked for a verdict or a ranking.
"""

from __future__ import annotations

from typing import Any, Mapping

from .chat import chat_complete
from .research_store import (
    _as_mapping,
    _compact_seq,
    _curate_conditions,
    _field,
    escape_untrusted,
    pack_sections,
)


# Cap on the serialized snapshot grounding (chars). Keeps the opt-in
# explanation prompt bounded regardless of how large a screen run was.
_CONTEXT_CAP = 8000


_EXPLAIN_EN = (
    "Explain this market-screen result as research commentary (NOT advice). "
    "For the top cross-strategy picks and each strategy's triggered tickers: "
    "(1) why the screen surfaced them, in plain language; (2) what each "
    "strategy's logic/Notes indicators mean; (3) a credibility/confidence read "
    "(e.g. how many strategies agree, any stale-data caveats). Do NOT invent "
    "numbers beyond the context. Do NOT recommend a trade, contract, strike, "
    "expiry, or position size; do NOT propose shorts, naked, or 0DTE; do NOT "
    "place, submit, or describe how to place an order; and do NOT produce a "
    "verdict or any verdict outside {SKIP, LONG_CALL, LONG_PUT}. End with a "
    "one-line reminder that this is research only."
)

_EXPLAIN_ZH = (
    "把这份市场筛选结果当作研究性评论来解释(不是投资建议)。针对跨策略最佳候选"
    "和各策略触发的股票:(1)用通俗语言说明筛选为什么选中它们;(2)解释每个策略的"
    "逻辑和 Notes 里指标的含义;(3)给出可信度/置信度判断(例如几个策略共振、是否"
    "有数据过期的警示)。不要编造上下文之外的数字;不要推荐任何交易/合约/行权价/"
    "到期/仓位大小;不要建议卖出期权、裸期权或 0DTE;不要下单、提交订单或说明如何"
    "下单;也不要给出 verdict 或 {SKIP, LONG_CALL, LONG_PUT} 之外的任何 verdict。"
    "结尾附一句:本内容仅供研究参考。"
)


def build_explain_message(lang: str = "en") -> str:
    """The user-turn instruction for screen explanation (pure, testable)."""

    return _EXPLAIN_ZH if lang == "zh" else _EXPLAIN_EN


def build_snapshot_context_block(screen_snapshot: Mapping[str, Any] | None) -> str:
    """Render a screen snapshot into an injection-safe, per-strategy-budgeted
    `<analysis_context>` block.

    Each strategy gets its OWN section via `pack_sections`, so every selected
    strategy receives a guaranteed share of the char budget and none can fall
    off the end of a single truncated blob. All embedded fields are escaped, so
    untrusted ticker/note text cannot break the delimiter or inject phrases.
    """

    snap = _as_mapping(screen_snapshot)
    sections: list[tuple[str, list[str]]] = []

    syn = _compact_seq(snap.get("synthesis"), 10)
    if syn:
        sections.append((
            "## Top cross-strategy picks",
            [
                f"- {escape_untrusted(_field(p, 'ticker'))} "
                f"dir={escape_untrusted(_field(p, 'direction'))} "
                f"resonance={escape_untrusted(_field(p, 'resonance'))} "
                f"score={escape_untrusted(_field(p, 'combined_score'))} "
                f"strategies=[{','.join(escape_untrusted(s) for s in (_field(p, 'supporting') or []))}]"
                for p in syn
            ],
        ))

    strategies = _as_mapping(snap.get("strategies"))
    for sid, sres in strategies.items():
        body: list[str] = []
        if _field(sres, "error"):
            body.append(f"error: {escape_untrusted(_field(sres, 'error'))}")
        else:
            body.append(
                f"triggered={escape_untrusted(_field(sres, 'n_triggered'))} "
                f"evaluated={escape_untrusted(_field(sres, 'n_evaluated'))}"
            )
            stale_set = set(_compact_seq(_field(sres, "stale_tickers"), 1000))
            for s in _compact_seq(_field(sres, "signals"), 8):
                cond = _curate_conditions(_field(s, "conditions"), 12)
                cond_str = ", ".join(
                    f"{escape_untrusted(k)}={escape_untrusted(v)}" for k, v in cond.items()
                )
                notes = "; ".join(escape_untrusted(n) for n in _compact_seq(_field(s, "notes"), 2))
                reward = _as_mapping(_field(s, "reward"))
                tgt = reward.get("target_price")
                st = " [stale]" if _field(s, "ticker") in stale_set else ""
                body.append(
                    f"- {escape_untrusted(_field(s, 'ticker'))}{st} "
                    f"dir={escape_untrusted(_field(s, 'direction'))} "
                    f"score={escape_untrusted(_field(s, 'score'))}"
                    + (f" target={escape_untrusted(tgt)}" if tgt is not None else "")
                    + (f" conditions=[{cond_str}]" if cond_str else "")
                    + (f" notes=[{notes}]" if notes else "")
                )
        sections.append((f"## strategy {escape_untrusted(sid)}", body))

    if not sections:
        sections = [("## Market screen", ["(not available)"])]
    return pack_sections(sections, _CONTEXT_CAP)


def explain_screen(
    screen_snapshot: Mapping[str, Any] | None,
    *,
    lang: str = "en",
    provider: str | None = None,
    disclaimer: str = "RESEARCH ONLY — NOT FINANCIAL ADVICE.",
) -> str:
    """Return plain-language research commentary on a screen snapshot.

    Raises RuntimeError (from chat_complete) when no LLM provider is
    configured. The snapshot is grounding DATA; the reply is prose only.
    """

    return chat_complete(
        history=[],
        user_message=build_explain_message(lang),
        context_bundle=None,
        context_block=build_snapshot_context_block(screen_snapshot),
        lang=lang,
        provider=provider,
        disclaimer=disclaimer,
    )
