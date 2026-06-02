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

import json
from typing import Any, Mapping

from .chat import chat_complete
from .research_store import json_safe


_EXPLAIN_EN = (
    "Explain this market-screen result as research commentary (NOT advice). "
    "For the top cross-strategy picks and each strategy's triggered tickers: "
    "(1) why the screen surfaced them, in plain language; (2) what each "
    "strategy's logic/Notes indicators mean; (3) a credibility/confidence read "
    "(e.g. how many strategies agree, any stale-data caveats). Do NOT invent "
    "numbers beyond the context, do NOT recommend a trade, contract, strike, or "
    "size, and do NOT produce a verdict. End with a one-line reminder that this "
    "is research only."
)

_EXPLAIN_ZH = (
    "把这份市场筛选结果当作研究性评论来解释(不是投资建议)。针对跨策略最佳候选"
    "和各策略触发的股票:(1)用通俗语言说明筛选为什么选中它们;(2)解释每个策略的"
    "逻辑和 Notes 里指标的含义;(3)给出可信度/置信度判断(例如几个策略共振、是否"
    "有数据过期的警示)。不要编造上下文之外的数字,不要推荐任何交易/合约/行权价/"
    "仓位,也不要给出 verdict。结尾附一句:本内容仅供研究参考。"
)


def build_explain_message(lang: str = "en") -> str:
    """The user-turn instruction for screen explanation (pure, testable)."""

    return _EXPLAIN_ZH if lang == "zh" else _EXPLAIN_EN


def build_snapshot_context_block(screen_snapshot: Mapping[str, Any] | None) -> str:
    """Serialize a screen snapshot into an injection-safe `<analysis_context>`
    block. Angle brackets in the JSON are neutralized so untrusted ticker /
    note text cannot close the delimiter or inject a pseudo-tag.
    """

    payload = json.dumps(json_safe(dict(screen_snapshot or {})), ensure_ascii=False, indent=2)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"<analysis_context>\n{payload}\n</analysis_context>"


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
