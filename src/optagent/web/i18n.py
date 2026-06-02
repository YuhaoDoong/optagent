"""Minimal i18n table — English ↔ Simplified Chinese.

Why hand-rolled rather than gettext: only ~80 strings, deterministic
shape, and the tests need a stable key list. Adding a third language
later: copy an existing column and translate.

Keys are dotted (`tab.analyze.title`) so the lookup is namespaced and
linters can grep for orphan strings.
"""

from __future__ import annotations


SUPPORTED_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("en", "English"),
    ("zh", "中文"),
)


_TABLE: dict[str, dict[str, str]] = {
    # Sidebar
    "sidebar.title": {"en": "optagent", "zh": "optagent"},
    "sidebar.version": {"en": "v{version}", "zh": "v{version}"},
    "sidebar.lang_label": {"en": "Language", "zh": "语言"},
    "sidebar.adapters": {"en": "Optional adapters", "zh": "可选数据源"},
    "sidebar.fred_label": {"en": "FRED_API_KEY", "zh": "FRED API 密钥"},
    "sidebar.fred_help": {
        "en": "Set to enable macro context (10y/2y yields, VIX, CPI, Fed funds, USD index).",
        "zh": "设置后启用宏观因子（10/2 年国债收益率、VIX、CPI、联邦基金利率、美元指数）。",
    },
    "sidebar.sec_label": {"en": "SEC EDGAR User-Agent", "zh": "SEC EDGAR User-Agent"},
    "sidebar.sec_help": {
        "en": "SEC EDGAR REQUIRES a contact email. Adapter fails closed without one.",
        "zh": "SEC EDGAR 强制要求包含联系邮箱。没填则适配器关闭。",
    },
    "sidebar.llm_header": {"en": "LLM (optional)", "zh": "LLM（可选）"},
    "sidebar.enable_llm": {"en": "Enable LLM synthesis", "zh": "启用 LLM 综合分析"},
    "sidebar.provider": {"en": "Provider", "zh": "供应商"},
    "sidebar.enable_ml": {
        "en": "Enable ML direction signal (Alt-3 v0)",
        "zh": "启用 ML 方向模型（Alt-3 v0）",
    },
    "sidebar.auto_detect": {"en": "auto-detect", "zh": "自动检测"},
    "sidebar.use_moomoo": {
        "en": "Use Moomoo OpenD for option quotes (works after-hours)",
        "zh": "用 Moomoo OpenD 获取期权报价(收盘也可用)",
    },
    "sidebar.moomoo_help": {
        "en": "Requires Moomoo OpenD running locally on 127.0.0.1:11111. Falls back to yfinance if unreachable. yfinance zeroes bid/ask/OI when the US market is closed; Moomoo does not.",
        "zh": "需要本地运行 Moomoo OpenD(127.0.0.1:11111)。连不上会自动回退到 yfinance。yfinance 在美股收盘时会把买卖盘口和持仓量清零,Moomoo 不会。",
    },

    # Disclaimer banner
    "disclaimer.banner": {
        "en": "⚠️ {disclaimer} optagent v{version} is a research tool only.",
        "zh": "⚠️ {disclaimer} optagent v{version} 仅为研究工具，不构成投资建议。",
    },

    # Tab headers
    "tab.analyze": {"en": "📊 Analyze ticker", "zh": "📊 单股票分析"},
    "tab.screen": {"en": "🔭 Screen market", "zh": "🔭 市场筛选"},
    "tab.ml": {"en": "🧠 ML signal", "zh": "🧠 ML 信号"},
    "tab.ledger": {"en": "📒 Ledger", "zh": "📒 历史账本"},
    "tab.chat": {"en": "💬 Ask the agent", "zh": "💬 与 agent 对话"},

    # Analyze tab
    "analyze.header": {"en": "📊 Single-ticker analysis", "zh": "📊 单股票分析"},
    "analyze.caption": {
        "en": "Equivalent to running `optagent analyze <ticker>` from the CLI. Outputs a structured research memo with the same disclaimer-first contract.",
        "zh": "与命令行 `optagent analyze <ticker>` 等价。输出结构化研究备忘，沿用相同的免责声明优先约定。",
    },
    "analyze.ticker_label": {"en": "Ticker", "zh": "股票代码"},
    "analyze.horizon_label": {"en": "Horizon (days)", "zh": "持仓周期（天）"},
    "analyze.max_loss_label": {"en": "Max-loss budget (USD, optional)", "zh": "最大亏损预算（美元，可选）"},
    "analyze.run_btn": {"en": "Analyze", "zh": "分析"},
    "analyze.spinner": {"en": "Running analysis on {ticker}...", "zh": "正在分析 {ticker}……"},
    "analyze.primary_reasons": {"en": "**Primary reasons:**", "zh": "**主要理由：**"},
    "analyze.candle_title": {"en": "Recent price action (60d)", "zh": "近期价格走势（60 日）"},
    "analyze.envelopes_title": {"en": "Upstream envelopes", "zh": "上游数据 Envelope"},
    "analyze.candidates_title": {"en": "Screener candidates ({n})", "zh": "筛选器候选合约（{n} 个）"},
    "analyze.smile_title": {"en": "Options chain IV smile", "zh": "期权链 IV 笑容曲线"},
    "analyze.memo_title": {"en": "Memo (text)", "zh": "研究备忘（文本）"},

    # Screen tab
    "screen.header": {"en": "🔭 Cross-ticker screen", "zh": "🔭 跨股票筛选"},
    "screen.caption": {
        "en": "Equivalent to `optagent screen --strategy ... --sector ...`. The framework runs a quant strategy across the universe and ranks the top candidates by score.",
        "zh": "与命令行 `optagent screen --strategy ... --sector ...` 等价。在指定 universe 上跑量化策略并按得分排序。",
    },
    "screen.strategy_label": {"en": "Strategy", "zh": "策略"},
    "screen.sector_label": {"en": "Sector (optional)", "zh": "扇区（可选）"},
    "screen.sector_any": {"en": "(any)", "zh": "(全部)"},
    "screen.limit_label": {"en": "Top N", "zh": "前 N 个"},
    "screen.run_btn": {"en": "Run screen", "zh": "运行筛选"},
    "screen.spinner": {"en": "Running {strategy} across {n} tickers...", "zh": "正在用 {strategy} 扫描 {n} 只股票……"},
    "screen.metric_universe": {"en": "Universe size", "zh": "Universe 大小"},
    "screen.metric_evaluated": {"en": "Evaluated", "zh": "评估完成"},
    "screen.metric_triggered": {"en": "Triggered", "zh": "已触发"},
    "screen.metric_near_misses": {"en": "Top near-misses", "zh": "差一点的候选"},
    "screen.stale_warning": {
        "en": "{n} ticker(s) had stale OHLCV bars (US market holidays / long weekends). They were still evaluated; verify before action.",
        "zh": "{n} 只股票的 OHLCV 数据过期（美国市场假期 / 长周末），仍参与了评估，请人工确认后再行动。",
    },
    "screen.top_candidates": {"en": "Top candidates", "zh": "前 N 候选"},
    "screen.stale_details": {"en": "Stale-bar details", "zh": "数据过期明细"},
    "screen.score_chart": {"en": "Scores", "zh": "得分"},
    "screen.no_trigger": {"en": "No tickers triggered the strategy.", "zh": "没有股票触发该策略。"},
    "screen.near_misses_expander": {"en": "Near-misses ({n})", "zh": "差一点触发的候选（{n} 个）"},
    "screen.sector_empty_warning": {
        "en": "Sector '{sector}' has no overlap with the built-in universe.",
        "zh": "扇区 '{sector}' 与内置 universe 没有交集。",
    },

    # ML tab
    "ml.header": {"en": "🧠 ML direction model", "zh": "🧠 ML 方向模型"},
    "ml.caption": {
        "en": "Per-ticker `GradientBoostingClassifier` with walk-forward OOS validation, Wilson 95% CI, and class-baseline comparison. Output is INFORMATIONAL — the fail-closed validator does not let it become the sole reason for a non-SKIP verdict.",
        "zh": "每只股票一个 `GradientBoostingClassifier`，含 walk-forward 样本外验证、Wilson 95% 置信区间、与基线类别准确率对比。结果仅供参考——失败闭合验证器不允许它单独成为非 SKIP verdict 的主要依据。",
    },
    "ml.run_btn": {"en": "Compute signal", "zh": "计算信号"},
    "ml.ticker_label": {"en": "Ticker", "zh": "股票代码"},
    "ml.spinner": {"en": "Training / loading model for {ticker}...", "zh": "正在训练/加载 {ticker} 的模型……"},
    "ml.unavailable": {
        "en": "ML signal unavailable (no yfinance history or invalid ticker).",
        "zh": "ML 信号不可用（缺少 yfinance 历史数据或股票代码无效）。",
    },
    "ml.gauge_title": {"en": "ML direction signal", "zh": "ML 方向信号"},
    "ml.feature_snapshot": {"en": "Feature snapshot", "zh": "特征快照"},

    # Ledger tab
    "ledger.header": {"en": "📒 Audit ledger viewer", "zh": "📒 审计账本浏览器"},
    "ledger.caption": {
        "en": "Browse recent runs persisted to `data/ledger/YYYY-MM-DD.jsonl`. Useful for post-hoc inspection without re-running.",
        "zh": "浏览存于 `data/ledger/YYYY-MM-DD.jsonl` 的近期运行记录。事后审计无需重跑。",
    },
    "ledger.days_back_label": {"en": "Days back", "zh": "回看天数"},
    "ledger.dir_label": {"en": "Ledger directory", "zh": "账本目录"},
    "ledger.empty": {
        "en": "No ledger rows found. Run `optagent analyze <ticker>` (CLI or this UI) to populate the ledger.",
        "zh": "未找到账本记录。请先运行 `optagent analyze <ticker>`（命令行或本 UI）。",
    },
    "ledger.count": {"en": "**{n}** recent runs found.", "zh": "找到 **{n}** 条近期运行。"},
    "ledger.pie_title": {"en": "Verdict distribution", "zh": "Verdict 分布"},

    # Chat tab
    "chat.header": {"en": "💬 Ask the agent", "zh": "💬 与 agent 对话"},
    "chat.caption": {
        "en": "Ask questions about the most-recent analysis (envelopes, candidates, verdict, ML signal). The LLM grounds answers in that context but does NOT recommend trades.",
        "zh": "对最近一次分析（envelope、候选合约、verdict、ML 信号）提问。LLM 会根据该上下文作答，但不会推荐具体交易。",
    },
    "chat.no_context": {
        "en": "Run an analysis on the 📊 Analyze tab first; the chat uses its data as grounding context.",
        "zh": "请先在 📊 单股票分析 标签页跑一次分析；对话会以此为 grounding 上下文。",
    },
    "chat.context_summary": {
        "en": "Discussing {ticker} analysis from {ts}.",
        "zh": "当前讨论 {ticker} 的分析（{ts}）。",
    },
    "chat.placeholder": {"en": "Ask anything about this analysis...", "zh": "对这次分析提问……"},
    "chat.spinner": {"en": "Thinking...", "zh": "思考中……"},
    "chat.no_llm": {
        "en": "Enable LLM in the sidebar (Anthropic / OpenAI / Gemini key required) to use chat.",
        "zh": "请在左侧启用 LLM（需要 Anthropic / OpenAI / Gemini 任一 API key）后再使用对话功能。",
    },
    "chat.error": {"en": "Chat call failed: {err}", "zh": "对话调用失败：{err}"},
    "chat.clear_btn": {"en": "Clear conversation", "zh": "清空对话"},
    "chat.panel_grounding": {
        "en": "Grounded on the latest results across all views.",
        "zh": "以各视图最新结果为对话上下文。",
    },

    # View navigation (replaces tabs)
    "nav.label": {"en": "View", "zh": "视图"},

    # Multi-strategy screen additions
    "screen.multiselect_label": {"en": "Strategies (select one or more)", "zh": "策略(可多选)"},
    "screen.select_prompt": {"en": "Select at least one strategy.", "zh": "请至少选择一个策略。"},
    "screen.per_strategy_title": {"en": "Per-strategy results", "zh": "各策略结果"},
    "screen.strategy_error": {"en": "Strategy '{sid}' failed: {err}", "zh": "策略 '{sid}' 出错:{err}"},
    "screen.synthesis_title": {"en": "Top cross-strategy picks", "zh": "跨策略最佳候选"},
    "screen.synthesis_caption": {
        "en": "Deterministic ranking: more strategies agreeing (resonance) first, then combined normalized score. The LLM only explains this order.",
        "zh": "确定性排序:多策略共振优先,再看综合归一化得分。LLM 只负责解释这个排名,不参与排序。",
    },
    "screen.col_resonance": {"en": "Strategies agreeing", "zh": "共振策略数"},
    "screen.col_support": {"en": "Supporting strategies", "zh": "支持策略"},
    "screen.col_score": {"en": "Combined score", "zh": "综合得分"},
    "screen.drill_ml": {"en": "ML signal", "zh": "ML 信号"},
    "screen.drill_analyze": {"en": "Analyze", "zh": "单股分析"},
    "screen.explain_btn": {"en": "🤖 Explain results with AI", "zh": "🤖 让 AI 解释结果"},
    "screen.explain_spinner": {"en": "Asking the LLM to explain...", "zh": "正在让 LLM 解释……"},
    "screen.explain_title": {"en": "AI explanation (research commentary)", "zh": "AI 解释(研究性评论)"},
    "screen.run_first": {"en": "Run a screen first, then ask the AI to explain it.", "zh": "请先运行筛选,再让 AI 解释。"},
    "screen.no_synthesis": {"en": "No tickers triggered across the selected strategies.", "zh": "所选策略均未触发任何股票。"},
    "screen.synthesis_filtered": {
        "en": "Strategies triggered tickers, but none qualified for the cross-strategy ranking (e.g. excluded for stale OHLCV bars). See the per-strategy tables below.",
        "zh": "策略触发了一些股票,但没有进入跨策略排名(例如因 OHLCV 数据过期被排除)。详见下方各策略表格。",
    },

    # Drill-down auto-run
    "drill.autorun_note": {
        "en": "Auto-running for {ticker} (from screen drill-down).",
        "zh": "正在自动分析 {ticker}(来自筛选钻取)。",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    """Look up a translation. Falls back to English on missing key/lang."""

    row = _TABLE.get(key)
    if row is None:
        return key  # surface the missing key in the UI rather than crash
    val = row.get(lang) or row.get("en") or key
    if kwargs:
        try:
            return val.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return val
    return val


def supported_languages() -> tuple[tuple[str, str], ...]:
    return SUPPORTED_LANGUAGES


def all_keys() -> list[str]:
    """Sorted key list — used by tests to assert coverage parity across langs."""

    return sorted(_TABLE.keys())
