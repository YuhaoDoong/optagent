# optagent/web — local Streamlit UI

> **RESEARCH ONLY — NOT FINANCIAL ADVICE.** The web UI is a presentation
> layer; the disclaimer + fail-closed validator + bounded-verdict
> invariants from the CLI all still apply.

## Quickstart

```bash
pip install -e .[adapters,ui]   # add `llm` if you want --enable-llm
optagent-ui
```

Opens a local Streamlit server (default http://localhost:8501). Hit
Ctrl-C to stop.

Want a different port? `optagent-ui --server.port 8765` — any flag is
forwarded to `streamlit run`.

## Tabs

### 📊 Analyze ticker
The visual version of `optagent analyze <ticker>`. Type a ticker, pick a
horizon + max-loss budget, hit **Analyze**. Renders:
- Verdict badge (colour-coded; bounded to SKIP / LONG_CALL / LONG_PUT).
- Primary reasons.
- All upstream envelopes (source, profile, confidence, cache age, warnings).
- Screener candidate table (Greeks, breakeven, max-loss, scores).
- ML signal gauge (if `Enable ML direction signal` ticked).
- Plain-text memo (the same one CLI prints).

### 🔭 Screen market
The visual version of `optagent screen --strategy ... --sector ...`.
- Pick a strategy + optional sector + top-N.
- Bar chart of triggered candidates by score.
- Stale-bar warnings (Codex R5 finding: US-market-holiday footgun).
- Near-misses (skip verdicts with score > 0) in a collapsible panel.

### 🧠 ML signal
Per-ticker GradientBoosting direction model with:
- prob_up gauge (Plotly indicator).
- Wilson 95% CI + class baseline + n_oos_samples annotations.
- Feature snapshot table.

### 📒 Ledger
Browse the persistent audit-ledger JSONL (`data/ledger/YYYY-MM-DD.jsonl`).
Picks the last N days, surfaces a verdict-distribution pie chart, and shows
one row per run (ticker / action / skip_reason / envelope and candidate
counts / run_id).

## Docker

A `Dockerfile`, `docker-compose.yml`, and `.dockerignore` ship at the repo
root. The image runs the same fail-closed validator and bounded
`VerdictAction` enum as the CLI; nothing in the container path can promote
a SKIP into a LONG verdict.

```bash
# Build once
docker build -t optagent:0.4.0-dev .

# Run; ledger / ML cache / IV history persist into ./data
docker run -it --rm -p 127.0.0.1:8501:8501 \
    -e OPTAGENT_USER_AGENT="me/0.0.1 (me@example.com)" \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    -v "$(pwd)/data:/home/app/data" \
    optagent:0.4.0-dev

# Or compose (reads optional .env beside docker-compose.yml)
docker compose up --build
```

Image runs as a non-root `app` user, binds Streamlit to `127.0.0.1` on the
host by default, and exposes a `HEALTHCHECK` against
`/_stcore/health`. Multi-stage build keeps the runtime image lean.

## Going public later

The UI is intentionally framework-light so the hosting path is open:

| Target | Steps |
|---|---|
| Streamlit Community Cloud | Push to GitHub, link the repo, set FRED/SEC env vars in the cloud dashboard. Done. |
| Docker on any cloud | `docker build` with a thin image that runs `optagent-ui`. Forward port 8501. |
| FastAPI + React port (real domain) | Reuse `optagent.web.components` helpers (pure functions, no Streamlit dep); call `analyze()` / `screen_universe()` from FastAPI endpoints; build a React/Vue/HTMX front-end. |

The pure helpers in `components.py` (verdict_badge, candidate_table,
envelope_summary, ml_signal_gauge, feature_radar, candle_chart,
strategy_signal_table) have **no Streamlit dependency** — they return
plain dicts / DataFrames, so they port directly to a FastAPI JSON API.

## Sidebar options

The sidebar exposes:
- `FRED_API_KEY` — paste once; the analyze tab will register the FRED
  adapter automatically.
- `OPTAGENT_USER_AGENT` — required for the SEC EDGAR adapter (the
  underlying adapter fail-closes without it).
- `Enable LLM synthesis` + provider picker — auto-detects from
  `OPENROUTER_API_KEY` (preferred) / `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` / `GEMINI_API_KEY`. Keys can live in a gitignored
  `.env` at the repo root (copy `.env.example`); shell-exported vars
  override `.env`. OpenRouter is a multi-model gateway — set
  `OPENROUTER_MODEL` to any slug (default `anthropic/claude-sonnet-4.6`).
- `Enable ML direction signal` — first run trains a fresh model (~5s);
  subsequent runs within 7 days hit the cache (~50ms).
- `Use Moomoo OpenD for option quotes` (on by default) — sources the option
  chain from a local Moomoo/Futu OpenD gateway (`127.0.0.1:11111`) instead of
  yfinance. yfinance/Yahoo zeroes out bid/ask AND open-interest whenever the
  US market is closed, so after-hours runs otherwise SKIP with
  `stale_required_input`; Moomoo returns real bid/ask/OI/IV even at EOD.
  Falls back to yfinance automatically if OpenD is unreachable. Install with
  `pip install -e .[moomoo]` and start OpenD first.

Sidebar inputs do NOT persist across sessions; treat them as a
per-session override of environment variables.

## Safety

Every page renders the canonical disclaimer banner at the top. The
underlying calls (`analyze()`, `screen_universe()`, `MLDirectionAdapter`)
all go through the same fail-closed validator and bounded-verdict enum
as the CLI — no UI feature can promote a SKIP into a LONG verdict, and
no UI page contains the strings `place_order` / `submit_order` /
`new_order` (CI grep test still active).
