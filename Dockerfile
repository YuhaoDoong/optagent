# optagent — local Streamlit UI image
#
# Build:
#   docker build -t optagent:0.4.0-dev .
# Run (local-only, no external network exposure):
#   docker run -it --rm -p 8501:8501 \
#       -e OPTAGENT_USER_AGENT="me/0.0.1 (me@example.com)" \
#       -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#       optagent:0.4.0-dev
#
# Open http://localhost:8501.
#
# > RESEARCH ONLY — NOT FINANCIAL ADVICE.
# > Image runs the same fail-closed validator and bounded VerdictAction
# > as the CLI; nothing in the container path can promote a SKIP into a
# > LONG verdict.

# ---- builder stage: install deps into a venv ----
FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --upgrade pip && \
    pip install ".[adapters,llm,ui]"

# ---- runtime stage: minimal image ----
FROM python:3.11-slim AS runtime

# Non-root user — Streamlit serves untrusted env inputs (sidebar text),
# so we limit damage potential.
RUN groupadd -r app && useradd -r -g app -m -d /home/app app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY --from=builder /opt/venv /opt/venv
WORKDIR /home/app
RUN mkdir -p data/ledger data/iv_history data/ml_cache && \
    chown -R app:app /home/app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else 1)" || exit 1

# `optagent-ui` is the console-script entry point installed by pip.
# Pass --server.address=0.0.0.0 so the container is reachable from the host.
CMD ["optagent-ui", "--server.port=8501", "--server.address=0.0.0.0"]
