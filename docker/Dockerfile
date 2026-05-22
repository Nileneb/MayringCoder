FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# mayring-core shared library (memory, llm routing, identity, ollama client).
# Installed editable so `import mayring_core` resolves from /app/core; src/ and
# tools/ import exclusively from it (#267).
COPY core/ ./core/
RUN pip install --no-cache-dir -e ./core

COPY src/ ./src/
COPY prompts/ ./prompts/
COPY codebooks/ ./codebooks/
COPY config/ ./config/
# tools/ enthält tools.conversation_watcher (Watcher-Daemon auf User-Laptops,
# Entry-Point aus dem Dashboard) und tools.ingest_conversations (wird
# server-side von POST /conversation/micro-batch für die Summary-Generierung
# importiert). Ohne diesen COPY scheitern beide Paths mit ModuleNotFoundError.
COPY tools/ ./tools/
COPY run.sh ./

RUN mkdir -p /app/cache /app/reports

ENV PYTHONUNBUFFERED=1 \
    MCP_TRANSPORT=stdio

# mayring-api exposes /health on port 8090 (src/api/server.py).
# Gilt nur für Container die src.api.server hosten — andere Services
# (mcp, pi, webui) überschreiben CMD in docker-compose.mayring.yml und können
# diesen Healthcheck ignorieren oder per --no-healthcheck override deaktivieren.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8090/health || exit 1

CMD ["python", "-m", "src.main"]
