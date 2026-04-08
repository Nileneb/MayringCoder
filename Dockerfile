FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# Copy application code (only what's needed)
COPY src/ ./src/
COPY .env.example . 2>/dev/null || true

# Create cache directory for SQLite + ChromaDB persistence
RUN mkdir -p /app/cache

# Transport mode: stdio (default) or http
ENV MCP_TRANSPORT=stdio
ENV PYTHONUNBUFFERED=1

# Default command: stdio transport for Claude Code
CMD ["python", "-m", "src.mcp_server"]
