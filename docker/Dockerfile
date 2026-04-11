FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn gradio

# Copy application code (only what's needed)
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY codebook.yaml codebook_sozialforschung.yaml ./
COPY checker.py run.sh ./
COPY .env.example ./

# Create directories for persistence
RUN mkdir -p /app/cache /app/reports

# Transport mode: stdio (default) or http
ENV MCP_TRANSPORT=stdio
ENV PYTHONUNBUFFERED=1

# Default command: stdio transport for Claude Code
CMD ["python", "-m", "src.mcp_server"]
