FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY prompts/ ./prompts/
COPY codebooks/ ./codebooks/
COPY checker.py run.sh ./

RUN mkdir -p /app/cache /app/reports

ENV PYTHONUNBUFFERED=1 \
    MCP_TRANSPORT=stdio

CMD ["python", "-m", "src.main"]
