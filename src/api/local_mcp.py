"""Local MCP server — agent tools only (pi_task, duel, benchmark_tasks).

Talks to local Ollama and uses the synced local memory DB
(populated by tools/memory_sync.py from the cloud).

Add to Claude Code MCP settings alongside the cloud memory server:
{
    "mcpServers": {
        "claude.ai Memory": { ...cloud SSE config... },
        "memory-agents": {
            "command": "/path/to/MayringCoder/.venv/bin/python",
            "args": ["-m", "src.api.local_mcp"],
            "cwd": "/path/to/MayringCoder"
        }
    }
}

Environment variables (all optional, sensible defaults):
  OLLAMA_URL          Local Ollama endpoint (default: http://localhost:11434)
  MAYRING_LOCAL_DB    Synced SQLite path   (default: ~/.cache/mayringcoder/memory.db)
  MAYRING_LOCAL_CHROMA Synced Chroma dir   (default: ~/.cache/mayringcoder/chroma)
  PI_AGENT_URL        Set to "direct" to bypass pi_server (default: direct)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")

_local_cache = Path.home() / ".cache" / "mayringcoder"

os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("MAYRING_LOCAL_DB", str(_local_cache / "memory.db"))
os.environ.setdefault("MAYRING_LOCAL_CHROMA", str(_local_cache / "chroma"))
os.environ.setdefault("PI_AGENT_URL", "direct")

from src.api.mcp_agent_tools import register_agent_tools  # noqa: E402

mcp = FastMCP("memory-agents")
register_agent_tools(mcp)


if __name__ == "__main__":
    mcp.run()
