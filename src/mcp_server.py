"""Backward-compat shim: src.mcp_server → src.api.mcp"""
from src.api.mcp import *  # noqa: F401, F403

if __name__ == "__main__":
    from src.api.mcp import main
    main()
