"""Backward-compat shim: src.pi_server → src.agents.pi_server"""
from src.agents.pi_server import app  # noqa: F401
from src.agents.pi_server import *  # noqa: F401, F403

if __name__ == "__main__":
    from src.agents.pi_server import main
    main()
