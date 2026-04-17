"""Backward-compat shim: src.web_ui → src.api.web_ui"""
from src.api.web_ui import *  # noqa: F401, F403

if __name__ == "__main__":
    from src.api.web_ui import main
    main()
