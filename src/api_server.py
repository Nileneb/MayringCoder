"""Backward-compat shim: src.api_server → src.api.server"""
from src.api.server import app  # noqa: F401
from src.api.server import *  # noqa: F401, F403
