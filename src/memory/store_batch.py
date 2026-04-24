"""Batch-commit context manager for SQLite connections (#68)."""
from __future__ import annotations

import sqlite3
import threading as _threading
from contextlib import contextmanager

_batch_local = _threading.local()


def _batch_depth(conn: sqlite3.Connection) -> int:
    reg = getattr(_batch_local, "depth_map", None)
    if reg is None:
        return 0
    return reg.get(id(conn), 0)


def _batch_bump(conn: sqlite3.Connection, delta: int) -> int:
    reg = getattr(_batch_local, "depth_map", None)
    if reg is None:
        reg = {}
        _batch_local.depth_map = reg
    reg[id(conn)] = reg.get(id(conn), 0) + delta
    new_depth = reg[id(conn)]
    if new_depth <= 0:
        reg.pop(id(conn), None)
    return new_depth


@contextmanager
def batch_context(conn: sqlite3.Connection):
    """Defer commits until block end; rollback on exception. Nested-safe."""
    is_outer = _batch_depth(conn) == 0
    _batch_bump(conn, +1)
    try:
        yield conn
        if is_outer:
            conn.commit()
    except Exception:
        if is_outer:
            conn.rollback()
        raise
    finally:
        _batch_bump(conn, -1)


def _maybe_commit(conn: sqlite3.Connection) -> None:
    if _batch_depth(conn) == 0:
        conn.commit()
