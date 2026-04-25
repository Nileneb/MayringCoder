from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

_MEMORY_READY = False
_IMPORT_ERROR: str = ""

try:
    from src.memory.store import (
        init_memory_db,
        add_feedback,
        get_chunks_by_source,
    )
    from src.memory.ingest import ingest, ingest_conversation_summary
    from src.memory.retrieval import search
    from src.memory.schema import Source, Chunk
    from src.github import parse_github_input, GitHubInputError
    from src.config import CACHE_DIR
    import hashlib

    _MEMORY_READY = True
except Exception as exc:
    _IMPORT_ERROR = str(exc)

_conn: sqlite3.Connection | None = None
_chroma_collection: Any = None
_ollama_url: str = "http://localhost:11434"
_api_url: str = "http://localhost:8080"


def set_runtime_urls(ollama_url: str, api_url: str) -> None:
    global _ollama_url, _api_url
    _ollama_url = ollama_url
    _api_url = api_url


def _get_conn() -> sqlite3.Connection | None:
    global _conn
    if _conn is not None:
        return _conn
    if not _MEMORY_READY:
        return None
    try:
        _conn = init_memory_db()
        return _conn
    except Exception:
        return None


def _get_chroma() -> Any:
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if not _MEMORY_READY:
        return None
    try:
        from src.memory.store import get_chroma_collection as get_collection
        _chroma_collection = get_collection()
        return _chroma_collection
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_html(available: bool, models: list[str]) -> str:
    color = "#22c55e" if available else "#ef4444"
    label = "Ollama online" if available else "Ollama offline"
    model_hint = f" ({', '.join(models[:3])})" if models else ""
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f'padding:4px 12px;border-radius:9999px;background:{color}20;'
        f'border:1px solid {color};font-size:0.85rem;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{color};'
        f'display:inline-block;"></span>'
        f'<span style="color:{color};font-weight:600;">{label}{model_hint}</span>'
        f"</div>"
    )


def _do_search(query: str, top_k: int, ollama_available: bool) -> tuple[str, list[list]]:
    if not _MEMORY_READY:
        return f"Memory-Module nicht geladen: {_IMPORT_ERROR}", []
    if not query.strip():
        return "Bitte Suchbegriff eingeben.", []

    conn = _get_conn()
    if conn is None:
        return "SQLite-Datenbank nicht verfugbar.", []

    chroma = _get_chroma() if ollama_available else None

    try:
        results = search(
            query=query,
            conn=conn,
            chroma_collection=chroma,
            ollama_url=_ollama_url,
            opts={"top_k": int(top_k)},
        )
    except Exception as exc:
        return f"Suchfehler: {exc}", []

    if not results:
        mode = "symbolisch" if not ollama_available else "hybrid"
        return f"Keine Ergebnisse ({mode} Suche).", []

    rows = []
    for r in results:
        cats = ", ".join(r.category_labels) if r.category_labels else "-"
        text_preview = r.text[:120].replace("\n", " ") + ("..." if len(r.text) > 120 else "")
        rows.append([
            r.chunk_id[:12],
            f"{r.score_final:.3f}",
            r.source_id,
            cats,
            text_preview,
            ", ".join(r.reasons),
        ])

    mode = "symbolisch" if not ollama_available else "hybrid"
    return f"{len(results)} Ergebnis(se) ({mode} Suche).", rows


def _load_sources(repo_filter: str, category_filter: str) -> list[list]:
    if not _MEMORY_READY:
        return []
    conn = _get_conn()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT source_id, source_type, repo, path, captured_at FROM sources ORDER BY captured_at DESC"
        ).fetchall()
    except Exception:
        return []

    result = []
    for r in rows:
        sid, stype, srepo, spath, cat = r
        if repo_filter and repo_filter.lower() not in srepo.lower():
            continue
        result.append([sid, stype, srepo, spath, cat])
    return result


def _load_chunks_for_source(source_id: str) -> str:
    if not source_id or not _MEMORY_READY:
        return "_Keine Source ausgewahlt._"
    conn = _get_conn()
    if conn is None:
        return "_Datenbank nicht verfugbar._"
    try:
        chunks = get_chunks_by_source(conn, source_id, active_only=True)
    except Exception as exc:
        return f"_Fehler: {exc}_"
    if not chunks:
        return f"_Keine aktiven Chunks fur `{source_id}`._"

    lines = [f"## Chunks fur `{source_id}`", f"Gesamt: {len(chunks)}", ""]
    for c in chunks[:20]:
        cats = ", ".join(c.category_labels) or "-"
        preview = c.text[:200].replace("\n", " ")
        lines.append(f"**{c.chunk_id[:12]}** | Level: `{c.chunk_level}` | Kategorien: {cats}")
        lines.append(f"> {preview}{'...' if len(c.text) > 200 else ''}")
        lines.append("")
    if len(chunks) > 20:
        lines.append(f"_... und {len(chunks) - 20} weitere Chunks._")
    return "\n".join(lines)


def _do_ingest(
    text_input: str,
    file_upload,
    source_path: str,
    repo: str,
    categorize: bool,
    mode: str,
    codebook: str,
    model: str,
    ollama_available: bool,
) -> str:
    if not _MEMORY_READY:
        return json.dumps({"error": f"Memory-Module nicht geladen: {_IMPORT_ERROR}"}, indent=2)

    warning = (
        "[Warnung] Ollama nicht erreichbar. Embedding wird ubersprungen."
        if not ollama_available else ""
    )

    content = ""
    if file_upload is not None:
        try:
            content = Path(file_upload.name).read_text(encoding="utf-8", errors="replace")
            if not source_path.strip():
                source_path = Path(file_upload.name).name
        except Exception as exc:
            return json.dumps({"error": f"Datei konnte nicht gelesen werden: {exc}"}, indent=2)
    elif text_input.strip():
        try:
            gh_hint = parse_github_input(text_input.strip())
        except GitHubInputError:
            gh_hint = None
        if gh_hint is not None:
            return json.dumps({
                "error": (
                    f"Das sieht nach einem GitHub-Repo aus ({gh_hint.slug}). "
                    f"Bitte die Quelle 'GitHub-Repo' wählen, dort werden Sources + Issues "
                    f"als Job ingested."
                )
            }, indent=2)
        content = text_input.strip()
    else:
        return json.dumps({"error": "Kein Inhalt angegeben (Text oder Datei)."}, indent=2)

    if not source_path.strip():
        source_path = "manual_input.txt"

    try:
        content_hash = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        source_id = Source.make_id(repo.strip() or "ui", source_path.strip())
        source = Source(
            source_id=source_id,
            source_type="repo_file" if repo.strip() else "note",
            repo=repo.strip(),
            path=source_path.strip(),
            content_hash=content_hash,
            captured_at=_now_iso(),
        )
    except Exception as exc:
        return json.dumps({"error": f"Source-Erstellung fehlgeschlagen: {exc}"}, indent=2)

    conn = _get_conn()
    if conn is None:
        return json.dumps({"error": "SQLite-Datenbank nicht verfugbar."}, indent=2)

    chroma = _get_chroma() if ollama_available else None

    try:
        result = ingest(
            source=source,
            content=content,
            conn=conn,
            chroma_collection=chroma,
            ollama_url=_ollama_url,
            model=model,
            opts={
                "categorize": categorize and ollama_available,
                "mode": mode,
                "codebook": codebook,
                "log": True,
            },
        )
    except Exception as exc:
        return json.dumps({"error": f"Ingest fehlgeschlagen: {exc}"}, indent=2)

    output = dict(result)
    if warning:
        output["warning"] = warning
    return json.dumps(output, indent=2, ensure_ascii=False)


def _do_ingest_conversation(
    summary_text: str,
    session_id: str,
    run_id: str,
    model: str,
    ollama_available: bool,
) -> str:
    if not _MEMORY_READY:
        return json.dumps({"error": f"Memory-Module nicht geladen: {_IMPORT_ERROR}"}, indent=2)
    if not summary_text.strip():
        return json.dumps({"error": "Kein Inhalt angegeben."}, indent=2)

    conn = _get_conn()
    if conn is None:
        return json.dumps({"error": "SQLite-Datenbank nicht verfügbar."}, indent=2)

    chroma = _get_chroma() if ollama_available else None

    try:
        result = ingest_conversation_summary(
            summary_text=summary_text.strip(),
            conn=conn,
            chroma_collection=chroma,
            ollama_url=_ollama_url,
            model=model if ollama_available else "",
            session_id=session_id.strip() or None,
            run_id=run_id.strip() or None,
        )
    except Exception as exc:
        return json.dumps({"error": f"Conversation-Ingest fehlgeschlagen: {exc}"}, indent=2)

    return json.dumps(result, indent=2, ensure_ascii=False)


def _scan_compact_files(model: str, ollama_available: bool) -> str:
    if not _MEMORY_READY:
        return f"Memory nicht verfügbar: {_IMPORT_ERROR}"

    claude_base = Path.home() / ".claude" / "projects"
    if not claude_base.exists():
        return "Kein ~/.claude/projects/ Verzeichnis gefunden."

    md_files = list(claude_base.rglob("memory/*.md"))
    if not md_files:
        return "Keine Compact-Dateien gefunden."

    conn = _get_conn()
    if conn is None:
        return "SQLite nicht verfügbar."

    chroma = _get_chroma() if ollama_available else None
    ingested, skipped, errors = 0, 0, 0

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **_kw):  # type: ignore[misc]
            return it

    for f in _tqdm(md_files, desc="Claude-Dateien scannen", unit="Datei"):
        try:
            content = f.read_text(encoding="utf-8")
            if not content.strip():
                skipped += 1
                continue
            content_hash = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
            source_id = f"note:claude_memory:{f.stem}"
            source = Source(
                source_id=source_id,
                source_type="note",
                repo="claude_memory",
                path=str(f.relative_to(Path.home())),
                content_hash=content_hash,
                captured_at=_now_iso(),
            )
            result = ingest_conversation_summary(
                summary_text=content,
                conn=conn,
                chroma_collection=chroma,
                ollama_url=_ollama_url,
                model=model if ollama_available else "",
            )
            if result.get("skipped"):
                skipped += 1
            else:
                ingested += 1
        except Exception:
            errors += 1

    parts = []
    if ingested:
        parts.append(f"✅ {ingested} neu ingested")
    if skipped:
        parts.append(f"⏭ {skipped} unverändert übersprungen")
    if errors:
        parts.append(f"❌ {errors} Fehler")
    return " · ".join(parts) if parts else "Nichts zu tun."


def _do_feedback(chunk_id: str, signal: str, label: str, token: str) -> str:
    if not token:
        return "Erst einloggen."
    if not chunk_id.strip():
        return "Chunk-ID fehlt."
    payload: dict = {"chunk_id": chunk_id.strip(), "signal": signal}
    if label.strip():
        payload["metadata"] = {"label": label.strip()}
    r = _api_post("memory/feedback", payload, token)
    if "error" in r:
        return f"Fehler: {r['error']}"
    return f"Feedback gespeichert: chunk_id={chunk_id.strip()}, signal={signal}"


def _validate_token(token: str) -> tuple[bool, str]:
    if not token.strip():
        return False, "Kein Token eingegeben."
    if _HAS_HTTPX:
        try:
            r = _httpx.post(
                f"{_api_url}/memory/search",
                json={"query": "ping", "top_k": 1},
                headers={"Authorization": f"Bearer {token.strip()}"},
                timeout=5.0,
            )
            if r.status_code == 200:
                return True, r.json().get("workspace_id", "default")
            if r.status_code == 402:
                return False, "Kein aktives Mayring-Abo (€5/Monat auf app.linn.games)."
            if r.status_code == 401:
                return False, "Token ungültig oder abgelaufen."
        except Exception:
            pass
    try:
        from src.api.jwt_auth import validate_jwt_token
        info = validate_jwt_token(token.strip())
        if info:
            return True, info.workspace_id
        return False, "Token ungültig oder abgelaufen."
    except Exception as exc:
        return False, f"Auth-Fehler: {exc}"


def _laravel_base() -> str:
    return os.environ.get("LARAVEL_INTERNAL_URL", "https://app.linn.games").rstrip("/")


def _laravel_headers() -> dict[str, str]:
    base = _laravel_base()
    return {"Host": "app.linn.games"} if base.startswith("http://") else {}


def refresh_jwt(old_token: str) -> str | None:
    if not _HAS_HTTPX or not old_token:
        return None
    try:
        resp = _httpx.post(
            f"{_laravel_base()}/api/mayring/refresh-token",
            headers={
                "Authorization": f"Bearer {old_token}",
                **_laravel_headers(),
            },
            timeout=5.0,
        )
        if resp.status_code == 200:
            fresh = (resp.json() or {}).get("token", "").strip()
            return fresh or None
    except Exception:
        pass
    return None


def _api_post(endpoint: str, payload: dict, token: str) -> dict:
    if not _HAS_HTTPX:
        return {"error": "httpx nicht installiert"}
    if not token:
        return {"error": "Nicht eingeloggt."}
    url = f"{_api_url}/{endpoint.lstrip('/')}"
    try:
        r = _httpx.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        if r.status_code == 401:
            fresh = refresh_jwt(token)
            if fresh:
                r = _httpx.post(
                    url, json=payload,
                    headers={"Authorization": f"Bearer {fresh}"},
                    timeout=10.0,
                )
                if r.status_code == 200:
                    out = r.json()
                    if isinstance(out, dict):
                        out["_refreshed_token"] = fresh
                    return out
            return {"error": "Session abgelaufen. Bitte neu einloggen via app.linn.games/mayring/dashboard."}
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def _api_get_job(job_id: str, token: str) -> dict:
    if not _HAS_HTTPX or not token:
        return {"error": "Nicht verbunden."}
    try:
        r = _httpx.get(
            f"{_api_url}/jobs/{job_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def _poll_job(job_id: str, token: str) -> str:
    if not job_id or not token:
        return "Job-ID oder Token fehlt."
    r = _api_get_job(job_id, token)
    if "error" in r:
        return f"❌ {r['error']}"
    status = r.get("status", "?")
    output = r.get("output", "")
    return f"**Status:** {status}\n\n```\n{output[-3000:] if output else ''}\n```"


def _training_files() -> list[Path]:
    if not _MEMORY_READY:
        return []
    try:
        from src.config import CACHE_DIR
        return sorted(CACHE_DIR.glob("*_training_log.jsonl"))
    except Exception:
        return []
