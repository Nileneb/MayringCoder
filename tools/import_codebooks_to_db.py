"""Import YAML codebooks → SQLite (memory.db) + category embeddings → Chroma.

#workspace-uuid-sot v2.0 Phase 1.2 (Python-Äquivalent zu `php artisan codebook:import`).
Reads codebooks/profiles/*.yaml (assembly) + codebooks/categories/*.yaml +
code.yaml/social.yaml (inline) → codebooks + codebook_categories rows. Each
category's "name: description" is embedded into the Chroma collection
"codebook_categories" (embedding_id = "cb:<slug>:<name>") for the v2 deductive
cosine-match. evidence_count seeded from chunks.category_labels frequency.

Idempotent (INSERT OR IGNORE / upsert). --dry-run default; --apply writes.
YAML stays the source until Phase 4 cleanup — this just mirrors it into the DB.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

CB_DIR = ROOT / "codebooks"
EMBED_MODEL = os.environ.get("MAYRING_EMBED_MODEL", "nomic-embed-text")


def _load_category_defs() -> dict[str, dict]:
    """name → {description, risk_level, languages, patterns} from categories/*.yaml
    + inline code.yaml/social.yaml."""
    defs: dict[str, dict] = {}
    cat_dir = CB_DIR / "categories"
    if cat_dir.is_dir():
        for f in cat_dir.glob("*.yaml"):
            try:
                d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                if d.get("name"):
                    defs[d["name"]] = d
            except Exception as e:  # noqa: BLE001
                print(f"  skip {f.name}: {e}", file=sys.stderr)
    for inline in ("code.yaml", "social.yaml"):
        p = CB_DIR / inline
        if p.exists():
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            for cat in d.get("categories", []):
                if isinstance(cat, dict) and cat.get("name"):
                    defs.setdefault(cat["name"], cat)
    return defs


def _label_freq(conn: sqlite3.Connection) -> Counter:
    c: Counter = Counter()
    for (labels,) in conn.execute(
            "SELECT category_labels FROM chunks WHERE is_active=1 AND category_labels!=''"):
        for lbl in str(labels).split(","):
            lbl = lbl.strip().lstrip("[neu]").strip()
            if lbl:
                c[lbl] += 1
    return c


def run(db_path: Path, apply: bool, no_chroma: bool) -> dict:
    cat_defs = _load_category_defs()
    profiles = sorted((CB_DIR / "profiles").glob("*.yaml"))
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=20000")
    conn.execute("PRAGMA foreign_keys=ON")
    freq = _label_freq(conn)
    now = datetime.now(timezone.utc).isoformat()
    rep = {"codebooks": 0, "categories": 0, "embedded": 0}
    embed_jobs: list[tuple[str, str]] = []  # (embedding_id, text)

    for pf in profiles:
        d = yaml.safe_load(pf.read_text(encoding="utf-8")) or {}
        slug = d.get("name") or pf.stem
        rep["codebooks"] += 1
        if apply:
            conn.execute(
                "INSERT OR IGNORE INTO codebooks(slug, description, created_at, updated_at) "
                "VALUES (?,?,?,?)", (slug, d.get("description", ""), now, now))
            cb_id = conn.execute("SELECT id FROM codebooks WHERE slug=?", (slug,)).fetchone()[0]
        else:
            cb_id = -1
        for name in d.get("categories", []):
            if not isinstance(name, str):
                continue
            cd = cat_defs.get(name, {})
            rep["categories"] += 1
            emb_id = f"cb:{slug}:{name}"
            embed_jobs.append((emb_id, f"{name}: {cd.get('description', name)}"))
            if apply:
                conn.execute(
                    "INSERT OR IGNORE INTO codebook_categories(codebook_id, name, "
                    "description, examples, status, source, evidence_count, embedding_id, "
                    "risk_level, languages, patterns) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (cb_id, name, cd.get("description", ""), "[]", "active", "imported",
                     int(freq.get(name, 0)), emb_id, cd.get("risk_level", ""),
                     json.dumps(cd.get("languages", [])), json.dumps(cd.get("patterns", []))))
        if apply:
            conn.commit()

    if apply and not no_chroma and embed_jobs:
        from mayring_core.ollama_client import embed_batch
        from mayring_core.memory.store import get_chroma_collection
        url = os.environ.get("OLLAMA_URL", "https://three.linn.games")
        col = get_chroma_collection("codebook_categories")
        ids = [j[0] for j in embed_jobs]
        texts = [j[1] for j in embed_jobs]
        for i in range(0, len(ids), 64):
            embs = embed_batch(url, EMBED_MODEL, texts[i:i + 64], timeout=120)
            if embs:
                col.upsert(ids=ids[i:i + 64], embeddings=embs, documents=texts[i:i + 64])
                rep["embedded"] += len(embs)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(ROOT / "cache" / "memory.db"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-chroma", action="store_true")
    args = ap.parse_args()
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== import codebooks YAML → DB  [{mode}] ===")
    rep = run(Path(args.db), args.apply, args.no_chroma)
    print(f"  codebooks={rep['codebooks']} categories={rep['categories']} embedded={rep['embedded']}")
    print("\n" + ("APPLIED." if args.apply else "DRY-RUN — re-run with --apply."))


if __name__ == "__main__":
    main()
