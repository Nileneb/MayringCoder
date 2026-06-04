"""Migration: collapse all non-universal codebooks into the ONE 'universal' codebook.

Covers move (name free in universal), merge (name collision → repoint chunk_categories
with PK-conflict dedup, sum evidence, repoint parent refs, drop row+codebook), and the
dry_run plan. See src/api/routes/codebook_admin.py."""
from mayring_core.memory.store import init_memory_db

from src.api.routes.codebook_admin import collapse_codebooks


class _FakeChroma:
    def __init__(self):
        self.deleted = []

    def delete(self, ids):
        self.deleted.extend(ids)


def _seed(tmp_path):
    conn = init_memory_db(tmp_path / "m.db")
    conn.execute("INSERT INTO codebooks(id,slug,description,auto_promote_threshold,"
                 "created_at,updated_at) VALUES (5,'universal','',3,'t','t'),"
                 "(1,'generic','',3,'t','t')")
    # universal: api(10)
    conn.execute("INSERT INTO codebook_categories(id,codebook_id,name,status,embedding_id,"
                 "evidence_count) VALUES (10,5,'api','active','u:api',2)")
    # generic: api(20) collides→merge, laravel_thing(21) moves, sub(22) moves (parent=20→10)
    conn.execute("INSERT INTO codebook_categories(id,codebook_id,name,status,embedding_id,"
                 "evidence_count) VALUES (20,1,'api','active','g:api',3),"
                 "(21,1,'laravel_thing','active','g:lt',1)")
    conn.execute("INSERT INTO codebook_categories(id,codebook_id,name,status,embedding_id,"
                 "evidence_count,parent_id) VALUES (22,1,'sub','active','g:sub',1,20)")
    # chunk links: c1→both 10&20 (dedup on repoint), c2→20 (clean repoint), c3→21 (moves)
    conn.execute("INSERT INTO chunk_categories(chunk_id,category_id,confidence,source) VALUES "
                 "('c1',10,0.9,'deductive'),('c1',20,0.8,'deductive'),"
                 "('c2',20,0.7,'deductive'),('c3',21,0.6,'deductive')")
    conn.commit()
    return conn


def test_dry_run_classifies_without_writing(tmp_path):
    conn = _seed(tmp_path)
    plan = collapse_codebooks(conn, None, dry_run=True)
    assert plan["dry_run"] is True
    assert plan["universal_id"] == 5
    assert plan["merged"] == 1          # api
    assert plan["moved"] == 2           # laravel_thing, sub
    assert plan["codebooks_deleted"] == 1
    # nothing written
    assert conn.execute("SELECT COUNT(*) FROM codebooks").fetchone()[0] == 2
    assert conn.execute("SELECT codebook_id FROM codebook_categories WHERE id=21").fetchone()[0] == 1


def test_real_collapse_moves_merges_dedups_deletes(tmp_path):
    conn = _seed(tmp_path)
    chroma = _FakeChroma()
    plan = collapse_codebooks(conn, chroma, dry_run=False)

    assert plan["merged"] == 1
    assert plan["moved"] == 2
    assert plan["chunk_links_repointed"] == 1   # c2
    assert plan["chunk_links_deduped"] == 1     # c1 (already linked to 10)
    assert plan["codebooks_deleted"] == 1

    # generic codebook gone, universal remains
    assert conn.execute("SELECT COUNT(*) FROM codebooks").fetchone()[0] == 1
    assert conn.execute("SELECT slug FROM codebooks").fetchone()[0] == "universal"

    # merged 'api' row (20) gone, all categories now in universal (5)
    assert conn.execute("SELECT COUNT(*) FROM codebook_categories WHERE id=20").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM codebook_categories WHERE codebook_id!=5").fetchone()[0] == 0

    # evidence summed onto universal api (2 + 3)
    assert conn.execute("SELECT evidence_count FROM codebook_categories WHERE id=10").fetchone()[0] == 5

    # parent of moved 'sub' repointed 20→10
    assert conn.execute("SELECT parent_id FROM codebook_categories WHERE id=22").fetchone()[0] == 10

    # chunk_categories: c1→10 (deduped to one), c2→10 (repointed), c3→21 (moved cat)
    rows = dict(conn.execute("SELECT chunk_id, category_id FROM chunk_categories ORDER BY chunk_id").fetchall())
    assert rows == {"c1": 10, "c2": 10, "c3": 21}
    assert conn.execute("SELECT COUNT(*) FROM chunk_categories WHERE category_id=20").fetchone()[0] == 0

    # merged-away embedding dropped from chroma
    assert "g:api" in chroma.deleted


def test_collapse_idempotent(tmp_path):
    conn = _seed(tmp_path)
    collapse_codebooks(conn, _FakeChroma(), dry_run=False)
    plan2 = collapse_codebooks(conn, _FakeChroma(), dry_run=False)
    assert plan2["moved"] == 0 and plan2["merged"] == 0 and plan2["codebooks_deleted"] == 0
