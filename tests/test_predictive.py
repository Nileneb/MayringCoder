"""Tests for predictive topic transitions (Markov chain)."""
from __future__ import annotations
import json

from src.memory.predictive import (
    TopicTransition,
    _extract_topics_from_text,
    build_transition_matrix,
    predict_next_topics,
    persist_transitions,
    load_transitions,
    update_transitions_incremental,
    predict_next_topics_for_query,
)
from src.memory.store import init_memory_db


def test_extract_topics_ordered_and_dedup():
    kw_index = {"auth": ["Authentication"], "login": ["Authentication"], "payment": ["Billing"]}
    topics = _extract_topics_from_text("User auth login then payment; auth again", kw_index)
    assert topics == ["Authentication", "Billing"]


def test_extract_topics_empty_on_missing_keywords():
    assert _extract_topics_from_text("random text", {}) == []
    assert _extract_topics_from_text("", {"foo": ["Bar"]}) == []


def test_predict_next_topics_sorted_by_probability():
    matrix = {"A": {"B": 7, "C": 2, "D": 1}}
    preds = predict_next_topics("A", matrix, top_k=2)
    assert len(preds) == 2
    assert preds[0].to_topic == "B"
    assert preds[0].probability == 0.7
    assert preds[1].to_topic == "C"


def test_predict_next_empty_on_unknown_topic():
    assert predict_next_topics("Nope", {"A": {"B": 1}}) == []


def test_persist_and_load_transitions_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    conn = init_memory_db(db)
    matrix = {"A": {"B": 5, "C": 1}, "B": {"A": 3}}
    persist_transitions(matrix, conn)
    loaded = load_transitions(conn)
    assert loaded == matrix


def test_persist_transitions_upsert_updates_count(tmp_path):
    db = tmp_path / "t.db"
    conn = init_memory_db(db)
    persist_transitions({"A": {"B": 1}}, conn)
    persist_transitions({"A": {"B": 9}}, conn)
    loaded = load_transitions(conn)
    assert loaded["A"]["B"] == 9


def test_build_transition_matrix_from_summaries(tmp_path, monkeypatch):
    import src.memory.predictive as pred_mod
    db = tmp_path / "t.db"
    conn = init_memory_db(db)

    conn.execute(
        'INSERT INTO sources(source_id, source_type, repo, path, branch, "commit", content_hash, captured_at) VALUES(?,?,?,?,?,?,?,?)',
        ("conversation:demo:s1", "conversation_summary", "demo", "demo/1", "local", "", "sha256:abc", "2026-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT INTO chunks(chunk_id, source_id, chunk_level, ordinal, start_offset, end_offset, text, text_hash, dedup_key, created_at, is_active) VALUES(?,?,?,?,?,?,?,?,?,?,1)",
        ("c1", "conversation:demo:s1", "file", 0, 0, 10, "auth flow then billing", "h1", "d1", "2026-01-01T00:00:00"),
    )
    conn.commit()

    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"], "billing": ["Billing"]})

    matrix = build_transition_matrix(conn, repo_slug="demo", limit=10)
    assert matrix == {"Auth": {"Billing": 1}}


def test_build_transition_matrix_empty_when_no_index(tmp_path, monkeypatch):
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index", lambda slug: {})
    assert build_transition_matrix(conn) == {}


def test_incremental_update_bumps_counts(tmp_path, monkeypatch):
    """Issue #55 follow-up: per-conversation incremental updates instead
    of batch rebuild. First call inserts (count=1), second on overlapping
    pairs increments via UPSERT."""
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"], "billing": ["Billing"], "search": ["Search"]})

    # First summary: Auth → Billing
    n1 = update_transitions_incremental("auth flow then billing", conn, "demo")
    assert n1 == 1
    assert load_transitions(conn) == {"Auth": {"Billing": 1}}

    # Second summary: Auth → Billing again, plus Billing → Search
    n2 = update_transitions_incremental("auth then billing then search", conn, "demo")
    assert n2 == 2
    matrix = load_transitions(conn)
    assert matrix == {"Auth": {"Billing": 2}, "Billing": {"Search": 1}}


def test_incremental_skips_when_no_index(tmp_path, monkeypatch):
    """No wiki_index.json yet (cold-start): hook must not crash + return 0."""
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index", lambda slug: {})
    assert update_transitions_incremental("some text", conn, "demo") == 0


def test_incremental_skips_self_transitions(tmp_path, monkeypatch):
    """A → A is meaningless (text repeats same topic) — must not bump."""
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"], "login": ["Auth"]})
    # extract_topics dedup'es to ["Auth"] only → no transitions possible
    n = update_transitions_incremental("auth login auth login", conn, "demo")
    assert n == 0
    assert load_transitions(conn) == {}


def test_predict_for_query_uses_persisted_matrix(tmp_path, monkeypatch):
    """The retrieval-side predictor must read from DB, not need the in-memory
    matrix passed by the caller."""
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"], "billing": ["Billing"]})

    persist_transitions({"Auth": {"Billing": 4, "Search": 1}}, conn)

    preds = predict_next_topics_for_query("does auth work?", conn, "demo", top_k=2)
    assert len(preds) >= 1
    assert preds[0].from_topic == "Auth"
    assert preds[0].to_topic == "Billing"  # 4/5 = 0.8 wins over Search 1/5


def test_predict_for_query_empty_when_query_has_no_known_topics(tmp_path, monkeypatch):
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")
    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"]})
    persist_transitions({"Auth": {"Billing": 1}}, conn)
    assert predict_next_topics_for_query("nothing relevant here", conn, "demo") == []


def test_slug_aliases_includes_short_form_and_lowercase():
    """Conversation-watcher pre-2026-05 schrieb repo='mayringcoder' (kurz),
    der Analysis-Run nutzt repo='nileneb-mayringcoder'. Ohne Alias-Match
    findet build_transition_matrix nur 1-2% der eigentlich verfügbaren
    summaries."""
    from src.memory.predictive import _slug_aliases
    aliases = _slug_aliases("nileneb-mayringcoder")
    assert "nileneb-mayringcoder" in aliases
    assert "mayringcoder" in aliases  # short form
    # mixed-case input → lowercase variant
    aliases_mc = _slug_aliases("Nileneb-MayringCoder")
    assert "nileneb-mayringcoder" in aliases_mc


def test_load_keyword_index_rejects_path_traversal(tmp_path, monkeypatch):
    """Issue #185 / CodeQL #129+#130: repo_slug kommt aus user-controlled
    workspace_slug im /conversation/micro-batch endpoint. Setup einen
    'evil'-payload außerhalb von cache/ — ein NICHT-sanitizing
    _load_keyword_index würde ihn lesen.

    Construction: tmp_path/payload/EVIL_wiki_index.json existiert.
    Mit cwd=tmp_path/work und repo_slug='../payload/EVIL' baut der
    naive Code 'cache/../payload/EVIL_wiki_index.json' was AUF DAS
    EVIL FILE auflöst → liest Angreifer-payload. Sicherer Code muss
    ein leeres dict liefern."""
    from src.memory import predictive as pred_mod

    # Setup: evil payload outside cache/
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    (payload_dir / "EVIL_wiki_index.json").write_text('{"pwned": ["yes"]}')

    # Innocuous workdir with empty cache/
    work = tmp_path / "work"
    (work / "cache").mkdir(parents=True)
    monkeypatch.chdir(work)

    # Naive: Path("cache") / f"{slug}_wiki_index.json" mit slug='../../payload/EVIL'
    # → "cache/../../payload/EVIL_wiki_index.json" → resolved zu
    # tmp_path/payload/EVIL_wiki_index.json (das evil-file). Sanitized
    # code muss diesen Aufruf abwehren und {} zurückgeben.
    evil_slug = "../../payload/EVIL"
    naive_path = (work / "cache" / f"{evil_slug}_wiki_index.json").resolve()
    assert naive_path == (payload_dir / "EVIL_wiki_index.json").resolve(), \
        f"test setup wrong: naive_path={naive_path}"
    assert naive_path.exists(), "evil file must exist for the test to be meaningful"

    # Now the actual security check: sanitized loader must NOT read it
    assert pred_mod._load_keyword_index(evil_slug) == {}, \
        "PATH TRAVERSAL: function read evil payload outside cache/"

    # Other malicious patterns
    assert pred_mod._load_keyword_index("/etc/passwd") == {}
    assert pred_mod._load_keyword_index("..") == {}
    assert pred_mod._load_keyword_index("a/b") == {}
    assert pred_mod._load_keyword_index("") == {}
    # None must not crash
    assert pred_mod._load_keyword_index(None) == {}  # type: ignore[arg-type]


def test_load_keyword_index_accepts_valid_slug(tmp_path, monkeypatch):
    """Sanity: legitimate slugs keep working."""
    from src.memory import predictive as pred_mod

    # Setup: real keyword-index file under tmp_path/cache/
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "nileneb-mayringcoder_wiki_index.json").write_text(
        '{"auth": ["Auth"], "billing": ["Billing"]}'
    )
    # Patch the cache-base so the function looks under tmp_path
    monkeypatch.chdir(tmp_path)

    out = pred_mod._load_keyword_index("nileneb-mayringcoder")
    assert out == {"auth": ["Auth"], "billing": ["Billing"]}

    # Underscore + dash + digits all allowed
    (cache / "nileneb_mayring-coder_v2_wiki_index.json").write_text('{"x": ["X"]}')
    out2 = pred_mod._load_keyword_index("nileneb_mayring-coder_v2")
    assert out2 == {"x": ["X"]}


def test_search_boosts_chunks_matching_predicted_topics(tmp_path, monkeypatch):
    """Issue #184: when /memory/search predicts that the user-query about
    topic A is likely followed by topic B, chunks tagged with B (in
    category_labels OR text) should rank higher than otherwise-identical
    peers. End-to-end test through retrieval.search().

    Setup:
      - 2 chunks with same vector/symbolic features, same source
      - chunk_match.category_labels = ['MCP'] (matches a predicted topic)
      - chunk_other.category_labels = ['unrelated']
      - persisted topic_transitions: Auth → MCP (high count)
      - query 'auth setup' → predicted next-topics include 'MCP'
      - chunk_match must rank above chunk_other in the result
    """
    import src.memory.predictive as pred_mod
    from src.memory.retrieval import search
    from src.memory.schema import Chunk, Source
    from src.memory.store import insert_chunk, upsert_source

    conn = init_memory_db(tmp_path / "t.db")

    # Setup keyword-index so 'auth' maps to topic 'Auth'
    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"], "mcp": ["MCP"]})

    # Persist transition: Auth → MCP with high count
    pred_mod.persist_transitions({"Auth": {"MCP": 10}}, conn)

    # 2 sources + 2 chunks, identical retrieval features, different category
    src_match = Source(
        source_id="src::match", source_type="repo_file",
        repo="https://github.com/x/y", path="src/match.py",
        content_hash="sha256:m",
    )
    src_other = Source(
        source_id="src::other", source_type="repo_file",
        repo="https://github.com/x/y", path="src/other.py",
        content_hash="sha256:o",
    )
    upsert_source(conn, src_match)
    upsert_source(conn, src_other)

    chunk_match = Chunk(
        chunk_id=Chunk.make_id("src::match", 0, "function"),
        source_id="src::match", chunk_level="function", ordinal=0,
        text="def auth_handler(): return jwt", text_hash="sha256:tm",
        category_labels=["MCP"],   # matches predicted topic
        created_at="2026-04-08T10:00:00+00:00",
    )
    chunk_other = Chunk(
        chunk_id=Chunk.make_id("src::other", 0, "function"),
        source_id="src::other", chunk_level="function", ordinal=0,
        text="def auth_handler(): return jwt", text_hash="sha256:to",
        category_labels=["unrelated"],
        created_at="2026-04-08T10:00:00+00:00",
    )
    insert_chunk(conn, chunk_match)
    insert_chunk(conn, chunk_other)

    # Search with predicted-topic boost enabled. predict_next_topics_for_query
    # liest _load_keyword_index — wir patchen den slug "demo" damit das
    # zu unserer monkeypatched lambda matcht.
    results = search(
        "auth setup", conn, None, "http://fake-ollama",
        opts={"top_k": 5, "include_text": False, "llm_prefilter": False,
              "predicted_repo_slug": "demo"},
    )
    assert len(results) >= 2, f"got {len(results)} results, expected >=2"

    # chunk_match must rank above chunk_other
    rank_match = next(i for i, r in enumerate(results) if r.chunk_id == chunk_match.chunk_id)
    rank_other = next(i for i, r in enumerate(results) if r.chunk_id == chunk_other.chunk_id)
    assert rank_match < rank_other, (
        f"chunk_match (with predicted topic 'MCP') should rank ABOVE "
        f"chunk_other; got rank_match={rank_match}, rank_other={rank_other}"
    )

    # And the matched chunk's score_predicted_topic must be > 0
    match_record = results[rank_match]
    assert match_record.score_predicted_topic > 0, (
        f"score_predicted_topic must be > 0 on the matched chunk; got "
        f"{match_record.score_predicted_topic}"
    )
    # The unrelated chunk should have score_predicted_topic == 0
    other_record = results[rank_other]
    assert other_record.score_predicted_topic == 0


def test_build_transition_matches_via_short_slug(tmp_path, monkeypatch):
    """Production-Bug 2026-05-09: 89 conversation_summary mit repo='mayringcoder',
    1 mit 'nileneb-mayringcoder'. Vor Slug-Aliasing: build_transition_matrix
    fand nur die 1, jetzt findet sie alle 90."""
    import src.memory.predictive as pred_mod
    conn = init_memory_db(tmp_path / "t.db")

    # 1 source unter 'nileneb-mayringcoder', 1 unter 'mayringcoder' (legacy)
    for ix, repo_val in enumerate(["nileneb-mayringcoder", "mayringcoder"]):
        conn.execute(
            'INSERT INTO sources(source_id, source_type, repo, path, branch, '
            '"commit", content_hash, captured_at) VALUES(?,?,?,?,?,?,?,?)',
            (f"conversation:{repo_val}:s{ix}", "conversation_summary",
             repo_val, f"{repo_val}/x", "local", "", f"sha256:abc{ix}",
             f"2026-01-{ix+1:02d}T00:00:00"),
        )
        conn.execute(
            "INSERT INTO chunks(chunk_id, source_id, chunk_level, ordinal, "
            "start_offset, end_offset, text, text_hash, dedup_key, created_at, "
            "is_active) VALUES(?,?,?,?,?,?,?,?,?,?,1)",
            (f"chk_{ix}", f"conversation:{repo_val}:s{ix}", "section", 0, 0, 10,
             "auth flow then billing", f"h{ix}", f"d{ix}",
             f"2026-01-{ix+1:02d}T00:00:00"),
        )
    conn.commit()

    monkeypatch.setattr(pred_mod, "_load_keyword_index",
                        lambda slug: {"auth": ["Auth"], "billing": ["Billing"]})

    # Query mit langer Slug-Variante muss BEIDE summaries finden
    matrix = build_transition_matrix(conn, repo_slug="nileneb-mayringcoder", limit=10)
    # 2 summaries × 1 transition pro = 2 counts gesamt
    assert matrix == {"Auth": {"Billing": 2}}
