import json
import sqlite3

from src.api.routes.dashboard import (
    _latest_ci_success,
    _norm_repo,
    _supersede_stale_reds,
)


def _ci(repo, wf, conclusion, urgency, fired):
    return {"type": "repo_ci", "repo": repo, "workflow": wf,
            "conclusion": conclusion, "urgency": urgency, "fired_at": fired}


def _succ_map(items):
    """Den latest-success-Map aus In-Page-Items bauen (wie der Endpoint es früher tat) —
    für die reine Downgrade-Logik. Die LIMIT-Unabhängigkeit testen die DB-Fälle unten."""
    m: dict[tuple[str, str], str] = {}
    for n in items:
        if n["type"] == "repo_ci" and n["conclusion"] == "success":
            k = (_norm_repo(n["repo"]), n["workflow"])
            if k not in m or n["fired_at"] > m[k]:
                m[k] = n["fired_at"]
    return m


def _db(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE hook_events "
                 "(id INTEGER PRIMARY KEY, hook_type TEXT, payload TEXT, fired_at TEXT)")
    for ht, payload, fired in rows:
        conn.execute("INSERT INTO hook_events(hook_type,payload,fired_at) VALUES (?,?,?)",
                     (ht, json.dumps(payload), fired))
    conn.commit()
    return conn


def test_red_resolved_by_later_success_is_downgraded():
    items = [
        _ci("o/r", "ci", "success", "green", "2026-06-08T02:00:00Z"),
        _ci("o/r", "ci", "failure", "red", "2026-06-08T01:00:00Z"),
    ]
    _supersede_stale_reds(items, _succ_map(items))
    assert items[1]["urgency"] == "green" and items[1]["superseded"] is True


def test_red_without_later_success_stays_red():
    items = [
        _ci("o/r", "ci", "failure", "red", "2026-06-08T02:00:00Z"),
        _ci("o/r", "ci", "success", "green", "2026-06-08T01:00:00Z"),  # OLDER success
    ]
    _supersede_stale_reds(items, _succ_map(items))
    assert items[0]["urgency"] == "red" and "superseded" not in items[0]


def test_success_in_other_workflow_does_not_resolve():
    items = [
        _ci("o/r", "deploy", "success", "green", "2026-06-08T02:00:00Z"),
        _ci("o/r", "ci", "failure", "red", "2026-06-08T01:00:00Z"),
    ]
    _supersede_stale_reds(items, _succ_map(items))
    assert items[1]["urgency"] == "red"  # different workflow → not superseded


def test_latest_ci_success_is_limit_independent():
    # Der reale Vorfall: failure (T0), dann viele Events, dann der auflösende success (T2).
    # Läge die Auflösung nur auf der Top-50-Seite, fiele der success raus → bliebe rot.
    rows = [("repo_ci", {"repo": "o/r", "workflow": "ci", "conclusion": "failure"},
             "2026-06-22T01:00:00Z")]
    for i in range(60):  # > LIMIT 50 Distraktoren dazwischen
        rows.append(("repo_ci", {"repo": "o/other", "workflow": "ci", "conclusion": "failure"},
                     f"2026-06-22T02:{i % 60:02d}:00Z"))
    rows.append(("repo_ci", {"repo": "o/r", "workflow": "ci", "conclusion": "success"},
                 "2026-06-22T03:00:00Z"))
    succ = _latest_ci_success(_db(rows))
    assert succ[("o/r", "ci")] == "2026-06-22T03:00:00Z"

    items = [_ci("o/r", "ci", "failure", "red", "2026-06-22T01:00:00Z")]
    _supersede_stale_reds(items, succ)
    assert items[0]["urgency"] == "green" and items[0]["superseded"] is True


def test_latest_ci_success_normalizes_repo_format_drift():
    # failure als Webhook-Format, success als repo-watch-Format → trotzdem aufgelöst.
    rows = [
        ("repo_ci", {"repo": "https://github.com/O/R.git", "workflow": "ci",
                     "conclusion": "failure"}, "2026-06-22T01:00:00Z"),
        ("repo_ci", {"repo": "O/R", "workflow": "ci", "conclusion": "success"},
         "2026-06-22T02:00:00Z"),
    ]
    succ = _latest_ci_success(_db(rows))
    items = [_ci("https://github.com/O/R.git", "ci", "failure", "red", "2026-06-22T01:00:00Z")]
    _supersede_stale_reds(items, succ)
    assert items[0]["urgency"] == "green"
