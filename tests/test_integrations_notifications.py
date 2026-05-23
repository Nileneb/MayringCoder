"""Tests for #270 Phase 1: GitHub-Event-Classifier + Notifications-Store."""
import sqlite3

from src.api.integrations.github_events import classify_github_event
from src.api.integrations import notifications_store as ns


# --- classifier -------------------------------------------------------------

def test_workflow_run_failure_classified():
    out = classify_github_event("workflow_run", {
        "repository": {"full_name": "Nileneb/MayringCoder"},
        "workflow_run": {"status": "completed", "conclusion": "failure",
                         "name": "Post-deploy smoke", "html_url": "http://x"},
    })
    assert out["kind"] == "ci_failure"
    assert out["repo"] == "Nileneb/MayringCoder"


def test_workflow_run_success_ignored():
    out = classify_github_event("workflow_run", {
        "workflow_run": {"status": "completed", "conclusion": "success"},
    })
    assert out is None


def test_security_alert_classified_high():
    out = classify_github_event("code_scanning_alert", {
        "repository": {"full_name": "r"},
        "alert": {"severity": "critical", "html_url": "http://a"},
    })
    assert out["kind"] == "security"
    assert out["severity"] == "critical"


def test_push_to_default_branch_classified():
    out = classify_github_event("push", {
        "ref": "refs/heads/master",
        "after": "abc123",
        "repository": {"full_name": "r", "default_branch": "master"},
    })
    assert out["kind"] == "push"


def test_push_to_feature_branch_ignored():
    out = classify_github_event("push", {
        "ref": "refs/heads/feature/x",
        "repository": {"full_name": "r", "default_branch": "master"},
    })
    assert out is None


def test_unknown_event_ignored():
    assert classify_github_event("watch", {}) is None


# --- store ------------------------------------------------------------------

def _conn(tmp_path):
    return sqlite3.connect(str(tmp_path / "n.db"))


def test_record_and_query_unacked(tmp_path):
    conn = _conn(tmp_path)
    ns.record_notification(conn, kind="ci_failure", severity="medium",
                           title="CI failed", repo="r")
    ns.record_notification(conn, kind="security", severity="high", title="alert")
    items = ns.unacked_since(conn)
    assert len(items) == 2
    assert {i["kind"] for i in items} == {"ci_failure", "security"}


def test_ack_removes_from_unacked(tmp_path):
    conn = _conn(tmp_path)
    nid = ns.record_notification(conn, kind="pr", title="PR #1")
    ns.record_notification(conn, kind="push", title="push")
    ns.ack(conn, [nid])
    items = ns.unacked_since(conn)
    assert len(items) == 1
    assert items[0]["kind"] == "push"


def test_since_filter(tmp_path):
    conn = _conn(tmp_path)
    ns.record_notification(conn, kind="a", title="old")
    # since in the far future → nothing newer
    items = ns.unacked_since(conn, since="2999-01-01T00:00:00Z")
    assert items == []
