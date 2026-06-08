from src.api.routes.dashboard import _supersede_stale_reds


def _ci(repo, wf, conclusion, urgency, fired):
    return {"type": "repo_ci", "repo": repo, "workflow": wf,
            "conclusion": conclusion, "urgency": urgency, "fired_at": fired}


def test_red_resolved_by_later_success_is_downgraded():
    # fired_at DESC order (as the endpoint produces): newest success first
    items = [
        _ci("o/r", "ci", "success", "green", "2026-06-08T02:00:00Z"),
        _ci("o/r", "ci", "failure", "red", "2026-06-08T01:00:00Z"),
    ]
    _supersede_stale_reds(items)
    assert items[1]["urgency"] == "green" and items[1]["superseded"] is True


def test_red_without_later_success_stays_red():
    items = [
        _ci("o/r", "ci", "failure", "red", "2026-06-08T02:00:00Z"),
        _ci("o/r", "ci", "success", "green", "2026-06-08T01:00:00Z"),  # OLDER success
    ]
    _supersede_stale_reds(items)
    assert items[0]["urgency"] == "red" and "superseded" not in items[0]


def test_success_in_other_workflow_does_not_resolve():
    items = [
        _ci("o/r", "deploy", "success", "green", "2026-06-08T02:00:00Z"),
        _ci("o/r", "ci", "failure", "red", "2026-06-08T01:00:00Z"),
    ]
    _supersede_stale_reds(items)
    assert items[1]["urgency"] == "red"  # different workflow → not superseded
