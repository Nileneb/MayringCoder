from src.api.routes.dashboard import _workspace_kind


def test_workspace_kind_classification():
    assert _workspace_kind("unclaimed:abc", "myws") == "unclaimed"
    assert _workspace_kind("system", "myws") == "infra"
    assert _workspace_kind("public", "myws") == "infra"
    assert _workspace_kind("bene:logs", "myws") == "infra"
    assert _workspace_kind("myws", "myws") == "mine"
    assert _workspace_kind("019d6933-old", "myws") == "legacy"
