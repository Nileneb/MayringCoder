import importlib
from mayring_core import config as cfg


def test_defaults_present(monkeypatch):
    for var in ("MAYRING_EMBED_VERIFY_THRESHOLD", "MAYRING_EMBED_REPLICATION",
                "MAYRING_EMBED_TRUST_MIN_DEVICES", "MAYRING_EMBED_QUARANTINE_SECONDS",
                "MAYRING_EMBED_HEARTBEAT_FRESH_SECONDS", "MAYRING_EMBED_DUAL_CLAIM_TIMEOUT_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    importlib.reload(cfg)
    assert cfg.EMBED_VERIFY_THRESHOLD == 0.9999
    assert cfg.EMBED_REPLICATION == 2
    assert cfg.EMBED_TRUST_MIN_DEVICES == 20
    assert cfg.EMBED_QUARANTINE_SECONDS == 3600
    assert cfg.EMBED_HEARTBEAT_FRESH_SECONDS == 120
    assert cfg.EMBED_DUAL_CLAIM_TIMEOUT_SECONDS == 300


def test_env_override(monkeypatch):
    monkeypatch.setenv("MAYRING_EMBED_VERIFY_THRESHOLD", "0.95")
    monkeypatch.setenv("MAYRING_EMBED_TRUST_MIN_DEVICES", "5")
    importlib.reload(cfg)
    assert cfg.EMBED_VERIFY_THRESHOLD == 0.95
    assert cfg.EMBED_TRUST_MIN_DEVICES == 5
