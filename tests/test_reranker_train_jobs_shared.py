"""Train-job state must be visible across uvicorn workers (multi-worker bug
2026-05-28): the job was created in one worker's in-memory _TRAIN_JOBS and a
status GET routed elsewhere returned 'not found'. Now persisted to a shared
file (atomic, merge-on-save)."""
from __future__ import annotations


def test_train_job_state_cross_worker_via_shared_file(tmp_path, monkeypatch):
    from src.api.routes import reranker_admin as ra
    monkeypatch.setattr(ra, "_TRAIN_JOBS_FILE", tmp_path / "tj.json")

    # Worker A creates + persists a job
    monkeypatch.setattr(ra, "_TRAIN_JOBS", {"train-1": {"status": "running", "days": 30}})
    ra._save_train_job("train-1")

    # Worker B (fresh in-memory) sees it only via the shared file
    assert ra._load_train_jobs().get("train-1") == {"status": "running", "days": 30}


def test_save_does_not_clobber_other_workers_jobs(tmp_path, monkeypatch):
    from src.api.routes import reranker_admin as ra
    monkeypatch.setattr(ra, "_TRAIN_JOBS_FILE", tmp_path / "tj.json")

    monkeypatch.setattr(ra, "_TRAIN_JOBS", {"train-A": {"status": "done"}})
    ra._save_train_job("train-A")
    # A different worker, unaware of train-A, persists its own job
    monkeypatch.setattr(ra, "_TRAIN_JOBS", {"train-B": {"status": "running"}})
    ra._save_train_job("train-B")

    shared = ra._load_train_jobs()
    assert "train-A" in shared and "train-B" in shared  # merge-on-save, no clobber


def test_corrupt_state_file_is_non_fatal(tmp_path, monkeypatch):
    from src.api.routes import reranker_admin as ra
    f = tmp_path / "tj.json"
    f.write_text("{ not json")
    monkeypatch.setattr(ra, "_TRAIN_JOBS_FILE", f)
    assert ra._load_train_jobs() == {}  # corrupt → empty, doesn't raise
