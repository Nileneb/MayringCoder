"""#266: the externalized mayring-pi-agent package must expose the exact
symbols MayringCoder consumes in-process, so the src/agents → package swap
is a 1:1 replacement (no behavioural loss)."""


def test_package_exposes_consumed_symbols():
    from mayring_pi_agent.pi import run_task_with_memory, analyze_with_memory
    from mayring_pi_agent.pi_queue import get_pi_queue
    from mayring_pi_agent.pi_jobs import PiJob, classify_pi_job
    from mayring_pi_agent.vision import caption_image, get_image_metadata
    from mayring_pi_agent.diff_history import DiffHistoryError, run
    from mayring_pi_agent import pi_worker, pi_server
    assert callable(run_task_with_memory)
    assert callable(analyze_with_memory)
    assert callable(get_pi_queue)
    assert callable(classify_pi_job)
    assert callable(pi_worker.start)
    assert hasattr(pi_server, "app")
