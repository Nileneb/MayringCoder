"""Pi-Agent-Freiform-Task Workflow."""
from __future__ import annotations

from src.config import repo_slug as _repo_slug
from src.model_router import ModelRouter


def run_pi_task(args, ollama_url: str, model: str, router: ModelRouter | None = None) -> None:
    if router is not None and not model:
        if router.is_available("analysis"):
            model = router.resolve("analysis")

    from src.agents.pi import run_task_with_memory

    task = args.pi_task
    repo_url = getattr(args, "repo", None)
    repo_slug_val = _repo_slug(repo_url) if repo_url else None

    print(f"[pi-task] Auftrag: {task[:80]}{'...' if len(task) > 80 else ''}")
    if repo_slug_val:
        print(f"[pi-task] Memory-Scope: {repo_slug_val}")
    print()

    result = run_task_with_memory(
        task=task,
        ollama_url=ollama_url,
        model=model,
        repo_slug=repo_slug_val,
    )

    print(result)
