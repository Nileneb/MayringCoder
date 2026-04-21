"""Image-Ingest Workflow — Vision-Captioning + Memory-Ingest von Bildern."""
from __future__ import annotations

from src.model_router import ModelRouter


def run_ingest_images(args, ollama_url: str, model: str, router: ModelRouter | None = None) -> None:
    if router is not None and not model:
        if router.is_available("vision"):
            model = router.resolve("vision")

    from src.memory.ingest import run_image_ingest

    repo_url = args.ingest_images
    vision_model = getattr(args, "vision_model", "qwen2.5vl:3b")
    max_images = getattr(args, "max_images", 50)
    do_force = getattr(args, "force_reingest", False)

    print(f"[ingest-images] Starte Bild-Ingest für: {repo_url}")
    print(f"[ingest-images] Vision-Modell: {vision_model}, Max-Bilder: {max_images}")

    result = run_image_ingest(
        repo_url=repo_url,
        ollama_url=ollama_url,
        vision_model=vision_model,
        embed_model=model,
        max_images=max_images,
        force_reingest=do_force,
        workspace_id=getattr(args, "workspace_id", "default"),
    )

    print(
        f"\n[ingest-images] Fertig: {result['images_found']} Bilder total, "
        f"{result['images_captioned']} captioniert, "
        f"{result['images_skipped']} Dedup, "
        f"{result['images_failed']} Fehler."
    )
