"""CLI argument definitions for src.cli — kept separate to stay under the 20k-char analysis limit."""
from __future__ import annotations

import argparse

from src.config import EMBEDDING_MODEL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RepoChecker — lokale Code-Analyse mit Ollama")
    p.add_argument("--repo", help="GitHub-Repo URL (überschreibt .env)")
    p.add_argument("--model", help="Ollama-Modell (überschreibt .env)")
    p.add_argument("--full", action="store_true",
                   help="Full-Scan: Cache ignorieren, kein Datei-Limit (impliziert --no-limit)")
    p.add_argument("--dry-run", action="store_true", help="Nur Diff + Selektion zeigen, keine Analyse")
    p.add_argument("--show-selection", action="store_true", help="Zeigt ausgewählte Dateien inkl. Kategorie")
    p.add_argument("--prompt", help="Pfad zu einem alternativen Prompt")
    p.add_argument("--debug", action="store_true", help="Speichert Raw-Snapshot lokal unter cache/")
    p.add_argument("--reset", action="store_true", help="Cache-DB für das Repo löschen (alle Analysen zurücksetzen)")
    p.add_argument("--mode", choices=["analyze", "overview", "turbulence"], default="analyze",
                   help="Modus: 'overview' = Funktions-Übersicht, 'analyze' = Fehlersuche (Standard), "
                        "'turbulence' = Hot-Zone-Analyse (vermischte Verantwortlichkeiten)")
    p.add_argument("--llm", action="store_true",
                   help="Turbulenz-Modus: LLM für Chunk-Kategorisierung nutzen (langsamer, genauer). "
                        "Standard: Heuristik (kein Ollama nötig).")
    p.add_argument("--no-limit", action="store_true", help="Kein Datei-Limit pro Lauf (alle Dateien verarbeiten)")
    p.add_argument("--max-chars", type=int, metavar="N", help="Zeichenlimit pro Datei überschreiben (Kontextlimit wird automatisch angepasst)")
    p.add_argument("--budget", type=int, metavar="N", help="Datei-Limit pro Lauf überschreiben (Standard: 20)")
    p.add_argument("--log-training-data", action="store_true",
                   help="Jeden LLM-Call (Prompt + Antwort) in ein JSONL-Logfile schreiben. "
                        "Speicherort: cache/<slug>_training_log.jsonl.")
    p.add_argument("--time-budget", type=float, metavar="SECONDS",
                   help="Maximale Laufzeit in Sekunden. Nach Ablauf wird graceful gestoppt.")
    p.add_argument("--batch-size", type=int, metavar="N",
                   help="GPU-Pause alle N Dateien (0 = keine Pause, Standard: BATCH_SIZE aus config.py)")
    p.add_argument("--batch-delay", type=float, metavar="S",
                   help="Pausendauer in Sekunden (Standard: BATCH_DELAY_SECONDS aus config.py)")
    p.add_argument("--run-id", help="Logischer Run-Key für Cache + Report (ermöglicht Modell-/Run-Vergleiche)")
    p.add_argument("--cache-by-model", action="store_true", help="Modellnamen als Cache-Key verwenden (wenn kein --run-id gesetzt ist)")
    p.add_argument("--codebook", help="Pfad zu einem alternativen Codebook (YAML)")
    p.add_argument("--codebook-profile", metavar="PROFILE",
                   help="Codebook-Profil aus codebooks/profiles/ laden (z.B. laravel, python). "
                        "Überschreibt --codebook wenn gesetzt. Auto-Detection wenn nicht angegeben.")
    p.add_argument("--export", metavar="DATEI", help="Ergebnisse exportieren (.csv oder .json)")
    p.add_argument("--history", action="store_true", help="Vergangene Runs anzeigen")
    p.add_argument("--compare", nargs=2, metavar="RUN_ID", help="Zwei Runs vergleichen (alt neu)")
    p.add_argument("--cleanup", type=int, metavar="N", help="Nur die N neuesten Runs behalten, Rest löschen")
    p.add_argument("--resolve-model-only", action="store_true",
                   help="Gibt nur den aufgelösten Modellnamen aus und beendet (für Shell-Skripting)")
    p.add_argument("--min-confidence", choices=["high", "medium", "low"], default="low",
                   help="Minimale Confidence-Schwelle für Findings (Standard: low).")
    p.add_argument("--adversarial", action="store_true",
                   help="Jedes Finding wird durch einen zweiten LLM-Call (Advocatus Diaboli) geprüft.")
    p.add_argument("--adversarial-cost-report", action="store_true",
                   help="Zeigt nach der Analyse: wie viele Findings BESTÄTIGT vs. ABGELEHNT wurden.")
    p.add_argument("--second-opinion", metavar="MODEL", default=None,
                   help="Zweites Modell für unabhängige Validierung. "
                        "Überschreibt die Umgebungsvariable SECOND_OPINION_MODEL.")
    p.add_argument("--embedding-prefilter", action="store_true",
                   help="Aktiviert den Embedding-Vorfilter: Dateien werden anhand semantischer "
                        "Ähnlichkeit zur Forschungsfrage vorselektiert.")
    p.add_argument("--embedding-model", default=None, metavar="MODEL",
                   help=f"Ollama-Embedding-Modell für den Vorfilter (Standard: {EMBEDDING_MODEL})")
    p.add_argument("--embedding-top-k", type=int, default=20, metavar="N",
                   help="Maximale Anzahl Dateien nach Embedding-Vorfilter (Standard: 20, 0 = kein Limit)")
    p.add_argument("--embedding-threshold", type=float, default=None, metavar="F",
                   help="Minimale Kosinus-Ähnlichkeit für Embedding-Vorfilter (z. B. 0.3)")
    p.add_argument("--embedding-query", default=None, metavar="TEXT",
                   help="Forschungsfrage / Suchbegriffe für den Embedding-Vorfilter.")
    p.add_argument("--use-overview-cache", action="store_true",
                   help="Turbulence-Modus: Kategorien aus Overview-Cache übernehmen.")
    p.add_argument("--use-turbulence-cache", action="store_true",
                   help="Analyze-Modus: Hot-Zone-Kontext aus Turbulence-Cache laden.")
    p.add_argument("--rag-enrichment", action="store_true",
                   help="Finding-reaktive RAG-Queries: Jedes Finding bekommt semantisch passenden Projektkontext.")
    p.add_argument("--populate-memory", action="store_true",
                   help="Repo laden und alle Dateien in die Memory-Pipeline ingesten.")
    p.add_argument("--workers", type=int, default=1, metavar="N",
                   help="Parallele Ingest-Worker für populate-memory (Standard: 1).")
    p.add_argument("--no-memory-categorize", dest="memory_categorize", action="store_false",
                   help="Mayring-Kategorisierung während Memory-Ingest deaktivieren (Default ist an).")
    p.add_argument("--memory-categorize", dest="memory_categorize", action="store_true",
                   help=argparse.SUPPRESS)  # BC
    p.set_defaults(memory_categorize=True)
    p.add_argument("--generate-wiki", action="store_true",
                   help="Verknüpfungswiki aus Overview-Cache + Memory erzeugen (cache/<slug>_wiki.md)")
    p.add_argument("--rebuild-transitions", action="store_true",
                   help="Scan conversation-summaries + rebuild Markov topic-transition matrix")
    p.add_argument("--wiki-type", choices=["code", "paper"], default="code",
                   help="Wiki-Modus: code (Import/Call-Graph) oder paper (Paper-Verknüpfungen)")
    p.add_argument("--wiki-cluster-strategy", choices=["louvain", "full"], default="louvain",
                   help="Wiki 2.0 Clustering-Strategie: louvain (schnell) oder full (Louvain+Embedding+LLM)")
    p.add_argument("--wiki-second-opinion", metavar="MODEL", default=None,
                   help="Zweites Modell zur Validierung von Cluster-Zuordnungen und concept_link-Edges.")
    p.add_argument("--wiki-history", action="store_true",
                   help="Zeigt die letzten 20 Wiki-Snapshots für den aktuellen Workspace.")
    p.add_argument("--wiki-team-activity", action="store_true",
                   help="Zeigt Team-Contribution-Übersicht (letzte 30 Tage).")
    p.add_argument("--wiki-history-cleanup", type=int, metavar="N", default=None,
                   help="Behält nur die N neuesten Wiki-Snapshots, löscht ältere.")
    p.add_argument("--generate-ambient", action="store_true",
                   help="Ambient-Snapshot regenerieren (cache via SQLite, model required)")
    p.add_argument("--ingest-issues", metavar="REPO",
                   help="GitHub Issues von REPO (owner/name) in Memory laden (benötigt gh CLI).")
    p.add_argument("--issues-state", choices=["open", "closed", "all"], default="open",
                   help="Welche Issues laden (Standard: open)")
    p.add_argument("--issues-limit", type=int, default=100, metavar="N",
                   help="Maximale Anzahl Issues (Standard: 100)")
    p.add_argument("--multiview", action="store_true",
                   help="Multi-view Indexing für Issues: LLM extrahiert Fact/Impl/Decision/Entities-Sichten")
    p.add_argument("--force-reingest", action="store_true",
                   help="Bestehende Chunks invalidieren und neu ingesten (ignoriert Dedup-Schutz)")
    p.add_argument("--ingest-images", metavar="REPO_URL",
                   help="Repo-Bilder (PNG/JPG/SVG) captionieren und in Memory ingesten.")
    p.add_argument("--vision-model", default="qwen2.5vl:3b", metavar="MODEL",
                   help="Ollama Vision-Modell für Bild-Captioning (Standard: qwen2.5vl:3b)")
    p.add_argument("--max-images", type=int, default=50, metavar="N",
                   help="Maximale Anzahl Bilder pro Ingest-Lauf (Standard: 50)")
    p.add_argument("--gpu-metrics", action="store_true",
                   help="GPU-Metriken via nvidia-smi erfassen (VRAM, Auslastung, Watt, Temp).")
    p.add_argument("--no-pi", action="store_true",
                   help="Pi-Agent deaktivieren (Standard: an).")
    p.add_argument("--no-wiki-inject", action="store_true",
                   help="Wiki-Kontext-Injektion deaktivieren (für A/B-Tests).")
    p.add_argument("--pi-task", metavar="TASK",
                   help="Freier Auftrag an Pi mit Memory-Zugriff.")
    p.add_argument("--workspace-id", default="default", metavar="ID",
                   help="Tenant workspace für Multi-Tenancy (Standard: 'default').")
    p.add_argument("--generate-training-data", choices=["memory", "kategorie"], metavar="PIPELINE",
                   help="Training-Daten generieren. 'memory' = Memory-Context-Injection, "
                        "'kategorie' = Kategorie-Coaching (Issue #87). "
                        "Outputs: cache/finetuning/*.jsonl")
    p.add_argument("--skip-auto-feedback", action="store_true",
                   help="--generate-training-data memory: Feedback-Schreiben überspringen (read-only)")
    p.add_argument("--training-limit", type=int, default=500, metavar="N",
                   help="--generate-training-data memory: max. Quell-Dateien verarbeiten (Standard: 500)")
    return p.parse_args()
