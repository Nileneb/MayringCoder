#!/usr/bin/env python3
"""Retrieval-Benchmark für die Memory-Pipeline.

Lädt Test-Queries aus einer YAML-Datei, führt Hybrid-Search durch und berechnet
MRR (Mean Reciprocal Rank) sowie Recall@K.

Verwendung:
    .venv/bin/python src/benchmark_retrieval.py --queries benchmarks/retrieval_queries.yaml
    .venv/bin/python src/benchmark_retrieval.py --queries benchmarks/retrieval_queries.yaml --top-k 10
    .venv/bin/python src/benchmark_retrieval.py --queries benchmarks/retrieval_queries.yaml --repo myorg/myrepo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow running as `python src/benchmark_retrieval.py` from project root
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Metriken-Funktionen (rein, gut testbar)
# ---------------------------------------------------------------------------


def mrr(ranked_paths: list[list[str]], relevant_paths: list[list[str]]) -> float:
    """Mean Reciprocal Rank über alle Queries.

    Args:
        ranked_paths: Für jede Query — geordnete Liste der zurückgegebenen Pfade.
        relevant_paths: Für jede Query — Menge der relevanten Pfade (Ground Truth).

    Returns:
        MRR-Score zwischen 0.0 und 1.0.
    """
    if not ranked_paths:
        return 0.0
    reciprocal_ranks: list[float] = []
    for ranked, relevant in zip(ranked_paths, relevant_paths):
        relevant_set = set(relevant)
        rr = 0.0
        for rank, path in enumerate(ranked, start=1):
            if path in relevant_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def recall_at_k(
    ranked_paths: list[list[str]],
    relevant_paths: list[list[str]],
    k: int = 5,
) -> float:
    """Recall@K über alle Queries: Anteil relevanter Items in den Top-K Ergebnissen."""
    if not ranked_paths:
        return 0.0
    recalls: list[float] = []
    for ranked, relevant in zip(ranked_paths, relevant_paths):
        relevant_set = set(relevant)
        if not relevant_set:
            recalls.append(1.0)
            continue
        top_k = set(ranked[:k])
        hit_count = len(top_k & relevant_set)
        recalls.append(hit_count / len(relevant_set))
    return sum(recalls) / len(recalls)


def category_precision_at_k(
    records_per_query: list[list[Any]],
    expected_categories: list[str | None],
    k: int = 5,
) -> float:
    """Category Precision@K: Anteil der Queries, bei denen mind. ein Top-K-Ergebnis
    die erwartete Kategorie enthält.

    Queries ohne expected_category werden übersprungen (nicht mitgezählt).
    Returns 0.0 if no queries have expected_category.
    """
    hits = 0
    total = 0
    for records, expected in zip(records_per_query, expected_categories):
        if not expected:
            continue
        total += 1
        expected_lower = expected.lower()
        for rec in records[:k]:
            labels = [lbl.lower() for lbl in (getattr(rec, "category_labels", []) or [])]
            if any(expected_lower in lbl or lbl in expected_lower for lbl in labels):
                hits += 1
                break
    return hits / total if total else 0.0


def uncategorized_rate(conn: Any) -> float:
    """Anteil der aktiven Chunks ohne Kategorie-Labels.

    Returns 1.0 if no chunks exist.
    """
    try:
        row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN category_labels = '' OR category_labels IS NULL THEN 1 ELSE 0 END) "
            "FROM chunks WHERE is_active = 1"
        ).fetchone()
        total, uncategorized = row
        if not total:
            return 1.0
        return (uncategorized or 0) / total
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Query-Loader
# ---------------------------------------------------------------------------


def load_queries(yaml_path: str | Path) -> list[dict]:
    """Lädt Benchmark-Queries aus einer YAML-Datei."""
    try:
        import yaml  # type: ignore
    except ImportError:
        print("PyYAML fehlt. Installieren: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("queries", [])


# ---------------------------------------------------------------------------
# Benchmark-Runner
# ---------------------------------------------------------------------------


def run_benchmark(
    queries: list[dict],
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    top_k: int = 5,
    repo: str | None = None,
) -> dict:
    """Führt alle Queries gegen die Memory-Pipeline aus und berechnet Metriken.

    Returns:
        {mrr, recall_at_1, recall_at_k, category_precision_at_k, uncategorized_rate, k, results}
    """
    from src.memory_retrieval import search

    all_ranked: list[list[str]] = []
    all_relevant: list[list[str]] = []
    all_records: list[list[Any]] = []
    all_expected_cats: list[str | None] = []
    query_results: list[dict] = []

    for q in queries:
        query_text = q.get("query", "")
        relevant = q.get("relevant_paths", [])
        view = q.get("view")
        expected_cat = q.get("expected_category")

        opts: dict = {"top_k": top_k}
        if repo:
            opts["repo"] = repo
        if view:
            opts["source_type"] = None

        records: list[Any] = []
        normalized: list[str] = []
        try:
            records = search(query_text, conn, chroma_collection, ollama_url, opts=opts)
            for r in records:
                parts = r.source_id.split(":")
                for part in parts:
                    if part.startswith("issue/"):
                        normalized.append(part)
                        break
                else:
                    normalized.append(r.source_id)

            if view:
                view_level = f"view_{view}"
                filtered = [
                    p for r, p in zip(records, normalized)
                    if view_level in (getattr(r, "chunk_level", "") or "")
                ]
                if filtered:
                    normalized = filtered

        except Exception as exc:
            print(f"  [WARN] Query {q.get('id', '?')} fehlgeschlagen: {exc}")

        all_ranked.append(normalized)
        all_relevant.append(relevant)
        all_records.append(records)
        all_expected_cats.append(expected_cat)

        first_hit = next((p for p in normalized if p in set(relevant)), None)
        query_results.append({
            "id": q.get("id", "?"),
            "query": query_text[:80],
            "top_results": normalized[:3],
            "relevant": relevant,
            "first_hit": first_hit,
            "hit": first_hit is not None,
            "expected_category": expected_cat,
        })

    return {
        "mrr": mrr(all_ranked, all_relevant),
        "recall_at_1": recall_at_k(all_ranked, all_relevant, k=1),
        "recall_at_k": recall_at_k(all_ranked, all_relevant, k=top_k),
        "category_precision_at_k": category_precision_at_k(all_records, all_expected_cats, k=top_k),
        "uncategorized_rate": uncategorized_rate(conn),
        "k": top_k,
        "results": query_results,
    }


# ---------------------------------------------------------------------------
# Report-Ausgabe
# ---------------------------------------------------------------------------


def print_report(benchmark: dict) -> None:
    """Gibt den Benchmark-Report auf stdout aus."""
    results = benchmark["results"]
    k = benchmark["k"]

    print(f"\n{'='*70}")
    print(f"  Retrieval-Benchmark — {len(results)} Queries")
    print(f"{'='*70}")

    for r in results:
        hit_marker = "✓" if r["hit"] else "✗"
        print(f"\n[{hit_marker}] {r['id']}: {r['query']}")
        print(f"    Relevant:    {', '.join(r['relevant']) or '(keine)'}")
        print(f"    Top-3:       {', '.join(r['top_results']) or '(keine Ergebnisse)'}")
        if r["first_hit"]:
            print(f"    Erster Hit:  {r['first_hit']}")

    print(f"\n{'─'*70}")
    print(f"  MRR:                    {benchmark['mrr']:.4f}")
    print(f"  Recall@1:               {benchmark['recall_at_1']:.4f}")
    print(f"  Recall@{k}:               {benchmark['recall_at_k']:.4f}")
    cat_prec = benchmark.get("category_precision_at_k")
    if cat_prec is not None:
        print(f"  Category Precision@{k}:   {cat_prec:.4f}")
    unc = benchmark.get("uncategorized_rate")
    if unc is not None:
        print(f"  Uncategorized Rate:     {unc:.1%}")
    hits = sum(1 for r in results if r["hit"])
    print(f"  Hits:                   {hits}/{len(results)}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-Benchmark für MayringCoder Memory")
    parser.add_argument(
        "--queries",
        default="benchmarks/retrieval_queries.yaml",
        help="Pfad zur Queries-YAML-Datei (Standard: benchmarks/retrieval_queries.yaml)",
    )
    parser.add_argument("--top-k", type=int, default=5, metavar="K",
                        help="Anzahl zurückgegebener Ergebnisse pro Query (Standard: 5)")
    parser.add_argument("--repo", default=None, metavar="REPO",
                        help="Optional: nur Ergebnisse aus diesem Repo filtern")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama URL (Standard: http://localhost:11434)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    from src.memory_store import init_memory_db
    from src.memory_ingest import get_or_create_chroma_collection

    print(f"Lade Queries aus: {args.queries}")
    queries = load_queries(args.queries)
    print(f"{len(queries)} Queries geladen")

    print("Initialisiere Memory-DB und ChromaDB ...")
    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()

    print(f"Starte Benchmark (top_k={args.top_k}) ...\n")
    benchmark = run_benchmark(
        queries, conn, chroma, args.ollama_url,
        top_k=args.top_k,
        repo=args.repo,
    )
    conn.close()

    print_report(benchmark)


if __name__ == "__main__":
    main()
