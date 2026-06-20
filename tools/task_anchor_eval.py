"""Measure the two redesign hypotheses on a curated TASK eval fixture, locally,
with real bge-m3 retrieval — no Chroma, no deploy.

Hypotheses:
  H1 (task anchor): a DISTILLED TASK query retrieves the relevant chunks better
     than the RAW PROMPT (which drags conversational junk up).
  H2 (question-decomposition loop): fanning the task into sub-questions and
     looping recalls more of the relevant set than a single-shot query.

Metric: Recall@k of each task's relevant chunks, for three strategies:
  A) raw prompt → top-k          (status quo)
  B) derived task → top-k        (H1)
  C) task + question loop → set  (H1+H2)

The fixture is an EVAL FIXTURE (a yardstick), not a training corpus: the tasks
are real MayringCoder work items, the relevant snippets mirror real chunk content,
the distractors are real conversational-junk patterns (snapshots/smoke). Honest
A/B of retrieval quality, not a claim about production data volume.

Run: OLLAMA_URL=http://localhost:11434 python tools/task_anchor_eval.py
"""
from __future__ import annotations

import os

import httpx

try:
    from tools.sufficiency_gate import derive_task, run_task_loop, decompose_questions
except ImportError:
    from sufficiency_gate import derive_task, run_task_loop, decompose_questions

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = "bge-m3"

# --- curated fixture: real tasks, MULTI-FACET relevant snippets (scattered so a
# single query can't catch them all → the loop's question-decomposition has room),
# a realistic RECAP per task (the clean task statement the compact-summary holds),
# and real junk distractors. ---
TASKS = [
    {
        "raw": "JAAAA mach den reranker endlich, aber WANN wird das aktiv gesetzt???",
        "recap": "# Session Recap. Aktuelle Aufgabe: Reranker-Aktivierung absichern — eine Version darf nur aktiv werden, wenn sie auf der clean-eval ueber der v1-Baseline liegt; das Setzen erfolgt ueber einen Admin-Endpoint und persistiert in einer Datei.",
        "relevant": [
            "POST /stats/admin/reranker-active mit {versions:[v3]} setzt die aktive Reranker-Version; write_active_versions persistiert nach rerank_active.json.",
            "Das Qualitaets-Gate _assert_active_quality lehnt eine Version ab, deren clean-eval-nDCG unter der v1-Baseline liegt (HTTP 422), ausser force=true.",
            "Der Delete-Guard schuetzt jede serving-aktive Version; rerank_default.txt wird beim Schreiben mit der primaeren Version synchron gehalten.",
        ],
    },
    {
        "raw": "ok und das gemma ding das die chunks prueft, wie hoert das auf zu loopen?",
        "recap": "# Session Recap. Aufgabe: Mythos-Sufficiency-Gate — ein kleines Modell prueft iterativ ob die Chunks reichen; der Loop muss garantiert terminieren ueber mehrere Halt-Kriterien und darf den Hot-Path nicht belasten.",
        "relevant": [
            "Der Sufficiency-Loop haelt auf dem ersten von: sufficient, no_requery, no_progress, cap, budget — er kann nie haengen.",
            "run_task_loop zerlegt den Task in Sub-Fragen und haelt, wenn alle Fragen beantwortet sind oder keine neuen Chunks kommen.",
            "Das Gate sitzt auf dem Act-Pfad, nie im per-Prompt-Inject (9s-Budget, VRAM-Thrash) — gemma laeuft cloud oder ersetzt qwen im Gate.",
        ],
    },
    {
        "raw": "warum kommt beim suchen immer so ein gesabbel hoch statt dem was ich brauche",
        "recap": "# Session Recap. Befund: Goal/Task-Anker fehlt im Retrieval; die Such-Query ist der rohe Prompt. Aufgabe: Prompt zu Task destillieren und als Suchanker nutzen, Goal als Scope.",
        "relevant": [
            "Der Goal/Task-Anker fehlt im Retrieval: die Such-Query ist der rohe Prompt, daher matchen Gespraechs-Chunks flach und werden hochgespuelt.",
            "Prompt-zu-Task-Destillation (derive_task) extrahiert die sachliche Arbeitseinheit aus dem rohen Prompt und nutzt sie als Suchanker.",
            "tasks.status ist abschliessbar (done), goals nicht — der semantische Loop-Halt ist nur gegen einen Task wohldefiniert, das Goal bleibt Scope-Parent.",
        ],
    },
    {
        "raw": "die vram sache auf dem gpu host, wie war das nochmal mit den modellen",
        "recap": "# Session Recap. Infrastruktur-Invariante: lokaler Hot-Path auf zwei Modelle begrenzt wegen VRAM; GPU-Host-Parallelitaet ueber systemd-Variablen gesteuert.",
        "relevant": [
            "Der lokale Hot-Path ist bewusst auf zwei Modelle gepinnt (bge-m3 + qwen3.5-mayring:2b), damit sie zusammen in den VRAM passen.",
            "OLLAMA_NUM_PARALLEL und MAX_LOADED_MODELS auf dem GPU-Host steuern, wie viele Modelle gleichzeitig resident bleiben.",
            "Ein drittes lokales Modell (gemma4:e4b) wuerde thrashen — daher cloud-routen oder ein bestehendes ersetzen statt zusaetzlich laden.",
        ],
    },
    {
        "raw": "der span judge das training, das war doch der mit dem schwachen lehrer oder",
        "recap": "# Session Recap. Aufgabe: Reranker-Training analysieren — span_judge nutzte einen zu schwachen Teacher; Cache-only-Modus mit starkem Teacher als Fix.",
        "relevant": [
            "span_judge nutzte ministral-3:3b als Cloud-Teacher; ein zu schwacher Teacher vergiftet die Relevanz-Labels und invertiert das v-Gewicht.",
            "SPAN_JUDGE_CACHE_ONLY laesst den Export nur vorgewaermte Claude-Labels nutzen, kein frischer ministral-Call.",
            "Distillation taugt nur mit starkem Teacher; schwache Features deckeln den gelernten Reranker bei der Vektor-Baseline.",
        ],
    },
]

# real conversational-junk patterns (the chunks that get wrongly surfaced today)
DISTRACTORS = [
    "# Session 2026-04-17 | tmp Thema: Smoke Test in Claude-Code-Projekt. Entscheidungen: Ausfuehrung eines Rauchverfahrens zur Validierung der Grundfunktionalitaet.",
    "Aktueller Stand: In der aktuellen Version von Unserem Wiki sind keine neuesten Implementierungen enthalten. Architektur-Hotspots: Startseite, Menue, Unterseiten.",
    "Okay, hier ist ein Projekt-Snapshot basierend auf den gegebenen Informationen (derzeit nahezu leer): Das Projekt befindet sich im Startstadium.",
    "Antworte NUR mit JSON: {aktuell:..., hotspots:[...], offene_punkte:[...]} kein Markdown, keine Prosa. <think> </think>",
    "# Session unbekannt | mayringcoder Thema: Smoke Test der Claude-Code-Implementierung. Ergebnisse: Durchfuehrung erfolgreich, keine Fehler.",
    "using System.Collections; namespace BattlefieldCampaign.Battle { One ship in the hexless realtime battle, free steering toward the nearest enemy.",
    "Konz, Britta; Rohde-Abuba, Caterina. Flucht und Religion. Religioese Verortungen von Kindern mit Fluchterfahrungen. Bad Heilbrunn 2022.",
    "EmbeddingToVoxel.cs DATAMODEL-konformer Algorithmus: Embedding zu 8x8x12 Voxel Grid, deterministisch, gleiche Embeddings gleiche Voxels.",
    "# app.linn.games Sync fuer MayringCoder Commits. Implementiert die Server-Seite der Spec llm-endpoints-implementation auf der Laravel-Seite.",
    "Job 6: End-to-End-Tests Chat plus Worker plus RAG validieren. Parent Issue 156. Redis-Cache fuer Session-Protokolle.",
]


def embed(texts: list[str]) -> list[list[float]]:
    r = httpx.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


def _cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


def main() -> int:
    # build corpus: all relevant chunks (tagged by task) + shared distractors
    corpus: list[dict] = []
    for ti, t in enumerate(TASKS):
        for j, txt in enumerate(t["relevant"]):
            corpus.append({"chunk_id": f"t{ti}_r{j}", "text": txt, "task": ti})
    for j, txt in enumerate(DISTRACTORS):
        corpus.append({"chunk_id": f"d{j}", "text": txt, "task": -1})

    print(f"corpus: {len(corpus)} chunks ({sum(1 for c in corpus if c['task']>=0)} relevant, "
          f"{sum(1 for c in corpus if c['task']<0)} distractors)")
    vecs = embed([c["text"] for c in corpus])
    for c, v in zip(corpus, vecs):
        c["vec"] = v

    def top_k(query_vec, k):
        scored = sorted(corpus, key=lambda c: _cos(query_vec, c["vec"]), reverse=True)
        return scored[:k]

    def retrieve_fn(query: str, k: int = 3) -> list[dict]:
        qv = embed([query])[0]
        return [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in top_k(qv, k)]

    K = 3  # single-query budget; loop gets the same k per sub-question
    recall = {"A_raw": [], "B_task_prompt": [], "C_task_recap": [], "D_recap_loop": []}
    for ti, t in enumerate(TASKS):
        gold = {f"t{ti}_r{j}" for j in range(len(t["relevant"]))}
        n_gold = len(gold)

        # A: raw prompt query (status quo)
        a_hits = {c["chunk_id"] for c in top_k(embed([t["raw"]])[0], K)} & gold
        recall["A_raw"].append(len(a_hits) / n_gold)

        # B: task distilled from the RAW PROMPT
        task_p = derive_task(t["raw"], OLLAMA)
        b_hits = {c["chunk_id"] for c in top_k(embed([task_p])[0], K)} & gold
        recall["B_task_prompt"].append(len(b_hits) / n_gold)

        # C: task distilled from the RECAP (user's insight — the clean task statement)
        task_r = derive_task(t["recap"], OLLAMA)
        c_hits = {c["chunk_id"] for c in top_k(embed([task_r])[0], K)} & gold
        recall["C_task_recap"].append(len(c_hits) / n_gold)

        # D: recap-task + question-decomposition loop (fan-out over sub-questions)
        loop = run_task_loop(task_r, lambda q: retrieve_fn(q, K), OLLAMA, max_loops=2, max_q=4)
        d_hits = {c["chunk_id"] for c in loop["final_chunks"]} & gold
        recall["D_recap_loop"].append(len(d_hits) / n_gold)

        print(f"  task#{ti}: gold={n_gold} | A={len(a_hits)} B={len(b_hits)} "
              f"C={len(c_hits)} D={len(d_hits)} | halt={loop['halted_by']} qs={len(loop['questions'])}")
        print(f"           recap-task='{task_r[:70]}'")

    print("\n=== Recall@%d (mean over %d tasks) ===" % (K, len(TASKS)))
    print("  A_raw          (status quo, roher Prompt) :", f"{sum(recall['A_raw'])/len(TASKS):.3f}")
    print("  B_task_prompt  (Task aus Prompt, H1)      :", f"{sum(recall['B_task_prompt'])/len(TASKS):.3f}")
    print("  C_task_recap   (Task aus Recap, H1+Recap) :", f"{sum(recall['C_task_recap'])/len(TASKS):.3f}")
    print("  D_recap_loop   (Recap-Task + Loop, H2)    :", f"{sum(recall['D_recap_loop'])/len(TASKS):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
