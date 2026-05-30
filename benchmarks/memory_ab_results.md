# Memory A/B — ehrliche Messung (2026-05-30)

**Frage:** Macht das Cloud-Memory-System den Agenten besser bei Tasks?
**Methode:** `benchmarks/memory_ab_eval.py` — gleiches Modell (qwen3.5:9b), jeder Task
aus `task_suite.yaml` einmal mit Memory (ambient + `search_memory`, cloud-first) und
einmal `disable_memory=True`. Score = objektive Keyword-Treffer (kein LLM-Judge → keine
Judge-Bias). 1 Lauf/Arm (stochastisch — Caveat).

## Ergebnis

| Split | Memory ON | Memory OFF | Δ | n |
|---|---|---|---|---|
| **requires_memory = true** (context_injection) | 0.35 | 0.10 | **+0.25** | 2 |
| **requires_memory = false** (pico/code_review/conv_summary) | 0.84 | 0.84 | **+0.00** | 5 |
| **OVERALL** | 0.70 | 0.63 | +0.07 | 7 |

Task-wins: ON=2, OFF=1, tie=4.

## Interpretation (ungeschönt)

- **Memory hilft genau dort, wo es soll** (+0.25 auf Tasks, die gespeichertes Wissen
  brauchen). Bei `context_injection_02` holte es per `search_memory` die Architektur-
  Fakten (2/4 vs 0/4) — memory-off kann diese Fragen prinzipiell nicht beantworten.
- **Auf memory-irrelevanten Tasks: exakt neutral (Δ=0.00).** Das ist der ehrliche,
  erwartete Befund.
- **Memory kann sogar schaden:** `code_review_01` — memory-on machte eine `search_memory`
  ('Laravel mass assignment'), das Ergebnis lenkte ab → 4/5 statt 5/5 (OFF gewann). Genau
  das „Rauschen statt Signal" auf self-contained Tasks, das ich vorhergesagt hatte.
- **Tool-Invocation ist unzuverlässig:** `context_injection_01` → unentschieden, weil das
  Modell `search_memory` GAR NICHT aufrief (0 Abfragen) trotz memory-on. Realer Limiter.

## Fazit zur Ausgangsfrage ("würde /goal Opus 4.8 auf SWE-bench schlagen?")

**Nein — und die Daten bestätigen meinen Einwand.** SWE-bench ist `requires_memory=false`-
Terrain (self-contained Single-Repo-Fixes). Dort ist Δ=0.00, gelegentlich negativ (Rauschen).
Der Memory-Hebel greift bei `requires_memory=true` — also bei Aufgaben mit nötigem Vorwissen,
genau dem, was SWE-bench wegschneidet. Das gebaute System ist ein verifiziertes Infra-Outcome,
KEIN Modell-Fähigkeits-Uplift auf einem unverwandten Benchmark.

**Caveats:** kleine N (7 Tasks), 1 Lauf/Arm (lokales Modell stochastisch), Keyword-Scoring
ist grob. Für härtere Zahlen: `--repeats 3-5`. Die qualitative Richtung (Hilfe nur bei
memory-Tasks, neutral/Rauschen sonst) ist robust, weil sie dem Mechanismus entspricht.

## Update: repeats=5 (partial, 4/7 task-groups, A/B gestoppt wg. Cloud-Saturation)

Mit Wiederholung schmilzt der Einzellauf-Effekt:
- requires_memory=true: ON≈0.27 OFF≈0.25 (Δ≈+0.03, verrauscht: context_injection_01 5× tie weil
  qwen search_memory NIE aufruft; context_injection_02 2× ON> / 2× OFF> / 1× tie).
- requires_memory=false: ON≈0.88 OFF≈0.96 (Δ≈**−0.08** — Memory SCHADET: pico_02 OFF gewann 3×, ON nie;
  die Suche lenkt ab + verbrennt einen Tool-Call).
- **Ehrliches Fazit:** für qwen3.5:9b netto neutral bis leicht negativ; der Einzellauf-"+0.25" war Glück.
- Lehre: Ambient war bei diesen Läufen tot (jetzt reaktiviert) → der Tool-Pull allein reicht nicht;
  Tool-Invocation-Unzuverlässigkeit ist der Haupt-Limiter beim lokalen Modell.
- NB: Cloud /memory/search saturierte unter A/B-Last (30s-Timeout → nach Stop 3,5s) — Lasttest-Nebenbefund.
