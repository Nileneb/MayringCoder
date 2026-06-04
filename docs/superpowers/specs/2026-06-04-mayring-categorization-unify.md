# Mayring-Kategorienbildung vereinheitlichen — Spec

**Stand:** 2026-06-04
**Status:** 📋 Approved (Design) — Umsetzung in Folge-Session
**Auslöser:** User-Auftrag „Kategorienbildung VEREINHEITLICHEN" — eine hybride,
universell einsetzbare Mayring-Codierung statt 4-fach duplizierter Ketten.
**Verbunden mit:** `docs/superpowers/specs/2026-05-24-phase3-mayring-process.md`,
`[[feedback-mayring-canonical-method]]`, Anti-Pattern #1 (v2-frustration-patterns).

---

## 1. Problem / Context

Die vom User beschriebene Methode —

> Thema (Goal/Task) + Text → themenbezogene **Paraphrase** (was ist die relevante
> Aussage neutral?) → **Generalisierung** (Einzelfall → allgemeine Aussage) →
> **Reduktion** (was lässt sich weglassen ohne die Kernbotschaft zu verlieren? —
> Kernbotschaft wird **am Thema** erkannt) → **Embedding-Vergleich** mit
> vorhandenen Kategorien (während/nach der Reduktion, gegen Chaos/Duplikate)

— **existiert bereits korrekt** als kanonischer Code-Pfad in `mayring-core`:

- `mayring_core/memory/ingestion/mayring_process.py`
  - `reduce_prompt(text, task, example_categories)` — Paraphrase→Generalisierung→
    Reduktion, **ziel-gebunden** (task obligatorisch), granularitäts-kalibriert via
    `_granularity_hint` (Label landet auf dem Abstraktionsniveau der Bestands-Kategorien
    → mehr Merging).
  - `mayring_process()` (Einzel) / `categorize_chunks()` (Batch) → `_assign_or_create()`:
    cosine **≥0.70** = deduktiver Match auf vorhandene Kategorie; **>0.92** = induktives
    Dedup (active+proposed); sonst echte induktive Neu-Kategorie als `proposed` (+ Chroma-Embedding).

**Das Problem ist nicht fehlende Logik, sondern 4-fache Doppelung derselben Kette** —
3 Kopien lassen den Embedding-Schritt weg und erzeugen damit genau das Chaos
(driftende Vokabulare, Duplikat-Labels), das der Embedding-Vergleich verhindern soll:

| Ort | Kette | Embedding | Output | Caller |
|---|---|---|---|---|
| `reduce_prompt`+`mayring_process` (mayring-core) | ✅ kanonisch | ✅ 0.70/0.92 | strukturiert (FK `chunk_categories`) | ingest |
| 4× `prompts/mayring_{hybrid,induktiv,deduktiv,s7_reduktion}.md` | ✅ (Prosa, je verschieden) | ❌ | Komma-Liste | `pi_categorize` |
| `pi_categorize` (`src/api/mcp_agent_tools.py:662`) | via Template | ❌ | Labels | goal-Skill, ingest-fallback |
| `pi_summarize_for_memory` (`mcp_agent_tools.py:1073`) | ✅ eigene Prosa | ❌ + **hardcodierte** Meta-Liste `[architecture,debug,config,decision,session-memory,context]` | 3-Felder-JSON | goal-Skill |
| `pi_mark_categories` (`mcp_agent_tools.py:747`) | ✅ eigene Prosa | ❌ | Span-Markierungen | **nirgends (tot)** |

**Unverbundene Funktionalität** (gebaut, nie verwendet):
`pi_mark_categories`, `pi_category_evidence`, `pi_judge_relevance` (läuft parallel zum
alten stop_hook-LLM-Judge), `cleanup_hallucinated_categories` + `reduce_categories`/S7
(nur manuell via MCP, kein Auto-Trigger).

**Zielbild (User-Entscheidung):** *Code-Primitive + 1 Prompt-SoT + dünner Skill.*
EINE Code-Methode mit **intrinsischem** Embedding-Vergleich + EINE parametrisierte
Prompt-Quelle (Modi als Parameter statt 4 Dateien) + EIN dünner universeller Skill als
Einstieg. Alle Tools werden dünne Wrapper, die sich nur in *Scope* (Chunk/Prompt/Span)
und *Persistenz* unterscheiden.

---

## 2. Generalisierungs-Einsicht (Antwort: „lässt sich hier noch generalisieren?")

**Ja.** Alle fünf Einsatzgebiete sind **dieselbe Operation** auf verschiedener
Granularität. Eine einzige Primitive subsumiert sie:

```
mayring_reduce(text, theme, existing_categories, mode) ->
    {
      paraphrase,
      generalization,
      candidates: [ { label, match: deductive | dedup | inductive, score } ]
    }
```

Die Unterschiede sind **orthogonale Wrapper-Belange**, keine eigene Logik:

| Use-Case | Scope | Extra-Schritt | Persistenz |
|---|---|---|---|
| ingest | Chunk (batched) | — | `chunk_categories` FK |
| `pi_categorize` | Einzeltext | multi-label | keine (nur return) |
| `pi_summarize_for_memory` | Freitext | intermediate-fields zurückgeben | Memory-Note |
| `pi_mark_categories` | Span | char-offset-Lokalisierung | `wiki_category_evidence` |
| goal-Skill | Prompt | IGIO-Axis-Klassifikation | Memory-Goal |

**Konsequenz:** Der Embedding-Vergleich wird **intrinsischer Bestandteil des
Code-Primitives**, damit er nie wieder weggelassen werden kann. Ein reiner Prompt
oder Skill kann cosine-gegen-Chroma nicht ausführen — deshalb Code-Primitive als Kern,
Prompt **nur** als Single-Source-of-Truth für die LLM-Reduktionsstufe.

---

## 3. Maßnahmen

### 3.1 Prompt-SoT konsolidieren (4 Dateien + 2 Inline-Prompts → 1 parametrisiert)

- `reduce_prompt(text, task, example_categories, mode="hybrid")` um `mode`
  (`inductive` | `deductive` | `hybrid`) erweitern. Der Modus steuert nur den
  Zuordnungs-/Neubildungs-Teil (deduktiv: kein Neu; induktiv: kein Anker; hybrid: beides).
- Optional `multi_label: bool` + **strukturierte Rückgabe** (`paraphrase`,
  `generalization`, `candidates`) statt nur des finalen Labels — deckt `pi_summarize`
  (intermediate fields) und multi-label `pi_categorize` mit ab.
- **Löschen:** `prompts/mayring_hybrid.md`, `prompts/mayring_induktiv.md`,
  `prompts/mayring_deduktiv.md` und `_load_mayring_template()` in `categorization.py`.
- **Behalten:** `prompts/mayring_s7_reduktion.md` — andere Operation (Label-Set →
  Merge-Map), aber **dokumentieren**: durch das intrinsische Dedup (>0.92) im ingest
  nur noch Backfill für Altbestand, kein Normalpfad.
- **Legacy `mayring_categorize()`** in `categorization.py` entfernen — nur noch in
  Tests genutzt, `core.py:302` markiert es bereits als ersetzt. Tests auf
  `categorize_chunks()` umstellen.

### 3.2 Wrapper auf den Embedding-Pfad routen

- **`pi_categorize`** (`mcp_agent_tools.py:662`): statt
  `_load_mayring_template`→`_pi_run`→Komma-Liste nun `categorize_chunks()` (multi-label)
  mit `conn`+`chroma`. Liefert deduktiv-gematchte bzw. neu gebildete Labels →
  **bekommt den fehlenden Embedding-Vergleich**.
- **`pi_summarize_for_memory`** (`mcp_agent_tools.py:1073`): die **hardcodierte
  Meta-Liste** `[architecture,debug,config,decision,session-memory,context]` ersetzen
  durch den dynamischen `_granularity_hint` aus den Bestands-Kategorien. `generalize`
  = embedding-gematchte/neue Kategorie statt fixe Liste. `paraphrase`/`generalize`/`reduce`
  kommen aus der strukturierten Primitive-Rückgabe.
- **goal-Skill** (`mayring-claude-plugin/skills/goal/SKILL.md`): Schritte 2+3 auf die
  vereinheitlichten Tools zeigen — kein eigener Pfad ohne Embedding mehr.

### 3.3 Tote Tools (Entscheidung: „verdrahten wo sinnvoll, Rest entfernen")

- **`pi_mark_categories`** → **verdrahten**: Kategorie-Ableitung über die Primitive
  (inkl. Embedding-Match), Span-Lokalisierung (char-offsets) bleibt einziger
  Extra-Schritt. An goal/ingest anbinden, damit `wiki_category_evidence` tatsächlich
  befüllt wird.
- **`pi_category_evidence`** → **behalten** — wird durch 3.3.1 lebendig (liest die Belege).
- **`pi_judge_relevance`** → mit dem alten stop_hook-LLM-Judge **vereinheitlichen**
  (ein Relevanz-Scoring-Pfad, JSON+scores). Doppelten Judge im stop_hook entfernen.
- **`cleanup_hallucinated_categories` / `reduce_categories` (S7)** → **behalten** als
  Backfill/Operator-Tools, im Doc als „nicht Normalpfad" markieren (intrinsisches
  Dedup übernimmt 99 %). Kein Auto-Cron in dieser Runde.

### 3.4 Dünner universeller Skill

- Einen universellen `mayring-coder:categorize`-Skill als Einstiegspunkt, der die
  vereinheitlichte Primitive dokumentiert/aufruft (Thema→Paraphrase→Generalisierung→
  Reduktion→Embedding). **`goal` wird Spezialfall davon** (IGIO-Axis-Klassifikation
  als Zusatz) — kein paralleler Pfad mehr.

---

## 4. Betroffene Dateien (Folge-Umsetzung)

- `vendor/mayring-core/mayring_core/memory/ingestion/mayring_process.py` — `reduce_prompt` (mode/struct), Primitive
- `vendor/mayring-core/mayring_core/memory/ingestion/categorization.py` — `_load_mayring_template` + `mayring_categorize` entfernen
- `prompts/mayring_{hybrid,induktiv,deduktiv}.md` — löschen; `mayring_s7_reduktion.md` behalten
- `src/api/mcp_agent_tools.py` — `pi_categorize` / `pi_summarize_for_memory` / `pi_mark_categories` / `pi_judge_relevance`
- `src/api/mcp_memory_tools.py` — Doku-Vermerke S7/cleanup
- stop_hook (Judge-Pfad) — Judge vereinheitlichen
- `mayring-claude-plugin/skills/goal/SKILL.md` + neuer `skills/categorize/SKILL.md`
- Tests: `test_mayring_process.py`, `test_pi_specialized_tools.py`, `test_cleanup_categories.py`

---

## 5. Risiken / offene Punkte

- **Multi-label vs. single-label:** `categorize_chunks` liefert heute 1 Label/Chunk;
  `pi_categorize` braucht N. Die Primitive muss multi-label sauber durch `_assign_or_create`
  schleifen (N Kandidaten → je Embedding-Match).
- **Prompt-Drift bei Migration:** beim Kollabieren der 4 Templates Regressionstests gegen
  bekannte Beispieltexte, damit die Label-Granularität stabil bleibt.
- **Client- vs. Server-Pfad:** `pi_*`-Tools laufen serverseitig (haben `conn`+`chroma`)
  → die Embedding-Route ist verfügbar.

---

## 6. Verification (Umsetzungs-Session)

```bash
cd MayringCoder
pytest tests/test_mayring_process.py tests/test_pi_specialized_tools.py tests/test_cleanup_categories.py -q
```

- Manuell: `pi_categorize` auf Beispieltext → Labels müssen Bestands-Kategorien
  matchen statt neue Duplikate zu erzeugen.
- `reduce_categories` dry-run: `unique_before ≈ unique_after` (Dedup greift jetzt
  bereits beim ingest).
