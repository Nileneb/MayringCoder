# MayringCoder

> Lokale KI-gestützte Analyse für GitHub-Repositories und Textdokumente — vollständig offline, mit Ollama.

MayringCoder lädt ein beliebiges GitHub-Repository oder Textkorpus, kategorisiert alle Dateien automatisch und analysiert sie mit einem lokalen Sprachmodell. Die Ergebnisse werden als strukturierter Markdown-Report gespeichert.

Die Analyse-Pipeline folgt dem **Mayring-Verfahren** aus der qualitativen Inhaltsanalyse:
Strukturierung → Reduktion → Explikation → Zusammenführung.

**Zwei Einsatzbereiche — eine Pipeline:**

| Modus | Zweck | Codebook | Prompt |
|---|---|---|---|
| **Code-Review** | Code-Smells, Security, Architektur | `codebook.yaml` | `prompts/file_inspector.md` |
| **Sozialforschung** | Qualitative Inhaltsanalyse nach Mayring | `codebook_sozialforschung.yaml` | `prompts/mayring_deduktiv.md` oder `prompts/mayring_induktiv.md` |

---

## Features

- **Vollständig lokal** — kein Cloud-API-Key nötig, läuft über [Ollama](https://ollama.com)
- **Snapshot-basiertes Caching** — nur geänderte oder neue Dateien werden analysiert (SQLite)
- **Automatische Dateikategorisierung** — YAML-Codebook sortiert Dateien in Kategorien
- **Priorisierung nach Risiko** — sicherheitskritische Kategorien (api, data_access, domain) werden bevorzugt
- **Mehrere Analyse-Modi** — Code-Review, qualitative Inhaltsanalyse (deduktiv/induktiv), Explikation
- **Budget-Limit** — maximal 20 Dateien pro Lauf (konfigurierbar), Rest bleibt in der Queue
- **Explikations-Flag** — Findings mit niedriger Konfidenz werden markiert und können per Re-Run vertieft werden

---

## Wie es funktioniert

```
GitHub Repo / Textkorpus
    │
    ▼ 1. Fetch (gitingest)
Roher Repo-Snapshot
    │
    ▼ 2. Split
Einzelne Datei-Dicts (Pfad, Inhalt, SHA256-Hash, Größe)
    │
    ▼ 3. Kategorisierung (Mayring Stufe 1: Strukturierung)
Dateien mit Kategorie (api / domain / argumentation / methodik / ...)
    │
    ▼ 4. Diff (SQLite Cache)
Nur neue / geänderte Dateien → Analyse-Queue
    │
    ▼ 5. LLM-Analyse (Mayring Stufen 2+3: Reduktion + Explikation)
Strukturiertes JSON pro Datei: Findings/Codierungen, Severity, Konfidenz
    │
    ▼ 6. Aggregation (Mayring Stufe 4: Zusammenführung)
Top-Findings, deduplizierte Handlungsempfehlungen / Kategoriensystem
    │
    ▼ 7. Report
reports/repo-check-YYYY-MM-DD_HHMM.md + run_meta.json
```

---

## Voraussetzungen

- Python 3.11+
- [Ollama](https://ollama.com) lokal installiert und gestartet (`ollama serve`)
- Ein Ollama-Modell geladen, z. B. `ollama pull llama3.1:8b`
- Optional: GitHub Personal Access Token für private Repos oder höhere Rate Limits

---

## Installation

```bash
git clone https://github.com/Nileneb/MayringCoder.git
cd MayringCoder

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# .env anpassen (siehe Konfiguration)
```

---

## Konfiguration

`.env` Datei:

```env
GITHUB_REPO=https://github.com/dein/repo
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
GITHUB_TOKEN=                   # optional, für private Repos
```

---

## Nutzung

### Code-Review (Standard)

```bash
# Standard-Analyse (nur geänderte Dateien seit letztem Run)
python checker.py

# Anderes Repository analysieren
python checker.py --repo https://github.com/user/repo

# Anderes Modell verwenden
python checker.py --model qwen2.5-coder:7b

# Alle Dateien analysieren (Cache ignorieren)
python checker.py --full

# Vorschau: zeigt nur welche Dateien analysiert würden
python checker.py --dry-run

# Ausgewählte Dateien mit Kategorie anzeigen
python checker.py --show-selection

# Alternativen Prompt verwenden
python checker.py --prompt prompts/smell_inspector.md

# Raw-Snapshot lokal speichern (Debugging)
python checker.py --debug
```

### Qualitative Inhaltsanalyse (Sozialforschung)

```bash
# Deduktiv — feste Kategorien vorgegeben
python checker.py --repo https://github.com/user/repo \
  --codebook codebook_sozialforschung.yaml \
  --prompt prompts/mayring_deduktiv.md

# Induktiv — Kategorien entstehen aus dem Material
python checker.py --repo https://github.com/user/repo \
  --codebook codebook_sozialforschung.yaml \
  --prompt prompts/mayring_induktiv.md
```

**Was ist der Unterschied?**

- **Deduktiv:** Du gibst ein festes Kategoriensystem vor (z. B. `argumentation`, `methodik`, `ergebnis`, `limitation`). Das LLM ordnet jede Textstelle einer dieser Kategorien zu. Geeignet für: Literatur-Reviews, strukturierte Textanalysen mit klarem Raster.

- **Induktiv:** Das LLM liest den Text und entwickelt die Kategorien selbst aus dem Material. Am Ende liefert es eine `category_summary` mit Definitionen und Häufigkeiten. Geeignet für: explorative Analysen, Interview-Transkripte, offene Codierung.

---

## Analyse-Prompts

### Code-Review-Prompts

| Datei | Modus | Ausgabe |
|---|---|---|
| `prompts/file_inspector.md` | **Standard** — strukturierte JSON-Analyse | JSON mit `file_summary` + `potential_smells` |
| `prompts/smell_inspector.md` | Freie Code-Review-Analyse | Fließtext mit Severity-Emojis |
| `prompts/explainer.md` | Explikation für unklare Findings | Klärung ob Finding berechtigt + Fix-Vorschlag |

### Sozialforschungs-Prompts

| Datei | Modus | Ausgabe |
|---|---|---|
| `prompts/mayring_deduktiv.md` | Deduktive Inhaltsanalyse (feste Kategorien) | JSON mit `file_summary` + `codierungen` |
| `prompts/mayring_induktiv.md` | Induktive Inhaltsanalyse (offene Kategorien) | JSON mit `file_summary` + `codierungen` + `category_summary` |

### Explikations-Workflow

Findings mit `confidence: low` werden automatisch als `needs_explikation: true` markiert und im Report hervorgehoben. Für eine vertiefte Analyse:

```bash
python checker.py --prompt prompts/explainer.md --full
```

---

## Codebooks (Dateikategorisierung)

Ein Codebook definiert, wie Dateien automatisch in Kategorien einsortiert werden. Es bestimmt **nicht**, welche Fehler gesucht werden — das macht der Prompt. Das Codebook bestimmt nur die **Sortierung und Priorisierung**.

### `codebook.yaml` — Code-Review (Standard)

| Kategorie | Beschreibung | Risiko-Priorität |
|---|---|---|
| `api` | Routes, Controller, Endpoints | 🔴 hoch |
| `data_access` | ORM-Modelle, Migrations, Repositories | 🔴 hoch |
| `domain` | Business-Logik, Services, Use Cases | 🔴 hoch |
| `ui` | Templates, Komponenten, Views | normal |
| `config` | Settings, YAML, ENV-Dateien | normal |
| `utils` | Hilfsfunktionen, Helpers | normal |
| `tests` | Unit- und Integrationstests | normal |
| `temp_dummy` | Placeholder, TODOs, Backup-Dateien | normal |

### `codebook_sozialforschung.yaml` — Qualitative Inhaltsanalyse

| Kategorie | Beschreibung |
|---|---|
| `argumentation` | Thesen, Begründungen, Schlussfolgerungen |
| `methodik` | Forschungsdesign, Methoden, Stichproben |
| `ergebnis` | Befunde, Resultate, Daten, Kennzahlen |
| `limitation` | Einschränkungen, Schwächen, offene Fragen |
| `theorie` | Theoretische Rahmung, Konzepte, Definitionen |
| `kontext` | Hintergrund, Forschungsstand, Literaturverweise |
| `wertung` | Bewertungen, Empfehlungen, normative Aussagen |
| `unklar` | Mehrdeutige oder nicht zuordenbare Textstellen |

Eigene Kategorien können in beiden Codebooks ergänzt werden. Regex-Muster beginnen mit `re:`.

---

## Report-Format

Jeder Lauf erzeugt zwei Dateien in `reports/`:

**`repo-check-YYYY-MM-DD_HHMM.md`** — Markdown-Report mit:
- Summary (Dateien, Findings, Laufzeit)
- Category Digest
- Top-5 Findings (nach Severity × Konfidenz)
- Per-File Findings mit Codeausschnitt und Fix-Vorschlag
- Explikations-Liste (Findings zur manuellen Nachprüfung)
- Empfohlene nächste Schritte

**`repo-check-YYYY-MM-DD_HHMM_meta.json`** — Maschinenlesbare Metadaten (Diff-Stats, analysierte Dateien, Laufzeit)

### Finding-Schema — Code-Review (JSON)

```json
{
  "type": "zombie_code|redundancy|inconsistent_pattern|error_handling|overengineering|security|unclear",
  "severity": "critical|warning|info",
  "confidence": "high|medium|low",
  "line_hint": "~42",
  "evidence_excerpt": "max. 10 Zeilen relevanter Code",
  "fix_suggestion": "konkrete Handlungsempfehlung",
  "needs_explikation": false
}
```

### Codierungs-Schema — Sozialforschung (JSON)

```json
{
  "category": "argumentation|methodik|ergebnis|limitation|theorie|kontext|wertung|unklar",
  "confidence": "high|medium|low",
  "line_hint": "~42",
  "evidence_excerpt": "wörtliches Zitat, max. 3 Sätze",
  "reasoning": "Begründung der Zuordnung",
  "needs_explikation": false
}
```

Bei induktiver Analyse zusätzlich:

```json
{
  "category_summary": [
    {
      "category": "aus_dem_text_abgeleitete_kategorie",
      "definition": "Was diese Kategorie inhaltlich abdeckt (1 Satz)",
      "count": 3
    }
  ]
}
```

---

## Caching & Incremental Analysis

MayringCoder legt für jedes Repository eine SQLite-Datenbank unter `cache/<repo-slug>.db` an.

- Bei jedem Lauf wird ein neuer **Snapshot** erstellt
- Dateien werden per **SHA256-Hash** verglichen
- Nur Dateien ohne `analyzed_at`-Stempel kommen in die Queue
- Das **Budget-Limit** (Standard: 20 Dateien) verhindert zu lange Laufzeiten
- Verbleibende Dateien werden beim nächsten `python checker.py` automatisch fortgesetzt

```bash
# Cache zurücksetzen / alle Dateien erneut analysieren:
python checker.py --full
```

---

## Projektstruktur

```
MayringCoder/
├── checker.py                        # Einstiegspunkt & Pipeline-Orchestrierung
├── codebook.yaml                     # Dateikategorie-Definitionen (Code-Review)
├── codebook_sozialforschung.yaml     # Dateikategorie-Definitionen (Sozialforschung)
├── requirements.txt
├── .env.example
├── prompts/
│   ├── file_inspector.md             # Standard-Prompt (strukturiertes JSON, Code-Review)
│   ├── smell_inspector.md            # Alternativer Fließtext-Prompt (Code-Review)
│   ├── explainer.md                  # Explikations-Prompt für unklare Findings
│   ├── mayring_deduktiv.md           # Deduktive Inhaltsanalyse (feste Kategorien)
│   └── mayring_induktiv.md           # Induktive Inhaltsanalyse (offene Kategorien)
└── src/
    ├── fetcher.py                    # Repo laden via gitingest
    ├── splitter.py                   # gitingest-Output in Datei-Dicts aufteilen
    ├── categorizer.py                # Dateien per Codebook kategorisieren
    ├── cache.py                      # SQLite Snapshot-Diff & Queue-Management
    ├── analyzer.py                   # LLM-Analyse via Ollama (JSON-Parsing, Fallback)
    ├── aggregator.py                 # Findings zusammenführen, ranken, deduplizieren
    ├── report.py                     # Markdown-Report + run_meta.json generieren
    └── config.py                     # Zentrale Konstanten und Pfade
```

---

## Empfohlene Modelle

| Modell | Qualität | Geschwindigkeit | Hinweis |
|---|---|---|---|
| `llama3.1:8b` | gut | schnell | Standard, gute Balance |
| `qwen2.5-coder:7b` | sehr gut | schnell | speziell für Code |
| `deepseek-coder-v2:16b` | exzellent | langsam | für kritische Reviews |

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
```

---

## Limitierungen

- Dateien werden auf **3.000 Zeichen** gekürzt — sehr große Dateien werden nur teilweise analysiert
- Die LLM-Ausgabe ist nicht deterministisch — gleiche Dateien können bei unterschiedlichen Runs leicht abweichende Findings liefern
- Findings mit `confidence: low` sollten mit dem Explainer-Prompt manuell nachgeprüft werden
- Der `smell_inspector`-Prompt erzeugt Fließtext statt JSON — Aggregation und Report-Detail-Ansicht sind dort eingeschränkt
- Der Sozialforschungs-Modus erwartet Textdateien (.md, .txt, .pdf, .docx) — bei reinen Code-Repos liefert er keine sinnvollen Ergebnisse

---

## Lizenz

Proprietär — Alle Rechte vorbehalten.
