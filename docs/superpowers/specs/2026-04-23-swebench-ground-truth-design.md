# SWEbench Ground-Truth-Validierung für MayringCoder

**Datum:** 2026-04-23  
**Status:** Approved  
**Scope:** Standalone-Skript, keine Änderungen an bestehenden MayringCoder-Komponenten

---

## Kontext & Ziel

MayringCoder's aktueller Benchmark (`tools/benchmark.sh`) misst Throughput und Parsing-Qualität, aber hat keine Ground-Truth-Validierung: Es ist unbekannt wie viele echte Bugs MayringCoder findet (Recall) und wie viele Findings falsch positiv sind.

SWEbench-Lite enthält ~300 reale GitHub-Issues mit bekannten Bug-Commits und Patches. Pre-Fix-Code = bekannter Bug-Zustand. Das ermöglicht erstmals: **"Hat MayringCoder den Bug gefunden, der tatsächlich existierte?"**

---

## Architektur

### Neue Datei: `benchmarks/swebench_eval.py`

Standalone-Skript. Ruft `checker.py` als Subprocess auf — keinerlei Änderungen an MayringCoder selbst.

### Abhängigkeiten

```
datasets>=2.0        # HuggingFace — SWEbench-Lite laden
gitpython>=3.0       # Repo klonen + Commit auschecken
```

Werden in `pyproject.toml` als optionale Dev-Abhängigkeit eingetragen (`[project.optional-dependencies] bench = [...]`).

### Output-Verzeichnis

```
benchmarks/swebench_results/    ← gitignored
  run_2026-04-23_143022.csv
  run_2026-04-23_143022_details.json
```

---

## Ablauf (Schritt für Schritt)

### 1. Dataset laden

```python
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
instances = list(ds)[:10]  # konfigurierbar via --n
```

Relevante Felder je Instance:
- `instance_id` — eindeutige ID (z.B. `django__django-11099`)
- `repo` — GitHub-Pfad (z.B. `django/django`)
- `base_commit` — SHA des buggy Commits
- `patch` — Unified Diff des Fixes (enthält die Ground-Truth-Files)
- `problem_statement` — Issue-Beschreibung (optional für Logging)

### 2. Repo klonen + Commit auschecken

```python
import tempfile, git
with tempfile.TemporaryDirectory() as tmp:
    repo = git.Repo.clone_from(f"https://github.com/{instance['repo']}", tmp, depth=50)
    repo.git.checkout(instance['base_commit'])
    # → tmp/ enthält buggy code
```

### 3. Ground-Truth-Files aus Patch extrahieren

```python
def patched_files(patch: str) -> set[str]:
    # Parst "--- a/path/file.py" Zeilen aus Unified Diff
    return {line[6:] for line in patch.splitlines() if line.startswith("--- a/")}
```

### 4. MayringCoder aufrufen

`checker.py` ist ein HTTP-Client — kein `--output-format json`. Findings werden serverseitig als JSON-Dateien gespeichert.

```python
import httpx, time

# 1. Job starten
r = httpx.post(f"{api_url}/analyze", json={"repo": tmp, "workspace_id": ws_id}, timeout=10)
job_id = r.json()["job_id"]

# 2. Pollen bis fertig
while True:
    status = httpx.get(f"{api_url}/jobs/{job_id}").json()
    if status["status"] in ("done", "error"):
        break
    time.sleep(3)

# 3. run_id aus Output extrahieren + Run-JSON laden
run_id = status.get("run_id")  # API gibt run_id zurück
from src.analysis.history import load_run
run_data = load_run(run_id, repo_url=tmp, workspace_id=ws_id)
findings = run_data.get("results", [])
```

**Voraussetzung:** MayringCoder API-Server läuft lokal (`python -m src.api.server`). Das Skript wartet nicht auf den Start — User muss den Server selbst starten, Skript prüft Erreichbarkeit beim Start.

MayringCoder unterstützt lokale Pfade als `repo`-Parameter bereits.

### 5. Match-Bestimmung (File-Level)

```python
mc_files = {f["filename"] for f in findings.get("results", [])}
gt_files = patched_files(instance["patch"])

match = "TP" if mc_files & gt_files else "FN"
```

**TP**: MayringCoder hat ≥1 Finding in mindestens einer Ground-Truth-Datei.  
**FN**: Kein Finding in einer Ground-Truth-Datei.

FP ist auf File-Level nicht sauber bestimmbar (kein "korrekte Dateien"-Komplettsatz vorhanden) — wird als zukünftige Erweiterung vermerkt.

### 6. Output

**Konsole:**
```
instance_id                  repo                 gt_files         mc_hit  match
django__django-11099         django/django        views/generic.py  YES     TP
psf__requests-4356           psf/requests         auth.py           NO      FN
...
────────────────────────────────────────────────────────
Recall:  6/10 (60.0%)
Avg findings/instance: 4.2
Runtime: 8m 32s
```

**CSV** (`swebench_results/run_DATUM.csv`):
```
instance_id,repo,base_commit,gt_files,mc_files,match,findings_count,runtime_s
```

**JSON** (`_details.json`): Vollständige Findings je Instance für manuelle Analyse.

---

## CLI-Interface

```bash
python benchmarks/swebench_eval.py [OPTIONS]

Options:
  --n INT           Anzahl Instances (default: 10)
  --seed INT        Random-Seed für Instance-Auswahl (default: 42)
  --model TEXT      Ollama-Modell für checker.py (default: aus .env / checker default)
  --time-budget INT Sekunden je Instance für checker.py (default: 120)
  --output-dir PATH Ausgabeverzeichnis (default: benchmarks/swebench_results/)
```

---

## Einschränkungen & bekannte Lücken

| Einschränkung | Begründung | Workaround |
|---|---|---|
| Kein FP-Rate | File-Level-Match kann FP nicht isolieren | Function-Level als spätere Erweiterung |
| Klone dauern | Große Repos (django ~500MB) | `--depth 50` minimiert Download |
| Nur Python-Repos | SWEbench-Lite enthält nur Python | Ausreichend für ersten Benchmark |
| Netzwerkabhängig | HuggingFace + GitHub | Cache-Mechanismus als Option |

---

## Nicht in Scope

- Änderungen an `checker.py`, `benchmark.sh` oder `benchmark_summary.py`
- Automatische Integration in CI
- FP-Messung (erfordert vollständige Annotation der Bugs)
- Patch-Generierung (MayringCoder bleibt Analysis-Tool)
