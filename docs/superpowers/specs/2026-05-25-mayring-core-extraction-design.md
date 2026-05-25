# mayring-core in eigenes Repo extrahieren — Design (#267-Follow-up)

**Datum:** 2026-05-25
**Status:** ✅ Approved (User „sieht gut aus", pi-agent zusätzlich public gestellt)
**Verbunden mit:** #266 (Pi-Agent-Dedup, Follow-up „mayring-core eigenes Repo bricht den vendor↔core-Zyklus"), #267 (core/-Package-Extraktion, PR #271/#273), [[project_mayring_core_extraction_267]], [[project_pi_agent_dedup_266]], [[project_pi_agent_release_tagging]]

## Problem

`vendor/mayring-pi-agent` (eigenes, jetzt public Repo) hängt per git-dep an
`mayring-core @ git+.../MayringCoder.git@master#subdirectory=core`. MayringCoder
vendored pi-agent gleichzeitig als Submodul → **Zyklus**. Folge: das
pi-agent-Image muss das **gesamte** MayringCoder-Repo klonen, nur um `core/` zu
beziehen. Das blockiert den pi-agent-Image-Rebuild und den Prod-Container 0.1.x
(offener Punkt aus #266).

Zusätzlich ist **CI seit dem #266-Vendoring rot** (eigenständige Regression):
~10 Testdateien importieren `mayring_pi_agent`, aber `.github/workflows/tests.yml`
checkt das Submodul nicht aus und installiert das Paket nicht → ModuleNotFound
bei Collection → `pytest` exit 1. Lokal grün, weil das Submodul hier ausgecheckt
ist.

## Zielzustand (azyklisch)

```
mayring-core          neu, PUBLIC, keine Sibling-Deps
mayring-pi-agent  →   git-dep mayring-core@v0.1.0       (kein MayringCoder-Clone mehr)
MayringCoder      →   submodule vendor/mayring-core (editable)
                  →   submodule vendor/mayring-pi-agent@v0.1.3
```

Topologie nach Extraktion: `MayringCoder → mayring-core`,
`MayringCoder → pi-agent → mayring-core`, `mayring-core → ∅`. Kein Zyklus.

**Entscheidungen (vom User bestätigt):**
- **Konsum-Modell:** Submodul `vendor/mayring-core` in MayringCoder (konsistent
  mit pi-agent, editable Dev ohne Tag-Bump, commit-genaues Pinning). pi-agent
  bezieht core via git-dep auf einen Tag.
- **Git-History:** erhalten via `git filter-repo` (25 core/-Commits wandern mit).
- **CI-Rot:** zuerst eigenständig grün machen, dann auf grünem CI extrahieren.
- **Sichtbarkeit:** mayring-core **public** (wie MayringCoder + pi-agent) → kein
  `GH_PAT` für Submodul-Checkout/Build nötig; pi-agents core-Fetch bleibt
  creds-frei.

## Schritt 0 — CI grün (eigenständig, vor der Extraktion)

Reiner Fix der #266-Regression, unabhängig vom Rest. `.github/workflows/tests.yml`:

```yaml
- uses: actions/checkout@v4
  with:
    submodules: recursive        # beide Submodule public → kein token nötig

# Install deps — nach `pip install -e ./core`:
- pip install -e ./vendor/mayring-pi-agent --no-deps
```

`--no-deps` spiegelt `docker/Dockerfile` (Z. 18-20): pi-agents `mayring-core`-
git-dep ist bereits durch `-e ./core` befriedigt; restliche Runtime-Deps
(httpx/fastapi/uvicorn) liegen in `requirements.txt`. Ohne `--no-deps` würde pip
die URL-dep erneut aus dem Netz ziehen.

**Akzeptanz:** Push auf master → CI-Run grün (`gh run list --workflow=tests.yml`).

## Schritt 1 — Repo `Nileneb/mayring-core` mit History anlegen

```bash
git clone https://github.com/Nileneb/MayringCoder.git /tmp/mayring-core-extract
cd /tmp/mayring-core-extract
git filter-repo --path core/ --path-rename core/:    # core/ wird Repo-Root, 25 Commits bleiben
```

- `git-filter-repo` ist vorhanden (`/home/nileneb/miniconda3/bin/git-filter-repo`).
- Kein tracked junk in `core/` (`egg-info`/`cache`/`__pycache__` sind gitignored
  und nicht getrackt) → nichts zu bereinigen.

**Verify (lokal, vor Push):**
```bash
test -f pyproject.toml && test -d mayring_core      # auf Root-Ebene
pip install -e . && python -c "import mayring_core; print(mayring_core.__file__)"
```

**Minimal-CI** (`.github/workflows/tests.yml` im neuen Repo — KISS, core hat
heute keine eigenen Tests):
```yaml
name: tests
on: { push: { branches: [main] }, pull_request: { branches: [main] } }
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install -e .
      - run: python -c "import mayring_core"
```

**Publizieren + taggen:**
```bash
gh repo create Nileneb/mayring-core --public --source=. --push
git tag v0.1.0 && git push origin v0.1.0      # = pyproject version 0.1.0
```

## Schritt 2 — pi-agent umverdrahten (Repo `Nileneb/mayring-pi-agent`)

`pyproject.toml`-Dependency ändern:
```
- "mayring-core @ git+https://github.com/Nileneb/MayringCoder.git@master#subdirectory=core",
+ "mayring-core @ git+https://github.com/Nileneb/mayring-core.git@v0.1.0",
```

- `Dockerfile` bleibt unverändert (`pip install .` zieht jetzt das public
  mayring-core; `git` ist im Image vorhanden, keine Creds nötig).
- pi-agent-eigenes CI (`.github/workflows/tests.yml`: `pip install -e . pytest`)
  zieht core danach aus dem neuen public Repo → muss grün bleiben.
- **Release:** Commit + Tag `v0.1.3` (nur Dep-Quelle geändert → Patch).
  Konvention [[project_pi_agent_release_tagging]]: git-Tag `vX.Y.Z` → Image-Tag
  `X.Y.Z` (metadata-action strippt `v`); compose-Pin OHNE `v`.

**Akzeptanz:** pi-agent-CI grün; `pip install .` in sauberem venv löst
`mayring-core` aus dem neuen Repo auf (nicht mehr aus MayringCoder).

## Schritt 3 — MayringCoder auf Submodul umstellen

```bash
git rm -r core/
git submodule add https://github.com/Nileneb/mayring-core.git vendor/mayring-core
cd vendor/mayring-core && git checkout v0.1.0 && cd ../..
```

Pfad-Referenzen anpassen (vollständige Liste aus Grep):

| Datei | Alt | Neu |
|---|---|---|
| `docker/Dockerfile:14` | `COPY core/ ./core/` | `COPY vendor/mayring-core ./vendor/mayring-core` |
| `docker/Dockerfile:15` | `pip install -e ./core` | `pip install -e ./vendor/mayring-core` |
| `.github/workflows/tests.yml:36` | `pip install -e ./core` | `pip install -e ./vendor/mayring-core` |
| `pytest.ini:4` | `pythonpath = . core tools` | `pythonpath = . vendor/mayring-core tools` |
| `.github/workflows/build-and-push.yml:8` | Path-Filter `'core/**'` | `'vendor/mayring-core'` (Submodul-Gitlink ändert sich dort, nicht mehr unter `core/**`) |

- pi-agent-Submodul-Pointer auf `v0.1.3` bumpen.
- `src/`- und `tools/`-Importe `from mayring_core …` bleiben **unverändert**
  (Paketname identisch).
- `.gitmodules` enthält danach zwei Submodule.

**Akzeptanz:** lokal sauberes venv →
```bash
pip install -r requirements.txt
pip install -e ./vendor/mayring-core
pip install -e ./vendor/mayring-pi-agent --no-deps
pytest -q          # grün
```

## Schritt 4 — Verifizieren / Outcome

1. Extraktions-Commit auf master push → CI grün.
2. **pi-agent-Image standalone bauen ohne MayringCoder-Klon** → bestätigt, dass
   der Zyklus gebrochen ist (das eigentliche Outcome).
3. Entsperrt: pi-agent-Prod-Container auf 0.1.3 rebuilden (blockierter Punkt #266).

## Risiken / Gotchas

- **filter-repo-Korrektheit:** `pyproject.toml` + `mayring_core/` müssen nach
  `--path-rename core/:` auf Root liegen; `import mayring_core` muss ziehen
  (Verify in S1 vor Push).
- **`--no-deps` für pi-agent darf nicht wegfallen** (CI + Dockerfile), sonst
  Re-Fetch der core-URL-dep.
- **Reihenfolge S1→S2:** mayring-core muss als `v0.1.0` gepusht+getaggt sein,
  bevor pi-agents Dep auflöst (sonst „ref not found" im Build).
- **build-and-push.yml-Path-Filter:** ohne Anpassung triggert ein core-Update
  (= Submodul-Pointer-Bump) keinen MayringCoder-Image-Rebuild → stiller
  Fehl-Deploy.
- **Zwei Submodule:** Clones/CI brauchen `submodules: recursive`.
- **Cross-Repo-Atomarität:** Die drei Repos können nicht atomar in einem PR
  geändert werden. Direkter Push auf default-Branch ist für MayringCoder/
  pi-agent autorisiert; mayring-core ist neu (Erst-Push). S0 ist ein separater
  CI-Fix-Commit; S1-S3 folgen in Reihenfolge.
