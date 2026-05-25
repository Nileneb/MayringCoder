# Pi-Agent Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MayringCoder konsumiert `mayring-pi-agent` als Git-Submodule-Package und löscht das duplizierte `src/agents/`, sodass nur noch eine Pi-Agent-Codebase existiert.

**Architecture:** `mayring-pi-agent` wird als Submodule unter `vendor/mayring-pi-agent` (Tag `0.1.1`) eingebunden und im Docker-Build editable + `--no-deps` installiert (gegen die zirkuläre `mayring-core`-Git-Dep). Alle 22 `src.agents`-Importe über 11 Dateien werden per Prefix-Swap auf `mayring_pi_agent` umgestellt, dann `src/agents/` gelöscht. Bestehende Tests (`test_pi_*`, `test_duel`, `test_devices_registry`, `test_vision_captioner`) sind das Safety-Net; ein Grep-Gate-Test sichert die Vollständigkeit.

**Tech Stack:** Python 3.13, FastAPI, pytest, Docker, GitHub Actions, git-submodule.

**Spec:** `docs/superpowers/specs/2026-05-25-pi-agent-dedup-design.md`

---

## File Structure

| Datei | Verantwortung | Änderung |
|---|---|---|
| `.gitmodules` (Create) | Submodule-Registrierung | neu |
| `vendor/mayring-pi-agent` (Submodule) | Pi-Agent-Package, Pin `0.1.1` | neu |
| `docker/Dockerfile` | Image-Build | `COPY` + `pip install -e --no-deps` |
| `.github/workflows/build-and-push.yml` | CI-Image-Build | checkout `submodules: true` + token |
| `src/api/server.py`, `routes/{memory,devices,duel,pi_stats}.py`, `mcp_agent_tools.py`, `analysis/analyzer.py`, `workflows/pi_task.py`, `provider_setup.py`, `api/local_mcp.py`, `main.py` | Pi-Konsumenten | Import-Prefix-Swap |
| `src/agents/` | dupliziertes Paket | **gelöscht** |
| `tests/` (13 Dateien) | Test-Importe | Import-Prefix-Swap |
| `tests/test_no_src_agents_imports.py` (Create) | Grep-Gate | neu |

**Vollständige Swap-Liste (22 Stellen):** `src.X` → `mayring_pi_agent.X`
- `src/workflows/pi_task.py:13`, `src/provider_setup.py:33,38`, `src/main.py:57`, `src/analysis/analyzer.py:410`, `src/api/routes/pi_stats.py:11`, `src/api/server.py:144,145,151,188`, `src/api/mcp_agent_tools.py:75,91,189,222,252,496`, `src/api/routes/devices.py:33`, `src/api/routes/duel.py:46,129`, `src/api/local_mcp.py:51`, `src/api/routes/memory.py:69,70`

---

## Task 1: Submodule einbinden + Build-Integration

**Files:**
- Create: `.gitmodules`, `vendor/mayring-pi-agent` (submodule)
- Modify: `docker/Dockerfile`
- Test: `tests/test_pi_agent_package_smoke.py` (Create)

- [ ] **Step 1: Submodule hinzufügen + auf Tag 0.1.1 pinnen**

```bash
cd /home/nileneb/Desktop/MayringCoder
git submodule add https://github.com/Nileneb/mayring-pi-agent.git vendor/mayring-pi-agent
cd vendor/mayring-pi-agent && git checkout 0.1.1 && cd ../..
git add .gitmodules vendor/mayring-pi-agent
```
Expected: `.gitmodules` enthält `[submodule "vendor/mayring-pi-agent"]`; `git submodule status` zeigt `0.1.1`.

- [ ] **Step 2: Paket lokal editable installieren (ohne deps, mayring-core ist schon da)**

```bash
.venv/bin/pip install -e ./vendor/mayring-pi-agent --no-deps
```
Expected: `Successfully installed mayring-pi-agent-0.1.1`.

- [ ] **Step 3: Import-Smoke-Test schreiben (verifiziert API-Kompatibilität)**

`tests/test_pi_agent_package_smoke.py`:
```python
"""#266: the externalized mayring-pi-agent package must expose the exact
symbols MayringCoder consumes in-process, so the src/agents → package swap
is a 1:1 replacement (no behavioural loss)."""


def test_package_exposes_consumed_symbols():
    from mayring_pi_agent.pi import run_task_with_memory, analyze_with_memory
    from mayring_pi_agent.pi_queue import get_pi_queue
    from mayring_pi_agent.pi_jobs import PiJob, classify_pi_job
    from mayring_pi_agent.vision import caption_image, get_image_metadata
    from mayring_pi_agent.diff_history import DiffHistoryError, run
    from mayring_pi_agent import pi_worker, pi_server
    assert callable(run_task_with_memory)
    assert callable(analyze_with_memory)
    assert callable(get_pi_queue)
    assert callable(classify_pi_job)
    assert callable(pi_worker.start)
    assert hasattr(pi_server, "app")
```

- [ ] **Step 4: Test laufen lassen — muss PASSEN (Paket ist installiert)**

Run: `.venv/bin/python -m pytest tests/test_pi_agent_package_smoke.py -v`
Expected: PASS. (Falls FAIL mit ImportError → API-Drift; Symbol im Paket prüfen, bevor weitergemacht wird.)

- [ ] **Step 5: Dockerfile anpassen**

In `docker/Dockerfile`, direkt NACH dem Block `COPY core/ ./core/` + `RUN pip install --no-cache-dir -e ./core` einfügen:
```dockerfile
# mayring-pi-agent (git-submodule, vendor/) — installed editable WITHOUT deps:
# its mayring-core git-dependency is already satisfied by ./core above, and the
# remaining runtime deps (httpx/fastapi/uvicorn) live in requirements.txt.
COPY vendor/mayring-pi-agent ./vendor/mayring-pi-agent
RUN pip install --no-cache-dir -e ./vendor/mayring-pi-agent --no-deps
```

- [ ] **Step 6: Commit**

```bash
git add .gitmodules vendor/mayring-pi-agent docker/Dockerfile tests/test_pi_agent_package_smoke.py
git commit -m "feat(#266): vendor mayring-pi-agent submodule + build integration

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Grep-Gate-Test (RED bis Task 4)

**Files:**
- Test: `tests/test_no_src_agents_imports.py` (Create)

- [ ] **Step 1: Gate-Test schreiben**

`tests/test_no_src_agents_imports.py`:
```python
"""#266: after the dedup, NOTHING in src/ may import the deleted src/agents
package. This gate fails until every consumer is swapped to mayring_pi_agent."""
import pathlib
import re

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
_PAT = re.compile(r"\b(from|import)\s+src\.agents\b|['\"]src\.agents")


def test_no_src_agents_references_in_src():
    offenders = []
    for py in _SRC.rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if _PAT.search(line):
                offenders.append(f"{py.relative_to(_SRC.parent)}:{i}")
    assert not offenders, "src.agents still referenced:\n" + "\n".join(offenders)
```

- [ ] **Step 2: Test laufen lassen — muss FEHLSCHLAGEN (22 Offender)**

Run: `.venv/bin/python -m pytest tests/test_no_src_agents_imports.py -v`
Expected: FAIL, Liste mit ~22 `src.agents`-Referenzen.

(Kein Commit — der Gate-Test wird in Task 4 gemeinsam mit dem grünen Zustand committet.)

---

## Task 3: Import-Swap in src/ + Safety-Net-Tests

**Files:**
- Modify: alle 11 src-Dateien aus der Swap-Liste

- [ ] **Step 1: Prefix-Swap über die betroffenen src-Dateien**

```bash
cd /home/nileneb/Desktop/MayringCoder
grep -rl "src\.agents" src/ --include="*.py" | grep -v "^src/agents/" | \
  xargs sed -i 's/src\.agents/mayring_pi_agent/g'
```
Das ersetzt `from src.agents.X` → `from mayring_pi_agent.X`, `from src.agents import` → `from mayring_pi_agent import`, sowie die Strings in `main.py:57` (`"mayring_pi_agent.pi_server:app"`) und im Hint `mcp_agent_tools.py:75`.

- [ ] **Step 2: Verifizieren — kein src.agents mehr in src/ (außer src/agents/ selbst)**

Run: `grep -rn "src\.agents" src/ --include="*.py" | grep -v "^src/agents/"`
Expected: leere Ausgabe.

- [ ] **Step 3: Safety-Net-Tests der betroffenen Module laufen lassen**

Run:
```bash
.venv/bin/python -m pytest tests/test_pi_agent.py tests/test_pi_queue.py \
  tests/test_pi_jobs.py tests/test_duel.py tests/test_devices_registry.py \
  tests/test_vision_captioner.py tests/test_pi_task_refactor.py \
  tests/test_pi_endpoint_override.py tests/test_pi_web_fetch.py \
  tests/test_pi_worker_bounded.py tests/test_image_ingestion.py -v
```
Expected: alle PASS. (Diese Tests importieren teils noch `src.agents` — sie laufen, weil `src/agents/` bis Task 4 existiert. Ein FAIL hier = echte API-Drift an einer Call-Site → die betroffene Signatur gegen das Paket prüfen und anpassen.)

- [ ] **Step 4: Commit**

```bash
git add src/
git commit -m "refactor(#266): swap src.agents imports → mayring_pi_agent (src/)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: tests/ swappen, src/agents löschen, Gate + volle Suite grün

**Files:**
- Modify: 13 Test-Dateien
- Delete: `src/agents/`

- [ ] **Step 1: Prefix-Swap über tests/**

```bash
cd /home/nileneb/Desktop/MayringCoder
grep -rl "src\.agents" tests/ --include="*.py" | \
  xargs sed -i 's/src\.agents/mayring_pi_agent/g'
```

- [ ] **Step 2: `src/agents/` löschen**

```bash
git rm -r src/agents/
```
Expected: 8 Dateien entfernt (`pi.py`, `pi_jobs.py`, `pi_queue.py`, `pi_worker.py`, `pi_server.py`, `vision.py`, `diff_history.py`, `__init__.py`).

- [ ] **Step 3: Grep-Gate-Test laufen lassen — jetzt PASS**

Run: `.venv/bin/python -m pytest tests/test_no_src_agents_imports.py -v`
Expected: PASS (keine Offender mehr).

- [ ] **Step 4: Volle Test-Suite laufen lassen**

Run: `.venv/bin/python -m pytest -q`
Expected: alle PASS. (Bei FAIL in einem `test_pi_*`/`test_duel`/`test_devices` → Signatur-Drift zwischen gelöschtem `src/agents` und Paket an dieser Stelle; Call-Site an die Paket-API anpassen.)

- [ ] **Step 5: Commit**

```bash
git add tests/ src/
git commit -m "refactor(#266): swap test imports, delete src/agents/, gate green

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: CI-Submodule-Checkout

**Files:**
- Modify: `.github/workflows/build-and-push.yml:34`

- [ ] **Step 1: checkout-Step um Submodule + Token erweitern**

In `.github/workflows/build-and-push.yml`, `- uses: actions/checkout@v5` ersetzen durch:
```yaml
      - uses: actions/checkout@v5
        with:
          submodules: true
          token: ${{ secrets.GH_PAT }}
```
(`GH_PAT` existiert bereits im Workflow — Voraussetzung: PAT hat Read auf `Nileneb/mayring-pi-agent`.)

- [ ] **Step 2: YAML-Validität prüfen**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/build-and-push.yml')); print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit + push (löst Build & Push + Deploy + Smoke aus)**

```bash
git add .github/workflows/build-and-push.yml
git commit -m "ci(#266): checkout vendor/mayring-pi-agent submodule with GH_PAT

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin master
```

---

## Task 6: Deploy-Verifikation + #266-Akzeptanz

- [ ] **Step 1: Build & Push abwarten (Submodule-Checkout darf nicht failen)**

Run: `gh run list --workflow=build-and-push.yml --limit 1`
Expected: `success`. (Bei FAIL am checkout → `GH_PAT`-Scope für `mayring-pi-agent` prüfen.)

- [ ] **Step 2: Post-deploy-Smoke abwarten (Pi-Task-Roundtrip)**

Run: `gh run list --workflow=post-deploy-smoke.yml --limit 1`
Expected: `success`. Falls kein Auto-Trigger nach Deploy: `gh workflow run post-deploy-smoke.yml` und Ergebnis prüfen.

- [ ] **Step 3: #266 schließen mit Verweis auf den Dedup**

```bash
gh issue close 266 -c "Duplikat eliminiert: MayringCoder konsumiert mayring-pi-agent als Submodule-Package (vendor/, 0.1.1), src/agents/ gelöscht, 22 Importe geswappt, Grep-Gate + volle Suite grün, Build/Smoke grün. Drift weg (eine Codebase)."
```

---

## Notes / Risiken (aus Spec)

- **API-Drift einzelner Funktionen** ist das größte Restrisiko — abgefangen durch Task 3/4 Safety-Net-Tests; bei rotem Test die konkrete Signatur gegen das Paket prüfen, nicht raten.
- **Atomarer Übergang:** Prod läuft bis Task-5-Deploy auf dem alten Image; der Swap wird erst mit dem neuen Image live.
- **Submodule-Pin-Disziplin:** künftige Pi-Agent-Releases brauchen einen Submodule-Bump (`cd vendor/mayring-pi-agent && git fetch && git checkout <tag> && cd ../.. && git add vendor/mayring-pi-agent`).
- **Nicht in Scope:** Full-HTTP-Boundary, MCP-als-Paket, `local_mcp.py`-Deprecation (#270).
