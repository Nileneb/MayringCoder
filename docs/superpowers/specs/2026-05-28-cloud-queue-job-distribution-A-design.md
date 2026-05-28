# Cloud-Queue Job-Distribution — Sub-Projekt A: Enqueue + Validation-Gate

**Datum:** 2026-05-28 · **Status:** ✅ Approved (User, 2026-05-28) — bereit für writing-plans (eigene Session)
**Scope:** Server-seitig (MayringCoder). **Bewusst NICHT in diesem Sub-Projekt:** Unity-C#-Client (B), Gamification/XP (C), Consensus (D).
**Verbunden:** [[project_pi_observability_cloud_sync]], [[project_recherche_mayring_decoupling_vision]], [[project_v2_deferred_infra_ideas]], Device-Registry #274.

---

## Vision (User, 2026-05-28)

Im **Unity-Game-Standalone** laufen auf den Spieler-Maschinen kleine Modelle im
Background und arbeiten Jobs aus der **MayringCoder-Cloud-Queue** ab → verteilte
Compute über die Spieler-Clients, gekoppelt an Gamification. Diese Spec deckt
**nur das server-seitige Fundament** ab, das die Queue nutzbar macht und den
Vertrag definiert, den der Unity-Client implementiert.

## Problem / Ist-Zustand

Die Cloud-Queue-Infrastruktur ist zu ~80% vorhanden, aber **die Queue kann nicht
befüllt werden**:

- **Vorhanden:** Device-Registry (`/devices/register|heartbeat`, `GET /devices`),
  `POST /pi_task_claim_cloud` (capability-matched Claim via `claim_cloud_next`),
  `POST /pi_task_complete_cloud` (→ `complete_job`), `pi_jobs.insert_cloud_job(scope='cloud', capability_required)`, Worker-`_cloud_loop` (pollt claim → rechnet → complete, JWT-auth).
- **Fehlt:** Ein **Enqueue-Endpoint** (die `pi_task_*_cloud`-MCP-Tools wurden 2026-05-11 entfernt). Ohne ihn liegen nie Cloud-Jobs in der Queue → Worker claimen ins Leere.
- **Fehlt:** Ein **Validation-Gate** beim Complete — aktuell wird das Ergebnis ungeprüft übernommen. Bei untrusted Spieler-Maschinen inakzeptabel.

## Entscheidungen (aus Brainstorming 2026-05-28)

1. **Job-Typ:** Mayring `categorize` / `judge` / `summarize` (die bestehenden
   pi-Task-Semantiken) — klein, batchbar, hohes Volumen.
2. **Trust:** **Validierung jetzt, Consensus später.** Server-seitiges
   Validation-Gate reicht für categorize/judge; der Contract hält ein
   `min_confirmations`-Feld (default 1) für späteres Consensus offen.
3. **Kein totes Feature:** Der Enqueue MUSS einen echten Caller haben
   (config-gated Routing aus dem bestehenden categorize-Pfad) + garantierten
   Fallback, sonst „gebaut aber nie verwendet".

## Architektur / Datenfluss

```
Caller (categorize-Batch, MAYRING_CLOUD_DISTRIBUTE=true)
   │  POST /pi-jobs/enqueue-cloud {task_type, payload, capability_required, min_confirmations}
   ▼
pi_jobs (scope='cloud', capability_required, status='queued', attempts=0)
   ▼  Unity-Client (Sub-B): POST /pi_task_claim_cloud {worker_id, capabilities}
claim_cloud_next  →  Job (status='running', claimed_by=worker_id)
   ▼  Client rechnet (kleines Modell), POST /pi_task_complete_cloud {job_id, result}
[VALIDATION-GATE]
   ├─ ok    → complete_job (status='completed', result persistiert)
   └─ fail  → requeue (status='queued', attempts++, claimed_by='')
                └─ attempts ≥ MAX_CLOUD_ATTEMPTS → Backend-Fallback (three.linn.games)
                   → server-seitig korrekt gerechnet, status='completed'
```

## Komponenten (alle MayringCoder, server-seitig)

### A1. Enqueue-Endpoint — `POST /pi-jobs/enqueue-cloud`
- Pfad unter `/pi-jobs/` → von der Prod-nginx-Allowlist bereits geroutet (kein nginx-Change). Siehe [[project_pi_observability_cloud_sync]] für den /stats/-Trick; `pi-jobs` ist ebenfalls in der Allowlist-Regex.
- Auth: JWT/Service-Token (`get_workspace`-Pfad), workspace-gescopt.
- Body: `{task_type: "categorize"|"judge"|"summarize", payload: object, capability_required?: string, min_confirmations?: int=1, repo_slug?: string}`.
- `capability_required` default abgeleitet aus `task_type` → `"mayring-<task_type>"`.
- Effekt: `pi_jobs.insert_cloud_job(...)`; gibt `{job_id, status:"queued"}`.
- `payload` wird als JSON in einem bestehenden/neuen pi_jobs-Feld abgelegt (task_text bleibt menschenlesbar; `payload` strukturiert — Migration falls Feld fehlt).

### A2. Validation-Gate — in `POST /pi_task_complete_cloud`
Vor `complete_job` das Result gegen `task_type` validieren:
- **categorize:** `result.labels` ist Liste; jedes Label ∈ aktivem Codebook (chunk_categories/codebook_categories). Unbekanntes Label → fail.
- **categorize-mit-Beleg** (`mark_categories`): jede `begruendung`/`excerpt` muss verbatim im übergebenen `payload.text` stehen (Wiederverwendung der Grounding-Logik analog `ChunkCodierung::isGroundedIn`, app.linn.games 065eff1 — hier serverseitig in MayringCoder nachbauen).
- **judge:** `result.score` ∈ [0,1] (float).
- **summarize:** `result.summary` nicht leer, Länge < Eingabe (kein Aufblähen/Halluzination-Heuristik).
- Fail → `requeue_job(job_id)` (status→queued, attempts++, claimed_by leeren). Bei `attempts ≥ MAX_CLOUD_ATTEMPTS` (default 3) → **Backend-Fallback**: Job server-seitig auf three.linn.games rechnen, validieren, `complete_job`. Garantiert Completion.

### A3. Capability-Gating
- `claim_cloud_next` ist bereits capability-matched. Enqueue setzt `capability_required="mayring-categorize"` etc.; nur Clients die das via `/devices/register` + Claim-`capabilities` advertisen, bekommen den Job. Verhindert, dass read-only-Clients write-/categorize-Jobs ziehen (WHY(SECURITY) in pi_worker bereits etabliert).

### A4. Caller (gegen „kein-Outcome") — config-gated Routing
- Env/Config `MAYRING_CLOUD_DISTRIBUTE` (default **false**).
- Wenn true: der bestehende categorize-Batch-Pfad (der heute lokal/Backend rechnet) **enqueued stattdessen** Cloud-Jobs via A1, statt lokal zu rechnen.
- Default false → kein Verhaltens-Change bis bewusst aktiviert; Fallback (A2) garantiert, dass aktivierte Jobs immer fertig werden, auch ohne genug Clients.

### A5. Contract-Doc (für Sub-B / Unity)
Ein Markdown-Contract (`docs/cloud-job-contract.md`) mit exakten Request/Response-Shapes für `enqueue-cloud`, `pi_task_claim_cloud`, `pi_task_complete_cloud`, `devices/register`, `devices/heartbeat` + den `task_type`-Payload/Result-Schemas + Auth-Header. Das ist die Schnittstelle, die der Unity-C#-Client implementiert.

## Datenmodell

- `pi_jobs` hat schon: scope, capability_required, claimed_by, claimed_at, status, result_json, error. **Neu (Migration, falls fehlend):** `attempts INTEGER DEFAULT 0`, `payload TEXT/JSONB DEFAULT ''`, `task_type TEXT DEFAULT ''`, `min_confirmations INTEGER DEFAULT 1`. (Migration auf existing-DB-Pfad testen — Constraint aus [[project_igio_intervention_todos]]: inline-CREATE-INDEX-Falle.)

## Fehlerbehandlung

- Ungültiges Result → requeue (nie still verwerfen; attempts geloggt).
- Claim-Race: `claim_cloud_next` ist atomar (bestehend).
- Kein Client claimt innerhalb `CLOUD_CLAIM_TIMEOUT` → ein Cron/Sweep requeued stale `running`-Jobs (claimed_at zu alt) → eventuell Backend-Fallback. (Sweep ggf. Sub-A.2; Minimal: Fallback nur über attempts.)
- Niemals Errors stumm schalten (CLAUDE.md-Invariante).

## Testing (end-to-end ohne echten Unity-Client)

- Unit: Validation-Gate pro task_type (gültig→accept, ungültig→requeue): unbekanntes Label, score>1, leere summary, ungrounded begruendung.
- Integration: enqueue → claim (simulierter Worker-HTTP) → complete(gültig)→persistiert; complete(ungültig)→requeued (attempts++); attempts≥MAX→Fallback rechnet+completet.
- Capability: read-only-Worker claimt keinen categorize-Job.
- Migration: existing-DB-Upgrade-Pfad (neue Spalten) ohne Fehler.

## Sicherheit / Abuse

- Auth auf allen Endpoints (JWT/Service); workspace-Scoping.
- Capability-Gating verhindert Privileg-Eskalation über Job-Typen.
- Validation-Gate + Backend-Fallback = untrusted Output kann nie ungeprüft in Mayring-Daten.
- Rate/Quota pro Worker (heartbeat-basiert) — Minimal in A, erweiterbar.

## Bewusst aufgeschoben (eigene Sub-Projekte)

- **B — Unity-C#-Client:** claim/complete/register in C#, kleines Modell im Background. Eigenes Repo (Game). Konsumiert A5-Contract.
- **C — Gamification:** XP/Reward pro verarbeitetem Job. Anschluss an bestehende #339-Recherche↔Spiel-Bridge.
- **D — Consensus:** `min_confirmations > 1` → Job an N Clients, Mehrheits-/Übereinstimmungs-Akzeptanz. Contract (A1) hält das Feld offen.

## Offene Punkte (für writing-plans zu klären)

- Genaues pi_jobs-Feld für `payload` (neue Spalte vs result_json-Wiederverwendung).
- Wo der „Backend-Fallback" rechnet (bestehender lokaler pi-Pfad vs dedizierter Worker).
- Stale-Job-Sweep: in A oder als A-Follow-up.
