# Workspace Re-Point Migration — 019d6933 → app.linn.games-SoT

**Datum:** 2026-05-24
**Status:** Spec — wartet auf User-Approval vor Execution (User-Entscheid: „Inventur + Spec, dann re-point")
**Verbunden mit:** v2-workspaces-spec, #104/#225, [[project_mayringcoder_workspace_model]]

## Root Cause

app.linn.games wurde neu aufgesetzt: bene@linn.games ist jetzt **user id=1** mit
Workspace **`019e14d6`** "Bene Workspace" (+ 2 Org-Workspaces). Die alte Identität
(**user sub=2**, workspace **`019d6933`**) lebt nur noch in MayringCoder + in
zirkulierenden Tokens. → Neue JWTs zeigen leer, alte sehen Daten, Dashboard fällt
auf `system` zurück. Das `system` im Dashboard ist ein Symptom dieses Drifts.

## Inventur (Prod, 2026-05-24)

**59.566 Zeilen unter `019d6933`** über 13 Tabellen: ingestion_log 18.898,
llm_calls_log 17.800, chunk_source_refs 7.119, **chunks 6.650**, context_feedback_log
4.428, sources 3.590, chunk_feedback 969, research_questions 40, hook_events 54,
projects 8, devices 2, topic_transitions 6, workspace_aliases 2.
- **Chroma** `memory_chunks`: 6.650 Vektoren mit `workspace_id=019d6933` in den Metadaten.
- **user_id-Drift minimal:** chunks.user_id durchweg NULL; nur **15 sources**-Zeilen `user_id='2'`.
- Unangetastet: `system` (2.132 chunks, Service-Bucket), `bene:logs` (64), `public` (8).
- 8 Projekte unter 019d6933 (mayringcoder, app-linn-games, battlefield, dronedetect,
  linn-games-research, DiakonieWhisper, logs, +Router-Dup) → bleiben als Sub-Dimension.

## Ziel (User-Entscheid)

**Alles `019d6933` → `019e14d6` (Bene Workspace).** project_id-Dimension bleibt erhalten
(trennt weiterhin die Projekte INNERHALB der persönlichen Workspace). Org-Split
(DiakonieWhisper → Bergische-Diakonie) = späteres, separates project→workspace-Reassignment.

## Migration (geordnet)

1. **Backup:** `tools/migrate_workspace_repoint.py` kopiert `memory.db` → `.bak-<ts>` vor
   jedem Write. Chroma-Snapshot manuell empfohlen (Volume `linn-mayring-cache`).
2. **Dry-run** (read-only): Skript ohne APPLY → zeigt exakte Re-Point-Counts. Review.
3. **Apply** (`APPLY=1`), eine SQLite-Transaktion:
   - `workspaces`-Row für `019e14d6` anlegen (kind=user, falls fehlt).
   - `UPDATE workspace_id 019d6933→019e14d6` über alle 13 Tabellen.
   - `UPDATE sources.user_id 2→1` (15 Zeilen).
   - `workspace_aliases(019d6933 → 019e14d6, created_at)` registrieren.
   - **Chroma:** `memory_chunks` Metadaten-Update der 6.650 Vektoren (batched, 256er).
4. **Dashboard-Scoping-Fix** (app.linn.games, separater Commit): die user-scoped
   Livewire-Komponenten (`MemoryObservability`, `MemoryDashboard`, `PiAgentObservability`,
   `Recherche/MayringMemoryDashboard`) auf `->forWorkspace($user->activeWorkspaceId())`
   umstellen → scopen jetzt auf `019e14d6` (= Daten). `PluginObservability` bleibt
   bewusst cross-workspace (Device-Registry). Kein `system` mehr für persönliche Sicht.
5. **Token-Reissue:** alte API-Tokens (`019d6933`) brechen — `resolve_workspace_from_token`
   konsultiert Aliases NICHT (nur der CLI-6-Schritt-Resolver tut's). hook.jwt neu minten
   (app.linn.games refresh-token / re-login); Watcher refresht selbst.

## Verifikation

- Service-Token + `X-Workspace-Id: 019e14d6` → `/memory/search` liefert Treffer (Daten da).
- `019d6933` → leer (bis auf was system/public hält).
- Dashboard (nach Scoping-Fix) zeigt **Bene Workspace**, nicht `system`.

## Rollback

- SQLite: `cp memory.db.bak-<ts> memory.db` (Container-Restart lädt sie).
- Chroma: Reverse-Metadaten-Update `019e14d6→019d6933` (gleiches Skript, Richtung getauscht)
  ODER Volume-Snapshot zurückspielen.
- Alle Schritte reversibel; nichts wird gelöscht.

## Risiken / Caveats

- **Chroma-Metadaten-Update ist der riskanteste Schritt** (6.650 Vektoren, kein Transaktions-
  Schutz wie SQLite). Bei Fehler: SQLite ist schon committed → nur der Chroma-Block muss
  re-run (idempotent: `where workspace_id=OLD` findet dann 0).
- Alte API-Tokens brechen (s. Schritt 5) — bewusst, da Identitäts-Korrektur.
- `system`/`public`/`bene:logs` bleiben unberührt (kein Cross-Tenant-Move).

## Deferred (nicht in dieser Migration)

- `resolve_workspace_from_token` alias-aware machen (transparente Alt-Token-Kompat) —
  braucht conn-Zugriff im API-Pfad, separater Change.
- Org-Split: DiakonieWhisper-Projekt → Bergische-Diakonie-Workspace.
- Router-Dup-Projekt (`d30af420` "mayringcoder") + slug-`mayringcoder` dedupen.
