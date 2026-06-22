# Spec: Ampel-Supersede härten (#42 — der „echte Punkt" am stale Rot)

**Datum:** 2026-06-22 · **Repos:** MayringCoder (api) + ggf. mayring-core (PR, falls Resolution in core wandert) · **Status:** 🟡 GEPLANT
**Verbunden mit:** Issue #42 (notification_state + classify, MERGED+DEPLOYED), [[project_notifications_ampel_center]], [[project_igio_session_coherence_audit]]

## Auslöser (gemessen 2026-06-22)
Eine CI-Failure-Notification für `app.linn.games` blieb dem User „IMMERNOCH rot", obwohl die
Live-CI längst grün war. Root-Cause dieser Session:
- Der `tests`-Lauf auf `5746a3e` schlug in **Attempt 1** fehl (flaky `BlackholeTest`, random
  Teleport-Ziel → seitdem deterministisch geseedet, app.linn.games `f1d169c`). Das feuerte
  `workflow_run.conclusion=failure` → GitHub-Issue #542 + ein rotes `repo_ci`-hook_event.
- **Attempt 2** bestand → finale Conclusion `success`, Issue #542 auto-closed. GitHub ist
  vollständig grün. Das **Rot war ein stale Point-in-Time-Event**, kein Live-Ausfall.

**Zwei Oberflächen NICHT verwechseln:**
1. **Harness/Claude-Code Repo-CI-Notification** (die `app.linn.games · tests failure 05:55`-Zeile)
   — GitHub-getrieben, Point-in-Time, aktualisiert sich NICHT bei grünem Re-Run. Das ist eine
   Claude-Code-Surface, **nicht** von #42 regiert. Hier ist nichts zu „fixen" außer Erwartung.
2. **In-App Ampel-Notification-Center** (#42) — soll stale Rot AUTOMATISCH grün zeigen, sobald
   ein späterer Erfolg es auflöst. Das ist der adressierbare „echte Punkt".

## Ist-Zustand (verifiziert gegen Code+Prod, NICHT angenommen)
Das Memory ([[project_notifications_ampel_center]]) listet viel „OFFEN", das real längst LEBT.
Gemessen 2026-06-22:

| Baustein | Status | Beleg |
|---|---|---|
| `classify_notification` (deterministische Ampel) | ✅ live | `vendor/mayring-core/mayring_core/notifications.py`, core-Pin `a5a02ac` (v0.1.11-3) |
| `notification_state`-Tabelle (v23) + v24-Drop `integration_notifications` | ✅ live | `store.py:1449`, `test_notifications.py` |
| `GET /stats/notifications`, `POST .../ack`, `POST .../ingest` | ✅ deployed | prod → **401** (Auth, nicht 404 → Routen existieren) |
| **App-Notification-Center** (Consumer + UI) | ✅ **deployed** | `MayringStatsClient::getAmpelNotifications`+`ackNotification`; konsumiert von `PluginObservability` UND `MemoryObservability`; Blades rendern Ampel-Dots + `open_red` (`memory-observability.blade.php:284`) |
| `_supersede_stale_reds` (Rot→Grün bei späterem Success) | ⚠️ **die einzige echte Lücke** | `dashboard.py:820` — read-time, **fenster-begrenzt** |

**Fazit:** #42 ist zu ~95 % fertig und live. Es bleibt GENAU EIN substanzieller Korrektheits-Defekt
— die Supersede-Logik —, der das „immer noch rot" im In-App-Center erklärt.

## Die echte Lücke — Supersede ist read-time + fenster-begrenzt
`_supersede_stale_reds` (dashboard.py:820) downgradet ein rotes CI-Event nur, wenn ein späterer
`success` für dieselbe (`_norm_repo`, workflow) **in derselben zurückgegebenen `items`-Seite**
liegt. Die App ruft `getAmpelNotifications(50)` → **LIMIT 50**. Daraus drei Defekte:
1. **Fenster-Lücke:** liegt der auflösende Success außerhalb der Top-50 (viele Events dazwischen,
   oder Success deutlich älter/jünger sortiert ans Seitenende), bleibt das Rot rot. → der reale
   „still red"-Pfad, sobald ein Repo > ~50 Events in `hook_events` hat.
2. **Nicht persistiert:** die Auflösung wird pro Read neu berechnet und nur in der Response
   mutiert. Ein zweiter Consumer mit anderer Query/Limit sieht das alte Rot.
3. **Ungetestet:** `test_notifications.py` deckt classify + state-roundtrip ab, NICHT Supersede
   (die Logik sitzt in `dashboard.py`, nicht in core → kein Unit-Schutz gegen Regression).

## Ziel
Auflösung deterministisch und vollständig — unabhängig vom Lese-Fenster und ohne Logik-Duplikat.

## Ansatz (einer wählen in der Umsetzungs-Session)
1. **SQL-seitige Resolution (empfohlen):** „latest success per (`norm_repo`, workflow)" als CTE
   über ALLE `repo_ci`-events (nicht nur die Seite) berechnen; rote Events mit
   `fired_at < latest_success` als `green/superseded` markieren — VOR dem LIMIT. Korrekt
   unabhängig von der Seitengröße. `_norm_repo` als SQL-Expr nachbilden ODER die Erst-Normierung
   beim Producer in `hook_events` ablegen (s. Falle Format-Drift).
2. **Persistierte Resolution (Producer-seitig):** beim Eintreffen eines `success`-Events
   (`/repo-events`) die offenen roten gleicher (`norm_repo`, workflow, älter) in
   `notification_state` als `acked=1`/`resolved` upserten. Jeder Reader sieht es ohne Logik-Kopie.
   Nachteil: Producer-Kopplung; mayring-core-Berührung → **PR** ([[feedback_prod_change_discipline]]).
3. **Hybrid:** SQL-Resolution im Endpoint (1) als Korrektheitsgrenze + nightly-Backfill, der alte
   aufgelöste Rots in `notification_state.acked=1` schreibt (Feed bleibt schlank).

**Empfehlung:** Ansatz 1 + ein Supersede-Test. Ansatz 2 nur, wenn ein zweiter Consumer (Harness)
dieselbe Resolution ohne den Endpoint braucht — aktuell nicht der Fall.

## Falle
- **Repo-Format-Drift:** CI-Events kommen in zwei Repo-Formaten (`https://github.com/Owner/x` via
  repo-watch, `Owner/x` via Webhook). Resolution MUSS über `_norm_repo` matchen (schon in
  `_supersede_stale_reds` gefixt 2026-06-13) — beim SQL-Port NICHT verlieren.
- **Smoke-Artefakte:** synthetische `smoke/repo-<ts>`-failures (`_is_smoke_repo`) gehören NIE in
  den User-Feed ([[feedback_smoke_check_authoring]]) — beim SQL-Port mitfiltern.
- **mayring-core via PR**, nicht Direct-Push; bei core-Berührung frischen Tag pinnen + Image neu
  bauen ([[feedback_tag_pin_freshness]]).
- **nginx-Allowlist** unverändert (Routen existieren bereits); nur prüfen, falls ein neuer Pfad
  entsteht ([[project_nginx_mcp_conf_sot]]).
- **NICHT** das App-Center neu bauen — es existiert. Nur den Server-Feed korrekt machen; die
  Blades lesen `urgency`/`superseded` bereits.

## Akzeptanz (an den realen Vorfall gebunden)
1. **Auto-Resolve LIMIT-unabhängig:** zwei `repo_ci`-Events für (`app.linn.games`, `tests`) —
   `failure` (T0) dann `success` (T1>T0) — plus ≥60 Distraktor-Events dazwischen. `/stats/notifications?limit=50`
   liefert für das alte Rot `urgency=green, superseded=true` und `open_red=0`. (Heute: bleibt rot,
   weil der Success aus der Top-50 fällt.)
2. **Format-Drift:** Failure als `https://github.com/Nileneb/app.linn.games`, Success als
   `Nileneb/app.linn.games` → trotzdem aufgelöst.
3. **Kein Smoke-Leak:** ein `smoke/repo-<ts>`-failure erzeugt KEINE sichtbare rote User-Notification.
4. **In-App grün:** `MemoryObservability`-Panel zeigt nach dem Re-Run-Success keinen roten Dot
   für `app.linn.games tests` mehr, `open_red`-Badge = 0 (Screenshot, [[feedback_validate_frontend_screenshots]]).
5. Tests: Supersede-Unit (LIMIT-unabhängig + Format-Drift + Smoke-Filter) grün + Live-Smoke gegen prod.

## Nicht-Ziele
- Die **Harness/GitHub-CI-Notification** auto-clearen — andere Surface, nicht #42.
- Kein LLM in der Klassifikation (Payload trägt das Signal; deterministisch ist billiger+verlässlicher).
- Kein Re-Build des App-Centers (existiert) und keine Reaktivierung von `trigger_stats`/`integration_notifications` (v24 dropped).

## Verwandt
[[project_notifications_ampel_center]] (Vorzustand — teils veraltet, hier korrigiert),
[[project_device_hook_events_pipeline]] (Producer-Pipeline), [[feedback_smoke_check_authoring]]
(Smoke-Artefakte), [[feedback_prod_change_discipline]] (core via PR).
