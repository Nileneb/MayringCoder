# Spec: Ampel-Notification-Center fertigstellen (#42 — der „echte Punkt")

**Datum:** 2026-06-22 · **Repos:** mayring-core (PR) + MayringCoder (api) + app.linn.games (UI, Direct-Push) · **Status:** 🟡 GEPLANT
**Verbunden mit:** Issue #42 (notification_state + classify, MERGED), [[project_notifications_ampel_center]], [[project_igio_session_coherence_audit]]

## Auslöser (gemessen 2026-06-22)
Eine CI-Failure-Notification für `app.linn.games` blieb dem User „IMMERNOCH rot" angezeigt,
obwohl die Live-CI längst grün war. Root-Cause-Analyse dieser Session:
- Der `tests`-Lauf auf `5746a3e` schlug in **Attempt 1** fehl (flaky `BlackholeTest`,
  random Teleport-Ziel → seitdem deterministisch geseedet, app.linn.games `f1d169c`).
  Das feuerte ein `workflow_run.conclusion=failure`-Event → GitHub-Issue #542 + ein
  `repo_ci`-hook_event (rot).
- **Attempt 2** bestand → finale Conclusion `success`, Issue #542 auto-closed. GitHub ist
  vollständig grün. Das **Rot war ein stale Point-in-Time-Event**, kein Live-Ausfall.

Damit ist #42 der „echte Punkt": ein roter CI-Event muss **automatisch grün werden**, sobald
ein späterer Erfolg für dieselbe (repo, workflow) ihn auflöst — ohne manuelles Ack, in JEDER
konsumierenden Oberfläche.

## Ist-Zustand (verifiziert, nicht angenommen)
Das Memory ([[project_notifications_ampel_center]]) sagt „Endpoint + App-Center + Deploy offen".
Das ist **veraltet** — gemessen am 2026-06-22:

| Baustein | Status | Beleg |
|---|---|---|
| `classify_notification` (deterministische Ampel) | ✅ live | `vendor/mayring-core/mayring_core/notifications.py`, core-Pin `a5a02ac` (v0.1.11-3) |
| `notification_state`-Tabelle (v23) + v24-Drop des toten `integration_notifications` | ✅ live | `store.py:1449`, `test_notifications.py` roundtrip |
| `GET /stats/notifications` (Ampel-Feed, repo→project-Join, seen/ack) | ✅ deployed | prod `https://mcp.linn.games/stats/notifications` → **401** (Auth, nicht 404 → Route existiert) |
| `POST /stats/notifications/ack` (seen/ack-Triage) | ✅ deployed | `dashboard.py:930` |
| `POST /stats/notifications/ingest` (Hook-A: dependabot/pull/issue) | ✅ deployed | `dashboard.py:977`, privileged-only |
| `_supersede_stale_reds` (Rot→Grün bei späterem Success) | ⚠️ teilweise | `dashboard.py:820` — read-time, **fenster-begrenzt** |
| App-Notification-Center (User-Oberfläche) | ❌ **fehlt** | kein Consumer von `/stats/notifications` in app.linn.games (`MayringStatsClient` deckt nur plugin/memory-observability) |

**Fazit:** Backend ist da. Zwei echte Lücken bleiben — und genau die erklären das „immer noch rot".

## Lücke 1 — Backend ohne Caller (Anti-Pattern „gebaut & nie verwendet")
Es gibt KEINE app-seitige Oberfläche, die `/stats/notifications` rendert. Der User „sieht rot"
also über GitHub/Harness, nicht über das gebaute Center. Das ist exakt das in
`v2-frustration-patterns.md` benannte Top-1-Muster („Feature gebaut UND DANN NIE VERWENDET").

### Ziel
Ein Livewire-Notification-Center in app.linn.games, das `/stats/notifications` als Ampel-Feed
zeigt, `open_red` als Badge führt, und Seen/Ack gegen `/stats/notifications/ack` schreibt.

### Ansatz
- **Service:** `MayringStatsClient::notifications(limit, only_open)` + `::ackNotification(id, seen, acked)`
  (gleiche JWT-Auth wie die bestehenden Stats-Calls; nginx-mcp.conf-Allowlist prüfen — die
  beiden `/stats/notifications*`-Pfade MÜSSEN in der Regex-location stehen, sonst 401-body=None,
  [[project_nginx_mcp_conf_sot]]).
- **Livewire:** `app/Livewire/Mayring/NotificationCenter.php` + Blade. Polling analog
  `PluginObservability` (`wire:poll.30s`). Ampel-Farben deterministisch aus `urgency`.
  **FALLE:** `urgency` ist ein String — kein truthy/falsy-Tint-Bug wie lang_tier 0
  ([[project_marktradar_globe_v2]]).
- **Eintrittspunkt:** Header-Badge mit `open_red`-Count → MUSS sichtbar verlinkt sein
  (kein default-disabled Panel; sonst wieder „kein Eintrittspunkt").

## Lücke 2 — Supersede ist read-time + fenster-begrenzt
`_supersede_stale_reds` mutiert nur die zurückgegebene `items`-Seite (LIMIT 50–500, `fired_at DESC`).
Folgen:
- Ein Rot, dessen auflösender Success **außerhalb der Seite** liegt (älter als das Fenster, oder
  Seite voller anderer Events), bleibt rot.
- Die Auflösung ist **nicht persistiert** — jeder Reader rechnet sie neu; ein Consumer, der das
  nicht tut (oder eine andere Query), sieht das alte Rot. Genau die Frikition dieser Session.
- **Kein Test** deckt Supersede ab (test_notifications.py prüft classify + state-roundtrip, nicht
  die Auflösung).

### Ziel
Auflösung deterministisch und vollständig — unabhängig vom Lese-Fenster.

### Ansatz (einer wählen in der Umsetzungs-Session)
1. **SQL-seitige Resolution (empfohlen):** „latest success per (norm_repo, workflow)" als
   Subquery/CTE über ALLE `repo_ci`-events berechnen, nicht nur die Seite; rote Events mit
   `fired_at < latest_success` direkt als `green/superseded` markieren. Korrekt unabhängig von LIMIT.
2. **Persistierte Resolution:** beim Ingest eines `success`-Events die offenen roten
   (gleiche norm_repo+workflow, älter) in `notification_state` als `acked/resolved` upserten
   (Producer-seitig in `/repo-events`). Vorteil: jeder Reader sieht es, ohne Logik zu duplizieren.
   Nachteil: Producer-Kopplung.
3. **Hybrid:** SQL-Resolution im Endpoint (1) + ein nightly-Backfill, der alte aufgelöste Rots
   in `notification_state.acked=1` schreibt (Feed bleibt schlank).

**Empfehlung:** Ansatz 1 (SQL-CTE) als Korrektheitsgrenze + Supersede-Test im core/api. Ansatz 2
nur, wenn ein zweiter Consumer (Harness) dieselbe Resolution ohne den Endpoint braucht.

## Falle
- **Repo-Format-Drift:** CI-Events kommen in zwei Repo-Formaten (`https://github.com/Owner/x` via
  repo-watch, `Owner/x` via Webhook). Resolution MUSS über `_norm_repo` matchen (schon in
  `_supersede_stale_reds` gefixt 2026-06-13) — beim SQL-Port nicht verlieren.
- **Smoke-Artefakte:** synthetische `smoke/repo-<ts>`-failures (`_is_smoke_repo`) NIE in den
  User-Feed ([[feedback_smoke_check_authoring]]). Beim SQL-Port der Resolution mitfiltern.
- **mayring-core via PR**, nicht Direct-Push ([[feedback_prod_change_discipline]]). Falls Lücke 2
  Core-Code berührt (store/ingest): PR + frischen Tag pinnen + Image neu bauen
  ([[feedback_tag_pin_freshness]]).
- **nginx-Allowlist:** neue/ack-Pfade in `app.linn.games/docker/mayring/nginx/mcp.conf`.

## Akzeptanz (an den realen Vorfall gebunden)
1. **Auto-Resolve:** Zwei `repo_ci`-Events für (`app.linn.games`, `tests`) — erst `failure`
   (fired_at T0), dann `success` (T1>T0) — liefern im Feed `urgency=green, superseded=true` für
   das alte Rot, `open_red=0`, **ohne** manuelles Ack. Auch wenn das Success-Event außerhalb der
   ersten LIMIT-Seite läge (Test mit kleinem LIMIT + Distraktor-Events).
2. **App-Center sichtbar:** app.linn.games rendert das Center, Header-Badge zeigt `open_red`,
   ein echter offener roter Event ist klickbar (URL → GitHub-Run), Ack setzt ihn auf seen/acked
   und der Badge dekrementiert (Screenshot-Verifikation, [[feedback_validate_frontend_screenshots]]).
3. **Kein Smoke-Leak:** ein Smoke-Lauf erzeugt KEINE sichtbare rote User-Notification.
4. Tests: core/api Supersede-Unit (LIMIT-unabhängig, Format-Drift, Smoke-Filter) grün +
   ein Live-Smoke gegen prod.

## Nicht-Ziele
- Kein LLM in der Klassifikation — die Payload trägt das Signal (conclusion/severity),
  deterministisch ist billiger und verlässlicher (Designentscheidung in `notifications.py`).
- Keine Reaktivierung des toten `trigger_stats`/`integration_notifications`-Pfads (v24 dropped).

## Verwandt
[[project_notifications_ampel_center]] (Vorzustand), [[project_device_hook_events_pipeline]]
(Producer-Pipeline), [[project_nginx_mcp_conf_sot]] (Allowlist-Falle),
[[feedback_smoke_check_authoring]] (Smoke-Artefakte).
