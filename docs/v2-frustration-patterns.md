# Frustration-Pattern aus User-Quotes

**Quelle:** Grep auf 38MB-Transcript. **119× DUMM**, **815× scheiße**, **39× kotz/ärger/frust** in User-messages.

## Top-7 wiederkehrende Anti-Patterns die User explizit benennt

### 1. "Feature gebaut aber nie verwendet"
> "schon wieder ein feature, das gebaut UND DANN NIE VERWENDET WURDE!" (line 6611)
> "DEINE ARBEIT HAT KEIN OUTCOME" (line 3639)

**Symptom:** Code im Repo, Tests grün, aber kein Caller / kein UI-Eintrittspunkt / default-disabled.

### 2. "Halb-fertige Arbeit / Workflow nicht durchgezogen"
> "WAS NUTZT MIR DIE GANZE SCHEISS APP WENN DER PAPER DOWNLOAD IMMERNOCH NICHT RICHTIG GEHT" (line 9303)
> "WENN IN EINER AUFEINANDER AUFBAUENDEN LOGIK DIE HÄLFTE WEGGELASSEN WIRD" (line 9303)
> "Wäre deine Arbbeit VOLLSTÄNDIG, wenn du den nächsten schritt weglässt???" (line 10228)

**Symptom:** Pipeline besteht aus Step 1 → 2 → 3 → 4. Ich fixe Step 2, lasse Step 3+4 ungetestet. User entdeckt das später.

### 3. "Silent failure — Error stumm geschaltet statt behoben"
> "OFT EINFACH GEMUTET, STATT BEHOBEN!!! DAS IST SO DUMM!!!" (line 4222)
> "MACH DOCH BIG BANG UND LASS ALLE ERRORS KRACHEND SCHEITERN STATT SILENT" (line 8566)

**Symptom:** `try: ... except Exception: pass` oder `return None` bei error oder offline-fallback ohne sichtbarkeit.

### 4. "Manueller Step in einer Auto-Pipeline"
> "WANN HABE ICH JEMALS MANUELLE COPY-PASTE TOKEN SCHEISSE GEWÜNSCHT, DASS ICH DAS IMMER WIEDER MACHEN MUSS???" (line 3762)
> "WARUM werden EINGESCHLOSSENE Papers nicht AUTOMATISCH herunter geladen?? das ist doch wieder DUMM!" (line 7690)

**Symptom:** Eine 80%-automatisierte Pipeline, aber 1 manueller Schritt mittendrin. User hat 100% erwartet.

### 5. "Default-disabled Feature nach deploy"
> "WARUM SCHON WIEDER DINGE IMPLEMENTIEREN UND DANN AUSSCHALTEN IM BETRIEB?" (line 2794)
> "WELCHE FUNKTIONEN SIND NOCH DEFAULT AUSGESCHALTET?" (line 2816)

**Symptom:** Feature deployed mit `if not env('FEATURE_X'): return` — User muss das erstmal aktivieren.

### 6. "Test fake / dauerhaft scheiternder Test"
> "EIN DAUERHAFT SCHEITERNDER TEST IST DAS ALLER DÜMMSTE" (line 6428)
> "ECHTE BEWEISE!!!!!!! ES KOTZT MIC SO AN, DASS ICH STÄNDIG DIE GLEICHE SCHEISSE SCHREIBEN MUSS" (line 3762)

**Symptom:** Smoke-test schlägt fail tagelang an — generiert email-spam, User sieht nicht mehr ob's ein NEUER bug ist. Plus: Tests die mocken statt real prüfen.

### 7. "Workspace=system statt user / Bug-Schleife"
> "WARUM bekomme ich dann IMMERNOCH in der job history angezeigt, dass ÜBERALL WORKSPACE SYSTEM VERWENDET WIRD" (line 10035)
> "Wir drehen uns jedes mal im kreis und machen JEDEN FIX eTWA ZEHNMAL" (line 4222)

**Symptom:** Multi-Tenant-Feature gefixt, aber EINE ingestpfad-Stelle blieb übersehen → Daten landen weiter im falschen Bucket. Erst beim 6. Iteration komplett.

## Was User explizit in dieser Session als Korrektur gesagt hat

> "kein quick-fix, der mehr probleme erschafft, als löst, weil er wieder nur ein EINZELSYMPTOM fixt, aber die pipeline zerstört, weil es an anderer stelle silent bricht" (line 16017)

> "FINDE erst alle GOALS die wir hatten, dann prüfe welche INTERVENTIONEN (git diffs??!!) wir alle durchgeführt haben. WAS war der OUTCOME? UND DANN ERST: was muss geändert werden" (line 16322)

> "ARBEITE LIEBER LÄNGER, als mir jetzt in fünf minuten einen schnellschuss abzugeben" (line 16322)

## Architektur-Implikationen

Diese 7 Patterns sind nicht 7 separate Bugs — sie haben gemeinsame strukturelle Wurzeln:

1. **Fehlende End-to-End-Contract-Tests** decken halb-fertige pipelines auf (→ #2, #4, #7)
2. **Kein Default-On-Policy** für deployed features (→ #5)
3. **Silent-fallback-Verbot** als Code-Review-Regel (→ #3)
4. **"Feature-doneness"-Definition** mit acceptance-criteria PR-Required (→ #1)
5. **Smoke-test-Stabilität** (jeder fail muss innerhalb 1h fixed/closed werden, sonst alarm-fatigue) (→ #6)
6. **Bug-Schleife-Heuristik:** wenn das gleiche Symptom 2× gefixt wurde, das nächste Mal vorab-Audit statt nächster Patch (→ #7)
