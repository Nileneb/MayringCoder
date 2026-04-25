# MayringCoder Memory-Workflow

MCP-Server: `mcp.linn.games/sse` | Workspace-ID: `system` (intern) oder Projekt-Slug

## 1. Sessionbeginn / neuer Task
```
mcp__claude_ai_Memory__search_memory(query="<aktueller Task>", workspace_id="<slug>")
```
Die zurückgegebenen `chunk_id`s für Feedback merken.

## 2. Nach /compact
```
mcp__claude_ai_Memory__search_memory(query="<Task>", workspace_id="<slug>", compacted=True)
```

## 3. Chunk-Feedback (nach jedem abgeschlossenen Task)
- Hilfreich → `mcp__claude_ai_Memory__feedback(chunk_id="...", signal="positive", metadata={"task":"..."})`
- Irrelevant → `mcp__claude_ai_Memory__feedback(chunk_id="...", signal="negative")`

## 4. Session-Zusammenfassung speichern (Sessionende / nach Großaufgabe)
Ersetzt den Docker-Watcher. Zusammenfassung direkt übergeben, kein Ollama nötig:
```
mcp__claude_ai_Memory__conversation_ingest(
  turns=[{"role":"assistant","content":"<was wurde entschieden/gebaut>","timestamp":"<ISO>"}],
  session_id="<YYYY-MM-DD-kurzbeschreibung>",
  workspace_slug="<slug>",
  presumarized="<2-3 Satz Zusammenfassung: Was wurde gemacht, welche Entscheidungen, welche Dateien>"
)
```

## 5. Einzelne Erkenntnis direkt speichern
```
mcp__claude_ai_Memory__put(content="<Erkenntnis>", source_id="<kontext:datei>", workspace_id="<slug>")
```

## 6. Aufgaben an Pi-Agent delegieren

Für rechenintensive oder memory-gestützte Aufgaben den Pi-Agent nutzen:
```
mcp__claude_ai_Memory__pi_task(
  task="<Aufgabenbeschreibung>",
  repo_slug="<repo-slug>",
  timeout=180.0
)
```

**Wann pi_task nutzen:**
- Code-Analyse einzelner Dateien oder Module
- PICO-Suchterme aus Projektkontext entwickeln
- Konventionen und Muster aus dem Memory abrufen und anwenden
- Zusammenfassungen mit vollständigem Projektkontext erstellen

**Wann NICHT pi_task:**
- Architekturentscheidungen (Claude selbst hat breiteren Kontext)
- Multi-File-Refactoring (braucht Claudes direkten Datei-Zugriff)
- Wenn kein Ollama lokal läuft (pi_task schlägt mit `{"error": "..."}` fehl — dann selbst antworten)

**Hinweis:** pi_task läuft über `pi_server.py` (Port 8091, lokal) und nutzt Ollama + Memory.
Wenn `{"error": "..."}` zurückkommt, ist entweder Ollama oder pi_server.py nicht gestartet.
