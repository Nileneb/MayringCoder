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
