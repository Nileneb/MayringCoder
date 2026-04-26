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

## 6. Pi-Agent Dispatch Policy (default-first)

**Regel: Bei jeder Coding-Aufgabe zuerst pi_task verwenden. Claude-eigene Subagents sind der Fallback, nicht der Default.**

```
mcp__memory-agents__pi_task(
  task="<Aufgabenbeschreibung>",
  repo_slug="<repo-slug>",
  timeout=180.0
)
```

### pi_task verwenden für
- Konkrete Implementierungsaufgaben (Datei lesen, Funktion patchen, Bug fixen)
- "Find the file / locate the function / patch this bug"
- Iterative Loops: Tests laufen lassen → fixen → wiederholen
- Repo-/Konventionsfragen mit Memory-Kontext
- Code-Analyse, Refactors, TODOs abarbeiten

### pi_task NICHT verwenden für
- Reine Architektur-/Strategieentscheidungen ohne Code-Arbeit
- Multi-File-Refactoring das direkten Dateizugriff von Claude braucht
- Sensitive Secrets oder Credentials
- Wenn pi_task fehlschlägt (`{"error": "..."}`) — dann selbst antworten und Fehler nennen

### Wie pi_task funktioniert
pi_task via MCP (`memory-agents` Server) ruft `run_task_with_memory()` direkt auf — kein pi_server.py nötig.
Voraussetzung: Ollama läuft lokal (`http://localhost:11434`), PI_AGENT_URL=direct (Standardwert in local_mcp.py).

`pi_server.py` (Port 8091) ist ein optionaler HTTP-Proxy für nicht-MCP-Clients — nicht für Claude Code nötig.

Wenn `{"error": "Ollama nicht erreichbar"}` → Ollama fehlt lokal.
Wenn Tool `mcp__memory-agents__pi_task` nicht existiert → `memory-agents` MCP-Server nicht registriert (install.sh neu ausführen).
