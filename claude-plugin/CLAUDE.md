# MayringCoder Memory-Workflow

Memory-Server (`mcp.linn.games/sse`) wird über das **Claude.ai-Cloud-Profil**
des Users verbunden — nicht über dieses Plugin. Die `mcp__claude_ai_Memory__*`
Tools unten kommen also von dort, nicht aus `.mcp.json`.

Das Plugin selbst liefert ausschließlich den **lokalen** `memory-agents`
MCP-Server (Pi-Agent: `pi_task`, `ingest`, `duel`, `benchmark_tasks`),
plus die Hooks für Session-Lifecycle und Conversation-Capture.

Workspace-ID: `system` (intern) oder Projekt-Slug

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
```
mcp__claude_ai_Memory__ingest(
  source="<was wurde entschieden/gebaut — 2-3 Sätze: Entscheidungen, geänderte Dateien, Ergebnisse>",
  source_id="session:<YYYY-MM-DD-kurzbeschreibung>",
  workspace_id="<slug>"
)
```

## 5. Einzelne Erkenntnis direkt speichern
```
mcp__claude_ai_Memory__ingest(
  source="<Erkenntnis als vollständiger Satz>",
  source_id="context:<datei-oder-thema>",
  workspace_id="<slug>"
)
```

## 6. Pi-Agent Dispatch Policy (default-first)

**Regel: Bei jeder Coding-Aufgabe zuerst pi_task verwenden. Claude-eigene Subagents sind der Fallback, nicht der Default.**

```
mcp__plugin_mayring-coder_memory-agents__pi_task(
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
pi_task via MCP (Plugin `mayring-coder`, Server `memory-agents`) ruft `run_task_with_memory()` direkt auf — kein pi_server.py nötig.
Voraussetzung: Ollama läuft lokal (`http://localhost:11434`), PI_AGENT_URL=direct (Standardwert in local_mcp.py).

Verwandte Tools desselben Servers: `mcp__plugin_mayring-coder_memory-agents__ingest`, `__duel`, `__benchmark_tasks`.

`pi_server.py` (Port 8091) ist ein optionaler HTTP-Proxy für nicht-MCP-Clients — nicht für Claude Code nötig.

Wenn `{"error": "Ollama nicht erreichbar"}` → Ollama fehlt lokal.
Wenn Tool `mcp__plugin_mayring-coder_memory-agents__pi_task` nicht existiert → Plugin `mayring-coder@mayring-local` nicht enabled oder MCP-Server nicht gestartet (`claude plugin list`, dann `/reload-plugins` bzw. Restart).
