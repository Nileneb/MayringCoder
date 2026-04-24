# MayringCoder Memory Integration

## Arbeitsgedächtnis laden

Zu Sessionbeginn und bei neuen Tasks relevante Chunks laden:

```
mcp__claude_ai_Memory__search_memory(
  query="<aktueller Task oder Frage>",
  workspace_id="<dein Workspace-Slug>"
)
```

Die `chunk_id` jedes zurückgegebenen Ergebnisses ist für das Feedback nötig.

## Feedback nach Task-Abschluss

Nachdem ein Task abgeschlossen ist, bei dem Memory-Chunks geliefert wurden:

**Hilfreicher Chunk** (hat das Problem direkt beigetragen):
```
mcp__claude_ai_Memory__feedback(
  chunk_id="<chunk_id aus dem Suchergebnis>",
  signal="positive",
  metadata={"task": "<kurze Task-Beschreibung>"}
)
```

**Irrelevanter Chunk** (wurde geliefert, war aber nutzlos):
```
mcp__claude_ai_Memory__feedback(
  chunk_id="<chunk_id>",
  signal="negative"
)
```

Dieses Feedback fließt direkt ins Ranking ein — häufig positiv bewertete Chunks
erscheinen bei ähnlichen Queries künftig höher.
