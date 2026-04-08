# CLAUDE.md

## Zweck

Dieses Repository entwickelt eine **MCP-basierte, lokale Memory-Architektur für Claude Code** auf Basis von **MayringCoder**.

**Source of truth:** `Target-Architecture.md`

## Projektziel

Claude Code soll persistentes, extern gespeichertes Memory über MCP nutzen statt Wissen nur im Prompt-Kontext zu halten.

Zielzustand:

- lokaler Betrieb
- strukturiertes und versioniertes Memory
- Hybrid Retrieval aus Filter + Embedding + Re-Ranking
- inkrementelle Reingestion bei Quelländerungen
- spätere Trainings- und UI-Erweiterung

## Bereits vorhanden

Bestehende Architekturbausteine, die weiterverwendet werden sollen:

- Ollama für lokale Modelle
- Overview-Analyse als Projektinventar
- SQLite Snapshot Cache für Änderungsdetektion
- ChromaDB / Embedding Retrieval
- finding-reactive RAG als bestehendes Retrieval-Muster
- opt-in JSONL Logging im LLM-Call-Loop

Bereits in der VS Code IDE vorhanden:

- MCP `sequentialthinking`
- MCP `memory`

## Harte Regeln

- Nur im aktuell geöffneten Projektverzeichnis arbeiten.
- Keine Annahmen über andere Repositories treffen oder dort Dateien anlegen.
- `Target-Architecture.md` ist die maßgebliche Architekturdefinition.
- Änderungen lokal, inkrementell und rückwärtskompatibel halten, wenn kein bewusster Bruch verlangt ist.
- Kein Cloud-Zwang einführen, wenn es lokal lösbar ist.
- Keine Architektur bauen, die primär auf größerem Kontextfenster basiert.
- **Kategorie ist Metadatum, nicht Primäridentität eines Chunks.**
- **Induktive Mayring-Kategorisierung verfeinert Chunking, ersetzt es nicht vollständig.**

## Architekturentscheidungen

### Chunking

Verbindliche Reihenfolge:

1. strukturelles Chunking
2. optionale semantische Verfeinerung
3. Kategorien als Metadaten speichern
4. Embeddings auf finalen Chunks erzeugen

Nicht umsetzen:

- rein kategoriales `Kategorie -> Chunk` als einziges Datenmodell
- direkte Speicherung von Memory ohne Quellreferenz, Hash oder Version

## Bevorzugte Speicherarchitektur

Memory besteht aus mehreren Ebenen:

- **KV/Cache** für direkte und schnelle Zugriffe
- **SQLite** für Metadaten, Versionen, Status, Feedback
- **ChromaDB** für semantisches Retrieval
- **Artifact Storage** für Snapshots, Exporte, Trainingsdaten

## Bevorzugte Implementierungsreihenfolge

1. MCP Tool-Verträge definieren
2. Source-/Chunk-Datenmodell implementieren
3. Ingestion-Pipeline ergänzen
4. Retrieval-Policy koppeln
5. Versionierung und Dedup ergänzen
6. Logging/Feedback integrieren
7. UI erst danach

## MCP Tooling

Wenn neue MCP-Tools implementiert werden, diese Form bevorzugen:

- `memory.put`
- `memory.get`
- `memory.search`
- `memory.update`
- `memory.invalidate`
- `memory.list_by_source`
- `memory.explain`
- `memory.reindex`
- `memory.feedback`

## Bestehende Codebasis bevorzugt wiederverwenden

Vorhandene Module zuerst prüfen, bevor neue Parallelstrukturen entstehen:

- `cache.py` / bestehender SQLite-Diff-Ansatz
- `context.py` / Overview, Inventory, ChromaDB, Retrieval
- `analyzer.py` / Ollama-Aufrufe, Logging
- `checker.py` / Pipeline-Orchestrierung

Wenn neue Module nötig sind, sie klar entlang der Zielarchitektur trennen, z. B.:

- `memory_schema.py`
- `memory_store.py`
- `memory_ingest.py`
- `memory_retrieval.py`
- `mcp_server.py`

## Arbeitsmodus für Claude

### Für Architektur- oder Refactoring-Aufgaben

- zuerst Abhängigkeiten und Seiteneffekte durchdenken
- bevorzugt `sequentialthinking` nutzen bei:
  - Datenmodellen
  - Retrieval-Policies
  - Migrationspfaden
  - Refactorings über mehrere Module

### Für stabile Projektentscheidungen

- vorhandenes MCP `memory` nutzen, um langlebige Architekturentscheidungen, Annahmen und Konventionen wiederauffindbar zu halten
- nur stabile, projektweit relevante Informationen speichern
- keine flüchtigen Zwischenschritte als dauerhaftes Memory behandeln

## Qualitätsmaßstab

Eine gute Änderung erfüllt möglichst viele dieser Punkte:

- lokal ausführbar
- klein und testbar
- bestehende Pipeline nicht unnötig brechen
- Quelle, Version und Hash bleiben nachvollziehbar
- Retrieval bleibt erklärbar
- Logs und Metadaten sind für spätere Auswertung nutzbar

## Bei Implementierungen mitliefern

Wenn sinnvoll, zusätzlich erstellen oder aktualisieren:

- kurze technische Doku
- CLI-Beispiele
- einfache Tests für Datenmodell, Dedup, Retrieval oder Migration
- Migrationshinweise bei Schemaänderungen

## Vermeiden

- doppelte Cache- oder Retrieval-Logik neben bestehenden Modulen
- irreversibles Überschreiben von Daten ohne Versionierung
- globale Architekturwechsel ohne dokumentierten Migrationspfad
- unnötige Abstraktionen ohne konkreten Nutzen
- Fülltext in Architektur- oder Technikdokumenten

## Default-Annahme bei Unklarheit

Wenn eine Entscheidung nicht explizit festgelegt ist, bevorzuge:

- lokal statt extern
- einfach statt generisch
- versioniert statt implizit
- hybride Retrieval-Architektur statt nur Vektor-Suche
- Erweiterung bestehender Module statt kompletter Parallelimplementierung
- Dokumentiere am Ende einer Todo-Liste kurz und prägnant die umgesetzten Änderungen. Ebenso, wenn   Punkte offen geblieben sind und ncoh weiter bearbeitet werden müssen.

## Compact Instructions

Bei `/compact` folgende Informationen erhalten (werden für Memory-Retrieval nach Kompaktierung benötigt):

- **Architektur:** Zielzustand aus `Target-Architecture.md` + aktive Entscheidungen aus `CLAUDE.md`
- **Offene Tasks:** Akzeptanzkriterien aus Issue #27 (Nileneb/MayringCoder#27)
- **Editierte Module (diese Session):** Dateinamen und geänderte Funktionen
- **Aktive MCP-Tool-Verträge:** `memory.put`, `memory.get`, `memory.search_memory`,
  `memory.invalidate`, `memory.list_by_source`, `memory.explain`, `memory.reindex`, `memory.feedback`
- **Chunking-Invariante:** Strukturelles Chunking zuerst, Mayring-Kategorisierung als semantische Schicht danach
- **Codebook-Auto-Erkennung:** `repo_file`/`note` → `codebook.yaml`; `conversation`/`conversation_summary` → `codebook_sozialforschung.yaml`

Compact-Zusammenfassungen können via `mcp__memory__put` (source_type: conversation_summary) ins Memory gespeichert werden, damit sie bei zukünftigen Sessions per `mcp__memory__search_memory` abrufbar sind.