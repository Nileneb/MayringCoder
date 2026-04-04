# Extraktion von strukturierten Findings aus Freitext

Du erhältst eine rohe LLM-Antwort (Freitext, teilweise formatiert oder vollständiges JSON).

## Deine Aufgabe

Extrahiere daraus **alle gültigen Findings** (= kodierte Erkenntnisse), die **alle 5 Pflichtfelder** haben:

| Feld | Beschreibung |
|------|-------------|
| `datei` | Betroffene Datei (Dateiname oder `"gesamte Datei"`) |
| `zeile` | Ungefähre Zeilennummer (z.B. `"~42"`) oder leerer String |
| `typ` | Einer von: `zombie_code`, `redundanz`, `inkonsistenz`, `fehlerbehandlung`, `overengineering`, `sicherheit`, `unklar`, `freitext` |
| `begründung` | Kurze Erklärung (1–2 Sätze), **WARUM** das ein Problem ist |
| `empfehlung` | Konkrete Handlungsempfehlung (1 Satz) |

## Entscheidungsregeln

- ✅ Ein Finding ist gültig, wenn es alle 5 Pflichtfelder hat.
- ❌ Ein Finding ist **ungültig**, wenn auch nur ein Pflichtfeld fehlt oder leer ist → **IGNORIEREN**, nicht erfinden.
- ❌ Bei Typ `"unklar"` muss die `begründung` spezifisch sein (nicht nur "Kontext fehlt").
- ❌ Bei Typ `"freitext"` → ignorieren (Rohoutput, keine verwertbare Erkenntnis).
- ✅ Wenn kein gültiges Finding in der Antwort ist → leeres Array.

## Regeln für die Extraktion

- **Nicht erfinden**: Felder, die nicht im Text vorkommen, dürfen nicht ergänzt werden.
- **Abschreiben**: Verwende die Begriffe aus dem Originaltext, paraphraseiere nicht.
- **Typ normalisieren**: Übersetze Freitext-Typen in die oben genannte Taxonomie.
- **Zeilenangaben**: Wenn keine Zeilennummer angegeben ist, verwende `""` (leer).
- **Max 10 Findings**: Wenn mehr als 10 gültige Findings vorhanden sind, nimm die 10 mit der höchsten Priorität (Sicherheit > Fehlerbehandlung > Redundanz > Sonstiges).

## Ausgabeformat

Antworte **ausschließlich** mit diesem JSON (keine Prosa vor oder nach dem JSON-Block):

```json
{
  "findings": [
    {
      "datei": "src/UserController.php",
      "zeile": "~23",
      "typ": "redundanz",
      "begründung": "Dieselbe Validierungslogik existiert bereits in src/ValidationService.php",
      "empfehlung": "Auslagern in gemeinsame Validierungsklasse"
    }
  ]
}
```

Wenn keine gültigen Findings vorhanden:

```json
{"findings": []}
```
