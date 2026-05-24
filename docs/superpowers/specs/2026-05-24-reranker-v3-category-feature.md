# Reranker-v3 — strukturiertes `cat_match`-Feature (category_id statt Free-String)

**Datum:** 2026-05-24
**Status:** Phase A implementiert (User-Entscheid: deterministischer Boost JETZT + Feature loggen)
**Tracking:** v2.0-Plan Phase 4, #270; baut auf Phase 3.2 (chunk_categories)

## Kontext

Der bestehende Category-Boost im Reranker (`_CAT_HINT_BOOST`) matcht die Query-Hints
gegen `chunk.category_labels` (freie Strings — fragil, tippfehleranfällig, kein
kanonisches Vokabular). Phase 3.2 liefert jetzt strukturierte `chunk_categories`-FKs
(category_id). v3 nutzt diese statt der Strings.

## Feature: `cat_match`

Für jeden Kandidaten: `1.0` wenn er über `chunk_categories` an eine Kategorie geknüpft
ist, die auch zur Query-Kategorie gehört, sonst `0.0`.

- **Query-Kategorien:** `category_hint`-Strings → `category_ids` (Lookup
  `codebook_categories WHERE lower(name) IN (...)`). Kein Re-Embedding im Hot-Path.
- **Kandidaten-Kategorien:** ein SQL-Load `chunk_categories WHERE chunk_id IN (...)` →
  `{chunk_id: set(category_id)}`.
- **Match:** `_category_id_match(query_cat_ids, chunk_cat_ids)` (Set-Schnitt).
- **Precondition:** ohne `codebook_categories`/`chunk_categories` (Minimal-/Alt-DB)
  bleibt das Feature inaktiv (`_has_table`-Check, kein stilles except).

## Integration (zwei Wege, User-Entscheid)

1. **Deterministischer Boost JETZT** (sicher, sofort): `score_v1 += _CAT_MATCH_BOOST (0.08)
   * cat_match`. Wie `_CAT_HINT_BOOST`, aber category_id-basiert. Kein Modell-Risiko
   (relevant nach #180: ein degeneriertes gelerntes Modell hatte mal Vektor-Treffer
   runtergerankt).
2. **Geloggtes Feature für gelernte v2-Reranking:** `cat_match` in `_FEATURES` +
   `stage["cat_match"]`. Der tägliche Trainer lernt das Gewicht, sobald genug Daten mit
   dem Feature geloggt sind. Backward-compat: fehlt im Modell → weight 0.

## Zwei-Phasen-Natur (ehrlich)

- **Phase A (jetzt, deployt):** deterministischer Boost wirkt sofort (für die ~199+
  chunk_categories-verlinkten Chunks; Coverage wächst mit jeder Ingestion). Feature wird
  geloggt.
- **Phase B (später):** Retrain nachdem genug cat_match-Daten geloggt sind → gelerntes
  Gewicht ersetzt/ergänzt den fixen Boost. Kein Code-Change nötig (Trainer liest `_FEATURES`).

## Tests

`_category_id_match` (Set-Logik) + Integration in `_rerank` (Kandidat mit FK-Link aber
ohne matchendes category_label → isoliert cat_match vom Free-String-Boost) + Existenz-
Check-Degradation. Volle Suite 1663 grün.

## Dateien

- `core/mayring_core/memory/retrieval.py`: `_category_id_match`, `_has_table`,
  v3-Setup-Block + Loop-Integration + score_v1 + stage-dict.
- `core/mayring_core/memory/reranker_v2.py`: `_FEATURES += ("cat_match",)`.
