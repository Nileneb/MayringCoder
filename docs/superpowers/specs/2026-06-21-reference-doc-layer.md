# Spec: Reference-Doc-Layer (Cross-Repo-Dokus ohne Memory-Verrauschen)

**Datum:** 2026-06-21 · **Repo:** mayring-core + MayringCoder · **Status:** ✅ UMGESETZT 2026-06-21 (mayring-core PR#57; MayringCoder source_class-Ingest + /reference/search + Backfill-Endpoint). OFFEN: Migration v28 + `/stats/admin/backfill-chroma-metadata` einmal live laufen lassen.

## Problem
Externe Referenz-Dokus (Unity 6.3 Docs = 3495 Chunks, ~33% des Korpus, source_type=note) überstrahlten JEDE 3D/Graphics-Query und begruben den eigenen Code. Interim 2026-06-21: per `/stats/admin/deactivate-source-prefix {prefix:"unity-docs:"}` deaktiviert (reversibel, is_active=0). User-Wunsch: Dauerlösung — Dokus **verrauschen nie**, sind aber **IMMER da wenn WIRKLICH gebraucht**.

## Design: EIN Store, ein Flag, default-exclude, opt-in + repo-scoped
1. **Markierung:** `sources.source_class` ∈ {code, reference, conversation, …}. Referenz-Korpora beim Ingest als `reference` taggen. (Migration: bestehende `unity-docs:*` → reference; Chroma-Metadata `source_class` mitschreiben + Backfill — gemeinsam mit dem repo-Metadata-Backfill aus [[2026-06-21-retrieval-repo-scoping-hardfilter]].)
2. **Default-Ausschluss** in BEIDEN Stages: `build_chroma_where` ergänzt `{"source_class": {"$ne":"reference"}}`; `_scope_filter` ergänzt `AND s.source_class != 'reference'`. Normale Queries sehen die Docs nie.
3. **Opt-in-Verfügbarkeit (kombinierbar):**
   - Such-Flag `include_reference=true` → Reference kommt in die Kandidaten.
   - `/reference/search` (sucht NUR den Reference-Layer).
   - **Repo-gescopte Auto-Eligibility (Clou):** Reference-Korpus an die Repos linken die ihn brauchen (Unity-Docs ↔ Battlefield, via `chunk_project_links`). Ist DAS Projekt der aktive Scope → sein Reference-Layer automatisch eligible; sonst aus. „Unity-Docs nur bei Battlefield, nie bei app.linn.games."
4. **Ingest-Pfad:** ein `--reference --link-repo <repo>`-Flag bzw. Endpoint für Doku-Ingest, der source_class=reference setzt + an die Ziel-Repos linkt.

## Akzeptanz
- Default-Query nach „WebGL tier" → 0 unity-docs, eigener Code surfaced.
- `include_reference=true` ODER aktiver Battlefield-Scope → Unity-Docs erscheinen wieder.
- Reversibel + ohne separate Infra/Collection.

## Phasen
P1 (done) interim-deaktiviert. P2: source_class-Spalte + Metadata + Backfill + Default-Exclude. P3: include_reference-Flag + per-Repo-Linking.

## Verwandt
[[project_reference_doc_layer_design]] (Memory), [[project_c3_project_scoped_memory]].
