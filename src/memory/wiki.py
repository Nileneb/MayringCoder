"""Verknüpfungswiki — funktionale Zusammenhänge via Label-Co-Occurrence und Dependency-Analyse."""
from src.memory.wiki_core import *
from src.memory.wiki_paper import *
from src.memory.wiki_orchestration import *

# Private names not exported by * — explicit re-exports for external callers
from src.memory.wiki_core import (
    _build_class_index,
    _resolve_dep,
    _fn_field,
    _TYPE_RE,
    _DISPATCH_RE,
    _JOB_CLASS_RE,
)
from src.memory.wiki_paper import (
    _cache_get,
    _cache_put,
    _paper_source_ids,
    _chunk_text,
    _CITE_NUM_RE,
    _CITE_AUTH_RE,
    _KNOWN_METHODS,
    _KNOWN_DATASETS,
    _extract_methods_from_chunks,
    _extract_datasets_from_chunks,
)
from src.memory.wiki_orchestration import (
    _build_keyword_index,
    _build_cluster_embeddings,
)
