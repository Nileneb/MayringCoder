from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PiTaskRequest(BaseModel):
    task: str
    repo_slug: str | None = None
    system_prompt: str | None = None
    timeout: float = 180.0


class AnalyzeRequest(BaseModel):
    repo: str
    full: bool = False
    adversarial: bool = False
    no_pi: bool = False
    budget: int | None = None
    second_opinion: str | None = None


class RepoRequest(BaseModel):
    repo: str


class TurbulenceRequest(BaseModel):
    repo: str
    llm: bool = False


class BenchmarkRequest(BaseModel):
    top_k: int = 5
    repo: str | None = None


class IssuesIngestRequest(BaseModel):
    repo: str
    state: str = "open"
    force_reingest: bool = False


class PopulateRequest(BaseModel):
    repo: str
    force_reingest: bool = False
    # Issue #85: batch_delay throttles the populate-memory loop so the
    # GPU has breathing room between embedding batches. Default None
    # means use BATCH_DELAY_SECONDS from src/config.py. 0 disables the
    # delay entirely.
    batch_delay: float | None = None
    # WHY(#253): smoke-tests POST a known-bad repo that fails fast; tagging the
    # job source="smoke" lets the job-history default-filter it out as noise.
    source: str = ""


class PaperIngestRequest(BaseModel):
    papers_dir: str = "/data/papers"
    repo: str = ""
    force_reingest: bool = False


class MemorySearchRequest(BaseModel):
    query: str
    repo: str | None = None
    source_type: str | None = None
    top_k: int = 8
    char_budget: int = 6000
    task_context: str | None = None
    # When False, the retrieval skips the PI-advisor LLM stage entirely —
    # used by the UserPromptSubmit hook to stay inside its 9s timeout
    # budget on populated workspaces. Default None lets the existing
    # auto-trigger logic decide.
    llm_prefilter: bool | None = None
    # Memory-Injection v2.0 — pin a specific ranker for this request.
    # 'v1' / 'v2' / 'auto' (50/50 by query-hash). When unset, the env
    # variable RERANKER_VERSION decides; default is v1 so behaviour
    # never changes silently.
    reranker_version: str | None = None
    # WHY(#252): restrict the search to one logical sub-bucket of the
    # workspace — e.g. "project:<uuid>" so a Recherche search only sees that
    # project's papers, never another project's (same workspace). Always
    # type-prefixed; None = search the whole workspace.
    scope: str | None = None
    # #workspace-uuid-sot (v7): scope to ONE project (Project-ID-Dimension,
    # User-Diagramm) within the workspace. None = whole workspace.
    project: str | None = None
    # WHY(2026-05-10 multi-cat prompt-decomp): wenn der hook prompt-
    # kategorien via mistral:7b-instruct extrahiert hat, kann er sie als
    # hint mitsenden. Chunks deren category_labels mit diesen kategorien
    # überlappen bekommen einen score-boost im _rerank — der user-prompt
    # "auth + caching + deployment" rankt dann chunks mit auth/caching/
    # deployment labels höher als nur durch vector-similarity möglich.
    category_hint: list[str] | None = None
    # WHY(igio-search-2026-05-15): erlaubt hook + session_start die Suche
    # auf einen IGIO-Axis zu fokussieren ohne alle Chunks zu laden.
    # "goal" → zeigt was der User anstrebt; "issue" → offene Probleme etc.
    igio_intent: str | None = None


class MemoryPutRequest(BaseModel):
    source_id: str | None = None
    source_type: str = "repo_file"
    repo: str = ""
    path: str = ""
    content: str
    categorize: bool = False
    # WHY(2026-05-11): Mayring Selektionskriterium — das Thema worauf
    # kategorisiert wird (z.B. die forschungsfrage bei paper-ingest aus
    # app.linn.games). Empty → die prompts leiten das Thema aus dem
    # chunk selbst ab.
    task: str = ""
    # V2 visibility — optional; defaults to 'private' downstream.
    # 'org' requires membership in org_id (or first org_id if not given).
    visibility: str | None = None
    org_id: str | None = None
    # WHY(#252): typed logical sub-bucket within the workspace — e.g.
    # "project:<uuid>" for a Recherche project's papers. REQUIRED when
    # source_type in (paper, agent_result) — memory_put returns 422 without
    # it (no silent default). Optional/None for other source_types
    # (= workspace-global). Always type-prefixed: "<type>:<id>".
    scope: str | None = None


class LogEvent(BaseModel):
    """Single log-event from MemoryIngestHandler."""
    ts: str
    level: str
    logger: str = ""
    msg: str
    exc: str | None = None
    # error_signature wird server-side gehasht falls nicht mitgesendet,
    # damit Sender-Bugs nicht zu fehlender Dedup führen.
    error_signature: str | None = None
    # Free-form additional context the logger appended via extra={}.
    context: dict | None = None


class LogEventBatch(BaseModel):
    """Batch von Log-Events vom App-Logger.

    Trigger-Quelle: MemoryIngestHandler in MayringCoder bzw.
    äquivalenter Monolog-Handler in Laravel. Batches kommen rein
    wenn entweder (a) buffer voll (default 100) oder
    (b) ein Event mit level >= ERROR den sofort-Flush ausgelöst
    hat.
    """
    service: str  # "mayring-api", "laravel-app", etc.
    events: list[LogEvent]


class DuelRequest(BaseModel):
    task: str
    model_a: str
    model_b: str
    repo_slug: str | None = None
    system_prompt: str | None = None
    timeout: float = 180.0
    judge: bool = True
    judge_model: str | None = None
    no_memory_baseline: bool = False


class MemoryInvalidateRequest(BaseModel):
    source_id: str


class MemoryReindexRequest(BaseModel):
    source_id: str | None = None


class MemoryFeedbackRequest(BaseModel):
    chunk_id: str
    # WHY(2026-05-10 rating-migration): binary signal komplett raus, nur
    # noch rating 1-5. Auto-rater (path-match heuristik) lieferte gift-
    # signale für generic files (codebook.yaml +/web.php -) und der
    # binary-feature dominierte reranker-v2-weights mit 17× (issue #180).
    # Rating gibt skalares signal: 1=schadhaft, 3=neutral, 5=primärquelle.
    signal: Literal["1", "2", "3", "4", "5"]
    metadata: dict | None = None


class ConversationTurnModel(BaseModel):
    role: str
    content: str
    timestamp: str | None = None


class ConversationMicroBatchRequest(BaseModel):
    turns: list[ConversationTurnModel]
    session_id: str
    workspace_slug: str = "default"
    presumarized: str | None = None
    # WHY(igio-pipeline-2026-05-15): stop_hook extrahiert IGIO-Axis aus dem
    # User-Prompt via fast-hints (kein LLM-Aufruf nötig). Das erlaubt dem
    # Server, den resultierenden Chunk direkt zu taggen statt auf den
    # IGIO-Cron (async, Stunden später) zu warten.
    igio_hint: str | None = None  # "goal" | "issue" | "intervention" | "outcome"


class WikiGenerateRequest(BaseModel):
    repo: str = ""
    wiki_type: str = "code"
    workspace_id: str = ""


class AmbientSnapshotRequest(BaseModel):
    repo: str


class PredictiveRebuildRequest(BaseModel):
    repo: str | None = None


class BenchmarkTasksRequest(BaseModel):
    model_a: str
    model_b: str
    category: str | None = None
    repo_slug: str | None = None
    timeout: float = 180.0
    judge_model: str | None = None


class WikiRebuildRequest(BaseModel):
    workspace_id: str
    repo_slug: str = ""
    strategy: str = "louvain"
    ollama_url: str = ""
    model: str = "qwen2.5-coder:14b"


class WikiEdgeCreateRequest(BaseModel):
    source: str
    target: str
    type: str = "import"
    weight: float = 1.0
    context: str = ""
    user_id: str = "api"


class WikiConflictResolveRequest(BaseModel):
    source: str
    target: str
    user_id: str = "api"


class PatchVisibilityRequest(BaseModel):
    visibility: str
    org_id: str | None = None


class ShareSourceRequest(BaseModel):
    """Body for POST /sources/{id}/share (#195 Iter 4). All optional:
    no body / {} → share publicly; {"org_id": "<id>"} → share to that org."""
    org_id: str | None = None


class TaskCreateRequest(BaseModel):
    title: str
    description: str = ""
    priority: str = "medium"
    due_date: str | None = None
    tags: str = ""
    linked_chunk_id: str | None = None
    scope_key: str | None = None
    external_id: str | None = None


class TaskUpdateRequest(BaseModel):
    title: str | None = None
    description: str | None = None
    status: str | None = None
    priority: str | None = None
    due_date: str | None = None
    tags: str | None = None
    linked_chunk_id: str | None = None
    scope_key: str | None = None


class RepoEventRequest(BaseModel):
    event_type: str               # 'push' | 'workflow_run' | 'security'
    repo: str                     # repo URL (matches projects.source_ref)
    sha: str | None = None
    ref: str | None = None
    conclusion: str | None = None  # workflow_run: success|failure|...
    workflow: str | None = None
    severity: str | None = None    # security: low|moderate|high|critical
    summary: str | None = None
    url: str | None = None
