from __future__ import annotations

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


class MemoryPutRequest(BaseModel):
    source_id: str | None = None
    source_type: str = "repo_file"
    repo: str = ""
    path: str = ""
    content: str
    categorize: bool = False


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
    signal: str
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
