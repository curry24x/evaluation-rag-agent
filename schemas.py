from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QueryPlan(BaseModel):
    route: Literal["retrieve", "analyze", "both"]
    retrieval_queries: list[str] = Field(default_factory=list)
    analysis_tasks: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=20)
    chunk_size: int = Field(default=500, ge=100, le=2000)
    retrieval_mode: Literal["dense", "hybrid"] = "dense"
    use_reranker: bool = False

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be blank")
        return value


class EvidenceItem(BaseModel):
    source: str
    content: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    route: Literal["retrieve", "analyze", "both"]
    tools_used: list[str]
    evidence: list[EvidenceItem]
    latency_ms: float
    timings_ms: dict[str, float] = Field(default_factory=dict)
    retrieval_mode: Literal["dense", "hybrid"]
    synthesis_mode: str
    index_version: str
