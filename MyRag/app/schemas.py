"""Pydantic request and response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MachineCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    embed_model_id: str | None = None
    chunk_size: int | None = Field(default=None, ge=128, le=4096)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1024)


class MachineUpdate(BaseModel):
    embed_model_id: str | None = None
    reembed: bool = False


class MachineOut(BaseModel):
    id: int
    title: str
    embed_model_id: str
    embed_dim: int
    chunk_size: int
    chunk_overlap: int
    created_at: datetime


class MachineSummary(MachineOut):
    document_count: int = 0
    chunk_count: int = 0
    query_count: int = 0
    total_cost_usd: float = 0.0


class DocumentOut(BaseModel):
    id: int
    machine_id: int
    filename: str
    content_type: str
    byte_size: int
    status: str
    error: str | None = None
    created_at: datetime
    chunk_count: int = 0
    embed_cost_usd: float = 0.0
    token_estimate: int = 0


class WorkflowOut(BaseModel):
    id: int
    name: str
    description: str
    class_key: str
    enabled: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    workflow_id: int
    llm_model_id: str
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatResponse(BaseModel):
    query_id: int
    answer: str
    reasoning: str | None = None
    workflow_name: str
    llm_model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cost_source: str
    retrieval_ms: int
    total_ms: int
    retrieved_chunk_ids: list[int] = Field(default_factory=list)
    llm_calls: list[dict] = Field(default_factory=list)


class ModelOption(BaseModel):
    id: str
    label: str | None = None
    input_per_1m: float
    output_per_1m: float | None = None
    dim: int | None = None
    context_window: int | None = None


class ModelsOut(BaseModel):
    llms: list[ModelOption]
    embeddings: list[ModelOption]
    llm_api_base: str
    embed_api_base: str
    default_llm_model: str
    default_embed_model: str
    embed_dim: int


class WorkflowComparisonRow(BaseModel):
    workflow_id: int
    workflow_name: str
    runs: int
    avg_cost_usd: float
    avg_total_tokens: float
    avg_latency_ms: float
    total_cost_usd: float
    last_run_at: datetime | None = None