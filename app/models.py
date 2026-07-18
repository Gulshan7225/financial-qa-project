"""Pydantic schemas shared across the API layer."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class UploadResponse(BaseModel):
    report_id: str
    filename: str
    status: ProcessingStatus
    message: str


class StatusResponse(BaseModel):
    report_id: str
    status: ProcessingStatus
    pages: Optional[int] = None
    tables_found: Optional[int] = None
    charts_found: Optional[int] = None
    error: Optional[str] = None


class TableOut(BaseModel):
    table_id: str
    page: int
    title: Optional[str] = None
    headers: List[str]
    rows: List[List[str]]


class ChartOut(BaseModel):
    chart_id: str
    page: int
    description: str
    extracted_series: Optional[Dict[str, Any]] = None
    confidence: str  # "high" | "medium" | "low"


class FinancialFact(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    metric: str
    period: Optional[str] = None
    value: float
    unit: Optional[str] = None
    source_type: str  # "table" | "chart" | "text"
    source_id: str
    page: int
    raw_label: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question about the report")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceSnippet(BaseModel):
    source_type: str
    page: int
    reference: str
    snippet: str


class QueryResponse(BaseModel):
    report_id: str
    question: str
    answer: str
    confidence: str  # "high" | "medium" | "low"
    matched_facts: List[FinancialFact] = []
    sources: List[SourceSnippet] = []


class SummaryResponse(BaseModel):
    report_id: str
    key_metrics: Dict[str, Any]
    period_covered: List[str]
    notes: List[str] = []


class ReportListItem(BaseModel):
    report_id: str
    filename: str
    status: ProcessingStatus
    uploaded_at: str


class DeleteResponse(BaseModel):
    report_id: str
    deleted: bool
