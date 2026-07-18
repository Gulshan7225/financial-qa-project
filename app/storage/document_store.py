"""
In-process document store.

Keeps the extracted representation of every uploaded report keyed by
report_id. Backed by a simple thread-safe dict for this reference
implementation; swapping in Redis/Postgres later only requires
reimplementing this module's interface (get/put/delete/list), nothing
in the API or QA layers needs to change.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.extraction.chart_extractor import ExtractedChart
from app.extraction.pdf_extractor import ExtractionResult, FinancialFact
from app.models import ProcessingStatus


@dataclass
class ReportRecord:
    report_id: str
    filename: str
    file_path: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    uploaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extraction: Optional[ExtractionResult] = None
    charts: List[ExtractedChart] = field(default_factory=list)
    error: Optional[str] = None


class DocumentStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[str, ReportRecord] = {}

    def put(self, record: ReportRecord) -> None:
        with self._lock:
            self._records[record.report_id] = record

    def get(self, report_id: str) -> Optional[ReportRecord]:
        with self._lock:
            return self._records.get(report_id)

    def delete(self, report_id: str) -> bool:
        with self._lock:
            return self._records.pop(report_id, None) is not None

    def list(self) -> List[ReportRecord]:
        with self._lock:
            return list(self._records.values())

    def all_facts(self, report_id: str) -> List[FinancialFact]:
        record = self.get(report_id)
        if not record or not record.extraction:
            return []
        return record.extraction.facts


# Module-level singleton used by the FastAPI app (simple DI pattern).
store = DocumentStore()
