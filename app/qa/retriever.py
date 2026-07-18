"""Thin wrapper caching a RetrievalIndex per report_id."""
from __future__ import annotations

import threading
from typing import Dict

from app.storage.document_store import ReportRecord
from app.storage.vector_index import RetrievalIndex, build_index_for_report

_lock = threading.Lock()
_index_cache: Dict[str, RetrievalIndex] = {}


def get_index(record: ReportRecord) -> RetrievalIndex:
    with _lock:
        if record.report_id not in _index_cache:
            _index_cache[record.report_id] = build_index_for_report(record)
        return _index_cache[record.report_id]


def invalidate(report_id: str) -> None:
    with _lock:
        _index_cache.pop(report_id, None)
