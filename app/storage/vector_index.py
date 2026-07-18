"""
Lightweight retrieval index.

We deliberately avoid a heavyweight embedding server / external vector DB
for this reference implementation: scikit-learn's TF-IDF + cosine
similarity runs fully offline, has zero external dependencies or API
costs, and is entirely sufficient for retrieving the right paragraph/table
out of a single report. Swap this class for a FAISS/Chroma + sentence-
transformers index if scaling to a large multi-document corpus.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class IndexedChunk:
    chunk_id: str
    source_type: str  # "text" | "table"
    page: int
    reference: str    # human readable pointer, e.g. "Page 3, Balance Sheet"
    content: str       # the text actually indexed/searched
    display: str       # the text shown back to the user as a snippet


class RetrievalIndex:
    def __init__(self, chunks: List[IndexedChunk]):
        self.chunks = chunks
        self._vectorizer = None
        self._matrix = None
        if chunks:
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._matrix = self._vectorizer.fit_transform([c.content for c in chunks])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[IndexedChunk, float]]:
        if not self.chunks or self._vectorizer is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return [(c, s) for c, s in ranked[:top_k] if s > 0]


def build_index_for_report(record) -> RetrievalIndex:
    """Build a fresh retrieval index from a ReportRecord's extraction result."""
    chunks: List[IndexedChunk] = []
    if not record.extraction:
        return RetrievalIndex(chunks)

    for tc in record.extraction.text_chunks:
        chunks.append(
            IndexedChunk(
                chunk_id=tc.chunk_id,
                source_type="text",
                page=tc.page,
                reference=f"Page {tc.page}",
                content=tc.text,
                display=tc.text[:600],
            )
        )

    for table in record.extraction.tables:
        rendered = _render_table(table)
        chunks.append(
            IndexedChunk(
                chunk_id=table.table_id,
                source_type="table",
                page=table.page,
                reference=f"Page {table.page}, Table ({table.title or 'untitled'})",
                content=rendered,
                display=rendered[:800],
            )
        )

    return RetrievalIndex(chunks)


def _render_table(table) -> str:
    lines = ["\t".join(table.headers)]
    for row in table.rows:
        lines.append("\t".join(row))
    return "\n".join(lines)
