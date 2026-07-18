"""
Answer engine.

Design principle: numbers must never be "guessed" by an LLM. The engine
tries, in order:

  1. STRUCTURED FACT LOOKUP - if the question mentions a known financial
     metric (revenue, net income, EBITDA, ...), look it up directly in the
     FinancialFact table built deterministically from parsed table cells.
     This is exact and auditable back to a page/table.
  2. RETRIEVAL + (optional) LLM SYNTHESIS - for open-ended / explanatory
     questions, retrieve the most relevant text/table chunks via TF-IDF
     and, if an LLM key is configured, ask the LLM to answer using ONLY
     that retrieved context (explicitly instructed not to introduce
     numbers absent from the context). Without an LLM key, falls back to
     returning the best-matching snippet directly (still 100% grounded,
     just less fluent).
  3. NO MATCH - explicit "not found in report" rather than a hallucinated
     guess.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app import config
from app.extraction.normalizer import format_value
from app.extraction.pdf_extractor import KNOWN_METRIC_ALIASES, FinancialFact
from app.models import FinancialFact as FinancialFactOut
from app.models import QueryResponse, SourceSnippet
from app.qa import retriever
from app.storage.document_store import ReportRecord


def _detect_metric(question: str) -> Optional[str]:
    q = question.lower()
    best_key, best_len = None, 0
    for key, aliases in KNOWN_METRIC_ALIASES.items():
        for alias in aliases:
            if alias in q and len(alias) > best_len:
                best_key, best_len = key, len(alias)
    return best_key


def _detect_period_token(question: str) -> Optional[str]:
    """Extract a coarse period hint like 'q1', 'q1 fy24', '2023', 'fy24' from the question."""
    match = re.search(
        r"\b(q[1-4]\s?(?:fy)?\s?\d{0,4}|fy\s?\d{2,4}|\d{4})\b", question.lower()
    )
    return match.group(0).replace(" ", "") if match else None


def _facts_for_metric(facts: List[FinancialFact], metric_key: str) -> List[FinancialFact]:
    return [f for f in facts if f.metric == metric_key]


def _filter_by_period(facts: List[FinancialFact], period_token: Optional[str]) -> List[FinancialFact]:
    if not period_token:
        return facts
    normalized_token = re.sub(r"[^a-z0-9]", "", period_token.lower())
    filtered = [
        f for f in facts
        if f.period and normalized_token in re.sub(r"[^a-z0-9]", "", f.period.lower())
    ]
    return filtered or facts  # if nothing matches the period, don't over-filter


_STOPWORDS = {
    "what", "is", "the", "was", "were", "of", "in", "for", "my", "me", "value", "total",
    "how", "much", "fund", "plan", "regular", "growth", "and", "a", "an",
}


def _filter_by_scheme_name(facts: List[FinancialFact], question: str) -> List[FinancialFact]:
    """
    Per-holding portfolio facts are tagged with the scheme name (not a
    quarter/year) in `period`, so matching against the question is done by
    keyword overlap with the scheme name rather than the period regex used
    for company financial statements.
    """
    q_words = {w for w in re.findall(r"[a-z0-9&]+", question.lower()) if w not in _STOPWORDS and len(w) > 2}
    if not q_words:
        return facts

    scored = []
    for f in facts:
        scheme_words = {w for w in re.findall(r"[a-z0-9&]+", (f.period or "").lower()) if len(w) > 2}
        overlap = len(q_words & scheme_words)
        if overlap:
            scored.append((overlap, f))

    if not scored:
        return facts  # no scheme mentioned -> caller returns the full list
    scored.sort(key=lambda x: x[0], reverse=True)
    top_score = scored[0][0]
    return [f for score, f in scored if score == top_score]


def _structured_answer(record: ReportRecord, question: str) -> Optional[QueryResponse]:
    metric_key = _detect_metric(question)
    if not metric_key:
        return None

    all_facts = record.extraction.facts if record.extraction else []
    matches = _facts_for_metric(all_facts, metric_key)
    if not matches:
        return None

    if metric_key.startswith("portfolio_"):
        matches = _filter_by_scheme_name(matches, question)
    else:
        period_token = _detect_period_token(question)
        matches = _filter_by_period(matches, period_token)
    if not matches:
        return None

    metric_display = metric_key.replace("_", " ").title()

    if len(matches) == 1:
        f = matches[0]
        if f.unit == "%":
            value_str = f"{format_value(f.value)}%"
        elif f.unit in ("$", "₹", "€", "£"):
            value_str = f"{f.unit}{format_value(f.value)}"
        elif f.unit:
            value_str = f"{format_value(f.value)} {f.unit}"
        else:
            value_str = format_value(f.value)
        answer = (
            f"{metric_display} for {f.period or 'the reported period'} was "
            f"{value_str} "
            f"(source: page {f.page}, extracted table)."
        )
    else:
        def _fmt(f):
            if f.unit == "%":
                return f"{format_value(f.value)}%"
            if f.unit in ("$", "₹", "€", "£"):
                return f"{f.unit}{format_value(f.value)}"
            if f.unit:
                return f"{format_value(f.value)} {f.unit}"
            return format_value(f.value)

        lines = [f"- {f.period or 'N/A'}: {_fmt(f)} (page {f.page})" for f in matches]
        grouping = "by holding" if metric_key.startswith("portfolio_") else "by period"
        answer = f"{metric_display} {grouping}, as reported:\n" + "\n".join(lines)

    sources = [
        SourceSnippet(
            source_type="table",
            page=f.page,
            reference=f"Table on page {f.page}",
            snippet=f"{f.raw_label}: {f.value}",
        )
        for f in matches
    ]

    return QueryResponse(
        report_id=record.report_id,
        question=question,
        answer=answer,
        confidence="high",
        matched_facts=[
            FinancialFactOut(
                metric=f.metric, period=f.period, value=f.value, unit=f.unit,
                source_type=f.source_type, source_id=f.source_id, page=f.page,
                raw_label=f.raw_label,
            ) for f in matches
        ],
        sources=sources,
    )


def _retrieval_answer(record: ReportRecord, question: str, top_k: int) -> QueryResponse:
    index = retriever.get_index(record)
    results = index.search(question, top_k=top_k)

    if not results:
        return QueryResponse(
            report_id=record.report_id,
            question=question,
            answer=(
                "I couldn't find information related to this question in the "
                "uploaded report. Try rephrasing, or ask about a specific "
                "line item (e.g. revenue, net income, total assets)."
            ),
            confidence="low",
            matched_facts=[],
            sources=[],
        )

    sources = [
        SourceSnippet(
            source_type=chunk.source_type,
            page=chunk.page,
            reference=chunk.reference,
            snippet=chunk.display[:400],
        )
        for chunk, _score in results
    ]

    if config.ENABLE_LLM:
        answer, confidence = _synthesize_with_llm(question, results)
    else:
        top_chunk, top_score = results[0]
        confidence = "medium" if top_score > 0.3 else "low"
        answer = (
            f"Based on the most relevant section of the report (page {top_chunk.page}):\n\n"
            f"{top_chunk.display[:800]}"
        )

    return QueryResponse(
        report_id=record.report_id,
        question=question,
        answer=answer,
        confidence=confidence,
        matched_facts=[],
        sources=sources,
    )


def _synthesize_with_llm(question: str, results) -> Tuple[str, str]:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        context_blocks = "\n\n".join(
            f"[{c.reference}]\n{c.content[:1200]}" for c, _ in results
        )
        prompt = (
            "You are a financial analyst assistant. Answer the question using "
            "ONLY the context excerpts below, which were extracted from a "
            "financial report. Do not introduce any number that is not "
            "explicitly present in the context. If the context does not "
            "contain the answer, say so plainly. Cite the page reference(s) "
            "you used in your answer.\n\n"
            f"CONTEXT:\n{context_blocks}\n\nQUESTION: {question}\n\nANSWER:"
        )
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        return text.strip(), "medium"
    except Exception:
        top_chunk, top_score = results[0]
        return (
            f"(LLM synthesis unavailable, showing best-matching excerpt from "
            f"page {top_chunk.page})\n\n{top_chunk.display[:800]}",
            "low",
        )


def answer_question(record: ReportRecord, question: str, top_k: Optional[int] = None) -> QueryResponse:
    structured = _structured_answer(record, question)
    if structured is not None:
        return structured
    return _retrieval_answer(record, question, top_k or config.TOP_K_CHUNKS)
