"""
Deterministic PDF extraction: page text, layout-aware tables, and
line-item "facts" (metric/period/value triples) parsed out of those
tables. This module never calls an LLM -- it is the accuracy-critical
path and must be fully reproducible and auditable.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pdfplumber

from app.extraction.cas_extractor import looks_like_cas_statement, parse_cas_statement
from app.extraction.normalizer import looks_numeric, normalize_number

logger = logging.getLogger("finqa.extraction")

# Common financial statement line items we specifically look for so that
# structured lookups ("what was net income in Q2") can be answered exactly,
# without relying on semantic search or an LLM to read the number.
KNOWN_METRIC_ALIASES = {
    "revenue": ["revenue", "total revenue", "net sales", "net revenue", "turnover", "total income"],
    "cost_of_goods_sold": ["cost of goods sold", "cogs", "cost of sales", "cost of revenue"],
    "gross_profit": ["gross profit", "gross margin"],
    "operating_expenses": ["operating expenses", "opex", "total operating expenses"],
    "operating_income": ["operating income", "operating profit", "ebit"],
    "ebitda": ["ebitda"],
    "net_income": ["net income", "net profit", "profit after tax", "pat", "net earnings"],
    "total_assets": ["total assets"],
    "total_liabilities": ["total liabilities"],
    "total_equity": ["total equity", "shareholders equity", "stockholders equity"],
    "cash_and_equivalents": ["cash and cash equivalents", "cash & cash equivalents"],
    "eps": ["earnings per share", "eps", "diluted eps", "basic eps"],
    "total_debt": ["total debt", "long-term debt", "long term borrowings"],
    # Mutual fund / portfolio holdings statements (CAMS/KFintech CAS)
    "portfolio_market_value": ["market value"],
    "portfolio_cost_value": ["cost value", "investment value", "invested amount"],
    "portfolio_nav": ["nav", "net asset value"],
    "portfolio_unit_balance": ["unit balance", "units held"],
    "total_portfolio_market_value": ["total portfolio value", "total market value", "total value", "portfolio value"],
    "total_portfolio_cost_value": ["total cost value", "total investment"],
}


def _upright_page_text(page) -> str:
    """Extract text after dropping rotated characters (sidebar stamps,
    vertical watermarks) that otherwise scramble the reading order of the
    surrounding upright text."""
    filtered = page.filter(lambda obj: obj.get("object_type") != "char" or obj.get("upright", True))
    return filtered.extract_text() or ""


@dataclass
class ExtractedTable:
    table_id: str
    page: int
    title: Optional[str]
    headers: List[str]
    rows: List[List[str]]


@dataclass
class TextChunk:
    chunk_id: str
    page: int
    text: str


@dataclass
class FinancialFact:
    metric: str
    period: Optional[str]
    value: float
    unit: Optional[str]
    source_type: str
    source_id: str
    page: int
    raw_label: str = ""


@dataclass
class ExtractionResult:
    num_pages: int = 0
    tables: List[ExtractedTable] = field(default_factory=list)
    text_chunks: List[TextChunk] = field(default_factory=list)
    facts: List[FinancialFact] = field(default_factory=list)
    raw_page_images: List[dict] = field(default_factory=list)  # bbox metadata for chart pass


def _guess_metric_key(label: str) -> Optional[str]:
    label_norm = re.sub(r"[^a-z& ]", "", label.lower()).strip()
    for key, aliases in KNOWN_METRIC_ALIASES.items():
        for alias in aliases:
            if alias in label_norm:
                return key
    return None


def _clean_header(cell: Optional[str]) -> str:
    if not cell:
        return ""
    return re.sub(r"\s+", " ", cell).strip()


def _table_to_headers_rows(raw_table: List[List[Optional[str]]]) -> (List[str], List[List[str]]):
    """pdfplumber returns a list of rows (each a list of cell strings/None)."""
    if not raw_table:
        return [], []
    header_row = [_clean_header(c) for c in raw_table[0]]
    body_rows = [[_clean_header(c) for c in row] for row in raw_table[1:]]
    return header_row, body_rows


def _is_plausible_financial_table(headers: List[str], rows: List[List[str]]) -> bool:
    """
    Guard against false-positive "tables" detected inside ordinary paragraph
    text by the loose text-based extraction strategy. A wrapped paragraph
    split into pseudo-columns typically has long, sentence-like header
    cells and mostly non-numeric body cells -- a real financial table does
    not.
    """
    if len(headers) < 2 or not rows:
        return False

    if any(len(h) > 40 for h in headers):
        return False

    # Reject headers that are obviously cut mid-word (an all-lowercase
    # fragment such as "ndust" or "ries" from a broken sentence) -- but
    # don't penalize short, legitimate labels like "FY23" or "Q1", which
    # contain a digit or a capital letter.
    fragment_like = sum(1 for h in headers if re.fullmatch(r"[a-z]+", h or "") and len(h) <= 6)
    if headers and fragment_like / len(headers) > 0.5:
        return False

    data_cells = [cell for row in rows for cell in row[1:]]
    if not data_cells:
        return False
    numeric_cells = sum(1 for c in data_cells if looks_numeric(c) and len(c) < 20)
    if numeric_cells / len(data_cells) < 0.75:
        return False

    avg_row0_len = sum(len(row[0]) for row in rows) / len(rows)
    if avg_row0_len > 60:
        return False

    return True


def _extract_facts_from_table(table: ExtractedTable) -> List[FinancialFact]:
    """
    Treat the first column as the line-item label and remaining columns as
    period values (a very common layout for income statements / balance
    sheets). Each numeric cell becomes one auditable FinancialFact tied
    back to its exact table/page.
    """
    facts: List[FinancialFact] = []
    if not table.headers or not table.rows:
        return facts

    period_labels = table.headers[1:]
    for row in table.rows:
        if not row:
            continue
        label = row[0]
        if not label:
            continue
        metric_key = _guess_metric_key(label)
        for idx, cell in enumerate(row[1:]):
            if idx >= len(period_labels):
                break
            if not looks_numeric(cell):
                continue
            parsed = normalize_number(cell)
            if parsed is None:
                continue
            unit = "%" if parsed.is_percentage else (parsed.currency or None)
            facts.append(
                FinancialFact(
                    metric=metric_key or label.lower(),
                    period=period_labels[idx] or None,
                    value=parsed.value,
                    unit=unit,
                    source_type="table",
                    source_id=table.table_id,
                    page=table.page,
                    raw_label=label,
                )
            )
    return facts


def _facts_from_cas_holdings(cas_result, table_id: str, page_for_totals: int) -> List[FinancialFact]:
    facts: List[FinancialFact] = []
    for h in cas_result.holdings:
        scheme_label = h.scheme_name[:80]
        facts.append(
            FinancialFact(
                metric="portfolio_market_value", period=scheme_label, value=h.market_value,
                unit="₹", source_type="table", source_id=table_id, page=h.page, raw_label=h.scheme_name,
            )
        )
        facts.append(
            FinancialFact(
                metric="portfolio_cost_value", period=scheme_label, value=h.cost_value,
                unit="₹", source_type="table", source_id=table_id, page=h.page, raw_label=h.scheme_name,
            )
        )
        facts.append(
            FinancialFact(
                metric="portfolio_nav", period=scheme_label, value=h.nav,
                unit="₹", source_type="table", source_id=table_id, page=h.page, raw_label=h.scheme_name,
            )
        )
        facts.append(
            FinancialFact(
                metric="portfolio_unit_balance", period=scheme_label, value=h.unit_balance,
                unit=None, source_type="table", source_id=table_id, page=h.page, raw_label=h.scheme_name,
            )
        )
    if cas_result.total_market_value is not None:
        facts.append(
            FinancialFact(
                metric="total_portfolio_market_value", period=None, value=cas_result.total_market_value,
                unit="₹", source_type="table", source_id=table_id, page=page_for_totals, raw_label="Total",
            )
        )
    if cas_result.total_cost_value is not None:
        facts.append(
            FinancialFact(
                metric="total_portfolio_cost_value", period=None, value=cas_result.total_cost_value,
                unit="₹", source_type="table", source_id=table_id, page=page_for_totals, raw_label="Total",
            )
        )
    return facts


def extract_pdf(file_path: str) -> ExtractionResult:
    """
    Full deterministic extraction pass over the PDF:
      1. per-page text -> text_chunks (for semantic retrieval fallback)
      2. per-page tables -> structured ExtractedTable objects
      3. table rows -> FinancialFact entries with verified numeric parsing
      4. per-page embedded image bounding boxes -> handed to chart_extractor

    Mutual-fund "Consolidated Account Summary" statements (CAMS/KFintech)
    use a borderless, whitespace-aligned layout that generic ruled/text
    table detection mis-parses, so they are detected up front and routed
    through a dedicated line-pattern parser (`cas_extractor`) instead of
    the generic table pipeline.
    """
    result = ExtractionResult()

    with pdfplumber.open(file_path) as pdf:
        result.num_pages = len(pdf.pages)
        page_texts = [_upright_page_text(p) for p in pdf.pages]

        for page_index, text in enumerate(page_texts, start=1):
            if text.strip():
                result.text_chunks.append(
                    TextChunk(chunk_id=str(uuid.uuid4())[:8], page=page_index, text=text.strip())
                )

        is_cas = looks_like_cas_statement("\n".join(page_texts))

        if is_cas:
            cas_result = parse_cas_statement(file_path)
            if cas_result and cas_result.holdings:
                table_id = str(uuid.uuid4())[:8]
                table = ExtractedTable(
                    table_id=table_id,
                    page=cas_result.holdings[0].page,
                    title="Mutual Fund Holdings (Consolidated Account Summary)",
                    headers=["Folio No", "ISIN", "Scheme Name", "Cost Value", "Unit Balance", "NAV Date", "NAV", "Market Value", "Registrar"],
                    rows=[
                        [h.folio_no, h.isin, h.scheme_name, f"{h.cost_value:,.2f}", f"{h.unit_balance:,.3f}",
                         h.nav_date, f"{h.nav}", f"{h.market_value:,.2f}", h.registrar]
                        for h in cas_result.holdings
                    ],
                )
                result.tables.append(table)
                last_page = cas_result.holdings[-1].page
                result.facts.extend(_facts_from_cas_holdings(cas_result, table_id, last_page))
            return result

        for page_index, page in enumerate(pdf.pages, start=1):
            text = page_texts[page_index - 1]

            # ---- tables ----
            try:
                raw_tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                    }
                )
                if not raw_tables:
                    # fall back to a looser text-based strategy for
                    # tables without visible ruling lines
                    raw_tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                        }
                    )
            except Exception as exc:  # pdfplumber can raise on malformed pages
                logger.warning("Table extraction failed on page %s: %s", page_index, exc)
                raw_tables = []

            for raw_table in raw_tables:
                headers, rows = _table_to_headers_rows(raw_table)
                if not headers or not any(any(r) for r in rows):
                    continue
                if not _is_plausible_financial_table(headers, rows):
                    continue
                table = ExtractedTable(
                    table_id=str(uuid.uuid4())[:8],
                    page=page_index,
                    title=_infer_table_title(text, headers),
                    headers=headers,
                    rows=rows,
                )
                result.tables.append(table)
                result.facts.extend(_extract_facts_from_table(table))

            # ---- image bounding boxes (charts), extracted later ----
            for img in page.images:
                result.raw_page_images.append(
                    {
                        "page": page_index,
                        "x0": img["x0"],
                        "x1": img["x1"],
                        "top": img["top"],
                        "bottom": img["bottom"],
                    }
                )

    return result


def _infer_table_title(page_text: str, headers: List[str]) -> Optional[str]:
    """Best-effort: look for a heading line just above common statement names."""
    candidates = [
        "income statement", "statement of operations", "balance sheet",
        "cash flow statement", "statement of cash flows", "profit and loss",
    ]
    lowered = page_text.lower()
    for c in candidates:
        if c in lowered:
            return c.title()
    return None
