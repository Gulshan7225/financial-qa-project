"""
Unit tests for the accuracy-critical extraction and QA path.

Run with:  pytest tests/ -v

These tests run entirely offline against the generated sample report and
do not require FastAPI/uvicorn to be running -- they exercise the
extraction and answer-engine modules directly.
"""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "sample_reports" / "Acme_Quarterly_Report.pdf"


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_pdf():
    if not SAMPLE_PDF.exists():
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "sample_reports" / "generate_sample_report.py")],
            check=True,
        )
    yield


@pytest.fixture(scope="session")
def extraction_result():
    from app.extraction.pdf_extractor import extract_pdf
    return extract_pdf(str(SAMPLE_PDF))


def test_page_count(extraction_result):
    assert extraction_result.num_pages == 2


def test_tables_detected_and_no_false_positives(extraction_result):
    # Exactly the income statement + balance sheet tables should be found;
    # the narrative text on page 1 must NOT be misdetected as a table.
    assert len(extraction_result.tables) == 2
    for table in extraction_result.tables:
        assert table.page == 2


@pytest.mark.parametrize(
    "metric,period,expected",
    [
        ("revenue", "Q1 FY24", 42_500_000.0),
        ("revenue", "Q4 FY24", 58_300_000.0),
        ("cost_of_goods_sold", "Q3 FY24", 21_400_000.0),
        ("gross_profit", "Q2 FY24", 27_200_000.0),
        ("operating_expenses", "Q4 FY24", 17_800_000.0),
        ("net_income", "Q2 FY24", 9_100_000.0),
        ("eps", "Q3 FY24", 0.58),
        ("total_assets", "FY24", 248_900_000.0),
        ("total_liabilities", "FY23", 96_700_000.0),
        ("total_equity", "FY24", 144_700_000.0),
        ("total_debt", "FY23", 40_000_000.0),
        ("cash_and_equivalents", "FY24", 41_600_000.0),
    ],
)
def test_numeric_accuracy(extraction_result, metric, period, expected):
    """Every extracted number must exactly match the source document."""
    matches = [f for f in extraction_result.facts if f.metric == metric and f.period == period]
    assert matches, f"No fact found for {metric}/{period}"
    assert matches[0].value == pytest.approx(expected)


def test_no_facts_leak_from_narrative_text(extraction_result):
    """Facts must only come from real tables (page 2), not the narrative on page 1."""
    for fact in extraction_result.facts:
        assert fact.page == 2


class _FakeRecord:
    def __init__(self, extraction):
        self.report_id = "test"
        self.extraction = extraction
        self.charts = []


def test_structured_qa_exact_lookup(extraction_result):
    from app.qa.answer_engine import answer_question

    record = _FakeRecord(extraction_result)
    resp = answer_question(record, "What was total revenue in Q4 FY24?")
    assert resp.confidence == "high"
    assert "58.3" in resp.answer or "58,300,000" in resp.answer or "58.30M" in resp.answer


def test_structured_qa_all_periods(extraction_result):
    from app.qa.answer_engine import answer_question

    record = _FakeRecord(extraction_result)
    resp = answer_question(record, "Show me net income by quarter")
    assert resp.confidence == "high"
    for period in ["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"]:
        assert period in resp.answer


def test_retrieval_fallback_for_open_ended_question(extraction_result):
    from app.qa.answer_engine import answer_question

    record = _FakeRecord(extraction_result)
    resp = answer_question(record, "What is management's outlook for next year?")
    assert resp.answer  # should not be empty
    assert resp.confidence in {"low", "medium", "high"}


def test_normalizer_handles_common_financial_formats():
    from app.extraction.normalizer import normalize_number

    assert normalize_number("$1,234.50").value == 1234.50
    assert normalize_number("(500)").value == -500.0
    assert normalize_number("12.3%").value == 12.3
    assert normalize_number("1.2M").value == 1_200_000.0
    assert normalize_number("2bn").value == 2_000_000_000.0
    assert normalize_number("N/A") is None
    assert normalize_number("-") is None


def test_analysis_growth_computation(extraction_result):
    from app.analysis import compute_key_metrics

    metrics = compute_key_metrics(extraction_result.facts)
    assert "revenue" in metrics
    # (58.3 - 51.8) / 51.8 * 100 = 12.55
    assert metrics["revenue"]["period_over_period_growth_pct"] == pytest.approx(12.55, abs=0.01)


# ---------------------------------------------------------------------------
# CAMS / KFintech mutual-fund "Consolidated Account Summary" support
# ---------------------------------------------------------------------------

CAS_SAMPLE_PDF = PROJECT_ROOT / "sample_reports" / "CAS_sample.pdf"


def test_cas_statement_detection():
    from app.extraction.cas_extractor import looks_like_cas_statement

    assert looks_like_cas_statement("Consolidated Account Summary brought to you by CAMS and KFintech")
    assert not looks_like_cas_statement("Acme Industries Quarterly Financial Report")


@pytest.mark.skipif(not CAS_SAMPLE_PDF.exists(), reason="Sample CAS statement not present")
def test_cas_extraction_accuracy():
    from app.extraction.pdf_extractor import extract_pdf

    result = extract_pdf(str(CAS_SAMPLE_PDF))
    assert len(result.tables) == 1
    assert result.tables[0].headers[0] == "Folio No"

    totals = [f for f in result.facts if f.metric == "total_portfolio_market_value"]
    assert totals, "Expected a total_portfolio_market_value fact"
