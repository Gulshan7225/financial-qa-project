"""
Financial Report QA — FastAPI application.

  GET    /                               basic web UI (upload, ask, view tables/summary)

Endpoints (all prefixed /api/v1):
  POST   /reports/upload                 upload a PDF, kicks off extraction
  GET    /reports                        list uploaded reports
  GET    /reports/{id}/status            processing status
  GET    /reports/{id}/tables            structured tables extracted
  GET    /reports/{id}/charts            chart descriptions
  GET    /reports/{id}/summary           computed key metrics
  POST   /reports/{id}/query             ask a question, get a grounded answer
  GET    /reports/{id}/chart-image       render a PNG trend chart for a metric
  DELETE /reports/{id}                   delete stored data for a report

See API_DOCUMENTATION.md for full request/response details.
"""
from __future__ import annotations

import logging
import shutil
import traceback
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette import status as http_status

from app import config
from app.analysis import compute_key_metrics, render_metric_trend_chart, render_multi_metric_comparison
from app.extraction.chart_extractor import extract_charts
from app.extraction.pdf_extractor import extract_pdf
from app.models import (
    ChartOut,
    DeleteResponse,
    FinancialFact as FinancialFactModel,
    ProcessingStatus,
    QueryRequest,
    QueryResponse,
    ReportListItem,
    StatusResponse,
    SummaryResponse,
    TableOut,
    UploadResponse,
)
from app.qa import retriever
from app.qa.answer_engine import answer_question
from app.security import new_report_id, require_api_key, validate_pdf_upload
from app.storage.document_store import ReportRecord, store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finqa.api")

app = FastAPI(
    title="Financial Report QA API",
    description=(
        "Extracts structured data (text, tables, charts) from financial PDF "
        "reports and answers natural-language questions grounded in that "
        "extracted data."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Log the full traceback server-side (visible in the uvicorn console) and
    return a JSON body describing the failure, instead of a bare, opaque
    500 response -- makes issues far quicker to diagnose during development
    and integration.
    """
    logger.error("Unhandled error on %s %s: %s\n%s", request.method, request.url.path, exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {exc.__class__.__name__}: {exc}"},
    )

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
async def frontend_home():
    """Serves the basic web UI (upload a PDF, ask questions, view tables/summary)."""
    return FileResponse(STATIC_DIR / "index.html")


def _get_record_or_404(report_id: str) -> ReportRecord:
    record = store.get(report_id)
    if not record:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="report_id not found")
    return record


def _process_report(report_id: str) -> None:
    """Runs the full extraction pipeline synchronously and updates the store."""
    record = store.get(report_id)
    if not record:
        return
    try:
        record.status = ProcessingStatus.PROCESSING
        store.put(record)

        extraction = extract_pdf(record.file_path)
        record.extraction = extraction

        chart_dir = str(Path(config.PROCESSED_DIR) / report_id / "charts")
        record.charts = extract_charts(record.file_path, chart_dir)

        record.status = ProcessingStatus.DONE
        store.put(record)
        retriever.invalidate(report_id)
    except Exception as exc:
        logger.error("Processing failed for %s: %s\n%s", report_id, exc, traceback.format_exc())
        record.status = ProcessingStatus.FAILED
        record.error = str(exc)
        store.put(record)


@app.post("/api/v1/reports/upload", response_model=UploadResponse, tags=["Reports"])
async def upload_report(file: UploadFile, _auth=Depends(require_api_key)):
    """
    Upload a financial PDF report. Validates the file, stores it, and runs
    the extraction pipeline (text, tables, charts, financial facts).
    Processing is synchronous in this reference implementation for
    simplicity/determinism in the demo; swap to a background task queue
    (Celery/RQ) for large-scale or very large PDFs.
    """
    contents = await validate_pdf_upload(file)

    report_id = new_report_id()
    dest_path = Path(config.UPLOAD_DIR) / f"{report_id}.pdf"
    with open(dest_path, "wb") as f:
        f.write(contents)

    record = ReportRecord(report_id=report_id, filename=file.filename or "report.pdf", file_path=str(dest_path))
    store.put(record)

    _process_report(report_id)
    record = store.get(report_id)

    if record.status == ProcessingStatus.FAILED:
        return UploadResponse(
            report_id=report_id,
            filename=record.filename,
            status=record.status,
            message=f"Upload succeeded but extraction failed: {record.error}",
        )

    return UploadResponse(
        report_id=report_id,
        filename=record.filename,
        status=record.status,
        message="File uploaded and processed successfully.",
    )


@app.get("/api/v1/reports", response_model=List[ReportListItem], tags=["Reports"])
async def list_reports(_auth=Depends(require_api_key)):
    """List all uploaded reports (metadata only, no content)."""
    return [
        ReportListItem(
            report_id=r.report_id, filename=r.filename, status=r.status, uploaded_at=r.uploaded_at
        )
        for r in store.list()
    ]


@app.get("/api/v1/reports/{report_id}/status", response_model=StatusResponse, tags=["Reports"])
async def get_status(report_id: str, _auth=Depends(require_api_key)):
    record = _get_record_or_404(report_id)
    return StatusResponse(
        report_id=report_id,
        status=record.status,
        pages=record.extraction.num_pages if record.extraction else None,
        tables_found=len(record.extraction.tables) if record.extraction else None,
        charts_found=len(record.charts) if record.charts else None,
        error=record.error,
    )


@app.get("/api/v1/reports/{report_id}/tables", response_model=List[TableOut], tags=["Extraction"])
async def get_tables(report_id: str, _auth=Depends(require_api_key)):
    record = _get_record_or_404(report_id)
    if not record.extraction:
        return []
    return [
        TableOut(table_id=t.table_id, page=t.page, title=t.title, headers=t.headers, rows=t.rows)
        for t in record.extraction.tables
    ]


@app.get("/api/v1/reports/{report_id}/charts", response_model=List[ChartOut], tags=["Extraction"])
async def get_charts(report_id: str, _auth=Depends(require_api_key)):
    record = _get_record_or_404(report_id)
    return [
        ChartOut(
            chart_id=c.chart_id, page=c.page, description=c.description,
            extracted_series=c.extracted_series, confidence=c.confidence,
        )
        for c in record.charts
    ]


@app.get("/api/v1/reports/{report_id}/charts/{chart_id}/image", tags=["Extraction"])
async def get_chart_image(report_id: str, chart_id: str, _auth=Depends(require_api_key)):
    record = _get_record_or_404(report_id)
    for c in record.charts:
        if c.chart_id == chart_id:
            return Response(content=Path(c.image_path).read_bytes(), media_type="image/png")
    raise HTTPException(status_code=404, detail="chart_id not found")


@app.get("/api/v1/reports/{report_id}/summary", response_model=SummaryResponse, tags=["Analysis"])
async def get_summary(report_id: str, _auth=Depends(require_api_key)):
    record = _get_record_or_404(report_id)
    facts = record.extraction.facts if record.extraction else []
    metrics = compute_key_metrics(facts)
    periods = sorted({f.period for f in facts if f.period})
    return SummaryResponse(
        report_id=report_id,
        key_metrics=metrics,
        period_covered=periods,
        notes=[
            "Values are extracted directly from tables detected in the PDF.",
            "Growth % is period-over-period based on the order line items appear in the source table.",
        ],
    )


@app.get("/api/v1/reports/{report_id}/chart-image", tags=["Analysis"])
async def get_trend_chart_image(
    report_id: str,
    metric: str = Query(..., description="Metric key, e.g. revenue, net_income, gross_profit"),
    _auth=Depends(require_api_key),
):
    """Render a PNG bar chart of a single metric's trend across reported periods."""
    record = _get_record_or_404(report_id)
    facts = record.extraction.facts if record.extraction else []
    try:
        png_bytes = render_metric_trend_chart(metric, facts)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return Response(content=png_bytes, media_type="image/png")


@app.get("/api/v1/reports/{report_id}/compare-image", tags=["Analysis"])
async def get_comparison_chart_image(
    report_id: str,
    metrics: str = Query(..., description="Comma-separated metric keys, e.g. revenue,net_income"),
    _auth=Depends(require_api_key),
):
    """Render a PNG grouped-bar chart comparing several metrics across periods."""
    record = _get_record_or_404(report_id)
    facts = record.extraction.facts if record.extraction else []
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    png_bytes = render_multi_metric_comparison(metric_list, facts)
    return Response(content=png_bytes, media_type="image/png")


@app.post("/api/v1/reports/{report_id}/query", response_model=QueryResponse, tags=["Question Answering"])
async def query_report(report_id: str, body: QueryRequest, _auth=Depends(require_api_key)):
    """
    Ask a natural-language question about the uploaded report. The engine
    first attempts an exact, table-grounded answer for known financial
    metrics; if that fails it falls back to retrieval over the report's
    text/tables (optionally synthesized into fluent prose by an LLM,
    strictly grounded in the retrieved context).
    """
    record = _get_record_or_404(report_id)
    if record.status != ProcessingStatus.DONE:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"Report is not ready for querying (status: {record.status}).",
        )
    return answer_question(record, body.question, body.top_k)


@app.delete("/api/v1/reports/{report_id}", response_model=DeleteResponse, tags=["Reports"])
async def delete_report(report_id: str, _auth=Depends(require_api_key)):
    """Delete all stored data (file, extraction, charts) for a report."""
    record = store.get(report_id)
    if not record:
        raise HTTPException(status_code=404, detail="report_id not found")

    Path(record.file_path).unlink(missing_ok=True)
    chart_dir = Path(config.PROCESSED_DIR) / report_id
    if chart_dir.exists():
        shutil.rmtree(chart_dir, ignore_errors=True)

    store.delete(report_id)
    retriever.invalidate(report_id)
    return DeleteResponse(report_id=report_id, deleted=True)


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "llm_enabled": config.ENABLE_LLM}
