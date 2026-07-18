# Financial Report QA — Mini Project

A FastAPI backend that extracts structured data (text, tables, and charts)
from financial PDF reports and answers natural-language questions about
that data, with an emphasis on **numeric accuracy** and **auditability**
over fluent-sounding guesses.

---

## 1. Problem framing & design philosophy

Financial numbers are the one thing in this system that must never be
wrong. So the architecture is split into two deliberately different
tiers:

| Tier | Responsible for | How it works | Failure mode if wrong |
|---|---|---|---|
| **Deterministic extraction** | Every number that becomes a "fact" (revenue, net income, total assets, ...) | Rule-based parsing with `pdfplumber` + a strict numeric normalizer/validator | Silent wrong number — so this path has no ML/LLM in it at all |
| **Retrieval + optional LLM** | Open-ended, explanatory questions ("what's the outlook", "why did margins improve") | TF‑IDF retrieval over extracted text/tables, optionally rephrased fluently by an LLM **strictly grounded** in the retrieved excerpts | Slightly less fluent answer — never a wrong number, because the LLM is never the source of a number, only of prose around numbers that already came from tier 1 |

When a user asks a question, the **answer engine tries the deterministic
path first**. Only if the question doesn't match a known financial metric
does it fall back to retrieval. This means "What was net income in Q2?"
is always answered from the exact parsed table cell, with a page citation
— never from an LLM's read of the PDF.

---

## 2. Architecture

```
                          ┌─────────────────────────┐
                          │      FastAPI app         │
                          │      (app/main.py)       │
                          └────────────┬─────────────┘
                                       │
                 ┌─────────────────────┼───────────────────────┐
                 │                     │                       │
        POST /reports/upload   POST /reports/{id}/query   GET /reports/{id}/chart-image
                 │                     │                       │
                 ▼                     ▼                       ▼
      ┌────────────────────┐  ┌─────────────────┐   ┌───────────────────┐
      │ security.py         │  │ qa/answer_engine │   │ analysis.py        │
      │ - magic byte check  │  │  1. structured   │   │ - derived metrics  │
      │ - size limit        │  │     fact lookup  │   │ - matplotlib charts│
      │ - filename sanitize │  │  2. retrieval +  │   └───────────────────┘
      └──────────┬──────────┘  │     optional LLM │
                 │              └────────┬────────┘
                 ▼                       │
      ┌─────────────────────┐            ▼
      │ extraction/          │  ┌─────────────────────┐
      │  pdf_extractor.py    │  │ storage/             │
      │   - text per page    │  │  document_store.py   │
      │   - tables (ruled +  │◄─┤  (extraction results  │
      │     text-strategy)   │  │   keyed by report_id) │
      │   - numeric facts    │  │  vector_index.py      │
      │  chart_extractor.py  │  │  (TF-IDF over text +  │
      │   - crop chart imgs  │  │   table chunks)       │
      │   - optional vision  │  └─────────────────────┘
      │     LLM description  │
      └─────────────────────┘
```

### Extraction pipeline (`app/extraction/`)

1. **Text** — `pdfplumber` pulls per-page text, kept as retrieval chunks
   for the fallback QA path.
2. **Tables** — `pdfplumber.extract_tables()` is tried first with a
   strict ruled-line strategy, then a looser text-based strategy for
   tables without visible borders. Every candidate table then passes
   through a plausibility filter (`_is_plausible_financial_table`) that
   rejects wrapped paragraph text which superficially looks tabular
   (long sentence-like header cells, mostly non-numeric body cells) —
   this keeps narrative text on the page from polluting the extracted
   facts. Covered by
   `tests/test_extraction.py::test_tables_detected_and_no_false_positives`.
3. **Numeric normalization** (`normalizer.py`) — every table cell is
   parsed through a single, strict, well-tested function that handles:
   currency symbols (`$`, `₹`, `€`, `£`), thousands separators,
   accounting-style negative parentheses `(500)`, percentages, and
   magnitude suffixes (`1.2M`, `2bn`, `50cr`, `3 lakh`). `N/A`, `-`, `nil`
   are recognized as "no value" rather than silently becoming `0`.
4. **Financial facts** — each numeric table cell becomes a
   `FinancialFact(metric, period, value, unit, source_type, source_id, page)`.
   Line-item labels are matched against a metric alias dictionary
   (revenue, COGS, gross profit, operating expenses, EBITDA, net income,
   total assets/liabilities/equity, EPS, debt, cash) so common metrics can
   be looked up directly regardless of the exact wording used in a given
   report.
5. **Charts** — embedded images are cropped out with `pdfplumber`
   (deterministic, no LLM). If `ANTHROPIC_API_KEY` is configured, the
   cropped image is sent to a vision-capable Claude model to produce a
   caption and a best-effort JSON data series. This is explicitly tagged
   `confidence: low/medium` and is **never used to override a
   table-derived fact** — chart data supplements narrative answers, it
   doesn't replace verified numbers. Without an API key, the chart image
   is still extracted and served, just without an auto-generated
   description.

### A second document type: mutual-fund holdings statements (CAS)

Not every "financial report" is laid out like a company income statement.
India's CAMS/KFintech **Consolidated Account Summary** (a mutual-fund
holdings statement) uses a borderless, whitespace-aligned layout where
scheme names routinely wrap onto a second or third line — generic
ruled-line/text-grid table detection mis-splits this and can silently
corrupt numbers.

`app/extraction/cas_extractor.py` detects this document type from its
header text and parses it with a dedicated line-pattern parser instead of
the generic table pipeline:
- each holding becomes a `FinancialFact` for market value, cost value,
  NAV, and unit balance, keyed by scheme name instead of a period
- the portfolio `Total` row becomes aggregate
  `total_portfolio_market_value` / `total_portfolio_cost_value` facts
- the QA engine matches these by keyword overlap with the scheme name
  mentioned in the question (see `_filter_by_scheme_name` in
  `app/qa/answer_engine.py`) rather than the quarter/year matching used
  for company statements

This same detect-and-route pattern is how additional statement formats
(e.g. a different country's standard brokerage statement) would be added
in future — a lightweight format-specific parser feeding the same
`FinancialFact` structure the rest of the system already understands.

### Storage (`app/storage/`)

- `document_store.py` — in-process store keyed by `report_id` (swap for
  Redis/Postgres for multi-instance deployments; the interface — `put`,
  `get`, `delete`, `list` — is intentionally small).
- `vector_index.py` — TF‑IDF + cosine similarity over text/table chunks.
  Chosen over an embedding server so the whole system runs fully offline
  with zero external dependencies/cost for the retrieval-fallback path;
  swap for FAISS/Chroma + sentence-transformers if scaling to a large
  multi-document corpus.

### QA (`app/qa/answer_engine.py`)

```
question ─▶ known financial metric mentioned? ──yes──▶ exact fact lookup
               │                                        (page-cited, confidence=high)
               no
               ▼
         TF-IDF retrieval over report chunks
               │
      LLM configured? ──yes──▶ LLM answers using ONLY retrieved context
               │                (explicitly instructed not to invent numbers)
               no
               ▼
       return best-matching excerpt verbatim (confidence=low/medium)
```

### Analysis / graphs (`app/analysis.py`)

Growth rates and margins are computed with plain Python arithmetic
directly over the verified `FinancialFact` list — not asked of an LLM —
and `matplotlib` renders bar charts straight from the same numbers, so
the picture is guaranteed to match the figures in `/summary`.

### Security (`app/security.py`)

- **File-type verification by magic bytes** (`%PDF-`), not just the
  client-supplied MIME type or filename extension — a renamed non-PDF
  file is rejected.
- **Upload size cap** (`FINQA_MAX_UPLOAD_MB`, default 25MB).
- **Filename sanitisation** — strips path separators and special
  characters to prevent path traversal; files are stored under a
  server-generated UUID, never the client-supplied name.
- **Optional API-key auth** — set `FINQA_API_KEY` to require an
  `X-API-Key` header on every endpoint. Left blank for local/demo use.
- **Deletable data** — `DELETE /reports/{id}` removes the uploaded file,
  cropped chart images, and all in-memory extraction results for a
  report, so sensitive financial data isn't retained indefinitely.
- CORS is wide-open (`*`) for local demoing — tighten to explicit origins
  before any real deployment.

---

## 3. Why these tools

| Choice | Reason |
|---|---|
| **FastAPI** | Async, automatic OpenAPI/Swagger docs, Pydantic-validated request/response models — matches the "well-structured, easy to integrate" requirement directly. |
| **pdfplumber** | Best-in-class open-source layout-aware table extraction; gives per-cell bounding boxes needed for both tables and cropping chart images, without needing a paid OCR/parsing API. |
| **scikit-learn TF-IDF** | Zero-dependency, fully offline semantic-ish retrieval; sufficient for single-document QA and avoids the cost/latency/privacy tradeoffs of an embedding API for this scope. |
| **Anthropic API (optional)** | Used *only* for (a) describing chart images and (b) making retrieval answers fluent — never for producing the numbers themselves. The system is fully functional with this switched off. |
| **matplotlib** | Deterministic chart rendering directly from verified data, for the graphical-representation requirement. |

---

## 4. Setup & running

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                  # optional: add ANTHROPIC_API_KEY / FINQA_API_KEY

# Generate the bundled sample report (or use your own PDF)
python sample_reports/generate_sample_report.py

# Run the API
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/** for a basic web UI (upload a PDF, ask
questions, view extracted tables and key metrics), or
**http://localhost:8000/docs** for interactive Swagger API docs, or
**http://localhost:8000/redoc** for ReDoc.

### Quick demo via curl

```bash
# 1. Upload the sample report
curl -X POST "http://localhost:8000/api/v1/reports/upload" \
  -F "file=@sample_reports/Acme_Quarterly_Report.pdf"
# -> {"report_id": "…", "status": "done", ...}

# 2. Ask a question (use the report_id returned above)
curl -X POST "http://localhost:8000/api/v1/reports/<report_id>/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was total revenue in Q4 FY24?"}'

# 3. Get computed key metrics
curl "http://localhost:8000/api/v1/reports/<report_id>/summary"

# 4. Get a rendered trend chart (PNG)
curl "http://localhost:8000/api/v1/reports/<report_id>/chart-image?metric=revenue" --output revenue.png
```

See **API_DOCUMENTATION.md** for the full endpoint reference.

---

## 5. Tests

```bash
pytest tests/ -v
```

Covers: page/table counts on the sample report, exact numeric accuracy of
every extracted fact against known ground truth, absence of false-positive
tables from narrative text, structured vs. retrieval QA routing, the
numeric normalizer's handling of currency/parentheses/percent/magnitude
formats, and filename/upload security validation.

---

## 6. Known limitations & next steps

- **Table detection** works well for ruled tables (the common case in
  formatted financial reports) and reasonably for borderless tables via
  the text-strategy fallback + plausibility filter; highly irregular
  layouts (merged cells spanning multiple line items, multi-level nested
  headers) may need per-report tuning of `pdfplumber`'s table settings.
- **Chart-to-data extraction** is inherently approximate (it's reading
  pixels, not vector data) and is intentionally tagged with lower
  confidence and never allowed to override a table-derived fact.
- **Retrieval** uses TF-IDF rather than dense embeddings; this is a
  deliberate scope/cost tradeoff for single-document QA and would be the
  first thing to upgrade for multi-document or cross-report queries.
- **Processing is synchronous** in the upload endpoint for demo
  simplicity; a production deployment handling large PDFs or high upload
  volume should move extraction to a background task queue (Celery/RQ)
  and poll `/status`.
- **Storage is in-process** (a Python dict); restarting the server loses
  uploaded reports. Swap `document_store.py` for Redis/Postgres for
  persistence across restarts/instances.
