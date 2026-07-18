"""
Parser for CAMS / KFintech "Consolidated Account Summary" (CAS) documents --
the standard mutual-fund holdings statement issued by India's two RTAs.

Unlike a company income statement or balance sheet, a CAS has no ruled
table borders: rows are whitespace-aligned text, and scheme names
routinely wrap onto a second or third line. Generic ruled-line/text-grid
table detection (used for income-statement style reports) mis-splits this
layout, so CAS documents are detected up front and routed through this
line-pattern parser instead.

Detection is a simple keyword check on the extracted text; parsing relies
on the fact that, empirically, every holding row places all six numeric
columns (Cost Value, Unit Balance, NAV Date, NAV, Market Value) plus the
Registrar on the SAME line as the Folio No/ISIN -- only the scheme name
description wraps onto following lines.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import pdfplumber

CAS_MARKERS = ("consolidated account summary", "cams", "kfintech")

FOLIO_RE = r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)*"
ISIN_RE = r"[A-Z]{2}[A-Z0-9]{9}\d"
MONEY_RE = r"[\d,]+\.\d+"
DATE_RE = r"\d{1,2}-[A-Za-z]{3}-\d{4}"

ROW_HEAD_RE = re.compile(rf"^({FOLIO_RE})\s*({ISIN_RE})\s+(.+)$")

ROW_TAIL_RE = re.compile(
    rf"^(?P<name>.+?)\s+"
    rf"(?P<cost>{MONEY_RE})\s+"
    rf"(?P<units>{MONEY_RE})\s+"
    rf"(?P<navdate>{DATE_RE})\s+"
    rf"(?P<nav>[\d,]+\.?\d*)\s+"
    rf"(?P<market>{MONEY_RE})\s+"
    rf"(?P<registrar>CAMS|KFINTECH)\s*$"
)

TOTAL_RE = re.compile(rf"^Total\s+(?P<cost>{MONEY_RE})\s+(?P<market>{MONEY_RE})\s*$")

SKIP_LINE_PREFIXES = ("Page ", "Folio No", "(INR)")


@dataclass
class Holding:
    folio_no: str
    isin: str
    scheme_name: str
    cost_value: float
    unit_balance: float
    nav_date: str
    nav: float
    market_value: float
    registrar: str
    page: int


@dataclass
class CasResult:
    holdings: List[Holding]
    total_cost_value: Optional[float]
    total_market_value: Optional[float]
    account_holder_name: Optional[str] = None


def _to_float(token: str) -> float:
    return float(token.replace(",", ""))


def _upright_only_text(page) -> str:
    """
    Strip rotated characters (e.g. a vertical sidebar watermark/version
    stamp) before extracting text. Left in place, such text interleaves
    with the main content and can scramble the reading order of nearby
    lines.
    """
    filtered = page.filter(lambda obj: obj.get("object_type") != "char" or obj.get("upright", True))
    return filtered.extract_text() or ""


def looks_like_cas_statement(all_text: str) -> bool:
    lowered = all_text.lower()
    return sum(marker in lowered for marker in CAS_MARKERS) >= 2


def parse_cas_statement(file_path: str) -> Optional[CasResult]:
    holdings: List[Holding] = []
    total_cost: Optional[float] = None
    total_market: Optional[float] = None
    account_holder: Optional[str] = None

    with pdfplumber.open(file_path) as pdf:
        full_text = "\n".join(_upright_only_text(p) for p in pdf.pages)
        if not looks_like_cas_statement(full_text):
            return None

        name_match = re.search(r"^((?:[A-Z]{2,}\s){1,3}[A-Z]{2,})\s+[a-z]", full_text, re.MULTILINE)
        if name_match:
            account_holder = name_match.group(1).strip()

        for page_index, page in enumerate(pdf.pages, start=1):
            text = _upright_only_text(page)
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

            pending_row: Optional[dict] = None

            for line in lines:
                if any(line.startswith(p) for p in SKIP_LINE_PREFIXES):
                    continue

                total_match = TOTAL_RE.match(line)
                if total_match:
                    total_cost = _to_float(total_match.group("cost"))
                    total_market = _to_float(total_match.group("market"))
                    continue

                head_match = ROW_HEAD_RE.match(line)
                if head_match:
                    folio, isin, rest = head_match.groups()
                    tail_match = ROW_TAIL_RE.match(rest)
                    if tail_match:
                        holdings.append(
                            Holding(
                                folio_no=folio,
                                isin=isin,
                                scheme_name=tail_match.group("name").strip(),
                                cost_value=_to_float(tail_match.group("cost")),
                                unit_balance=_to_float(tail_match.group("units")),
                                nav_date=tail_match.group("navdate"),
                                nav=_to_float(tail_match.group("nav")),
                                market_value=_to_float(tail_match.group("market")),
                                registrar=tail_match.group("registrar"),
                                page=page_index,
                            )
                        )
                        pending_row = holdings[-1]
                    continue

                # Continuation line: scheme name wrapping onto the next
                # line (no folio/ISIN, no numeric columns of its own).
                if pending_row is not None and not re.search(r"\d{2},|\bTotal\b", line):
                    pending_row.scheme_name = f"{pending_row.scheme_name} {line}".strip()

    return CasResult(
        holdings=holdings,
        total_cost_value=total_cost,
        total_market_value=total_market,
        account_holder_name=account_holder,
    )
