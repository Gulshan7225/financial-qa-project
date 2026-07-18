"""
Numeric normalization utilities.

Financial PDFs encode numbers in many inconsistent ways:
  "1,234.5"   "(1,234.5)"  "$1,234,567"  "12.3%"  "1.2M"  "1.2bn"  "-"  "N/A"
Getting these wrong silently is the single biggest source of error in
financial-document extraction, so every number that enters the system
passes through this module rather than being parsed ad-hoc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_MULTIPLIERS = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "mm": 1_000_000,
    "million": 1_000_000,
    "bn": 1_000_000_000,
    "b": 1_000_000_000,
    "billion": 1_000_000_000,
    "cr": 10_000_000,       # crore (Indian reports)
    "crore": 10_000_000,
    "lakh": 100_000,
    "lac": 100_000,
}

_CURRENCY_SYMBOLS = r"[$₹€£]"
_NA_TOKENS = {"-", "--", "n/a", "na", "nil", "nm", ""}

_NUMBER_RE = re.compile(
    r"""
    (?P<neg_paren>\()?
    \s*
    (?P<currency>[$₹€£])?
    \s*
    (?P<neg_sign>-)?
    (?P<number>\d[\d,]*\.?\d*)
    \s*
    (?P<pct>%)?
    \s*
    (?P<multiplier>k|mm|m|bn|b|cr|crore|lakh|lac|thousand|million|billion)?
    \s*
    (?P<neg_paren_close>\))?
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class ParsedNumber:
    raw: str
    value: float
    is_percentage: bool
    currency: Optional[str]
    multiplier_applied: Optional[str]
    is_negative: bool

    def as_units(self) -> float:
        """Return the value scaled to plain units (multiplier applied)."""
        return self.value


def normalize_number(token: str) -> Optional[ParsedNumber]:
    """
    Parse a single numeric-looking token from a financial document.

    Returns None if the token is a recognised "not applicable" marker or
    contains no digits at all (so callers can distinguish "0" from
    "couldn't parse").
    """
    if token is None:
        return None
    cleaned = token.strip()
    if cleaned.lower() in _NA_TOKENS:
        return None

    match = _NUMBER_RE.search(cleaned)
    if not match or not match.group("number"):
        return None

    number_str = match.group("number").replace(",", "")
    try:
        value = float(number_str)
    except ValueError:
        return None

    is_negative = bool(match.group("neg_paren") or match.group("neg_sign")) or (
        match.group("neg_paren") is not None and match.group("neg_paren_close") is not None
    )
    # Parentheses around a number is standard accounting notation for negative
    if cleaned.strip().startswith("(") and cleaned.strip().endswith(")"):
        is_negative = True

    multiplier_key = (match.group("multiplier") or "").lower() or None
    multiplier = _MULTIPLIERS.get(multiplier_key, 1) if multiplier_key else 1
    value = value * multiplier

    if is_negative:
        value = -abs(value)

    return ParsedNumber(
        raw=token,
        value=value,
        is_percentage=bool(match.group("pct")),
        currency=match.group("currency"),
        multiplier_applied=multiplier_key,
        is_negative=is_negative,
    )


_CLEAN_NUMERIC_CELL_RE = re.compile(
    r"""^\(?
    \s*[$₹€£]?\s*
    -?
    \d[\d,]*\.?\d*
    \s*%?
    \s*(?:k|mm|m|bn|b|cr|crore|lakh|lac|thousand|million|billion)?
    \s*\)?$""",
    re.IGNORECASE | re.VERBOSE,
)


def looks_numeric(token: str) -> bool:
    """
    Strict check for whether a whole table cell IS a number (not merely
    'contains a digit somewhere'). This matters because loose text-based
    table detection can carve stray sentence fragments out of narrative
    paragraphs ("...grew from $42.5M in Q1 to $58...") that contain digits
    but are not, themselves, a numeric table cell -- treating those as
    numeric would corrupt both table-plausibility checks and fact
    extraction.
    """
    if token is None:
        return False
    t = token.strip()
    if t.lower() in _NA_TOKENS:
        return True  # counts as a numeric column, just a missing value
    return bool(_CLEAN_NUMERIC_CELL_RE.match(t))


def format_value(value: float, unit: Optional[str] = None) -> str:
    """Human friendly re-rendering of a normalized value, for answers."""
    if unit == "%":
        return f"{value:.2f}%"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{value:,.2f}"
    return f"{value:,.2f}"
