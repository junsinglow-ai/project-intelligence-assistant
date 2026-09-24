"""Value normalisation for messy source data.

`data/MESSINESS.md` records every intentional flaw in the corpus together with a
"how it is handled" column. That column is a contract, and this module is where
it is kept: each function below names the flaw IDs it covers, so the table and
the code can be checked against each other.

Two rules hold throughout:

- **Nothing is dropped.** Every parser returns a `Parsed`, which carries the
  normalised value *and* the raw string. A value that cannot be parsed comes
  back with `value=None` and an `issue`, never silently coerced or discarded.
- **Interpretation is recorded.** Where a value is ambiguous (a bare `0.72` that
  might be a ratio or a percentage, a `02/07/2026` that might be either day
  order), the chosen reading is noted on the result so an answer can cite the
  raw form.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

DateStyle = Literal["dmy", "mdy"]

# M02: tokens that mean "no value", not a value.
_NULL_TOKENS = {"", "-", "--", "n/a", "na", "n.a.", "none", "null", "tbc", "tbd", "?", "unknown"}

# M07: text that occupies a field without answering it.
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\bsee attached\b", re.I),
    re.compile(r"\bpending .{0,40}confirmation\b", re.I),
    re.compile(r"\bto be (confirmed|advised|decided)\b", re.I),
    re.compile(r"\blorem ipsum\b", re.I),
)

_CID_RE = re.compile(r"\(cid:\d+\)")
_WS_RE = re.compile(r"\s+")


@dataclass(slots=True)
class Parsed:
    """A normalised value alongside the text it came from.

    `issue` names the flaw code when the raw form needed interpreting or could
    not be read at all; `note` explains the reading that was chosen.
    """

    value: Any
    raw: str | None
    issue: str | None = None
    note: str | None = None

    def __bool__(self) -> bool:
        return self.value is not None


@dataclass(slots=True)
class Flaws:
    """Flaw codes raised while cleaning one file, for the ingest report."""

    codes: set[str] = field(default_factory=set)

    def add(self, parsed: Parsed) -> Parsed:
        if parsed.issue:
            self.codes.add(parsed.issue)
        return parsed


# --------------------------------------------------------------------------
# Text (F05, F10, M07)
# --------------------------------------------------------------------------


def collapse_ws(raw: str | None) -> str:
    """Strip and collapse whitespace, and drop PDF `(cid:NNN)` glyph artifacts.

    F05. The `(cid:` handling is not in MESSINESS.md: it is an extraction
    artifact from the bullet glyph in the status-report PDFs, which would
    otherwise appear in every chunk and be embedded as if it were content.
    """
    if raw is None:
        return ""
    text = unicodedata.normalize("NFKC", str(raw))
    text = _CID_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def null_or(raw: Any) -> Parsed:
    """Normalise the null-token list to a real null (M02).

    'tbc', 'N/A' and friends are absences. Treating them as values makes counts
    and aggregates wrong, which is exactly what the risk-register and financials
    questions depend on.
    """
    text = collapse_ws(raw if isinstance(raw, str) else ("" if raw is None else str(raw)))
    if text.casefold() in _NULL_TOKENS:
        return Parsed(None, text or None, issue="M02" if text else None)
    return Parsed(text, text)


def is_placeholder(raw: Any) -> bool:
    """True when text fills a field without answering it (M07).

    Flagged rather than removed, so an agent can decline to quote it as the
    substance of a mitigation or decision instead of presenting it as content.
    """
    text = collapse_ws(raw if raw is not None else "")
    if not text:
        return False
    if any(p.search(text) for p in _PLACEHOLDER_PATTERNS):
        return True
    # Truncated mid-sentence: ends without terminal punctuation on a comma or
    # a conjunction, e.g. "...changed, pending PMO confirmation".
    return bool(re.search(r"[,;]\s*\w+$", text)) and not text.endswith((".", "!", "?"))


# --------------------------------------------------------------------------
# Numbers and money (F02, F07, M06)
# --------------------------------------------------------------------------

_CURRENCY_SYMBOLS = {"$": "USD", "£": "GBP", "€": "EUR"}
_SCALE_SUFFIXES = {"k": 1_000, "m": 1_000_000, "bn": 1_000_000_000, "b": 1_000_000_000}


def _strip_separators(text: str) -> Decimal:
    """Read a number without assuming a locale (F07).

    The last separator present is the decimal separator, so `1.850.000,00` and
    `1,850,000.00` both read as 1850000.00. Sniffing per value rather than
    applying one global locale is what lets European and US notation coexist in
    the same column, which they do in the financial workbook.
    """
    has_dot, has_comma = "." in text, "," in text

    if has_dot and has_comma:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        text = text.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_comma:
        # A 3-digit final group is a thousands separator (6,315,000); anything
        # shorter is a decimal comma (1234,56).
        text = text.replace(",", "") if len(text.rsplit(",", 1)[-1]) == 3 else text.replace(",", ".")
    elif has_dot and len(text.split(".")) > 2:
        text = text.replace(".", "")  # 1.850.000

    return Decimal(text)


def parse_money(raw: Any) -> Parsed:
    """Parse an amount into a Decimal plus a currency code (F02, F07, M06).

    Covers every notation in the corpus: `$6,315,000`, `$6.3M`, `6,315,000 USD`,
    `120k`, `$0.4M`, the European `1.850.000,00`, and `' 300,000 '` stored as
    text. Parenthesised values read as negative, which is how the financials
    export writes overspends.
    """
    base = null_or(raw)
    if not base:
        return Parsed(None, base.raw, issue=base.issue)

    text = str(base.value)
    currency = None
    negative = False

    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1].strip()

    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            currency, text = code, text.replace(symbol, "")
    if match := re.search(r"\b(USD|GBP|EUR)\b", text, re.I):
        currency, text = match.group(1).upper(), text[: match.start()] + text[match.end() :]

    text = text.strip()
    scale = 1
    if match := re.fullmatch(r"(.*?)\s*(bn|[kmb])", text, re.I):
        text, scale = match.group(1).strip(), _SCALE_SUFFIXES[match.group(2).lower()]

    if text.startswith("-"):
        negative, text = True, text[1:].strip()

    try:
        value = _strip_separators(text) * scale
    except (InvalidOperation, ValueError):
        return Parsed(None, base.raw, issue="F02", note=f"unreadable amount {base.raw!r}")

    return Parsed(-value if negative else value, base.raw, note=currency)


def parse_percent(raw: Any, bare_scale: Literal["auto", "ratio", "percent"] = "auto") -> Parsed:
    """Parse a percentage to a 0-1 fraction, recording the reading (F06).

    The same column mixes `77%`, `0.72`, `82 pct` and `0.7708`. An explicit
    marker always means "divide by 100". Without one the value is ambiguous, and
    it cannot be resolved per value: `1.0931` is a 109% overspend in a
    ratio-scaled column but 1.09% in a percent-scaled one. `bare_scale` carries
    the decision made for the whole column -- see `detect_percent_scale` -- and
    "auto" falls back to reading anything above 1 as a percentage.
    """
    base = null_or(raw)
    if not base:
        return Parsed(None, base.raw, issue=base.issue)

    text = str(base.value)
    explicit = bool(re.search(r"%|\bpct\b|\bper ?cent\b", text, re.I))
    text = re.sub(r"%|\bpct\b|\bper ?cent\b", "", text, flags=re.I).strip()

    try:
        value = _strip_separators(text)
    except (InvalidOperation, ValueError):
        return Parsed(None, base.raw, issue="F06", note=f"unreadable percentage {base.raw!r}")

    if explicit:
        return Parsed(float(value) / 100, base.raw, note="explicit percentage")
    if bare_scale == "ratio":
        return Parsed(float(value), base.raw, issue="F06", note="bare value, column is ratio-scaled")
    if bare_scale == "percent" or value > 1:
        return Parsed(float(value) / 100, base.raw, issue="F06", note="bare value read as percentage")
    return Parsed(float(value), base.raw, note="bare value <=1 read as fraction")


def detect_percent_scale(values: list[Any]) -> Literal["ratio", "percent"]:
    """Decide once per column how to read its unmarked values (F06).

    A column written as ratios (0.77, 1.09) and one written as percentages
    (77, 109) are indistinguishable value by value once a figure exceeds 1. The
    column as a whole is not ambiguous though: if most unmarked values sit at or
    below 1 they are ratios, and the occasional overspend above 1 is a ratio too.
    """
    bare = []
    for raw in values:
        text = collapse_ws(raw)
        if not text or re.search(r"%|\bpct\b|\bper ?cent\b", text, re.I):
            continue
        try:
            bare.append(_strip_separators(text))
        except (InvalidOperation, ValueError):
            continue
    if not bare:
        return "percent"
    return "ratio" if sum(1 for v in bare if v <= 1) * 2 >= len(bare) else "percent"


def parse_number(raw: Any) -> Parsed:
    """Parse a plain number, coercing per cell rather than per column (M06)."""
    base = null_or(raw)
    if not base:
        return Parsed(None, base.raw, issue=base.issue)
    try:
        return Parsed(_strip_separators(str(base.value)), base.raw)
    except (InvalidOperation, ValueError):
        return Parsed(None, base.raw, issue="M06", note=f"not numeric: {base.raw!r}")


# --------------------------------------------------------------------------
# Dates (F01)
# --------------------------------------------------------------------------

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}
_MONTHS |= {
    m.lower(): i
    for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"], 1
    )
}

_NUMERIC_DATE = re.compile(r"^(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})$")
_TEXT_DATE = re.compile(r"^(\d{1,2})[\s-]+([A-Za-z]+)[\s-]+(\d{2,4})$")
_MONTH_YEAR = re.compile(r"^([A-Za-z]+)\s+(\d{4})$")


def _four_digit_year(year: int) -> int:
    return year if year > 99 else 2000 + year


def detect_date_style(values: list[Any]) -> DateStyle:
    """Infer a file's day/month order from the values it can't be wrong about.

    F01 says to resolve dd/mm vs mm/dd against the file's dominant style. A
    value above 12 in either position settles it (`03/20/2026` can only be
    mm/dd; `30/06/2026` can only be dd/mm). Files with no decisive value default
    to dd/mm, the convention the rest of the corpus uses.
    """
    dmy = mdy = 0
    for value in values:
        text = collapse_ws(value)
        if match := _NUMERIC_DATE.fullmatch(text):
            first, second, _ = (int(g) for g in match.groups())
            if len(match.group(1)) == 4:
                continue  # ISO, tells us nothing about the ambiguous ones
            if first > 12 >= second:
                dmy += 1
            elif second > 12 >= first:
                mdy += 1
    return "mdy" if mdy > dmy else "dmy"


def parse_date(raw: Any, style: DateStyle = "dmy") -> Parsed:
    """Normalise a date to ISO-8601, keeping the raw string (F01).

    Handles every convention in the corpus: `2026-03-31`, `03/20/2026`,
    `30/06/2026`, `18-Sep-26`, `3 December 2025` and the month-only `June 2026`,
    which returns `YYYY-MM` and says so rather than inventing a day.
    """
    base = null_or(raw)
    if not base:
        return Parsed(None, base.raw, issue=base.issue)
    if isinstance(raw, date):
        return Parsed(raw.isoformat(), base.raw)

    text = str(base.value)

    if match := _NUMERIC_DATE.fullmatch(text):
        first, second, third = (int(g) for g in match.groups())
        if len(match.group(1)) == 4:
            year, month, day = first, second, third
        else:
            year = _four_digit_year(third)
            ambiguous = first <= 12 and second <= 12
            if first > 12:
                day, month = first, second
            elif second > 12:
                month, day = first, second
            else:
                day, month = (first, second) if style == "dmy" else (second, first)
            if ambiguous:
                return _build(year, month, day, base.raw, "F01", f"ambiguous, read as {style}")
        return _build(year, month, day, base.raw)

    if match := _TEXT_DATE.fullmatch(text):
        day, month_name, year = match.groups()
        if (month := _MONTHS.get(month_name.lower())) is None:
            return Parsed(None, base.raw, issue="F01", note=f"unknown month {month_name!r}")
        return _build(_four_digit_year(int(year)), month, int(day), base.raw)

    if match := _MONTH_YEAR.fullmatch(text):
        month_name, year = match.groups()
        if (month := _MONTHS.get(month_name.lower())) is None:
            return Parsed(None, base.raw, issue="F01", note=f"unknown month {month_name!r}")
        return Parsed(f"{int(year):04d}-{month:02d}", base.raw, note="month precision only")

    return Parsed(None, base.raw, issue="F01", note=f"unrecognised date {base.raw!r}")


def _build(year: int, month: int, day: int, raw: str | None, issue: str | None = None,
           note: str | None = None) -> Parsed:
    try:
        return Parsed(date(year, month, day).isoformat(), raw, issue, note)
    except ValueError:
        return Parsed(None, raw, issue="F01", note=f"impossible date {raw!r}")


# --------------------------------------------------------------------------
# Controlled vocabularies (F03)
# --------------------------------------------------------------------------

_RAG = {
    "g": "green", "green": "green", "on track": "green", "ontrack": "green", "good": "green",
    "a": "amber", "amber": "amber", "yellow": "amber", "at risk": "amber", "watch": "amber",
    "r": "red", "red": "red", "off track": "red", "critical": "red",
}

_STATUS = {
    "open": "open", "new": "open", "raised": "open",
    "in-progress": "in_progress", "in progress": "in_progress", "inprogress": "in_progress",
    "wip": "in_progress", "started": "in_progress",
    "pending": "pending", "not started": "pending", "planned": "pending",
    "mitigated": "mitigated", "accepted": "mitigated",
    "closed": "closed", "done": "closed", "completed": "closed", "complete": "closed",
    "resolved": "closed", "delivered": "closed",
}


def _lookup(table: dict[str, str], raw: Any, issue: str) -> Parsed:
    base = null_or(raw)
    if not base:
        return Parsed(None, base.raw, issue=base.issue)
    key = re.sub(r"[^a-z0-9 -]", "", str(base.value).casefold()).strip()
    if (value := table.get(key)) is not None:
        return Parsed(value, base.raw)
    return Parsed(None, base.raw, issue=issue, note=f"unmapped value {base.raw!r}")


def normalise_rag(raw: Any) -> Parsed:
    """Map an overall status to green/amber/red (F03).

    The three reports spell the same thing as 'G', 'Yellow' and 'A'. Unmapped
    values come back flagged rather than dropped, so a new spelling surfaces
    instead of quietly becoming a missing status.
    """
    return _lookup(_RAG, raw, "F03")


def normalise_status(raw: Any) -> Parsed:
    """Map a milestone or risk status to a controlled vocabulary (F03).

    Covers both vocabularies, because the corpus mixes them: milestones use
    done/Closed/Completed/In-progress/pending, risks use open/OPEN/mitigated/
    Closed. Matching is case-insensitive, which the 'how many risks are open'
    question depends on.
    """
    return _lookup(_STATUS, raw, "F03")
