#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["reportlab>=4.0", "openpyxl>=3.1"]
# ///
"""Render a synthetic project dataset from a JSON spec.

Vendored from the `synthetic-project-data` skill so that `make data` works
from a clean checkout without it. Driven by generate_synthetic_data.py;
run directly for the flaw catalogue (--list-flaws) or a clean control set
(--flaws none).

Writes into --out:
  * one PDF status report per entry in spec["reports"] (three house styles)
  * a financial summary (XLSX and/or CSV)
  * exactly one risk register (CSV or XLSX)
  * manifest.json - files written, which flaws landed where, ground-truth facts
  * MESSINESS.md  - the flaw table, shaped for a data/README.md

The numbers in the reports are derived from the financial lines rather than
restated in the spec, so the documents never contradict each other: only their
formatting differs. Flaw injection is driven by a seeded RNG, so the same spec
and seed reproduce the same files and the documentation stays true.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Flaw catalogue
#
# Two families, matching the kinds of mess that actually survive into project
# document sets: values that mean the same thing written differently, and
# values that are simply absent or structurally wrong. Each entry carries the
# handling note that ends up in MESSINESS.md, because a flaw nobody documented
# is indistinguishable from a bug in the generator.
# --------------------------------------------------------------------------

FLAWS: dict[str, dict] = {
    "F01": dict(
        family="format",
        name="Mixed date formats",
        summary="Each document uses its own date convention (2026-03-31, 31/03/2026, "
                "31-Mar-26, March 2026, Q1 FY26); hand-edited rows drift from their "
                "file's house style.",
        handling="Parse with a multi-format date parser and normalise to ISO-8601 at "
                 "ingestion; resolve dd/mm vs mm/dd against the file's dominant style "
                 "and keep the raw string in metadata.",
    ),
    "F02": dict(
        family="format",
        name="Mixed currency notation",
        summary="The same amount appears as 1200000, \"1,200,000\", \"S$1.2M\", "
                "\"1200k\" and \"1,200,000 SGD\"; negatives as -45000 and (45,000).",
        handling="Strip symbols/separators and expand k/M suffixes to a numeric value "
                 "plus a currency code; read parenthesised numbers as negative.",
    ),
    "F03": dict(
        family="format",
        name="Inconsistent status vocabulary",
        summary="RAG and progress states are spelled every way a human would type "
                "them: Green / GREEN / On Track / G, Amber / At Risk / Yellow, "
                "Complete / Completed / DONE / done.",
        handling="Map to a controlled vocabulary with a case-insensitive synonym "
                 "table; surface unmapped values rather than silently dropping them.",
    ),
    "F04": dict(
        family="format",
        name="Header naming drift",
        summary="The same concept is headed differently across files and within one "
                "header row (Owner / Risk Owner / Accountable, Date Raised / raised_date).",
        handling="Resolve headers through a synonym map to canonical field names; "
                 "match case-insensitively after stripping punctuation and spaces.",
    ),
    "F05": dict(
        family="format",
        name="Whitespace, casing and filename drift",
        summary="Leading/trailing and doubled spaces in cells, the project name cased "
                "three ways, and filenames following three different conventions.",
        handling="Strip and collapse whitespace on ingestion; casefold before matching "
                 "entity names; never derive metadata from the filename alone.",
    ),
    "F06": dict(
        family="format",
        name="Mixed scale and percentage notation",
        summary="Likelihood is 'High' in one row, '4' in another and '80%' in a third; "
                "completion shows as 0.35, 35% and '35 pct'.",
        handling="Detect ordinal-vs-numeric-vs-percentage per value and convert to one "
                 "scale; record the interpretation so an answer can cite the raw form.",
    ),
    "F07": dict(
        family="format",
        name="Decimal and thousands separator drift",
        summary="One locale-confused export row writes 1.250,00 and another 1 250, "
                "alongside the file's normal 1,250.00.",
        handling="Sniff the separator per value (last separator wins as decimal) "
                 "instead of a global locale assumption; quarantine what stays ambiguous.",
    ),
    "F08": dict(
        family="format",
        name="UTF-8 BOM and non-ASCII text",
        summary="A CSV starts with a BOM and carries en-dashes, curly quotes and "
                "accented vendor names.",
        handling="Open CSVs as utf-8-sig and keep text as Unicode; normalise dashes and "
                 "quotes only for matching, not for display.",
    ),
    "F09": dict(
        family="format",
        name="Mixed line endings",
        summary="One CSV is written with CRLF line endings and a trailing blank line, "
                "as exported from a Windows finance tool.",
        handling="Read with universal newlines and drop empty trailing records.",
    ),
    "F10": dict(
        family="format",
        name="Repeated page furniture in PDFs",
        summary="Every PDF page repeats a confidentiality banner, project code and "
                "'Page x of y', which lands in the middle of extracted text.",
        handling="Strip repeated header/footer lines during PDF extraction (drop lines "
                 "that recur on most pages) before chunking, so chunks stay clean.",
    ),
    "M01": dict(
        family="missing",
        name="Blank required fields",
        summary="A milestone with no owner, a risk with no review date, a cost line "
                "with no vendor - cells simply left empty.",
        handling="Treat empty as unknown rather than zero; keep the row, flag the gap, "
                 "and say the field is missing rather than inventing a value.",
    ),
    "M02": dict(
        family="missing",
        name="Inconsistent null tokens",
        summary="Absent values are written as TBD, N/A, n/a, -, tbc, unknown and ??? "
                "in the same column.",
        handling="Normalise a null-token list to a real null at ingestion, case-"
                 "insensitively, so counts and aggregates don't treat 'N/A' as a value.",
    ),
    "M03": dict(
        family="missing",
        name="Risk recorded without mitigation",
        summary="At least one high-scoring risk has an empty mitigation/action cell.",
        handling="Surface it as an explicit gap ('no mitigation recorded') - a question "
                 "about mitigations must not silently skip the row.",
    ),
    "M04": dict(
        family="missing",
        name="Spreadsheet title rows and merged cells",
        summary="The workbook opens with a merged title banner and an export stamp "
                "above the real header row, so row 1 is not the header.",
        handling="Detect the header row by scanning for the first row that looks like "
                 "field names; unmerge and forward-fill merged cells before parsing.",
    ),
    "M05": dict(
        family="missing",
        name="Ragged CSV row",
        summary="One free-text field contains an unquoted comma, so that row parses "
                "with one column too many.",
        handling="Parse leniently and repair rows whose field count is off by one by "
                 "re-joining the overflow into the free-text column; log the repair.",
    ),
    "M06": dict(
        family="missing",
        name="Numbers stored as text",
        summary="Some spreadsheet amounts are strings (\" 1,200,000 \", \"1.2m\") rather "
                "than numeric cells.",
        handling="Coerce per cell rather than trusting the column type; keep the raw "
                 "string when coercion fails instead of dropping the row.",
    ),
    "M07": dict(
        family="missing",
        name="Placeholder and truncated text",
        summary="Notes read 'see attached', 'per last month', '???' or stop mid-sentence.",
        handling="Treat placeholder phrases as non-answers; never quote them as the "
                 "substance of a mitigation or decision.",
    ),
    "M08": dict(
        family="missing",
        name="Omitted report section",
        summary="One status report has no budget section at all - the author ran out "
                "of time that month.",
        handling="Answer from the period that does report it and say the section is "
                 "absent, rather than assuming the figure is unchanged or zero.",
    ),
    "M09": dict(
        family="missing",
        name="Trailing empty rows and columns",
        summary="The sheet's used range extends past the data with blank rows and a "
                "stray empty column.",
        handling="Trim wholly empty rows/columns after reading the used range so row "
                 "counts and aggregates are right.",
    ),
    "M10": dict(
        family="missing",
        name="Embedded total row",
        summary="A 'TOTAL' row sits inside the data rows of the export, not in a "
                "separate summary block.",
        handling="Detect and exclude label rows (TOTAL/Subtotal/Grand Total) before "
                 "aggregating, or the numbers double-count.",
    ),
}

FORMAT_FLAWS = [f for f, v in FLAWS.items() if v["family"] == "format"]
MISSING_FLAWS = [f for f, v in FLAWS.items() if v["family"] == "missing"]

NULL_TOKENS = ["TBD", "N/A", "n/a", "-", "tbc", "unknown", "???"]

STATUS_VARIANTS = {
    "green": ["Green", "GREEN", "On Track", "G", "on track"],
    "amber": ["Amber", "AMBER", "At Risk", "Yellow", "A"],
    "red": ["Red", "RED", "Off Track", "Critical", "R"],
}

MILESTONE_VARIANTS = {
    "complete": ["Complete", "Completed", "DONE", "done", "Closed"],
    "in_progress": ["In Progress", "In-progress", "WIP", "in progress", "Started"],
    "not_started": ["Not Started", "Not started", "NS", "pending", "Not yet started"],
    "slipped": ["Slipped", "SLIPPED", "Delayed", "At Risk", "Re-baselined"],
}

CURRENCY_SYMBOLS = {
    "SGD": "S$", "USD": "$", "EUR": "€", "GBP": "£",
    "AUD": "A$", "MYR": "RM", "INR": "₹", "JPY": "¥",
}


# --------------------------------------------------------------------------
# Flaw bookkeeping
# --------------------------------------------------------------------------


class Flaws:
    """Decides which flaws are live and records where each one actually landed.

    Recording matters as much as injecting: MESSINESS.md is generated from
    these occurrences, so it can never drift from what the files contain.
    """

    def __init__(self, enabled: set[str], rng: random.Random) -> None:
        self.enabled = enabled
        self.rng = rng
        self.occurrences: list[dict] = []

    def on(self, flaw_id: str) -> bool:
        return flaw_id in self.enabled

    def hit(self, flaw_id: str, file_label: str, detail: str) -> None:
        if not self.on(flaw_id):
            return
        self.occurrences.append({"flaw": flaw_id, "file": file_label, "detail": detail})

    def pick(self, flaw_id: str, options: list, default=None):
        """Return a seeded choice when the flaw is on, else the tidy default."""
        if not self.on(flaw_id):
            return default if default is not None else options[0]
        return self.rng.choice(options)


# --------------------------------------------------------------------------
# Value formatting - one concept, many renderings
# --------------------------------------------------------------------------

DATE_STYLES = ["iso", "dmy_slash", "dmy_dash_short", "long_month", "month_year", "us_slash"]


def parse_date(value) -> date:
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y", "%d %B %Y", "%B %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {value!r}")


def fmt_date(value: date, style: str) -> str:
    if style == "iso":
        return value.isoformat()
    if style == "dmy_slash":
        return value.strftime("%d/%m/%Y")
    if style == "dmy_dash_short":
        return value.strftime("%d-%b-%y")
    if style == "long_month":
        return value.strftime("%d %B %Y").lstrip("0")
    if style == "month_year":
        return value.strftime("%B %Y")
    if style == "us_slash":
        return value.strftime("%m/%d/%Y")
    return value.isoformat()


MONEY_STYLES = ["plain", "grouped", "symbol_grouped", "compact", "code_suffix", "k_suffix"]


def fmt_money(amount: float, style: str, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency.upper(), currency.upper() + " ")
    negative = amount < 0
    value = abs(amount)
    if style == "plain":
        text = f"{value:.0f}"
    elif style == "grouped":
        text = f"{value:,.0f}"
    elif style == "symbol_grouped":
        text = f"{symbol}{value:,.0f}"
    elif style == "compact":
        text = f"{symbol}{value / 1_000_000:.1f}M" if value >= 100_000 else f"{symbol}{value:,.0f}"
    elif style == "code_suffix":
        text = f"{value:,.0f} {currency.upper()}"
    elif style == "k_suffix":
        text = f"{value / 1000:.0f}k"
    else:
        text = f"{value:,.0f}"
    if not negative:
        return text
    # Finance exports mark negatives with a minus or with parentheses, rarely both.
    return f"({text})" if style in {"grouped", "symbol_grouped", "code_suffix"} else f"-{text}"


def fmt_percent(fraction: float, style: str) -> str:
    if style == "decimal":
        return f"{fraction:.2f}"
    if style == "pct_word":
        return f"{fraction * 100:.0f} pct"
    if style == "bare_number":
        return f"{fraction * 100:.0f}"
    return f"{fraction * 100:.0f}%"


def messy_spacing(text: str, rng: random.Random) -> str:
    """Add the whitespace a human leaves behind: a trailing space, a double space."""
    choice = rng.randint(0, 2)
    if choice == 0:
        return text + " "
    if choice == 1:
        return " " + text
    return re.sub(r"(\s)", r"\1 ", text, count=1)


def case_variant(name: str, index: int) -> str:
    return [name, name.upper(), name.lower(), name.title()][index % 4]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


IMPACT_SCALE = {"very low": 1, "low": 2, "medium": 3, "moderate": 3, "high": 4, "very high": 5}


def risk_score(risk: dict) -> int:
    """Likelihood x impact on a 1-5 scale - the score a register actually carries."""
    likelihood = IMPACT_SCALE.get(str(risk.get("likelihood", "")).strip().lower(), 3)
    impact = IMPACT_SCALE.get(str(risk.get("impact", "")).strip().lower(), 3)
    return likelihood * impact


def export_date(spec: dict) -> date:
    """The date the files pretend to have been exported.

    Derived from the spec rather than from today's clock, so regenerating the
    dataset next month produces the same bytes and the documentation stays true.
    """
    if spec.get("export_date"):
        return parse_date(spec["export_date"])
    return max(parse_date(r["as_of"]) for r in spec["reports"])


def quarter_label(value: date) -> str:
    return f"{value.year}-Q{(value.month - 1) // 3 + 1}"


# --------------------------------------------------------------------------
# Spec loading and validation
# --------------------------------------------------------------------------

REQUIRED_TOP = ["project", "periods", "financials", "risk_register", "reports"]


def validate_spec(spec: dict) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_TOP:
        if key not in spec:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors

    project = spec["project"]
    for key in ("name", "code", "currency"):
        if not project.get(key):
            errors.append(f"project.{key} is required")

    periods = spec["periods"]
    if not isinstance(periods, list) or not periods:
        errors.append("periods must be a non-empty list of period labels")

    lines = spec["financials"].get("lines") or []
    if not lines:
        errors.append("financials.lines must contain at least one cost line")
    for i, line in enumerate(lines):
        if not line.get("category"):
            errors.append(f"financials.lines[{i}].category is required")
        if "budget" not in line:
            errors.append(f"financials.lines[{i}].budget is required")
        for period in (line.get("actuals") or {}):
            if period not in periods:
                errors.append(
                    f"financials.lines[{i}].actuals has period {period!r} "
                    f"which is not in periods"
                )

    risks = spec["risk_register"].get("risks") or []
    if not risks:
        errors.append("risk_register.risks must contain at least one risk")
    risk_ids = {r.get("id") for r in risks}
    for i, risk in enumerate(risks):
        for key in ("id", "title", "owner", "status"):
            if not risk.get(key):
                errors.append(f"risk_register.risks[{i}].{key} is required")

    reports = spec["reports"]
    if not reports:
        errors.append("reports must contain at least one status report")
    for i, report in enumerate(reports):
        for key in ("id", "period", "as_of", "rag", "executive_summary"):
            if not report.get(key):
                errors.append(f"reports[{i}].{key} is required")
        if report.get("period") and report["period"] not in periods:
            errors.append(f"reports[{i}].period {report['period']!r} is not in periods")
        try:
            parse_date(report.get("as_of", ""))
        except ValueError:
            errors.append(f"reports[{i}].as_of is not a parseable date")
        for rid in report.get("highlight_risks", []):
            if rid not in risk_ids:
                errors.append(f"reports[{i}].highlight_risks references unknown risk {rid!r}")
        if report.get("rag", "").lower() not in STATUS_VARIANTS:
            errors.append(f"reports[{i}].rag must be one of green/amber/red")
    return errors


def financial_view(spec: dict) -> dict:
    """Roll the cost lines up once so every output reads from the same numbers."""
    periods = spec["periods"]
    lines = []
    for raw in spec["financials"]["lines"]:
        actuals = {p: float(raw.get("actuals", {}).get(p, 0) or 0) for p in periods}
        budget = float(raw["budget"])
        spent = sum(actuals.values())
        lines.append({
            "cost_code": raw.get("cost_code", ""),
            "category": raw["category"],
            "vendor": raw.get("vendor", ""),
            "budget": budget,
            "actuals": actuals,
            "spent": spent,
            "variance": budget - spent,
            "note": raw.get("note", ""),
        })
    totals = {
        "budget": sum(line["budget"] for line in lines),
        "spent": sum(line["spent"] for line in lines),
        "by_period": {p: sum(line["actuals"][p] for line in lines) for p in periods},
    }
    totals["variance"] = totals["budget"] - totals["spent"]
    return {"periods": periods, "lines": lines, "totals": totals}


def budget_as_of(view: dict, period: str) -> dict:
    """Cumulative position at the end of `period` - what a status report quotes."""
    periods = view["periods"]
    upto = periods[: periods.index(period) + 1]
    per_line = {line["category"]: sum(line["actuals"][p] for p in upto) for line in view["lines"]}
    spent = sum(per_line.values())
    budget = view["totals"]["budget"]
    period_spend = sum(line["actuals"][period] for line in view["lines"])
    # Forecast at completion = approved budget plus the overruns already visible
    # on individual cost lines. A report cannot see future actuals, so neither
    # does this: it only knows which lines have already passed their budget.
    overruns = sum(max(0.0, per_line[line["category"]] - line["budget"]) for line in view["lines"])
    return {
        "budget": budget,
        "spent_to_date": spent,
        "period_spend": period_spend,
        "variance": budget - spent,
        "pct_spent": (spent / budget) if budget else 0.0,
        "forecast": budget + overruns,
        "line_overruns": overruns,
    }


# --------------------------------------------------------------------------
# PDF status reports
# --------------------------------------------------------------------------

from xml.sax.saxutils import escape as _xml_escape  # noqa: E402

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#1a1a1a")
RULE = colors.HexColor("#9aa0a6")
BAND = colors.HexColor("#eceff1")
TEMPLATES = ["classic", "memo", "table"]


def esc(text) -> str:
    return _xml_escape("" if text is None else str(text))


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=16, leading=20, textColor=INK, alignment=0),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=11, leading=14, spaceBefore=10, spaceAfter=4,
                             textColor=INK),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=4),
        "small": ParagraphStyle("s", parent=base["BodyText"], fontName="Helvetica",
                                fontSize=8, leading=10.5, textColor=INK),
        "cell": ParagraphStyle("c", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=8.5, leading=11, textColor=INK, spaceAfter=0),
        "cellb": ParagraphStyle("cb", parent=base["BodyText"], fontName="Helvetica-Bold",
                                fontSize=8.5, leading=11, textColor=colors.white,
                                spaceAfter=0),
    }


def _table(rows, widths, styles, header=True):
    data = [[Paragraph(esc(c), styles["cellb"] if (header and r == 0) else styles["cell"])
             for c in row] for r, row in enumerate(rows)]
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#455a64")))
    else:
        commands.append(("BACKGROUND", (0, 0), (0, -1), BAND))
    table.setStyle(TableStyle(commands))
    return table


class _NumberedCanvas(canvas.Canvas):
    """Draws the repeated page furniture once the total page count is known.

    Page furniture is flaw F10, so it has to be real furniture: drawn outside
    the text frame on every page, exactly as a template would produce it.
    """

    banner = ""
    footer = ""
    furniture = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._page_states: list[dict] = []

    def showPage(self):
        self._page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._page_states)
        for state in self._page_states:
            self.__dict__.update(state)
            self._draw_furniture(total)
            super().showPage()
        super().save()

    def _draw_furniture(self, total: int):
        self.saveState()
        self.setFillColor(RULE)
        if self.furniture:
            self.setFont("Helvetica-Bold", 7)
            self.drawString(18 * mm, A4[1] - 12 * mm, self.banner)
            self.drawRightString(A4[0] - 18 * mm, A4[1] - 12 * mm,
                                 "CONFIDENTIAL - INTERNAL USE ONLY")
            self.setLineWidth(0.3)
            self.setStrokeColor(RULE)
            self.line(18 * mm, A4[1] - 14 * mm, A4[0] - 18 * mm, A4[1] - 14 * mm)
            self.setFont("Helvetica", 7)
            self.drawString(18 * mm, 12 * mm, self.footer)
            self.drawRightString(A4[0] - 18 * mm, 12 * mm,
                                 f"Page {self._pageNumber} of {total}")
        else:
            self.setFont("Helvetica", 7.5)
            self.drawRightString(A4[0] - 18 * mm, 12 * mm, str(self._pageNumber))
        self.restoreState()


def build_pdf(path: Path, story, *, title: str, author: str,
              banner: str, footer: str, furniture: bool) -> None:
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm, title=title, author=author,
        subject="Project status report", creator="render_dataset.py", invariant=True,
    )
    maker = type("_Furnished", (_NumberedCanvas,),
                 {"banner": banner, "footer": footer, "furniture": furniture})
    doc.build(story, canvasmaker=maker)


def _milestone_rows(report, rng, flaws, date_style, label):
    rows = [["Milestone", "Owner", "Due", "Status", "Comment"]]
    milestones = report.get("milestones", [])
    blank_at = rng.randrange(len(milestones)) if milestones else -1
    drift_at = rng.randrange(len(milestones)) if milestones else -1
    for i, milestone in enumerate(milestones):
        owner = milestone.get("owner", "")
        due = milestone.get("due", "")
        status_key = milestone.get("status", "in_progress")
        status = flaws.pick("F03", MILESTONE_VARIANTS.get(status_key, [status_key]),
                            default=status_key.replace("_", " ").title())
        comment = milestone.get("comment", "")

        # A hand-typed row uses a different date convention from the rest (F01).
        row_style = date_style
        if flaws.on("F01") and i == drift_at and due:
            row_style = flaws.rng.choice([s for s in DATE_STYLES if s != date_style])
            flaws.hit("F01", label,
                      f"milestone '{milestone.get('name', '')}' shows its due date as "
                      f"{fmt_date(parse_date(due), row_style)} rather than the "
                      f"{fmt_date(parse_date(due), date_style)} form used by the rest of "
                      f"the table")
        due_text = fmt_date(parse_date(due), row_style) if due else ""

        if i == blank_at and flaws.on("M01"):
            owner = ""
            flaws.hit("M01", label, f"milestone '{milestone.get('name', '')}' has no owner")
        elif not owner and flaws.on("M02"):
            owner = flaws.rng.choice(NULL_TOKENS)
        if not due_text and flaws.on("M02"):
            due_text = flaws.rng.choice(NULL_TOKENS)
            flaws.hit("M02", label, f"milestone due date written as '{due_text}'")
        if flaws.on("F05") and i == (blank_at + 1) % max(len(milestones), 1):
            comment = messy_spacing(comment, flaws.rng) if comment else comment
        rows.append([milestone.get("name", ""), owner, due_text, status, comment])
    return rows


def _risk_rows(report, risks_by_id, flaws, label, currency, money_style):
    rows = [["ID", "Risk", "Owner", "Score", "Mitigation"]]
    for rid in report.get("highlight_risks", []):
        risk = risks_by_id[rid]
        mitigation = risk.get("mitigation", "")
        if not mitigation and flaws.on("M03"):
            mitigation = ""
            flaws.hit("M03", label, f"{rid} is highlighted with no mitigation recorded")
        score = risk.get("score") or risk_score(risk)
        rows.append([rid, risk.get("title", ""), risk.get("owner", ""), str(score),
                     mitigation or ""])
    return rows


def render_reports(spec, view, flaws, out_dir: Path, rng) -> list[dict]:
    styles = _styles()
    project = spec["project"]
    currency = project["currency"]
    risks_by_id = {r["id"]: r for r in spec["risk_register"]["risks"]}
    reports = spec["reports"]
    written: list[dict] = []

    # Pick the report that will be missing its budget section (M08) unless the
    # spec already says which one, so a default spec still shows the flaw.
    omit_target = None
    if flaws.on("M08") and len(reports) >= 2:
        declared = [i for i, r in enumerate(reports) if "budget" in r.get("omit_sections", [])]
        omit_target = declared[0] if declared else len(reports) // 2

    for index, report in enumerate(reports):
        template = report.get("template") or TEMPLATES[index % len(TEMPLATES)]
        date_style = DATE_STYLES[index % len(DATE_STYLES)] if flaws.on("F01") else "iso"
        money_style = MONEY_STYLES[(index + 2) % len(MONEY_STYLES)] if flaws.on("F02") else "grouped"
        pct_style = ["percent", "decimal", "pct_word"][index % 3] if flaws.on("F06") else "percent"
        as_of = parse_date(report["as_of"])
        name = case_variant(project["name"], index) if flaws.on("F05") else project["name"]
        slug = slugify(project["name"])

        # Filename conventions drift the way they do across a real shared drive (F05).
        if flaws.on("F05"):
            filename = [
                f"{project['name'].replace(' ', '_')}_Status_Report_{report['period']}.pdf",
                f"{slug}-status-memo-{as_of.strftime('%b%Y').lower()}.pdf",
                f"{project['code']}_PSR_{as_of.isoformat()}.pdf",
            ][index % 3]
        else:
            filename = f"{slug}_status_report_{report['period']}.pdf"
        path = out_dir / filename
        label = filename

        if flaws.on("F01"):
            flaws.hit("F01", label, f"dates are written like {fmt_date(as_of, date_style)}")
        if flaws.on("F02"):
            flaws.hit("F02", label, "amounts are written like "
                                    f"{fmt_money(view['totals']['budget'], money_style, currency)}")
        if flaws.on("F05"):
            flaws.hit("F05", label, f"project name appears as '{name}'; filename follows a "
                                    f"different convention from the other reports")

        rag_key = report["rag"].lower()
        rag_text = flaws.pick("F03", STATUS_VARIANTS[rag_key], default=rag_key.title())
        if flaws.on("F03"):
            flaws.hit("F03", label, f"overall status written as '{rag_text}'")

        snapshot = budget_as_of(view, report["period"])
        story = []

        # --- header -------------------------------------------------------
        if template == "memo":
            story.append(Paragraph("PROJECT STATUS MEMORANDUM", styles["title"]))
            story.append(Spacer(1, 6))
            meta = [
                ["TO:", ", ".join(report.get("distribution", ["Steering Committee"]))],
                ["FROM:", report.get("author", project.get("manager", ""))],
                ["DATE:", fmt_date(as_of, date_style)],
                ["RE:", f"{name} - {report['period']} status ({rag_text})"],
                ["REF:", report["id"]],
            ]
            story.append(_table(meta, [22 * mm, 150 * mm], styles, header=False))
        elif template == "table":
            story.append(Paragraph(f"{project['code']} / {report['id']}", styles["small"]))
            story.append(Paragraph(f"{name}", styles["title"]))
            story.append(Spacer(1, 4))
            meta = [
                ["Reporting period", report["period"], "Report date", fmt_date(as_of, date_style)],
                ["Overall health", rag_text, "Prepared by", report.get("author", "")],
                ["Client", project.get("client", ""), "Project manager", project.get("manager", "")],
            ]
            story.append(_table(meta, [32 * mm, 54 * mm, 32 * mm, 54 * mm], styles, header=False))
        else:
            story.append(Paragraph(f"{name}", styles["title"]))
            story.append(Paragraph(
                f"Project status report &nbsp;|&nbsp; {esc(report['period'])} "
                f"&nbsp;|&nbsp; {esc(fmt_date(as_of, date_style))}", styles["small"]))
            story.append(Spacer(1, 8))
            story.append(_table([
                ["Report ID", report["id"], "Status", rag_text],
                ["Prepared by", report.get("author", project.get("manager", "")),
                 "Sponsor", project.get("sponsor", "")],
            ], [26 * mm, 60 * mm, 26 * mm, 60 * mm], styles, header=False))
        story.append(Spacer(1, 10))

        # --- narrative ----------------------------------------------------
        story.append(Paragraph("1. Executive summary", styles["h2"]))
        story.append(Paragraph(esc(report["executive_summary"]), styles["body"]))

        if report.get("progress"):
            story.append(Paragraph("2. Progress this period", styles["h2"]))
            for item in report["progress"]:
                story.append(Paragraph(f"&bull;&nbsp; {esc(item)}", styles["body"]))

        if report.get("milestones"):
            story.append(Paragraph("3. Milestones", styles["h2"]))
            rows = _milestone_rows(report, rng, flaws, date_style, label)
            story.append(_table(rows, [52 * mm, 26 * mm, 26 * mm, 24 * mm, 46 * mm], styles))

        # --- budget (sometimes absent) ---------------------------------------
        if index == omit_target:
            flaws.hit("M08", label, "the budget / financial position section is absent "
                                    "from this report")
        else:
            story.append(Paragraph("4. Financial position", styles["h2"]))
            pct = fmt_percent(snapshot["pct_spent"], pct_style)
            if flaws.on("F06"):
                flaws.hit("F06", label, f"percentage spent written as '{pct}'")
            if template == "memo":
                story.append(Paragraph(
                    f"Approved budget stands at {esc(fmt_money(snapshot['budget'], money_style, currency))} "
                    f"with {esc(fmt_money(snapshot['spent_to_date'], money_style, currency))} spent to "
                    f"{esc(fmt_date(as_of, date_style))} ({esc(pct)} of budget). Spend in the period was "
                    f"{esc(fmt_money(snapshot['period_spend'], money_style, currency))}; variance against "
                    f"budget is {esc(fmt_money(snapshot['variance'], money_style, currency))}.",
                    styles["body"]))
            else:
                rows = [
                    ["Measure", "Amount", "Notes"],
                    ["Approved budget", fmt_money(snapshot["budget"], money_style, currency),
                     report.get("budget_note", "")],
                    ["Spend to date", fmt_money(snapshot["spent_to_date"], money_style, currency),
                     f"{pct} of approved budget"],
                    ["Spend this period", fmt_money(snapshot["period_spend"], money_style, currency), ""],
                    ["Variance", fmt_money(snapshot["variance"], money_style, currency),
                     "Under budget" if snapshot["variance"] >= 0 else "Over budget"],
                    ["Forecast at completion", fmt_money(snapshot["forecast"], money_style, currency),
                     report.get("forecast_note", "")],
                ]
                story.append(_table(rows, [42 * mm, 44 * mm, 88 * mm], styles))

        # --- risks and issues -------------------------------------------------
        if report.get("highlight_risks"):
            story.append(Paragraph("5. Key risks", styles["h2"]))
            rows = _risk_rows(report, risks_by_id, flaws, label, currency, money_style)
            story.append(_table(rows, [18 * mm, 54 * mm, 26 * mm, 18 * mm, 58 * mm], styles))

        if report.get("issues"):
            story.append(Paragraph("6. Issues and decisions required", styles["h2"]))
            issues = list(report["issues"])
            if flaws.on("M07") and issues:
                target = rng.randrange(len(issues))
                issues[target] = flaws.rng.choice([
                    "Vendor commercials - see attached.",
                    "Carried over, per last month.",
                    "Awaiting confirmation from the infrastructure team on the",
                ])
                flaws.hit("M07", label, f"issue {target + 1} is a placeholder or stops mid-sentence")
            for item in issues:
                story.append(Paragraph(f"&bull;&nbsp; {esc(item)}", styles["body"]))

        if report.get("next_period"):
            story.append(Paragraph("7. Focus for next period", styles["h2"]))
            for item in report["next_period"]:
                story.append(Paragraph(f"&bull;&nbsp; {esc(item)}", styles["body"]))

        story.append(Spacer(1, 12))
        story.append(Paragraph(
            f"Prepared by {esc(report.get('author', project.get('manager', '')))} "
            f"&nbsp;|&nbsp; {esc(report['id'])} &nbsp;|&nbsp; "
            f"issued {esc(fmt_date(as_of, date_style))}", styles["small"]))

        banner = f"{project['code']} - {name} - {report['period']}"
        footer = f"{report['id']} - issued {fmt_date(as_of, date_style)}"
        build_pdf(path, story,
                  title=f"{project['name']} status report {report['period']}",
                  author=report.get("author", project.get("manager", "")),
                  banner=banner, footer=footer, furniture=flaws.on("F10"))
        if flaws.on("F10"):
            flaws.hit("F10", label, "confidentiality banner, project code and 'Page x of y' "
                                    "repeat on every page")
        written.append({
            "file": filename,
            "type": "PDF",
            "purpose": f"Status report - {report['period']} ({report['id']})",
            "template": template,
        })
    return written


# --------------------------------------------------------------------------
# Financial summary (XLSX / CSV)
# --------------------------------------------------------------------------

from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

HEADER_FILL = PatternFill("solid", fgColor="455A64")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=10)


def _header_variants(canonical: list[str], flaws: Flaws, label: str) -> list[str]:
    """Return the header row a real file would carry: same fields, uneven spelling."""
    if not flaws.on("F04"):
        return canonical
    styled = []
    for i, head in enumerate(canonical):
        mode = flaws.rng.randint(0, 3)
        if mode == 0:
            styled.append(head)
        elif mode == 1:
            styled.append(head.lower().replace(" ", "_"))
        elif mode == 2:
            styled.append(head.upper())
        else:
            styled.append(head + " ")
    flaws.hit("F04", label, "header row mixes Title Case, snake_case and UPPER CASE "
                            "spellings of the same fields")
    return styled


def save_workbook(wb, path: Path, when: date) -> None:
    """Save a workbook that is byte-identical on every run.

    Two things vary otherwise: the document properties openpyxl stamps with the
    current time, and the entry timestamps inside the xlsx zip container. Both
    are pinned to the dataset's export date, so a regenerated workbook can be
    diffed against the committed one.
    """
    moment = datetime(when.year, when.month, when.day, 9, 0, 0)
    wb.properties.created = moment
    wb.properties.modified = moment
    wb.save(path)

    stamp = (when.year, when.month, when.day, 9, 0, 0)
    iso = f"{when.isoformat()}T09:00:00Z".encode()
    with zipfile.ZipFile(path) as archive:
        entries = [(item, archive.read(item.filename)) for item in archive.infolist()]
    # openpyxl rewrites dcterms:modified with the wall clock as it saves, which
    # would otherwise leave every regenerated workbook a few bytes different.
    entries = [
        (item, re.sub(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:)",
                      lambda m: m.group(1) + iso + m.group(2), payload)
         if item.filename == "docProps/core.xml" else payload)
        for item, payload in entries
    ]
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for item, payload in entries:
            info = zipfile.ZipInfo(item.filename, date_time=stamp)
            info.compress_type = item.compress_type
            info.external_attr = item.external_attr
            info.create_system = item.create_system
            archive.writestr(info, payload)
    temporary.replace(path)


def render_financials(spec, view, flaws, out_dir: Path, rng) -> list[dict]:
    project = spec["project"]
    currency = project["currency"].upper()
    slug = slugify(project["name"])
    formats = [f.lower() for f in spec["financials"].get("formats", ["xlsx", "csv"])]
    fiscal = spec["financials"].get("fiscal_label", f"FY{str(view['periods'][-1])[2:4]}")
    written: list[dict] = []
    lines = view["lines"]
    periods = view["periods"]

    # Deterministic targets for the row-level flaws.
    text_row = rng.randrange(len(lines))
    euro_row = (text_row + 1) % len(lines)
    blank_row = (text_row + 2) % len(lines)
    note_row = (text_row + 3) % len(lines)

    if "xlsx" in formats:
        filename = f"{project['name'].replace(' ', '_')}_Financial_Summary_{fiscal}.xlsx"
        path = out_dir / filename
        wb = Workbook()
        ws = wb.active
        ws.title = "Summary"

        row_cursor = 1
        if flaws.on("M04"):
            ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=8)
            ws.cell(row=1, column=1, value=f"{project['name']} - cost summary {fiscal}").font = \
                Font(bold=True, size=13)
            ws.cell(row=1, column=1).alignment = Alignment(horizontal="center")
            ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=8)
            ws.cell(row=2, column=1,
                    value=f"Exported from FinOps on {fmt_date(export_date(spec), 'dmy_slash')} "
                          f"by {project.get('finance_contact', 'Finance Ops')} - "
                          f"all amounts in {currency} unless stated")
            ws.cell(row=2, column=1).font = Font(italic=True, size=9)
            row_cursor = 4  # row 3 left blank, as these exports always are
            flaws.hit("M04", filename, "rows 1-2 are a merged title banner and an export "
                                       "stamp; the real header row is row 4")

        canonical = ["Cost Code", "Category", "Vendor", f"Approved Budget ({currency})",
                     "Actual To Date", "Variance", "% Spent", "Notes"]
        headers = _header_variants(canonical, flaws, filename)
        for col, head in enumerate(headers, start=1):
            cell = ws.cell(row=row_cursor, column=col, value=head)
            cell.font = HEADER_FONT
            cell.fill = HEADER_FILL
        header_row = row_cursor
        row_cursor += 1

        for i, line in enumerate(lines):
            vendor = line["vendor"]
            note = line["note"]
            budget: object = line["budget"]
            spent: object = line["spent"]

            if flaws.on("M06") and i == text_row:
                budget = f" {line['budget']:,.0f} "
                flaws.hit("M06", filename,
                          f"the approved budget for '{line['category']}' is the text "
                          f"'{budget}' rather than a number")
            if flaws.on("F07") and i == euro_row:
                spent = f"{line['spent']:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")
                flaws.hit("F07", filename,
                          f"actuals for '{line['category']}' use European separators ({spent})")
            if flaws.on("M01") and i == blank_row:
                vendor = ""
                flaws.hit("M01", filename, f"'{line['category']}' has no vendor recorded")
            if flaws.on("M02") and i == note_row:
                note = flaws.rng.choice(NULL_TOKENS)
                flaws.hit("M02", filename, f"an empty note is written as '{note}'")
            if flaws.on("M07") and i == (note_row + 1) % len(lines):
                note = flaws.rng.choice(["see attached", "per last month", "???",
                                         "awaiting vendor confirmation of the"])
                flaws.hit("M07", filename, f"the note for '{line['category']}' is a "
                                           f"placeholder ('{note}')")
            if flaws.on("F05") and i == euro_row:
                vendor = messy_spacing(vendor, flaws.rng) if vendor else vendor
                flaws.hit("F05", filename, "vendor names carry stray leading/trailing spaces")

            values = [line["cost_code"], line["category"], vendor, budget, spent,
                      line["variance"],
                      round(line["spent"] / line["budget"], 4) if line["budget"] else 0,
                      note]
            for col, value in enumerate(values, start=1):
                ws.cell(row=row_cursor, column=col, value=value)
            row_cursor += 1

        if flaws.on("M10"):
            ws.cell(row=row_cursor, column=2, value="TOTAL")
            ws.cell(row=row_cursor, column=2).font = Font(bold=True)
            ws.cell(row=row_cursor, column=4, value=view["totals"]["budget"])
            ws.cell(row=row_cursor, column=5, value=view["totals"]["spent"])
            ws.cell(row=row_cursor, column=6, value=view["totals"]["variance"])
            flaws.hit("M10", filename, f"a bold TOTAL row sits at row {row_cursor} inside "
                                       f"the data block, not in a separate summary area")
            row_cursor += 1

        if flaws.on("M09"):
            ws.cell(row=row_cursor + 3, column=10, value=None)
            ws.cell(row=row_cursor + 3, column=10, value="")
            flaws.hit("M09", filename, "the used range runs past the data with blank rows "
                                       "and an empty column J")

        widths = [12, 26, 22, 20, 18, 16, 10, 38]
        for col, width in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(col)].width = width
        ws.freeze_panes = ws.cell(row=header_row + 1, column=1)

        # --- second sheet: actuals by period -----------------------------
        ws2 = wb.create_sheet("By Period")
        period_labels = []
        for i, period in enumerate(periods):
            if flaws.on("F01"):
                year, quarter = period.split("-Q")
                period_labels.append([period, f"Q{quarter} {year}", f"{quarter}Q{year[2:]}",
                                      f"Qtr {quarter} {year}"][i % 4])
            else:
                period_labels.append(period)
        if flaws.on("F01"):
            flaws.hit("F01", filename, "the 'By Period' sheet labels the same quarters "
                                       f"{', '.join(period_labels[:3])} and so on")
        ws2.append(["Category"] + period_labels + ["Total"])
        for cell in ws2[1]:
            cell.font = HEADER_FONT
            cell.fill = HEADER_FILL
        for line in lines:
            ws2.append([line["category"]] + [line["actuals"][p] for p in periods] + [line["spent"]])
        ws2.column_dimensions["A"].width = 26

        # --- third sheet: notes ------------------------------------------
        ws3 = wb.create_sheet("Notes")
        ws3.append(["Date", "Author", "Note"])
        for cell in ws3[1]:
            cell.font = HEADER_FONT
            cell.fill = HEADER_FILL
        for i, note in enumerate(spec["financials"].get("notes", [])):
            style = DATE_STYLES[i % len(DATE_STYLES)] if flaws.on("F01") else "iso"
            when = parse_date(note["date"]) if isinstance(note, dict) else date.today()
            text = note["text"] if isinstance(note, dict) else str(note)
            author = note.get("author", "") if isinstance(note, dict) else ""
            ws3.append([fmt_date(when, style), author, text])
        ws3.column_dimensions["A"].width = 16
        ws3.column_dimensions["B"].width = 18
        ws3.column_dimensions["C"].width = 90

        save_workbook(wb, path, export_date(spec))
        written.append({"file": filename, "type": "XLSX",
                        "purpose": f"Financial summary - budget vs actual by cost line and period"})

    if "csv" in formats:
        filename = f"{slug}_financials_export.csv"
        path = out_dir / filename
        canonical = ["Cost Code", "Category", "Vendor", "Approved Budget", "Actual To Date",
                     "Variance", "% Spent", "Notes"]
        # A different system exported this one, so it names its columns differently.
        headers = (["cost_code", "cost category", "SUPPLIER", f"budget_{currency.lower()}",
                    "actual_to_date", "variance", "pct_spent", "comments"]
                   if flaws.on("F04") else canonical)
        if flaws.on("F04"):
            flaws.hit("F04", filename, "the same fields are named differently here than in "
                                       "the workbook (SUPPLIER vs Vendor, comments vs Notes)")
        rows = [headers]
        for i, line in enumerate(lines):
            budget = f"{line['budget']:.0f}"
            spent = f"{line['spent']:.0f}"
            if flaws.on("F02") and i == text_row:
                budget = fmt_money(line["budget"], "compact", currency)
                spent = fmt_money(line["spent"], "k_suffix", currency)
                flaws.hit("F02", filename, f"'{line['category']}' is reported as {budget} / "
                                           f"{spent} while every other row is plain digits")
            variance = f"{line['variance']:.0f}"
            if flaws.on("F02") and line["variance"] < 0:
                variance = f"({abs(line['variance']):,.0f})"
                flaws.hit("F02", filename, "overspends are written in parentheses rather "
                                           "than with a minus sign")
            pct = fmt_percent(line["spent"] / line["budget"] if line["budget"] else 0,
                              "decimal" if (flaws.on("F06") and i % 2) else "percent")
            vendor = line["vendor"]
            if flaws.on("M01") and i == blank_row:
                vendor = ""
            rows.append([line["cost_code"], line["category"], vendor, budget, spent,
                         variance, pct, line["note"]])
        if flaws.on("F06"):
            flaws.hit("F06", filename, "'% spent' alternates between 0.42 and 42% down the "
                                       "same column")
        if flaws.on("M10"):
            rows.append(["", "TOTAL", "", f"{view['totals']['budget']:.0f}",
                         f"{view['totals']['spent']:.0f}", f"{view['totals']['variance']:.0f}",
                         "", ""])
            flaws.hit("M10", filename, "the last data row is a TOTAL row with the same shape "
                                       "as a cost line")

        newline_mode = "\r\n" if flaws.on("F09") else "\n"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator=newline_mode)
            writer.writerows(rows)
            if flaws.on("F09"):
                handle.write(newline_mode)
        if flaws.on("F09"):
            flaws.hit("F09", filename, "CRLF line endings and a trailing blank line, as "
                                       "written by a Windows finance tool")
        written.append({"file": filename, "type": "CSV",
                        "purpose": "Financial summary - flat export of the cost lines"})
    return written


# --------------------------------------------------------------------------
# Risk register (exactly one)
# --------------------------------------------------------------------------

OPEN_STATES = {"closed", "close", "mitigated", "resolved", "done", "retired"}


def _risk_table(spec, view, flaws, rng, currency, label, for_excel: bool):
    risks = spec["risk_register"]["risks"]
    canonical = ["Risk ID", "Risk Title", "Description", "Category", "Likelihood", "Impact",
                 "Risk Score", "Owner", "Status", "Mitigation", "Date Raised", "Next Review",
                 f"Exposure ({currency})"]
    if flaws.on("F04"):
        headers = ["Risk ID", "risk_title", "Description ", "CATEGORY", "likelihood", "Impact",
                   "Risk Score", "Risk Owner", "status", "Mitigation / Action", "raised",
                   "review date", "Exposure"]
        flaws.hit("F04", label, "columns are named differently from the status reports "
                                "(Risk Owner vs Owner, raised vs Date Raised) and mix "
                                "snake_case with Title Case")
    else:
        headers = canonical

    no_mitigation = next((i for i, r in enumerate(risks) if not r.get("mitigation")), None)
    if no_mitigation is None and flaws.on("M03") and risks:
        scored = sorted(range(len(risks)), key=lambda i: -float(risks[i].get("exposure", 0) or 0))
        no_mitigation = scored[0]
    blank_owner = rng.randrange(len(risks)) if risks else -1
    ragged = rng.randrange(len(risks)) if risks else -1
    nonascii = (ragged + 1) % len(risks) if risks else -1

    rows = [headers]
    for i, risk in enumerate(risks):
        date_style = DATE_STYLES[i % len(DATE_STYLES)] if flaws.on("F01") else "iso"
        description = risk.get("description", "")
        mitigation = risk.get("mitigation", "")
        owner = risk.get("owner", "")
        status_key = risk.get("status", "Open")
        review = risk.get("review_date", "")
        exposure = float(risk.get("exposure", 0) or 0)

        status = status_key
        if flaws.on("F03"):
            variants = [status_key, status_key.upper(), status_key.lower(), status_key.title()]
            status = flaws.rng.choice(variants)

        likelihood = risk.get("likelihood", "")
        if flaws.on("F06"):
            scale = {"low": ("Low", "1", "20%"), "medium": ("Medium", "3", "50%"),
                     "high": ("High", "4", "80%"), "very high": ("Very High", "5", "90%")}
            options = scale.get(str(likelihood).lower())
            if options:
                likelihood = options[i % 3]

        if flaws.on("M03") and i == no_mitigation:
            mitigation = ""
            flaws.hit("M03", label, f"{risk['id']} ('{risk.get('title', '')}') is open with "
                                    f"no mitigation recorded")
        elif flaws.on("M07") and i == (no_mitigation or 0) + 1 and i < len(risks):
            mitigation = flaws.rng.choice(["see attached", "as previously agreed", "???"])
            flaws.hit("M07", label, f"the mitigation for {risk['id']} is a placeholder "
                                    f"('{mitigation}')")
        if flaws.on("M01") and i == blank_owner:
            owner = ""
            flaws.hit("M01", label, f"{risk['id']} has no owner recorded")
        if flaws.on("M02") and not review:
            review = flaws.rng.choice(NULL_TOKENS)
            flaws.hit("M02", label, f"the next review date for {risk['id']} is '{review}'")
        elif review:
            review = fmt_date(parse_date(review), date_style)
        if flaws.on("F05") and i == blank_owner - 1:
            description = messy_spacing(description, flaws.rng)
            flaws.hit("F05", label, f"the description for {risk['id']} has stray whitespace")
        if flaws.on("F08") and i == nonascii:
            description = description.replace(" - ", " – ").replace("'", "’")
            if "–" not in description and "’" not in description:
                description = description.rstrip(".") + " – raised at the vendor review."
            flaws.hit("F08", label, f"the description for {risk['id']} contains an en-dash "
                                    f"and a curly apostrophe")

        exposure_text: object = exposure
        if flaws.on("F02"):
            style = MONEY_STYLES[i % len(MONEY_STYLES)]
            exposure_text = fmt_money(exposure, style, currency)
        elif for_excel:
            exposure_text = exposure

        score: object = risk.get("score") or risk_score(risk)
        if flaws.on("M01") and i == (blank_owner + 2) % len(risks):
            score = ""
            flaws.hit("M01", label, f"{risk['id']} has no risk score")
        elif flaws.on("F06") and i % 3 == 1:
            # Some rows record the score as an L/I code instead of the product.
            score = (f"{str(risk.get('likelihood', ''))[:1].upper()}"
                     f"/{str(risk.get('impact', ''))[:1].upper()}")

        rows.append([risk["id"], risk.get("title", ""), description, risk.get("category", ""),
                     likelihood, risk.get("impact", ""), score, owner, status, mitigation,
                     fmt_date(parse_date(risk["raised_date"]), date_style)
                     if risk.get("raised_date") else "", review, exposure_text])
    if flaws.on("F01"):
        flaws.hit("F01", label, "every row carries its own date convention, because rows "
                                "were pasted in from different trackers")
    if flaws.on("F02"):
        flaws.hit("F02", label, "exposure is written as plain digits, S$1.2M, 1200k and "
                                "'1,200,000 SGD' in different rows")
    if flaws.on("F03"):
        flaws.hit("F03", label, "status values appear as Open / OPEN / open / Closed / CLOSED")
    if flaws.on("F06"):
        flaws.hit("F06", label, "likelihood is recorded as 'High', '4' and '80%' for the "
                                "same level of risk")
    return rows, ragged


def render_risk_register(spec, view, flaws, out_dir: Path, rng) -> list[dict]:
    project = spec["project"]
    currency = project["currency"].upper()
    slug = slugify(project["name"])
    fmt = spec["risk_register"].get("format", "csv").lower()
    version = spec["risk_register"].get("version", "v3")

    if fmt == "xlsx":
        filename = f"{slug}_risk_register_{version}.xlsx"
        path = out_dir / filename
        rows, _ = _risk_table(spec, view, flaws, rng, currency, filename, for_excel=True)
        wb = Workbook()
        ws = wb.active
        ws.title = "Risk Register"
        start = 1
        if flaws.on("M04"):
            ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(rows[0]))
            ws.cell(row=1, column=1,
                    value=f"{project['name']} - Risk Register ({version})").font = Font(bold=True, size=13)
            ws.cell(row=2, column=1, value=f"Maintained by the PMO - last reviewed "
                                           f"{fmt_date(export_date(spec), 'dmy_slash')}")
            start = 4
            flaws.hit("M04", filename, "a merged title banner and a 'last reviewed' stamp sit "
                                       "above the header row")
        for r, row in enumerate(rows):
            for c, value in enumerate(row, start=1):
                cell = ws.cell(row=start + r, column=c, value=value)
                if r == 0:
                    cell.font = HEADER_FONT
                    cell.fill = HEADER_FILL
        if flaws.on("M09"):
            ws.cell(row=start + len(rows) + 3, column=len(rows[0]) + 2, value="")
            flaws.hit("M09", filename, "blank rows and an empty column extend the used range")
        for col, width in enumerate([12, 34, 52, 16, 14, 12, 12, 20, 14, 46, 14, 14, 16], start=1):
            ws.column_dimensions[get_column_letter(col)].width = width
        ws.freeze_panes = ws.cell(row=start + 1, column=1)
        save_workbook(wb, path, export_date(spec))
        return [{"file": filename, "type": "XLSX", "purpose": "Risk register"}]

    filename = f"{slug}_risk_register_{version}.csv"
    path = out_dir / filename
    rows, ragged = _risk_table(spec, view, flaws, rng, currency, filename, for_excel=False)

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows([[str(v) for v in row] for row in rows])
    lines = buffer.getvalue().split("\n")

    if flaws.on("M05") and 0 <= ragged < len(rows) - 1:
        target = ragged + 1  # +1 for the header line
        description = str(rows[target][2])
        if "," not in description:
            description = description.rstrip(".") + ", pending PMO confirmation"
            rows[target][2] = description
            rewritten = io.StringIO()
            csv.writer(rewritten, lineterminator="\n").writerows(
                [[str(v) for v in row] for row in rows])
            lines = rewritten.getvalue().split("\n")
        quoted = '"' + description.replace('"', '""') + '"'
        if quoted in lines[target]:
            lines[target] = lines[target].replace(quoted, description, 1)
            flaws.hit("M05", filename, f"the description for {rows[target][0]} contains an "
                                       f"unquoted comma, so that row parses with one extra column")

    encoding = "utf-8-sig" if flaws.on("F08") else "utf-8"
    if flaws.on("F08"):
        flaws.hit("F08", filename, "the file starts with a UTF-8 BOM, so a naive reader sees "
                                   "'\\ufeffRisk ID' as the first column name")
    with open(path, "w", encoding=encoding, newline="") as handle:
        handle.write("\n".join(lines))
    return [{"file": filename, "type": "CSV", "purpose": "Risk register"}]


# --------------------------------------------------------------------------
# Ground truth, manifest and documentation
# --------------------------------------------------------------------------


def ground_truth_facts(spec, view, files, flaws) -> list[dict]:
    """Answers a reader can verify, so the dataset can seed an evaluation set."""
    project = spec["project"]
    currency = project["currency"].upper()
    risks = spec["risk_register"]["risks"]
    reports = spec["reports"]
    fin_files = [f["file"] for f in files if "Financial" in f["purpose"]]
    risk_files = [f["file"] for f in files if f["purpose"].startswith("Risk register")]
    report_files = [f["file"] for f in files if f["type"] == "PDF"]
    last = reports[-1]
    snapshot = budget_as_of(view, last["period"])
    biggest = max(view["lines"], key=lambda line: line["budget"])
    facts = [
        {"question": f"What is the approved budget for {project['name']}?",
         "answer": f"{view['totals']['budget']:,.0f} {currency}",
         "sources": fin_files,
         "watch_out": "The same figure is written as plain digits, with separators and in "
                      "compact form (F02)."},
        {"question": f"How much had been spent by the end of {last['period']}?",
         "answer": f"{snapshot['spent_to_date']:,.0f} {currency} "
                   f"({snapshot['pct_spent'] * 100:.0f}% of budget)",
         "sources": fin_files + report_files[-1:],
         "watch_out": "A TOTAL row sits inside the cost lines; including it double-counts (M10)."},
        {"question": f"Is the project over or under budget as at {last['period']}?",
         "answer": ("Under budget by " if snapshot["variance"] >= 0 else "Over budget by ")
                   + f"{abs(snapshot['variance']):,.0f} {currency}",
         "sources": fin_files,
         "watch_out": "Overspends appear in parentheses rather than with a minus sign (F02)."},
        {"question": "Which cost category carries the largest approved budget?",
         "answer": f"{biggest['category']} ({biggest['budget']:,.0f} {currency})",
         "sources": fin_files,
         "watch_out": "One budget cell is stored as text (M06)."},
        {"question": "How many risks are currently open?",
         "answer": str(sum(1 for r in risks if str(r.get("status", "")).lower() not in OPEN_STATES)),
         "sources": risk_files,
         "watch_out": "Status casing varies (Open / OPEN / open) so matching must be "
                      "case-insensitive (F03)."},
    ]
    with_exposure = [r for r in risks if r.get("exposure")]
    if with_exposure:
        top = max(with_exposure, key=lambda r: float(r["exposure"]))
        facts.append({
            "question": "Which risk has the highest financial exposure, and who owns it?",
            "answer": f"{top['id']} - {top.get('title', '')} "
                      f"({float(top['exposure']):,.0f} {currency}), owned by {top.get('owner', '')}",
            "sources": risk_files,
            "watch_out": "Exposure is written in several notations, so values need parsing "
                         "before they can be compared (F02).",
        })
    missing_mitigation = [o["detail"].split(" ")[0] for o in flaws.occurrences if o["flaw"] == "M03"]
    if missing_mitigation:
        facts.append({
            "question": "Which risks have no mitigation recorded?",
            "answer": ", ".join(sorted(set(missing_mitigation))),
            "sources": risk_files,
            "watch_out": "The cell is empty rather than marked; the gap must be reported, "
                         "not skipped (M03).",
        })
    facts.append({
        "question": "What was the overall status in each reporting period?",
        "answer": "; ".join(f"{r['period']}: {r['rag'].title()}" for r in reports),
        "sources": report_files,
        "watch_out": "Each report spells its status differently (Green / On Track / G) (F03).",
    })
    omitted = [o for o in flaws.occurrences if o["flaw"] == "M08"]
    if omitted:
        facts.append({
            "question": f"What does {omitted[0]['file']} say about the budget?",
            "answer": "Nothing - that report has no financial section; the figure has to come "
                      "from another period or from the financial summary.",
            "sources": [omitted[0]["file"]],
            "watch_out": "A confident answer here is a hallucination (M08).",
        })
    return facts


def messiness_doc(spec, files, flaws, seed: int) -> str:
    project = spec["project"]
    stamp = export_date(spec).isoformat()
    order = {f["file"]: i for i, f in enumerate(files)}
    grouped: dict[tuple[str, str], list[str]] = {}
    for occurrence in flaws.occurrences:
        key = (occurrence["file"], occurrence["flaw"])
        grouped.setdefault(key, []).append(occurrence["detail"])

    lines = [
        f"# Sample data - {project['name']}",
        "",
        f"Synthetic project documents generated by `render_dataset.py` (seed `{seed}`), "
        f"reporting as at {stamp}. Regenerating from the same spec and seed reproduces "
        "the documents byte for byte.",
        "",
        "| File | Type | Purpose |",
        "|---|---|---|",
    ]
    for entry in files:
        lines.append(f"| `{entry['file']}` | {entry['type']} | {entry['purpose']} |")
    lines += [
        "",
        "## Intentional messiness",
        "",
        "Every row was recorded as the file was written, so this table describes what the "
        "data actually contains. The handling column is the pipeline's contract: if a "
        "behaviour is not implemented yet, say so rather than deleting the row.",
        "",
        "| File | Issue | How it is handled |",
        "|---|---|---|",
    ]
    for (file_name, flaw_id), details in sorted(
        grouped.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])
    ):
        flaw = FLAWS[flaw_id]
        detail = "; ".join(dict.fromkeys(details))
        lines.append(
            f"| `{file_name}` | **{flaw_id} {flaw['name']}** - {detail} | {flaw['handling']} |"
        )
    lines += [
        "",
        "## Regenerating",
        "",
        "```bash",
        f"uv run render_dataset.py <spec>.json --out <dir> --seed {seed}",
        "```",
        "",
        "Pass `--flaws none` for a clean control copy of the same dataset, which is useful "
        "for isolating whether a pipeline problem comes from the data or from the code.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def resolve_flaws(selector) -> set[str]:
    """Accept 'all', 'none', a family name, ids, and '-ID' to subtract from all."""
    if selector is None:
        return set(FLAWS)
    if isinstance(selector, list):
        tokens = [str(t).strip() for t in selector]
    else:
        tokens = [t.strip() for t in str(selector).split(",") if t.strip()]
    if not tokens:
        return set(FLAWS)

    enabled: set[str] = set()
    if tokens[0].startswith("-"):
        enabled = set(FLAWS)
    for token in tokens:
        negate = token.startswith("-")
        name = token.lstrip("-")
        if name.lower() == "all":
            target = set(FLAWS)
        elif name.lower() == "none":
            target = set()
            enabled = set()
            continue
        elif name.lower() == "format":
            target = set(FORMAT_FLAWS)
        elif name.lower() in {"missing", "malformed"}:
            target = set(MISSING_FLAWS)
        elif name.upper() in FLAWS:
            target = {name.upper()}
        else:
            raise SystemExit(f"unknown flaw selector: {token!r} "
                             f"(try --list-flaws to see the catalogue)")
        enabled = enabled - target if negate else enabled | target
    return enabled


def print_catalogue() -> None:
    for family, title in (("format", "Format inconsistency"), ("missing", "Missing and malformed")):
        print(f"\n{title}")
        print("-" * len(title))
        for flaw_id, flaw in FLAWS.items():
            if flaw["family"] == family:
                print(f"  {flaw_id}  {flaw['name']}")
                print(f"        {flaw['summary']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render synthetic project documents (status report PDFs, financial "
                    "summary, risk register) with documented, reproducible data flaws.")
    parser.add_argument("spec", nargs="?", help="path to the dataset spec JSON")
    parser.add_argument("--out", help="output directory (created if missing)")
    parser.add_argument("--seed", type=int, help="RNG seed; overrides spec.seed (default 7)")
    parser.add_argument("--flaws", help="all | none | format | missing | F01,M03 | all,-M08")
    parser.add_argument("--validate-only", action="store_true",
                        help="check the spec and exit without writing files")
    parser.add_argument("--list-flaws", action="store_true", help="print the flaw catalogue")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.list_flaws:
        print_catalogue()
        return 0
    if not args.spec:
        parser.error("a spec file is required (or use --list-flaws)")

    spec_path = Path(args.spec).expanduser()
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"spec not found: {spec_path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"spec is not valid JSON: {exc}", file=sys.stderr)
        return 2

    errors = validate_spec(spec)
    if errors:
        print(f"{len(errors)} problem(s) in {spec_path}:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    if args.validate_only:
        print(f"{spec_path}: valid ({len(spec['reports'])} reports, "
              f"{len(spec['financials']['lines'])} cost lines, "
              f"{len(spec['risk_register']['risks'])} risks)")
        return 0
    if not args.out:
        parser.error("--out is required")

    seed = args.seed if args.seed is not None else int(spec.get("seed", 7))
    enabled = resolve_flaws(args.flaws if args.flaws is not None else spec.get("flaws"))
    rng = random.Random(seed)
    flaws = Flaws(enabled, rng)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    view = financial_view(spec)
    files: list[dict] = []
    files += render_reports(spec, view, flaws, out_dir, rng)
    files += render_financials(spec, view, flaws, out_dir, rng)
    files += render_risk_register(spec, view, flaws, out_dir, rng)

    facts = ground_truth_facts(spec, view, files, flaws)
    by_file: dict[str, list[str]] = {}
    for occurrence in flaws.occurrences:
        by_file.setdefault(occurrence["file"], [])
        if occurrence["flaw"] not in by_file[occurrence["file"]]:
            by_file[occurrence["file"]].append(occurrence["flaw"])

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "render_dataset.py",
        "spec": str(spec_path),
        "seed": seed,
        "project": spec["project"],
        "periods": view["periods"],
        "files": files,
        "totals": {
            "approved_budget": view["totals"]["budget"],
            "actual_to_date": view["totals"]["spent"],
            "variance": view["totals"]["variance"],
            "by_period": view["totals"]["by_period"],
            "risks": len(spec["risk_register"]["risks"]),
        },
        "flaws_enabled": sorted(enabled),
        "flaws_by_file": by_file,
        "flaw_occurrences": flaws.occurrences,
        "flaw_catalogue": {fid: {k: v for k, v in flaw.items()} for fid, flaw in FLAWS.items()
                           if fid in enabled},
        "ground_truth_facts": facts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "MESSINESS.md").write_text(
        messiness_doc(spec, files, flaws, seed), encoding="utf-8")

    if not args.quiet:
        print(f"Wrote {len(files)} document(s) to {out_dir}:")
        for entry in files:
            marks = ", ".join(by_file.get(entry["file"], [])) or "clean"
            print(f"  {entry['file']:<52} {entry['type']:<5} [{marks}]")
        print(f"  manifest.json  ({len(facts)} ground-truth facts, "
              f"{len(flaws.occurrences)} flaw occurrences)")
        print(f"  MESSINESS.md   (paste into data/README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
