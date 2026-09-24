"""Loading and chunking the real corpus.

Each test names the flaw from `data/MESSINESS.md` that it pins down.
"""

import pytest

from app.ingestion.chunking import chunk_pdf, chunk_table
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.tabular_loader import load_tabular
from tests.conftest import (
    FINANCIALS_CSV,
    FINANCIALS_XLSX,
    RISK_REGISTER_CSV,
    STATUS_MEMO_Q2,
    STATUS_REPORT_Q1,
    STATUS_REPORT_Q3,
)


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def test_repeated_page_furniture_is_stripped():
    """F10: the banner and `Page N of 2` footer repeat on every page."""
    document = load_pdf(STATUS_REPORT_Q1)
    text = " ".join(block.text for block in document.blocks if block.kind == "narrative")

    assert "CONFIDENTIAL" not in text
    assert "Page 1 of 2" not in text
    assert "F10" in document.flaws


def test_pdf_glyph_artifacts_do_not_reach_chunks():
    """The bullet glyph extracts as `(cid:127)` and would be embedded as content."""
    chunks = chunk_pdf(load_pdf(STATUS_REPORT_Q1))

    assert not any("(cid:" in chunk.text for chunk in chunks)


def test_the_sign_off_line_is_not_indexed_as_content():
    """It appears once, so furniture detection cannot see it; it is metadata."""
    document = load_pdf(STATUS_REPORT_Q1)
    narrative = [block for block in document.blocks if block.kind == "narrative"]

    assert not any("Prepared by Arjun Mehta |" in block.text for block in narrative)


def test_tables_are_extracted_structurally_not_as_flattened_text():
    """`extract_text` splits 'Hannah Okonkwo' across lines and interleaves the comment."""
    document = load_pdf(STATUS_REPORT_Q1)
    milestones = _table(document, "3. Milestones")

    metering = next(row for row in milestones.rows if "metering" in row[0].lower())
    assert metering[1] == "Hannah Okonkwo"
    assert metering[2] == "03/20/2026"


def test_a_table_split_by_a_page_break_is_stitched_back_together():
    """The Key Risks table starts on page 1 and continues on page 2."""
    risks = _table(load_pdf(STATUS_REPORT_Q1), "5. Key risks")

    assert risks.page_start == 1
    assert risks.page_end == 2
    assert {row[0] for row in risks.rows} == {"R-001", "R-002", "R-009"}


def test_sections_become_the_chunk_boundary():
    sections = [block.section for block in load_pdf(STATUS_REPORT_Q1).blocks]

    assert "1. Executive summary" in sections
    assert "7. Focus for next period" in sections
    assert None not in sections  # no orphaned preamble


@pytest.mark.parametrize(
    "path,period,status",
    [(STATUS_REPORT_Q1, "2026-Q1", "green"),
     (STATUS_MEMO_Q2, "2026-Q2", "amber"),
     (STATUS_REPORT_Q3, "2026-Q3", "amber")],
)
def test_status_and_period_are_recovered_from_all_three_templates(path, period, status):
    """F03 across three layouts: `Status: G`, `Overall health: A`, and the memo,
    which records it only inside `Re: ... 2026-Q2 status (Yellow)`."""
    meta = load_pdf(path).meta

    assert meta["period"] == period
    assert meta["status_normalised"] == status


def test_the_memo_has_no_financial_section():
    """M08: answering a budget question from this file is a hallucination."""
    sections = {block.section for block in load_pdf(STATUS_MEMO_Q2).blocks}

    assert not any("Financial" in (s or "") for s in sections)
    assert any("Financial" in (s or "") for s in {b.section for b in load_pdf(STATUS_REPORT_Q1).blocks})


def test_citations_name_the_section_and_page():
    chunks = chunk_pdf(load_pdf(STATUS_REPORT_Q1))
    citations = {chunk.citation for chunk in chunks}

    assert "§5. Key risks (pp. 1-2)" in citations
    assert "§1. Executive summary (p. 1)" in citations


def _table(document, section):
    return next(b for b in document.blocks if b.kind == "table" and b.section == section)


# --------------------------------------------------------------------------
# Tabular
# --------------------------------------------------------------------------


def test_risk_register_reads_through_its_bom():
    """F09: written by a tool that emits a UTF-8 BOM."""
    table = load_tabular(RISK_REGISTER_CSV)[0]

    assert table.columns[0] == "risk_id"
    assert len(table) == 12


def test_a_row_broken_by_an_unquoted_comma_is_repaired_not_discarded():
    """R-005 has 14 fields against a 13-column header."""
    table = load_tabular(RISK_REGISTER_CSV)[0]
    row = next(r for r in table.rows if r["risk_id"] == "R-005")

    assert row["owner"] == "Lena Petrova"
    assert row["status"] == "closed"
    assert row["exposure"] is not None
    assert "pending PMO confirmation" in row["description"]


def test_the_workbook_header_is_found_behind_the_title_banner():
    """M04: rows 1-2 are a merged title, so the real header is row 4."""
    summary = next(t for t in load_tabular(FINANCIALS_XLSX) if t.sheet == "Summary")

    assert summary.header_row == 4
    assert "budget" in summary.columns
    assert "M04" in summary.flaws


def test_the_embedded_total_row_is_kept_out_of_the_data_rows():
    """M10: including it double-counts every aggregate."""
    for path in (FINANCIALS_CSV, FINANCIALS_XLSX):
        table = next(t for t in load_tabular(path) if t.name == "financials")

        assert len(table) == 8
        assert table.total_row is not None
        assert sum(row["budget"] for row in table.rows) == 6_315_000


def test_both_financial_exports_agree_after_cleaning():
    """The same eight cost lines, written with different separators and headers."""
    csv_table = load_tabular(FINANCIALS_CSV)[0]
    xlsx_table = next(t for t in load_tabular(FINANCIALS_XLSX) if t.name == "financials")

    assert {r["cost_code"] for r in csv_table.rows} == {r["cost_code"] for r in xlsx_table.rows}
    assert sum(r["actual"] for r in csv_table.rows) == sum(r["actual"] for r in xlsx_table.rows)


def test_quarter_headers_are_normalised():
    """F01 applied to headers: `2025-Q4`, `Q1 2026`, `2Q26`, `Qtr 3 2026`."""
    by_period = next(t for t in load_tabular(FINANCIALS_XLSX) if t.name == "financials_by_period")

    assert by_period.columns[:6] == ["category", "2025-Q4", "2026-Q1", "2026-Q2", "2026-Q3", "total"]


def test_an_empty_field_is_reported_as_missing_not_as_a_value():
    """M03: R-011 has no mitigation, and that is the answer to a test query."""
    table = load_tabular(RISK_REGISTER_CSV)[0]
    missing = [r["risk_id"] for r in table.rows if r["mitigation"] is None]

    assert missing == ["R-011"]


def test_table_chunks_carry_row_level_citations():
    """Citations record the location; `source` is a separate metadata field,
    matching the `Citation` model the agents return."""
    table = load_tabular(RISK_REGISTER_CSV)[0]
    chunks = chunk_table(table)

    assert all(chunk.source == RISK_REGISTER_CSV.name for chunk in chunks)
    assert all(chunk.citation.startswith(("row ", "rows ", "total")) for chunk in chunks)


def test_workbook_citations_name_the_sheet_and_original_row():
    """Row numbers refer to the sheet as a reader sees it, not a post-trim index."""
    summary = next(t for t in load_tabular(FINANCIALS_XLSX) if t.sheet == "Summary")
    chunks = chunk_table(summary)

    assert all(chunk.citation.startswith("Summary!") for chunk in chunks)
    assert summary.row_numbers[0] == 5  # header at row 4, first data row below it


def test_chunk_text_keeps_the_original_notation_alongside_the_parsed_value():
    """BM25 can only match text that is present, and a question may quote the source."""
    table = next(t for t in load_tabular(FINANCIALS_CSV) if t.name == "financials")
    text = "\n".join(chunk.text for chunk in chunk_table(table))

    assert "as written:" in text
