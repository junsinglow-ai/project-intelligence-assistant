"""Value normalisation, checked against the notations the corpus really uses.

Every case here is a value that appears in `data/raw/`, cross-referenced with
the flaw it demonstrates in `data/MESSINESS.md`. These are the contract that
file describes, so a regression shows up as a failing flaw ID rather than as a
wrong number several layers downstream.
"""

from decimal import Decimal

import pytest

from app.ingestion.cleaning import (
    collapse_ws,
    detect_date_style,
    detect_percent_scale,
    is_placeholder,
    normalise_rag,
    normalise_status,
    null_or,
    parse_date,
    parse_money,
    parse_number,
    parse_percent,
)


@pytest.mark.parametrize(
    "raw,expected,currency",
    [
        ("$6,315,000", Decimal("6315000"), "USD"),     # F02 symbol + separators
        ("$6.3M", Decimal("6300000"), "USD"),          # F02 compact form
        ("6,315,000 USD", Decimal("6315000"), "USD"),  # F02 trailing code
        ("120k", Decimal("120000"), None),             # F02 k suffix
        ("$0.3M", Decimal("300000"), "USD"),           # F02 sub-million compact
        ("1.850.000,00", Decimal("1850000"), None),    # F07 European separators
        (" 300,000 ", Decimal("300000"), None),        # M06 number stored as text
        ("(1,234)", Decimal("-1234"), None),           # F02 parenthesised negative
        ("90,000 USD", Decimal("90000"), "USD"),
        ("480000", Decimal("480000"), None),
    ],
)
def test_parse_money_handles_every_notation_in_the_corpus(raw, expected, currency):
    parsed = parse_money(raw)

    assert parsed.value == expected
    assert parsed.note == currency
    assert parsed.raw == raw.strip()


def test_money_keeps_the_raw_string_when_it_cannot_be_read():
    parsed = parse_money("M/M")

    assert parsed.value is None
    assert parsed.raw == "M/M"
    assert parsed.issue == "F02"


@pytest.mark.parametrize(
    "raw,expected",
    [("77%", 0.77), ("82 pct", 0.82), ("42%", 0.42), ("30%", 0.30)],
)
def test_parse_percent_with_an_explicit_marker(raw, expected):
    assert parse_percent(raw).value == pytest.approx(expected)


def test_percent_scale_is_decided_per_column_not_per_value():
    """F06: `1.0931` is a 109% overspend, not 1.09%.

    Read on its own the value is ambiguous, and the "anything above 1 must be a
    percentage" reading silently divides a real overspend by 100. The column it
    sits in settles it: its neighbours are 0.77 and 0.72.
    """
    column = ["0.7708", "0.7244", "1.0931", "0.6667"]
    assert detect_percent_scale(column) == "ratio"
    assert parse_percent("1.0931", "ratio").value == pytest.approx(1.0931)

    assert detect_percent_scale(["77", "72", "109"]) == "percent"
    assert parse_percent("109", "percent").value == pytest.approx(1.09)


def test_date_style_is_inferred_from_the_values_that_cannot_be_ambiguous():
    """F01: a value above 12 in either position settles the file's convention."""
    assert detect_date_style(["22/01/2026", "2026-05-08"]) == "dmy"
    assert detect_date_style(["03/20/2026"]) == "mdy"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-03-31", "2026-03-31"),
        ("18-Sep-26", "2026-09-18"),
        ("3 December 2025", "2025-12-03"),
        ("30/06/2026", "2026-06-30"),
        ("03/20/2026", "2026-03-20"),  # unambiguous mm/dd even in a dd/mm file
    ],
)
def test_parse_date_normalises_every_convention(raw, expected):
    assert parse_date(raw, "dmy").value == expected


def test_ambiguous_dates_record_the_reading_that_was_chosen():
    parsed = parse_date("02/07/2026", "dmy")

    assert parsed.value == "2026-07-02"
    assert parsed.issue == "F01"
    assert "dmy" in (parsed.note or "")


def test_month_only_dates_do_not_invent_a_day():
    parsed = parse_date("June 2026")

    assert parsed.value == "2026-06"
    assert "month precision" in (parsed.note or "")


@pytest.mark.parametrize("raw,expected", [("G", "green"), ("Green", "green"), ("On Track", "green"),
                                          ("A", "amber"), ("Yellow", "amber"), ("Red", "red")])
def test_rag_vocabulary_is_unified_across_the_three_reports(raw, expected):
    """F03: the same status is written 'G', 'Yellow' and 'A'."""
    assert normalise_rag(raw).value == expected


@pytest.mark.parametrize("raw,expected", [("open", "open"), ("OPEN", "open"), ("Open", "open"),
                                          ("mitigated", "mitigated"), ("Closed", "closed"),
                                          ("done", "closed"), ("Completed", "closed"),
                                          ("In-progress", "in_progress")])
def test_status_matching_is_case_insensitive(raw, expected):
    """F03: counting open risks depends on this."""
    assert normalise_status(raw).value == expected


def test_an_unmapped_status_is_surfaced_rather_than_dropped():
    parsed = normalise_rag("chartreuse")

    assert parsed.value is None
    assert parsed.issue == "F03"
    assert parsed.raw == "chartreuse"


@pytest.mark.parametrize("raw", ["tbc", "N/A", "-", "TBD", ""])
def test_null_tokens_become_real_nulls(raw):
    """M02: treating 'tbc' as a value makes counts and aggregates wrong."""
    assert null_or(raw).value is None


def test_real_text_survives_null_normalisation():
    assert null_or("Two squads: routing and billing").value == "Two squads: routing and billing"


@pytest.mark.parametrize("raw", ["see attached", "Vendor commercials - see attached.",
                                 "buffering defaults changed, pending PMO confirmation"])
def test_placeholders_are_recognised(raw):
    """M07: text that fills a field without answering it."""
    assert is_placeholder(raw) is True


def test_real_content_is_not_mistaken_for_a_placeholder():
    assert is_placeholder("Grew after the May rate-limit incident.") is False


def test_pdf_glyph_artifacts_are_stripped():
    """Not in MESSINESS.md: the bullet glyph extracts as a literal `(cid:127)`."""
    assert collapse_ws("(cid:127) Unified  API released.") == "Unified API released."


def test_numbers_are_coerced_per_cell_not_per_column():
    """M06: one budget cell is text while the rest of the column is numeric."""
    assert parse_number(" 300,000 ").value == Decimal("300000")
    assert parse_number("not a number").issue == "M06"
