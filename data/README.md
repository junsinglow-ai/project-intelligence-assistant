# Sample data

Synthetic project documents used for development, demos and evaluation.

They describe one fictional programme, **Conduit Unified LLM Gateway** (`CND-26`):
an internal platform that puts a single API in front of three upstream model
providers, with routing, failover, per-key budgets and usage metering. Three
quarters of reporting take it from green, through a quarter where pass-through
token spend runs ahead of plan and a rate-limit cascade dents availability, to a
partial recovery. The arc is deliberate: it gives the assistant questions that
have findable answers and that need more than one document to answer.

Generated from [`scripts/llm_gateway_spec.json`](scripts/llm_gateway_spec.json)
with seed `11`. The spec is the source; the documents in `raw/` are output.

| File | Type | Purpose |
|---|---|---|
| `Conduit_Unified_LLM_Gateway_Status_Report_2026-Q1.pdf` | PDF | Status report - 2026-Q1 (PSR-2026-03) |
| `conduit_unified_llm_gateway-status-memo-jun2026.pdf` | PDF | Status report - 2026-Q2 (PSR-2026-06) |
| `CND-26_PSR_2026-09-18.pdf` | PDF | Status report - 2026-Q3 (PSR-2026-09) |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | XLSX | Financial summary - budget vs actual by cost line and period |
| `conduit_unified_llm_gateway_financials_export.csv` | CSV | Financial summary - flat export of the cost lines |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | CSV | Risk register |

Two generated companions sit beside this file rather than inside `raw/`:
[`manifest.json`](manifest.json), which records which flaw landed in which file
plus ground-truth facts plus the trap in each, and `MESSINESS.md`, the generated
form of the table below. They stay out of `raw/` deliberately - the manifest
answers the evaluation questions outright, so indexing it would let the
assistant retrieve its own ground truth. Facts worth reusing in
`eval/test_queries.yaml`:

- **What is the approved budget for Conduit Unified LLM Gateway?** 6,315,000 USD
- **How much had been spent by the end of 2026-Q3?** 5,183,000 USD (82% of budget)
- **Is the project over or under budget as at 2026-Q3?** Under budget by 1,132,000 USD
- **Which cost category carries the largest approved budget?** Platform engineering (2,400,000 USD)
- **How many risks are currently open?** 9
- **Which risk has the highest financial exposure, and who owns it?** R-003 - Prompt logging retains request bodies beyond data residency commitments (520,000 USD), owned by Yusuf Karim

## Intentional messiness

Every row below was recorded as the file was written, so the table describes
what the data actually contains rather than what was intended. The handling
column is the pipeline's contract: where a behaviour is not implemented yet, say
so rather than deleting the row.

Two kinds of mess are deliberately absent, because they change what the
documents *mean* rather than how they are written: cross-document
contradictions and prompt-injection text. Every financial figure in a report is
derived from the cost lines, so the corpus is internally consistent by
construction - a disagreement between two files would be a bug, not a fixture.

| File | Issue | How it is handled |
|---|---|---|
| `Conduit_Unified_LLM_Gateway_Status_Report_2026-Q1.pdf` | **F01 Mixed date formats** - dates are written like 2026-03-31; milestone 'Usage metering pipeline' shows its due date as 03/20/2026 rather than the 2026-03-20 form used by the rest of the table | Parse with a multi-format date parser and normalise to ISO-8601 at ingestion; resolve dd/mm vs mm/dd against the file's dominant style and keep the raw string in metadata. |
| `Conduit_Unified_LLM_Gateway_Status_Report_2026-Q1.pdf` | **F02 Mixed currency notation** - amounts are written like $6,315,000 | Strip symbols/separators and expand k/M suffixes to a numeric value plus a currency code; read parenthesised numbers as negative. |
| `Conduit_Unified_LLM_Gateway_Status_Report_2026-Q1.pdf` | **F03 Inconsistent status vocabulary** - overall status written as 'G' | Map to a controlled vocabulary with a case-insensitive synonym table; surface unmapped values rather than silently dropping them. |
| `Conduit_Unified_LLM_Gateway_Status_Report_2026-Q1.pdf` | **F05 Whitespace, casing and filename drift** - project name appears as 'Conduit Unified LLM Gateway'; filename follows a different convention from the other reports | Strip and collapse whitespace on ingestion; casefold before matching entity names; never derive metadata from the filename alone. |
| `Conduit_Unified_LLM_Gateway_Status_Report_2026-Q1.pdf` | **F06 Mixed scale and percentage notation** - percentage spent written as '30%' | Detect ordinal-vs-numeric-vs-percentage per value and convert to one scale; record the interpretation so an answer can cite the raw form. |
| `Conduit_Unified_LLM_Gateway_Status_Report_2026-Q1.pdf` | **F10 Repeated page furniture in PDFs** - confidentiality banner, project code and 'Page x of y' repeat on every page | Strip repeated header/footer lines during PDF extraction (drop lines that recur on most pages) before chunking, so chunks stay clean. |
| `Conduit_Unified_LLM_Gateway_Status_Report_2026-Q1.pdf` | **M01 Blank required fields** - milestone 'SOC 2 Type I readiness review' has no owner | Treat empty as unknown rather than zero; keep the row, flag the gap, and say the field is missing rather than inventing a value. |
| `Conduit_Unified_LLM_Gateway_Status_Report_2026-Q1.pdf` | **M07 Placeholder and truncated text** - issue 1 is a placeholder or stops mid-sentence | Treat placeholder phrases as non-answers; never quote them as the substance of a mitigation or decision. |
| `conduit_unified_llm_gateway-status-memo-jun2026.pdf` | **F01 Mixed date formats** - dates are written like 30/06/2026; milestone 'Automatic provider failover' shows its due date as 2026-05-29 rather than the 29/05/2026 form used by the rest of the table | Parse with a multi-format date parser and normalise to ISO-8601 at ingestion; resolve dd/mm vs mm/dd against the file's dominant style and keep the raw string in metadata. |
| `conduit_unified_llm_gateway-status-memo-jun2026.pdf` | **F02 Mixed currency notation** - amounts are written like $6.3M | Strip symbols/separators and expand k/M suffixes to a numeric value plus a currency code; read parenthesised numbers as negative. |
| `conduit_unified_llm_gateway-status-memo-jun2026.pdf` | **F03 Inconsistent status vocabulary** - overall status written as 'Yellow' | Map to a controlled vocabulary with a case-insensitive synonym table; surface unmapped values rather than silently dropping them. |
| `conduit_unified_llm_gateway-status-memo-jun2026.pdf` | **F05 Whitespace, casing and filename drift** - project name appears as 'CONDUIT UNIFIED LLM GATEWAY'; filename follows a different convention from the other reports | Strip and collapse whitespace on ingestion; casefold before matching entity names; never derive metadata from the filename alone. |
| `conduit_unified_llm_gateway-status-memo-jun2026.pdf` | **F10 Repeated page furniture in PDFs** - confidentiality banner, project code and 'Page x of y' repeat on every page | Strip repeated header/footer lines during PDF extraction (drop lines that recur on most pages) before chunking, so chunks stay clean. |
| `conduit_unified_llm_gateway-status-memo-jun2026.pdf` | **M01 Blank required fields** - milestone 'Per-key budget caps' has no owner | Treat empty as unknown rather than zero; keep the row, flag the gap, and say the field is missing rather than inventing a value. |
| `conduit_unified_llm_gateway-status-memo-jun2026.pdf` | **M07 Placeholder and truncated text** - issue 3 is a placeholder or stops mid-sentence | Treat placeholder phrases as non-answers; never quote them as the substance of a mitigation or decision. |
| `conduit_unified_llm_gateway-status-memo-jun2026.pdf` | **M08 Omitted report section** - the budget / financial position section is absent from this report | Answer from the period that does report it and say the section is absent, rather than assuming the figure is unchanged or zero. |
| `CND-26_PSR_2026-09-18.pdf` | **F01 Mixed date formats** - dates are written like 18-Sep-26; milestone 'SOC 2 Type II window close' shows its due date as 2026-11-28 rather than the 28-Nov-26 form used by the rest of the table | Parse with a multi-format date parser and normalise to ISO-8601 at ingestion; resolve dd/mm vs mm/dd against the file's dominant style and keep the raw string in metadata. |
| `CND-26_PSR_2026-09-18.pdf` | **F02 Mixed currency notation** - amounts are written like 6,315,000 USD | Strip symbols/separators and expand k/M suffixes to a numeric value plus a currency code; read parenthesised numbers as negative. |
| `CND-26_PSR_2026-09-18.pdf` | **F03 Inconsistent status vocabulary** - overall status written as 'A' | Map to a controlled vocabulary with a case-insensitive synonym table; surface unmapped values rather than silently dropping them. |
| `CND-26_PSR_2026-09-18.pdf` | **F05 Whitespace, casing and filename drift** - project name appears as 'conduit unified llm gateway'; filename follows a different convention from the other reports | Strip and collapse whitespace on ingestion; casefold before matching entity names; never derive metadata from the filename alone. |
| `CND-26_PSR_2026-09-18.pdf` | **F06 Mixed scale and percentage notation** - percentage spent written as '82 pct' | Detect ordinal-vs-numeric-vs-percentage per value and convert to one scale; record the interpretation so an answer can cite the raw form. |
| `CND-26_PSR_2026-09-18.pdf` | **F10 Repeated page furniture in PDFs** - confidentiality banner, project code and 'Page x of y' repeat on every page | Strip repeated header/footer lines during PDF extraction (drop lines that recur on most pages) before chunking, so chunks stay clean. |
| `CND-26_PSR_2026-09-18.pdf` | **M01 Blank required fields** - milestone 'Per-key budget caps and hard stops' has no owner | Treat empty as unknown rather than zero; keep the row, flag the gap, and say the field is missing rather than inventing a value. |
| `CND-26_PSR_2026-09-18.pdf` | **M07 Placeholder and truncated text** - issue 2 is a placeholder or stops mid-sentence | Treat placeholder phrases as non-answers; never quote them as the substance of a mitigation or decision. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **F01 Mixed date formats** - the 'By Period' sheet labels the same quarters 2025-Q4, Q1 2026, 2Q26 and so on | Parse with a multi-format date parser and normalise to ISO-8601 at ingestion; resolve dd/mm vs mm/dd against the file's dominant style and keep the raw string in metadata. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **F04 Header naming drift** - header row mixes Title Case, snake_case and UPPER CASE spellings of the same fields | Resolve headers through a synonym map to canonical field names; match case-insensitively after stripping punctuation and spaces. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **F05 Whitespace, casing and filename drift** - vendor names carry stray leading/trailing spaces | Strip and collapse whitespace on ingestion; casefold before matching entity names; never derive metadata from the filename alone. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **F07 Decimal and thousands separator drift** - actuals for 'Platform engineering' use European separators (1.850.000,00) | Sniff the separator per value (last separator wins as decimal) instead of a global locale assumption; quarantine what stays ambiguous. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **M01 Blank required fields** - 'Reliability engineering' has no vendor recorded | Treat empty as unknown rather than zero; keep the row, flag the gap, and say the field is missing rather than inventing a value. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **M02 Inconsistent null tokens** - an empty note is written as 'tbc' | Normalise a null-token list to a real null at ingestion, case-insensitively, so counts and aggregates don't treat 'N/A' as a value. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **M04 Spreadsheet title rows and merged cells** - rows 1-2 are a merged title banner and an export stamp; the real header row is row 4 | Detect the header row by scanning for the first row that looks like field names; unmerge and forward-fill merged cells before parsing. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **M06 Numbers stored as text** - the approved budget for 'Contingency' is the text ' 300,000 ' rather than a number | Coerce per cell rather than trusting the column type; keep the raw string when coercion fails instead of dropping the row. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **M07 Placeholder and truncated text** - the note for 'Cloud and edge infrastructure' is a placeholder ('see attached') | Treat placeholder phrases as non-answers; never quote them as the substance of a mitigation or decision. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **M09 Trailing empty rows and columns** - the used range runs past the data with blank rows and an empty column J | Trim wholly empty rows/columns after reading the used range so row counts and aggregates are right. |
| `Conduit_Unified_LLM_Gateway_Financial_Summary_FY26.xlsx` | **M10 Embedded total row** - a bold TOTAL row sits at row 13 inside the data block, not in a separate summary area | Detect and exclude label rows (TOTAL/Subtotal/Grand Total) before aggregating, or the numbers double-count. |
| `conduit_unified_llm_gateway_financials_export.csv` | **F02 Mixed currency notation** - overspends are written in parentheses rather than with a minus sign; 'Contingency' is reported as $0.3M / 120k while every other row is plain digits | Strip symbols/separators and expand k/M suffixes to a numeric value plus a currency code; read parenthesised numbers as negative. |
| `conduit_unified_llm_gateway_financials_export.csv` | **F04 Header naming drift** - the same fields are named differently here than in the workbook (SUPPLIER vs Vendor, comments vs Notes) | Resolve headers through a synonym map to canonical field names; match case-insensitively after stripping punctuation and spaces. |
| `conduit_unified_llm_gateway_financials_export.csv` | **F06 Mixed scale and percentage notation** - '% spent' alternates between 0.42 and 42% down the same column | Detect ordinal-vs-numeric-vs-percentage per value and convert to one scale; record the interpretation so an answer can cite the raw form. |
| `conduit_unified_llm_gateway_financials_export.csv` | **F09 Mixed line endings** - CRLF line endings and a trailing blank line, as written by a Windows finance tool | Read with universal newlines and drop empty trailing records. |
| `conduit_unified_llm_gateway_financials_export.csv` | **M10 Embedded total row** - the last data row is a TOTAL row with the same shape as a cost line | Detect and exclude label rows (TOTAL/Subtotal/Grand Total) before aggregating, or the numbers double-count. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **F01 Mixed date formats** - every row carries its own date convention, because rows were pasted in from different trackers | Parse with a multi-format date parser and normalise to ISO-8601 at ingestion; resolve dd/mm vs mm/dd against the file's dominant style and keep the raw string in metadata. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **F02 Mixed currency notation** - exposure is written as plain digits, S$1.2M, 1200k and '1,200,000 SGD' in different rows | Strip symbols/separators and expand k/M suffixes to a numeric value plus a currency code; read parenthesised numbers as negative. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **F03 Inconsistent status vocabulary** - status values appear as Open / OPEN / open / Closed / CLOSED | Map to a controlled vocabulary with a case-insensitive synonym table; surface unmapped values rather than silently dropping them. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **F04 Header naming drift** - columns are named differently from the status reports (Risk Owner vs Owner, raised vs Date Raised) and mix snake_case with Title Case | Resolve headers through a synonym map to canonical field names; match case-insensitively after stripping punctuation and spaces. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **F05 Whitespace, casing and filename drift** - the description for R-011 has stray whitespace | Strip and collapse whitespace on ingestion; casefold before matching entity names; never derive metadata from the filename alone. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **F06 Mixed scale and percentage notation** - likelihood is recorded as 'High', '4' and '80%' for the same level of risk | Detect ordinal-vs-numeric-vs-percentage per value and convert to one scale; record the interpretation so an answer can cite the raw form. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **F08 UTF-8 BOM and non-ASCII text** - the description for R-006 contains an en-dash and a curly apostrophe; the file starts with a UTF-8 BOM, so a naive reader sees '\ufeffRisk ID' as the first column name | Open CSVs as utf-8-sig and keep text as Unicode; normalise dashes and quotes only for matching, not for display. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **M01 Blank required fields** - R-002 has no risk score; R-012 has no owner recorded | Treat empty as unknown rather than zero; keep the row, flag the gap, and say the field is missing rather than inventing a value. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **M02 Inconsistent null tokens** - the next review date for R-011 is 'TBD' | Normalise a null-token list to a real null at ingestion, case-insensitively, so counts and aggregates don't treat 'N/A' as a value. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **M03 Risk recorded without mitigation** - R-011 ('Latency-aware routing degrades quality on long-context requests') is open with no mitigation recorded | Surface it as an explicit gap ('no mitigation recorded') - a question about mitigations must not silently skip the row. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **M05 Ragged CSV row** - the description for R-005 contains an unquoted comma, so that row parses with one extra column | Parse leniently and repair rows whose field count is off by one by re-joining the overflow into the free-text column; log the repair. |
| `conduit_unified_llm_gateway_risk_register_v4.csv` | **M07 Placeholder and truncated text** - the mitigation for R-012 is a placeholder ('see attached') | Treat placeholder phrases as non-answers; never quote them as the substance of a mitigation or decision. |

## Regenerating

```bash
make data
```

The renderer (`scripts/render_dataset.py`, vendored from the
`synthetic-project-data` skill) and its spec are both committed, so regeneration
needs nothing outside this repository. Rendering libraries live in the backend's
optional `data` dependency group and are not installed into the runtime image.

For a clean control copy of the same dataset - identical content, none of the
messiness - which is the quickest way to tell whether a pipeline problem comes
from the data or from the code:

```bash
uv run --project backend --group data python data/scripts/render_dataset.py \
    data/scripts/llm_gateway_spec.json --out /tmp/clean --flaws none
```

`--list-flaws` on the same script prints the catalogue of the 20 flaws with the
identifiers used in the table above.
