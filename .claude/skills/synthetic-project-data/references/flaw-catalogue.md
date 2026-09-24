# Flaw catalogue

Twenty deliberate data flaws in two families, every one of them documented and
reproducible. The generator records each flaw as it writes it, so `MESSINESS.md`
and `manifest.json` always describe the files that actually exist.

The point of the catalogue is not to corrupt the data - it is to reproduce the
specific ways real project documents go wrong, so a pipeline can be exercised
against them and its handling written down. Every flaw therefore carries a
**handling** note: the behaviour a pipeline needs in order to cope. That note is
what lands in the "How it is handled" column of a data README.

Two things this catalogue deliberately leaves out, because they change what the
documents *mean* rather than how they are written: cross-document contradictions
(the same figure disagreeing between a report and the finance sheet) and
adversarial text (prompt-injection strings planted in a document). The generator
derives every figure in the reports from the financial lines precisely so the
corpus stays internally consistent. If you ever need those, they belong behind an
explicit opt-in flag, never in the default set.

Selecting flaws: `--flaws all` (default), `none`, `format`, `missing`,
`F01,M03`, or `all,-M08` to subtract. The authoritative list is
`uv run render_dataset.py --list-flaws`.

| Family | Meaning |
|---|---|
| Format (F01-F10) | The value is present and correct, but written a different way each time. Costs you joins, aggregates and matching. |
| Missing (M01-M10) | The value is absent, structurally wrong, or a placeholder. Costs you counts, totals and truthful answers. |


## Format inconsistency

### F01 - Mixed date formats

**In the data:** Each document uses its own date convention (2026-03-31, 31/03/2026, 31-Mar-26, March 2026, Q1 FY26); hand-edited rows drift from their file's house style.

**Handling:** Parse with a multi-format date parser and normalise to ISO-8601 at ingestion; resolve dd/mm vs mm/dd against the file's dominant style and keep the raw string in metadata.

### F02 - Mixed currency notation

**In the data:** The same amount appears as 1200000, "1,200,000", "S$1.2M", "1200k" and "1,200,000 SGD"; negatives as -45000 and (45,000).

**Handling:** Strip symbols/separators and expand k/M suffixes to a numeric value plus a currency code; read parenthesised numbers as negative.

### F03 - Inconsistent status vocabulary

**In the data:** RAG and progress states are spelled every way a human would type them: Green / GREEN / On Track / G, Amber / At Risk / Yellow, Complete / Completed / DONE / done.

**Handling:** Map to a controlled vocabulary with a case-insensitive synonym table; surface unmapped values rather than silently dropping them.

### F04 - Header naming drift

**In the data:** The same concept is headed differently across files and within one header row (Owner / Risk Owner / Accountable, Date Raised / raised_date).

**Handling:** Resolve headers through a synonym map to canonical field names; match case-insensitively after stripping punctuation and spaces.

### F05 - Whitespace, casing and filename drift

**In the data:** Leading/trailing and doubled spaces in cells, the project name cased three ways, and filenames following three different conventions.

**Handling:** Strip and collapse whitespace on ingestion; casefold before matching entity names; never derive metadata from the filename alone.

### F06 - Mixed scale and percentage notation

**In the data:** Likelihood is 'High' in one row, '4' in another and '80%' in a third; completion shows as 0.35, 35% and '35 pct'.

**Handling:** Detect ordinal-vs-numeric-vs-percentage per value and convert to one scale; record the interpretation so an answer can cite the raw form.

### F07 - Decimal and thousands separator drift

**In the data:** One locale-confused export row writes 1.250,00 and another 1 250, alongside the file's normal 1,250.00.

**Handling:** Sniff the separator per value (last separator wins as decimal) instead of a global locale assumption; quarantine what stays ambiguous.

### F08 - UTF-8 BOM and non-ASCII text

**In the data:** A CSV starts with a BOM and carries en-dashes, curly quotes and accented vendor names.

**Handling:** Open CSVs as utf-8-sig and keep text as Unicode; normalise dashes and quotes only for matching, not for display.

### F09 - Mixed line endings

**In the data:** One CSV is written with CRLF line endings and a trailing blank line, as exported from a Windows finance tool.

**Handling:** Read with universal newlines and drop empty trailing records.

### F10 - Repeated page furniture in PDFs

**In the data:** Every PDF page repeats a confidentiality banner, project code and 'Page x of y', which lands in the middle of extracted text.

**Handling:** Strip repeated header/footer lines during PDF extraction (drop lines that recur on most pages) before chunking, so chunks stay clean.


## Missing and malformed

### M01 - Blank required fields

**In the data:** A milestone with no owner, a risk with no review date, a cost line with no vendor - cells simply left empty.

**Handling:** Treat empty as unknown rather than zero; keep the row, flag the gap, and say the field is missing rather than inventing a value.

### M02 - Inconsistent null tokens

**In the data:** Absent values are written as TBD, N/A, n/a, -, tbc, unknown and ??? in the same column.

**Handling:** Normalise a null-token list to a real null at ingestion, case-insensitively, so counts and aggregates don't treat 'N/A' as a value.

### M03 - Risk recorded without mitigation

**In the data:** At least one high-scoring risk has an empty mitigation/action cell.

**Handling:** Surface it as an explicit gap ('no mitigation recorded') - a question about mitigations must not silently skip the row.

### M04 - Spreadsheet title rows and merged cells

**In the data:** The workbook opens with a merged title banner and an export stamp above the real header row, so row 1 is not the header.

**Handling:** Detect the header row by scanning for the first row that looks like field names; unmerge and forward-fill merged cells before parsing.

### M05 - Ragged CSV row

**In the data:** One free-text field contains an unquoted comma, so that row parses with one column too many.

**Handling:** Parse leniently and repair rows whose field count is off by one by re-joining the overflow into the free-text column; log the repair.

### M06 - Numbers stored as text

**In the data:** Some spreadsheet amounts are strings (" 1,200,000 ", "1.2m") rather than numeric cells.

**Handling:** Coerce per cell rather than trusting the column type; keep the raw string when coercion fails instead of dropping the row.

### M07 - Placeholder and truncated text

**In the data:** Notes read 'see attached', 'per last month', '???' or stop mid-sentence.

**Handling:** Treat placeholder phrases as non-answers; never quote them as the substance of a mitigation or decision.

### M08 - Omitted report section

**In the data:** One status report has no budget section at all - the author ran out of time that month.

**Handling:** Answer from the period that does report it and say the section is absent, rather than assuming the figure is unchanged or zero.

### M09 - Trailing empty rows and columns

**In the data:** The sheet's used range extends past the data with blank rows and a stray empty column.

**Handling:** Trim wholly empty rows/columns after reading the used range so row counts and aggregates are right.

### M10 - Embedded total row

**In the data:** A 'TOTAL' row sits inside the data rows of the export, not in a separate summary block.

**Handling:** Detect and exclude label rows (TOTAL/Subtotal/Grand Total) before aggregating, or the numbers double-count.


## Where each flaw can appear

| Flaw | Status report PDFs | Financial summary | Risk register |
|---|---|---|---|
| F01 mixed dates | yes | yes | yes |
| F02 currency notation | yes | CSV only | yes |
| F03 status vocabulary | yes | - | yes |
| F04 header drift | - | yes | yes |
| F05 whitespace/casing/filenames | yes | yes | yes |
| F06 scale and percentages | yes | yes | yes |
| F07 separator drift | - | XLSX only | - |
| F08 BOM and non-ASCII | - | - | CSV only |
| F09 line endings | - | CSV only | - |
| F10 page furniture | yes | - | - |
| M01 blank fields | yes | yes | yes |
| M02 null tokens | yes | yes | yes |
| M03 risk without mitigation | yes | - | yes |
| M04 title rows / merged cells | - | XLSX only | XLSX only |
| M05 ragged row | - | - | CSV only |
| M06 numbers as text | - | XLSX only | - |
| M07 placeholder text | yes | yes | yes |
| M08 omitted section | yes | - | - |
| M09 trailing empty rows | - | XLSX only | XLSX only |
| M10 embedded total row | - | yes | - |

A dash means the flaw has nowhere sensible to live in that file type, so asking
for it there is a no-op rather than an error. The per-file breakdown of what
actually landed is in `manifest.json` under `flaws_by_file`.
