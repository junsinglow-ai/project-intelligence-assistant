# Dataset spec reference

One JSON file describes the project: who is doing what, how the money is
spread, what could go wrong, and what each report says. The renderer turns that
into documents and injects the flaws. You own the *content*; the script owns the
*mechanics*.

Validate before rendering - the errors name the exact path that is wrong:

```bash
uv run render_dataset.py my_spec.json --validate-only
```

## Top level

| Key | Required | Notes |
|---|---|---|
| `project` | yes | Identity of the project (below). |
| `periods` | yes | Ordered reporting periods, e.g. `["2025-Q3","2025-Q4","2026-Q1","2026-Q2"]`. Every actual and every report period must use one of these labels. |
| `financials` | yes | Cost lines and their actuals. The single source of truth for money. |
| `risk_register` | yes | Exactly one register. |
| `reports` | yes | One entry per status report; three is the useful default. |
| `seed` | no | Integer, default `7`. `--seed` overrides it. |
| `flaws` | no | `"all"` (default), `"none"`, `"format"`, `"missing"`, or a list like `["F01","M03"]`. `--flaws` overrides it. |
| `export_date` | no | The date the spreadsheets claim to have been exported. Defaults to the latest report's `as_of`, which keeps regeneration reproducible. |

## `project`

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Full project name. Appears in documents, filenames and the messiness table. |
| `code` | yes | Short code, e.g. `APL-2026`. Used in page furniture and one filename convention. |
| `currency` | yes | ISO code: `SGD`, `USD`, `EUR`, `GBP`, `AUD`, `MYR`, `INR`, `JPY`. Drives the symbol used in mixed-notation amounts. |
| `client`, `manager`, `sponsor`, `finance_contact` | no | Named people and organisations shown in report headers and the export stamp. |
| `start_date`, `end_date` | no | ISO dates, for context in the manifest. |

## `financials`

| Field | Required | Notes |
|---|---|---|
| `formats` | no | Any of `["xlsx","csv"]`, default both. Generating both is worth it: the workbook carries the structural flaws, the CSV the encoding and notation flaws. |
| `fiscal_label` | no | Appears in the workbook filename and title, e.g. `FY26`. |
| `lines[]` | yes | One row per cost line. |
| `notes[]` | no | `{date, author, text}` objects; they become a Notes sheet with dates in three conventions. |

Each entry in `lines[]`:

| Field | Required | Notes |
|---|---|---|
| `category` | yes | Cost line name, e.g. `Systems integrator`. |
| `budget` | yes | Approved budget, a plain number - never a formatted string. |
| `actuals` | no | `{period: amount}` using the labels from `periods`. Missing periods count as zero. |
| `cost_code`, `vendor`, `note` | no | Shown in the summary; some are blanked or replaced with placeholders by the flaw pass. |

Let one line exceed its budget across the reporting periods. That single
overrun is what makes negative variances, parenthesised numbers and a forecast
above budget appear anywhere in the dataset - without it the whole corpus reads
suspiciously tidy.

## `risk_register`

| Field | Required | Notes |
|---|---|---|
| `format` | no | `csv` (default) or `xlsx`. CSV carries the ragged-row and BOM flaws; XLSX carries title rows and merged cells. |
| `version` | no | Appears in the filename, e.g. `v3`. |
| `risks[]` | yes | Ten to fourteen risks reads like a real register. |

Each entry in `risks[]`:

| Field | Required | Notes |
|---|---|---|
| `id` | yes | e.g. `R-001`. Reports reference these. |
| `title` | yes | One line, specific to the project. |
| `owner` | yes | A person's name; reuse the same handful across the register. |
| `status` | yes | `Open`, `Closed`, `Mitigated`, `In Progress`. Casing is varied for you. |
| `description` | no | A sentence of detail; this is where the ragged-row and non-ASCII flaws land. |
| `category` | no | `Technical`, `Vendor`, `Regulatory`, `Resource`, `Data`, `Security`, `Schedule`, `Financial`, `Change`. |
| `likelihood`, `impact` | no | `Low` / `Medium` / `High` / `Very High`. The score is computed as likelihood x impact on a 1-5 scale, then some rows are re-rendered as `4`, `80%` or `H/M`. |
| `mitigation` | no | Leave one high-exposure risk's mitigation empty on purpose - that is flaw M03. If you fill them all, the generator empties the highest-exposure one for you. |
| `raised_date`, `review_date` | no | ISO dates. Missing review dates become `TBD`/`tbc`/`-`. |
| `exposure` | no | Plain number in the project currency; drives the "highest exposure" ground-truth fact. |

## `reports[]`

| Field | Required | Notes |
|---|---|---|
| `id` | yes | e.g. `PSR-2026-03`. |
| `period` | yes | Must be one of `periods`. |
| `as_of` | yes | ISO date the report was issued. |
| `rag` | yes | `green`, `amber` or `red`. Rendered as `On Track`, `AMBER`, `G` and so on. |
| `executive_summary` | yes | Three to five sentences. This is the paragraph most questions get answered from - make it carry real information. |
| `template` | no | `classic`, `memo` or `table`. Defaults to rotating through all three, which is what makes the corpus look like it came from different authors. |
| `author`, `distribution` | no | Header/addressee detail. |
| `progress[]`, `issues[]`, `next_period[]` | no | Bullet lists. |
| `milestones[]` | no | `{name, owner, due, status, comment}` with status one of `complete`, `in_progress`, `not_started`, `slipped`. |
| `highlight_risks[]` | no | Risk ids to pull into the report's risk table. They must exist in the register - validation enforces it. |
| `budget_note`, `forecast_note` | no | Short notes shown beside the figures. |
| `omit_sections` | no | `["budget"]` forces this report to be the one missing its financial section (M08). If no report declares it, the middle report is chosen. |

## What the renderer derives - do not restate these

Writing these by hand is how a corpus ends up contradicting itself:

- **Every figure in a report's financial section**: approved budget, spend to
  date, spend in period, variance, percentage spent. All computed from
  `financials.lines` as at that report's period.
- **Forecast at completion**: approved budget plus overruns already visible on
  individual cost lines at that point in time.
- **Risk scores** shown in reports and the register.
- **Totals**, the `TOTAL` row and the by-period matrix.
- **Filenames**, which follow three different conventions on purpose.

Prose is the exception the generator cannot check: if an executive summary says
"contingency of 85k was drawn down", make sure a contingency line really shows
85,000 in that period. Keep numbers in prose sparse and make the ones you write
agree with the cost lines.

## Validation rules

- Every `actuals` key and every `reports[].period` must appear in `periods`.
- `highlight_risks` must reference risks that exist.
- `rag` must be green, amber or red; `as_of` must parse as a date.
- Each cost line needs a `category` and a numeric `budget`; each risk needs
  `id`, `title`, `owner` and `status`.

A complete, realistic example lives in `scripts/example_spec.json` - a four-quarter
core banking migration with seven cost lines, twelve risks and three reports.
Copy it and rewrite the content rather than starting from an empty file.
