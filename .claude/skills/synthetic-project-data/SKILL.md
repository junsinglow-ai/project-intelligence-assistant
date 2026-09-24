---
name: synthetic-project-data
description: >-
  Generates a realistic synthetic project document set - project status reports
  as PDFs, a financial summary as Excel and CSV, and one risk register - carrying
  intentional, documented data messiness (mixed date and currency formats,
  inconsistent status vocabulary, blank and placeholder fields, spreadsheet title
  rows and merged cells, ragged CSV rows, BOM encoding) plus a manifest of
  ground-truth answers. Use this whenever someone needs sample, demo, dummy or
  test project data, fixtures for a document ingestion or RAG pipeline, or
  documents that exercise real-world data quality problems. Trigger it on
  phrasings like "generate some fake project reports", "I need test PDFs and a
  risk register", "make a demo dataset for the ingestion pipeline", "synthetic
  financials with missing fields", "sample data with realistic messiness", or
  "populate data/raw" - and also when a repo has an unimplemented synthetic-data
  script, an empty data directory, or a `make data` target waiting to be filled.
---

# Synthetic project data

## What this produces

A small corpus that looks like it came off a real project's shared drive:

| Output | Detail |
|---|---|
| Status report PDFs | One per reporting period, rotating through three house styles (formal report, memo, dense one-pager) so the set looks authored by different people. |
| Financial summary | An XLSX workbook (summary, by-period matrix, notes) and a flat CSV export. |
| Risk register | Exactly one, CSV by default or XLSX. |
| `manifest.json` | Files written, which flaws landed in which file, and ground-truth facts with the answer and the trap in each. |
| `MESSINESS.md` | A file table and a flaw table (file / issue / how it is handled), shaped to drop into a data README. |

Everything is seeded: the same spec and seed regenerate the documents byte for
byte, which is what lets the documentation stay true over time.

## How the work splits

You write a **spec** - the project's story, its money, its risks. A bundled
script renders it and injects the flaws. That split matters: prose is what makes
synthetic data convincing and is worth your attention, while PDF layout and
seeded flaw injection are mechanical and shouldn't be re-derived on each run.
The script also derives every financial figure in the reports from the cost
lines, so the corpus cannot contradict itself by accident.

## Workflow

```bash
SKILL=~/.claude/skills/synthetic-project-data          # adjust to where this skill lives
cp $SKILL/scripts/example_spec.json ./spec.json        # 1. start from the worked example
# 2. rewrite spec.json for the project you were asked for
uv run $SKILL/scripts/render_dataset.py spec.json --validate-only   # 3. check it
uv run $SKILL/scripts/render_dataset.py spec.json --out data/raw    # 4. render
uv run $SKILL/scripts/inspect_dataset.py data/raw                   # 5. verify
```

**1. Settle the brief.** What kind of project, how many reports, where the files
go, which formats. Infer what you reasonably can and ask only where the answer
changes the data - the domain, if there is no hint of one, is usually the only
real question. Sensible defaults: three reports over three consecutive quarters,
six to eight cost lines, ten to fourteen risks, financials as both XLSX and CSV,
register as CSV, all flaws on.

**2. Write the spec.** Copy `scripts/example_spec.json` and rewrite its content -
starting from an empty file wastes effort on structure you can get for free.
Field-by-field reference: `references/spec-schema.md`.

**3. Validate, render, inspect.** `--validate-only` names the exact path of
anything wrong. `inspect_dataset.py` reads the output back the way an ingestion
pipeline would and reports extracted PDF text, header rows, ragged rows, null
tokens, merged cells and numbers stored as text. Run it - it is how you find out
that a PDF has no extractable text or that a flaw silently did not land.

## Writing a spec that reads like real project data

The generator can make data messy; only you can make it *plausible*. What
separates a convincing corpus from an obviously synthetic one:

- **Give it an arc.** Something degrades, gets escalated, and partially
  recovers. A set of three green reports answers no interesting question, and
  the whole point of a demo corpus is that questions have findable answers.
- **Be specific.** "Rehearsal 1 took 19 hours against a 14-hour rollback window"
  is worth ten sentences of "cutover preparation is progressing well". Specifics
  are what retrieval has to grab onto.
- **Let one cost line overrun its budget.** That single overrun is what makes
  negative variances, parenthesised amounts and a forecast above budget appear
  anywhere in the set. Without it everything reads suspiciously tidy.
- **Carry risks across reports.** A risk raised in one period, highlighted in
  the next and closed in the third is what makes multi-document questions work.
- **Reuse a small cast.** Four to six people, each owning a consistent area.
  Cross-document questions need entities that actually repeat.
- **Keep names fictional but plausible.** Invented vendors and clients, never
  real companies presented as doing real work.
- **Don't restate derived numbers.** Budget figures, variances, forecasts,
  totals and risk scores are computed. Prose is the one place the generator
  can't check you: if a summary mentions "85k of contingency", make sure the
  contingency line really shows 85,000 that period.

## Choosing the messiness

All twenty flaws are on by default, in two families - **format** (the value is
right but written differently every time) and **missing** (the value is absent,
placeheld or structurally wrong). Select with `--flaws`:

```bash
--flaws none            # clean control set - useful as an A/B against the messy one
--flaws format          # or: missing
--flaws all,-M08        # everything except the omitted report section
--flaws F01,F02,M03     # just these
```

`uv run scripts/render_dataset.py --list-flaws` prints the catalogue; the full
descriptions and the handling note for each are in `references/flaw-catalogue.md`.

Two kinds of mess are deliberately **not** in the default set, because they
change what the documents mean rather than how they are written: cross-document
contradictions and adversarial/prompt-injection text. Don't hand-add them to a
spec without saying so explicitly - a corpus that quietly disagrees with itself
is indistinguishable from a bug in whatever pipeline consumes it.

## Fitting it into a repository

When the request comes from inside a project, place the output where that
project already expects data (`data/raw/` or similar) and then close the loop:

- If a data README has a messiness or file table, **merge** the rows from
  `MESSINESS.md` into it rather than leaving a second, competing document.
- If the repo has a generator stub or a `make data` target, wire it up. The
  renderer lives with this skill, outside the repo, so a repo that must
  regenerate on its own needs the script and the spec committed to it - copy
  them in, or say plainly that the target still depends on the skill. Leaving a
  make target that only works on your machine is worse than leaving it stubbed.
- Commit the spec. It is the reproducible source; the documents are output.

## Before you hand it over

- `inspect_dataset.py` runs clean: every PDF has extractable text, and the
  ragged row, BOM, merged cells and title rows are reported where you expect.
- The ground-truth facts in `manifest.json` are genuinely answerable from the
  documents - they are what someone will use to test a pipeline, so skim them.
- The flaw table describes files that exist, with the filenames as written.
- If you claimed reproducibility, re-render to a temp directory and diff.

Then report back with where the files went, what the corpus is *about* in a
sentence, a couple of the ground-truth questions it can answer, and where the
flaw table is. Don't paste the whole messiness table into chat - it is long by
design and it already lives in a file.

## Dependencies

`uv run` installs what the scripts need from their inline metadata (reportlab
and openpyxl to render; pypdf to inspect) without touching the project's
environment. Without uv: `pip install reportlab openpyxl pypdf` and run the
scripts with `python3`.
