# Using the synthetic-data skill with your coding agent

The documents in `data/raw/` were not written by hand. They were produced by the
`synthetic-project-data` agent skill, which turns a JSON **spec** (the project's
story, money and risks) into status-report PDFs, a financial summary (XLSX + CSV)
and a risk register, then plants 20 documented data-quality flaws in them. This
guide is for anyone who wants their own coding agent to change that corpus or
make a new one.

**You only need the skill to *change* the data.** Regenerating the committed
corpus as it is needs nothing but `make data`: the renderer and spec are
committed, and the same spec and seed reproduce `data/raw/` byte for byte.

## Where things live

| Path | What it is |
|---|---|
| `.claude/skills/synthetic-project-data/SKILL.md` | The skill: the workflow and the advice on writing a believable spec. This is what your agent reads. |
| `.claude/skills/synthetic-project-data/references/` | `spec-schema.md` (every spec field) and `flaw-catalogue.md` (the 20 flaws and how a pipeline should handle each one). |
| `.claude/skills/synthetic-project-data/scripts/` | `example_spec.json` (a worked example), `inspect_dataset.py` (reads the output back the way an ingestion pipeline would) and `render_dataset.py`. |
| `data/scripts/render_dataset.py` | The renderer. The skill's `render_dataset.py` is a symlink to this file, so there is only one copy to keep up to date. |
| `data/scripts/llm_gateway_spec.json` | The spec for the committed corpus (Conduit Unified LLM Gateway, seed `11`). **This is the source. The PDFs and spreadsheets are generated from it.** |
| `data/scripts/generate_synthetic_data.py` | What `make data` runs: it renders the spec into `data/raw/` and moves `manifest.json` and `MESSINESS.md` up to `data/`. |
| `data/README.md` | The file table, ground-truth facts and messiness table. These are deliverables and must match what is in `raw/`. |

## Loading the skill into your agent

### Claude Code

You don't need to install anything. Claude Code discovers project skills in
`.claude/skills/`, so the skill is available as soon as you open the repo. To check:

- Run `/skills` and look for `synthetic-project-data`, or
- ask "what skills do you have for generating test data?"

You don't need to name the skill. Asking for something like "regenerate the
sample data with a construction project" or "give me a clean copy of the
dataset" will load it.

### Other agents (Codex, Cursor, Copilot, Gemini CLI, Aider, …)

`SKILL.md` is plain Markdown with YAML frontmatter (the Agent Skills format).
Several agents can load these files natively. The rest can read them like any
other document:

1. **If your agent supports skills**, copy or symlink
   `.claude/skills/synthetic-project-data/` into the skills directory it scans.
   Check your agent's docs for that path. Keep the `scripts/render_dataset.py`
   symlink intact, or replace it with a copy of `data/scripts/render_dataset.py`.
2. **Otherwise, point your agent at the file at the start of the task**, e.g.:

   > Read `.claude/skills/synthetic-project-data/SKILL.md` and the two files in
   > its `references/` folder, then follow that workflow, together with the
   > repo rules in `docs/synthetic-data-skill.md`, to &lt;your task&gt;.

   To make this automatic, add that sentence to your agent's standing
   instructions file (`AGENTS.md`, `.cursor/rules`, `.github/copilot-instructions.md`, …).

Every agent also needs the **repo rules** below. The skill is written for any
repository. These rules apply only to this one, and they win where the two differ.

### No agent

Everything the skill does is plain commands. See [Doing it by hand](#doing-it-by-hand).

## Repo rules your agent must follow

Paste or reference this section when you brief a non-Claude agent. Claude Code
picks up enough from `CLAUDE.md` and `data/README.md`, but it's worth pointing
it here as well.

1. **Edit the spec, never the documents.** Every change goes into
   `data/scripts/llm_gateway_spec.json` (or a new spec beside it), followed by a
   re-render. A hand-edited PDF or CSV is overwritten by the next `make data`.
2. **Keep `manifest.json` and `MESSINESS.md` out of `data/raw/`.** The manifest
   lists the evaluation answers outright. If it gets indexed, the assistant
   retrieves its own ground truth. `make data` moves both files to `data/`. If
   your agent calls `render_dataset.py --out data/raw` directly, it must move
   them itself.
3. **Merge into `data/README.md`, don't add a second table.** Replace the file
   table, the ground-truth bullets and the messiness rows with the new
   `data/MESSINESS.md` and `data/manifest.json` content. Leave the README's prose
   (the corpus description, "Regenerating", the note on deliberately absent mess)
   alone unless it is now wrong.
4. **Delete stale files.** The renderer overwrites files but never removes any.
   If a spec change renames the project or a report, the old files stay in
   `data/raw/` and get indexed alongside the new ones.
5. **Don't plant contradictions or prompt-injection text.** Both are left out of
   the default flaw set on purpose (see `data/README.md`). A corpus that
   disagrees with itself can't be told apart from a pipeline bug. If you
   genuinely need adversarial documents, put them in a separate, clearly
   labelled corpus and record the reason in `DECISIONS.md`.
6. **Don't restate derived numbers in prose.** Budgets, variances, forecasts,
   totals and risk scores are computed from the cost lines. Summary text is the
   one place the renderer can't check, so any figure written there must match.
7. **Keep the seed** unless you mean to change the documents. Changing it
   reshuffles which flaw lands where, and every messiness row changes with it.
8. **Re-index after regenerating.** `make reindex` (or `make ingest` against a
   fresh index), otherwise the assistant keeps answering from the old chunks.
   The ground-truth facts in `eval/test_queries.yaml` should also be updated
   from the new `manifest.json`.

## Common tasks and prompts to give your agent

These prompts work as written in Claude Code. For other agents, add the "Read
SKILL.md…" preamble above.

**Change the story of the existing corpus**
> In `data/scripts/llm_gateway_spec.json`, make the Q2 availability incident
> worse and add a new vendor-lock-in risk that is raised in Q2 and closed in Q3.
> Validate, run `make data`, inspect the output, update `data/README.md`, and
> tell me which ground-truth facts changed.

**A clean control copy (to decide whether the data or the code is at fault)**
> Render the committed spec with `--flaws none` into a scratch directory and
> diff its extracted text against `data/raw`.

**Isolate one flaw while debugging ingestion**
> Render the committed spec with only F04 (header naming drift) into a scratch
> directory, so I can test the header synonym map on its own.

**A second corpus for a different domain**
> Using the synthetic-project-data skill, write a new spec
> `data/scripts/hospital_wing_spec.json` for a hospital wing construction
> programme, with three quarterly reports and the register as XLSX. Render it into
> a scratch directory, not `data/raw`. I want to check the pipeline copes with a
> second domain before we decide whether to commit it.

**Replace the corpus entirely.** This is a larger change, so ask for a plan first:
> Plan a replacement of the committed corpus with a payments-platform migration.
> List every file you would touch (spec, `generate_synthetic_data.py`'s `SPEC`
> constant, `data/README.md`, eval queries, anything that hard-codes "Conduit" or
> `CND-26`) before making changes.

## Doing it by hand

```bash
SPEC=data/scripts/llm_gateway_spec.json
RENDER="uv run --project backend --group data python data/scripts/render_dataset.py"
INSPECT="uv run .claude/skills/synthetic-project-data/scripts/inspect_dataset.py"

$RENDER --list-flaws                         # the flaw catalogue with IDs
$RENDER $SPEC --validate-only                # errors name the exact JSON path
make data                                    # render into data/raw, move manifest + MESSINESS.md to data/
$INSPECT data/raw                            # extracted PDF text, ragged rows, BOM, merged cells…
$RENDER $SPEC --out /tmp/clean --flaws none  # clean control set
$RENDER $SPEC --out /tmp/f01 --flaws F01,M03 # only these flaws
make reindex                                 # rebuild the index from the new documents
```

`--flaws` also accepts `all`, `format`, `missing` and exclusions like `all,-M08`.
The full field reference for a spec is
`.claude/skills/synthetic-project-data/references/spec-schema.md`. Start any new
spec from `example_spec.json` or the committed spec rather than from an empty file.

## Before you open a PR

- [ ] The spec is committed, and `make data` on a clean checkout reproduces
      `data/raw/` exactly (`git status` shows nothing new after a second run).
- [ ] `inspect_dataset.py data/raw` reports extractable text for every PDF, and
      the flaws you expect show up where you expect them.
- [ ] `data/raw/` contains only documents: no `manifest.json`, no
      `MESSINESS.md`, no files left over from an earlier spec.
- [ ] `data/README.md`'s file table, ground-truth facts and messiness table
      match the new files, filenames included.
- [ ] Eval ground truth is updated from `data/manifest.json`, and `make test`
      still passes.

## Troubleshooting

- **The skill's `render_dataset.py` is a one-line text file** (Windows, or
  `core.symlinks=false`): git checked the symlink out as a plain file. Run
  `data/scripts/render_dataset.py` directly, or replace the file with a copy.
- **`uv: command not found`**: install uv (the whole repo depends on it). If you
  can't, `pip install reportlab openpyxl pypdf` and run the scripts with `python3`.
- **The agent wrote a spec that validates but reads as obviously fake**: point
  it back to the "Writing a spec that reads like real project data" section of
  `SKILL.md`. The corpus needs an arc, specifics, one cost line that overruns,
  risks that carry across reports, and a small cast of people who reappear.
  Retrieval questions only have answers when the documents give them something
  to find.
- **The answers didn't change after regenerating**: the index is stale. Run
  `make reindex`.
