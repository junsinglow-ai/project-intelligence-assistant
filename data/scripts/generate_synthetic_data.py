#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["reportlab>=4.0", "openpyxl>=3.1"]
# ///
"""Generate synthetic project documents into data/raw/.

Produces 3 status report PDFs, a financial summary (XLSX + CSV) and a risk
register from the committed spec, with deliberate inconsistencies (mixed
formats, missing fields, placeholder values) documented in data/README.md.

The documents are output, not source: `llm_gateway_spec.json` is the source, and
the same spec and seed regenerate the same files, so the messiness table in
data/README.md stays true. Run it with `make data`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_dataset import main as render  # noqa: E402

SPEC = Path(__file__).resolve().with_name("llm_gateway_spec.json")
OUT = Path(__file__).resolve().parents[1] / "raw"


def main() -> None:
    code = render([str(SPEC), "--out", str(OUT)])
    if code:
        raise SystemExit(code)

    # Keep data/raw a pure document corpus. The manifest answers the evaluation
    # questions outright and the messiness table describes every planted flaw,
    # so indexing either of them would let the assistant retrieve its own ground
    # truth instead of reading the documents.
    for name in ("manifest.json", "MESSINESS.md"):
        generated = OUT / name
        if generated.exists():
            generated.replace(OUT.parent / name)


if __name__ == "__main__":
    main()
