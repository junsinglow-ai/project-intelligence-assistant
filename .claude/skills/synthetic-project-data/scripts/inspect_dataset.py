#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pypdf>=4.0", "openpyxl>=3.1"]
# ///
"""Inspect a generated dataset the way an ingestion pipeline would see it.

Reads every PDF, CSV and XLSX in a directory and reports what a loader hits:
extracted text, header rows, ragged rows, null tokens, merged cells, blank
fields. Run it after rendering - it is the fastest way to confirm the documents
are genuinely readable and that the flaws you documented are really there.

    uv run inspect_dataset.py <dir> [--full] [--file NAME]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from openpyxl import load_workbook
from pypdf import PdfReader

NULL_TOKENS = {"tbd", "n/a", "na", "-", "--", "tbc", "unknown", "???", "none"}


def inspect_pdf(path: Path, full: bool) -> None:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated = {line for line in lines if lines.count(line) >= max(2, len(reader.pages))}
    print(f"  pages: {len(reader.pages)}   extracted characters: {len(text)}")
    if not text.strip():
        print("  !! no extractable text - the PDF is image-only and will break ingestion")
    headings = [line for line in lines if line[:2].rstrip(".").isdigit() and len(line) < 60]
    if headings:
        print(f"  sections: {', '.join(headings[:9])}")
    if repeated:
        print(f"  repeated on every page (page furniture): "
              f"{'; '.join(sorted(repeated)[:3])}")
    print("  --- extracted text " + "-" * 40)
    body = lines if full else lines[:28]
    for line in body:
        print(f"    {line[:110]}")
    if not full and len(lines) > 28:
        print(f"    ... {len(lines) - 28} more lines (--full to see them)")


def inspect_csv(path: Path, full: bool) -> None:
    raw = path.read_bytes()
    bom = raw[:3] == b"\xef\xbb\xbf"
    crlf = b"\r\n" in raw
    print(f"  BOM: {'yes (read with utf-8-sig)' if bom else 'no'}   "
          f"line endings: {'CRLF' if crlf else 'LF'}   bytes: {len(raw)}")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        print("  !! empty file")
        return
    header, data = rows[0], rows[1:]
    print(f"  columns: {len(header)}   data rows: {len(data)}")
    print(f"  header: {header}")
    odd = [(i + 2, len(row)) for i, row in enumerate(data) if len(row) != len(header)]
    for line_no, width in odd:
        print(f"  !! row {line_no} has {width} fields, not {len(header)} "
              f"(unquoted delimiter in a free-text field)")
    blanks: dict[str, int] = {}
    nulls: dict[str, int] = {}
    for row in data:
        for name, value in zip(header, row):
            key = name.strip() or "(unnamed)"
            if not value.strip():
                blanks[key] = blanks.get(key, 0) + 1
            elif value.strip().lower() in NULL_TOKENS:
                nulls[key] = nulls.get(key, 0) + 1
    if blanks:
        print(f"  blank cells: {blanks}")
    if nulls:
        print(f"  null tokens: {nulls}")
    label_rows = [row[:2] for row in data
                  if any(str(cell).strip().lower() in {"total", "subtotal", "grand total"}
                         for cell in row)]
    if label_rows:
        print(f"  !! label rows inside the data: {label_rows}")
    if any(ord(ch) > 127 for ch in raw.decode("utf-8-sig", "replace")):
        print("  contains non-ASCII characters")
    shown = data if full else data[:5]
    print("  --- rows " + "-" * 50)
    for row in shown:
        print(f"    {row}")
    if not full and len(data) > 5:
        print(f"    ... {len(data) - 5} more rows (--full to see them)")


def _looks_like_a_number(value) -> bool:
    """True for a string that is really an amount - not a date or a period label.

    Dates ('2025-11-04'), quarter labels ('Q4 2025') and codes all contain
    letters or separators that an amount never does, so they are left alone.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or any(ch.isalpha() for ch in text):
        return False
    core = text.strip("()$\u20ac\u00a3\u00a5 ").replace(",", "").replace(".", "").replace(" ", "")
    return core.isdigit() and len(core) >= 3


def inspect_xlsx(path: Path, full: bool) -> None:
    wb = load_workbook(path)
    print(f"  sheets: {wb.sheetnames}")
    for name in wb.sheetnames:
        ws = wb[name]
        print(f"  [{name}] used range {ws.dimensions} "
              f"({ws.max_row} rows x {ws.max_column} cols)")
        if ws.merged_cells.ranges:
            print(f"    merged: {[str(r) for r in ws.merged_cells.ranges]}")
        rows = list(ws.iter_rows(values_only=True))
        header_row = next(
            (i for i, row in enumerate(rows)
             if sum(1 for c in row if isinstance(c, str) and c.strip()) >= max(3, len(row) // 2)),
            0)
        if header_row:
            print(f"    !! header is row {header_row + 1}, not row 1 "
                  f"(title/banner rows above it)")
        text_numbers = [(cell.coordinate, cell.value)
                        for row in ws.iter_rows() for cell in row
                        if _looks_like_a_number(cell.value)]
        if text_numbers:
            print(f"    !! numbers stored as text: {text_numbers[:4]}")
        trailing = sum(1 for row in reversed(rows) if all(c is None or c == "" for c in row))
        if trailing:
            print(f"    !! {trailing} trailing empty row(s) in the used range")
        shown = rows if full else rows[:8]
        for row in shown:
            print(f"      {['' if v is None else v for v in row]}")
        if not full and len(rows) > 8:
            print(f"      ... {len(rows) - 8} more rows (--full to see them)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a generated dataset directory.")
    parser.add_argument("directory")
    parser.add_argument("--full", action="store_true", help="print every line and row")
    parser.add_argument("--file", help="inspect only files whose name contains this")
    args = parser.parse_args(argv)

    root = Path(args.directory).expanduser()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"manifest: {manifest['project']['name']} (seed {manifest['seed']}), "
              f"{len(manifest['files'])} documents, "
              f"{len(manifest['flaws_enabled'])} flaw types enabled")
        missing = [f["file"] for f in manifest["files"] if not (root / f["file"]).exists()]
        if missing:
            print(f"!! manifest lists files that are not on disk: {missing}")
        print()

    handlers = {".pdf": inspect_pdf, ".csv": inspect_csv, ".xlsx": inspect_xlsx}
    seen = 0
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in handlers:
            continue
        if args.file and args.file.lower() not in path.name.lower():
            continue
        seen += 1
        print("=" * 78)
        print(f"{path.name}  ({path.stat().st_size:,} bytes)")
        print("=" * 78)
        handlers[path.suffix.lower()](path, args.full)
        print()
    if not seen:
        print("no PDF, CSV or XLSX files found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
