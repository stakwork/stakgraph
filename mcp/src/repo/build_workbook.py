#!/usr/bin/env python3
"""
build_workbook.py — openpyxl workbook builder for the generate_xlsx agent tool.

Usage: python3 build_workbook.py <input.json>

Input JSON schema:
{
  "output": "/path/to/output.xlsx",
  "filename": "optional-hint",   // ignored; output path is canonical
  "sheets": [
    {
      "name": "Sheet1",
      "rows": [["Header A", "Header B"], [1, 2], ...],  // optional row-based data
      "cells": [                                          // optional fine-grained cells
        { "ref": "A1", "value": "Label" },
        { "ref": "B2", "formula": "=Sheet2!B2+1" }
      ]
    }
  ]
}

- "rows" are written first (1-indexed, starting at row 1).
- "cells" are applied after rows and may override individual cells.
- "formula" values are written as-is (including cross-sheet refs like =Sheet2!B2).
- Both "value" and "formula" may coexist in a cell entry; "formula" takes precedence.
"""
import sys
import json
import os

try:
    from openpyxl import Workbook
except ImportError as e:
    print(f"openpyxl is not installed: {e}", file=sys.stderr)
    sys.exit(1)


def build(spec: dict) -> None:
    output_path = spec.get("output")
    if not output_path:
        print("Missing 'output' key in input spec", file=sys.stderr)
        sys.exit(1)

    sheets_spec = spec.get("sheets", [])
    if not sheets_spec:
        print("No sheets specified", file=sys.stderr)
        sys.exit(1)

    wb = Workbook()
    # Remove the default empty sheet so we control creation order.
    wb.remove(wb.active)  # type: ignore[arg-type]

    for sheet_spec in sheets_spec:
        name = sheet_spec.get("name", "Sheet")
        ws = wb.create_sheet(title=name)

        # Write row-based data first.
        for row_data in sheet_spec.get("rows") or []:
            ws.append(list(row_data))

        # Apply fine-grained cell overrides. Assigning via the A1-style ref
        # (ws["B2"] = ...) avoids depending on openpyxl.utils helpers whose
        # import paths have shifted across openpyxl versions.
        for cell_spec in sheet_spec.get("cells") or []:
            ref = cell_spec.get("ref")
            if not ref:
                continue
            if "formula" in cell_spec:
                ws[ref] = cell_spec["formula"]
            elif "value" in cell_spec:
                ws[ref] = cell_spec["value"]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    wb.save(output_path)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: build_workbook.py <input.json>", file=sys.stderr)
        sys.exit(1)
    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    build(spec)


if __name__ == "__main__":
    main()
