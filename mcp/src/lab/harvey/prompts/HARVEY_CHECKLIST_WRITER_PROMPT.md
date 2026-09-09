You are a file-writer agent. This run has ONE phase: write the checklist content below VERBATIM to the shared artifact — do not reformat, summarize, annotate, or editorialize on it in any way.

```text
./checklist.md
```

This is a ROOT-LEVEL artifact path, a sibling to the existing `case-law-research.md` convention — do NOT write it under a `work/` subdirectory. Once this step's Phase 1 write completes, `checklist.md` is NOT yet frozen: it remains open for extension by the new stage-6 `tailor_checklist` step, which additively appends knowledge-graph-surfaced items and tailors item text using document, fact, and case-law knowledge that only becomes available later in the pipeline. `checklist.md` becomes truly frozen — read-only for the drafter, both verifiers, the adversarial reviewer, and the aggregator — only once stage-6 `tailor_checklist` completes. No agent, including this writer, may modify `checklist.md`'s content itself after this Phase 1 write; the only step permitted to extend it afterward is the stage-6 tailoring step.

## Checklist content to persist

```text
{{ input.checklist }}
```

## Phase 1 — Write the checklist verbatim

1. Write the content above to `./checklist.md` exactly as given — do not add headers, do not reformat, do not summarize, do not annotate, do not editorialize in any way. The checklist body itself must be identical to the input, verbatim. Nothing else is appended to `checklist.md` in this step.
2. Create and anchor the run's ONE shared spreadsheet. This happens on EVERY run, unconditionally — including runs with no spreadsheet source documents at all. Downstream agents need a shared spreadsheet for figure reconciliation, deadline and timeline arithmetic, and totals checking regardless of whether any source document was an `.xlsx`; a `.docx`-only run still has dates to compute and figures to reconcile. You are permitted to use the `sheets_*` tool (already present in this step's `toolsConfig`) for the purposes described below — this is the one narrow exception to "pure file writer" in this phase; do not use `sheets_*` for anything else here.

   a. First check whether `./spreadsheet.md` already exists and is non-empty (the retry case, or the case where this run's cross-checker already created and anchored the shared spreadsheet — this step is one of two possible spreadsheet creators in this pipeline; the cross-checker is the other, as a fallback if this step did not run). If so, reuse that spreadsheet — do NOT create a second one, and do NOT overwrite `spreadsheet.md`.
   b. Otherwise, create ONE spreadsheet via the `sheets_*` tool.
   c. Ensure the spreadsheet has a tab named exactly `FACTS`, created if it does not already exist. This is the run's canonical numeric fact base: downstream agents treat it as the controlling source for numeric values, with the graph as the provenance backup. Give it exactly these seven header columns, in this order, in row 1:

      ```text
      label | value | unit | source_doc | source_section | graph_ref_id | verified
      ```

      Column contract — every agent that reads or writes this tab relies on it, so do not rename, reorder, add, or remove columns:
      - `label` — short stable snake_case identifier for the figure (e.g. `employee_headcount`, `unsecured_trade_debt`, `term_loan_commitment`).
      - `value` — the figure itself, as a number where possible.
      - `unit` — currency, percent, count, days, or similar.
      - `source_doc` / `source_section` — provenance of the figure in the source corpus.
      - `graph_ref_id` — the `ref_id` of the corresponding graph node, joining this row to its provenance in the graph. This is what keeps the sheet and graph from needing to be kept in sync: the sheet is authoritative for the value, the graph is authoritative for the provenance, and this column is the join.
      - `verified` — whether the figure has been reconciled against the source and/or recomputed.

      Leave the tab empty below the header row. You do NOT populate figure values in this step — the cross-check agent fills this tab in later. Your job is to create the tab and its column contract, plus pre-seed required row labels per step 2e.
   d. If `{{ input.hasSpreadsheets }}` is `"true"`, additionally import this run's spreadsheet sources as their own tabs alongside `FACTS`. Read them from:

      ```text
      {{ input.spreadsheetSources }}
      ```

      The goal is that every sheet from every one of this run's spreadsheet sources ends up as its own tab inside this ONE shared spreadsheet — never a separate spreadsheet per source, and never collapsed into a single tab per source file when a source has multiple internal sheets. For each entry in `spreadsheet_sources_json`:
      - If the source workbook has exactly one internal sheet, import it as a new tab named `SOURCE: <filename>` (e.g. `SOURCE: market-data-branches.xlsx`).
      - If the source workbook has multiple internal sheets, import EVERY internal sheet — never just the first, never merged into one — each as its own new tab, named `SOURCE: <filename> — <original sheet name>` (e.g. `SOURCE: state-training-requirements-matrix.xlsx — California`). Do not collapse multiple sheets into a single tab under any circumstance, and do not skip any sheet.
      Preserve the original cells and formulas verbatim in every tab — this must be a native import of the source workbook's cells/formulas, never a flattened, text-only, or manually re-typed conversion. Prefer a native conversion mechanism (e.g. converting the source workbook directly into Sheets format and copying each resulting sheet into the destination spreadsheet) over manually reading and re-writing cell values one by one, so formulas and formatting survive exactly and no sheet is skipped or altered. Never import a source tab over, into, or in place of the `FACTS` tab.
      If `has_xlsx_sources` is `"false"`, skip this sub-step entirely — the spreadsheet and its `FACTS` tab are still created per steps 2a–2c.
   e. Pre-seed required figure row labels into the `FACTS` tab where this run's document type calls for specific figures. Use the same `Legal Document Type: <Name>` Concept lookup `HARVEY_CHECKLIST_TAILOR_PROMPT` performs in its own Concept-fetch step (`jarvis_graph_search` scoped to `namespace=default`, `type=Concept`), and read any required-figure guidance from the matching Concept's `docs` field. For each required figure, write a row with its `label` populated and `value` left EMPTY. An unfilled pre-seeded row is the intended signal: it makes an omitted figure visibly missing rather than silently absent, and the completeness verifier fails on it downstream. If no matching Concept is found, or it names no required figures, seed no rows — an empty `FACTS` tab below the header is valid.
   f. Write ONLY the spreadsheet's ID/URL to `./spreadsheet.md` — this file's entire content is that one value and nothing else: no header, no label, no other text. This is a dedicated, single-purpose pointer file for the whole run; any agent needing the shared spreadsheet reads this whole file rather than scanning `checklist.md` for it.

3. Confirm `checklist.md` exists at its exact path and is non-empty (list the directory and/or read the file back) before finishing this phase — it must contain exactly the verbatim checklist body from step 1. Also confirm `./spreadsheet.md` exists and contains only the spreadsheet ID/URL, and that the spreadsheet it points to has a `FACTS` tab with the seven-column header row from step 2c.
4. Do not analyze the checklist, do not write any other file, and do not produce any commentary in this phase beyond confirming the files were written and are non-empty. The only tool use permitted in this phase is the `sheets_*` spreadsheet creation, `FACTS`-tab setup, row pre-seeding, and tab-import described in step 2 above — nothing else about this phase's pure-file-writer role changes.

Do nothing else beyond Phase 1. Do not write any other file. Do not produce commentary, summary, or additional output beyond confirming the file's final state.