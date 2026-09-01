import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";
import { execFile } from "node:child_process";
import { mkdir, readFile } from "node:fs/promises";
import { isAbsolute, join, resolve, sep, dirname } from "node:path";

/**
 * The `generate_xlsx` equivalent for the harvey-deliver pipeline (equivalent
 * functionality, openpyxl subprocess — the same dependency the reading
 * instructions already assume on PATH). Builds a real .xlsx from a sheets
 * definition, written INSIDE this run's artifacts dir. Cell strings starting
 * with "=" are live formulas (openpyxl semantics).
 */
const BUILDER = `
import json, sys
from openpyxl import Workbook
spec = json.load(sys.stdin)
wb = Workbook()
wb.remove(wb.active)
for sheet in spec["sheets"]:
    ws = wb.create_sheet(title=str(sheet["name"])[:31])
    for row in sheet.get("rows", []):
        ws.append(row)
wb.save(spec["out"])
`;

async function artifactsPath(ctx: StepContext | undefined, filename: string): Promise<string> {
  const c = ctx as StepContext<VeinCapabilities> | undefined;
  const artifacts = c?.services?.artifacts;
  if (!artifacts) throw new Error("artifacts capability unavailable — is this the lab vein?");
  const base = await artifacts.dir(c!.runId);
  if (isAbsolute(filename) || filename.split(/[\\/]/).includes("..")) {
    throw new Error(`filename must be a relative path without '..': "${filename}"`);
  }
  const path = resolve(join(base, filename));
  if (path !== base && !path.startsWith(base + sep)) {
    throw new Error(`filename escapes the run's artifacts dir: "${filename}"`);
  }
  await mkdir(dirname(path), { recursive: true });
  return path;
}

export default defineStep({
  type: "harvey/generate-xlsx",
  description:
    "Generate a real .xlsx spreadsheet, written INSIDE this run's artifacts directory (your working " +
    "directory). `filename` is relative to it — e.g. 'output/model.xlsx' (parent dirs are created). " +
    "`sheets` is a list of { name, rows } where rows is a 2D array of cell values (strings/numbers); a " +
    "string cell starting with '=' becomes a LIVE formula (e.g. '=SUM(B2:B9)'), so derive computed values " +
    "with formulas over input cells rather than hardcoding them. Returns { path, filename, bytes }.",
  input: z.object({
    filename: z
      .string()
      .min(1)
      .describe("Output .xlsx path RELATIVE to the run's artifacts dir, e.g. 'output/model.xlsx'."),
    sheets: z
      .array(
        z.object({
          name: z.string().min(1).describe("Sheet (tab) name, max 31 chars."),
          rows: z
            .array(z.array(z.union([z.string(), z.number(), z.null()])))
            .describe("2D array of cell values, row-major. '=...' strings are live formulas."),
        }),
      )
      .min(1)
      .describe("The workbook's sheets."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const outFile = await artifactsPath(ctx as StepContext, cfg.filename);
      const spec = JSON.stringify({ out: outFile, sheets: cfg.sheets });
      await new Promise<void>((resolvePromise, reject) => {
        const child = execFile("python3", ["-c", BUILDER], { timeout: 60_000 }, (err, _stdout, stderr) => {
          if (err) reject(new Error(`${err.message}${stderr ? ` — ${stderr.slice(0, 500)}` : ""}`));
          else resolvePromise();
        });
        child.stdin!.end(spec);
      });
      const bytes = (await readFile(outFile)).length;
      if (bytes === 0) return `harvey/generate-xlsx failed: empty file at ${cfg.filename}`;
      return { path: outFile, filename: cfg.filename, bytes };
    } catch (err: any) {
      return `harvey/generate-xlsx failed: ${err?.message ?? String(err)} (needs python3 + openpyxl on PATH)`;
    }
  },
});
