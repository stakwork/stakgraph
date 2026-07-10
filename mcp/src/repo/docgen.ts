import { randomUUID } from "crypto";
import { writeFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { resolve, isAbsolute, join, sep } from "node:path";
import { AGENT_ARTIFACTS_DIR } from "./artifacts.js";

const execFileAsync = promisify(execFile);

/** Resolved artifacts output directory: durable volume if set, otherwise /tmp */
const artifactsDir = AGENT_ARTIFACTS_DIR ?? tmpdir();

/**
 * Bundled docgen templates directory — resolve against import.meta.url so it
 * works both from source (ts-node) and compiled (build/repo/docgen.js).
 */
const TEMPLATES_DIR = path.join(
  path.dirname(new URL(import.meta.url).pathname),
  "docgen-templates"
);

/** Containment guard: resolve p against TEMPLATES_DIR and throw if it escapes. */
function resolveTemplate(template: string): string {
  const root = resolve(TEMPLATES_DIR);
  const target = resolve(isAbsolute(template) ? template : join(root, template));
  if (!(target === root || target.startsWith(root + sep))) {
    throw new Error(`template path "${template}" escapes the templates directory`);
  }
  return target;
}

export interface DocxInput {
  markdown: string;
  template?: string;
}

export interface ComputedCell {
  ref: string;
  op: "sum" | "percent_of_total" | "ratio";
  range?: string;
  value_ref?: string;
  total_ref?: string;
  denominator_ref?: string;
  decimals?: number;
  as_fraction?: boolean;
}

export interface XlsxSheet {
  name: string;
  rows?: (string | number)[][];
  cells?: Array<{
    ref: string;
    value?: string | number;
    formula?: string;
  }>;
  computed?: ComputedCell[];
}

export interface XlsxInput {
  filename?: string;
  sheets: XlsxSheet[];
}

/**
 * Generate a .docx file from Markdown via Pandoc.
 * Returns a string with the download path on success, or a non-fatal error string.
 */
export async function runDocx(input: DocxInput): Promise<string> {
  const base = input.markdown
    .split("\n")[0]
    .replace(/^#+\s*/, "")
    .trim()
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .slice(0, 32) || "document";
  const uuid = randomUUID();
  const outFile = path.join(artifactsDir, `${base}-${uuid}.docx`);
  const tmpMd = path.join(tmpdir(), `docgen-${uuid}.md`);

  console.log(`===> generate_docx: ${outFile}`);

  writeFileSync(tmpMd, input.markdown, "utf8");

  const args = [tmpMd, "-o", outFile];

  if (input.template) {
    try {
      const resolvedTemplate = resolveTemplate(input.template);
      args.push(`--reference-doc=${resolvedTemplate}`);
    } catch (e) {
      console.warn(`[docgen] ignoring template: ${(e as Error).message}`);
    }
  }

  try {
    await execFileAsync("pandoc", args);
    console.log(`===> generate_docx: written ${outFile}`);
    return `Generated: /repo/agent/file?path=${encodeURIComponent(outFile)}`;
  } catch (e: any) {
    const stderr = e?.stderr || String(e);
    console.error(`===> generate_docx failed: ${stderr}`);
    return `generate_docx failed: ${stderr}`;
  } finally {
    try { unlinkSync(tmpMd); } catch {}
  }
}

/**
 * Generate a .xlsx file from a workbook definition via build_workbook.py (openpyxl).
 * Returns a string with the download path on success, or a non-fatal error string.
 * @param logLabel - Label used in log/error strings (default: "generate_xlsx")
 */
export async function runXlsx(input: XlsxInput, logLabel = "generate_xlsx"): Promise<string> {
  const base = (input.filename || "workbook")
    .replace(/\.xlsx$/i, "")
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .slice(0, 32);
  const uuid = randomUUID();
  const outFile = path.join(artifactsDir, `${base}-${uuid}.xlsx`);
  const tmpJson = path.join(tmpdir(), `docgen-${uuid}.json`);

  console.log(`===> ${logLabel}: ${outFile}`);

  const payload = JSON.stringify({ ...input, output: outFile });
  writeFileSync(tmpJson, payload, "utf8");

  const scriptPath = path.join(
    path.dirname(new URL(import.meta.url).pathname),
    "build_workbook.py"
  );

  try {
    await execFileAsync("python3", [scriptPath, tmpJson]);
    console.log(`===> ${logLabel}: written ${outFile}`);
    return `Generated: /repo/agent/file?path=${encodeURIComponent(outFile)}`;
  } catch (e: any) {
    const stderr = e?.stderr || String(e);
    console.error(`===> ${logLabel} failed: ${stderr}`);
    return `${logLabel} failed: ${stderr}`;
  } finally {
    try { unlinkSync(tmpJson); } catch {}
  }
}
