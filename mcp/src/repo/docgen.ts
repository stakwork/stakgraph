import { randomUUID } from "crypto";
import { writeFileSync, readFileSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { resolve, isAbsolute, join, sep } from "node:path";
import JSZip from "jszip";
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
  // Inline Markdown content. For large documents prefer markdownPath: a
  // single tool call carrying a whole document as one JSON string must fit
  // in one model message (output-token capped) and fails for models that
  // can't reliably emit huge arguments.
  markdown?: string;
  // Path to a Markdown file to convert instead of inline content — build it
  // incrementally (e.g. bash appends), then convert. Relative paths resolve
  // against the repo directory.
  markdownPath?: string;
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
 * Generate a unique 8-digit uppercase hex ID, avoiding the reserved value
 * "00000000" and any IDs already in the `used` set. Adds the new ID to `used`.
 */
function genParaId(used: Set<string>): string {
  while (true) {
    const id = Math.floor(Math.random() * 0xffffffff + 1)
      .toString(16)
      .toUpperCase()
      .padStart(8, "0");
    if (id !== "00000000" && !used.has(id)) {
      used.add(id);
      return id;
    }
  }
}

/**
 * Post-process a `word/document.xml` string to stamp `w14:paraId` and
 * `w14:textId` attributes onto every `<w:p>` element that lacks them.
 *
 * - Declares `xmlns:w14` on the document root (idempotent).
 * - Registers `w14` in `mc:Ignorable` (appends if present, creates if absent).
 * - Matches `<w:p>`, `<w:p …>`, and `<w:p/>` but NOT `<w:pPr>`, `<w:pStyle>`, etc.
 * - Self-closing `<w:p/>` paragraphs are also stamped.
 * - Pre-scans existing `w14:paraId`/`w14:textId` values to avoid collisions.
 * - Never emits the reserved value `00000000`.
 *
 * Returns the rewritten XML and the count of paragraphs stamped.
 */
export function injectParaIds(documentXml: string): { xml: string; count: number } {
  // ── 1. Seed the used-ID set from pre-existing values in the XML ─────────
  const used = new Set<string>();
  const existingIds = documentXml.matchAll(/w14:(?:paraId|textId)="([0-9A-Fa-f]{8})"/g);
  for (const m of existingIds) {
    used.add(m[1].toUpperCase());
  }

  // ── 2. Ensure xmlns:w14 is declared on the <w:document …> opening tag ───
  // Match the opening tag of <w:document (everything up to the first >)
  let xml = documentXml.replace(
    /(<w:document\b[^>]*)(>)/,
    (full, attrs, close) => {
      if (attrs.includes("xmlns:w14=")) return full; // already declared

      // ── 2a. Handle mc:Ignorable ─────────────────────────────────────────
      let newAttrs = attrs;

      // Ensure xmlns:mc is declared
      if (!newAttrs.includes("xmlns:mc=")) {
        newAttrs += ' xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"';
      }

      // Append w14 to mc:Ignorable or create it
      if (/mc:Ignorable="([^"]*)"/.test(newAttrs)) {
        newAttrs = newAttrs.replace(/mc:Ignorable="([^"]*)"/, (_m: string, tokens: string) => {
          const list = tokens.split(/\s+/).filter(Boolean);
          if (!list.includes("w14")) list.push("w14");
          return `mc:Ignorable="${list.join(" ")}"`;
        });
      } else {
        newAttrs += ' mc:Ignorable="w14"';
      }

      // Add xmlns:w14
      newAttrs += ' xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"';

      return newAttrs + close;
    }
  );

  // ── 3. Stamp each <w:p …> / <w:p> / <w:p/> that lacks w14:paraId ───────
  // Lookahead ensures we match only true paragraph tags, not <w:pPr>, <w:pStyle>, etc.
  let count = 0;
  xml = xml.replace(/<w:p(?=[ />])((?:[^>]|>(?!\/w:p>))*?)(\/?)\s*>/g, (full, inner, selfClose) => {
    // Skip if already has a paraId
    if (/w14:paraId=/.test(inner)) return full;

    const paraId = genParaId(used);
    const textId = genParaId(used);
    count++;

    // Reconstruct: <w:p {existing attrs} w14:paraId="…" w14:textId="…"[/]>
    const attrs = inner.trimEnd();
    const sep = " ";
    const sc = selfClose === "/" ? "/" : "";
    return `<w:p${attrs}${sep}w14:paraId="${paraId}" w14:textId="${textId}"${sc}>`;
  });

  return { xml, count };
}

/**
 * Generate a .docx file from Markdown via Pandoc.
 * Returns a string with the download path on success, or a non-fatal error string.
 */
export async function runDocx(input: DocxInput, repoPath?: string): Promise<string> {
  let markdown: string;
  if (input.markdownPath) {
    const resolved = path.isAbsolute(input.markdownPath)
      ? input.markdownPath
      : path.join(repoPath ?? process.cwd(), input.markdownPath);
    try {
      markdown = readFileSync(resolved, "utf8");
    } catch (e) {
      return `generate_docx failed: could not read markdownPath "${resolved}": ${(e as Error).message}`;
    }
  } else if (input.markdown) {
    markdown = input.markdown;
  } else {
    return "generate_docx failed: provide either 'markdown' (inline) or 'markdownPath' (path to a .md file).";
  }

  const base = markdown
    .split("\n")[0]
    .replace(/^#+\s*/, "")
    .trim()
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .slice(0, 32) || "document";
  const uuid = randomUUID();
  const outFile = path.join(artifactsDir, `${base}-${uuid}.docx`);
  const tmpMd = path.join(tmpdir(), `docgen-${uuid}.md`);

  console.log(`===> generate_docx: ${outFile}`);

  writeFileSync(tmpMd, markdown, "utf8");

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

    // ── Post-process: stamp w14:paraId / w14:textId onto every <w:p> ──────
    try {
      const zipData = readFileSync(outFile);
      const zip = await JSZip.loadAsync(zipData);
      const docEntry = zip.file("word/document.xml");
      if (!docEntry) throw new Error("word/document.xml not found in docx");

      const docXml = await docEntry.async("string");
      const { xml: stamped, count } = injectParaIds(docXml);
      zip.file("word/document.xml", stamped);

      const buf = await zip.generateAsync({
        type: "nodebuffer",
        compression: "DEFLATE",
        compressionOptions: { level: 6 },
      });
      writeFileSync(outFile, buf);
      console.log(`===> generate_docx: stamped ${count} paraIds`);
    } catch (e: any) {
      const msg = e?.message || String(e);
      console.error(`===> generate_docx failed (paraId stamping): ${msg}`);
      return `generate_docx failed: ${msg}`;
    }

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
