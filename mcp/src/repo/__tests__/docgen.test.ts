/**
 * Tests for generate_docx / generate_xlsx tools:
 *   - Unit: registration gating in get_tools (off by default, on when truthy, description override)
 *   - Unit: normalizeToolsConfig flat-string branch resolves new names
 *   - Integration: runDocx / runXlsx produce real files (requires pandoc + python3/openpyxl)
 */
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { existsSync, rmSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execSync } from "node:child_process";

// ─── helpers ────────────────────────────────────────────────────────────────

function hasBinary(name: string): boolean {
  try { execSync(`which ${name}`, { stdio: "ignore" }); return true; }
  catch { return false; }
}

function hasPythonModule(mod: string): boolean {
  try {
    execSync(`python3 -c "import ${mod}"`, { stdio: "ignore" });
    return true;
  } catch { return false; }
}

const hasPandoc = hasBinary("pandoc");
const hasOpenpyxl = hasPythonModule("openpyxl");

// ─── Unit: normalizeToolsConfig flat-string branch ───────────────────────────

describe("normalizeToolsConfig — generate_docx / generate_xlsx", () => {
  it("parses 'generate_docx true' from a flat string", async () => {
    const { normalizeToolsConfig } = await import("../tools.js");
    const cfg = normalizeToolsConfig("generate_docx true");
    assert.ok(cfg, "expected a config object");
    assert.strictEqual((cfg as any).generate_docx, true);
  });

  it("parses 'generate_xlsx true' from a flat string", async () => {
    const { normalizeToolsConfig } = await import("../tools.js");
    const cfg = normalizeToolsConfig("generate_xlsx true");
    assert.ok(cfg, "expected a config object");
    assert.strictEqual((cfg as any).generate_xlsx, true);
  });

  it("parses 'generate_docx false' correctly", async () => {
    const { normalizeToolsConfig } = await import("../tools.js");
    const cfg = normalizeToolsConfig("generate_docx false");
    assert.ok(cfg, "expected a config object");
    assert.strictEqual((cfg as any).generate_docx, false);
  });

  it("treats a non-boolean-keyword token as a custom description override", async () => {
    const { normalizeToolsConfig } = await import("../tools.js");
    const cfg = normalizeToolsConfig("generate_xlsx my-custom-desc");
    assert.ok(cfg, "expected a config object");
    assert.strictEqual((cfg as any).generate_xlsx, "my-custom-desc");
  });

  it("parses both names together in one flat string", async () => {
    const { normalizeToolsConfig } = await import("../tools.js");
    const cfg = normalizeToolsConfig("generate_docx true generate_xlsx true");
    assert.ok(cfg);
    assert.strictEqual((cfg as any).generate_docx, true);
    assert.strictEqual((cfg as any).generate_xlsx, true);
  });
});

// ─── Unit: get_tools registration gating ────────────────────────────────────

describe("get_tools — generate_docx / generate_xlsx registration", () => {
  // get_tools requires a repoPath + apiKey; we use /tmp and an empty key for unit tests.
  const REPO = "/tmp";
  const KEY = "";

  it("does NOT include generate_docx when toolsConfig is undefined", async () => {
    const { get_tools } = await import("../tools.js");
    const tools = await get_tools(REPO, KEY, undefined, undefined);
    assert.ok(!("generate_docx" in tools), "generate_docx must be absent by default");
  });

  it("does NOT include generate_xlsx when toolsConfig is undefined", async () => {
    const { get_tools } = await import("../tools.js");
    const tools = await get_tools(REPO, KEY, undefined, undefined);
    assert.ok(!("generate_xlsx" in tools), "generate_xlsx must be absent by default");
  });

  it("does NOT include generate_docx when toolsConfig.generate_docx is false", async () => {
    const { get_tools } = await import("../tools.js");
    const tools = await get_tools(REPO, KEY, undefined, { generate_docx: false });
    assert.ok(!("generate_docx" in tools));
  });

  it("registers generate_docx when toolsConfig.generate_docx is true", async () => {
    const { get_tools } = await import("../tools.js");
    const tools = await get_tools(REPO, KEY, undefined, { generate_docx: true });
    assert.ok("generate_docx" in tools, "generate_docx must be registered when truthy");
  });

  it("registers generate_xlsx when toolsConfig.generate_xlsx is true", async () => {
    const { get_tools } = await import("../tools.js");
    const tools = await get_tools(REPO, KEY, undefined, { generate_xlsx: true });
    assert.ok("generate_xlsx" in tools, "generate_xlsx must be registered when truthy");
  });

  it("uses a custom string as the description for generate_docx", async () => {
    const { get_tools } = await import("../tools.js");
    const customDesc = "My custom docx tool description";
    const tools = await get_tools(REPO, KEY, undefined, { generate_docx: customDesc });
    const tool = (tools as any).generate_docx;
    assert.ok(tool, "generate_docx must be registered");
    assert.strictEqual(tool.description, customDesc);
  });

  it("uses a custom string as the description for generate_xlsx", async () => {
    const { get_tools } = await import("../tools.js");
    const customDesc = "My custom xlsx tool description";
    const tools = await get_tools(REPO, KEY, undefined, { generate_xlsx: customDesc });
    const tool = (tools as any).generate_xlsx;
    assert.ok(tool, "generate_xlsx must be registered");
    assert.strictEqual(tool.description, customDesc);
  });
});

// ─── Integration: runDocx ────────────────────────────────────────────────────

describe("runDocx — integration", { skip: !hasPandoc ? "pandoc not installed" : undefined }, () => {
  let tmpArtifacts: string;

  before(() => {
    tmpArtifacts = mkdtempSync(join(tmpdir(), "docgen-test-"));
    process.env.AGENT_ARTIFACTS_DIR = tmpArtifacts;
  });

  after(() => {
    delete process.env.AGENT_ARTIFACTS_DIR;
    try { rmSync(tmpArtifacts, { recursive: true, force: true }); } catch {}
  });

  it("produces a .docx file and returns a download path", async () => {
    // Re-import after env is set so artifactsDir picks it up
    const { runDocx } = await import("../docgen.js?t=" + Date.now());
    const result = await runDocx({ markdown: "# Hello\n\nThis is a test document." });
    assert.match(result, /Generated:.*\/repo\/agent\/file\?path=/, "result must contain download path");

    // extract path from the URL-encoded result
    const match = result.match(/path=(.+)$/);
    assert.ok(match, "result must contain a path= query param");
    const filePath = decodeURIComponent(match[1]);
    assert.ok(filePath.endsWith(".docx"), "output must be a .docx file");
    assert.ok(existsSync(filePath), `file must exist at ${filePath}`);
  });

  it("returns a non-fatal error string on invalid input (empty markdown is ok, pandoc error would be bad args)", async () => {
    // We test the failure path by supplying an invalid template name that
    // doesn't crash the agent (template is silently ignored, docx still generated or error returned non-fatally).
    const { runDocx } = await import("../docgen.js");
    const result = await runDocx({ markdown: "# test", template: "../../etc/passwd" });
    // Either succeeds (template ignored) or returns non-fatal error string — must not throw
    assert.ok(typeof result === "string", "result must be a string (non-fatal)");
  });
});

// ─── Integration: runXlsx ───────────────────────────────────────────────────

describe("runXlsx — integration", { skip: !hasOpenpyxl ? "openpyxl not installed" : undefined }, () => {
  let tmpArtifacts: string;

  before(() => {
    tmpArtifacts = mkdtempSync(join(tmpdir(), "xlsxgen-test-"));
    process.env.AGENT_ARTIFACTS_DIR = tmpArtifacts;
  });

  after(() => {
    delete process.env.AGENT_ARTIFACTS_DIR;
    try { rmSync(tmpArtifacts, { recursive: true, force: true }); } catch {}
  });

  it("produces a .xlsx file with multiple sheets and returns a download path", async () => {
    const { runXlsx } = await import("../docgen.js?t=" + Date.now());
    const result = await runXlsx({
      filename: "test-workbook",
      sheets: [
        {
          name: "Sheet1",
          rows: [["A", "B"], [1, 2], [3, 4]],
          cells: [{ ref: "C1", value: "Total" }, { ref: "C2", formula: "=Sheet2!B1" }],
        },
        {
          name: "Sheet2",
          rows: [["X", "Y"], [10, 20]],
        },
      ],
    });

    assert.match(result, /Generated:.*\/repo\/agent\/file\?path=/, "result must contain download path");

    const match = result.match(/path=(.+)$/);
    assert.ok(match, "result must contain a path= query param");
    const filePath = decodeURIComponent(match[1]);
    assert.ok(filePath.endsWith(".xlsx"), "output must be a .xlsx file");
    assert.ok(existsSync(filePath), `file must exist at ${filePath}`);
  });

  it("returns a non-fatal error string on malformed input (empty sheets)", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({ sheets: [] });
    assert.ok(typeof result === "string", "result must be a string (non-fatal)");
    assert.match(result, /generate_xlsx failed:/, "must be a non-fatal failure string");
  });
});

// ─── Unit: normalizeToolsConfig — generate_xlsx_computed ────────────────────

describe("normalizeToolsConfig — generate_xlsx_computed", () => {
  it("parses 'generate_xlsx_computed true' from a flat string", async () => {
    const { normalizeToolsConfig } = await import("../tools.js");
    const cfg = normalizeToolsConfig("generate_xlsx_computed true");
    assert.ok(cfg, "expected a config object");
    assert.strictEqual((cfg as any).generate_xlsx_computed, true);
  });

  it("parses 'generate_xlsx_computed false' correctly", async () => {
    const { normalizeToolsConfig } = await import("../tools.js");
    const cfg = normalizeToolsConfig("generate_xlsx_computed false");
    assert.ok(cfg, "expected a config object");
    assert.strictEqual((cfg as any).generate_xlsx_computed, false);
  });

  it("treats a non-boolean-keyword token as a custom description override", async () => {
    const { normalizeToolsConfig } = await import("../tools.js");
    const cfg = normalizeToolsConfig("generate_xlsx_computed my-custom-desc");
    assert.ok(cfg, "expected a config object");
    assert.strictEqual((cfg as any).generate_xlsx_computed, "my-custom-desc");
  });

  it("parses generate_xlsx and generate_xlsx_computed together", async () => {
    const { normalizeToolsConfig } = await import("../tools.js");
    const cfg = normalizeToolsConfig("generate_xlsx true generate_xlsx_computed true");
    assert.ok(cfg);
    assert.strictEqual((cfg as any).generate_xlsx, true);
    assert.strictEqual((cfg as any).generate_xlsx_computed, true);
  });
});

// ─── Unit: get_tools — generate_xlsx_computed registration ──────────────────

describe("get_tools — generate_xlsx_computed registration", () => {
  const REPO = "/tmp";
  const KEY = "";

  it("does NOT include generate_xlsx_computed when toolsConfig is undefined", async () => {
    const { get_tools } = await import("../tools.js");
    const tools = await get_tools(REPO, KEY, undefined, undefined);
    assert.ok(!("generate_xlsx_computed" in tools), "generate_xlsx_computed must be absent by default");
  });

  it("does NOT include generate_xlsx_computed when toolsConfig.generate_xlsx_computed is false", async () => {
    const { get_tools } = await import("../tools.js");
    const tools = await get_tools(REPO, KEY, undefined, { generate_xlsx_computed: false });
    assert.ok(!("generate_xlsx_computed" in tools));
  });

  it("registers generate_xlsx_computed when toolsConfig.generate_xlsx_computed is true", async () => {
    const { get_tools } = await import("../tools.js");
    const tools = await get_tools(REPO, KEY, undefined, { generate_xlsx_computed: true });
    assert.ok("generate_xlsx_computed" in tools, "generate_xlsx_computed must be registered when truthy");
  });

  it("uses a custom string as the description for generate_xlsx_computed", async () => {
    const { get_tools } = await import("../tools.js");
    const customDesc = "My custom computed xlsx description";
    const tools = await get_tools(REPO, KEY, undefined, { generate_xlsx_computed: customDesc });
    const t = (tools as any).generate_xlsx_computed;
    assert.ok(t, "generate_xlsx_computed must be registered");
    assert.strictEqual(t.description, customDesc);
  });

  it("generate_xlsx is unaffected when generate_xlsx_computed is enabled", async () => {
    const { get_tools } = await import("../tools.js");
    const tools = await get_tools(REPO, KEY, undefined, {
      generate_xlsx: true,
      generate_xlsx_computed: true,
    });
    assert.ok("generate_xlsx" in tools, "generate_xlsx must still be registered");
    assert.ok("generate_xlsx_computed" in tools, "generate_xlsx_computed must be registered");
  });
});

// ─── Integration: runXlsx (generate_xlsx_computed) ──────────────────────────

describe("runXlsx computed — integration", { skip: !hasOpenpyxl ? "openpyxl not installed" : undefined }, () => {
  let tmpArtifacts: string;

  before(() => {
    tmpArtifacts = mkdtempSync(join(tmpdir(), "xlsxcomputed-test-"));
    process.env.AGENT_ARTIFACTS_DIR = tmpArtifacts;
  });

  after(() => {
    delete process.env.AGENT_ARTIFACTS_DIR;
    try { rmSync(tmpArtifacts, { recursive: true, force: true }); } catch {}
  });

  /** Helper: read a cell value from a generated xlsx file using python3+openpyxl */
  async function readCell(filePath: string, sheet: string, ref: string): Promise<any> {
    const { execSync } = await import("node:child_process");
    const script = `
import openpyxl, sys
wb = openpyxl.load_workbook(sys.argv[1], data_only=True)
ws = wb[sys.argv[2]]
print(repr(ws[sys.argv[3]].value))
`.trim();
    const out = execSync(`python3 -c "${script.replace(/"/g, '\\"').replace(/\n/g, ';')}" "${filePath}" "${sheet}" "${ref}"`, {
      encoding: "utf8",
    }).trim();
    // Parse Python repr: None -> null, numbers as numbers, strings
    if (out === "None") return null;
    return JSON.parse(out);
  }

  /** Simpler helper using inline python */
  function readCellSync(filePath: string, sheet: string, ref: string): any {
    const { execSync } = require("child_process");
    const out = execSync(
      `python3 -c "import openpyxl; wb=openpyxl.load_workbook('${filePath}', data_only=True); ws=wb['${sheet}']; print(ws['${ref}'].value)"`,
      { encoding: "utf8" }
    ).trim();
    if (out === "None") return null;
    const n = Number(out);
    return isNaN(n) ? out : n;
  }

  it("column sum: writes literal numeric value (not formula string)", async () => {
    const { runXlsx } = await import("../docgen.js?t=" + Date.now());
    const result = await runXlsx({
      filename: "test-sum",
      sheets: [{
        name: "Sheet1",
        rows: [["Val"], [10], [20], [30]],
        computed: [{ ref: "A5", op: "sum", range: "A2:A4" }],
      }],
    }, "generate_xlsx_computed");

    assert.match(result, /Generated:.*\/repo\/agent\/file\?path=/);
    const filePath = decodeURIComponent(result.match(/path=(.+)$/)![1]);
    assert.ok(existsSync(filePath));
    const val = readCellSync(filePath, "Sheet1", "A5");
    assert.strictEqual(val, 60, "column sum must be 60");
  });

  it("row sum: writes literal numeric value", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-rowsum",
      sheets: [{
        name: "Sheet1",
        rows: [["A", "B", "C", "Total"], [5, 15, 30, null]],
        computed: [{ ref: "D2", op: "sum", range: "A2:C2" }],
      }],
    }, "generate_xlsx_computed");

    assert.match(result, /Generated:/);
    const filePath = decodeURIComponent(result.match(/path=(.+)$/)![1]);
    const val = readCellSync(filePath, "Sheet1", "D2");
    assert.strictEqual(val, 50, "row sum must be 50");
  });

  it("percent_of_total: default percent-scaled (×100), rounded to 2 decimals", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-pct",
      sheets: [{
        name: "Sheet1",
        rows: [["Val", "Total", "Pct"], [25, 100, null]],
        computed: [{ ref: "C2", op: "percent_of_total", value_ref: "A2", total_ref: "B2" }],
      }],
    }, "generate_xlsx_computed");

    assert.match(result, /Generated:/);
    const filePath = decodeURIComponent(result.match(/path=(.+)$/)![1]);
    const val = readCellSync(filePath, "Sheet1", "C2");
    assert.strictEqual(val, 25.00, "percent should be 25.00");
  });

  it("percent_of_total with as_fraction:true writes raw 0-1 ratio", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-fraction",
      sheets: [{
        name: "Sheet1",
        rows: [["Val", "Total", "Frac"], [25, 100, null]],
        computed: [{ ref: "C2", op: "percent_of_total", value_ref: "A2", total_ref: "B2", as_fraction: true }],
      }],
    }, "generate_xlsx_computed");

    assert.match(result, /Generated:/);
    const filePath = decodeURIComponent(result.match(/path=(.+)$/)![1]);
    const val = readCellSync(filePath, "Sheet1", "C2");
    assert.strictEqual(val, 0.25, "as_fraction should be 0.25");
  });

  it("ratio: writes value_ref/denominator_ref as literal number", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-ratio",
      sheets: [{
        name: "Sheet1",
        rows: [["Num", "Den", "Ratio"], [10, 4, null]],
        computed: [{ ref: "C2", op: "ratio", value_ref: "A2", denominator_ref: "B2", decimals: 4 }],
      }],
    }, "generate_xlsx_computed");

    assert.match(result, /Generated:/);
    const filePath = decodeURIComponent(result.match(/path=(.+)$/)![1]);
    const val = readCellSync(filePath, "Sheet1", "C2");
    assert.strictEqual(val, 2.5, "ratio 10/4 = 2.5");
  });

  it("ordered eval: sum written first feeds later percent_of_total", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-ordered",
      sheets: [{
        name: "Sheet1",
        rows: [["Val"], [10], [20], [30]],
        computed: [
          // First compute sum into A5
          { ref: "A5", op: "sum", range: "A2:A4" },
          // Then percent_of_total referencing A5 (which is 60)
          { ref: "B2", op: "percent_of_total", value_ref: "A2", total_ref: "A5" },
        ],
      }],
    }, "generate_xlsx_computed");

    assert.match(result, /Generated:/);
    const filePath = decodeURIComponent(result.match(/path=(.+)$/)![1]);
    const sumVal = readCellSync(filePath, "Sheet1", "A5");
    assert.strictEqual(sumVal, 60, "sum must be 60");
    const pctVal = readCellSync(filePath, "Sheet1", "B2");
    // 10/60 * 100 = 16.67
    assert.ok(Math.abs(pctVal - 16.67) < 0.01, `percent should be ~16.67, got ${pctVal}`);
  });

  it("cross-sheet ref: range and value_ref using Sheet2! qualifier", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-crosssheet",
      sheets: [
        {
          name: "Sheet1",
          rows: [["CrossSum", "CrossPct"]],
          computed: [
            { ref: "A2", op: "sum", range: "Sheet2!A1:A3" },
            { ref: "B2", op: "percent_of_total", value_ref: "Sheet2!A1", total_ref: "A2" },
          ],
        },
        {
          name: "Sheet2",
          rows: [[100], [200], [300]],
        },
      ],
    }, "generate_xlsx_computed");

    assert.match(result, /Generated:/);
    const filePath = decodeURIComponent(result.match(/path=(.+)$/)![1]);
    const sumVal = readCellSync(filePath, "Sheet1", "A2");
    assert.strictEqual(sumVal, 600, "cross-sheet sum should be 600");
    const pctVal = readCellSync(filePath, "Sheet1", "B2");
    // 100/600 * 100 = 16.67
    assert.ok(Math.abs(pctVal - 16.67) < 0.01, `cross-sheet percent should be ~16.67, got ${pctVal}`);
  });

  // ── Error class tests ────────────────────────────────────────────────────

  it("error: malformed/invalid cell ref returns non-fatal failure string", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-badref",
      sheets: [{
        name: "Sheet1",
        rows: [[1, 2, 3]],
        computed: [{ ref: "NOTAREF!", op: "sum", range: "A1:C1" }],
      }],
    }, "generate_xlsx_computed");
    assert.ok(typeof result === "string");
    assert.match(result, /generate_xlsx_computed failed:/, "must be non-fatal failure string");
  });

  it("error: empty/unresolved range returns non-fatal failure string", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-emptyrange",
      sheets: [{
        name: "Sheet1",
        rows: [[null, null, null]],
        computed: [{ ref: "A5", op: "sum", range: "A1:A3" }],
      }],
    }, "generate_xlsx_computed");
    assert.ok(typeof result === "string");
    assert.match(result, /generate_xlsx_computed failed:/, "empty range must produce non-fatal error");
  });

  it("error: non-numeric text in sum range returns non-fatal failure string", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-textinrange",
      sheets: [{
        name: "Sheet1",
        // Row 1 is a header that bleeds into the sum range
        rows: [["Header"], [10], [20]],
        computed: [{ ref: "A5", op: "sum", range: "A1:A3" }],
      }],
    }, "generate_xlsx_computed");
    assert.ok(typeof result === "string");
    assert.match(result, /generate_xlsx_computed failed:/, "non-numeric text in range must produce non-fatal error");
  });

  it("error: divide-by-zero in percent_of_total returns non-fatal failure string", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-divzero-pct",
      sheets: [{
        name: "Sheet1",
        rows: [[10, 0]],
        computed: [{ ref: "C1", op: "percent_of_total", value_ref: "A1", total_ref: "B1" }],
      }],
    }, "generate_xlsx_computed");
    assert.ok(typeof result === "string");
    assert.match(result, /generate_xlsx_computed failed:/, "divide-by-zero must produce non-fatal error");
  });

  it("error: divide-by-zero in ratio returns non-fatal failure string", async () => {
    const { runXlsx } = await import("../docgen.js");
    const result = await runXlsx({
      filename: "test-divzero-ratio",
      sheets: [{
        name: "Sheet1",
        rows: [[10, 0]],
        computed: [{ ref: "C1", op: "ratio", value_ref: "A1", denominator_ref: "B1" }],
      }],
    }, "generate_xlsx_computed");
    assert.ok(typeof result === "string");
    assert.match(result, /generate_xlsx_computed failed:/, "divide-by-zero in ratio must produce non-fatal error");
  });
});
