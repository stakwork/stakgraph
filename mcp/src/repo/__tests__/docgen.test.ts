/**
 * Tests for generate_docx / generate_xlsx tools:
 *   - Unit: registration gating in get_tools (off by default, on when truthy, description override)
 *   - Unit: normalizeToolsConfig flat-string branch resolves new names
 *   - Integration: runDocx / runXlsx produce real files (requires pandoc + python3/openpyxl)
 */
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { existsSync, rmSync, mkdtempSync, writeFileSync } from "node:fs";
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

  it("converts from markdownPath, resolving relative paths against repoPath", async () => {
    const { runDocx } = await import("../docgen.js?t=path" + Date.now());
    const repoDir = mkdtempSync(join(tmpdir(), "docgen-repo-"));
    try {
      writeFileSync(join(repoDir, "doc.md"), "# From File\n\nBuilt incrementally.", "utf8");
      const result = await runDocx({ markdownPath: "doc.md" }, repoDir);
      assert.match(result, /Generated:.*\/repo\/agent\/file\?path=/, "result must contain download path");
    } finally {
      try { rmSync(repoDir, { recursive: true, force: true }); } catch {}
    }
  });

  it("returns non-fatal errors for a missing markdownPath file and for neither input", async () => {
    const { runDocx } = await import("../docgen.js");
    const missing = await runDocx({ markdownPath: "/nonexistent/nope.md" });
    assert.match(missing, /generate_docx failed: could not read markdownPath/);
    const neither = await runDocx({});
    assert.match(neither, /generate_docx failed: provide either/);
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

// ─── Unit: injectParaIds ─────────────────────────────────────────────────────

describe("injectParaIds — unit tests", () => {
  // Minimal realistic document.xml snippet that exercises all tricky shapes
  const FIXTURE_XML = `<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" mc:Ignorable="wpc w14c" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">
<w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Title</w:t></w:r></w:p>
<w:p w:rsidR="00A1B2C3"><w:r><w:t>Para with attr</w:t></w:r></w:p>
<w:p/>
<w:p w14:paraId="AABBCCDD" w14:textId="11223344" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"><w:r><w:t>Already stamped</w:t></w:r></w:p>
<w:pStyle w:val="ShouldNotMatch"/>
<w:pPr><w:pStyle w:val="AlsoShouldNotMatch"/></w:pPr>
</w:body>
</w:document>`;

  it("returns xml and count", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const result = injectParaIds(FIXTURE_XML);
    assert.ok(typeof result.xml === "string", "xml must be a string");
    assert.ok(typeof result.count === "number", "count must be a number");
  });

  it("declares xmlns:w14 on the document root exactly once", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const { xml } = injectParaIds(FIXTURE_XML);
    const matches = xml.match(/xmlns:w14=/g) ?? [];
    // The existing pre-stamped paragraph also has xmlns:w14 on itself — only root is ours
    assert.ok(
      xml.includes('xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"'),
      "must declare w14 namespace"
    );
    // Should not be added twice to the <w:document> tag
    const docTag = xml.match(/<w:document[^>]*>/)?.[0] ?? "";
    assert.strictEqual(
      (docTag.match(/xmlns:w14=/g) ?? []).length,
      1,
      "xmlns:w14 must appear exactly once on <w:document>"
    );
  });

  it("is idempotent: re-running adds xmlns:w14 exactly once", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const { xml: once } = injectParaIds(FIXTURE_XML);
    const { xml: twice } = injectParaIds(once);
    const docTag = twice.match(/<w:document[^>]*>/)?.[0] ?? "";
    assert.strictEqual(
      (docTag.match(/xmlns:w14=/g) ?? []).length,
      1,
      "idempotent: xmlns:w14 still appears exactly once"
    );
  });

  it("appends w14 to existing mc:Ignorable token list without duplication", async () => {
    const { injectParaIds } = await import("../docgen.js");
    // FIXTURE_XML already has mc:Ignorable="wpc w14c" — w14 should be appended
    const { xml } = injectParaIds(FIXTURE_XML);
    const docTag = xml.match(/<w:document[^>]*>/)?.[0] ?? "";
    assert.ok(
      /mc:Ignorable="[^"]*\bw14\b/.test(docTag),
      "w14 must appear in mc:Ignorable"
    );
    // Must not duplicate existing tokens
    assert.ok(
      /mc:Ignorable="wpc w14c w14"/.test(docTag) ||
      /mc:Ignorable="[^"]*wpc[^"]*w14c[^"]*w14[^"]*"/.test(docTag),
      "existing tokens preserved, w14 appended"
    );
  });

  it("creates mc:Ignorable='w14' when attribute is absent", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const noIgnorable = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hi</w:t></w:r></w:p></w:body></w:document>`;
    const { xml } = injectParaIds(noIgnorable);
    const docTag = xml.match(/<w:document[^>]*>/)?.[0] ?? "";
    assert.ok(docTag.includes('mc:Ignorable="w14"'), "must add mc:Ignorable=\"w14\"");
    assert.ok(docTag.includes("xmlns:mc="), "must add xmlns:mc when absent");
  });

  it("does not add xmlns:mc if already present", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const withMc = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"><w:body><w:p/></w:body></w:document>`;
    const { xml } = injectParaIds(withMc);
    const docTag = xml.match(/<w:document[^>]*>/)?.[0] ?? "";
    // Should have exactly one xmlns:mc
    assert.strictEqual(
      (docTag.match(/xmlns:mc=/g) ?? []).length,
      1,
      "xmlns:mc must appear exactly once"
    );
  });

  it("stamps a plain <w:p> paragraph", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>`;
    const { xml: out, count } = injectParaIds(xml);
    assert.ok(out.includes("w14:paraId="), "must inject w14:paraId");
    assert.ok(out.includes("w14:textId="), "must inject w14:textId");
    assert.strictEqual(count, 1, "must stamp exactly 1 paragraph");
    assert.ok(!out.includes("<w:pw14"), "tag name must not merge into paraId");
    assert.match(out, /<w:p w14:paraId="[0-9A-F]{8}"/);
  });

  it("stamps a <w:p attr='x'> paragraph with existing attributes", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p w:rsidR="001122"><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>`;
    const { xml: out, count } = injectParaIds(xml);
    assert.ok(out.includes("w14:paraId="), "must inject w14:paraId");
    assert.strictEqual(count, 1, "must stamp exactly 1 paragraph");
    // Original attribute must be preserved
    assert.ok(out.includes('w:rsidR="001122"'), "must preserve existing attributes");
    assert.ok(!out.includes("<w:pw14"), "tag name must not merge into paraId");
    assert.match(out, /<w:p w:rsidR="001122" w14:paraId="[0-9A-F]{8}"/);
  });

  it("stamps a self-closing <w:p/> paragraph", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p/></w:body></w:document>`;
    const { xml: out, count } = injectParaIds(xml);
    assert.ok(out.includes("w14:paraId="), "self-closing <w:p/> must be stamped");
    assert.ok(out.includes("w14:textId="), "self-closing <w:p/> must have textId");
    assert.strictEqual(count, 1, "must stamp exactly 1 self-closing paragraph");
    // Must remain self-closing
    assert.ok(/w14:paraId="[0-9A-F]{8}"[^/]*\/>/.test(out) || /w14:paraId="[0-9A-F]{8}" w14:textId="[0-9A-F]{8}"\/>/.test(out),
      "self-closing tag must still close with />"
    );
    assert.ok(!out.includes("<w:pw14"), "tag name must not merge into paraId");
    assert.match(out, /w14:paraId="[0-9A-F]{8}" w14:textId="[0-9A-F]{8}"\/>/);
  });

  it("does NOT match <w:pPr> elements", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:pPr><w:pStyle w:val="Header"/></w:pPr></w:p></w:body></w:document>`;
    const { xml: out } = injectParaIds(xml);
    // <w:pPr> and <w:pStyle> must NOT have w14:paraId injected into them
    const pPrMatch = out.match(/<w:pPr[^>]*>/);
    assert.ok(!pPrMatch?.[0].includes("w14:paraId"), "<w:pPr> must not receive w14:paraId");
    const pStyleMatch = out.match(/<w:pStyle[^>]*>/);
    assert.ok(!pStyleMatch?.[0].includes("w14:paraId"), "<w:pStyle> must not receive w14:paraId");
  });

  it("does NOT match <w:pStyle> or other <w:p*> elements", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:pStyle w:val="Normal"/><w:pPr/><w:p/></w:body></w:document>`;
    const { count } = injectParaIds(xml);
    assert.strictEqual(count, 1, "only the <w:p/> should be counted, not <w:pStyle> or <w:pPr>");
  });

  it("leaves an existing w14:paraId untouched and does not reuse its value", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const existingId = "AABBCCDD";
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"><w:body>
<w:p w14:paraId="${existingId}" w14:textId="11223344"><w:r><w:t>Pre-stamped</w:t></w:r></w:p>
<w:p><w:r><w:t>Unstamped</w:t></w:r></w:p>
</w:body></w:document>`;
    const { xml: out, count } = injectParaIds(xml);
    // Pre-stamped paragraph must retain its original ID
    assert.ok(out.includes(`w14:paraId="${existingId}"`), "pre-existing paraId must be unchanged");
    // Only 1 new paragraph stamped
    assert.strictEqual(count, 1, "only 1 unstamped paragraph should be stamped");
    // New paraId must not be the same as the existing one
    const allParaIds = [...out.matchAll(/w14:paraId="([0-9A-F]{8})"/g)].map(m => m[1]);
    assert.strictEqual(new Set(allParaIds).size, allParaIds.length, "all paraIds must be unique");
  });

  it("generates uppercase 8-hex-digit IDs only", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p/><w:p/><w:p/>
</w:body></w:document>`;
    const { xml: out } = injectParaIds(xml);
    const ids = [...out.matchAll(/w14:(?:paraId|textId)="([^"]+)"/g)].map(m => m[1]);
    assert.ok(ids.length >= 6, "3 paragraphs × 2 IDs each = at least 6");
    for (const id of ids) {
      assert.match(id, /^[0-9A-F]{8}$/, `ID "${id}" must be uppercase 8-hex digits`);
      assert.notStrictEqual(id, "00000000", "must never emit 00000000");
    }
  });

  it("generates IDs within the OOXML-valid range (nonzero, high bit clear)", async () => {
    const { injectParaIds } = await import("../docgen.js");
    // 50 paragraphs × 2 IDs = 100 samples; under the old unmasked generator
    // each had a ~50% chance of an out-of-range value, so all passing by luck
    // is astronomically unlikely
    const paras = Array(50).fill("<w:p/>").join("\n");
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>${paras}</w:body></w:document>`;
    const { xml: out } = injectParaIds(xml);
    const ids = [...out.matchAll(/w14:(?:paraId|textId)="([0-9A-F]{8})"/g)].map(m => m[1]);
    assert.ok(ids.length >= 100, "50 paragraphs × 2 IDs each = at least 100");
    for (const id of ids) {
      const n = parseInt(id, 16);
      assert.ok(n >= 0x00000001 && n <= 0x7fffffff, `ID "${id}" must be in 0x00000001–0x7FFFFFFF`);
    }
  });

  it("generates unique IDs across multiple paragraphs", async () => {
    const { injectParaIds } = await import("../docgen.js");
    // Generate 20 paragraphs to surface collision issues
    const paras = Array(20).fill("<w:p/>").join("\n");
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>${paras}</w:body></w:document>`;
    const { xml: out, count } = injectParaIds(xml);
    assert.strictEqual(count, 20, "must stamp all 20 paragraphs");
    const ids = [...out.matchAll(/w14:(?:paraId|textId)="([^"]+)"/g)].map(m => m[1]);
    assert.strictEqual(new Set(ids).size, ids.length, "all generated IDs must be unique");
  });

  it("deduplicates new IDs against pre-seeded existing IDs from template", async () => {
    const { injectParaIds } = await import("../docgen.js");
    // Construct XML with many pre-existing IDs to exercise dedup logic
    const preSeeded = Array.from({ length: 100 }, (_, i) =>
      `w14:paraId="${i.toString(16).toUpperCase().padStart(8, "0")}" w14:textId="${(i + 200).toString(16).toUpperCase().padStart(8, "0")}"`
    );
    const existingParas = preSeeded
      .map(attrs => `<w:p ${attrs}><w:r><w:t>x</w:t></w:r></w:p>`)
      .join("\n");
    const newPara = `<w:p><w:r><w:t>new</w:t></w:r></w:p>`;
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"><w:body>${existingParas}${newPara}</w:body></w:document>`;
    const { xml: out, count } = injectParaIds(xml);
    assert.strictEqual(count, 1, "only the 1 new paragraph should be stamped");
    const allIds = [...out.matchAll(/w14:(?:paraId|textId)="([^"]+)"/g)].map(m => m[1]);
    assert.strictEqual(new Set(allIds).size, allIds.length, "no duplicate IDs after dedup");
  });

  it("never emits 00000000 as an ID", async () => {
    const { injectParaIds } = await import("../docgen.js");
    const paras = Array(50).fill("<w:p/>").join("\n");
    const xml = `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>${paras}</w:body></w:document>`;
    const { xml: out } = injectParaIds(xml);
    assert.ok(!out.includes('w14:paraId="00000000"'), "must never emit 00000000 as paraId");
    assert.ok(!out.includes('w14:textId="00000000"'), "must never emit 00000000 as textId");
  });

  it("fixture: combined shapes in one document", async () => {
    const { injectParaIds } = await import("../docgen.js");
    // Exercises: plain <w:p>, attributed <w:p>, self-closing <w:p/>,
    // pre-stamped paragraph, <w:pPr>, <w:pStyle> — all in one string
    const { xml: out, count } = injectParaIds(FIXTURE_XML);
    // 3 unstamped paragraphs: plain, with-attr, self-closing
    assert.strictEqual(count, 3, "must stamp exactly 3 paragraphs (not the pre-stamped one)");
    // Pre-existing ID must be preserved
    assert.ok(out.includes('w14:paraId="AABBCCDD"'), "pre-existing paraId must be preserved");
    // All generated IDs must be uppercase 8-hex
    const ids = [...out.matchAll(/w14:(?:paraId|textId)="([^"]+)"/g)].map(m => m[1]);
    for (const id of ids) {
      assert.match(id, /^[0-9A-F]{8}$/, `ID "${id}" must be uppercase 8-hex`);
    }
    // All IDs must be unique
    assert.strictEqual(new Set(ids).size, ids.length, "all IDs in fixture output must be unique");
    // w14 namespace declared on root
    const docTag = out.match(/<w:document[^>]*>/)?.[0] ?? "";
    assert.ok(docTag.includes("xmlns:w14="), "root must declare xmlns:w14");
    assert.ok(/mc:Ignorable="[^"]*\bw14\b/.test(docTag), "mc:Ignorable must include w14");
    assert.ok(!out.includes("<w:pw14"), "no merged tag/attribute in any stamped paragraph");
  });
});

// ─── Integration: runDocx paraId stamping ────────────────────────────────────

describe(
  "runDocx — paraId stamping integration",
  { skip: !hasPandoc ? "pandoc not installed" : undefined },
  () => {
    let tmpArtifacts: string;

    before(() => {
      tmpArtifacts = mkdtempSync(join(tmpdir(), "docgen-paraid-test-"));
      process.env.AGENT_ARTIFACTS_DIR = tmpArtifacts;
    });

    after(() => {
      delete process.env.AGENT_ARTIFACTS_DIR;
      try { rmSync(tmpArtifacts, { recursive: true, force: true }); } catch {}
    });

    it("output docx has w14 namespace and every <w:p> has w14:paraId", async () => {
      const { runDocx } = await import("../docgen.js?t=" + Date.now());
      const result = await runDocx({
        markdown: [
          "# Test Document",
          "",
          "First paragraph.",
          "",
          "Second paragraph with **bold** text.",
          "",
          "Third paragraph.",
        ].join("\n"),
      });
      assert.match(result, /Generated:.*\/repo\/agent\/file\?path=/, "must return download path");

      const match = result.match(/path=(.+)$/);
      assert.ok(match, "result must have path= param");
      const filePath = decodeURIComponent(match[1]);
      assert.ok(existsSync(filePath), `docx file must exist at ${filePath}`);

      // Unzip and inspect with JSZip
      const JSZip = (await import("jszip")).default;
      const { readFileSync: rfs } = await import("node:fs");
      const zip = await JSZip.loadAsync(rfs(filePath));

      // [Content_Types].xml must still be present
      assert.ok(zip.file("[Content_Types].xml") !== null, "[Content_Types].xml must be present");

      // word/document.xml must exist
      const docEntry = zip.file("word/document.xml");
      assert.ok(docEntry !== null, "word/document.xml must exist in the output docx");

      const docXml = await docEntry!.async("string");

      // Must declare xmlns:w14 on root
      const docTag = docXml.match(/<w:document[^>]*>/)?.[0] ?? "";
      assert.ok(
        docTag.includes('xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"'),
        "document root must declare xmlns:w14"
      );

      // Must list w14 in mc:Ignorable
      assert.ok(
        /mc:Ignorable="[^"]*\bw14\b/.test(docTag),
        "mc:Ignorable must include w14"
      );

      // Every <w:p …> must have w14:paraId
      const paragraphTags = [...docXml.matchAll(/<w:p(?=[ />])[^>]*>/g)].map(m => m[0]);
      assert.ok(paragraphTags.length > 0, "output document must contain at least one <w:p>");
      for (const tag of paragraphTags) {
        assert.ok(
          tag.includes("w14:paraId="),
          `paragraph tag missing w14:paraId: ${tag.slice(0, 120)}`
        );
      }

      // All paraIds must be unique uppercase 8-hex
      const allIds = [...docXml.matchAll(/w14:(?:paraId|textId)="([^"]+)"/g)].map(m => m[1]);
      assert.ok(allIds.length > 0, "must have at least some IDs");
      for (const id of allIds) {
        assert.match(id, /^[0-9A-F]{8}$/, `ID "${id}" must be uppercase 8-hex`);
        assert.notStrictEqual(id, "00000000", "must never emit 00000000");
      }
      assert.strictEqual(new Set(allIds).size, allIds.length, "all paraIds must be unique");
    });

    it("all non-document.xml entries are preserved intact", async () => {
      const { runDocx } = await import("../docgen.js");
      const result = await runDocx({ markdown: "# Preservation Test\n\nHello world." });
      assert.match(result, /Generated:/);

      const filePath = decodeURIComponent(result.match(/path=(.+)$/)![1]);
      const JSZip = (await import("jszip")).default;
      const { readFileSync: rfs } = await import("node:fs");
      const zip = await JSZip.loadAsync(rfs(filePath));

      // Verify essential docx structure entries are present
      const entryNames = Object.keys(zip.files);
      assert.ok(entryNames.includes("[Content_Types].xml"), "[Content_Types].xml must be present");
      assert.ok(
        entryNames.some(n => n.startsWith("word/")),
        "word/ directory entries must be present"
      );
      assert.ok(
        entryNames.some(n => n.startsWith("_rels/") || n === "_rels/.rels"),
        "_rels/ entries must be present"
      );
    });
  }
);
