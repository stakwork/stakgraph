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
