import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolveInCwd, textEdit } from "./textEdit.js";

describe("resolveInCwd — multi-root path guard", () => {
  let cwd: string;
  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), "mcp-textedit-"));
  });
  afterEach(() => {
    rmSync(cwd, { recursive: true, force: true });
  });

  it("resolves a relative path under the primary root", () => {
    const result = resolveInCwd("foo.txt", [cwd, tmpdir()]);
    assert.equal(result, join(cwd, "foo.txt"));
  });

  it("accepts an absolute path under the repo root", () => {
    const abs = join(cwd, "bar.txt");
    assert.equal(resolveInCwd(abs, [cwd, tmpdir()]), abs);
  });

  it("accepts an absolute path under os.tmpdir()", () => {
    const tmp = join(tmpdir(), "scratch.py");
    assert.equal(resolveInCwd(tmp, [cwd, tmpdir()]), tmp);
  });

  it("throws for a path outside all roots (e.g. /etc/passwd)", () => {
    assert.throws(
      () => resolveInCwd("/etc/passwd", [cwd, tmpdir()]),
      /escapes the working directory/
    );
  });

  it("throws for traversal escape from the repo root", () => {
    assert.throws(
      () => resolveInCwd("../../../etc/passwd", [cwd, tmpdir()]),
      /escapes the working directory/
    );
  });
});

describe("textEdit — multi-root sandbox", () => {
  let cwd: string;
  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), "mcp-textedit-"));
  });
  afterEach(() => {
    rmSync(cwd, { recursive: true, force: true });
  });

  it("views a file via relative path under repo root (unchanged behaviour)", () => {
    writeFileSync(join(cwd, "a.txt"), "line1\nline2");
    const out = textEdit({ command: "view", path: "a.txt" }, [cwd, tmpdir()]);
    assert.equal(out, "1: line1\n2: line2");
  });

  it("accepts absolute path under the repo root", () => {
    const abs = join(cwd, "abs.txt");
    writeFileSync(abs, "hello");
    const out = textEdit({ command: "view", path: abs }, [cwd, tmpdir()]);
    assert.equal(out, "1: hello");
  });

  it("accepts absolute path under os.tmpdir() — create then str_replace round-trip", () => {
    const scratchPath = join(tmpdir(), `mcp-scratch-${Date.now()}.py`);
    try {
      // Create the scratch file under tmpdir()
      const createOut = textEdit(
        { command: "create", path: scratchPath, file_text: "x = 1\n" },
        [cwd, tmpdir()]
      );
      assert.match(createOut, /Successfully created/);
      // Edit it — this used to fail with "escapes the working directory"
      const replaceOut = textEdit(
        { command: "str_replace", path: scratchPath, old_str: "x = 1", new_str: "x = 42" },
        [cwd, tmpdir()]
      );
      assert.match(replaceOut, /Successfully replaced/);
      assert.equal(readFileSync(scratchPath, "utf-8"), "x = 42\n");
    } finally {
      rmSync(scratchPath, { force: true });
    }
  });

  it("refuses absolute path outside all roots (e.g. /etc/passwd)", () => {
    const out = textEdit({ command: "view", path: "/etc/passwd" }, [cwd, tmpdir()]);
    assert.match(out, /escapes the working directory/);
  });

  it("refuses traversal path escaping the working dir", () => {
    const out = textEdit({ command: "view", path: "../../../etc/passwd" }, [cwd, tmpdir()]);
    assert.match(out, /escapes the working directory/);
  });
});
