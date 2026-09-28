/**
 * Seeded step files are published VERBATIM into the strut workspace
 * (`<workspace>/steps/_graph/<type>.ts` on a graph workspace), so any
 * relative import that escapes the step's own directory resolves against
 * the workspace, not this source tree. strut's discovery loader swallows
 * the resulting ERR_MODULE_NOT_FOUND and simply drops the step from the
 * registry — the workflow then fails at run time with an unknown step.
 *
 * `import type` is fine (erased by tsx). Value imports must be bare
 * specifiers that resolve from the app's node_modules (`strut`, `aieo`, …).
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const LAB = dirname(fileURLToPath(import.meta.url));

function stepFiles(): string[] {
  const out: string[] = [];
  for (const exp of readdirSync(LAB)) {
    const dir = join(LAB, exp, "steps");
    let entries: string[];
    try {
      entries = readdirSync(dir);
    } catch {
      continue;
    }
    for (const f of entries) {
      const full = join(dir, f);
      if (statSync(full).isFile() && f.endsWith(".ts") && !f.endsWith(".test.ts")) out.push(full);
    }
  }
  return out;
}

// A value import (not `import type`) whose specifier starts with `../`.
const ESCAPING_VALUE_IMPORT = /^import\s+(?!type\s)[^;]*?from\s+["']\.\.\//m;

describe("seeded lab steps", () => {
  const files = stepFiles();
  it("finds the seeded step files", () => {
    assert.ok(files.length > 20, `expected many step files, found ${files.length}`);
  });
  for (const file of files) {
    it(`${file.slice(LAB.length + 1)} has no value import escaping its step dir`, () => {
      const src = readFileSync(file, "utf-8");
      const m = src.match(ESCAPING_VALUE_IMPORT);
      assert.equal(
        m,
        null,
        `${file}: "${m?.[0]}" resolves against the workspace once seeded and breaks the step`,
      );
    });
  }
});
