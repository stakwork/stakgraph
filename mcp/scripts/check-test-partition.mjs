#!/usr/bin/env node
/**
 * Fails when a `*.test.ts` file is claimed by neither test runner.
 *
 * `test:node` uses an opt-in glob list, so a new directory silently defaults
 * to "runs nowhere" — which is how src/gitree's 13 tests went unrun. This
 * makes that default loud.
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fg from "fast-glob";

const mcpRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Standalone scripts that assert at import time and call process.exit — they
// are not test files and must not be collected by either runner.
const ALLOWLIST = ["src/aieo/**"];

function nodeGlobs() {
  const pkg = JSON.parse(
    fs.readFileSync(path.join(mcpRoot, "package.json"), "utf-8"),
  );
  return [...pkg.scripts["test:node"].matchAll(/"([^"]*\.test\.ts)"/g)].map(
    (m) => m[1],
  );
}

function playwrightFiles() {
  // Report to a file, not stdout: a node:test file wrongly collected by
  // Playwright prints TAP at import time, which would corrupt stdout JSON.
  const out = path.join(
    fs.mkdtempSync(path.join(os.tmpdir(), "mcp-partition-")),
    "report.json",
  );
  execFileSync("npx", ["playwright", "test", "--list", "--reporter=json"], {
    cwd: mcpRoot,
    stdio: "ignore",
    env: { ...process.env, PLAYWRIGHT_JSON_OUTPUT_NAME: out },
  });
  const report = JSON.parse(fs.readFileSync(out, "utf-8"));
  fs.rmSync(path.dirname(out), { recursive: true, force: true });
  const files = new Set();
  const walk = (suites) => {
    for (const s of suites ?? []) {
      if (s.file) files.add(path.posix.join("src", s.file));
      walk(s.suites);
    }
  };
  walk(report.suites);
  return files;
}

const all = await fg("src/**/*.test.ts", { cwd: mcpRoot, ignore: ALLOWLIST });
const claimedByNode = new Set(await fg(nodeGlobs(), { cwd: mcpRoot }));
const claimedByPlaywright = playwrightFiles();

const orphans = all.filter(
  (f) => !claimedByNode.has(f) && !claimedByPlaywright.has(f),
);
const both = all.filter(
  (f) => claimedByNode.has(f) && claimedByPlaywright.has(f),
);

if (orphans.length || both.length) {
  for (const f of orphans) {
    console.error(`orphan: ${f} runs in neither test:node nor playwright`);
  }
  for (const f of both) {
    console.error(`double-claimed: ${f} runs in both runners`);
  }
  console.error(
    "\nFix by adding the file's directory to the test:node globs in " +
      "package.json (node tests) or to testIgnore in playwright.config.js " +
      "(playwright tests).",
  );
  process.exit(1);
}

console.log(
  `test partition ok: ${claimedByNode.size} node, ${claimedByPlaywright.size} playwright, 0 orphans`,
);
