#!/usr/bin/env node
// Publish the built staktrak bundles into hive's public assets so the host page
// and the in-iframe recorder never run different builds (bundle skew was a real
// source of "which version failed?" ambiguity — see notes-staktrak-architecture.md P0).
//
// Usage:
//   npm run build && HIVE_DIR=../../../../hive npm run sync:hive
// If HIVE_DIR is unset, tries the conventional sibling checkout ../../../../hive
// relative to this package.
import { cp, mkdir, readFile, access } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgDir = resolve(here, "..");
const distDir = join(pkgDir, "dist");

const hiveDir = resolve(pkgDir, process.env.HIVE_DIR || "../../../../hive");
const targetDir = join(hiveDir, "public", "js");
const bundles = ["staktrak.js", "playwright-generator.js"];

async function exists(p) {
  try { await access(p); return true; } catch { return false; }
}

const { version } = JSON.parse(await readFile(join(pkgDir, "package.json"), "utf8"));

if (!(await exists(targetDir))) {
  console.error(`[sync:hive] target not found: ${targetDir}`);
  console.error(`[sync:hive] set HIVE_DIR to your hive checkout (got HIVE_DIR=${process.env.HIVE_DIR ?? "<unset>"})`);
  process.exit(1);
}

await mkdir(targetDir, { recursive: true });
for (const f of bundles) {
  const src = join(distDir, f);
  if (!(await exists(src))) {
    console.error(`[sync:hive] missing built bundle: ${src} — run \`npm run build\` first`);
    process.exit(1);
  }
  await cp(src, join(targetDir, f));
  console.log(`[sync:hive] ${f} → ${join(targetDir, f)}`);
}
console.log(`[sync:hive] published staktrak v${version} to hive.`);
