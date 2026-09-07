#!/usr/bin/env node
/**
 * Stage a self-contained vein directory for embedding in a native desktop
 * app (plans/local-desktop-and-stt.md §2.3, "phase A").
 *
 *   npm run package:desktop -- [--platform darwin-arm64] [--out dist-desktop] [--smoke] [--skip-build] [--embeddings]
 *
 * Output: <out>/vein/ containing package.json, build/ (server + steps as
 * loose files — the registry scans them), web/dist/, and a production-only
 * node_modules/ with exactly one sherpa-onnx platform package and only this
 * platform's onnxruntime-node binaries. The host adds an official Node
 * binary beside it and runs `node build/server.js` with the env in §2.5.
 *
 * Two size cuts, both on by default:
 *   - the embeddings stack (@huggingface/transformers + onnxruntime-web/-node
 *     + sharp, ~200 MB) is uninstalled — MiniLM only serves graph search,
 *     which needs a Neo4j the desktop build doesn't ship; `--embeddings`
 *     keeps it, and graph/embeddings.ts fails with a clear message without it;
 *   - sourcemaps, typings, and docs are stripped from node_modules (~100 MB).
 *
 * --smoke boots the staged copy from a temp directory (so nothing can leak
 * in from this checkout), with a workspace outside the package tree that
 * holds a custom step importing "vein", and checks /health, /audio/models
 * (`available: true` = the sherpa addon loaded) and that the step registered.
 *
 * Not a single-file build on purpose: the step loader scans directories and
 * the sherpa addon must be a real file beside the binary either way (§2.4).
 */
import { cp, mkdir, mkdtemp, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? (args[i + 1] && !args[i + 1].startsWith("--") ? args[i + 1] : true) : dflt;
};
const platform = String(flag("platform", `${process.platform === "win32" ? "win" : process.platform}-${process.arch}`));
const out = resolve(ROOT, String(flag("out", "dist-desktop")));
const smoke = flag("smoke", false) === true;
const skipBuild = flag("skip-build", false) === true;
const embeddings = flag("embeddings", false) === true;
const stage = join(out, "vein");

const SHERPA_PLATFORMS = ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64", "win-x64", "win-ia32"];
if (!SHERPA_PLATFORMS.includes(platform)) {
  console.error(`unknown platform ${platform}; one of ${SHERPA_PLATFORMS.join(", ")}`);
  process.exit(2);
}
// onnxruntime-node lays its binaries out as bin/napi-v6/<os>/<arch>.
const [ortOs, ortArch] = platform.replace(/^win-/, "win32-").split("-");

const log = (m) => console.log(`[package-desktop] ${m}`);
const run = (cmd, cmdArgs, cwd = ROOT) => {
  const r = spawnSync(cmd, cmdArgs, { cwd, stdio: "inherit", env: process.env });
  if (r.status !== 0) throw new Error(`${cmd} ${cmdArgs.join(" ")} failed (${r.status})`);
};
const sizeOf = async (p) => {
  const s = await stat(p);
  if (!s.isDirectory()) return s.size;
  let n = 0;
  for (const e of await readdir(p)) n += await sizeOf(join(p, e));
  return n;
};
const mb = (n) => `${Math.round(n / 1e6)} MB`;

async function build() {
  if (skipBuild) return log("skipping build (--skip-build)");
  log("building server (tsc) and web UI (vite)");
  run("npm", ["run", "build"]);
  run("npm", ["run", "build:web"]);
}

async function stageFiles() {
  await rm(stage, { recursive: true, force: true });
  await mkdir(stage, { recursive: true });
  const pkg = JSON.parse(await readFile(join(ROOT, "package.json"), "utf-8"));
  // The host runs `node build/server.js`; drop the dev surface from the manifest.
  delete pkg.devDependencies;
  delete pkg.scripts;
  pkg.private = true;
  await writeFile(join(stage, "package.json"), JSON.stringify(pkg, null, 2) + "\n");
  await cp(join(ROOT, "build"), join(stage, "build"), { recursive: true });
  await cp(join(ROOT, "web", "dist"), join(stage, "web", "dist"), { recursive: true });
  log(`staged build/ + web/dist/ → ${stage}`);
}

async function installDeps() {
  log("installing production dependencies (npm install --omit=dev)");
  // optionalDependencies (sherpa-onnx-node + its platform packages) come along;
  // the other platforms are removed below.
  run("npm", ["install", "--omit=dev", "--no-audit", "--no-fund", "--no-package-lock", "--ignore-scripts"], stage);
  const nm = join(stage, "node_modules");

  // One sherpa platform package.
  for (const p of SHERPA_PLATFORMS) {
    if (p === platform) continue;
    await rm(join(nm, `sherpa-onnx-${p}`), { recursive: true, force: true });
  }
  const keep = join(nm, `sherpa-onnx-${platform}`);
  try {
    await stat(keep);
  } catch {
    log(`sherpa-onnx-${platform} isn't installed on this machine's npm view; fetching it explicitly`);
    const ver = JSON.parse(await readFile(join(nm, "sherpa-onnx-node", "package.json"), "utf-8")).version;
    run("npm", ["install", "--no-save", "--no-audit", "--no-fund", "--no-package-lock", "--ignore-scripts", "--force", `sherpa-onnx-${platform}@${ver}`], stage);
  }

  // Only this platform's onnxruntime binaries (the package ships all of them).
  const ortBin = join(nm, "onnxruntime-node", "bin", "napi-v6");
  try {
    for (const os of await readdir(ortBin)) {
      if (os !== ortOs) {
        await rm(join(ortBin, os), { recursive: true, force: true });
        continue;
      }
      for (const arch of await readdir(join(ortBin, os))) {
        if (arch !== ortArch) await rm(join(ortBin, os, arch), { recursive: true, force: true });
      }
    }
  } catch {
    /* onnxruntime-node absent — nothing to prune */
  }
  if (!embeddings) {
    // npm removes the package and everything only it depended on, and drops
    // it from the staged package.json so the manifest matches the build.
    log("removing the embeddings stack (@huggingface/transformers and its deps); --embeddings keeps it");
    run("npm", ["uninstall", "--omit=dev", "--no-audit", "--no-fund", "--no-package-lock", "--ignore-scripts", "@huggingface/transformers"], stage);
  }
  await rm(join(nm, ".package-lock.json"), { force: true });
  await strip(nm);
}

// Sourcemaps, typings, and docs are dead weight in a shipped app. Licenses
// stay. Only node_modules is touched (vein's own build/ keeps its .d.ts).
const STRIP_EXT = [".map", ".d.ts", ".d.mts", ".d.cts", ".md", ".markdown"];
const STRIP_NAMES = new Set(["CHANGELOG", "CHANGES", "HISTORY", ".github", ".vscode", ".idea"]);
async function strip(dir) {
  let files = 0;
  let bytes = 0;
  const walk = async (d) => {
    for (const e of await readdir(d, { withFileTypes: true })) {
      const p = join(d, e.name);
      const base = e.name.replace(/\.(md|txt|markdown)$/i, "");
      if (STRIP_NAMES.has(base) || STRIP_NAMES.has(e.name)) {
        bytes += await sizeOf(p);
        files++;
        await rm(p, { recursive: true, force: true });
        continue;
      }
      if (e.isDirectory()) {
        await walk(p);
        continue;
      }
      if (/^licen[cs]e/i.test(e.name)) continue;
      if (STRIP_EXT.some((x) => e.name.endsWith(x))) {
        bytes += (await stat(p)).size;
        files++;
        await rm(p, { force: true });
      }
    }
  };
  await walk(dir);
  log(`stripped ${files} sourcemap/typing/doc files (${mb(bytes)}) from node_modules`);
}

async function report() {
  const nm = join(stage, "node_modules");
  const total = await sizeOf(stage);
  const rows = [];
  for (const e of await readdir(nm)) {
    if (e.startsWith(".")) continue;
    if (e.startsWith("@")) {
      for (const s of await readdir(join(nm, e))) rows.push([`${e}/${s}`, await sizeOf(join(nm, e, s))]);
    } else rows.push([e, await sizeOf(join(nm, e))]);
  }
  rows.sort((a, b) => b[1] - a[1]);
  log(`staged ${stage}: ${mb(total)} total, ${rows.length} packages; largest:`);
  for (const [name, size] of rows.slice(0, 8)) console.log(`    ${mb(size).padStart(7)}  ${name}`);
  const addons = [];
  const walk = async (d) => {
    for (const e of await readdir(d, { withFileTypes: true })) {
      const p = join(d, e.name);
      if (e.isDirectory()) await walk(p);
      else if (e.name.endsWith(".node")) addons.push(p.slice(nm.length + 1));
    }
  };
  await walk(nm);
  log(`native addons the host must code-sign (${addons.length}):`);
  for (const a of addons) console.log(`    ${a}`);
}

const freePort = () =>
  new Promise((res, rej) => {
    const s = createServer();
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address();
      s.close(() => res(port));
    });
    s.on("error", rej);
  });

async function smokeTest() {
  // Run from a temp copy so resolution can't fall back into this checkout.
  const tmp = await mkdtemp(join(tmpdir(), "vein-desktop-"));
  const app = join(tmp, "vein");
  const workspace = join(tmp, "Application Support", "workspace");
  const cache = join(tmp, "cache");
  log(`smoke: copying stage → ${app}`);
  await cp(stage, app, { recursive: true });
  await mkdir(join(workspace, "steps", "custom"), { recursive: true });
  await writeFile(
    join(workspace, "steps", "custom", "smoke-step.ts"),
    `import { z, defineStep } from "vein";
export default defineStep({
  type: "smoke-step",
  input: z.object({ name: z.string() }),
  output: z.string(),
  async run({ input }) { return "hi " + input.name; },
});
`,
  );
  const port = await freePort();
  const key = "smoke-key";
  const env = {
    ...process.env,
    VEIN_HOST: "127.0.0.1",
    VEIN_PORT: String(port),
    VEIN_WORKSPACE: workspace,
    VEIN_WORKSPACE_BACKEND: "fs",
    VEIN_API_KEY: key,
    VEIN_CACHE_DIR: cache,
  };
  log(`smoke: node build/server.js on 127.0.0.1:${port} (workspace + cache under ${tmp})`);
  const child = spawn(process.execPath, ["build/server.js"], { cwd: app, env, stdio: ["ignore", "pipe", "pipe"] });
  let output = "";
  child.stdout.on("data", (d) => (output += d));
  child.stderr.on("data", (d) => (output += d));
  const base = `http://127.0.0.1:${port}`;
  const headers = { authorization: `Bearer ${key}` };
  const deadline = Date.now() + 30_000;
  let up = false;
  while (Date.now() < deadline && !up) {
    if (child.exitCode !== null) break;
    try {
      up = (await fetch(`${base}/health`)).ok;
    } catch {
      await new Promise((r) => setTimeout(r, 250));
    }
  }
  const failures = [];
  try {
    if (!up) throw new Error(`server did not come up:\n${output}`);
    const health = await (await fetch(`${base}/health`)).json();
    if (!health.ok) failures.push("health not ok");
    if (health.dataDir !== workspace) failures.push(`dataDir ${health.dataDir} != ${workspace}`);

    const steps = await (await fetch(`${base}/steps`)).text();
    if (!steps.includes("smoke-step")) failures.push('custom step importing "vein" from an out-of-tree workspace did not register');

    const unauth = await fetch(`${base}/audio/models`);
    if (unauth.status !== 401) failures.push(`/audio/models without key: ${unauth.status}, expected 401`);
    const models = await (await fetch(`${base}/audio/models`, { headers })).json();
    if (!models.available) failures.push("sherpa addon did not load in the staged copy (available: false)");
    if (!models.modelDir.startsWith(cache)) failures.push(`modelDir ${models.modelDir} not under VEIN_CACHE_DIR`);

    const ui = await fetch(`${base}/`);
    if (!ui.ok || !(await ui.text()).includes("<div id=\"app\"")) failures.push("web UI index did not serve");
  } catch (e) {
    failures.push(String(e));
  } finally {
    child.kill("SIGTERM");
  }
  if (/Warning: failed to load step/.test(output)) failures.push(`step load warning:\n${output}`);
  await rm(tmp, { recursive: true, force: true });
  if (failures.length) {
    console.error(`[package-desktop] SMOKE FAILED\n - ${failures.join("\n - ")}`);
    process.exit(1);
  }
  log("smoke: OK — boots from an unrelated path, addon loads, out-of-tree custom step resolves, UI serves");
}

await build();
await stageFiles();
await installDeps();
await report();
if (smoke) await smokeTest();
log(`done. Host contract: plans/local-desktop-and-stt.md §2.5; run \`node build/server.js\` inside ${stage}`);
