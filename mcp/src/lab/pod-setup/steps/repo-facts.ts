import { z, defineStep } from "strut";
import { open, readdir, readFile } from "node:fs/promises";
import { basename, dirname, join, relative } from "node:path";

/**
 * pod-setup/repo-facts — what production's services agent pre-scanned
 * (mcp repo/services_utils.ts buildRepoFacts), as a seeded step: per
 * checked-out repository, its package files with the package manager their
 * lockfile names and their scripts, its compose files, its env example
 * files, and the env var names its code reads. Facts code can establish, so
 * the agents never have to guess them. Self-contained: a seeded step imports
 * only `strut` and Node builtins.
 *
 * KNOWN EXCEPTION to the env boundary (strut src/step-lint.ts): this step
 * reads files OUTSIDE the run's own directory — the checkouts `git/checkout`
 * made, under strut's data dir — with `node:fs`, and strut warns about the
 * import at boot ("reaches past ctx.services"). There is no honest
 * `ctx.services` equivalent yet: `artifacts` is the run's files, `shell`
 * would mean a `find` + `head` pipeline for a walk that reads thousands of
 * files. A later strut refuses the import, so this step needs a strut-side
 * door (a read-only files capability over a checkout) before then; until
 * it has one, the read stays and is listed in the PR that drew the line.
 */

/** Directories never walked: dependencies, build output, VCS and editor state. */
const SKIP_DIRS = new Set([
  "node_modules", "vendor", "dist", "build", "out", "target", "venv", "logs", "temp", "tmp",
  "__pycache__", "coverage",
]);
const PACKAGE_FILES: Record<string, string> = {
  "package.json": "node",
  "requirements.txt": "python",
  "pyproject.toml": "python",
  Pipfile: "python",
  "Cargo.toml": "rust",
  "go.mod": "go",
  Gemfile: "ruby",
  "pom.xml": "java",
  "build.gradle": "jvm",
  "build.gradle.kts": "jvm",
  "composer.json": "php",
  "mix.exs": "elixir",
  "CMakeLists.txt": "cpp",
  Podfile: "swift",
};
/** A lockfile names the package manager of the package file beside it (or above it, in a monorepo). */
const LOCKFILES: Record<string, string> = {
  "yarn.lock": "yarn",
  "pnpm-lock.yaml": "pnpm",
  "package-lock.json": "npm",
  "bun.lockb": "bun",
  "bun.lock": "bun",
  "uv.lock": "uv",
  "poetry.lock": "poetry",
  "Pipfile.lock": "pipenv",
  "Cargo.lock": "cargo",
  "Gemfile.lock": "bundler",
  "composer.lock": "composer",
};
const COMPOSE_FILE = /^(docker-)?compose(\.[\w-]+)?\.ya?ml$/;
const ENV_EXAMPLE = /^\.env[\w.-]*\.(example|sample|template|dist)$/;
/** Env var reads, by file extension. */
const ENV_READS: Array<[RegExp, RegExp[]]> = [
  [/\.(m|c)?[jt]sx?$/, [/process\.env\.([A-Za-z_]\w*)/g, /process\.env\[['"]([A-Za-z_]\w*)['"]\]/g, /import\.meta\.env\.([A-Za-z_]\w*)/g]],
  [/\.py$/, [/os\.environ\[['"]([A-Za-z_]\w*)['"]\]/g, /os\.environ\.get\(\s*['"]([A-Za-z_]\w*)['"]/g, /os\.getenv\(\s*['"]([A-Za-z_]\w*)['"]/g]],
  [/\.rb$/, [/ENV\[['"]([A-Za-z_]\w*)['"]\]/g, /ENV\.fetch\(\s*['"]([A-Za-z_]\w*)['"]/g]],
  [/\.go$/, [/os\.(?:Getenv|LookupEnv)\("([A-Za-z_]\w*)"\)/g]],
  [/\.rs$/, [/env::var\("([A-Za-z_]\w*)"\)/g]],
  [/\.(java|kt|kts)$/, [/System\.getenv\("([A-Za-z_]\w*)"\)/g]],
  [/\.swift$/, [/ProcessInfo\.processInfo\.environment\["([A-Za-z_]\w*)"\]/g]],
  [/\.(c|cc|cpp|h|hpp)$/, [/getenv\("([A-Za-z_]\w*)"\)/g]],
  [/\.php$/, [/getenv\(['"]([A-Za-z_]\w*)['"]\)/g, /\$_ENV\[['"]([A-Za-z_]\w*)['"]\]/g]],
];
const MAX_FILES = 20_000;
const MAX_READ = 512 * 1024;
const MAX_ENV_FILES_LISTED = 120;

export interface PackageFile {
  path: string;
  kind: string;
  /** The lockfile that decides its package manager, relative to the repo, or null. */
  lockfile: string | null;
  manager: string | null;
  /** package.json only: its scripts, each command cut at 200 characters. */
  scripts?: Record<string, string>;
}

export interface RepoFacts {
  name: string;
  path: string;
  url: string;
  packageFiles: PackageFile[];
  composeFiles: string[];
  envExamples: Array<{ file: string; names: string[] }>;
  envReads: Array<{ file: string; names: string[] }>;
  /** The walk stopped at MAX_FILES. */
  truncated: boolean;
}

async function head(file: string): Promise<string | null> {
  const fh = await open(file, "r");
  try {
    const buf = Buffer.alloc(MAX_READ);
    const { bytesRead } = await fh.read(buf, 0, MAX_READ, 0);
    const bytes = buf.subarray(0, bytesRead);
    if (bytes.subarray(0, 8000).includes(0)) return null; // binary
    return bytes.toString("utf8");
  } finally {
    await fh.close();
  }
}

/** Every file under `root` but the skipped directories and dot-directories. */
async function walk(root: string): Promise<{ files: string[]; truncated: boolean }> {
  const files: string[] = [];
  const queue = [root];
  while (queue.length) {
    const dir = queue.shift()!;
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const e of entries) {
      const full = join(dir, e.name);
      if (e.isDirectory()) {
        if (!e.name.startsWith(".") && !SKIP_DIRS.has(e.name)) queue.push(full);
      } else if (e.isFile()) {
        files.push(full);
        if (files.length >= MAX_FILES) return { files, truncated: true };
      }
    }
  }
  return { files, truncated: false };
}

export async function scanRepo(checkout: { path: string; repo?: string; url?: string }): Promise<RepoFacts> {
  const root = checkout.path;
  const { files, truncated } = await walk(root);
  const rel = (f: string) => relative(root, f) || ".";
  const present = new Set(files.map(rel));
  const lockFor = (pkgRel: string): { lockfile: string | null; manager: string | null } => {
    // Beside it, then each directory up to the repository's root (a monorepo's one lockfile).
    for (let dir = dirname(pkgRel); ; dir = dirname(dir)) {
      for (const [lock, manager] of Object.entries(LOCKFILES)) {
        const candidate = dir === "." ? lock : `${dir}/${lock}`;
        if (present.has(candidate)) return { lockfile: candidate, manager };
      }
      if (dir === "." || dir === "/" || dir === "") return { lockfile: null, manager: null };
    }
  };
  const packageFiles: PackageFile[] = [];
  const composeFiles: string[] = [];
  const envExamples: RepoFacts["envExamples"] = [];
  const envReads: RepoFacts["envReads"] = [];
  for (const file of files) {
    const name = basename(file);
    const r = rel(file);
    const kind = PACKAGE_FILES[name];
    if (kind) {
      const pkg: PackageFile = { path: r, kind, ...lockFor(r) };
      if (name === "package.json") {
        try {
          const scripts = JSON.parse(await readFile(file, "utf8"))?.scripts;
          if (scripts && typeof scripts === "object") {
            pkg.scripts = Object.fromEntries(
              Object.entries(scripts)
                .filter(([, v]) => typeof v === "string")
                .map(([k, v]) => [k, String(v).slice(0, 200)]),
            );
          }
        } catch {
          // an unparseable package.json is still a package file
        }
      }
      packageFiles.push(pkg);
      continue;
    }
    if (COMPOSE_FILE.test(name)) {
      composeFiles.push(r);
      continue;
    }
    if (ENV_EXAMPLE.test(name)) {
      const text = await head(file).catch(() => null);
      const names = text ? [...text.matchAll(/^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=/gm)].map((m) => m[1]!) : [];
      envExamples.push({ file: r, names: [...new Set(names)] });
      continue;
    }
    const reads = ENV_READS.find(([ext]) => ext.test(name))?.[1];
    if (!reads) continue;
    const text = await head(file).catch(() => null);
    if (!text) continue;
    const names = new Set<string>();
    for (const re of reads) for (const m of text.matchAll(re)) names.add(m[1]!);
    if (names.size) envReads.push({ file: r, names: [...names].sort() });
  }
  envReads.sort((a, b) => a.file.localeCompare(b.file));
  return {
    name: basename(root),
    path: root,
    url: checkout.url ?? "",
    packageFiles,
    composeFiles,
    envExamples,
    envReads,
    truncated,
  };
}

/** The facts as the agents read them. */
export function factsText(repos: RepoFacts[]): string {
  const out = ["REPO FACTS — scanned from the checkouts by code. Ground truth: do not re-discover these."];
  for (const r of repos) {
    out.push("", `## ${r.name} — /workspaces/${r.name}${r.url ? ` (${r.url})` : ""}`);
    out.push(r.packageFiles.length ? "Package files:" : "Package files: none found");
    for (const p of r.packageFiles) {
      const manager = p.manager ? `${p.manager}, from ${p.lockfile}` : "no lockfile";
      const scripts = p.scripts && Object.keys(p.scripts).length
        ? ` scripts: ${Object.entries(p.scripts).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")}`
        : "";
      out.push(`  ${p.path} (${p.kind}; ${manager})${scripts}`);
    }
    if (r.composeFiles.length) out.push(`Compose files in the repository: ${r.composeFiles.join(", ")}`);
    for (const e of r.envExamples) out.push(`Env example ${e.file}: ${e.names.join(", ") || "(no names)"}`);
    if (r.envReads.length) {
      out.push("Env vars the code reads:");
      for (const e of r.envReads.slice(0, MAX_ENV_FILES_LISTED)) out.push(`  ${e.file}: ${e.names.join(", ")}`);
      if (r.envReads.length > MAX_ENV_FILES_LISTED) out.push(`  … and ${r.envReads.length - MAX_ENV_FILES_LISTED} more files`);
    } else {
      out.push("Env vars the code reads: none found");
    }
    if (r.truncated) out.push(`(The scan stopped at ${MAX_FILES} files.)`);
  }
  return out.join("\n");
}

export default defineStep({
  type: "pod-setup/repo-facts",
  description:
    "Scan checked-out repositories for what a pod setup rests on, in code: per repository its package files (with the package manager their lockfile " +
    "names, and package.json's scripts), its compose files, its env example files and the env var names its code reads. Output: { repos: [{ name, " +
    "path, url, packageFiles, composeFiles, envExamples, envReads, truncated }], names (each repository's directory name — its /workspaces/<name> in " +
    "a pod), text (the facts as a prompt reads them) }.",
  input: z.object({
    checkouts: z
      .array(z.object({ path: z.string().min(1), repo: z.string().optional(), url: z.string().optional() }).passthrough())
      .min(1)
      .describe("git/checkout outputs, one per repository"),
  }),
  output: z.object({ repos: z.array(z.any()), names: z.array(z.string()), text: z.string() }),
  async run(cfg) {
    const repos = await Promise.all(cfg.checkouts.map((c) => scanRepo(c)));
    const names = repos.map((r) => r.name);
    const twice = names.filter((n, i) => names.indexOf(n) !== i);
    if (twice.length) {
      throw new Error(`pod-setup/repo-facts: two repositories share the directory name "${twice[0]}" — a pod mounts each at /workspaces/<name>, so they cannot both be in one workspace`);
    }
    return { repos, names, text: factsText(repos) };
  },
});
