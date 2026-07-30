#!/usr/bin/env node
/**
 * Vendor skill packs from upstream plugin repos into ./skills, which the
 * Dockerfile copies to $SKILLS_ROOT (/usr/src/skills — deliberately outside
 * /root, which sphinx-swarm shadows with a named volume).
 *
 * Upstream repos are Claude Code *plugins*: each top-level plugin holds a
 * skills/ dir plus a large CLAUDE.md that its skills assume is loaded. We
 * flatten `<plugin>/skills/<name>/` up one level and fold the plugin CLAUDE.md
 * into a PACK.md preamble, which the loader prepends the first time any skill
 * from that pack is loaded.
 *
 *   node scripts/vendor-skills.mjs           # re-vendor at the pinned SHAs
 *   node scripts/vendor-skills.mjs --check   # verify tree matches, don't write
 *
 * Re-run after bumping a SHA below and commit the result.
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "skills");
const CHECK = process.argv.includes("--check");

const SOURCES = [
  {
    repo: "https://github.com/anthropics/claude-for-legal.git",
    // Pinned so the vendored tree is reproducible. Bump deliberately.
    sha: "4a6c651889c97cc9140580363c73e0eb17379c2b",
    license: "Apache-2.0",
    attribution: "Anthropic — claude-for-legal",
    // Group label. A caller can enable every pack from this source with
    // `skills: { legal: true }` instead of naming each pack.
    tags: ["legal"],
  },
];

/**
 * Prepended to every PACK.md. The upstream skills are written for the Claude
 * Code plugin runtime; this states plainly which parts of that runtime are
 * absent here, so the model adapts instead of chasing files that don't exist.
 *
 * Kept short on purpose — the plugin CLAUDE.md that follows is already ~9k
 * tokens, and this has to be read before it to reframe what comes after.
 */
function adaptationHeader(packName) {
  return `## Running outside Claude Code

This pack was written for the Claude Code plugin runtime. You are not running in
it. The guidance below is otherwise accurate — adapt these points as you read:

- **Slash commands are unavailable.** Where a skill says to run
  \`/${packName}:some-command\`, there is no such command. Either perform that step's
  work directly, or load the correspondingly-named skill with
  \`load_skill("${packName}/some-command")\` and follow it.
- **Matter workspaces are off.** Treat \`## Matter workspaces\` as disabled
  (\`Enabled: ✗\`), which is the documented default. Work at the practice level and
  skip matter-switching, per-matter folders, and cross-matter rules entirely.
- **Plugin config paths do not exist.** Ignore instructions to read or write
  \`~/.claude/plugins/config/...\`. Write outputs to the working directory unless
  the user names a destination, and never assume a prior run's files are present.
- **This file is the practice-level CLAUDE.md.** When a skill refers to "the
  practice-level CLAUDE.md" or "this plugin's CLAUDE.md", it means the sections
  below. There is no separate file to open.
- **Customization is unsaved.** The profile sections below hold upstream defaults,
  not this user's real practice. Where a decision depends on firm-specific
  configuration that is clearly still a placeholder, ask rather than assume.
- **Plugin hooks, sub-agents, and MCP connectors are absent.** Anything relying on
  them needs to be done with the tools you actually have.

---

`;
}

function sh(cmd, args, cwd) {
  return execFileSync(cmd, args, { cwd, encoding: "utf-8", stdio: ["ignore", "pipe", "pipe"] });
}

function parseFrontmatterName(text) {
  const m = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text);
  return m ? m[1] : "";
}

/** Quote a description for YAML frontmatter as a folded block scalar. */
function yamlDescription(desc) {
  const wrapped = String(desc)
    .replace(/\s+/g, " ")
    .trim()
    .replace(/(.{1,88})(\s|$)/g, "$1\n")
    .trim()
    .split("\n")
    .map((l) => `  ${l.trim()}`)
    .join("\n");
  return `description: >\n${wrapped}`;
}

function copyDir(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const d of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, d.name);
    const t = path.join(dest, d.name);
    if (d.isDirectory()) copyDir(s, t);
    else if (d.isFile()) fs.copyFileSync(s, t);
  }
}

/** Every `<plugin>/skills` dir in the checkout, at any depth (external_plugins nests one deeper). */
function findPluginDirs(root) {
  const found = [];
  const walk = (dir, depth) => {
    if (depth > 2) return;
    for (const d of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!d.isDirectory() || d.name.startsWith(".")) continue;
      const sub = path.join(dir, d.name);
      if (fs.existsSync(path.join(sub, "skills"))) found.push({ name: d.name, dir: sub });
      else walk(sub, depth + 1);
    }
  };
  walk(root, 0);
  return found.sort((a, b) => a.name.localeCompare(b.name));
}

function vendor(source, tmp) {
  const checkout = path.join(tmp, path.basename(source.repo, ".git"));
  sh("git", ["clone", "--quiet", source.repo, checkout]);
  sh("git", ["checkout", "--quiet", source.sha], checkout);

  const plugins = findPluginDirs(checkout);
  let packCount = 0;
  let skillCount = 0;

  for (const plugin of plugins) {
    const skillsDir = path.join(plugin.dir, "skills");
    const skillDirs = fs
      .readdirSync(skillsDir, { withFileTypes: true })
      .filter((d) => d.isDirectory() && fs.existsSync(path.join(skillsDir, d.name, "SKILL.md")))
      .map((d) => d.name)
      .sort();
    if (skillDirs.length === 0) continue;

    const destPack = path.join(OUT, plugin.name);
    fs.rmSync(destPack, { recursive: true, force: true });
    fs.mkdirSync(destPack, { recursive: true });

    for (const name of skillDirs) {
      copyDir(path.join(skillsDir, name), path.join(destPack, name));
      skillCount++;
    }

    // PACK.md: frontmatter (drives the one-line prompt index) + adaptation
    // header + the plugin's own CLAUDE.md (what its skills actually cite).
    let description = "";
    const manifest = path.join(plugin.dir, ".claude-plugin", "plugin.json");
    if (fs.existsSync(manifest)) {
      try {
        description = JSON.parse(fs.readFileSync(manifest, "utf-8")).description || "";
      } catch {
        /* leave empty */
      }
    }
    const claudeMd = path.join(plugin.dir, "CLAUDE.md");
    const pluginContext = fs.existsSync(claudeMd) ? fs.readFileSync(claudeMd, "utf-8") : "";

    const packMd = [
      "---",
      `name: ${plugin.name}`,
      yamlDescription(description),
      ...(source.tags?.length ? [`tags: [${source.tags.join(", ")}]`] : []),
      `source: ${source.repo.replace(/\.git$/, "")}`,
      `source_ref: ${source.sha}`,
      `license: ${source.license}`,
      `attribution: ${source.attribution}`,
      "---",
      "",
      `# ${plugin.name}`,
      "",
      adaptationHeader(plugin.name),
      pluginContext.replace(/^#\s+.*\n/, "").trimStart(),
      "",
    ].join("\n");

    fs.writeFileSync(path.join(destPack, "PACK.md"), packMd);
    packCount++;
    console.log(`  ${plugin.name.padEnd(22)} ${String(skillDirs.length).padStart(3)} skills`);
  }
  return { packCount, skillCount };
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "vendor-skills-"));
try {
  if (CHECK) {
    console.log("--check: verifying ./skills matches the pinned sources");
  }
  const before = CHECK && fs.existsSync(OUT) ? sh("git", ["status", "--porcelain", "skills"], ROOT) : "";
  fs.mkdirSync(OUT, { recursive: true });

  let packs = 0;
  let skills = 0;
  for (const source of SOURCES) {
    console.log(`${source.repo} @ ${source.sha.slice(0, 7)}`);
    const r = vendor(source, tmp);
    packs += r.packCount;
    skills += r.skillCount;
  }
  console.log(`\n${packs} packs, ${skills} skills -> ${path.relative(ROOT, OUT)}/`);

  if (CHECK) {
    const after = sh("git", ["status", "--porcelain", "skills"], ROOT);
    if (after !== before) {
      console.error("\nERROR: ./skills is out of date. Run: node scripts/vendor-skills.mjs");
      process.exit(1);
    }
    console.log("./skills is up to date.");
  }
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
