import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import yaml from "js-yaml";

/**
 * Inline skills: compiled into the binary and concatenated into the system
 * prompt eagerly when enabled. This tier is for short *style* guidance that the
 * model must follow without being prompted to go fetch it (a formatting guide
 * it never thinks to load is a formatting guide that does nothing).
 *
 * Procedural skills belong on disk under SKILLS_ROOT instead, where they're
 * surfaced as a one-line index and pulled on demand via `load_skill`.
 */
export const SKILLS: Record<string, string> = {};

export const mermaid = `
## Mermaid Diagram Style Guide

When generating mermaid diagrams for workspaces, follow these rules:

### Structure
- Use \`graph TD\` (top-down) for system/architecture diagrams
- Group related nodes with \`subgraph Name["Label"]\`
- Use cylinder syntax \`[("label")]\` for databases/stores
- Use descriptive edge labels: \`-->|"verb phrase"|\`
- Add a \`%% ── SECTION NAME ──\` comment before each subgraph block

### Color Classes
Always end the diagram with classDef definitions and class assignments.
Use this palette (dark-mode optimized, muted fills, bright borders):

\`\`\`mermaid
classDef client    fill:#1e3a5f,stroke:#5b9cf6,color:#c7e2ff
classDef gateway   fill:#431c0d,stroke:#fb923c,color:#ffe4cc
classDef service   fill:#2e1a4a,stroke:#a78bfa,color:#ede9fe
classDef data      fill:#2a1040,stroke:#c084fc,color:#f3e8ff
classDef external  fill:#3b1030,stroke:#f472b6,color:#fce7f3
classDef observe   fill:#0d3328,stroke:#34d399,color:#d1fae5
\`\`\`

Assign every node to a class. No unstyled nodes.

### Grouping Rules
- Adapt the number of subgraphs to the actual content — could be 1 node with no subgraphs, could be 30 nodes across 6 layers. Don't force structure that isn't there.
- Only create a subgraph when there are 2+ related nodes that benefit from grouping.
- A workspace with a single repo is just a single node. That's fine. Don't pad it.
- As complexity grows, common layers emerge naturally:
  - Clients (apps, web, admin)
  - Gateway / Ingress (LB, API gateway, auth)
  - Services (core business logic)
  - Data (DBs, caches, queues, object stores)
  - External (3rd party APIs, email, payments)
  - Observability (logging, metrics, tracing)
- Use these as inspiration, not a checklist. If a workspace is just 3 backend services and a database, that's 4 nodes and maybe 1 subgraph or none. Let the content dictate the shape.

### Edge Labels
- Keep labels short: 2-3 words (\`read/write\`, \`publish event\`, \`HTTPS\`)
- Use \`-->\` for sync, \`-.->\` for async/optional if relevant

### Syntax Rules
- Every \`subgraph\` MUST have a matching \`end\` keyword on its own line
- Every edge must have exactly one source and one target: \`A -->|"label"| B\`. Never chain multiple labels or arrows in a single edge.
- Every node referenced in an edge or \`class\` assignment must be defined (e.g. \`NODE_ID["Label"]\`) inside a subgraph first
- Do not nest subgraphs — close each subgraph with \`end\` before opening the next

### Avoid
- Don't use \`style\` on individual nodes — only \`classDef\` + \`class\`
- Don't exceed ~30 nodes (split into multiple diagrams instead)
- Don't leave subgraphs with only 1 node
`

SKILLS["mermaid"] = mermaid;

// ---------------------------------------------------------------------------
// On-disk skills (progressive disclosure)
// ---------------------------------------------------------------------------
//
// Layout under SKILLS_ROOT, two levels deep at most:
//
//   <root>/frontend-design/SKILL.md          -> a skill
//   <root>/commercial-legal/PACK.md          -> a pack's shared preamble
//   <root>/commercial-legal/nda-review/SKILL.md
//
// A directory holding a SKILL.md is a skill; a directory holding skill
// directories is a pack. Nothing special-cases any particular pack name.
//
// Only `name: description` reaches the system prompt — a pack contributes one
// line regardless of how many skills it holds. Bodies are pulled through
// `load_skill`, which is worth ~40x on a large library: the bundled legal packs
// are ~13k tokens of descriptions against ~525k tokens of bodies.

/**
 * Where skills live. Read per call rather than bound at module load so tests
 * (and a dev volume mount) can point it elsewhere after import.
 */
export function skillsRoot(): string {
  return process.env.SKILLS_ROOT || path.join(os.homedir(), ".agents", "skills");
}

export interface SkillEntry {
  kind: "skill";
  /** Bare directory name, e.g. "nda-review" */
  name: string;
  /** How `load_skill` addresses it: "nda-review" or "commercial-legal/nda-review" */
  qualified: string;
  description: string;
  file: string;
  pack?: string;
}

export interface PackEntry {
  kind: "pack";
  name: string;
  description: string;
  dir: string;
  /** PACK.md — shared context every skill in the pack assumes is loaded. */
  preambleFile?: string;
  /**
   * Group labels from PACK.md frontmatter (`tags: [legal]`). Enabling a tag in
   * the `skills` map turns on every pack carrying it, so a caller can say
   * `{ legal: true }` instead of naming all 13 legal packs — and a 14th pack
   * joins the group by shipping the tag, with no code change here.
   */
  tags: string[];
  skills: SkillEntry[];
}

export type IndexEntry = SkillEntry | PackEntry;

/** Split YAML frontmatter off a markdown file. Malformed frontmatter is ignored. */
function parseFrontmatter(text: string): {
  data: Record<string, any>;
  body: string;
} {
  const m = /^---\r?\n([\s\S]*?)\r?\n---[ \t]*\r?\n?/.exec(text);
  if (!m) return { data: {}, body: text };
  let data: Record<string, any> = {};
  try {
    const parsed = yaml.load(m[1]);
    if (parsed && typeof parsed === "object") data = parsed as Record<string, any>;
  } catch {
    // A skill with unparseable frontmatter still has a usable body; fall back
    // to deriving its description from the directory name.
  }
  return { data, body: text.slice(m[0].length) };
}

/** Collapse a description to a single line so the prompt index stays one-line-per-entry. */
function oneLine(s: string): string {
  return String(s).replace(/\s+/g, " ").trim();
}

function readSkillDir(dir: string, pack?: string): SkillEntry | null {
  const file = path.join(dir, "SKILL.md");
  if (!fs.existsSync(file)) return null;
  let data: Record<string, any> = {};
  try {
    data = parseFrontmatter(fs.readFileSync(file, "utf-8")).data;
  } catch {
    return null;
  }
  const name = typeof data.name === "string" && data.name ? data.name : path.basename(dir);
  return {
    kind: "skill",
    name,
    qualified: pack ? `${pack}/${name}` : name,
    description: oneLine(data.description || ""),
    file,
    pack,
  };
}

function toTags(raw: unknown): string[] {
  if (Array.isArray(raw)) return raw.filter((t) => typeof t === "string" && t).map((t) => t.trim());
  if (typeof raw === "string") return raw.split(",").map((t) => t.trim()).filter(Boolean);
  return [];
}

/** Pack metadata: PACK.md frontmatter, else the Claude Code plugin manifest, else empty. */
function readPackMeta(dir: string): {
  description: string;
  preambleFile?: string;
  tags: string[];
} {
  const packFile = path.join(dir, "PACK.md");
  if (fs.existsSync(packFile)) {
    try {
      const { data } = parseFrontmatter(fs.readFileSync(packFile, "utf-8"));
      return {
        description: data.description ? oneLine(data.description) : "",
        preambleFile: packFile,
        tags: toTags(data.tags),
      };
    } catch {
      /* fall through */
    }
  }
  const manifest = path.join(dir, ".claude-plugin", "plugin.json");
  if (fs.existsSync(manifest)) {
    try {
      const j = JSON.parse(fs.readFileSync(manifest, "utf-8"));
      if (j && typeof j.description === "string") {
        return { description: oneLine(j.description), tags: toTags(j.tags) };
      }
    } catch {
      /* fall through */
    }
  }
  return { description: "", tags: [] };
}

const scanCache = new Map<string, IndexEntry[]>();

/**
 * Enumerate everything installed under `root`. Skills are baked into the image,
 * so the result is cached per-process; set SKILLS_NO_CACHE when mounting the
 * skills directory as a volume for authoring.
 */
export function scanSkills(root: string = skillsRoot()): IndexEntry[] {
  const useCache = !process.env.SKILLS_NO_CACHE;
  if (useCache) {
    const hit = scanCache.get(root);
    if (hit) return hit;
  }

  const entries: IndexEntry[] = [];
  let dirents: fs.Dirent[] = [];
  try {
    dirents = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    // No skills installed — an empty index is a valid state, not an error.
    if (useCache) scanCache.set(root, entries);
    return entries;
  }

  for (const d of dirents.sort((a, b) => a.name.localeCompare(b.name))) {
    if (!d.isDirectory() || d.name.startsWith(".")) continue;
    const dir = path.join(root, d.name);

    const asSkill = readSkillDir(dir);
    if (asSkill) {
      entries.push(asSkill);
      continue;
    }

    let children: fs.Dirent[] = [];
    try {
      children = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    const skills: SkillEntry[] = [];
    for (const c of children.sort((a, b) => a.name.localeCompare(b.name))) {
      if (!c.isDirectory() || c.name.startsWith(".")) continue;
      const s = readSkillDir(path.join(dir, c.name), d.name);
      if (s) skills.push(s);
    }
    if (skills.length === 0) continue;

    const { description, preambleFile, tags } = readPackMeta(dir);
    entries.push({ kind: "pack", name: d.name, description, dir, preambleFile, tags, skills });
  }

  if (useCache) scanCache.set(root, entries);
  return entries;
}

/** Test/dev hook — drop the memoized scan so a re-scan picks up edits. */
export function clearSkillsCache(): void {
  scanCache.clear();
}

/** Every skill on disk, flattened, pack-qualified. */
export function allDiskSkills(root: string = skillsRoot()): SkillEntry[] {
  return scanSkills(root).flatMap((e) => (e.kind === "skill" ? [e] : e.skills));
}

/**
 * The entries a caller switched on via the `skills` request field. A key is one
 * of, resolved in this order:
 *
 *   "commercial-legal/nda-review"  one skill out of a pack
 *   "commercial-legal"             an entry name (a pack, or a bare skill)
 *   "legal"                        a tag — every pack carrying it
 *
 * An entry name always wins over a tag of the same spelling, so shipping a pack
 * named after an existing tag narrows the selection rather than silently
 * widening it.
 */
export function enabledEntries(
  skills: Record<string, boolean | undefined> | undefined,
  root: string = skillsRoot(),
): IndexEntry[] {
  const on = Object.entries(skills || {})
    .filter(([, v]) => v)
    .map(([k]) => k);
  if (on.length === 0) return [];

  const installed = scanSkills(root);
  const out: IndexEntry[] = [];
  const seen = new Set<string>();

  for (const name of on) {
    if (name.includes("/")) {
      const [packName, skillName] = name.split("/");
      const pack = installed.find((e) => e.kind === "pack" && e.name === packName) as
        | PackEntry
        | undefined;
      const skill = pack?.skills.find((s) => s.name === skillName);
      if (skill && !seen.has(skill.qualified)) {
        seen.add(skill.qualified);
        out.push(skill);
      }
      continue;
    }
    const entry = installed.find((e) => e.name === name);
    if (entry) {
      if (!seen.has(entry.name)) {
        seen.add(entry.name);
        out.push(entry);
      }
      continue;
    }
    // Not an entry name — try it as a tag. Keeps scan order so the index is
    // stable regardless of which key the caller used to reach a pack.
    for (const tagged of installed) {
      if (tagged.kind !== "pack" || !tagged.tags.includes(name)) continue;
      if (seen.has(tagged.name)) continue;
      seen.add(tagged.name);
      out.push(tagged);
    }
  }
  return out;
}

/**
 * The block appended to the system prompt: one line per entry. Packs collapse
 * to a single line with a skill count, so enabling all 13 legal packs costs
 * ~600 tokens instead of the ~13k a flat listing of their 151 skills would.
 */
export function renderSkillIndex(entries: IndexEntry[]): string {
  if (entries.length === 0) return "";
  const lines = entries.map((e) => {
    const desc = e.description || "(no description)";
    return e.kind === "pack"
      ? `  - ${e.name} [pack, ${e.skills.length} skill${e.skills.length === 1 ? "" : "s"}]: ${desc}`
      : `  - ${e.qualified}: ${desc}`;
  });
  return `\n\nSKILLS:
Guidance available to you on demand. Only the names and descriptions are shown below — call \`load_skill\` to read one in full BEFORE doing work it covers.
${lines.join("\n")}
- \`load_skill(name)\` returns a skill's full instructions. Use the name exactly as listed (pack skills are addressed as "pack/skill").
- \`list_skills(pack)\` expands a pack into its individual skills. Called with no argument it lists everything installed, including packs not shown above.
Load a skill when the task plausibly falls in its area. Don't guess at a skill's contents from its description alone.`;
}

/** One pack's skills, for `list_skills(pack)`. */
export function listPack(packName: string, root: string = skillsRoot()): SkillEntry[] | null {
  const pack = scanSkills(root).find((e) => e.kind === "pack" && e.name === packName) as
    | PackEntry
    | undefined;
  return pack ? pack.skills : null;
}

export interface LoadedSkill {
  name: string;
  body: string;
  /** Pack preamble, returned only the first time a pack is loaded in a request. */
  preamble?: string;
  packName?: string;
}

/**
 * Resolve "pack/skill", "skill", or an inline skill name to its full text.
 * A bare name that is ambiguous across packs resolves to nothing — the caller
 * gets the list of qualified candidates instead of an arbitrary pick, which
 * matters because names like `customize` repeat across every legal pack.
 */
export function loadSkill(
  name: string,
  root: string = skillsRoot(),
): LoadedSkill | { error: string; candidates?: string[] } {
  const wanted = name.trim().replace(/^\/+|\/+$/g, "");
  if (!wanted) return { error: "No skill name provided" };

  if (SKILLS[wanted]) return { name: wanted, body: SKILLS[wanted] };

  const installed = scanSkills(root);
  const flat = installed.flatMap((e) => (e.kind === "skill" ? [e] : e.skills));

  let match = flat.find((s) => s.qualified === wanted);
  if (!match) {
    const byName = flat.filter((s) => s.name === wanted);
    if (byName.length === 1) match = byName[0];
    else if (byName.length > 1) {
      return {
        error: `"${wanted}" is ambiguous — it exists in ${byName.length} packs. Load it with its pack prefix.`,
        candidates: byName.map((s) => s.qualified),
      };
    }
  }
  if (!match) {
    return {
      error: `Skill "${wanted}" not found. Call list_skills to see what is available.`,
    };
  }

  let body: string;
  try {
    body = fs.readFileSync(match.file, "utf-8");
  } catch (e: any) {
    return { error: `Could not read skill "${match.qualified}": ${e?.message || e}` };
  }

  let preamble: string | undefined;
  if (match.pack) {
    const pack = installed.find(
      (e) => e.kind === "pack" && e.name === match!.pack,
    ) as PackEntry | undefined;
    if (pack?.preambleFile) {
      try {
        // Frontmatter only feeds the index; the model just needs the body.
        preamble = parseFrontmatter(fs.readFileSync(pack.preambleFile, "utf-8")).body.trim();
      } catch {
        // A missing preamble degrades the skill but doesn't break it.
      }
    }
  }

  return { name: match.qualified, body, preamble, packName: match.pack };
}