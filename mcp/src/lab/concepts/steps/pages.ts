import { createHash } from "node:crypto";
import { z, defineStep, composeNodeKey } from "strut";
import type { GraphBackend, NodeSchema } from "strut";
import type { ConceptServices } from "../services.js";

/**
 * The two levels ABOVE a repository's feature Concepts (strut's
 * plans/concept-tree.md): one page per repository under a `Products` root,
 * each joined by PARENT_OF the way every strut Concept reader walks it.
 *
 *   Products                      one per workspace; what the team builds
 *   └── owner/repo                one per repository; what it is, how it is run
 *       └── <feature>             the Concepts the sweep already makes
 *
 * Runs at the end of every sweep (process-repo-chronological), for the
 * repository just swept. A page is a FUNCTION OF ITS INPUTS and is rewritten
 * only when they changed: the hash of the inputs rides in its
 * `unique_source_id` stamp (`lab/concepts/<name>@<inputHash>.<textHash>`),
 * so a run whose inputs match the stamp writes nothing — never a threshold,
 * a timer or a model's judgement. The repository page depends on the doc
 * files at HEAD (README, AGENTS.md, CLAUDE.md — `docFiles`) and its
 * features' names and descriptions (not their docs, which the sweep
 * rewrites every run); the root on its children's name, description and
 * docs. The second hash in the stamp is of the page's own text as written:
 * a page a person edited in the graph (`edited`) or wrote under one of
 * these names (`foreign`) is left alone and reported, never overwritten.
 * The first sweep after this lands finds no stamp, which is the migration.
 *
 * The pages carry no `repo` property: every reader of features filters on
 * it (`getAllConcepts(repo)`, hive's list, `collect-for-eval`), so the two
 * new levels are invisible to all of them. Nodes and edges go through
 * strut's graph seams on `ctx.services.graph` (the node writer by (type,
 * name), the edge writer), as strut's own seeded pages do, so every reader
 * sees them exactly as it sees a seeded page. The prompts are the calling
 * workflow's `params` (the experiment surface), never baked in here.
 *
 * Output: { repoPage, products, reparented, unparented?, usage }, each page
 * one of written | unchanged | edited | foreign | empty | skipped.
 */

export const STAMP_PREFIX = "lab/concepts/";
export const PRODUCTS = "Products";
const TYPE = "Concept";
const EDGE = "PARENT_OF";
const DEFAULT_DOC_FILES = ["README.md", "AGENTS.md", "CLAUDE.md"];
/** The root's stamp before it has been written from its children: never a hash, so the first run with any child rewrites it. */
const PENDING = "000000000000";
/** What the root says until its children give it something to say. */
export const PRODUCTS_PLACEHOLDER: Page = { description: "What this workspace builds.", docs: "" };

export interface DocFile {
  name: string;
  body: string;
}
export interface FeatureLine {
  name: string;
  description: string;
}
export interface ChildPage extends FeatureLine {
  docs: string;
}
export interface Page {
  description: string;
  docs: string;
}
export interface Usage {
  input: number;
  cache_read: number;
  cache_write: number;
  output: number;
  total: number;
}
export interface ExistingPage {
  ref_id: string;
  deleted: boolean;
  unique_source_id?: unknown;
  description?: unknown;
  docs?: unknown;
}
export type PageAction = "create" | "unchanged" | "edited" | "foreign" | "rewrite";
export type PageReport = "written" | "unchanged" | "edited" | "foreign" | "empty" | "skipped";

const str = (v: unknown): string => (typeof v === "string" ? v : "");
const hash = (s: string): string => createHash("sha256").update(s, "utf8").digest("hex").slice(0, 12);
const byName = <T extends { name: string }>(a: T, b: T) => a.name.localeCompare(b.name);

/** What a repository page is made from: the doc files and the features' name + description, each sorted. */
export function repoInputHash(docs: DocFile[], features: FeatureLine[]): string {
  const d = [...docs].sort(byName).map((f) => `${f.name}\0${f.body}`).join("\n");
  const f = [...features].sort(byName).map((x) => `${x.name}\t${x.description ?? ""}`).join("\n");
  return hash(`${d}\n\n${f}`);
}

/** What the root is made from: its children's name, description and docs, sorted. */
export function rootInputHash(children: ChildPage[]): string {
  return hash([...children].sort(byName).map((c) => `${c.name}\t${c.description ?? ""}\t${c.docs ?? ""}`).join("\n"));
}

/** The page's own text as written — how a later edit in the graph is told apart. */
export function textHash(page: Page): string {
  return hash(`${page.description}\n${page.docs}`);
}

export function stampOf(name: string, inputHash: string, text: string): string {
  return `${STAMP_PREFIX}${name}@${inputHash}.${text}`;
}

const STAMP_RE = /^lab\/concepts\/(.+)@([0-9a-f]{12})\.([0-9a-f]{12})$/;

export function parseStamp(usid: unknown): { name: string; inputHash: string; textHash: string } | null {
  if (typeof usid !== "string") return null;
  const m = STAMP_RE.exec(usid);
  return m ? { name: m[1]!, inputHash: m[2]!, textHash: m[3]! } : null;
}

/** The decision table: no page → create; not our stamp → foreign; text changed since we wrote it → edited; same inputs → unchanged; else rewrite. */
export function planPage(existing: ExistingPage | null, inputHash: string): PageAction {
  if (!existing || existing.deleted) return "create";
  const stamp = parseStamp(existing.unique_source_id);
  if (!stamp) return "foreign";
  if (textHash({ description: str(existing.description), docs: str(existing.docs) }) !== stamp.textHash) return "edited";
  return stamp.inputHash === inputHash ? "unchanged" : "rewrite";
}

/** The model's answer: a `Description:` line, then the page. Without the line, the first non-empty line stands in. */
export function parsePage(text: string): Page {
  let body = text.replace(/\r\n/g, "\n").trim();
  const fenced = /^```[a-z]*\n([\s\S]*?)\n```$/.exec(body);
  if (fenced) body = fenced[1]!.trim();
  const lines = body.split("\n");
  let description = "";
  let from = 0;
  const m = /^\s*(?:#+\s*)?\**\s*description\s*\**\s*:\s*\**\s*(.+)$/i.exec(lines[0] ?? "");
  if (m) {
    description = m[1]!.replace(/\*+$/, "").trim();
    from = 1;
  }
  const docs = lines.slice(from).join("\n").trim();
  if (!description) description = (docs.split("\n").find((l) => l.trim()) ?? "").replace(/^#+\s*/, "").trim().slice(0, 200);
  return { description, docs };
}

export function sumUsage(list: Array<Record<string, unknown> | undefined>): Usage {
  const out: Usage = { input: 0, cache_read: 0, cache_write: 0, output: 0, total: 0 };
  for (const u of list) {
    if (!u) continue;
    for (const k of Object.keys(out) as Array<keyof Usage>) {
      const v = u[k];
      if (typeof v === "number" && Number.isFinite(v)) out[k] += v;
    }
  }
  return out;
}

export function repoPrompt(repoName: string, docs: DocFile[], features: FeatureLine[]): string {
  const parts = [`Repository: ${repoName}`, "", "## The repository's own docs"];
  if (docs.length) for (const d of [...docs].sort(byName)) parts.push("", `### ${d.name}`, "", d.body);
  else parts.push("", "(none of the doc files were found at the repository root)");
  parts.push("", "## Its features, as the knowledge graph has them", "");
  parts.push(features.length ? [...features].sort(byName).map((f) => `- ${f.name} — ${f.description}`).join("\n") : "(none yet)");
  return parts.join("\n");
}

export function productsPrompt(children: ChildPage[]): string {
  return ["The repositories, one page each:", ...[...children].sort(byName).flatMap((c) => ["", `## ${c.name}`, "", c.description, "", c.docs])].join("\n");
}

// ── the graph ──────────────────────────────────────────────────────────────

async function findPage(graph: GraphBackend, schema: NodeSchema, name: string): Promise<ExistingPage | null> {
  const key = composeNodeKey(schema, { name });
  return graph.tx(async (tx) => {
    const ref = await graph.nodeEngine.refByKey(tx, schema.type, key, graph.namespace);
    const raw = ref ? await graph.nodeEngine.readRaw(tx, ref) : null;
    if (!raw) return null;
    const p = raw.props;
    return {
      ref_id: raw.ref_id,
      deleted: p["deleted_at"] != null || p["is_deleted"] === true,
      unique_source_id: p["unique_source_id"],
      description: p["description"],
      docs: p["docs"],
    };
  });
}

async function writePage(
  graph: GraphBackend,
  name: string,
  page: Page,
  inputHash: string,
  extra: Record<string, unknown>,
  existing: boolean,
): Promise<string> {
  const data = { name, description: page.description, docs: page.docs, unique_source_id: stampOf(name, inputHash, textHash(page)), ...extra };
  const r = await graph.nodes.write({ type: TYPE, data }, existing ? "upsert" : "create");
  return r.ref_id;
}

/** The live children of a page along PARENT_OF: name, description, docs. */
async function childrenOf(graph: GraphBackend, ref: string): Promise<ChildPage[]> {
  return graph.tx(async (tx) => {
    const live = await graph.edgeEngine.liveEdgesOf(tx, ref, EDGE, true);
    const out: ChildPage[] = [];
    for (const e of live?.edges ?? []) {
      const raw = await graph.nodeEngine.readRaw(tx, e.other);
      if (!raw) continue;
      const p = raw.props;
      if (p["deleted_at"] != null || p["is_deleted"] === true) continue;
      out.push({ name: str(p["name"]), description: str(p["description"]), docs: str(p["docs"]) });
    }
    return out;
  });
}

type Github = ReturnType<ConceptServices["github"]>;

/** The doc files at HEAD through the contents API: a clone exists only for a brand-new repository. */
async function readDocFiles(github: Github, owner: string, repo: string, files: string[], maxChars: number): Promise<DocFile[]> {
  const out: DocFile[] = [];
  for (const path of files) {
    try {
      const res = await github.rest.repos.getContent({ owner, repo, path });
      const d = res.data as { type?: string; content?: string; encoding?: string };
      if (Array.isArray(res.data) || d.type !== "file" || typeof d.content !== "string") continue;
      const body = Buffer.from(d.content, d.encoding === "base64" ? "base64" : "utf8").toString("utf8");
      out.push({ name: path, body: body.length > maxChars ? `${body.slice(0, maxChars)}\n\n[cut at ${maxChars} chars]` : body });
    } catch (err) {
      if ((err as { status?: number }).status === 404) continue;
      throw err;
    }
  }
  return out;
}

export default defineStep({
  type: "concepts/pages",
  description:
    "The pages above a repository's feature Concepts: its own page (docs distilled from README / AGENTS.md / CLAUDE.md at HEAD) under a `Products` root (what the team builds), joined by PARENT_OF, every feature parented under the repository. Each page is rewritten only when the hash of what it is made from differs from the stamp on it; a page someone edited is left and reported. Required config: repoPageSystem, productsSystem — the calling workflow's `params`. Output: { repoPage, products, reparented, unparented?, usage }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    /** The files at the repository root a page is distilled from, in order. */
    docFiles: z.array(z.string()).default(DEFAULT_DOC_FILES),
    /** Each file is cut here before it reaches the model. */
    maxDocChars: z.number().int().positive().default(40_000),
    repoPageSystem: z.string(),
    productsSystem: z.string(),
    /** The root page's name. `Products` everywhere but a test over a shared graph. */
    root: z.string().default(PRODUCTS),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const services = ctx.services as ConceptServices;
    const skipped = (reason: string) => ({ repoPage: "skipped" as PageReport, products: "skipped" as PageReport, reason, reparented: 0, usage: sumUsage([]) });
    const graph = services.graph;
    if (!graph) return skipped("no graph on the services bag");
    const schema = await graph.schemas.schema(TYPE);
    if (!schema) return skipped(`no "${TYPE}" schema in the graph`);

    const repoName = `${cfg.owner}/${cfg.repo}`;
    const token = await services.secrets?.get("GITHUB_TOKEN");
    const docs = await readDocFiles(services.github(token), cfg.owner, cfg.repo, cfg.docFiles, cfg.maxDocChars);
    const features = (await services.storage.getAllConcepts(repoName)).map((c) => ({ name: c.name, description: c.description ?? "", ref_id: c.ref_id }));
    const usages: Array<Record<string, unknown> | undefined> = [];

    // 1. The repository's page.
    const inputHash = repoInputHash(docs, features);
    const existing = await findPage(graph, schema, repoName);
    const action = planPage(existing, inputHash);
    let repoRef = existing && !existing.deleted ? existing.ref_id : undefined;
    let repoPage: PageReport = action === "create" || action === "rewrite" ? "empty" : action;
    if ((action === "create" || action === "rewrite") && (docs.length || features.length)) {
      const { text, usage } = await services.summarizer.generatePage({ system: cfg.repoPageSystem, prompt: repoPrompt(repoName, docs, features) });
      usages.push(usage as unknown as Record<string, unknown>);
      repoRef = await writePage(graph, repoName, parsePage(text), inputHash, { id: repoName }, !!existing);
      repoPage = "written";
    }

    // 2. Every feature of the repository under its page — the existing ones too.
    let reparented = 0;
    const unparented: string[] = [];
    if (repoRef) {
      for (const f of features) {
        if (!f.ref_id) {
          unparented.push(f.name);
          continue;
        }
        try {
          const e = await graph.edges.write({ edge: EDGE, source_ref_id: repoRef, target_ref_id: f.ref_id });
          if (e.created) reparented++;
        } catch (err) {
          unparented.push(f.name);
          console.warn(`[concepts/pages] could not parent "${f.name}" under ${repoName}:`, err instanceof Error ? err.message : err);
        }
      }
    }

    // 3. The root, over every repository page under it.
    let root = await findPage(graph, schema, cfg.root);
    if (!root || root.deleted) {
      await writePage(graph, cfg.root, PRODUCTS_PLACEHOLDER, PENDING, {}, !!root);
      root = await findPage(graph, schema, cfg.root);
    }
    let products: PageReport = "skipped";
    if (root) {
      if (repoRef) await graph.edges.write({ edge: EDGE, source_ref_id: root.ref_id, target_ref_id: repoRef });
      const children = await childrenOf(graph, root.ref_id);
      const rootHash = rootInputHash(children);
      const rootAction = planPage(root, rootHash);
      products = rootAction === "create" || rootAction === "rewrite" ? "empty" : rootAction;
      if ((rootAction === "create" || rootAction === "rewrite") && children.length) {
        const { text, usage } = await services.summarizer.generatePage({ system: cfg.productsSystem, prompt: productsPrompt(children) });
        usages.push(usage as unknown as Record<string, unknown>);
        await writePage(graph, cfg.root, parsePage(text), rootHash, {}, true);
        products = "written";
      }
    }

    return { repoPage, products, reparented, ...(unparented.length ? { unparented } : {}), usage: sumUsage(usages) };
  },
});
