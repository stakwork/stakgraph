import { generateText, ModelMessage, LanguageModel, type Tool } from "ai";
import { listConcepts } from "../gitree/service.js";
import { getProviderOptions } from "../aieo/src/index.js";
import { extractLeadingJsonObject, maxOutputTokensFor } from "./utils.js";
import type { ReflectedConcept } from "./session.js";

/**
 * Concept reflection.
 *
 * Two halves, deliberately kept separate so a bad model turn never costs us
 * the hard data:
 *
 *  1. Collection (deterministic) — which gitree Concepts the agent actually
 *     READ during a run. Recorded from tool results, not from the model.
 *  2. Reflection (opt-in, `reflect`) — one extra call appended to the finished
 *     transcript asking the agent to rank those concepts by how load-bearing
 *     they were. Runs immediately after the run ends, while the provider's
 *     prompt cache is still warm, so the transcript re-read is ~0.1x input
 *     price.
 */

/** Caller config for the `reflect` request field: `true` or an object. */
export type ReflectConfig = boolean | { prompt?: string };

export function reflectEnabled(reflect?: ReflectConfig): boolean {
  if (reflect === true) return true;
  if (reflect && typeof reflect === "object") return true;
  return false;
}

export function reflectPromptOverride(reflect?: ReflectConfig): string | undefined {
  if (reflect && typeof reflect === "object" && typeof reflect.prompt === "string") {
    const trimmed = reflect.prompt.trim();
    if (trimmed) return trimmed;
  }
  return undefined;
}

/** One read of a Concept's body during a run. */
export interface ConceptRead {
  /** gitree concept id (what `learn_concept` takes). */
  id?: string;
  /** Graph ref_id (what `graph_get` takes). */
  ref_id?: string;
  name?: string;
  repo?: string;
  /** Raw tool name the read came through, e.g. "learn_concept" | "graph_get". */
  via: string;
}

export interface ConceptCollector {
  reads: ConceptRead[];
}

/**
 * Tools that return a Concept's BODY. Deliberately excludes `graph_search` /
 * `graph_neighbors` / `list_concepts`: those surface a name and description
 * without the documentation, and `list_concepts` returns the whole catalog, so
 * appearing in one says nothing about what the agent chose to read.
 */
const CONCEPT_READ_TOOLS = ["learn_concept", "graph_get", "graph_get_batched"];

function parseMaybeJson(result: unknown): any {
  if (typeof result === "string") {
    try {
      return JSON.parse(result);
    } catch {
      return null;
    }
  }
  return result;
}

/**
 * Pull a Concept read out of one tool result, or null if it wasn't one.
 *
 * Keyed off the RESULT rather than the tool name wherever possible: the graph
 * tools stamp `node_type` on what they return, so a new tool that resolves
 * nodes is picked up by adding its name to CONCEPT_READ_TOOLS, with no new
 * shape-specific parsing.
 */
export function conceptReadFrom(
  toolName: string,
  input: any,
  result: unknown,
): ConceptRead | null {
  const parsed = parseMaybeJson(result);
  if (!parsed || typeof parsed !== "object") return null;
  if (parsed.error) return null;

  if (parsed.node_type === "Concept") {
    if (!parsed.ref_id) return null;
    return {
      ref_id: parsed.ref_id,
      id: parsed.properties?.id,
      name: parsed.name ?? parsed.properties?.name,
      repo: parsed.properties?.repo,
      via: toolName,
    };
  }

  // learn_concept resolves through gitree rather than the graph, so it returns
  // documentation with no node_type. Its input carries the id.
  if (toolName === "learn_concept") {
    const id = parsed.id ?? input?.concept_id;
    if (!id) return null;
    if (!parsed.documentation && !parsed.description) return null;
    return { id, name: parsed.name, via: toolName };
  }

  return null;
}

/**
 * Every Concept read in one tool result.
 *
 * `graph_get` resolves one node, so it yields at most one read. Batched tools
 * return `{ nodes: [...] }`, and each entry is scored on its own — without this
 * the batched form would record nothing at all, since the envelope carries no
 * `node_type` of its own.
 */
export function conceptReadsFrom(
  toolName: string,
  input: any,
  result: unknown,
): ConceptRead[] {
  const parsed = parseMaybeJson(result);
  if (parsed && typeof parsed === "object" && Array.isArray(parsed.nodes)) {
    return parsed.nodes
      .map((node: unknown) => conceptReadFrom(toolName, input, node))
      .filter((r: ConceptRead | null): r is ConceptRead => r !== null);
  }
  const one = conceptReadFrom(toolName, input, result);
  return one ? [one] : [];
}

/**
 * Wrap the concept-reading tools so every read is recorded as it happens.
 *
 * Returns a new tool map — the originals are not mutated. Collection is
 * best-effort: a throw inside the recorder must never fail the tool call the
 * agent is actually making.
 */
export function withConceptCollection(
  tools: Record<string, Tool<any, any>>,
  collector: ConceptCollector,
): Record<string, Tool<any, any>> {
  const wrapped: Record<string, Tool<any, any>> = { ...tools };
  for (const name of CONCEPT_READ_TOOLS) {
    const original = tools[name];
    if (!original || typeof (original as any).execute !== "function") continue;
    const execute = (original as any).execute.bind(original);
    wrapped[name] = {
      ...original,
      execute: async (input: any, options: any) => {
        const result = await execute(input, options);
        try {
          collector.reads.push(...conceptReadsFrom(name, input, result));
        } catch (e) {
          console.error(`[concepts] failed to record ${name} read:`, e);
        }
        return result;
      },
    } as Tool<any, any>;
  }
  return wrapped;
}

/**
 * Identity for merging, both within a run and across the turns of a session.
 *
 * `ref_id` leads because it is the identifier the node itself always carries:
 * a Concept created directly in the graph has no gitree `id` at all, and a
 * `learn_concept` read only gains a ref_id once the catalog resolves it. Under
 * an id-first key, a concept recorded as ref_id-only on one turn (catalog
 * lookup failed) and as id+ref_id on the next would land in the sidecar twice.
 */
export function conceptKey(c: { id?: string; ref_id?: string; name?: string }): string {
  return c.ref_id ?? c.id ?? c.name ?? "";
}

/** One concept as the catalog knows it, carrying both identifiers. */
export interface ConceptIdentity {
  id?: string;
  ref_id?: string;
  name?: string;
  repo?: string;
}

/**
 * Collapse raw reads into one entry per concept, resolving both identifiers
 * against a catalog.
 *
 * The two read paths key on different things — `learn_concept` on gitree's
 * `id`, `graph_get` on the graph `ref_id` — so the same concept reached both
 * ways arrives as two entries with no overlapping key. The catalog is what
 * lets them collapse; without it the ranking list shows the same concept
 * twice and every downstream count is inflated.
 *
 * Pure, so the merge is testable without a graph. `normalizeConceptReads`
 * supplies the real catalog.
 */
export function mergeConceptReads(
  reads: ConceptRead[],
  catalog: ConceptIdentity[],
): ConceptRead[] {
  const byId = new Map<string, ConceptIdentity>();
  const byRefId = new Map<string, ConceptIdentity>();
  for (const c of catalog) {
    if (c.id) byId.set(c.id, c);
    if (c.ref_id) byRefId.set(c.ref_id, c);
  }

  const merged = new Map<string, ConceptRead>();
  for (const read of reads) {
    const known =
      (read.id ? byId.get(read.id) : undefined) ??
      (read.ref_id ? byRefId.get(read.ref_id) : undefined);
    const resolved: ConceptRead = {
      id: known?.id ?? read.id,
      ref_id: known?.ref_id ?? read.ref_id,
      name: known?.name ?? read.name,
      repo: known?.repo ?? read.repo,
      via: read.via,
    };
    const key = conceptKey(resolved);
    if (!key) continue;
    const prev = merged.get(key);
    if (prev) {
      if (prev.via !== resolved.via) prev.via = `${prev.via},${resolved.via}`;
      continue;
    }
    merged.set(key, resolved);
  }
  return [...merged.values()];
}

/**
 * Resolve reads against the live concept catalog.
 *
 * Best-effort: if the catalog lookup fails, reads are still returned, deduped
 * on whichever key each one already has. A run's concept record is worth
 * keeping even when it can't be fully cross-referenced.
 */
export async function normalizeConceptReads(
  reads: ConceptRead[],
  repo?: string,
): Promise<ConceptRead[]> {
  if (reads.length === 0) return [];
  let catalog: ConceptIdentity[] = [];
  try {
    catalog = (await listConcepts(repo)).concepts;
  } catch (e) {
    console.error("[concepts] could not load concept list for normalization:", e);
  }
  return mergeConceptReads(reads, catalog);
}

const DEFAULT_REFLECT_PROMPT = `REFLECTION — a review turn on the work you just finished. Do not call any tools. Answer with JSON only.

1. Rank the concepts listed below from most to least load-bearing for the answer you gave, and for each give one line of evidence: where you used it and what it changed. A concept you read and never ended up using is a useful answer, not a failure — say so plainly.
2. Did anything you read in those concepts contradict what you found in the source? If so, quote both sides in \`contradicts\`. Omit the field when nothing did.
3. Was there anything you had to work out from the source that one of these concepts should have covered? Put it in \`gap\` as one sentence, or null.

Respond with JSON and nothing else:
{"ranking":[{"id":"<id>","rank":1,"evidence":"...","contradicts":"..."}],"gap":"..." or null}`;

/**
 * The identifier to show the model for one concept.
 *
 * Deliberately not the storage key: this is the id the agent actually passed
 * or saw during the run — `learn_concept` takes gitree's id, the graph tools
 * take ref_id — so the entry is recognizable to it. `parseReflection` accepts
 * either identifier back, so a mismatch costs nothing.
 */
function displayId(c: ConceptRead): string {
  if (c.via.includes("learn_concept")) return c.id ?? c.ref_id ?? "";
  return c.ref_id ?? c.id ?? "";
}

/**
 * Build the single user turn appended to the finished transcript.
 *
 * The concept list is always appended by us, including under a caller-supplied
 * prompt: the caller has no way of knowing which concepts a run read, so it
 * can't supply this part itself.
 */
export function buildReflectTurn(
  concepts: ConceptRead[],
  promptOverride?: string,
): ModelMessage {
  const list = concepts.map((c) => `  ${displayId(c)}  ${c.name ?? "(unnamed)"}`).join("\n");
  return {
    role: "user",
    content: `${promptOverride ?? DEFAULT_REFLECT_PROMPT}\n\nConcepts you read this session:\n${list}`,
  };
}

/**
 * Parse the reflect turn's output onto the concepts we know were read.
 *
 * Lenient by design. The ranking is only ever an overlay on the deterministic
 * read list: ids the model invented are dropped, ids it omitted stay with
 * `rank: null`, and text that isn't JSON at all (likely under a custom prompt)
 * is preserved as `raw` instead of being thrown away.
 */
export function parseReflection(
  text: string,
  concepts: ConceptRead[],
): { concepts: ReflectedConcept[]; gap: string | null; raw?: string } {
  const base: ReflectedConcept[] = concepts.map((c) => ({
    id: c.id,
    ref_id: c.ref_id,
    repo: c.repo,
    name: c.name,
    rank: null,
  }));

  const parsed = extractLeadingJsonObject(text);
  const ranking = Array.isArray(parsed?.ranking) ? parsed.ranking : null;
  if (!ranking) {
    return { concepts: base, gap: null, raw: text.trim() || undefined };
  }

  const byKey = new Map<string, ReflectedConcept>();
  for (const c of base) {
    if (c.id) byKey.set(c.id, c);
    if (c.ref_id) byKey.set(c.ref_id, c);
  }
  for (const entry of ranking) {
    const target = byKey.get(entry?.id) ?? byKey.get(entry?.ref_id);
    if (!target) continue;
    const rank = Number(entry?.rank);
    if (Number.isFinite(rank)) target.rank = rank;
    if (typeof entry?.evidence === "string") target.evidence = entry.evidence;
    if (typeof entry?.contradicts === "string" && entry.contradicts.trim()) {
      target.contradicts = entry.contradicts;
    }
  }

  const gap = typeof parsed?.gap === "string" && parsed.gap.trim() ? parsed.gap : null;
  return { concepts: base, gap };
}

export interface ReflectRunArgs {
  model: LanguageModel;
  modelId: string;
  provider: string;
  /** The run's system prompt, byte-identical. */
  system: string | undefined;
  /** The run's tool set, byte-identical. */
  tools: Record<string, Tool<any, any>>;
  /** The full model-facing transcript of the finished run. */
  messages: ModelMessage[];
  concepts: ConceptRead[];
  promptOverride?: string;
}

/**
 * Run the reflect call against the finished transcript.
 *
 * Everything before the appended turn is sent byte-identical to what the run
 * itself sent — same tools, same system prompt, same provider options — so the
 * provider serves it from cache. That is what makes replaying a long transcript
 * affordable, and it is fragile in a specific way: changing the tool set
 * invalidates the whole prefix, and changing `tool_choice` or the thinking
 * setting invalidates the messages half of it (which is nearly all of a long
 * agentic run). So the model is told not to call tools rather than being
 * prevented from doing so.
 */
export async function runReflection(args: ReflectRunArgs): Promise<{
  concepts: ReflectedConcept[];
  gap: string | null;
  raw?: string;
}> {
  const { model, modelId, provider, system, tools, messages, concepts } = args;
  const providerOptions = getProviderOptions(provider as any, undefined, modelId);
  const result = await generateText({
    model,
    ...(system ? { system } : {}),
    messages: [...messages, buildReflectTurn(concepts, args.promptOverride)],
    tools,
    providerOptions: providerOptions as any,
    maxOutputTokens: maxOutputTokensFor(provider),
  });
  return parseReflection(result.text ?? "", concepts);
}
