import { generateText, ModelMessage, LanguageModel, type Tool } from "ai";
import {
  listConcepts,
  listProposals,
  createProposal,
  reviseProposal,
  type CreateProposalInput,
} from "../gitree/service.js";
import { GraphStorage } from "../gitree/store/index.js";
import type { ConceptProposal, ConceptProposalAction } from "../gitree/types.js";
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
 *     they were, and to propose changes to the Concept knowledge base its work
 *     supports. Runs immediately after the run ends, while the provider's
 *     prompt cache is still warm, so the transcript re-read is ~0.1x input
 *     price. Proposals are filed by deterministic code (never a tool call —
 *     that would either break the cached prefix or trust the model with the
 *     upsert), keyed on (session, target) so a multi-turn session revises its
 *     one standing draft per topic instead of filing a sibling each turn.
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
2. Did anything you read in those concepts contradict what you found in the source? If so, quote both sides in \`contradicts\`. If you verified the source is right and the concept is wrong, that is grounds for an "update" proposal below.
3. \`proposals\` — optional, and empty on most runs. The Concept knowledge base exists to orient FUTURE sessions, so the test for filing one is: did this session uncover knowledge the next agent could NOT cheaply re-derive by reading the code? A pattern, best practice, or gotcha — a convention that spans files, a constraint the code doesn't make obvious, a mistake the code invites — is worth remembering. Feature-level and implementation-level detail is not: the next agent can always read the code itself for low-level specifics.

File a proposal only when this session produced that kind of evidence:
- You had to work out a pattern, convention, or gotcha from source that a concept should have covered → "create" a new concept (or "update" the nearest existing one).
- A concept's documentation contradicted the source and you verified it → "update" with the FULL revised documentation (it replaces the whole body).
- Two concepts turned out to cover the same ground → "merge". A concept describes something that no longer exists → "delete".

Do NOT propose: knowledge specific to this one question, feature-level details readable from the code, minor wording preferences, suspicions you didn't verify, or reorganizations of concepts you only skimmed. Every proposal needs a rationale citing what happened this session — a human reviews it before anything takes effect.

If a STANDING DRAFT proposal from this session is shown below, do NOT file a second proposal for the same target. Either return a revised full version of it (same target), leave it out of \`proposals\` to keep it unchanged, or return it with "withdraw": true if it no longer meets the bar above (the conversation moved on, or the learning turned out to be feature-specific after all).

Respond with JSON and nothing else:
{"ranking":[{"id":"<id>","rank":1,"evidence":"...","contradicts":"..."}],"proposals":[{"action":"create|update|delete|merge","concept_id":"<target concept id; omit for create>","merge_into_concept_id":"<merge only: surviving concept>","name":"<create only>","description":"...","documentation":"...","rationale":"...","withdraw":false}]}`;

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
 * A standing draft, rendered for the model in the same field names the
 * response schema uses, so revising is a copy-and-edit rather than a mapping
 * exercise. Full documentation included — a revision replaces the whole body,
 * so the model must see what it is revising.
 */
function renderDraft(p: ConceptProposal): string {
  return JSON.stringify({
    action: p.action,
    concept_id: p.conceptId,
    merge_into_concept_id: p.mergeIntoConceptId,
    name: p.name,
    description: p.description,
    documentation: p.documentation,
    rationale: p.rationale,
  });
}

/**
 * Build the single user turn appended to the finished transcript.
 *
 * The concept list and the session's standing draft proposals are always
 * appended by us, including under a caller-supplied prompt: the caller has no
 * way of knowing which concepts a run read or what it proposed on an earlier
 * turn, so it can't supply this part itself.
 */
export function buildReflectTurn(
  concepts: ConceptRead[],
  promptOverride?: string,
  drafts?: ConceptProposal[],
): ModelMessage {
  const list = concepts.map((c) => `  ${displayId(c)}  ${c.name ?? "(unnamed)"}`).join("\n");
  const draftBlock =
    drafts && drafts.length > 0
      ? `\n\nSTANDING DRAFT proposals from this session (pending human review):\n${drafts
          .map(renderDraft)
          .join("\n")}`
      : "";
  return {
    role: "user",
    content: `${promptOverride ?? DEFAULT_REFLECT_PROMPT}\n\nConcepts you read this session:\n${list}${draftBlock}`,
  };
}

/**
 * One proposed change parsed from the reflect turn, in the response schema's
 * own field names. Mapped onto CreateProposalInput at apply time.
 */
export interface ReflectionProposal {
  action: ConceptProposalAction;
  concept_id?: string;
  merge_into_concept_id?: string;
  name?: string;
  description?: string;
  documentation?: string;
  rationale?: string;
  /** True = withdraw this session's standing draft for the same target. */
  withdraw?: boolean;
}

const REFLECT_PROPOSAL_ACTIONS = new Set<ConceptProposalAction>([
  "create",
  "update",
  "delete",
  "merge",
]);

function cleanString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

/** Lenient parse of the `proposals` array; malformed entries are dropped. */
function parseReflectionProposals(parsed: any): ReflectionProposal[] {
  if (!Array.isArray(parsed?.proposals)) return [];
  const out: ReflectionProposal[] = [];
  for (const entry of parsed.proposals) {
    if (!entry || typeof entry !== "object") continue;
    if (!REFLECT_PROPOSAL_ACTIONS.has(entry.action)) continue;
    out.push({
      action: entry.action,
      concept_id: cleanString(entry.concept_id),
      merge_into_concept_id: cleanString(entry.merge_into_concept_id),
      name: cleanString(entry.name),
      description: cleanString(entry.description),
      documentation:
        typeof entry.documentation === "string" ? entry.documentation : undefined,
      rationale: cleanString(entry.rationale),
      withdraw: entry.withdraw === true,
    });
  }
  return out;
}

/**
 * Parse the reflect turn's output onto the concepts we know were read.
 *
 * Lenient by design. The ranking is only ever an overlay on the deterministic
 * read list: ids the model invented are dropped, ids it omitted stay with
 * `rank: null`, and text that isn't JSON at all (likely under a custom prompt)
 * is preserved as `raw` instead of being thrown away. Proposals are parsed
 * independently of the ranking, so a turn that answers only one half still
 * yields the other.
 */
export function parseReflection(
  text: string,
  concepts: ConceptRead[],
): { concepts: ReflectedConcept[]; proposals: ReflectionProposal[]; raw?: string } {
  const base: ReflectedConcept[] = concepts.map((c) => ({
    id: c.id,
    ref_id: c.ref_id,
    repo: c.repo,
    name: c.name,
    rank: null,
  }));

  const parsed = extractLeadingJsonObject(text);
  const proposals = parseReflectionProposals(parsed);
  const ranking = Array.isArray(parsed?.ranking) ? parsed.ranking : null;
  if (!ranking) {
    return { concepts: base, proposals, raw: text.trim() || undefined };
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

  return { concepts: base, proposals };
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
  /** This session's standing pending proposals, shown for revise-not-refile. */
  drafts?: ConceptProposal[];
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
  proposals: ReflectionProposal[];
  raw?: string;
}> {
  const { model, modelId, provider, system, tools, messages, concepts } = args;
  const providerOptions = getProviderOptions(provider as any, undefined, modelId);
  const result = await generateText({
    model,
    ...(system ? { system } : {}),
    messages: [
      ...messages,
      buildReflectTurn(concepts, args.promptOverride, args.drafts),
    ],
    tools,
    providerOptions: providerOptions as any,
    maxOutputTokens: maxOutputTokensFor(provider),
  });
  return parseReflection(result.text ?? "", concepts);
}

// ─── Reflection proposals (create-or-revise, one draft per target) ──────────

/**
 * The standing pending proposals attributable to a session — the drafts the
 * reflect turn shows the model, and the upsert targets at apply time. Only
 * reflection stamps sessionIds today, but the filter deliberately doesn't
 * check `source`: any future producer that stamps the session gets revised
 * rather than duplicated.
 */
export async function sessionPendingProposals(
  sessionId: string,
  repo?: string,
): Promise<ConceptProposal[]> {
  const all = await listProposals(repo, "pending");
  return all.filter((p) => p.sessionIds?.includes(sessionId));
}

/**
 * Upsert identity for a proposal within a session: the concept it targets, or
 * for creates the proposed name. This is what makes a session revise its one
 * draft per topic instead of filing a sibling each turn — and what keeps two
 * genuinely unrelated proposals from clobbering each other.
 */
export function proposalTargetKey(p: {
  action: string;
  conceptId?: string;
  name?: string;
}): string {
  if (p.action === "create") {
    return `create:${(p.name ?? "").trim().toLowerCase()}`;
  }
  return `concept:${p.conceptId ?? ""}`;
}

/**
 * The graph tools show the model ref_ids, but the proposal service resolves
 * targets by gitree concept id — so map a ref_id back through what this run
 * actually read before treating it as a target.
 */
function resolveConceptId(
  raw: string | undefined,
  known: ConceptRead[],
): string | undefined {
  if (!raw) return undefined;
  const match = known.find((c) => c.ref_id === raw || c.id === raw);
  return match?.id ?? raw;
}

export interface ApplyReflectionResult {
  filed: ConceptProposal[];
  revised: ConceptProposal[];
  withdrawn: string[];
}

/**
 * File the reflect turn's proposals: create-or-revise keyed on
 * (session, target), withdraw on request.
 *
 * The model half is already over by the time this runs — everything here is
 * deterministic code, which is the point: session-stamping, dedup against the
 * standing drafts, and the pending-only revision guard are enforced rather
 * than asked of the model. Per-proposal failures are logged and skipped so one
 * bad entry never blocks the rest, and the caller treats the whole thing as
 * best-effort.
 */
export async function applyReflectionProposals(
  storage: GraphStorage,
  args: {
    sessionId: string;
    repo?: string;
    proposals: ReflectionProposal[];
    drafts: ConceptProposal[];
    /** This run's concept reads, for ref_id -> gitree id translation. */
    known?: ConceptRead[];
  },
): Promise<ApplyReflectionResult> {
  const { sessionId, repo, proposals, drafts } = args;
  const known = args.known ?? [];
  const result: ApplyReflectionResult = { filed: [], revised: [], withdrawn: [] };

  const draftByTarget = new Map<string, ConceptProposal>();
  for (const d of drafts) draftByTarget.set(proposalTargetKey(d), d);

  for (const p of proposals) {
    try {
      const conceptId = resolveConceptId(p.concept_id, known);
      const targetKey = proposalTargetKey({
        action: p.action,
        conceptId,
        name: p.name,
      });
      const draft = draftByTarget.get(targetKey);

      if (p.withdraw) {
        if (draft) {
          await storage.claimProposal(
            draft.id,
            "rejected",
            "reflection",
            "Withdrawn by session reflection",
          );
          result.withdrawn.push(draft.id);
        }
        continue;
      }

      const input: CreateProposalInput = {
        action: p.action,
        repo,
        conceptId,
        mergeIntoConceptId: resolveConceptId(p.merge_into_concept_id, known),
        name: p.name,
        description: p.description,
        documentation: p.documentation,
        rationale: p.rationale,
        source: "reflection",
        sessionIds: [sessionId],
      };

      if (draft) {
        result.revised.push(await reviseProposal(storage, draft.id, input));
      } else {
        result.filed.push(await createProposal(storage, input));
      }
    } catch (e) {
      console.error(
        `[concepts] could not apply reflection proposal (${p.action} ${p.concept_id ?? p.name ?? ""}):`,
        e,
      );
    }
  }
  return result;
}

/** applyReflectionProposals against the live graph. */
export async function fileReflectionProposals(args: {
  sessionId: string;
  repo?: string;
  proposals: ReflectionProposal[];
  drafts: ConceptProposal[];
  known?: ConceptRead[];
}): Promise<ApplyReflectionResult> {
  const storage = new GraphStorage();
  await storage.initialize();
  return applyReflectionProposals(storage, args);
}
