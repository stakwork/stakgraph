/**
 * Concept collection + reflection.
 *
 * The invariants worth pinning down are the ones where a plausible-looking
 * implementation still loses data:
 *
 *   1. A Concept reached through `graph_get` is recorded, not just one reached
 *      through `learn_concept`. The two tools key on different identifiers, so
 *      the same concept read both ways must collapse to one entry.
 *   2. The deterministic read list survives a model turn that returns garbage,
 *      and a failed reflect never clobbers a ranking an earlier turn produced.
 */

import { test, expect } from "../../testkit.js";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

import type { ConceptCollector } from "../concepts.js";

// session.ts reads SESSIONS_DIR once at module load, so it must be set before
// anything that reaches session.ts is imported, and stay fixed for the
// lifetime of the process.
const tmpSessionsDir = path.join(os.tmpdir(), `test-concepts-${randomUUID()}`);
fs.mkdirSync(tmpSessionsDir, { recursive: true });
process.env.SESSIONS_DIR = tmpSessionsDir;
process.env.NO_DB = "true";

// Imported here, not at the top: concepts.ts reaches session.ts through
// utils.ts, so a static import would bind SESSIONS_DIR to the real .sessions
// directory before the override above ran — and these tests would write their
// fixtures into it. (Type-only imports are erased, so they're safe up top.)
const {
  conceptReadFrom,
  withConceptCollection,
  mergeConceptReads,
  conceptKey,
  parseReflection,
  buildReflectTurn,
  reflectEnabled,
  reflectPromptOverride,
  proposalTargetKey,
  applyReflectionProposals,
} = await import("../concepts.js");

/** A graph_get result for a Concept node, as the tool actually returns it. */
function graphGetConcept(overrides: Record<string, any> = {}) {
  return JSON.stringify({
    ref_id: "ref-abc",
    node_type: "Concept",
    name: "auth-session-lifecycle",
    properties: { id: "c_4f2a", repo: "stakwork/hive" },
    edges: { PARENT_OF: 2 },
    ...overrides,
  });
}

test.describe("recording concept reads", () => {
  test("records a Concept resolved through graph_get", () => {
    const read = conceptReadFrom("graph_get", { ref_id: "ref-abc" }, graphGetConcept());
    expect(read?.ref_id).toBe("ref-abc");
    expect(read?.id).toBe("c_4f2a");
    expect(read?.name).toBe("auth-session-lifecycle");
    expect(read?.via).toBe("graph_get");
  });

  test("ignores graph_get results for other node types", () => {
    const other = JSON.stringify({
      ref_id: "ref-xyz",
      node_type: "Function",
      name: "handleAuth",
      properties: {},
    });
    expect(conceptReadFrom("graph_get", { ref_id: "ref-xyz" }, other)).toBeNull();
  });

  test("records a concept read through learn_concept", () => {
    const doc = {
      id: "c_91bd",
      name: "stakwork-workflow-runs",
      description: "how runs are stored",
      documentation: "# Runs\n...",
    };
    const read = conceptReadFrom("learn_concept", { concept_id: "c_91bd" }, doc);
    expect(read?.id).toBe("c_91bd");
    expect(read?.via).toBe("learn_concept");
  });

  test("a concept miss is not a read", () => {
    const miss = { error: "Concept not found" };
    expect(conceptReadFrom("learn_concept", { concept_id: "nope" }, miss)).toBeNull();
  });

  test("wrapping collects reads and passes results through untouched", async () => {
    const collector: ConceptCollector = { reads: [] };
    const raw = graphGetConcept();
    const tools: Record<string, any> = {
      graph_get: { description: "resolve a node", execute: async () => raw },
      learn_concept: {
        description: "read a concept",
        execute: async () => ({ id: "c_91bd", name: "runs", documentation: "x" }),
      },
      graph_search: { description: "search", execute: async () => "[]" },
    };

    const wrapped = withConceptCollection(tools, collector);
    const passthrough = await wrapped.graph_get.execute({ ref_id: "ref-abc" }, {});
    await wrapped.learn_concept.execute({ concept_id: "c_91bd" }, {});
    await wrapped.graph_search.execute({ q: "auth" }, {});

    expect(passthrough).toBe(raw);
    expect(collector.reads).toHaveLength(2);
    expect(wrapped.graph_get.description).toBe("resolve a node");
    // graph_search surfaces concepts without their body — not a read.
    expect(collector.reads.map((r) => r.via)).toEqual(["graph_get", "learn_concept"]);
  });
});

test.describe("normalizing across the two id spaces", () => {
  const catalog = [
    { id: "c_4f2a", ref_id: "ref-abc", name: "auth-session-lifecycle", repo: "stakwork/hive" },
    { id: "c_91bd", ref_id: "ref-def", name: "stakwork-workflow-runs", repo: "stakwork/hive" },
  ];

  test("the same concept read both ways collapses to one entry", () => {
    const merged = mergeConceptReads(
      [
        { id: "c_4f2a", name: "auth-session-lifecycle", via: "learn_concept" },
        { ref_id: "ref-abc", name: "auth-session-lifecycle", via: "graph_get" },
      ],
      catalog,
    );
    expect(merged).toHaveLength(1);
    expect(merged[0].id).toBe("c_4f2a");
    expect(merged[0].ref_id).toBe("ref-abc");
    // Both paths are retained — how a concept gets reached is worth knowing.
    expect(merged[0].via).toBe("learn_concept,graph_get");
  });

  test("fills in the identifier the read path didn't carry", () => {
    const merged = mergeConceptReads([{ ref_id: "ref-def", via: "graph_get" }], catalog);
    expect(merged[0].id).toBe("c_91bd");
    expect(merged[0].name).toBe("stakwork-workflow-runs");
    expect(merged[0].repo).toBe("stakwork/hive");
  });

  test("keeps reads the catalog can't resolve", () => {
    const merged = mergeConceptReads([{ ref_id: "ref-gone", via: "graph_get" }], []);
    expect(merged).toHaveLength(1);
    expect(merged[0].ref_id).toBe("ref-gone");
  });

  test("a concept with no gitree id is still recorded", () => {
    // Concepts created directly in the graph never enter gitree's id space.
    const merged = mergeConceptReads([{ ref_id: "ref-only", via: "graph_get" }], [
      { ref_id: "ref-only", name: "graph-native-concept" },
    ]);
    expect(merged).toHaveLength(1);
    expect(merged[0].id).toBeUndefined();
    expect(conceptKey(merged[0])).toBe("ref-only");
  });

  test("ref_id keys ahead of id, so a partial record merges with a full one", () => {
    // Turn 1's catalog lookup failed, leaving ref_id only; turn 2's succeeded.
    // Keying on id first would file these as two separate concepts.
    const partial = mergeConceptReads([{ ref_id: "ref-abc", via: "graph_get" }], []);
    const full = mergeConceptReads([{ id: "c_4f2a", via: "learn_concept" }], catalog);
    expect(conceptKey(partial[0])).toBe(conceptKey(full[0]));
  });

  test("distinct concepts stay distinct", () => {
    const merged = mergeConceptReads(
      [
        { id: "c_4f2a", via: "learn_concept" },
        { ref_id: "ref-def", via: "graph_get" },
      ],
      catalog,
    );
    expect(merged).toHaveLength(2);
  });
});

test.describe("parsing the reflect turn", () => {
  const concepts = [
    { id: "c_4f2a", ref_id: "ref-abc", name: "auth", via: "learn_concept" },
    { id: "c_91bd", ref_id: "ref-def", name: "runs", via: "graph_get" },
  ];

  test("overlays rank, evidence and contradictions onto the read list", () => {
    const out = parseReflection(
      JSON.stringify({
        ranking: [
          { id: "c_4f2a", rank: 1, evidence: "used for the sidecar answer" },
          { id: "c_91bd", rank: 2, evidence: "opened on a wrong guess", contradicts: "says X; source says Y" },
        ],
      }),
      concepts,
    );
    expect(out.concepts[0].rank).toBe(1);
    expect(out.concepts[1].contradicts).toBe("says X; source says Y");
    expect(out.proposals).toEqual([]);
  });

  test("a concept the model skipped stays unranked rather than disappearing", () => {
    const out = parseReflection(
      JSON.stringify({ ranking: [{ id: "c_4f2a", rank: 1 }] }),
      concepts,
    );
    expect(out.concepts).toHaveLength(2);
    expect(out.concepts[1].rank).toBeNull();
  });

  test("parses proposals, dropping malformed entries", () => {
    const out = parseReflection(
      JSON.stringify({
        ranking: [],
        proposals: [
          {
            action: "update",
            concept_id: "c_4f2a",
            documentation: "# Auth\nrevised",
            rationale: "source contradicts the docs",
          },
          { action: "explode", concept_id: "c_4f2a" }, // invalid action
          "not an object",
          { action: "create", name: "session-reflection", documentation: "# New", rationale: "missing" },
        ],
      }),
      concepts,
    );
    expect(out.proposals).toHaveLength(2);
    expect(out.proposals[0].action).toBe("update");
    expect(out.proposals[0].concept_id).toBe("c_4f2a");
    expect(out.proposals[1].action).toBe("create");
    expect(out.proposals[1].name).toBe("session-reflection");
  });

  test("proposals survive a reply with no ranking", () => {
    const out = parseReflection(
      JSON.stringify({
        proposals: [{ action: "delete", concept_id: "c_91bd", rationale: "obsolete" }],
      }),
      concepts,
    );
    expect(out.proposals).toHaveLength(1);
    // The ranking half still degrades the same way it always did.
    expect(out.concepts.every((c) => c.rank === null)).toBe(true);
  });

  test("ids the model invented are dropped", () => {
    const out = parseReflection(
      JSON.stringify({ ranking: [{ id: "c_does_not_exist", rank: 1 }] }),
      concepts,
    );
    expect(out.concepts).toHaveLength(2);
    expect(out.concepts.every((c) => c.rank === null)).toBe(true);
  });

  test("non-JSON output is kept as raw, with the read list intact", () => {
    const out = parseReflection("The first concept was the useful one.", concepts);
    expect(out.concepts).toHaveLength(2);
    expect(out.raw).toBe("The first concept was the useful one.");
  });
});

test.describe("the reflect turn itself", () => {
  test("lists the concepts read, ids and names only", () => {
    const turn = buildReflectTurn([
      { id: "c_4f2a", ref_id: "ref-abc", name: "auth-session-lifecycle", via: "learn_concept" },
    ]);
    expect(turn.role).toBe("user");
    expect(turn.content).toContain("c_4f2a");
    expect(turn.content).toContain("auth-session-lifecycle");
  });

  test("a caller prompt replaces the instructions but keeps the concept list", () => {
    const turn = buildReflectTurn(
      [{ id: "c_4f2a", name: "auth", via: "learn_concept" }],
      "Which of these would you keep?",
    );
    expect(turn.content).toContain("Which of these would you keep?");
    expect(turn.content).toContain("c_4f2a");
    expect(turn.content).not.toContain("REFLECTION");
  });

  test("standing drafts are embedded with their full documentation", () => {
    const turn = buildReflectTurn(
      [{ id: "c_4f2a", name: "auth", via: "learn_concept" }],
      undefined,
      [
        {
          id: "prop-1",
          action: "update",
          status: "pending",
          conceptId: "c_4f2a",
          documentation: "# Auth\ndraft body from turn 1",
          rationale: "docs were stale",
          createdAt: new Date(),
        },
      ],
    );
    expect(turn.content).toContain("pending human review");
    expect(turn.content).toContain("draft body from turn 1");
    // No drafts -> no draft block at all (the prompt's own mention of
    // standing drafts remains, but no block header or content).
    const bare = buildReflectTurn([{ id: "c_4f2a", name: "auth", via: "learn_concept" }]);
    expect(bare.content).not.toContain("pending human review");
  });

  test("reflect config accepts true or an object", () => {
    expect(reflectEnabled(true)).toBe(true);
    expect(reflectEnabled({ prompt: "hi" })).toBe(true);
    expect(reflectEnabled(false)).toBe(false);
    expect(reflectEnabled(undefined)).toBe(false);
    expect(reflectPromptOverride({ prompt: "  hi  " })).toBe("hi");
    expect(reflectPromptOverride(true)).toBeUndefined();
    expect(reflectPromptOverride({ prompt: "   " })).toBeUndefined();
  });
});

test.describe("the reflection sidecar", () => {
  test("merges turns and never lets a failed reflect clobber a ranking", async () => {
    const { mergeReflection, loadReflection } = await import("../session.js");
    const sessionId = `sess-${randomUUID()}`;

    mergeReflection(sessionId, {
      concepts: [
        { id: "c_4f2a", name: "auth", rank: 1, evidence: "load-bearing" },
        { id: "c_91bd", name: "runs", rank: 2 },
      ],
    });

    // A later turn reads a third concept, but its reflect call fails — the
    // reads still land, carrying no ranks of their own.
    mergeReflection(sessionId, {
      concepts: [
        { id: "c_91bd", name: "runs", rank: null },
        { id: "c_02e1", name: "ontology", rank: null },
      ],
    });

    const saved = loadReflection(sessionId);
    expect(saved?.concepts).toHaveLength(3);
    const byId = Object.fromEntries((saved?.concepts ?? []).map((c) => [c.id, c]));
    expect(byId["c_4f2a"].rank).toBe(1);
    expect(byId["c_4f2a"].evidence).toBe("load-bearing");
    expect(byId["c_91bd"].rank).toBe(2);
    expect(byId["c_02e1"].rank).toBeNull();
    // Unranked concepts sort after ranked ones.
    expect(saved?.concepts[2].id).toBe("c_02e1");
  });

  test("a ref_id-only record and a fuller one are the same concept", async () => {
    const { mergeReflection, loadReflection } = await import("../session.js");
    const sessionId = `sess-${randomUUID()}`;

    mergeReflection(sessionId, {
      concepts: [{ ref_id: "ref-abc", name: "auth", rank: 1, evidence: "used it" }],
    });
    mergeReflection(sessionId, {
      concepts: [{ id: "c_4f2a", ref_id: "ref-abc", name: "auth", rank: null }],
    });

    const saved = loadReflection(sessionId);
    expect(saved?.concepts).toHaveLength(1);
    expect(saved?.concepts[0].id).toBe("c_4f2a");
    expect(saved?.concepts[0].rank).toBe(1);
    expect(saved?.concepts[0].evidence).toBe("used it");
  });

  test("a later ranking supersedes an earlier one", async () => {
    const { mergeReflection, loadReflection } = await import("../session.js");
    const sessionId = `sess-${randomUUID()}`;

    mergeReflection(sessionId, { concepts: [{ id: "c_4f2a", rank: 1, evidence: "first pass" }] });
    mergeReflection(sessionId, { concepts: [{ id: "c_4f2a", rank: 3, evidence: "second pass" }] });

    const saved = loadReflection(sessionId);
    expect(saved?.concepts[0].rank).toBe(3);
    expect(saved?.concepts[0].evidence).toBe("second pass");
  });

  test("read order is assigned with no model involved", async () => {
    const { mergeReflection, loadReflection } = await import("../session.js");
    const sessionId = `sess-${randomUUID()}`;

    // What a run without `reflect` produces: reads, no ranks.
    mergeReflection(sessionId, {
      concepts: [
        { ref_id: "ref-a", name: "first", rank: null },
        { ref_id: "ref-b", name: "second", rank: null },
      ],
    });

    const saved = loadReflection(sessionId);
    expect(saved?.concepts.map((c) => c.read_order)).toEqual([1, 2]);
    expect(saved?.concepts.every((c) => c.rank === null)).toBe(true);
  });

  test("later turns append without renumbering earlier reads", async () => {
    const { mergeReflection, loadReflection } = await import("../session.js");
    const sessionId = `sess-${randomUUID()}`;

    mergeReflection(sessionId, { concepts: [{ ref_id: "ref-a", rank: null }] });
    mergeReflection(sessionId, {
      concepts: [
        { ref_id: "ref-a", rank: null }, // re-read: keeps its position
        { ref_id: "ref-b", rank: null },
      ],
    });

    const saved = loadReflection(sessionId);
    const byRef = Object.fromEntries((saved?.concepts ?? []).map((c) => [c.ref_id, c]));
    expect(byRef["ref-a"].read_order).toBe(1);
    expect(byRef["ref-b"].read_order).toBe(2);
  });

  test("a reflect-off turn doesn't disturb ranks an earlier reflect produced", async () => {
    const { mergeReflection, loadReflection } = await import("../session.js");
    const sessionId = `sess-${randomUUID()}`;

    mergeReflection(sessionId, {
      concepts: [
        { ref_id: "ref-a", rank: 2, evidence: "some help" },
        { ref_id: "ref-b", rank: 1, evidence: "decisive" },
      ],
    });
    mergeReflection(sessionId, { concepts: [{ ref_id: "ref-c", rank: null }] });

    const saved = loadReflection(sessionId);
    // Judged concepts stay ordered by rank; the new read falls in behind them.
    expect(saved?.concepts.map((c) => c.ref_id)).toEqual(["ref-b", "ref-a", "ref-c"]);
    expect(saved?.concepts[0].rank).toBe(1);
    expect(saved?.concepts[2].rank).toBeNull();
    expect(saved?.concepts[2].read_order).toBe(3);
  });

  test("returns what it wrote, so the run result can carry it", async () => {
    // ContextResult.reflection (and the /progress terminal payload) is this
    // return value — a void merge would silently ship `undefined` to callers.
    const { mergeReflection } = await import("../session.js");
    const sessionId = `sess-${randomUUID()}`;
    const returned = mergeReflection(sessionId, {
      concepts: [{ ref_id: "ref-abc", name: "auth", rank: 1 }],
    });
    expect(returned.session_id).toBe(sessionId);
    expect(returned.concepts).toHaveLength(1);
    expect(returned.concepts[0].rank).toBe(1);
  });

  test("no sidecar for a session that read nothing", async () => {
    const { loadReflection } = await import("../session.js");
    expect(loadReflection(`sess-${randomUUID()}`)).toBeNull();
  });
});

// ─── Filing reflection proposals (create-or-revise per (session, target)) ────

function makeConcept(id: string, documentation: string) {
  const now = new Date();
  return {
    id,
    name: id,
    description: "",
    prNumbers: [],
    commitShas: [],
    createdAt: now,
    lastUpdated: now,
    documentation,
  };
}

/** In-memory mock of the GraphStorage surface the proposal service touches. */
function makeProposalStorage(seedConcepts: any[] = [], seedProposals: any[] = []) {
  const concepts: Record<string, any> = {};
  for (const c of seedConcepts) concepts[c.id] = c;
  const proposals: Record<string, any> = {};
  for (const p of seedProposals) proposals[p.id] = p;

  return {
    _proposals: proposals,
    initialize: async () => {},
    getConcept: async (id: string) => concepts[id] ?? null,
    saveProposal: async (p: any) => {
      proposals[p.id] = p;
    },
    getProposal: async (id: string) => proposals[id] ?? null,
    updateProposal: async (p: any) => {
      const existing = proposals[p.id];
      if (!existing || existing.status !== "pending") return false;
      proposals[p.id] = { ...p, status: existing.status };
      return true;
    },
    claimProposal: async (id: string, status: string, decidedBy?: string) => {
      const existing = proposals[id];
      if (!existing || existing.status !== "pending") return false;
      existing.status = status;
      existing.decidedBy = decidedBy;
      return true;
    },
  } as any;
}

test.describe("applying reflection proposals", () => {
  test("target keys: creates by name, everything else by concept", () => {
    expect(proposalTargetKey({ action: "create", name: "Session Reflection" })).toBe(
      proposalTargetKey({ action: "create", name: "session reflection" }),
    );
    expect(proposalTargetKey({ action: "update", conceptId: "c_1" })).toBe(
      proposalTargetKey({ action: "delete", conceptId: "c_1" }),
    );
    expect(proposalTargetKey({ action: "update", conceptId: "c_1" })).not.toBe(
      proposalTargetKey({ action: "update", conceptId: "c_2" }),
    );
  });

  test("files a fresh proposal stamped with the session and reflection source", async () => {
    const storage = makeProposalStorage([makeConcept("c_1", "old docs")]);
    const result = await applyReflectionProposals(storage, {
      sessionId: "sess-1",
      proposals: [
        { action: "update", concept_id: "c_1", documentation: "new docs", rationale: "stale" },
      ],
      drafts: [],
    });
    expect(result.filed).toHaveLength(1);
    expect(result.filed[0].source).toBe("reflection");
    expect(result.filed[0].sessionIds).toEqual(["sess-1"]);
    expect(result.filed[0].baseDocs).toBe("old docs");
  });

  test("revises the session's standing draft instead of filing a sibling", async () => {
    const draft = {
      id: "prop-1",
      action: "update",
      status: "pending",
      conceptId: "c_1",
      documentation: "turn-1 docs",
      baseDocs: "docs as of turn 1",
      sessionIds: ["sess-1"],
      source: "reflection",
      createdAt: new Date(),
    };
    // The concept drifted while the draft sat in the queue.
    const storage = makeProposalStorage([makeConcept("c_1", "current docs")], [draft]);

    const result = await applyReflectionProposals(storage, {
      sessionId: "sess-1",
      proposals: [
        { action: "update", concept_id: "c_1", documentation: "turn-2 docs", rationale: "requirements changed" },
      ],
      drafts: [draft as any],
    });

    expect(result.filed).toHaveLength(0);
    expect(result.revised).toHaveLength(1);
    expect(result.revised[0].id).toBe("prop-1");
    expect(result.revised[0].documentation).toBe("turn-2 docs");
    // baseDocs re-snapshots at revision time, so accept-time staleness checks
    // compare against what the reflection actually saw.
    expect(result.revised[0].baseDocs).toBe("current docs");
    expect(Object.keys(storage._proposals)).toHaveLength(1);
  });

  test("withdraw rejects the standing draft and files nothing", async () => {
    const draft = {
      id: "prop-1",
      action: "create",
      status: "pending",
      name: "obsolete-idea",
      documentation: "x",
      sessionIds: ["sess-1"],
      createdAt: new Date(),
    };
    const storage = makeProposalStorage([], [draft]);
    const result = await applyReflectionProposals(storage, {
      sessionId: "sess-1",
      proposals: [{ action: "create", name: "obsolete-idea", withdraw: true }],
      drafts: [draft as any],
    });
    expect(result.withdrawn).toEqual(["prop-1"]);
    expect(result.filed).toHaveLength(0);
    expect(storage._proposals["prop-1"].status).toBe("rejected");
  });

  test("a ref_id the model echoed back resolves to the gitree concept id", async () => {
    const storage = makeProposalStorage([makeConcept("c_1", "docs")]);
    const result = await applyReflectionProposals(storage, {
      sessionId: "sess-1",
      proposals: [
        { action: "update", concept_id: "ref-abc", documentation: "revised", rationale: "r" },
      ],
      drafts: [],
      known: [{ id: "c_1", ref_id: "ref-abc", via: "graph_get" }],
    });
    expect(result.filed).toHaveLength(1);
    expect(result.filed[0].conceptId).toBe("c_1");
  });

  test("one bad proposal never blocks the rest", async () => {
    const storage = makeProposalStorage([makeConcept("c_1", "docs")]);
    const result = await applyReflectionProposals(storage, {
      sessionId: "sess-1",
      proposals: [
        { action: "update", concept_id: "c_missing", documentation: "x", rationale: "r" },
        { action: "update", concept_id: "c_1", documentation: "y", rationale: "r" },
      ],
      drafts: [],
    });
    expect(result.filed).toHaveLength(1);
    expect(result.filed[0].conceptId).toBe("c_1");
  });
});
