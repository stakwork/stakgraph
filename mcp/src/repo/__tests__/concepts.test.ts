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

import {
  conceptReadFrom,
  withConceptCollection,
  mergeConceptReads,
  conceptKey,
  parseReflection,
  buildReflectTurn,
  reflectEnabled,
  reflectPromptOverride,
  type ConceptCollector,
} from "../concepts.js";

// session.ts reads SESSIONS_DIR once at module load, so it must be set before
// the dynamic import below and stay fixed for the lifetime of the process.
const tmpSessionsDir = path.join(os.tmpdir(), `test-concepts-${randomUUID()}`);
fs.mkdirSync(tmpSessionsDir, { recursive: true });
process.env.SESSIONS_DIR = tmpSessionsDir;
process.env.NO_DB = "true";

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
        gap: "provenance sidecar format",
      }),
      concepts,
    );
    expect(out.concepts[0].rank).toBe(1);
    expect(out.concepts[1].contradicts).toBe("says X; source says Y");
    expect(out.gap).toBe("provenance sidecar format");
  });

  test("a concept the model skipped stays unranked rather than disappearing", () => {
    const out = parseReflection(
      JSON.stringify({ ranking: [{ id: "c_4f2a", rank: 1 }], gap: null }),
      concepts,
    );
    expect(out.concepts).toHaveLength(2);
    expect(out.concepts[1].rank).toBeNull();
    expect(out.gap).toBeNull();
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
      gap: "sidecar format",
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
    expect(saved?.gap).toBe("sidecar format");
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

  test("no sidecar for a session that read nothing", async () => {
    const { loadReflection } = await import("../session.js");
    expect(loadReflection(`sess-${randomUUID()}`)).toBeNull();
  });
});
