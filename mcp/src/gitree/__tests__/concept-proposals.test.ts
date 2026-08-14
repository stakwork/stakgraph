/**
 * Unit tests for applying ConceptProposals (accept-time semantics).
 *
 * Strategy: mock GraphStorage so no real Neo4j connection is needed, and call
 * the real applyProposal used by POST /gitree/proposals/:id/accept. Covers:
 *   - update: docs applied through saveDocumentation
 *   - update with description: goes through saveConcept (embedding refresh)
 *   - update stale base → 409 with code "stale_base", nothing written
 *   - update stale base + force → applied
 *   - delete: concept removed, stale base honored
 *   - merge: provenance unioned, surviving concept updated, absorbed deleted
 *   - merge stale base on the surviving concept → 409
 *   - create: delegates to createConceptDirect, returns createdConceptId
 *   - create when concept appeared while pending → 409
 *   - update when target vanished while pending → 404
 */
import { test, expect } from "../../testkit.js";
import { applyProposal } from "../proposals.js";
import { HttpError } from "../service.js";
import type { Concept, ConceptProposal } from "../types.js";

// ─── Helpers ────────────────────────────────────────────────────────────────

function makeConcept(
  overrides: Partial<Concept> & { id: string; name: string }
): Concept {
  const now = new Date();
  return {
    repo: undefined,
    description: "",
    prNumbers: [],
    commitShas: [],
    createdAt: now,
    lastUpdated: now,
    documentation: "",
    ...overrides,
  };
}

function makeProposal(
  overrides: Partial<ConceptProposal> & { action: ConceptProposal["action"] }
): ConceptProposal {
  return {
    id: "test-proposal-id",
    status: "pending",
    createdAt: new Date(),
    ...overrides,
  };
}

/** In-memory mock of the GraphStorage surface applyProposal touches. */
function makeStorage(seed: Concept[] = []) {
  const store: Record<string, Concept> = {};
  for (const concept of seed) store[concept.id] = concept;
  const savedDocs: Array<{ conceptId: string; documentation: string }> = [];
  const savedConcepts: Concept[] = [];

  const storage: any = {
    _store: store,
    _savedDocs: savedDocs,
    _savedConcepts: savedConcepts,
    initialize: async () => {},
    getConcept: async (id: string, repo?: string) => {
      const fullId = repo && !id.includes("/") ? `${repo}/${id}` : id;
      return store[fullId] ?? null;
    },
    saveConcept: async (concept: Concept) => {
      store[concept.id] = concept;
      savedConcepts.push(concept);
    },
    saveDocumentation: async (conceptId: string, documentation: string) => {
      savedDocs.push({ conceptId, documentation });
      if (store[conceptId]) store[conceptId].documentation = documentation;
    },
    deleteConcept: async (id: string, repo?: string) => {
      const fullId = repo && !id.includes("/") ? `${repo}/${id}` : id;
      delete store[fullId];
    },
    linkConceptParent: async () => {},
  };
  return storage;
}

async function expectHttpError(
  fn: () => Promise<any>,
  statusCode: number
): Promise<HttpError> {
  try {
    await fn();
  } catch (err: any) {
    expect(err instanceof HttpError).toBe(true);
    expect(err.statusCode).toBe(statusCode);
    return err;
  }
  throw new Error(`Expected HttpError ${statusCode}, but nothing was thrown`);
}

// ─── update ─────────────────────────────────────────────────────────────────

test.describe("applyProposal — update", () => {
  test("applies documentation through saveDocumentation when base matches", async () => {
    const storage = makeStorage([
      makeConcept({
        id: "owner/repo/auth",
        name: "Auth",
        repo: "owner/repo",
        documentation: "old docs",
      }),
    ]);
    const proposal = makeProposal({
      action: "update",
      conceptId: "owner/repo/auth",
      repo: "owner/repo",
      baseDocs: "old docs",
      documentation: "new docs",
    });

    const result = await applyProposal(storage, proposal, false);

    expect(result.createdConceptId).toBe(undefined);
    expect(storage._savedDocs).toEqual([
      { conceptId: "owner/repo/auth", documentation: "new docs" },
    ]);
    expect(storage._savedConcepts.length).toBe(0);
    expect(storage._store["owner/repo/auth"].documentation).toBe("new docs");
  });

  test("goes through saveConcept when the description also changes", async () => {
    const storage = makeStorage([
      makeConcept({
        id: "owner/repo/auth",
        name: "Auth",
        repo: "owner/repo",
        description: "old description",
        documentation: "old docs",
      }),
    ]);
    const proposal = makeProposal({
      action: "update",
      conceptId: "owner/repo/auth",
      repo: "owner/repo",
      baseDocs: "old docs",
      documentation: "new docs",
      description: "new description",
    });

    await applyProposal(storage, proposal, false);

    expect(storage._savedConcepts.length).toBe(1);
    expect(storage._savedConcepts[0].description).toBe("new description");
    expect(storage._savedConcepts[0].documentation).toBe("new docs");
    expect(storage._savedDocs.length).toBe(0);
  });

  test("rejects a stale base with 409 stale_base and writes nothing", async () => {
    const storage = makeStorage([
      makeConcept({
        id: "owner/repo/auth",
        name: "Auth",
        repo: "owner/repo",
        documentation: "docs changed by someone else",
      }),
    ]);
    const proposal = makeProposal({
      action: "update",
      conceptId: "owner/repo/auth",
      repo: "owner/repo",
      baseDocs: "old docs",
      documentation: "new docs",
    });

    const err = await expectHttpError(
      () => applyProposal(storage, proposal, false),
      409
    );
    expect(err.extra?.code).toBe("stale_base");
    expect(storage._savedDocs.length).toBe(0);
    expect(storage._store["owner/repo/auth"].documentation).toBe(
      "docs changed by someone else"
    );
  });

  test("force overrides a stale base", async () => {
    const storage = makeStorage([
      makeConcept({
        id: "owner/repo/auth",
        name: "Auth",
        repo: "owner/repo",
        documentation: "docs changed by someone else",
      }),
    ]);
    const proposal = makeProposal({
      action: "update",
      conceptId: "owner/repo/auth",
      repo: "owner/repo",
      baseDocs: "old docs",
      documentation: "new docs",
    });

    await applyProposal(storage, proposal, true);

    expect(storage._store["owner/repo/auth"].documentation).toBe("new docs");
  });

  test("404s when the target vanished while the proposal was pending", async () => {
    const storage = makeStorage([]);
    const proposal = makeProposal({
      action: "update",
      conceptId: "owner/repo/gone",
      repo: "owner/repo",
      baseDocs: "old docs",
      documentation: "new docs",
    });

    await expectHttpError(() => applyProposal(storage, proposal, false), 404);
  });
});

// ─── delete ─────────────────────────────────────────────────────────────────

test.describe("applyProposal — delete", () => {
  test("deletes the concept when base matches", async () => {
    const storage = makeStorage([
      makeConcept({
        id: "owner/repo/dead",
        name: "Dead",
        repo: "owner/repo",
        documentation: "docs",
      }),
    ]);
    const proposal = makeProposal({
      action: "delete",
      conceptId: "owner/repo/dead",
      repo: "owner/repo",
      baseDocs: "docs",
    });

    await applyProposal(storage, proposal, false);

    expect(storage._store["owner/repo/dead"]).toBe(undefined);
  });

  test("refuses to delete when docs drifted since propose time", async () => {
    const storage = makeStorage([
      makeConcept({
        id: "owner/repo/dead",
        name: "Dead",
        repo: "owner/repo",
        documentation: "docs got rewritten",
      }),
    ]);
    const proposal = makeProposal({
      action: "delete",
      conceptId: "owner/repo/dead",
      repo: "owner/repo",
      baseDocs: "docs",
    });

    await expectHttpError(() => applyProposal(storage, proposal, false), 409);
    expect(storage._store["owner/repo/dead"]).toBeDefined();
  });
});

// ─── merge ──────────────────────────────────────────────────────────────────

test.describe("applyProposal — merge", () => {
  test("unions provenance into the survivor and deletes the absorbed concept", async () => {
    const storage = makeStorage([
      makeConcept({
        id: "owner/repo/auth",
        name: "Auth",
        repo: "owner/repo",
        description: "surviving",
        documentation: "auth docs",
        prNumbers: [1, 2],
        commitShas: ["aaa"],
      }),
      makeConcept({
        id: "owner/repo/login",
        name: "Login",
        repo: "owner/repo",
        documentation: "login docs",
        prNumbers: [2, 3],
        commitShas: ["bbb"],
      }),
    ]);
    const proposal = makeProposal({
      action: "merge",
      conceptId: "owner/repo/login",
      mergeIntoConceptId: "owner/repo/auth",
      repo: "owner/repo",
      baseDocs: "auth docs",
      absorbedDocs: "login docs",
      documentation: "merged docs",
    });

    await applyProposal(storage, proposal, false);

    const survivor = storage._store["owner/repo/auth"];
    expect(survivor.documentation).toBe("merged docs");
    expect(survivor.prNumbers).toEqual([1, 2, 3]);
    expect(survivor.commitShas).toEqual(["aaa", "bbb"]);
    expect(survivor.description).toBe("surviving");
    expect(storage._store["owner/repo/login"]).toBe(undefined);
  });

  test("409s when the surviving concept's docs drifted", async () => {
    const storage = makeStorage([
      makeConcept({
        id: "owner/repo/auth",
        name: "Auth",
        repo: "owner/repo",
        documentation: "auth docs drifted",
      }),
      makeConcept({
        id: "owner/repo/login",
        name: "Login",
        repo: "owner/repo",
        documentation: "login docs",
      }),
    ]);
    const proposal = makeProposal({
      action: "merge",
      conceptId: "owner/repo/login",
      mergeIntoConceptId: "owner/repo/auth",
      repo: "owner/repo",
      baseDocs: "auth docs",
      documentation: "merged docs",
    });

    await expectHttpError(() => applyProposal(storage, proposal, false), 409);
    expect(storage._store["owner/repo/login"]).toBeDefined();
    expect(storage._store["owner/repo/auth"].documentation).toBe(
      "auth docs drifted"
    );
  });
});

// ─── create ─────────────────────────────────────────────────────────────────

test.describe("applyProposal — create", () => {
  test("creates the concept and reports its minted id", async () => {
    const storage = makeStorage([]);
    const proposal = makeProposal({
      action: "create",
      repo: "owner/repo",
      name: "Rate Limiting",
      description: "How rate limits work",
      documentation: "rate limit docs",
    });

    const result = await applyProposal(storage, proposal, false);

    expect(result.createdConceptId).toBe("owner/repo/rate-limiting");
    const created = storage._store["owner/repo/rate-limiting"];
    expect(created).toBeDefined();
    expect(created.name).toBe("Rate Limiting");
    expect(created.description).toBe("How rate limits work");
    expect(created.documentation).toBe("rate limit docs");
  });

  test("409s when the concept appeared while the proposal was pending", async () => {
    const storage = makeStorage([
      makeConcept({
        id: "owner/repo/rate-limiting",
        name: "Rate Limiting",
        repo: "owner/repo",
      }),
    ]);
    const proposal = makeProposal({
      action: "create",
      repo: "owner/repo",
      name: "Rate Limiting",
      documentation: "rate limit docs",
    });

    const err = await expectHttpError(
      () => applyProposal(storage, proposal, false),
      409
    );
    expect(err.extra?.conceptId).toBe("owner/repo/rate-limiting");
  });
});
