import { GraphStorage } from "./store/index.js";
import { Concept } from "./types.js";
import { generateSlug, makeRepoId } from "./store/utils.js";

/**
 * Error carrying an HTTP status (and optional extra response fields) so
 * service-layer logic can be shared across route handlers without each one
 * re-implementing the status mapping.
 */
export class HttpError extends Error {
  constructor(
    public statusCode: number,
    message: string,
    public extra?: Record<string, unknown>
  ) {
    super(message);
  }
}

export interface CreateConceptDirectInput {
  name: string;
  documentation: string;
  description?: string;
  repo?: string;
  parent?: string;
}

/**
 * Create a concept from caller-supplied content (no repo analysis). Shared by
 * POST /gitree/create-concept-direct and accepted "create" proposals so both
 * paths mint ids, validate parents, and roll back identically.
 */
export async function createConceptDirect(
  storage: GraphStorage,
  input: CreateConceptDirectInput
): Promise<{ concept: Concept; parentId?: string }> {
  const name = typeof input.name === "string" ? input.name.trim() : "";
  if (!name) {
    throw new HttpError(400, "name is required");
  }
  if (typeof input.documentation !== "string") {
    throw new HttpError(400, "documentation is required and must be a string");
  }

  const slug = generateSlug(name);
  if (!slug) {
    throw new HttpError(400, "name must contain alphanumeric characters");
  }
  const repoId =
    typeof input.repo === "string" && input.repo.trim()
      ? input.repo.trim()
      : undefined;
  const conceptId = repoId ? makeRepoId(repoId, slug) : slug;
  const parentId =
    typeof input.parent === "string" && input.parent.trim()
      ? input.parent.trim()
      : undefined;

  const existing = await storage.getConcept(conceptId, repoId);
  if (existing) {
    throw new HttpError(409, `Concept ${conceptId} already exists`, {
      conceptId,
    });
  }

  // Validate parent before any persistence — a bad parent must never leave
  // a partially-created concept behind.
  let parentConcept = null;
  if (parentId) {
    parentConcept = await storage.getConcept(parentId, repoId);
    if (!parentConcept) {
      throw new HttpError(400, `Parent concept ${parentId} not found`);
    }
    if (parentConcept.id === conceptId) {
      throw new HttpError(400, "A concept cannot be its own parent");
    }
  }

  const now = new Date();
  const concept: Concept = {
    id: conceptId,
    repo: repoId,
    name,
    description: typeof input.description === "string" ? input.description : "",
    prNumbers: [],
    commitShas: [],
    createdAt: now,
    lastUpdated: now,
    documentation: input.documentation,
  };

  await storage.saveConcept(concept);
  await storage.saveDocumentation(conceptId, input.documentation);

  // Link to parent if provided — compensating rollback on failure.
  if (parentId && parentConcept) {
    try {
      await storage.linkConceptParent(parentConcept.id, conceptId);
      console.log(
        `🔗 Linked concept ${conceptId} under parent ${parentConcept.id}`
      );
    } catch (linkErr: any) {
      try {
        await storage.deleteConcept(conceptId, repoId);
      } catch (rollbackErr: any) {
        console.error(
          `Rollback failed for orphaned concept ${conceptId}:`,
          rollbackErr
        );
      }
      throw new HttpError(
        500,
        `Concept created but parent link failed; rolled back: ${linkErr.message}`
      );
    }
  }

  return { concept, parentId: parentConcept?.id };
}

export interface ConceptSummary {
  id: string;
  repo?: string;
  ref_id?: string;
  name: string;
  description: string;
  prCount: number;
  commitCount: number;
  lastUpdated: string;
  hasDocumentation: boolean;
}

export interface ConceptDocumentation {
  id: string;
  name: string;
  description: string;
  documentation?: string;
}

export async function listConcepts(repo?: string): Promise<{
  concepts: ConceptSummary[];
  total: number;
}> {
  const storage = new GraphStorage();
  await storage.initialize();

  const concepts = await storage.getAllConcepts(repo);

  return {
    concepts: concepts.map((f) => ({
      id: f.id,
      repo: f.repo,
      ref_id: f.ref_id,
      name: f.name,
      description: f.description,
      prCount: f.prNumbers.length,
      commitCount: (f.commitShas || []).length,
      lastUpdated: f.lastUpdated.toISOString(),
      hasDocumentation: !!f.documentation,
    })),
    total: concepts.length,
  };
}

export async function getConceptDocumentation(
  conceptId: string,
  repo?: string
): Promise<ConceptDocumentation | null> {
  const storage = new GraphStorage();
  await storage.initialize();

  const concept = await storage.getConcept(conceptId, repo);

  if (!concept) {
    return null;
  }

  return {
    id: concept.id,
    name: concept.name,
    description: concept.description,
    documentation: concept.documentation,
  };
}
