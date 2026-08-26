import { randomUUID } from "crypto";
import { GraphStorage } from "./store/index.js";
import {
  Concept,
  ConceptProposal,
  ConceptProposalAction,
  ConceptProposalStatus,
} from "./types.js";
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

// ─── Concept Proposals ───────────────────────────────────────────────────────

export const VALID_PROPOSAL_ACTIONS: ConceptProposalAction[] = [
  "create",
  "update",
  "delete",
  "merge",
];

export interface CreateProposalInput {
  action: ConceptProposalAction;
  repo?: string;
  conceptId?: string;
  mergeIntoConceptId?: string;
  name?: string;
  description?: string;
  documentation?: string;
  parent?: string;
  rationale?: string;
  source?: string;
  prNumbers?: number[];
  sessionIds?: string[];
}

function optionalString(value: any): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

export async function requireConcept(
  storage: GraphStorage,
  conceptId: string,
  repo?: string
): Promise<Concept> {
  const concept = await storage.getConcept(conceptId, repo);
  if (!concept) {
    throw new HttpError(404, `Concept ${conceptId} not found`);
  }
  return concept;
}

function newProposalShell(input: CreateProposalInput): ConceptProposal {
  const action = input.action;
  if (!VALID_PROPOSAL_ACTIONS.includes(action)) {
    throw new HttpError(
      400,
      `action must be one of: ${VALID_PROPOSAL_ACTIONS.join(", ")}`
    );
  }
  return {
    id: randomUUID(),
    action,
    status: "pending",
    repo: optionalString(input.repo),
    rationale: optionalString(input.rationale),
    source: optionalString(input.source),
    prNumbers: Array.isArray(input.prNumbers)
      ? input.prNumbers.filter((n: any) => Number.isFinite(n)).map(Number)
      : undefined,
    sessionIds: Array.isArray(input.sessionIds)
      ? input.sessionIds.filter((s: any) => typeof s === "string")
      : undefined,
    createdAt: new Date(),
  };
}

/**
 * Per-action validation and target snapshotting, shared by create and revise.
 * Fills the proposal's content fields in place, re-reading the target so
 * baseDocs/absorbedDocs always reflect the target's docs AT THIS MOMENT —
 * which is what lets a revised proposal pass the accept-time staleness check
 * even when the target drifted while an earlier version sat in the queue.
 */
async function validateAndFillProposal(
  storage: GraphStorage,
  proposal: ConceptProposal,
  input: CreateProposalInput
): Promise<void> {
  const action = proposal.action;

  if (action === "create") {
    const name = optionalString(input.name);
    if (!name) {
      throw new HttpError(400, "name is required for a create proposal");
    }
    if (typeof input.documentation !== "string") {
      throw new HttpError(
        400,
        "documentation is required and must be a string"
      );
    }
    const slug = generateSlug(name);
    if (!slug) {
      throw new HttpError(400, "name must contain alphanumeric characters");
    }
    // Early feedback only — existence is re-checked authoritatively at
    // accept time, since concepts can appear while the proposal is pending.
    const targetId = proposal.repo ? makeRepoId(proposal.repo, slug) : slug;
    const existing = await storage.getConcept(targetId, proposal.repo);
    if (existing) {
      throw new HttpError(409, `Concept ${targetId} already exists`, {
        conceptId: targetId,
      });
    }
    const parent = optionalString(input.parent);
    if (parent) {
      const parentConcept = await storage.getConcept(parent, proposal.repo);
      if (!parentConcept) {
        throw new HttpError(400, `Parent concept ${parent} not found`);
      }
      proposal.parent = parentConcept.id;
    }
    proposal.name = name;
    proposal.description = optionalString(input.description);
    proposal.documentation = input.documentation;
  } else if (action === "update") {
    const conceptId = optionalString(input.conceptId);
    if (!conceptId) {
      throw new HttpError(400, "conceptId is required for an update proposal");
    }
    if (typeof input.documentation !== "string") {
      throw new HttpError(
        400,
        "documentation is required and must be a string"
      );
    }
    const concept = await requireConcept(storage, conceptId, proposal.repo);
    proposal.conceptId = concept.id;
    proposal.repo = proposal.repo || concept.repo;
    proposal.baseDocs = concept.documentation ?? "";
    proposal.documentation = input.documentation;
    proposal.description = optionalString(input.description);
  } else if (action === "delete") {
    const conceptId = optionalString(input.conceptId);
    if (!conceptId) {
      throw new HttpError(400, "conceptId is required for a delete proposal");
    }
    const concept = await requireConcept(storage, conceptId, proposal.repo);
    proposal.conceptId = concept.id;
    proposal.repo = proposal.repo || concept.repo;
    proposal.baseDocs = concept.documentation ?? "";
  } else {
    // merge: conceptId is absorbed into mergeIntoConceptId
    const conceptId = optionalString(input.conceptId);
    const mergeIntoConceptId = optionalString(input.mergeIntoConceptId);
    if (!conceptId || !mergeIntoConceptId) {
      throw new HttpError(
        400,
        "conceptId (absorbed) and mergeIntoConceptId (surviving) are required for a merge proposal"
      );
    }
    if (typeof input.documentation !== "string") {
      throw new HttpError(
        400,
        "documentation (the merged docs for the surviving concept) is required and must be a string"
      );
    }
    const absorbed = await requireConcept(storage, conceptId, proposal.repo);
    const into = await requireConcept(
      storage,
      mergeIntoConceptId,
      proposal.repo
    );
    if (absorbed.id === into.id) {
      throw new HttpError(400, "A concept cannot be merged into itself");
    }
    proposal.conceptId = absorbed.id;
    proposal.mergeIntoConceptId = into.id;
    proposal.repo = proposal.repo || into.repo;
    proposal.baseDocs = into.documentation ?? "";
    proposal.absorbedDocs = absorbed.documentation ?? "";
    proposal.documentation = input.documentation;
    proposal.description = optionalString(input.description);
  }
}

/**
 * Validate and persist a ConceptProposal. Shared by POST /gitree/proposals
 * and the propose_concept_change agent tool. Snapshots the target's current
 * docs (baseDocs/absorbedDocs) server-side so accept can detect drift.
 */
export async function createProposal(
  storage: GraphStorage,
  input: CreateProposalInput
): Promise<ConceptProposal> {
  const proposal = newProposalShell(input);
  await validateAndFillProposal(storage, proposal, input);
  await storage.saveProposal(proposal);
  return proposal;
}

/**
 * Replace a PENDING proposal's content in place — same validation and target
 * snapshotting as createProposal, keeping the proposal's id and createdAt.
 * Used by session reflection to keep one evolving draft per (session, target)
 * instead of filing a sibling every turn.
 *
 * Evidence accumulates: prNumbers and sessionIds are unioned with what the
 * proposal already carries. The pending-only guard is enforced twice — an
 * early check for a friendly error, and atomically in storage.updateProposal
 * so a reviewer deciding the proposal mid-revision always wins.
 */
export async function reviseProposal(
  storage: GraphStorage,
  id: string,
  input: CreateProposalInput
): Promise<ConceptProposal> {
  const existing = await storage.getProposal(id);
  if (!existing) {
    throw new HttpError(404, `Proposal ${id} not found`);
  }
  if (existing.status !== "pending") {
    throw new HttpError(409, `Proposal ${id} was already ${existing.status}`, {
      status: existing.status,
    });
  }

  const shell = newProposalShell(input);
  const proposal: ConceptProposal = {
    ...shell,
    id: existing.id,
    createdAt: existing.createdAt,
    source: shell.source ?? existing.source,
    prNumbers: unionEvidence(existing.prNumbers, shell.prNumbers),
    sessionIds: unionEvidence(existing.sessionIds, shell.sessionIds),
  };
  await validateAndFillProposal(storage, proposal, input);

  const updated = await storage.updateProposal(proposal);
  if (!updated) {
    throw new HttpError(409, `Proposal ${id} was decided while being revised`);
  }
  return proposal;
}

function unionEvidence<T>(a?: T[], b?: T[]): T[] | undefined {
  if (!a?.length && !b?.length) return undefined;
  return Array.from(new Set([...(a ?? []), ...(b ?? [])]));
}

export async function listProposals(
  repo?: string,
  status?: ConceptProposalStatus
): Promise<ConceptProposal[]> {
  const storage = new GraphStorage();
  await storage.initialize();
  return storage.getAllProposals(repo, status);
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
