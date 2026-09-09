import { Request, Response } from "express";
import { GraphStorage } from "./store/index.js";
import { Concept, ConceptProposal, ConceptProposalStatus } from "./types.js";
import {
  createConceptDirect,
  createProposal,
  requireConcept,
  HttpError,
} from "./service.js";
import { parseRepoParam } from "./routes.js";

const VALID_STATUSES: ConceptProposalStatus[] = [
  "pending",
  "accepted",
  "rejected",
];

function sendError(res: Response, error: any, fallback: string) {
  if (error instanceof HttpError) {
    res
      .status(error.statusCode)
      .json({ error: error.message, ...(error.extra || {}) });
    return;
  }
  console.error(fallback, error);
  res.status(500).json({ error: error.message || fallback });
}

function optionalString(value: any): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

// A pending proposal's target may drift while it sits in the queue (another
// proposal accepted first, or a manual edit). Refuse to clobber the newer
// docs unless the reviewer explicitly forces it.
function checkStaleBase(
  proposal: ConceptProposal,
  concept: Concept,
  force: boolean
) {
  if (force) return;
  if ((proposal.baseDocs ?? "") !== (concept.documentation ?? "")) {
    throw new HttpError(
      409,
      `Documentation of concept ${concept.id} has changed since this proposal was created; re-review or pass force=true`,
      { code: "stale_base", conceptId: concept.id }
    );
  }
}

function unionArray<T>(a: T[], b: T[]): T[] {
  return Array.from(new Set([...a, ...b]));
}

/**
 * Create a concept proposal
 * POST /gitree/proposals
 * Body: { action, repo?, conceptId?, mergeIntoConceptId?, name?, description?,
 *         documentation?, parent?, rationale?, source?, prNumbers?, sessionIds? }
 */
export async function gitree_create_proposal(req: Request, res: Response) {
  console.log("===> gitree_create_proposal", req.path, req.method);
  try {
    const storage = new GraphStorage();
    await storage.initialize();

    const proposal = await createProposal(storage, req.body || {});

    console.log(
      `✅ Concept proposal created: ${proposal.id} (${proposal.action})`
    );
    res.json({ status: "success", proposal });
  } catch (error: any) {
    sendError(res, error, "Failed to create proposal");
  }
}

/**
 * List proposals
 * GET /gitree/proposals?repo=owner/repo&status=pending (both optional)
 */
export async function gitree_list_proposals(req: Request, res: Response) {
  try {
    const repo = parseRepoParam(req);
    const statusParam = req.query.status as string | undefined;
    if (
      statusParam &&
      !VALID_STATUSES.includes(statusParam as ConceptProposalStatus)
    ) {
      res.status(400).json({
        error: `status must be one of: ${VALID_STATUSES.join(", ")}`,
      });
      return;
    }

    const storage = new GraphStorage();
    await storage.initialize();
    const proposals = await storage.getAllProposals(
      repo,
      statusParam as ConceptProposalStatus | undefined
    );

    res.json({ proposals, count: proposals.length, repo: repo || "all" });
  } catch (error: any) {
    sendError(res, error, "Failed to list proposals");
  }
}

/**
 * Get a specific proposal
 * GET /gitree/proposals/:id
 */
export async function gitree_get_proposal(req: Request, res: Response) {
  try {
    const storage = new GraphStorage();
    await storage.initialize();
    const proposal = await storage.getProposal(req.params.id as string);
    if (!proposal) {
      res.status(404).json({ error: "Proposal not found" });
      return;
    }
    res.json({ proposal });
  } catch (error: any) {
    sendError(res, error, "Failed to get proposal");
  }
}

/**
 * Accept a proposal — applies the change to the Concept graph through the
 * same write paths direct edits use, then stamps the proposal accepted.
 * POST /gitree/proposals/:id/accept
 * Body: { decidedBy?, force? } — force overrides the stale-base check
 */
export async function gitree_accept_proposal(req: Request, res: Response) {
  console.log("===> gitree_accept_proposal", req.path, req.method);
  try {
    const body = req.body || {};
    const decidedBy = optionalString(body.decidedBy);
    const force = body.force === true || body.force === "true";

    const storage = new GraphStorage();
    await storage.initialize();

    const proposal = await storage.getProposal(req.params.id as string);
    if (!proposal) {
      res.status(404).json({ error: "Proposal not found" });
      return;
    }
    if (proposal.status !== "pending") {
      throw new HttpError(
        409,
        `Proposal ${proposal.id} was already ${proposal.status}`,
        { status: proposal.status }
      );
    }

    // Claim before applying so a concurrent accept can never apply twice;
    // rolled back to pending if applying fails.
    const claimed = await storage.claimProposal(
      proposal.id,
      "accepted",
      decidedBy
    );
    if (!claimed) {
      throw new HttpError(
        409,
        `Proposal ${proposal.id} was already decided by another request`
      );
    }

    try {
      const { createdConceptId } = await applyProposal(
        storage,
        proposal,
        force
      );
      if (createdConceptId) {
        await storage.setProposalCreatedConcept(proposal.id, createdConceptId);
      }
      const updated = await storage.getProposal(proposal.id);
      console.log(
        `✅ Concept proposal accepted: ${proposal.id} (${proposal.action})`
      );
      res.json({ status: "success", proposal: updated });
    } catch (applyError: any) {
      await storage.releaseProposalClaim(proposal.id).catch((releaseError) => {
        console.error(
          `Failed to release claim on proposal ${proposal.id}:`,
          releaseError
        );
      });
      throw applyError;
    }
  } catch (error: any) {
    sendError(res, error, "Failed to accept proposal");
  }
}

/**
 * Reject a proposal — stamps the decision, touches no Concept.
 * POST /gitree/proposals/:id/reject
 * Body: { decidedBy?, reason? }
 */
export async function gitree_reject_proposal(req: Request, res: Response) {
  console.log("===> gitree_reject_proposal", req.path, req.method);
  try {
    const body = req.body || {};
    const storage = new GraphStorage();
    await storage.initialize();

    const proposal = await storage.getProposal(req.params.id as string);
    if (!proposal) {
      res.status(404).json({ error: "Proposal not found" });
      return;
    }
    const claimed = await storage.claimProposal(
      proposal.id,
      "rejected",
      optionalString(body.decidedBy),
      optionalString(body.reason)
    );
    if (!claimed) {
      throw new HttpError(
        409,
        `Proposal ${proposal.id} was already ${proposal.status}`,
        { status: proposal.status }
      );
    }
    const updated = await storage.getProposal(proposal.id);
    res.json({ status: "success", proposal: updated });
  } catch (error: any) {
    sendError(res, error, "Failed to reject proposal");
  }
}

export async function applyProposal(
  storage: GraphStorage,
  proposal: ConceptProposal,
  force: boolean
): Promise<{ createdConceptId?: string }> {
  switch (proposal.action) {
    case "create": {
      const { concept } = await createConceptDirect(storage, {
        name: proposal.name || "",
        documentation: proposal.documentation ?? "",
        description: proposal.description,
        repo: proposal.repo,
        parent: proposal.parent,
      });
      return { createdConceptId: concept.id };
    }

    case "update": {
      const concept = await requireConcept(
        storage,
        proposal.conceptId!,
        proposal.repo
      );
      checkStaleBase(proposal, concept, force);
      if (proposal.description !== undefined) {
        // saveConcept refreshes the embedding from name + description
        await storage.saveConcept({
          ...concept,
          description: proposal.description,
          documentation: proposal.documentation ?? "",
          lastUpdated: new Date(),
        });
      } else {
        await storage.saveDocumentation(
          concept.id,
          proposal.documentation ?? ""
        );
      }
      return {};
    }

    case "delete": {
      const concept = await requireConcept(
        storage,
        proposal.conceptId!,
        proposal.repo
      );
      checkStaleBase(proposal, concept, force);
      await storage.deleteConcept(concept.id, proposal.repo);
      return {};
    }

    case "merge": {
      const absorbed = await requireConcept(
        storage,
        proposal.conceptId!,
        proposal.repo
      );
      const into = await requireConcept(
        storage,
        proposal.mergeIntoConceptId!,
        proposal.repo
      );
      checkStaleBase(proposal, into, force);
      // saveConcept also re-creates TOUCHES edges for the unioned provenance
      await storage.saveConcept({
        ...into,
        description: proposal.description ?? into.description,
        documentation: proposal.documentation ?? into.documentation,
        prNumbers: unionArray(into.prNumbers, absorbed.prNumbers),
        commitShas: unionArray(
          into.commitShas || [],
          absorbed.commitShas || []
        ),
        lastUpdated: new Date(),
      });
      await storage.deleteConcept(absorbed.id, proposal.repo);
      return {};
    }
  }
}
