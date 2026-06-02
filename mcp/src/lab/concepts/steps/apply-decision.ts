import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";
import type {
  Concept,
  LLMDecision,
  PRRecord,
  CommitRecord,
  ChronologicalCheckpoint,
} from "../types.js";
import type { Storage } from "../store/index.js";

/**
 * Persist the change record, apply the LLM decision to concepts
 * (add/create/update + themes), and advance the checkpoint.
 *
 * Self-contained: the apply algorithm (how decisions mutate concepts, how
 * records are saved, how the checkpoint advances) lives inline so it's an
 * editable experiment seam. Clients come from `ctx.services`; agentic doc
 * generation for brand-new concepts is delegated to `services.exploreConcept`.
 * Output: { modifiedConceptIds, usage }.
 */

/** Repo-prefixed slug-style concept id, e.g. "owner/repo/feature-slug". */
function generateConceptId(repo: string, name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `${repo}/${slug}`;
}

/** Advance the chronological checkpoint after processing a change. */
async function updateCheckpoint(
  storage: Storage,
  repo: string,
  date: Date,
  id: string,
): Promise<void> {
  const current = await storage.getChronologicalCheckpoint(repo);
  const dateString = date.toISOString();
  if (!current || current.lastProcessedTimestamp < dateString) {
    const next: ChronologicalCheckpoint = {
      lastProcessedTimestamp: dateString,
      processedAtTimestamp: [id],
    };
    await storage.setChronologicalCheckpoint(repo, next);
  } else if (current.lastProcessedTimestamp === dateString) {
    if (!current.processedAtTimestamp.includes(id)) {
      current.processedAtTimestamp.push(id);
      await storage.setChronologicalCheckpoint(repo, current);
    }
  }
}

interface ApplyConfig {
  addToConcept: (concept: Concept) => Promise<boolean>;
  createConceptWith: (base: Omit<Concept, "prNumbers" | "commitShas">) => Concept;
}

async function applyDecisionToConcepts(
  storage: Storage,
  repoId: string,
  decision: LLMDecision,
  changeDate: Date,
  modified: Set<string>,
  repoPath: string | undefined,
  exploreConcept: ConceptServices["exploreConcept"],
  config: ApplyConfig,
): Promise<void> {
  for (const action of decision.actions) {
    if (action === "add_to_existing") {
      for (const conceptId of decision.existingConceptIds || []) {
        const concept = await storage.getConcept(conceptId, repoId);
        if (concept) {
          const wasAdded = await config.addToConcept(concept);
          if (wasAdded) {
            concept.lastUpdated = changeDate;
            await storage.saveConcept(concept);
            modified.add(concept.id);
          }
        }
      }
    }

    if (action === "create_new") {
      for (const newConcept of decision.newConcepts || []) {
        const base = {
          id: generateConceptId(repoId, newConcept.name),
          repo: repoId,
          name: newConcept.name,
          description: newConcept.description,
          createdAt: changeDate,
          lastUpdated: changeDate,
        };
        const concept = config.createConceptWith(base);
        await storage.saveConcept(concept);
        modified.add(concept.id);

        // Explore the codebase to seed initial docs when a local clone is
        // available (agentic — delegated to services).
        if (repoPath) {
          await exploreConcept(concept, repoPath);
        }
      }
    }
  }

  // Update concept descriptions (and attach the change that caused the update)
  for (const update of decision.updateConcepts || []) {
    const concept = await storage.getConcept(update.conceptId, repoId);
    if (concept) {
      concept.description = update.newDescription;
      concept.lastUpdated = changeDate;
      await config.addToConcept(concept);
      await storage.saveConcept(concept);
      modified.add(concept.id);
    }
  }

  if (decision.themes && decision.themes.length > 0) {
    await storage.addThemes(repoId, decision.themes);
  }
}

export default defineStep({
  type: "concepts/apply-decision",
  description: "Apply an LLM decision to concepts and advance the checkpoint. Output: { modifiedConceptIds, usage }.",
  input: z.object({
    change: z.any(),
    decision: z.any(),
    owner: z.string(),
    repo: z.string(),
    repoPath: z.string().optional(),
    usage: z.any().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { storage, octokit, exploreConcept } = ctx.services as ConceptServices;
    const change = cfg.change;
    const decision = cfg.decision as LLMDecision;
    const repoId = `${cfg.owner}/${cfg.repo}`;
    const repoPath = cfg.repoPath;
    const changeDate = new Date(change.date);
    const modified = new Set<string>();

    if (change.type === "pr") {
      const pr = change.data;
      const { data: files } = await octokit.pulls.listFiles({
        owner: cfg.owner,
        repo: cfg.repo,
        pull_number: pr.number,
        per_page: 100,
      });
      const prRecord: PRRecord = {
        number: pr.number,
        repo: repoId,
        title: pr.title,
        summary: decision.summary,
        mergedAt: new Date(pr.mergedAt),
        url: pr.url,
        files: files.map((f) => f.filename),
        newDeclarations: decision.newDeclarations,
        usage: cfg.usage,
      };
      await storage.savePR(prRecord);

      await applyDecisionToConcepts(
        storage, repoId, decision, changeDate, modified, repoPath, exploreConcept,
        {
          addToConcept: async (concept) => {
            if (!concept.prNumbers.includes(pr.number)) {
              concept.prNumbers.push(pr.number);
              return true;
            }
            return false;
          },
          createConceptWith: (base) => ({ ...base, prNumbers: [pr.number], commitShas: [] }),
        },
      );
    } else {
      const commit = change.data;
      const { data: commitData } = await octokit.repos.getCommit({
        owner: cfg.owner,
        repo: cfg.repo,
        ref: commit.sha,
      });
      const files = commitData.files || [];
      const commitRecord: CommitRecord = {
        sha: commit.sha,
        repo: repoId,
        message: commit.message,
        summary: decision.summary,
        author: commit.author,
        committedAt: new Date(commit.committedAt),
        url: commit.url,
        files: files.map((f: any) => f.filename),
        newDeclarations: decision.newDeclarations,
        usage: cfg.usage,
      };
      await storage.saveCommit(commitRecord);

      await applyDecisionToConcepts(
        storage, repoId, decision, changeDate, modified, repoPath, exploreConcept,
        {
          addToConcept: async (concept) => {
            if (!concept.commitShas) concept.commitShas = [];
            if (!concept.commitShas.includes(commit.sha)) {
              concept.commitShas.push(commit.sha);
              return true;
            }
            return false;
          },
          createConceptWith: (base) => ({ ...base, prNumbers: [], commitShas: [commit.sha] }),
        },
      );
    }

    await updateCheckpoint(storage, repoId, changeDate, String(change.id));

    return { modifiedConceptIds: [...modified], usage: cfg.usage };
  },
});
