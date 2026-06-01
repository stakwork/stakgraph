import { Octokit } from "@octokit/rest";
import { Storage } from "./store/index.js";
import {
  Concept,
  PRRecord,
  CommitRecord,
  LLMDecision,
  ChronologicalCheckpoint,
} from "./types.js";
import { SYSTEM_PROMPT, DECISION_GUIDELINES } from "./llm.js";

/**
 * Pure pipeline helpers extracted from gitree's `StreamingFeatureBuilder`.
 * Each concepts step is a thin vein adapter over these — the engine logic
 * lives here so it stays testable and free of vein/runner concerns.
 */

export interface Change {
  type: "pr" | "commit";
  data: any;
  date: Date;
  id: string;
}

const SKIP_PATTERNS = [/^bump/i, /^chore:/i, /dependabot/i, /^docs:/i, /typo/i, /^ci:/i];

/** Quick heuristic filter (no LLM) — skip obvious maintenance changes. */
export function shouldSkip(text: string): boolean {
  return SKIP_PATTERNS.some((p) => p.test(text));
}

/** Repo-prefixed slug-style concept id, e.g. "owner/repo/feature-slug". */
export function generateConceptId(repo: string, name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `${repo}/${slug}`;
}

// ── Checkpoints ──────────────────────────────────────────────────────────

/** Get the chronological checkpoint for a repo, migrating legacy ones. */
export async function resolveCheckpoint(
  storage: Storage,
  repo: string,
): Promise<ChronologicalCheckpoint | null> {
  const existing = await storage.getChronologicalCheckpoint(repo);
  if (existing) return existing;
  return migrateOldCheckpoint(storage, repo);
}

async function migrateOldCheckpoint(
  storage: Storage,
  repo: string,
): Promise<ChronologicalCheckpoint | null> {
  const lastPR = await storage.getLastProcessedPR(repo);
  const lastCommit = await storage.getLastProcessedCommit(repo);
  if (lastPR === 0 && !lastCommit) return null;

  let latestDate: Date | null = null;
  if (lastPR > 0) {
    const pr = await storage.getPR(lastPR, repo);
    if (pr) latestDate = pr.mergedAt;
  }
  if (lastCommit) {
    const commit = await storage.getCommit(lastCommit, repo);
    if (commit && (!latestDate || commit.committedAt > latestDate)) {
      latestDate = commit.committedAt;
    }
  }
  if (!latestDate) return null;

  const checkpoint: ChronologicalCheckpoint = {
    lastProcessedTimestamp: latestDate.toISOString(),
    processedAtTimestamp: [],
  };
  await storage.setChronologicalCheckpoint(repo, checkpoint);
  return checkpoint;
}

/** Advance the checkpoint after processing a change. */
export async function updateCheckpoint(
  storage: Storage,
  repo: string,
  date: Date,
  id: string,
): Promise<void> {
  const current = await storage.getChronologicalCheckpoint(repo);
  const dateString = date.toISOString();

  if (!current || current.lastProcessedTimestamp < dateString) {
    await storage.setChronologicalCheckpoint(repo, {
      lastProcessedTimestamp: dateString,
      processedAtTimestamp: [id],
    });
  } else if (current.lastProcessedTimestamp === dateString) {
    if (!current.processedAtTimestamp.includes(id)) {
      current.processedAtTimestamp.push(id);
      await storage.setChronologicalCheckpoint(repo, current);
    }
  }
}

// ── Fetch changes ──────────────────────────────────────────────────────────

/**
 * Fetch all changes (merged PRs + standalone commits) since the checkpoint,
 * sorted chronologically (oldest first). Ported from
 * `StreamingFeatureBuilder.fetchAllChanges`.
 */
export async function fetchAllChanges(
  octokit: Octokit,
  owner: string,
  repo: string,
  checkpoint: ChronologicalCheckpoint | null,
): Promise<Change[]> {
  const changes: Change[] = [];
  const sinceDate = checkpoint ? new Date(checkpoint.lastProcessedTimestamp) : null;
  const processedIds = new Set(checkpoint?.processedAtTimestamp || []);

  // PRs
  const allPRs = await octokit.paginate(octokit.pulls.list, {
    owner,
    repo,
    state: "closed",
    sort: "created",
    direction: "asc",
    per_page: 100,
  });

  const mergedPRs = allPRs.filter((pr) => {
    if (!pr.merged_at) return false;
    const mergedDate = new Date(pr.merged_at);
    if (!sinceDate) return true;
    if (mergedDate > sinceDate) return true;
    if (
      mergedDate.getTime() === sinceDate.getTime() &&
      !processedIds.has(pr.number.toString())
    ) {
      return true;
    }
    return false;
  });

  for (const pr of mergedPRs) {
    changes.push({
      type: "pr",
      data: {
        number: pr.number,
        title: pr.title,
        body: pr.body,
        url: pr.html_url,
        mergedAt: new Date(pr.merged_at!),
        additions: 0,
        deletions: 0,
        filesChanged: [],
      },
      date: new Date(pr.merged_at!),
      id: pr.number.toString(),
    });
  }

  // Standalone commits (not associated with any merged PR)
  const commits = await octokit.paginate(octokit.repos.listCommits, {
    owner,
    repo,
    per_page: 100,
    ...(sinceDate ? { since: sinceDate.toISOString() } : {}),
  });

  for (const commit of commits) {
    try {
      if (!commit?.sha || !commit?.commit) continue;

      const { data: prs } =
        await octokit.repos.listPullRequestsAssociatedWithCommit({
          owner,
          repo,
          commit_sha: commit.sha,
        });
      const mergedAssociated = prs.filter((pr) => pr.merged_at);

      if (mergedAssociated.length === 0) {
        const committedAt = new Date(commit.commit.author?.date || Date.now());
        if (sinceDate) {
          if (committedAt < sinceDate) continue;
          if (
            committedAt.getTime() === sinceDate.getTime() &&
            processedIds.has(commit.sha)
          ) {
            continue;
          }
        }
        changes.push({
          type: "commit",
          data: {
            sha: commit.sha,
            message: commit.commit.message,
            author: commit.commit.author?.name || commit.author?.login || "Unknown",
            committedAt,
            url: commit.html_url,
          },
          date: committedAt,
          id: commit.sha,
        });
      }
    } catch (error) {
      console.error(`   ❌ Error checking PR association for ${commit?.sha}:`, error);
    }
  }

  // Chronological (oldest first)
  changes.sort((a, b) => a.date.getTime() - b.date.getTime());
  return changes;
}

// ── Decision prompt ──────────────────────────────────────────────────────────

export interface PromptOverrides {
  systemPrompt?: string;
  guidelines?: string;
}

/** Build the LLM decision prompt: system + concept context + themes + content + guidelines. */
export async function buildDecisionPrompt(
  storage: Storage,
  repo: string,
  content: string,
  overrides: PromptOverrides = {},
): Promise<string> {
  const concepts = await storage.getAllConcepts(repo);
  const themes = await storage.getRecentThemes(repo);
  const system = overrides.systemPrompt ?? SYSTEM_PROMPT;
  const guidelines = overrides.guidelines ?? DECISION_GUIDELINES;

  return `${system}

${formatConceptContext(concepts)}

${formatThemeContext(themes)}

${content}

${guidelines}`;
}

function formatConceptContext(concepts: Concept[]): string {
  if (concepts.length === 0) return "## Current Concepts\n\nNo concepts yet.";
  const list = concepts
    .slice()
    .sort((a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime())
    .map((f) => {
      const prCount = f.prNumbers.length;
      const commitCount = (f.commitShas || []).length;
      const summary =
        commitCount > 0 ? `[${prCount} PRs, ${commitCount} commits]` : `[${prCount} PRs]`;
      return `- **${f.name}** (\`${f.id}\`): ${f.description} ${summary}`;
    })
    .join("\n");
  return `## Current Concepts\n\n${list}`;
}

function formatThemeContext(themes: string[]): string {
  if (themes.length === 0) return "## Recent Technical Themes\n\nNo recent themes.";
  const list = themes.slice().reverse().slice(0, 100).join(", ");
  return `## Recent Technical Themes (last 100 of ${themes.length})\n\n${list}`;
}

// ── Apply decision ──────────────────────────────────────────────────────────

/**
 * Persist the change record and apply the LLM decision to concepts.
 * Returns the ids of concepts that were created/modified.
 * Ported from applyPrDecision / applyCommitDecision / applyDecisionToConcepts.
 */
export async function applyDecision(
  storage: Storage,
  octokit: Octokit,
  owner: string,
  repo: string,
  change: Change,
  decision: LLMDecision,
  usage?: unknown,
  repoPath?: string,
): Promise<string[]> {
  const repoId = `${owner}/${repo}`;
  const modified = new Set<string>();

  if (change.type === "pr") {
    const pr = change.data;
    const { data: files } = await octokit.pulls.listFiles({
      owner,
      repo,
      pull_number: pr.number,
      per_page: 100,
    });
    const prRecord: PRRecord = {
      number: pr.number,
      repo: repoId,
      title: pr.title,
      summary: decision.summary,
      mergedAt: pr.mergedAt,
      url: pr.url,
      files: files.map((f) => f.filename),
      newDeclarations: decision.newDeclarations,
      usage: usage as PRRecord["usage"],
    };
    await storage.savePR(prRecord);

    await applyDecisionToConcepts(storage, repoId, decision, change.date, modified, repoPath, {
      addToConcept: async (concept) => {
        if (!concept.prNumbers.includes(pr.number)) {
          concept.prNumbers.push(pr.number);
          return true;
        }
        return false;
      },
      createConceptWith: (base) => ({ ...base, prNumbers: [pr.number], commitShas: [] }),
    });
  } else {
    const commit = change.data;
    const { data: commitData } = await octokit.repos.getCommit({
      owner,
      repo,
      ref: commit.sha,
    });
    const files = commitData.files || [];
    const commitRecord: CommitRecord = {
      sha: commit.sha,
      repo: repoId,
      message: commit.message,
      summary: decision.summary,
      author: commit.author,
      committedAt: commit.committedAt,
      url: commit.url,
      files: files.map((f: any) => f.filename),
      newDeclarations: decision.newDeclarations,
      usage: usage as CommitRecord["usage"],
    };
    await storage.saveCommit(commitRecord);

    await applyDecisionToConcepts(storage, repoId, decision, change.date, modified, repoPath, {
      addToConcept: async (concept) => {
        if (!concept.commitShas) concept.commitShas = [];
        if (!concept.commitShas.includes(commit.sha)) {
          concept.commitShas.push(commit.sha);
          return true;
        }
        return false;
      },
      createConceptWith: (base) => ({ ...base, prNumbers: [], commitShas: [commit.sha] }),
    });
  }

  return [...modified];
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

        // Explore codebase to seed initial docs when a local clone is
        // available. Lazy-imported so the agentic/repo deps (and their
        // Neo4j-at-load side effects) only load when actually exploring.
        if (repoPath) {
          const { exploreNewConcept } = await import("./bootstrap.js");
          await exploreNewConcept(concept, repoPath, storage);
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
