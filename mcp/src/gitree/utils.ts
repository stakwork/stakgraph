import { Concept } from "./types.js";
import { Storage } from "./store/index.js";

/**
 * Format a concept with its PRs and commits for API response
 * Limits PRs and commits to first + last 20 if there are more than 21
 */
export async function formatConceptWithDetails(
  concept: Concept,
  storage: Storage
) {
  const prs = await storage.getPRsForConcept(concept.id);
  const commits = await storage.getCommitsForConcept(concept.id);

  // Limit PRs: if > 21, include first PR and last 20 PRs
  let limitedPrs = prs;
  if (prs.length > 21) {
    const firstPR = prs[0];
    const last20PRs = prs.slice(-20);
    limitedPrs = [firstPR, ...last20PRs];
  }

  // Limit commits: if > 21, include first commit and last 20 commits
  let limitedCommits = commits;
  if (commits.length > 21) {
    const firstCommit = commits[0];
    const last20Commits = commits.slice(-20);
    limitedCommits = [firstCommit, ...last20Commits];
  }

  return {
    concept: {
      id: concept.id,
      name: concept.name,
      description: concept.description,
      documentation: concept.documentation,
    },
    prs: limitedPrs.map((pr) => ({
      number: pr.number,
      title: pr.title,
      summary: pr.summary,
    })),
    commits: limitedCommits.map((commit) => ({
      message: commit.message.split("\n")[0],
      summary: commit.summary,
      author: commit.author,
    })),
  };
}
