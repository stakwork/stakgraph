import { Feature } from "./types.js";
import { Storage } from "./store/index.js";

/**
 * Format a feature with its PRs and commits for API response
 * Limits PRs to first + last 20 if there are more than 21
 */
export async function formatFeatureWithDetails(
  feature: Feature,
  storage: Storage
) {
  const prs = await storage.getPRsForFeature(feature.id);
  const commits = await storage.getCommitsForFeature(feature.id);

  // Limit PRs: if > 21, include first PR and last 20 PRs
  let limitedPrs = prs;
  if (prs.length > 21) {
    const firstPR = prs[0];
    const last20PRs = prs.slice(-20);
    limitedPrs = [firstPR, ...last20PRs];
  }

  return {
    feature: {
      id: feature.id,
      name: feature.name,
      description: feature.description,
      documentation: feature.documentation,
      prNumbers: feature.prNumbers,
      commitShas: feature.commitShas || [],
      createdAt: feature.createdAt.toISOString(),
      lastUpdated: feature.lastUpdated.toISOString(),
    },
    prs: limitedPrs.map((pr) => ({
      number: pr.number,
      title: pr.title,
      summary: pr.summary,
      mergedAt: pr.mergedAt.toISOString(),
      url: pr.url,
    })),
    commits: commits.map((commit) => ({
      sha: commit.sha,
      message: commit.message.split("\n")[0],
      summary: commit.summary,
      author: commit.author,
      committedAt: commit.committedAt.toISOString(),
      url: commit.url,
    })),
  };
}
