import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";

/**
 * Fetch all merged PRs + standalone commits since the checkpoint, sorted
 * chronologically (oldest first). Output: { changes, count }.
 *
 * Self-contained on purpose: the entire fetch/filter/sort algorithm lives
 * inline so it's the primary "swap the implementation" experiment seam. The
 * only external dep is the GitHub client (`ctx.services.octokit`).
 */
export default defineStep({
  type: "concepts/fetch-changes",
  description: "Fetch merged PRs and standalone commits from GitHub since the checkpoint. Output: { changes, count }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    checkpoint: z.any().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { octokit } = ctx.services as ConceptServices;
    const { owner, repo } = cfg;
    const checkpoint = cfg.checkpoint ?? null;

    const changes: Array<{
      type: "pr" | "commit";
      data: any;
      date: Date;
      id: string;
    }> = [];
    const sinceDate = checkpoint
      ? new Date(checkpoint.lastProcessedTimestamp)
      : null;
    const processedIds = new Set<string>(checkpoint?.processedAtTimestamp ?? []);

    // ── Merged PRs ──────────────────────────────────────────────────────────
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

    // ── Standalone commits (not part of any merged PR) ──────────────────────
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
        if (mergedAssociated.length > 0) continue;

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
            author:
              commit.commit.author?.name || commit.author?.login || "Unknown",
            committedAt,
            url: commit.html_url,
          },
          date: committedAt,
          id: commit.sha,
        });
      } catch (error) {
        console.error(
          `   ❌ Error checking PR association for ${commit?.sha}:`,
          error,
        );
      }
    }

    // Chronological (oldest first)
    changes.sort((a, b) => a.date.getTime() - b.date.getTime());
    return { changes, count: changes.length };
  },
});
