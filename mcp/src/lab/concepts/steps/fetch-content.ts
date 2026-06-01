import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";
import { shouldSkip } from "../pipeline.js";
import { fetchPullRequestContent } from "../pr.js";
import { fetchCommitContent } from "../commit.js";

const MAX_PATCH_LINES = 100;

/**
 * Fetch the full markdown content for one change (PR or commit) for the
 * LLM. Applies the maintenance-noise skip heuristic — skipped changes
 * still get a minimal record saved (matching gitree). Output:
 * { skipped, markdown, change }.
 */
export default defineStep({
  type: "concepts/fetch-content",
  description: "Fetch a single PR/commit's full content as markdown. Skips maintenance changes (bump/chore/dependabot/docs/typo/ci). Output: { skipped, markdown, change }.",
  input: z.object({
    change: z.any(),
    owner: z.string(),
    repo: z.string(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { octokit, storage } = ctx.services as ConceptServices;
    const change = cfg.change;
    const repoId = `${cfg.owner}/${cfg.repo}`;

    if (change.type === "pr") {
      const pr = change.data;
      if (shouldSkip(pr.title)) {
        await storage.savePR({
          number: pr.number,
          repo: repoId,
          title: pr.title,
          summary: "Skipped (maintenance/trivial)",
          mergedAt: new Date(pr.mergedAt),
          url: pr.url,
          files: [],
        });
        return { skipped: true, markdown: null, change };
      }

      // Enrich additions/deletions (lazy per-PR detail fetch)
      const { data: fullPR } = await octokit.pulls.get({
        owner: cfg.owner,
        repo: cfg.repo,
        pull_number: pr.number,
      });
      pr.additions = fullPR.additions || 0;
      pr.deletions = fullPR.deletions || 0;

      const markdown = await fetchPullRequestContent(
        octokit,
        { owner: cfg.owner, repo: cfg.repo, pull_number: pr.number },
        { maxPatchLines: MAX_PATCH_LINES },
      );
      return { skipped: false, markdown, change };
    }

    // commit
    const commit = change.data;
    if (shouldSkip(commit.message)) {
      await storage.saveCommit({
        sha: commit.sha,
        repo: repoId,
        message: commit.message,
        summary: "Skipped (maintenance/trivial)",
        author: commit.author,
        committedAt: new Date(commit.committedAt),
        url: commit.url,
        files: [],
      });
      return { skipped: true, markdown: null, change };
    }

    const markdown = await fetchCommitContent(
      octokit,
      { owner: cfg.owner, repo: cfg.repo, sha: commit.sha },
      { maxPatchLines: MAX_PATCH_LINES },
    );
    return { skipped: false, markdown, change };
  },
});
