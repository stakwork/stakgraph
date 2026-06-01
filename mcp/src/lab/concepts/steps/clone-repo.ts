import { z, defineStep } from "vein";

/**
 * Clone (or update) the repo locally so bootstrap / doc generation can
 * explore the codebase. Output: { repoPath }.
 */
export default defineStep({
  type: "concepts/clone-repo",
  description: "Clone or update a GitHub repo locally. Output: { repoPath }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    token: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg) {
    const { cloneOrUpdateRepo } = await import("../../../repo/clone.js");
    const token = cfg.token ?? process.env["GITHUB_TOKEN"];
    const repoPath = await cloneOrUpdateRepo(
      `https://github.com/${cfg.owner}/${cfg.repo}`,
      undefined,
      token,
    );
    return { repoPath };
  },
});
