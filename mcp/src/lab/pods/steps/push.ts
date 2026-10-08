import { z, defineStep } from "strut";
import { brief, gitCredentials, podCall, type PodCtx } from "./_shared.js";

export default defineStep({
  type: "pod/push",
  description:
    "Commit everything in a pod repository's working tree and push it as the run's GitHub identity, opening a pull request unless told not to. " +
    "A NEW change: `branch_name` names the branch (made unique if taken), the pull request's title is the commit message's first sentence, and its base is the " +
    "repository's default branch, which the pod reads from the remote — leave `base_branch` out; a guess (`main` on a `master` repository) is refused with the real default named. " +
    "A REVISION of a pull request: `stay_on_branch: true` commits on the branch the pod is on — the pull request's — and pushes it, so the same pull request " +
    "gains the commit and comes back as `pr_url`; never open a second one for the same change. Only while the pod is still on that branch: after pod/latest " +
    "or a new claim it sits on the base, and `stay_on_branch` there commits to the base branch itself. Nothing to commit is fine — unpushed commits still go; " +
    "a pull request with nothing ahead of the base fails. A pull request the pod could not open fails this step with the pod's reason: fix what it names and call pod/push again — " +
    "never have the agent in the pod open it. Output: { branch, pr_url (null without a pull request), commits: a link to each pushed head commit }.",
  input: z.object({
    control: z.string().min(1).describe("The pod's control URL (pod/claim)"),
    sealed: z.string().min(1).describe("The pod's sealed password (pod/claim)"),
    repo_url: z.string().min(1).describe("The repository as the pod knows it: https://github.com/<owner>/<name>.git"),
    branch_name: z.string().min(1).describe("The branch to push: a new one for a new change; ignored with stay_on_branch"),
    commit_message: z.string().min(1).describe("The commit message; its first sentence titles the pull request"),
    base_branch: z.string().min(1).optional().describe("The pull request's base. Omit for a new change: the pod uses the repository's default branch. Give it only when the base is another branch, such as a pull request's."),
    create_pr: z.boolean().default(true).describe("Open a pull request for the branch, or return the one it already has (a revision)"),
    stay_on_branch: z.boolean().default(false).describe("Commit on the branch the pod is on instead of making a new one — a revision of its pull request"),
    githubTokenSecret: z.string().default("GITHUB_TOKEN").describe("NAME of the secret with the run's GitHub token"),
    timeoutMs: z.number().int().positive().default(600_000),
  }),
  output: z.object({
    branch: z.string(),
    pr_url: z.string().nullable(),
    commits: z.array(z.string()),
  }),
  async run(cfg, ctx: PodCtx) {
    const creds = await gitCredentials(ctx, cfg.githubTokenSecret);
    if (!creds) throw new Error(`pod/push: no ${cfg.githubTokenSecret} for this run — the pod pushes as the run's GitHub identity, and it has none`);
    const q = new URLSearchParams({ commit: "true", pr: String(cfg.create_pr) });
    if (cfg.stay_on_branch) q.set("stayOnCurrentBranch", "true");
    const path = `/push?${q}`;
    const body = {
      tasks: [],
      repos: [{ url: cfg.repo_url, branch_name: cfg.branch_name, commit_name: cfg.commit_message, ...(cfg.base_branch ? { base_branch: cfg.base_branch } : {}) }],
      git_credentials: creds,
    };
    const res = await podCall(ctx, cfg.control, cfg.sealed, path, { method: "POST", body, timeout: cfg.timeoutMs });
    if (!res.ok) throw new Error(`POST ${path}: ${res.status} — ${brief(res.body)}`);
    const b: any = res.body ?? {};
    if (b.error) throw new Error(`pod/push: ${b.message ?? b.error}`);
    const branch = Object.values((b.branches ?? {}) as Record<string, string>)[0];
    const pr_url = Object.values((b.prs ?? {}) as Record<string, string>)[0] ?? null;
    if (b.prErrors && Object.keys(b.prErrors).length) throw new Error(`pod/push: pushed ${branch ?? "?"} but the pull request failed — ${brief(b.prErrors)}`);
    if (!branch) throw new Error(`pod/push: the pod reported no branch: ${brief(b)}`);
    if (cfg.create_pr && !pr_url) throw new Error(`pod/push: pushed ${branch} but the pod reported no pull request: ${brief(b)}`);
    return { branch, pr_url, commits: Array.isArray(b.commits) ? b.commits.map(String) : [] };
  },
});
