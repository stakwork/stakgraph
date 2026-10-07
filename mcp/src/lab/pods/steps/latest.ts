import { z, defineStep } from "strut";
import { failure, gitCredentials, settle, submit, type PodCtx } from "./_shared.js";

const Repo = z.object({
  url: z.string().min(1).describe("The repository as the pod knows it: https://github.com/<owner>/<name>.git"),
  base_branch: z.string().min(1).optional().describe("The branch to reset to; the repository's default branch when omitted"),
});

export default defineStep({
  type: "pod/latest",
  description:
    "Reset a pod's repositories to the latest of a branch: each is checked out at `base_branch` (default: its default branch), fetched, hard-reset and cleaned; " +
    "then the pod reinstalls and restarts its services. Do this once after a claim — the pod comes with the workspace's repositories at whatever state the pool " +
    "left them — and to start a task from a fresh base. Everything uncommitted in those repositories is lost. Waits for the pod (default 15 min); the run's GitHub " +
    "token authenticates the fetch when it has one. Output: { request_id, credentials: actor | pool, repos }.",
  input: z.object({
    control: z.string().min(1).describe("The pod's control URL (pod/claim)"),
    sealed: z.string().min(1).describe("The pod's sealed password (pod/claim)"),
    repos: z.array(Repo).min(1).describe("The repositories to reset, in order"),
    githubTokenSecret: z.string().default("GITHUB_TOKEN").describe("NAME of the secret with the run's GitHub token"),
    pollMs: z.number().int().positive().default(5000),
    timeoutMs: z.number().int().positive().default(900_000),
  }),
  output: z.object({
    request_id: z.string(),
    credentials: z.enum(["actor", "pool"]),
    repos: z.array(Repo),
  }),
  async run(cfg, ctx: PodCtx) {
    const creds = await gitCredentials(ctx, cfg.githubTokenSecret);
    const repos = cfg.repos.map((r) => (r.base_branch ? { url: r.url, base_branch: r.base_branch } : { url: r.url }));
    const body = { tasks: [], repos, ...(creds ? { git_credentials: creds } : {}) };
    const first = await submit(ctx, cfg.control, cfg.sealed, "/latest", "PUT", body);
    const p = await settle(ctx, cfg.control, cfg.sealed, first, { pollMs: cfg.pollMs, timeoutMs: cfg.timeoutMs, what: "pod/latest" });
    if (p.status === "failed") {
      throw new Error(`pod/latest: ${failure(p.body)} (the pod resets repositories one after another with no per-repository result; some may have been reset)`);
    }
    return { request_id: first.requestId, credentials: creds ? ("actor" as const) : ("pool" as const), repos: cfg.repos };
  },
});
