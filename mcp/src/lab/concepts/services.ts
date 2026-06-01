import { Octokit } from "@octokit/rest";
import { Storage, GraphStorage } from "./store/index.js";
import { LLMClient } from "./llm.js";
import { Summarizer } from "./summarizer.js";
import { getApiKeyForProvider, type Provider } from "../../aieo/src/provider.js";
import type { Concept, Usage } from "./types.js";

/**
 * Capabilities bag injected into every concepts step via `ctx.services`.
 *
 * This is the single seam for environment-specific implementations and the
 * boundary that keeps steps **self-contained**: steps import only `vein` and
 * reach every external capability through here, so their source can live as
 * an editable, portable file in the workspace (`steps/custom/`).
 *
 * The split is deliberate:
 *   - **clients** (`storage`, `octokit`, `llm`, `summarizer`) — runtime objects
 *     you can't serialize into an editable file.
 *   - **infra capabilities** (`clone`, `bootstrap`, `exploreConcept`,
 *     `prContent`, `commitContent`) — heavy/agentic operations bound to repo
 *     and Neo4j internals. The *algorithm* a step author wants to experiment
 *     with (fetch/filter/sort, the decision prompt, ordering, apply logic)
 *     lives **inline in the step**, not here.
 */
export interface ConceptServices {
  storage: Storage;
  octokit: Octokit;
  llm: LLMClient;
  /** Free-text documentation generator (uses callGenerateText, not the
   *  structured-decision LLMClient). */
  summarizer: Summarizer;

  /** Clone (or update) a GitHub repo locally; returns the local path. */
  clone(owner: string, repo: string, token?: string): Promise<string>;

  /** Agentically explore a fresh clone to seed initial concepts and set the
   *  checkpoint to a recent lookback window. */
  bootstrap(
    owner: string,
    repo: string,
    repoPath: string,
    lookbackDays?: number,
  ): Promise<{ concepts: Concept[]; usage: Usage }>;

  /** Agentically generate documentation for a newly-created concept. */
  exploreConcept(concept: Concept, repoPath: string): Promise<void>;

  /** Build full markdown content for a single PR (for the LLM). */
  prContent(
    args: { owner: string; repo: string; pull_number: number },
    opts?: { maxPatchLines?: number },
  ): Promise<string>;

  /** Build full markdown content for a single commit (for the LLM). */
  commitContent(
    args: { owner: string; repo: string; sha: string },
    opts?: { maxPatchLines?: number },
  ): Promise<string>;
}

export interface BuildServicesOptions {
  /** GitHub token for Octokit. Falls back to `process.env.GITHUB_TOKEN`. */
  githubToken?: string;
  /** LLM provider. Defaults to "anthropic". */
  provider?: Provider;
}

/**
 * Build the default filesystem/Neo4j-backed services bag. Mirrors how
 * `gitree/routes.ts` constructs its storage/octokit/llm, but as one
 * injectable object.
 *
 * The infra capabilities lazy-import their heavy/agentic dependencies (repo
 * clone, the bootstrap explorer, PR/commit markdown builders) so those — and
 * their Neo4j-at-load side effects — only load when actually exercised.
 */
export async function buildConceptServices(
  opts: BuildServicesOptions = {},
): Promise<ConceptServices> {
  const provider: Provider = opts.provider ?? "anthropic";
  const apiKey = getApiKeyForProvider(provider);

  const storage = new GraphStorage();
  await storage.initialize();

  const githubToken = opts.githubToken ?? process.env["GITHUB_TOKEN"];
  const octokit = new Octokit(githubToken ? { auth: githubToken } : {});

  const llm = new LLMClient(provider, apiKey);
  const summarizer = new Summarizer(storage, provider, apiKey);

  return {
    storage,
    octokit,
    llm,
    summarizer,

    async clone(owner, repo, token) {
      const { cloneOrUpdateRepo } = await import("../../repo/clone.js");
      return cloneOrUpdateRepo(
        `https://github.com/${owner}/${repo}`,
        undefined,
        token ?? githubToken,
      );
    },

    async bootstrap(owner, repo, repoPath, lookbackDays) {
      const { bootstrapConcepts } = await import("./bootstrap.js");
      const result = await bootstrapConcepts(
        owner,
        repo,
        repoPath,
        storage,
        undefined,
        lookbackDays,
      );
      return { concepts: result.features, usage: result.usage };
    },

    async exploreConcept(concept, repoPath) {
      const { exploreNewConcept } = await import("./bootstrap.js");
      await exploreNewConcept(concept, repoPath, storage);
    },

    async prContent(args, opts) {
      const { fetchPullRequestContent } = await import("./pr.js");
      return fetchPullRequestContent(octokit, args, opts);
    },

    async commitContent(args, opts) {
      const { fetchCommitContent } = await import("./commit.js");
      return fetchCommitContent(octokit, args, opts);
    },
  };
}
