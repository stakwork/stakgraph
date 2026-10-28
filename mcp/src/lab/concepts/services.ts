import { Octokit } from "@octokit/rest";
import { Storage, GraphStorage } from "./store/index.js";
import { LLMClient } from "./llm.js";
import { Summarizer } from "./summarizer.js";
import { getApiKeyForProvider, type Provider } from "../../aieo/src/provider.js";
import type { Concept, Usage } from "./types.js";
import type { GraphBackend } from "strut";
import { env } from "../../env.js";

/**
 * Capabilities bag injected into every concepts step via `ctx.services`.
 *
 * This is the single seam for environment-specific implementations and the
 * boundary that keeps steps **self-contained**: steps import only `strut` and
 * reach every external capability through here, so their source can live as
 * an editable, portable file in the workspace (`steps/custom/`).
 *
 * The split is deliberate:
 *   - **clients** (`storage`, `github`, `llm`, `summarizer`) — runtime objects
 *     you can't serialize into an editable file.
 *   - **infra capabilities** (`clone`, `bootstrap`, `exploreConcept`,
 *     `prContent`, `commitContent`) — heavy/agentic operations bound to repo
 *     and Neo4j internals. The *algorithm* a step author wants to experiment
 *     with (fetch/filter/sort, the decision prompt, ordering, apply logic)
 *     lives **inline in the step**, not here.
 */
export interface CodeNodeToDescribe {
  ref_id: string;
  labels: string[];
  name: string;
  /** The node's `file` property: its path, `owner/repo/…`. */
  file: string;
  docs: string;
  body: string;
}

/** A node another one CONTAINS (a File's Functions, Classes, …). */
export interface ContainedNode {
  type: string;
  name: string;
  description: string;
}

export interface ConceptServices {
  storage: Storage;
  /** strut's graph backend, the one behind the lab's graph workspace —
   *  put on the bag by createStrut (`ctx.services.graph`, what the graph/*
   *  steps read), never built here. `concepts/pages` writes the Concept
   *  pages above the features through it, so every reader sees them as it
   *  sees a seeded page. Absent on a bag built outside createStrut. */
  graph?: GraphBackend;
  /** An Octokit for a GitHub token — the run's `GITHUB_TOKEN` secret, which
   *  the step reads — or the deployment's env token when none. Built per
   *  token and cached. Never built at boot: a boot-time client would be
   *  the deployment's identity for every actor's run. */
  github(token?: string): Octokit;
  /** strut's standard secrets capability, bound to the run's principal (an
   *  actor's secret first, then the deployment's, then env). Absent outside
   *  the runner (unit tests). */
  secrets?: { get(name: string): Promise<string | undefined> };
  llm: LLMClient;
  /** Free-text documentation generator (uses callGenerateText, not the
   *  structured-decision LLMClient). */
  summarizer: Summarizer;

  /** Clone (or update) a GitHub repo locally; returns the local path. */
  clone(owner: string, repo: string, token?: string): Promise<string>;

  /** Agentically explore a fresh clone to seed initial concepts and set the
   *  checkpoint to a recent lookback window. The prompt (`system` + `template`)
   *  is the experiment surface — supplied by the workflow's `params` via the
   *  `concepts/bootstrap-explore` step, not baked into the service. */
  bootstrap(
    owner: string,
    repo: string,
    repoPath: string,
    promptConfig: { system: string; template: string },
    lookbackDays?: number,
  ): Promise<{ concepts: Concept[]; usage: Usage }>;

  /** Agentically generate documentation for a newly-created concept. */
  exploreConcept(concept: Concept, repoPath: string): Promise<void>;

  /** Build full markdown content for a single PR (for the LLM), with the
   *  given GitHub token (see `github`). */
  prContent(
    args: { owner: string; repo: string; pull_number: number },
    opts?: { maxPatchLines?: number },
    token?: string,
  ): Promise<string>;

  /** Code nodes for `concepts/describe-code`: the ones of `types` (labels)
   *  with a body and an empty `field` — `description` or `summary`, nothing
   *  else — optionally limited to repos, as `owner/repo`, and file path
   *  suffixes; what nodes CONTAIN (of the labels `types`, in file order:
   *  how many, and the first `max`); and the write-back to `field`. Writes
   *  the field only — strut's embedding pass re-embeds the node from its
   *  Schema `index`. */
  codeNodes: {
    withoutDescription(
      limit: number,
      repoPaths: string[],
      filePaths: string[],
      opts: { types: string[]; field: string },
    ): Promise<CodeNodeToDescribe[]>;
    contained(
      refIds: string[],
      types: string[],
      max: number,
    ): Promise<Record<string, { total: number; nodes: ContainedNode[] }>>;
    writeDescriptions(batch: Array<{ ref_id: string; description: string }>, field: string): Promise<void>;
  };

  /** Build full markdown content for a single commit (for the LLM), with the
   *  given GitHub token (see `github`). */
  commitContent(
    args: { owner: string; repo: string; sha: string },
    opts?: { maxPatchLines?: number },
    token?: string,
  ): Promise<string>;
}

export interface BuildServicesOptions {
  /** The deployment's GitHub token, used when a run has none of its own.
   *  Falls back to `process.env.GITHUB_TOKEN`. */
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

  const envToken = opts.githubToken ?? env["GITHUB_TOKEN"];
  // One Octokit per distinct token. A deployment sees a handful of actors'
  // tokens; the cap only guards against a pathological churn of them.
  const clients = new Map<string, Octokit>();
  const github = (token?: string): Octokit => {
    const auth = token || envToken;
    const key = auth ?? "";
    let client = clients.get(key);
    if (!client) {
      if (clients.size >= 64) clients.clear();
      client = new Octokit(auth ? { auth } : {});
      clients.set(key, client);
    }
    return client;
  };

  const llm = new LLMClient(provider, apiKey);
  const summarizer = new Summarizer(storage, provider, apiKey);

  return {
    storage,
    github,
    llm,
    summarizer,

    async clone(owner, repo, token) {
      const { cloneOrUpdateRepo } = await import("../../repo/clone.js");
      return cloneOrUpdateRepo(
        `https://github.com/${owner}/${repo}`,
        undefined,
        token || envToken,
      );
    },

    async bootstrap(owner, repo, repoPath, promptConfig, lookbackDays) {
      const { bootstrapConcepts } = await import("./bootstrap.js");
      const result = await bootstrapConcepts(
        owner,
        repo,
        repoPath,
        storage,
        promptConfig,
        undefined,
        lookbackDays,
      );
      return { concepts: result.features, usage: result.usage };
    },

    async exploreConcept(concept, repoPath) {
      const { exploreNewConcept } = await import("./bootstrap.js");
      await exploreNewConcept(concept, repoPath, storage);
    },

    async prContent(args, opts, token) {
      const { fetchPullRequestContent } = await import("./pr.js");
      return fetchPullRequestContent(github(token), args, opts);
    },

    async commitContent(args, opts, token) {
      const { fetchCommitContent } = await import("./commit.js");
      return fetchCommitContent(github(token), args, opts);
    },

    codeNodes: {
      async withoutDescription(limit, repoPaths, filePaths, { types, field }) {
        const { db } = await import("../../graph/neo4j.js");
        const nodes = await db.get_nodes_without_description(limit, repoPaths, filePaths, types, field);
        return nodes
          .filter((n) => !!n.ref_id)
          .map((n) => ({
            ref_id: n.ref_id!,
            labels: n.labels,
            name: String(n.properties.name ?? ""),
            file: String(n.properties.file ?? ""),
            docs: String(n.properties.docs ?? ""),
            body: String(n.properties.body ?? ""),
          }));
      },
      async contained(refIds, types, max) {
        const { db } = await import("../../graph/neo4j.js");
        return db.get_contained_nodes(refIds, types, max);
      },
      async writeDescriptions(batch, field) {
        const { db } = await import("../../graph/neo4j.js");
        await db.bulk_update_descriptions(batch, field);
      },
    },
  };
}
