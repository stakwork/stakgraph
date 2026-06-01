import { Octokit } from "@octokit/rest";
import { Storage, GraphStorage } from "./store/index.js";
import { LLMClient } from "./llm.js";
import { Summarizer } from "./summarizer.js";
import { getApiKeyForProvider, type Provider } from "../../aieo/src/provider.js";

/**
 * Capabilities bag injected into every concepts step via `ctx.services`.
 *
 * This is the single seam for environment-specific implementations:
 * swap `storage` (Neo4j vs in-memory), `octokit` (real vs fake), or
 * `llm` (real provider vs stub) without touching any workflow or step.
 */
export interface ConceptServices {
  storage: Storage;
  octokit: Octokit;
  llm: LLMClient;
  /** Free-text documentation generator (uses callGenerateText, not the
   *  structured-decision LLMClient). */
  summarizer: Summarizer;
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

  return { storage, octokit, llm, summarizer };
}
