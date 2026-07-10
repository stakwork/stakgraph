import { test, expect } from "../../testkit.js";

const describe = test.describe;

/**
 * Unit tests for the ignoreRepoInfo bypass logic in repo_agent.
 *
 * The logic under test (from index.ts):
 *   const promptWithRepoInfo = (isExistingSession || body.ignoreRepoInfo)
 *     ? body.prompt
 *     : prependRepoInfo(body.prompt, body.repoList, graphRepos);
 *
 * We test this by reproducing the same conditional inline.
 */

function prependRepoInfo(prompt: any, clonedRepos: string[], graphRepos: string[]): any {
  const uniqueGraphRepos = [...new Set(graphRepos.filter(Boolean))];
  const uniqueClonedRepos = [...new Set(clonedRepos.filter(Boolean))];
  const clonedOnlyRepos = uniqueClonedRepos.filter((repo) => !uniqueGraphRepos.includes(repo));

  const lines: string[] = ["Repository availability for this session:"];

  if (uniqueGraphRepos.length > 0) {
    lines.push(
      "Graph-backed repos (prefer repo_overview, stakgraph_search, stakgraph_map, and stakgraph_code for these):",
      ...uniqueGraphRepos.map((repo) => `- ${repo}`)
    );
  }

  if (clonedOnlyRepos.length > 0) {
    lines.push(
      "Additional repos cloned locally under /tmp/{owner}/{repo} but not ingested in the graph (use bash/fulltext_search for these):",
      ...clonedOnlyRepos.map((repo) => `- ${repo}`)
    );
  }

  if (uniqueClonedRepos.length === 0 && uniqueGraphRepos.length > 0) {
    lines.push(
      "No repo_url was supplied. Use the ingested repos above; they are also available locally under /tmp/{owner}/{repo}."
    );
  }

  if (uniqueGraphRepos.length === 0 && uniqueClonedRepos.length > 0) {
    lines.push(
      "No ingested Repository nodes were found for the requested repos, so rely on bash/fulltext_search over the cloned repos."
    );
  }

  const repoInfo = `${lines.join("\n")}\n\n`;
  if (typeof prompt === "string") {
    return repoInfo + prompt;
  }
  if (Array.isArray(prompt)) {
    return prompt.map((msg: any, i: number) => {
      if (i === 0 && msg.role === "user") {
        return { ...msg, content: repoInfo + msg.content };
      }
      return msg;
    });
  }
  return prompt;
}

/** Mirrors the ternary in repo_agent. */
function resolvePrompt(
  prompt: any,
  repoList: string[],
  graphRepos: string[],
  isExistingSession: boolean,
  ignoreRepoInfo: boolean | undefined
): any {
  return isExistingSession || ignoreRepoInfo
    ? prompt
    : prependRepoInfo(prompt, repoList, graphRepos);
}

const GRAPH_REPOS = ["stakwork/stakgraph"];
const RAW_PROMPT = "What does this repo do?";

describe("ignoreRepoInfo bypass logic", () => {
  test("new session + ignoreRepoInfo: true → raw prompt returned unchanged", () => {
    const result = resolvePrompt(RAW_PROMPT, [], GRAPH_REPOS, false, true);
    expect(result).toBe(RAW_PROMPT);
  });

  test("new session + ignoreRepoInfo: false → repo info prepended", () => {
    const result = resolvePrompt(RAW_PROMPT, [], GRAPH_REPOS, false, false);
    expect(typeof result).toBe("string");
    expect(result).toContain("Repository availability for this session:");
    expect(result).toContain(RAW_PROMPT);
  });

  test("new session + ignoreRepoInfo omitted (undefined) → repo info prepended", () => {
    const result = resolvePrompt(RAW_PROMPT, [], GRAPH_REPOS, false, undefined);
    expect(typeof result).toBe("string");
    expect(result).toContain("Repository availability for this session:");
    expect(result).toContain(RAW_PROMPT);
  });

  test("existing session + ignoreRepoInfo: true → raw prompt returned (session takes precedence)", () => {
    const result = resolvePrompt(RAW_PROMPT, [], GRAPH_REPOS, true, true);
    expect(result).toBe(RAW_PROMPT);
  });

  test("existing session + ignoreRepoInfo: false → raw prompt returned (session takes precedence)", () => {
    const result = resolvePrompt(RAW_PROMPT, [], GRAPH_REPOS, true, false);
    expect(result).toBe(RAW_PROMPT);
  });

  test("new session + ignoreRepoInfo: true with array prompt → array returned unchanged", () => {
    const arrayPrompt = [{ role: "user", content: RAW_PROMPT }];
    const result = resolvePrompt(arrayPrompt, [], GRAPH_REPOS, false, true);
    expect(result).toEqual(arrayPrompt);
  });

  test("new session + ignoreRepoInfo: false with array prompt → repo info prepended to first user message", () => {
    const arrayPrompt = [{ role: "user", content: RAW_PROMPT }];
    const result = resolvePrompt(arrayPrompt, [], GRAPH_REPOS, false, false);
    expect(Array.isArray(result)).toBe(true);
    expect(result[0].content).toContain("Repository availability for this session:");
    expect(result[0].content).toContain(RAW_PROMPT);
  });
});
