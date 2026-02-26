import test from "node:test";
import assert from "node:assert/strict";

import {
  getNodesQuerySchema,
  postNodesBodySchema,
  getEdgesQuerySchema,
  refIdQuerySchema,
  workflowQuerySchema,
  repoMapQuerySchema,
  shortestPathQuerySchema,
  mapQuerySchema,
  graphQuerySchema,
  searchQuerySchema,
} from "../../graph/routes/schemas/graph.js";
import {
  exploreQuerySchema,
  understandQuerySchema,
  seedUnderstandingQuerySchema,
  askQuerySchema,
  getLearningsQuerySchema,
  createLearningBodySchema,
  createPullRequestBodySchema,
  seedStoriesQuerySchema,
  reconnectQuerySchema,
} from "../../graph/routes/schemas/knowledge.js";
import {
  gitseeBodySchema,
  gitseeAgentQuerySchema,
  gitseeEventsParamsSchema,
  gitseeServicesQuerySchema,
  requestIdQuerySchema,
} from "../../graph/routes/schemas/gitsee.js";
import {
  getServicesQuerySchema,
  mocksInventoryQuerySchema,
} from "../../graph/routes/schemas/services.js";
import {
  mcpServerSchema,
  sessionConfigSchema as repoSessionConfigSchema,
  getLeaksQuerySchema,
  repoAgentBodySchema,
  getAgentSessionQuerySchema,
} from "../../repo/schemas/routes.js";
import {
  logsAgentBodySchema,
  sessionConfigSchema as logSessionConfigSchema,
  stakworkRunSummarySchema,
} from "../../log/schemas/routes.js";
import {
  gitreeRepoQuerySchema,
  gitreeFeatureParamsSchema,
  gitreeGetFeatureQuerySchema,
  gitreeAnalyzeChangesQuerySchema,
  gitreeAnalyzeCluesQuerySchema,
  gitreeGetCommitParamsSchema,
  gitreeFeatureFilesQuerySchema,
  gitreeSummarizeFeatureParamsSchema,
  gitreeLinkFilesQuerySchema,
  gitreeAllFeaturesGraphQuerySchema,
  gitreeRelevantFeaturesBodySchema,
  gitreeCreateFeatureBodySchema,
  gitreeListCluesQuerySchema,
  gitreeClueParamsSchema,
  gitreeLinkCluesQuerySchema,
  gitreeGetPrParamsSchema,
  gitreeProcessQuerySchema,
  gitreeSearchCluesBodySchema,
  gitreeSearchCluesQuerySchema,
  gitreeProvenanceBodySchema,
} from "../../gitree/schemas/routes.js";

test("graph schemas validate edge/search/map/query constraints", () => {
  const getNodes = getNodesQuerySchema.safeParse({
    node_type: "Function",
    concise: "false",
    ref_ids: "a,b",
    output: "json",
  });
  assert.equal(getNodes.success, true);

  const getNodesInvalid = getNodesQuerySchema.safeParse({ node_type: "InvalidType" });
  assert.equal(getNodesInvalid.success, false);

  assert.equal(
    postNodesBodySchema.safeParse({ node_type: "Class", ref_ids: ["r1"], output: "snippet" }).success,
    true
  );

  const edgesValid = getEdgesQuerySchema.safeParse({ edge_type: "CALLS", concise: "true" });
  assert.equal(edgesValid.success, true);

  const edgesInvalid = getEdgesQuerySchema.safeParse({ edge_type: "NotAnEdge" });
  assert.equal(edgesInvalid.success, false);

  const searchValid = searchQuerySchema.safeParse({
    query: "auth flow",
    node_types: "Function,Class",
    limit: "10",
  });
  assert.equal(searchValid.success, true);
  if (searchValid.success) {
    assert.deepEqual(searchValid.data.node_types, ["Function", "Class"]);
    assert.equal(searchValid.data.limit, 10);
  }

  const searchInvalid = searchQuerySchema.safeParse({ query: "", node_types: "Nope" });
  assert.equal(searchInvalid.success, false);

  assert.equal(refIdQuerySchema.safeParse({ ref_id: "abc" }).success, true);
  assert.equal(refIdQuerySchema.safeParse({ ref_id: "" }).success, false);

  assert.equal(workflowQuerySchema.safeParse({ ref_id: "r1", concise: "true" }).success, true);

  assert.equal(repoMapQuerySchema.safeParse({ node_type: "Repository", ref_id: "r1" }).success, true);
  assert.equal(repoMapQuerySchema.safeParse({ node_type: "NoType" }).success, false);

  const mapInvalid = mapQuerySchema.safeParse({ depth: "2" });
  assert.equal(mapInvalid.success, false);
  const mapValid = mapQuerySchema.safeParse({ node_type: "Function", name: "run" });
  assert.equal(mapValid.success, true);

  assert.equal(shortestPathQuerySchema.safeParse({ start_ref_id: "a", end_ref_id: "b" }).success, true);

  const graphValid = graphQuerySchema.safeParse({
    concise: "1",
    edges: "false",
    node_types: "Function,Class",
    limit: "25",
    since: "1700000000",
  });
  assert.equal(graphValid.success, true);
  if (graphValid.success) {
    assert.equal(graphValid.data.concise, true);
    assert.equal(graphValid.data.edges, false);
    assert.equal(graphValid.data.limit, 25);
    assert.equal(graphValid.data.since, 1700000000);
  }
});

test("knowledge schemas enforce required fields and refinement", () => {
  assert.equal(exploreQuerySchema.safeParse({ prompt: "explore this" }).success, true);
  assert.equal(exploreQuerySchema.safeParse({ prompt: "" }).success, false);

  const understand = understandQuerySchema.safeParse({
    question: "How does auth work?",
    threshold: "0.5",
    provider: "openrouter",
  });
  assert.equal(understand.success, true);
  if (understand.success) {
    assert.equal(understand.data.threshold, 0.5);
  }

  assert.equal(seedUnderstandingQuerySchema.safeParse({ budget: "3" }).success, true);

  assert.equal(
    askQuerySchema.safeParse({ question: "What changed?", forceRefresh: "True" }).success,
    true
  );

  assert.equal(getLearningsQuerySchema.safeParse({}).success, true);

  assert.equal(
    createPullRequestBodySchema.safeParse({ name: "feat", docs: "docs", number: "42" }).success,
    true
  );

  const learningMissingContext = createLearningBodySchema.safeParse({
    question: "Q",
    answer: "A",
  });
  assert.equal(learningMissingContext.success, false);

  assert.equal(
    createLearningBodySchema.safeParse({
      question: "Q",
      answer: "A",
      context: "repo context",
    }).success,
    true
  );

  assert.equal(
    createLearningBodySchema.safeParse({
      question: "Q",
      answer: "A",
      featureIds: ["f1"],
    }).success,
    true
  );

  assert.equal(seedStoriesQuerySchema.safeParse({ prompt: "stories", budget: "2" }).success, true);
  assert.equal(reconnectQuerySchema.safeParse({ provider: "openrouter" }).success, true);
});

test("gitsee and services schemas validate endpoint contracts", () => {
  const gitseeValid = gitseeBodySchema.safeParse({
    owner: "stakwork",
    repo: "hive",
    data: ["repo_info", "contributors"],
    useCache: "0",
  });
  assert.equal(gitseeValid.success, true);
  if (gitseeValid.success) {
    assert.equal(gitseeValid.data.useCache, false);
  }

  assert.equal(gitseeBodySchema.safeParse({ owner: "stakwork", repo: "hive", data: [] }).success, false);
  assert.equal(gitseeEventsParamsSchema.safeParse({ owner: "o", repo: "r" }).success, true);
  assert.equal(gitseeAgentQuerySchema.safeParse({ owner: "o", repo: "r", prompt: "summarize" }).success, true);
  assert.equal(gitseeAgentQuerySchema.safeParse({ owner: "o", repo: "r" }).success, false);
  assert.equal(gitseeServicesQuerySchema.safeParse({ owner: "o", repo: "r", username: "u" }).success, true);
  assert.equal(requestIdQuerySchema.safeParse({ request_id: "req-1" }).success, true);

  assert.equal(getServicesQuerySchema.safeParse({ clone: "true" }).success, false);
  assert.equal(
    getServicesQuerySchema.safeParse({ clone: "true", repo_url: "https://github.com/org/repo" }).success,
    true
  );

  const inventory = mocksInventoryQuerySchema.safeParse({ limit: "20", offset: "5" });
  assert.equal(inventory.success, true);
  if (inventory.success) {
    assert.equal(inventory.data.limit, 20);
    assert.equal(inventory.data.offset, 5);
  }
});

test("repo and log schemas validate required body/query fields", () => {
  assert.equal(
    mcpServerSchema.safeParse({
      name: "graph",
      url: "https://example.com/mcp",
      headers: { authorization: "Bearer token" },
      toolFilter: ["search"],
    }).success,
    true
  );
  assert.equal(mcpServerSchema.safeParse({ name: "", url: "" }).success, false);

  assert.equal(repoSessionConfigSchema.safeParse({ maxToolResultChars: 100 }).success, true);
  assert.equal(repoSessionConfigSchema.safeParse({ maxToolResultChars: 0 }).success, false);

  assert.equal(getLeaksQuerySchema.safeParse({ repo_url: "https://github.com/org/repo" }).success, true);
  assert.equal(getLeaksQuerySchema.safeParse({}).success, false);

  const repoAgentValid = repoAgentBodySchema.safeParse({
    repo_url: "https://github.com/org/repo",
    prompt: "Explain architecture",
    toolsConfig: { dryRun: true, model: "x", note: null },
  });
  assert.equal(repoAgentValid.success, true);

  assert.equal(
    repoAgentBodySchema.safeParse({
      repo_url: "https://github.com/org/repo",
      prompt: "Explain architecture",
      sessionConfig: { maxToolResultLines: 0 },
    }).success,
    false
  );

  assert.equal(logsAgentBodySchema.safeParse({ prompt: "inspect failed run" }).success, true);
  assert.equal(logsAgentBodySchema.safeParse({ prompt: "x", stakworkRuns: [{ projectId: "1" }] }).success, false);

  assert.equal(getAgentSessionQuerySchema.safeParse({ session_id: "s1" }).success, true);

  assert.equal(logSessionConfigSchema.safeParse({ maxToolResultLines: 50 }).success, true);
  assert.equal(logSessionConfigSchema.safeParse({ maxToolResultLines: -1 }).success, false);

  assert.equal(
    stakworkRunSummarySchema.safeParse({
      projectId: 1,
      type: "ingest",
      status: "done",
      createdAt: new Date().toISOString(),
      agentLogs: [{ agent: "planner", url: "https://example.com/log" }],
    }).success,
    true
  );
});

test("gitree schemas parse/coerce values and enforce required fields", () => {
  const process = gitreeProcessQuerySchema.safeParse({
    owner: "stakwork",
    repo: "hive",
    summarize: "true",
    link: "False",
    analyze_clues: "1",
  });
  assert.equal(process.success, true);
  if (process.success) {
    assert.equal(process.data.summarize, true);
    assert.equal(process.data.link, false);
    assert.equal(process.data.analyze_clues, true);
  }

  assert.equal(gitreeRepoQuerySchema.safeParse({ repo: "hive" }).success, true);
  assert.equal(gitreeFeatureParamsSchema.safeParse({ id: "feat-1" }).success, true);
  assert.equal(gitreeGetFeatureQuerySchema.safeParse({ include: "files", repo: "hive" }).success, true);

  const prParams = gitreeGetPrParamsSchema.safeParse({ number: "7" });
  assert.equal(prParams.success, true);
  if (prParams.success) {
    assert.equal(prParams.data.number, 7);
  }
  assert.equal(gitreeGetPrParamsSchema.safeParse({ number: "0" }).success, false);

  assert.equal(gitreeGetCommitParamsSchema.safeParse({ sha: "abc123" }).success, true);
  assert.equal(gitreeFeatureFilesQuerySchema.safeParse({ expand: "true", output: "json" }).success, true);
  assert.equal(gitreeSummarizeFeatureParamsSchema.safeParse({ id: "feat-1" }).success, true);
  assert.equal(gitreeLinkFilesQuerySchema.safeParse({ feature_id: "feat-1" }).success, true);

  const allFeaturesGraph = gitreeAllFeaturesGraphQuerySchema.safeParse({
    repo: "hive",
    concise: "true",
    limit: "100",
    depth: "2",
    node_types: "Feature,File",
  });
  assert.equal(allFeaturesGraph.success, true);
  if (allFeaturesGraph.success) {
    assert.equal(allFeaturesGraph.data.concise, true);
  }

  assert.equal(gitreeRelevantFeaturesBodySchema.safeParse({ prompt: "user auth feature" }).success, true);

  assert.equal(gitreeCreateFeatureBodySchema.safeParse({ prompt: "p", name: "n", owner: "o", repo: "r" }).success, true);
  assert.equal(gitreeCreateFeatureBodySchema.safeParse({ prompt: "p", name: "n", owner: "o" }).success, false);

  assert.equal(gitreeAnalyzeCluesQuerySchema.safeParse({ owner: "o", repo: "r", force: "1" }).success, true);
  assert.equal(gitreeAnalyzeChangesQuerySchema.safeParse({ owner: "o", repo: "r", force: "0" }).success, true);

  assert.equal(gitreeListCluesQuerySchema.safeParse({ repo: "hive", feature_id: "f1" }).success, true);
  assert.equal(gitreeClueParamsSchema.safeParse({ id: "clue-1" }).success, true);
  assert.equal(gitreeLinkCluesQuerySchema.safeParse({ owner: "o", repo: "r", force: "false" }).success, true);

  assert.equal(gitreeSearchCluesBodySchema.safeParse({ query: "auth", limit: 5 }).success, true);
  assert.equal(gitreeSearchCluesBodySchema.safeParse({ query: "auth", limit: 0 }).success, false);
  assert.equal(gitreeSearchCluesQuerySchema.safeParse({ repo: "hive" }).success, true);

  assert.equal(gitreeProvenanceBodySchema.safeParse({ conceptIds: ["c1", "c2"] }).success, true);
  assert.equal(gitreeProvenanceBodySchema.safeParse({ conceptIds: [] }).success, true);
});
