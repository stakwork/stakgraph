/**
 * Unit tests for stakwork_run_step — workflow_version_id passthrough.
 *
 * Strategy: intercept axios.post (used by stakworkPost) via module-level
 * patching so we can assert on the exact body sent to the endpoint without
 * making real HTTP calls.  axios.get is stubbed to return a terminal status
 * so the polling loop resolves immediately.
 */
import { test, expect } from "../testkit.js";
import { registerStakworkTools } from "../repo/toolsStakwork.js";
import type { Tool } from "ai";
import axios from "axios";

// ---------------------------------------------------------------------------
// Axios stub helpers
// ---------------------------------------------------------------------------

type AxiosPostCall = { url: string; data: unknown };
type AxiosGetCall = { url: string };

let postCalls: AxiosPostCall[] = [];
let getCalls: AxiosGetCall[] = [];

// Saved originals
const _origPost = axios.post.bind(axios);
const _origGet = axios.get.bind(axios);

/** Reset captured calls and install stubs. */
function installStubs(
  postResponse: object,
  getResponse: object = { data: JSON.stringify({ status: "completed" }) },
) {
  postCalls = [];
  getCalls = [];

  (axios as any).post = async (url: string, data: unknown) => {
    postCalls.push({ url, data });
    return {
      status: 200,
      data: JSON.stringify(postResponse),
    };
  };

  (axios as any).get = async (url: string) => {
    getCalls.push({ url });
    return {
      status: 200,
      data: JSON.stringify(getResponse),
    };
  };
}

/** Restore real axios methods. */
function restoreStubs() {
  (axios as any).post = _origPost;
  (axios as any).get = _origGet;
}

// ---------------------------------------------------------------------------
// Helper: build a tools map with stakwork_run_step registered
// ---------------------------------------------------------------------------

function makeTools(baseUrl = "https://test.stakwork.example/api/v1"): Record<string, Tool<any, any>> {
  const allTools: Record<string, Tool<any, any>> = {};
  registerStakworkTools(allTools, {
    apiKey: "test-key",
    baseUrl,
    runStep: true,
  });
  return allTools;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("stakwork_run_step — workflow_version_id passthrough", () => {
  test.afterEach(() => {
    restoreStubs();
  });

  // 1. Omitted: body unchanged (no workflow_version_id key added)
  test("omitting workflow_version_id keeps body unchanged (published-version fallback)", async () => {
    installStubs({ project_id: 42 });

    const tools = makeTools();
    await tools.stakwork_run_step.execute({
      step_id: "my_step",
      workflow_id: 100,
      params: { ancestor: { key: "val" } },
      wait_seconds: 0,
    });

    expect(postCalls.length).toBe(1);
    const body = postCalls[0].data as any;
    expect(body).not.toHaveProperty("workflow_version_id");
    expect(body.step_id).toBe("my_step");
    expect(body.params).toEqual({ ancestor: { key: "val" } });
  });

  // 2. Provided: key appears in JSON body only, never in URL path
  test("providing workflow_version_id adds it to POST body, not the URL path", async () => {
    installStubs({ project_id: 43 });

    const tools = makeTools("https://test.stakwork.example/api/v1");
    await tools.stakwork_run_step.execute({
      step_id: "my_step",
      workflow_id: 200,
      workflow_version_id: 77,
      params: { ancestor: { key: "val" } },
      wait_seconds: 0,
    });

    expect(postCalls.length).toBe(1);

    // Body must contain workflow_version_id
    const body = postCalls[0].data as any;
    expect(body.workflow_version_id).toBe(77);

    // URL path must remain /workflows/:id/run_step_from_template (unchanged)
    const url = postCalls[0].url;
    expect(url).toMatch(/\/workflows\/200\/run_step_from_template$/);
    expect(url).not.toContain("workflow_version_id");
  });

  // 3. Resume exclusivity: project_id resume never emits workflow_version_id
  test("project_id resume path never sends workflow_version_id", async () => {
    // No POST should happen in a pure resume; stub GET to return terminal state
    installStubs({}, { status: "completed" });
    // Make GET for IO also return something reasonable
    (axios as any).get = async (url: string) => {
      getCalls.push({ url });
      if (url.includes("/io")) {
        return { status: 200, data: JSON.stringify({ inputs: {}, outputs: {} }) };
      }
      return { status: 200, data: JSON.stringify({ status: "completed" }) };
    };

    const tools = makeTools();
    await tools.stakwork_run_step.execute({
      step_id: "my_step",
      project_id: 999,
      workflow_version_id: 77,
      wait_seconds: 0,
    });

    // Resume path never POSTs — it only polls GET /projects/:id/status
    expect(postCalls.length).toBe(0);
    expect(getCalls.some((c) => c.url.includes("/projects/999/status"))).toBe(true);
  });

  // 4. Probe branch: probe body also receives workflow_version_id when provided
  test("probe launch (no params) includes workflow_version_id when provided", async () => {
    // Return a discovery-style error (success: false) so the tool doesn't poll
    installStubs({
      success: false,
      errors: "Missing required ancestor keys: foo.bar",
    });

    const tools = makeTools();
    const result = await tools.stakwork_run_step.execute({
      step_id: "my_step",
      workflow_id: 300,
      workflow_version_id: 55,
      // No params → probe path
      wait_seconds: 0,
    });

    expect(postCalls.length).toBe(1);

    // Probe body uses synthetic _probe params
    const body = postCalls[0].data as any;
    expect(body.params).toEqual({ _probe: { _: "_" } });

    // workflow_version_id must be present in probe body too
    expect(body.workflow_version_id).toBe(55);

    // URL unchanged
    const url = postCalls[0].url;
    expect(url).toMatch(/\/workflows\/300\/run_step_from_template$/);

    // Result should be the discovery/error response
    const parsed = JSON.parse(result as string);
    expect(parsed.discovery).toBe(true);
    expect(parsed.launched).toBe(false);
  });

  // 5. Schema: workflow_version_id is registered as optional number
  test("inputSchema includes workflow_version_id as optional number", () => {
    const tools = makeTools();
    const schema = (tools.stakwork_run_step as any).inputSchema;
    expect(schema).toBeDefined();

    // Parse with version omitted (should succeed)
    const withoutVersion = schema.safeParse({ step_id: "s", workflow_id: 1 });
    expect(withoutVersion.success).toBe(true);
    expect(withoutVersion.data).not.toHaveProperty("workflow_version_id");

    // Parse with version provided as number (should succeed)
    const withVersion = schema.safeParse({ step_id: "s", workflow_id: 1, workflow_version_id: 42 });
    expect(withVersion.success).toBe(true);
    expect(withVersion.data?.workflow_version_id).toBe(42);

    // Parse with version provided as string (should fail)
    const withStringVersion = schema.safeParse({ step_id: "s", workflow_id: 1, workflow_version_id: "42" });
    expect(withStringVersion.success).toBe(false);
  });
});
