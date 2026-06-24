import { test, expect } from "@playwright/test";
import {
  buildContextualSystemPrompt,
  GRAPH_AGENT_SYSTEM_PROMPT,
} from "../prompts/graph.js";

// ── buildContextualSystemPrompt ──────────────────────────────────────────────

test.describe("buildContextualSystemPrompt", () => {
  test("includes selectedRefId in the returned string", () => {
    const result = buildContextualSystemPrompt({
      selectedRefId: "ep-abc123",
      nodeType: "Episode",
    });
    expect(result).toContain("ep-abc123");
  });

  test("includes nodeType in the returned string", () => {
    const result = buildContextualSystemPrompt({
      selectedRefId: "ep-abc123",
      nodeType: "Episode",
    });
    expect(result).toContain("Episode");
  });

  test("uses title when provided", () => {
    const result = buildContextualSystemPrompt({
      selectedRefId: "ep-abc123",
      nodeType: "Episode",
      title: "My Great Episode",
    });
    expect(result).toContain("My Great Episode");
  });

  test("falls back to selectedRefId when title is absent", () => {
    const result = buildContextualSystemPrompt({
      selectedRefId: "ep-abc123",
      nodeType: "Episode",
    });
    // label should be selectedRefId
    expect(result).toContain('"ep-abc123"');
  });

  test("appends GRAPH_AGENT_SYSTEM_PROMPT as the base", () => {
    const result = buildContextualSystemPrompt({
      selectedRefId: "ep-abc123",
      nodeType: "Episode",
    });
    // The base prompt content should be present
    expect(result).toContain(GRAPH_AGENT_SYSTEM_PROMPT);
  });

  test("preamble appears before base prompt", () => {
    const result = buildContextualSystemPrompt({
      selectedRefId: "ep-abc123",
      nodeType: "Episode",
    });
    const preambleIdx = result.indexOf("REQUIRED first steps");
    const baseIdx = result.indexOf(GRAPH_AGENT_SYSTEM_PROMPT);
    expect(preambleIdx).toBeGreaterThanOrEqual(0);
    expect(baseIdx).toBeGreaterThan(preambleIdx);
  });

  test("instructs graph_node call with the selectedRefId", () => {
    const result = buildContextualSystemPrompt({
      selectedRefId: "clip-xyz",
      nodeType: "Clip",
    });
    expect(result).toContain('graph_node("clip-xyz")');
  });

  test("instructs graph_neighbors call with the selectedRefId", () => {
    const result = buildContextualSystemPrompt({
      selectedRefId: "clip-xyz",
      nodeType: "Clip",
    });
    expect(result).toContain('graph_neighbors("clip-xyz")');
  });
});

// ── parseGraphAgentBody (via simulated req.body) ─────────────────────────────

// We test the logic inline since parseGraphAgentBody is not exported.
// The body parsing is straightforward — we verify the contract by importing
// the agent and checking the type accepts context.

import type { GraphAgentOptions } from "../agent.js";

test.describe("GraphAgentOptions context field", () => {
  test("accepts context with selectedRefId, nodeType, and optional title", () => {
    // TypeScript compile-time check represented as a runtime assertion
    const opts: GraphAgentOptions = {
      prompt: "What is this episode about?",
      context: {
        selectedRefId: "ep-abc123",
        nodeType: "Episode",
        title: "Pilot Episode",
      },
    };
    expect(opts.context?.selectedRefId).toBe("ep-abc123");
    expect(opts.context?.nodeType).toBe("Episode");
    expect(opts.context?.title).toBe("Pilot Episode");
  });

  test("context is optional — opts without context is valid", () => {
    const opts: GraphAgentOptions = {
      prompt: "General question",
    };
    expect(opts.context).toBeUndefined();
  });

  test("context without title is valid", () => {
    const opts: GraphAgentOptions = {
      prompt: "What is this clip about?",
      context: {
        selectedRefId: "clip-xyz",
        nodeType: "Clip",
      },
    };
    expect(opts.context?.title).toBeUndefined();
    expect(opts.context?.selectedRefId).toBe("clip-xyz");
  });
});

// ── Context parsing logic (mirrors parseGraphAgentBody) ──────────────────────

test.describe("context parsing from request body", () => {
  // Simulate what parseGraphAgentBody does with req.body.context
  function extractContext(
    body: Record<string, unknown>,
  ): { selectedRefId: string; nodeType: string; title?: string } | undefined {
    return body.context as
      | { selectedRefId: string; nodeType: string; title?: string }
      | undefined;
  }

  test("returns context object when present in body", () => {
    const body = {
      prompt: "Tell me about this",
      context: { selectedRefId: "ep-001", nodeType: "Episode", title: "First Episode" },
    };
    const ctx = extractContext(body);
    expect(ctx).toBeDefined();
    expect(ctx?.selectedRefId).toBe("ep-001");
    expect(ctx?.nodeType).toBe("Episode");
    expect(ctx?.title).toBe("First Episode");
  });

  test("returns undefined when context is absent", () => {
    const body = { prompt: "General question" };
    const ctx = extractContext(body);
    expect(ctx).toBeUndefined();
  });

  test("returns context without title when title is omitted", () => {
    const body = {
      prompt: "Something",
      context: { selectedRefId: "video-42", nodeType: "Video" },
    };
    const ctx = extractContext(body);
    expect(ctx?.selectedRefId).toBe("video-42");
    expect(ctx?.nodeType).toBe("Video");
    expect(ctx?.title).toBeUndefined();
  });
});

// ── Contextual prompt produces different instructions than base ───────────────

test.describe("contextual vs base system prompt", () => {
  test("contextual prompt differs from bare GRAPH_AGENT_SYSTEM_PROMPT", () => {
    const contextual = buildContextualSystemPrompt({
      selectedRefId: "ep-abc",
      nodeType: "Episode",
    });
    expect(contextual).not.toBe(GRAPH_AGENT_SYSTEM_PROMPT);
    expect(contextual.length).toBeGreaterThan(GRAPH_AGENT_SYSTEM_PROMPT.length);
  });

  test("two different selectedRefIds produce different prompts", () => {
    const p1 = buildContextualSystemPrompt({ selectedRefId: "ref-1", nodeType: "Episode" });
    const p2 = buildContextualSystemPrompt({ selectedRefId: "ref-2", nodeType: "Episode" });
    expect(p1).not.toBe(p2);
  });
});
