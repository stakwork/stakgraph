/**
 * Recovering concept reads from stored transcripts.
 *
 * The backfill reads tool-call INPUTS rather than results, which makes it
 * robust to `truncateToolResults` but means it can't tell a Concept ref_id
 * from any other node's. Catalog membership is the only thing standing between
 * "recovered the read log" and "invented one", so that's what these cover.
 */

import { test, expect } from "../../testkit.js";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import type { ModelMessage } from "ai";

const tmpSessionsDir = path.join(os.tmpdir(), `test-backfill-${randomUUID()}`);
fs.mkdirSync(tmpSessionsDir, { recursive: true });
process.env.SESSIONS_DIR = tmpSessionsDir;
process.env.NO_DB = "true";

// Must come after the env override — session.ts captures SESSIONS_DIR at
// module load, and this module reaches it. A static import here would point
// the sweep at the real .sessions directory, where it would write sidecars
// and a marker into live data.
const { conceptReadsFromTranscript, confirmConceptReads } = await import(
  "../conceptBackfill.js"
);

/** An assistant turn carrying tool calls, as stored in the session JSONL. */
function assistantCalls(
  calls: { toolName: string; input: Record<string, unknown> }[],
): ModelMessage {
  return {
    role: "assistant",
    content: calls.map((c, i) => ({
      type: "tool-call",
      toolCallId: `toolu_${i}`,
      toolName: c.toolName,
      input: c.input,
    })),
  } as ModelMessage;
}

const catalog = [
  { id: "c_4f2a", ref_id: "ref-abc", name: "auth-session-lifecycle", repo: "stakwork/hive" },
  { id: "c_91bd", ref_id: "ref-def", name: "stakwork-workflow-runs", repo: "stakwork/hive" },
];

test.describe("reading concept reads out of a transcript", () => {
  test("picks up both tool paths, in call order", () => {
    const reads = conceptReadsFromTranscript([
      { role: "user", content: "how does auth work?" },
      assistantCalls([
        { toolName: "graph_get", input: { ref_id: "ref-def" } },
        { toolName: "learn_concept", input: { concept_id: "c_4f2a" } },
      ]),
    ]);
    expect(reads).toHaveLength(2);
    expect(reads[0].ref_id).toBe("ref-def");
    expect(reads[0].via).toBe("graph_get");
    expect(reads[1].id).toBe("c_4f2a");
  });

  test("ignores tools that don't return a concept body", () => {
    const reads = conceptReadsFromTranscript([
      assistantCalls([
        { toolName: "graph_search", input: { q: "auth" } },
        { toolName: "list_concepts", input: {} },
        { toolName: "stakgraph_code", input: { ref_id: "ref-abc" } },
      ]),
    ]);
    expect(reads).toHaveLength(0);
  });

  test("survives a transcript with no tool calls at all", () => {
    expect(
      conceptReadsFromTranscript([
        { role: "user", content: "hi" },
        { role: "assistant", content: "hello" },
      ]),
    ).toHaveLength(0);
  });
});

test.describe("confirming reads against the catalog", () => {
  test("drops a graph_get on a node that isn't a Concept", () => {
    // The input alone looks identical to a concept read — only the catalog
    // says otherwise. Recording this would invent a read that never happened.
    const confirmed = confirmConceptReads(
      [
        { ref_id: "ref-abc", via: "graph_get" },
        { ref_id: "ref-some-function", via: "graph_get" },
      ],
      catalog,
    );
    expect(confirmed).toHaveLength(1);
    expect(confirmed[0].ref_id).toBe("ref-abc");
  });

  test("fills in the identifiers and name the transcript didn't carry", () => {
    const confirmed = confirmConceptReads([{ ref_id: "ref-def", via: "graph_get" }], catalog);
    expect(confirmed[0].id).toBe("c_91bd");
    expect(confirmed[0].name).toBe("stakwork-workflow-runs");
    expect(confirmed[0].repo).toBe("stakwork/hive");
  });

  test("the same concept read both ways collapses, as it does live", () => {
    const confirmed = confirmConceptReads(
      [
        { id: "c_4f2a", via: "learn_concept" },
        { ref_id: "ref-abc", via: "graph_get" },
      ],
      catalog,
    );
    expect(confirmed).toHaveLength(1);
  });

  test("an empty catalog recovers nothing rather than everything", () => {
    // A failed or empty lookup must not be read as "none of these are known,
    // so record them all" — that's the failure mode that would fabricate data.
    expect(confirmConceptReads([{ ref_id: "ref-abc", via: "graph_get" }], [])).toHaveLength(0);
  });
});

test.describe("the backfill sweep", () => {
  test("does nothing when there are no sessions", async () => {
    const { backfillConceptReads } = await import("../conceptBackfill.js");
    const result = await backfillConceptReads();
    expect(result).toEqual({ scanned: 0, written: 0, concepts: 0 });
  });

  test("leaves a session that already has a sidecar alone", async () => {
    const { mergeReflection, loadReflection } = await import("../session.js");
    const { backfillConceptReads } = await import("../conceptBackfill.js");
    const sessionId = `sess-${randomUUID()}`;

    // A live-collected sidecar, with a ranking worth protecting.
    mergeReflection(sessionId, {
      concepts: [{ ref_id: "ref-abc", name: "auth", rank: 1, evidence: "decisive" }],
    });
    fs.writeFileSync(
      path.join(tmpSessionsDir, `${sessionId}.jsonl`),
      JSON.stringify(assistantCalls([{ toolName: "graph_get", input: { ref_id: "ref-def" } }])) + "\n",
    );

    await backfillConceptReads();

    const saved = loadReflection(sessionId);
    expect(saved?.concepts).toHaveLength(1);
    expect(saved?.concepts[0].rank).toBe(1);
    expect(saved?.concepts[0].evidence).toBe("decisive");
  });
});
