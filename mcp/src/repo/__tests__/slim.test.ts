import { test, expect } from "@playwright/test";
import { ModelMessage } from "ai";
import { slimToolResult, slimOldToolResults } from "../utils.js";

const SEARCH_RESULT = JSON.stringify([
  { name: "acceptAndPayBounty", node_type: "Function", file: "src/lib/bounty/payment-service.ts", lines: "63-451", ref_id: "abc123", description: "Accepts a proof submission and initiates Lightning payment to the bounty hunter" },
  { name: "POST", node_type: "Endpoint", file: "src/app/api/bounties/[id]/accept-and-pay/route.ts", lines: "1-80", ref_id: "def456", description: "HTTP endpoint for accepting proof and paying a bounty" },
]);

const CODE_RESULT = `export async function acceptAndPayBounty(params: AcceptAndPayParams): Promise<AcceptAndPayResult> {
  const { bountyId, proofId, feedback, memo, adminPubkey, requestId } = params;
  const startTime = Date.now();

  logInfo("Starting accept-and-pay flow", { bountyId, proofId, adminPubkey });

  // Validate bounty status
  const bounty = await db.bounty.findUnique({ where: { id: bountyId } });
  if (!bounty) throw new Error("Bounty not found");
  if (bounty.status !== BountyStatus.IN_REVIEW && bounty.status !== BountyStatus.ASSIGNED) {
    throw new Error("Invalid status");
  }

  // Process payment
  const keysendResult = await lightningClient.keysend({ dest: bounty.assigneePubkey, amount: bounty.amount });
  return { success: true, paymentHash: keysendResult.tag };
}`;

const BASH_RESULT = `total 120
drwxr-xr-x  15 user  staff   480 Jun 12 10:00 .
drwxr-xr-x   8 user  staff   256 Jun 12 09:00 ..
-rw-r--r--   1 user  staff  1234 Jun 12 10:00 agent.ts
-rw-r--r--   1 user  staff  5678 Jun 12 10:00 bash.ts
-rw-r--r--   1 user  staff  2345 Jun 12 10:00 clone.ts
-rw-r--r--   1 user  staff  3456 Jun 12 10:00 events.ts
-rw-r--r--   1 user  staff  4567 Jun 12 10:00 index.ts
-rw-r--r--   1 user  staff  1111 Jun 12 10:00 services.ts
-rw-r--r--   1 user  staff  2222 Jun 12 10:00 session.ts
-rw-r--r--   1 user  staff  3333 Jun 12 10:00 tools.ts
-rw-r--r--   1 user  staff  4444 Jun 12 10:00 utils.ts`;

function toolTurn(
  toolName: string,
  output: string,
  id?: string
): ModelMessage[] {
  const callId = id || `call_${toolName}_${Math.random().toString(36).slice(2, 6)}`;
  return [
    {
      role: "assistant",
      content: [{ type: "tool-call", toolCallId: callId, toolName, input: {} }],
    } as ModelMessage,
    {
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: callId,
          toolName,
          output: { type: "text", value: output },
        },
      ],
    } as ModelMessage,
  ];
}

function userMsg(text: string): ModelMessage {
  return { role: "user", content: text };
}

function assistantMsg(text: string): ModelMessage {
  return { role: "assistant", content: text };
}

test.describe("slimToolResult", () => {
  test("slims search results by stripping descriptions and lines", () => {
    const result = slimToolResult("stakgraph_search", SEARCH_RESULT);
    expect(result).toContain("[SLIMMED");
    expect(result).toContain("ref_id");
    expect(result).not.toContain("Accepts a proof submission");
    expect(result).not.toContain("63-451");
    const parsed = JSON.parse(result.split("\n").slice(1).join("\n"));
    expect(parsed).toHaveLength(2);
    expect(parsed[0].name).toBe("acceptAndPayBounty");
    expect(parsed[0].ref_id).toBe("abc123");
    expect(parsed[0].description).toBeUndefined();
  });

  test("slims code results to signature only", () => {
    const result = slimToolResult("stakgraph_code", CODE_RESULT);
    expect(result).toContain("[SLIMMED");
    expect(result).toContain("acceptAndPayBounty");
    expect(result).not.toContain("keysendResult");
    expect(result).toContain("more lines");
  });

  test("slims bash results to head/tail", () => {
    const result = slimToolResult("bash", BASH_RESULT);
    expect(result).toContain("[SLIMMED");
    expect(result).toContain("total 120");
    expect(result).toContain("utils.ts");
    expect(result).toContain("lines omitted");
  });

  test("nukes repo_overview entirely", () => {
    const result = slimToolResult("repo_overview", "big tree output here\nline2\nline3");
    expect(result).toContain("[SLIMMED");
    expect(result).toContain("repo_overview");
    expect(result).not.toContain("big tree");
  });

  test("leaves short content unchanged", () => {
    const short = "just a few lines";
    expect(slimToolResult("bash", short)).toBe(short);
  });

  test("leaves unknown tools unchanged", () => {
    expect(slimToolResult("some_custom_tool", "output")).toBe("output");
  });

  test("applies same logic to vector_search as stakgraph_search", () => {
    const result = slimToolResult("vector_search", SEARCH_RESULT);
    expect(result).toContain("[SLIMMED");
  });

  test("handles malformed JSON in search gracefully", () => {
    const result = slimToolResult("stakgraph_search", "not json at all");
    expect(result).toBe("not json at all");
  });
});

test.describe("slimOldToolResults", () => {
  test("no-op when fewer than RECENT_TOOL_RESULTS tool results", () => {
    const messages: ModelMessage[] = [
      userMsg("q1"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT),
      assistantMsg("a1"),
    ];
    const result = slimOldToolResults(messages);
    expect(result).toBe(messages);
  });

  test("slims old results, keeps recent 6 intact", () => {
    const messages: ModelMessage[] = [
      userMsg("q1"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "call_1"),
      ...toolTurn("stakgraph_code", CODE_RESULT, "call_2"),
      ...toolTurn("bash", BASH_RESULT, "call_3"),
      ...toolTurn("repo_overview", "big\ntree\noutput", "call_4"),
      assistantMsg("a1"),
      userMsg("q2"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "call_5"),
      ...toolTurn("stakgraph_code", CODE_RESULT, "call_6"),
      ...toolTurn("bash", BASH_RESULT, "call_7"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "call_8"),
      assistantMsg("a2"),
    ];
    const result = slimOldToolResults(messages);
    expect(result).not.toBe(messages);

    const getOutput = (msgs: ModelMessage[], callId: string) => {
      for (const msg of msgs) {
        if (msg.role !== "tool" || !Array.isArray(msg.content)) continue;
        for (const part of msg.content as any[]) {
          if (part.toolCallId === callId) return part.output.value;
        }
      }
      return undefined;
    };

    expect(getOutput(result, "call_1")).toContain("[SLIMMED");
    expect(getOutput(result, "call_2")).toContain("[SLIMMED");

    expect(getOutput(result, "call_5")).toBe(SEARCH_RESULT);
    expect(getOutput(result, "call_6")).toBe(CODE_RESULT);
    expect(getOutput(result, "call_7")).toBe(BASH_RESULT);
    expect(getOutput(result, "call_8")).toBe(SEARCH_RESULT);
  });

  test("preserves message count and structure", () => {
    const messages: ModelMessage[] = [
      userMsg("q"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c1"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c2"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c3"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c4"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c5"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c6"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c7"),
      assistantMsg("a"),
    ];
    const result = slimOldToolResults(messages);
    expect(result.length).toBe(messages.length);
    for (let i = 0; i < result.length; i++) {
      expect(result[i].role).toBe(messages[i].role);
    }
  });

  test("does not mutate original messages", () => {
    const messages: ModelMessage[] = [
      userMsg("q"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c1"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c2"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c3"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c4"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c5"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c6"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c7"),
      assistantMsg("a"),
    ];
    slimOldToolResults(messages);
    expect((messages[2] as any).content[0].output.value).toBe(SEARCH_RESULT);
  });

  test("protects error outputs from slimming", () => {
    const messages: ModelMessage[] = [
      userMsg("q"),
      ...toolTurn("bash", "Error: command not found\n" + BASH_RESULT, "c1"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c2"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c3"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c4"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c5"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c6"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c7"),
      assistantMsg("a"),
    ];
    const result = slimOldToolResults(messages);
    const errOutput = (result[2] as any).content[0].output.value;
    expect(errOutput).toContain("Error: command not found");
  });

  test("is idempotent", () => {
    const messages: ModelMessage[] = [
      userMsg("q"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c1"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c2"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c3"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c4"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c5"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c6"),
      ...toolTurn("stakgraph_search", SEARCH_RESULT, "c7"),
      assistantMsg("a"),
    ];
    const once = slimOldToolResults(messages);
    const twice = slimOldToolResults(once);
    expect(twice).toBe(once);
  });
});
