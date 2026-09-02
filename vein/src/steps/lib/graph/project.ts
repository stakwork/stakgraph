import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText } from "./_shared.js";

export default defineStep({
  type: "graph/project",
  description:
    "Project vein's own run and chat history into the knowledge graph (VeinRun / VeinAgentSession / " +
    "VeinToolCall / VeinChat / VeinTurn nodes with EXECUTED, IN_RUN, IN_SESSION, SPAWNED, IN_CHAT edges), " +
    "reading the raw logs under the server's data dir. Idempotent — re-run any time; settled runs are " +
    "skipped unless skipSettled is false. This is what makes 'which runs executed this version' and " +
    "'which chat launched this run' one-hop graph questions.",
  input: z.object({
    dataDir: z
      .string()
      .optional()
      .describe("Local data dir holding workflows/<name>/runs and chats/ (default: VEIN_WORKSPACE, else ./workspace)."),
    workflows: z
      .array(z.string())
      .optional()
      .describe("Workflows whose runs to project (default: every workflow the workspace lists)."),
    limitPerWorkflow: z.number().int().positive().optional().describe("Newest N runs per workflow (default: all)."),
    skipSettled: z.boolean().optional().describe("Skip runs already in the graph with a terminal status (default true)."),
    chats: z.boolean().optional().describe("Also project chats + turns (default true)."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const dataDir = cfg.dataDir ?? process.env["VEIN_WORKSPACE"] ?? "./workspace";
      const [{ FileRunStore }, { FileChatStore }, { FileWorkspaceStore }, { Neo4jWorkspaceStore }, { projectAll }, { graphWorkspaceRequested }] =
        await Promise.all([
          import("../../../store.js"),
          import("../../../chat-store.js"),
          import("../../../workspace.js"),
          import("../../../graph/workspace-store.js"),
          import("../../../graph/projector.js"),
          import("../../../graph/wiring.js"),
        ]);
      const workflows =
        cfg.workflows ??
        (await (graphWorkspaceRequested() ? new Neo4jWorkspaceStore(b) : new FileWorkspaceStore(dataDir)).listWorkflows()).map((w) => w.name);
      const report = await projectAll(
        b,
        { store: new FileRunStore(dataDir), chatStore: cfg.chats === false ? undefined : new FileChatStore(dataDir), workflows },
        { limitPerWorkflow: cfg.limitPerWorkflow, skipSettled: cfg.skipSettled },
      );
      return { dataDir, workflows, ...report };
    } catch (e) {
      return errText("graph/project", e);
    }
  },
});
