import type { CanvasData, CanvasNode, CanvasEdge } from "system-canvas";

/**
 * Step data as returned by the API (serialized from the Flow object).
 * The API returns the raw step objects, not the typed Step interface.
 */
export interface StepData {
  id: string;
  type: string;
  config: Record<string, any>;
  options?: {
    retry?: { max: number; delayMs: number };
    onError?: StepData;
  };
}

export interface FlowData {
  name: string;
  steps: StepData[];
}

export interface RunEventData {
  ts: string;
  runId: string;
  path: string;
  type: string;
  stepType?: string;
  output?: any;
  error?: { message: string };
  durationMs?: number;
  iteration?: number;
}

// ── Node dimensions ────────────────────────────────────────────────────────

const NODE_W = 180;
const NODE_H = 60;
const GAP_Y = 40;
const GAP_X = 60;
const PARALLEL_GAP = 220;

// ── Step type → category colors ────────────────────────────────────────────

const STEP_CATEGORIES: Record<string, { fill: string; stroke: string }> = {
  http:     { fill: "rgba(6, 78, 59, 0.4)",   stroke: "#34d399" },
  log:      { fill: "rgba(30, 58, 138, 0.4)", stroke: "#60a5fa" },
  if:       { fill: "rgba(120, 53, 15, 0.4)", stroke: "#f59e0b" },
  loop:     { fill: "rgba(88, 28, 135, 0.4)", stroke: "#a78bfa" },
  parallel: { fill: "rgba(127, 29, 29, 0.4)", stroke: "#f87171" },
  subflow:  { fill: "rgba(21, 94, 117, 0.4)", stroke: "#22d3ee" },
  llm:      { fill: "rgba(76, 29, 149, 0.4)", stroke: "#c084fc" },
  default:  { fill: "rgba(38, 38, 38, 0.6)",  stroke: "#737373" },
};

function categoryFor(type: string) {
  return STEP_CATEGORIES[type] ?? STEP_CATEGORIES["default"]!;
}

// ── Status colors (for run overlay) ────────────────────────────────────────

function statusColor(status?: string): string | undefined {
  switch (status) {
    case "success": return "#22c55e";
    case "error":   return "#ef4444";
    case "running": return "#f59e0b";
    default:        return undefined;
  }
}

// ── Convert Flow → CanvasData ──────────────────────────────────────────────

interface LayoutResult {
  nodes: CanvasNode[];
  edges: CanvasEdge[];
  width: number;
  height: number;
}

function layoutSteps(
  steps: StepData[],
  startX: number,
  startY: number,
  pathPrefix: string,
  runEvents?: RunEventData[],
): LayoutResult {
  const nodes: CanvasNode[] = [];
  const edges: CanvasEdge[] = [];
  let curY = startY;
  let maxWidth = NODE_W;

  for (let i = 0; i < steps.length; i++) {
    const s = steps[i]!;
    const stepPath = `${pathPrefix}/${s.id}`;
    const cat = categoryFor(s.type);

    // Determine status from run events
    const stepEvents = runEvents?.filter((e) => e.path === stepPath) ?? [];
    const endEvent = stepEvents.find((e) => e.type === "step.end");
    const errorEvent = stepEvents.find((e) => e.type === "step.error");
    const status = errorEvent ? "error" : endEvent ? "success" : stepEvents.length > 0 ? "running" : undefined;
    const borderColor = statusColor(status);

    const label = `${s.id}\n${s.type}`;

    if (s.type === "parallel" && s.config.branches) {
      // Parallel: lay out branches side by side
      const branches = Object.entries(s.config.branches as Record<string, FlowData>);
      const branchLayouts: { name: string; layout: LayoutResult }[] = [];
      let branchX = startX;

      for (const [name, branchFlow] of branches) {
        const branchSteps = branchFlow.steps ?? [];
        const layout = layoutSteps(
          branchSteps,
          branchX,
          curY + NODE_H + GAP_Y,
          `${stepPath}.${name}`,
          runEvents,
        );
        branchLayouts.push({ name, layout });
        branchX += Math.max(layout.width, NODE_W) + GAP_X;
      }

      // Fork node
      const forkId = `${pathPrefix}/${s.id}__fork`;
      const totalBranchWidth = branchX - startX - GAP_X;
      const forkX = startX + totalBranchWidth / 2 - NODE_W / 2;
      nodes.push({
        id: forkId,
        type: "text",
        text: label,
        x: forkX,
        y: curY,
        width: NODE_W,
        height: NODE_H,
        color: borderColor ?? cat.stroke,
        customData: { stepId: s.id, stepIndex: i },
      });

      // Add branch nodes and connect
      for (const { name, layout } of branchLayouts) {
        nodes.push(...layout.nodes);
        edges.push(...layout.edges);

        // Edge from fork to first node in branch
        if (layout.nodes.length > 0) {
          edges.push({
            id: `${forkId}__to__${layout.nodes[0]!.id}`,
            fromNode: forkId,
            toNode: layout.nodes[0]!.id,
            label: name,
          });
        }
      }

      // Join node
      const joinId = `${pathPrefix}/${s.id}__join`;
      const maxBranchH = Math.max(...branchLayouts.map((b) => b.layout.height), 0);
      const joinY = curY + NODE_H + GAP_Y + maxBranchH + GAP_Y;
      nodes.push({
        id: joinId,
        type: "text",
        text: `${s.id}\njoin`,
        x: forkX,
        y: joinY,
        width: NODE_W,
        height: NODE_H / 2,
        color: cat.stroke,
      });

      // Edges from last node of each branch to join
      for (const { layout } of branchLayouts) {
        if (layout.nodes.length > 0) {
          const lastNode = layout.nodes[layout.nodes.length - 1]!;
          edges.push({
            id: `${lastNode.id}__to__${joinId}`,
            fromNode: lastNode.id,
            toNode: joinId,
          });
        }
      }

      maxWidth = Math.max(maxWidth, totalBranchWidth);
      curY = joinY + NODE_H / 2 + GAP_Y;

      // Connect to previous step
      if (i > 0) {
        const prevId = nodes.find(
          (n) => n.id === `${pathPrefix}/${steps[i - 1]!.id}` ||
                 n.id === `${pathPrefix}/${steps[i - 1]!.id}__join`,
        )?.id;
        if (prevId) {
          edges.push({
            id: `${prevId}__to__${forkId}`,
            fromNode: prevId,
            toNode: forkId,
          });
        }
      }
      continue;
    }

    if (s.type === "if") {
      // If: show condition with then/else branches
      const condId = `${pathPrefix}/${s.id}`;
      nodes.push({
        id: condId,
        type: "text",
        text: `${s.id}\nif`,
        x: startX,
        y: curY,
        width: NODE_W,
        height: NODE_H,
        color: borderColor ?? cat.stroke,
        customData: { stepId: s.id, stepIndex: i },
      });

      const thenStep = s.config.then as StepData | undefined;
      const elseStep = s.config.else as StepData | undefined;

      if (thenStep) {
        const thenX = startX - NODE_W / 2 - GAP_X / 2;
        const thenId = `${condId}/then/${thenStep.id}`;
        const thenCat = categoryFor(thenStep.type);
        nodes.push({
          id: thenId,
          type: "text",
          text: `${thenStep.id}\n${thenStep.type}`,
          x: thenX,
          y: curY + NODE_H + GAP_Y,
          width: NODE_W,
          height: NODE_H,
          color: thenCat.stroke,
        });
        edges.push({
          id: `${condId}__then__${thenId}`,
          fromNode: condId,
          toNode: thenId,
          label: "then",
        });
      }

      if (elseStep) {
        const elseX = startX + NODE_W / 2 + GAP_X / 2;
        const elseId = `${condId}/else/${elseStep.id}`;
        const elseCat = categoryFor(elseStep.type);
        nodes.push({
          id: elseId,
          type: "text",
          text: `${elseStep.id}\n${elseStep.type}`,
          x: elseX,
          y: curY + NODE_H + GAP_Y,
          width: NODE_W,
          height: NODE_H,
          color: elseCat.stroke,
        });
        edges.push({
          id: `${condId}__else__${elseId}`,
          fromNode: condId,
          toNode: elseId,
          label: "else",
        });
      }

      maxWidth = Math.max(maxWidth, NODE_W * 2 + GAP_X);
      curY += NODE_H + GAP_Y + NODE_H + GAP_Y;

      if (i > 0) {
        const prevNodeId = `${pathPrefix}/${steps[i - 1]!.id}`;
        const prevNode = nodes.find(
          (n) => n.id === prevNodeId || n.id === `${prevNodeId}__join`,
        );
        if (prevNode) {
          edges.push({
            id: `${prevNode.id}__to__${condId}`,
            fromNode: prevNode.id,
            toNode: condId,
          });
        }
      }
      continue;
    }

    if (s.type === "loop") {
      // Loop: show loop node with a self-referencing note
      const loopId = `${pathPrefix}/${s.id}`;
      const bodyStep = s.config.body as StepData | undefined;
      const bodyLabel = bodyStep ? `${bodyStep.id} (${bodyStep.type})` : "body";
      nodes.push({
        id: loopId,
        type: "text",
        text: `${s.id}\nloop: ${bodyLabel}`,
        x: startX,
        y: curY,
        width: NODE_W + 40,
        height: NODE_H + 10,
        color: borderColor ?? cat.stroke,
        customData: { stepId: s.id, stepIndex: i },
      });

      if (i > 0) {
        const prevNodeId = `${pathPrefix}/${steps[i - 1]!.id}`;
        const prevNode = nodes.find(
          (n) => n.id === prevNodeId || n.id === `${prevNodeId}__join`,
        );
        if (prevNode) {
          edges.push({
            id: `${prevNode.id}__to__${loopId}`,
            fromNode: prevNode.id,
            toNode: loopId,
          });
        }
      }

      curY += NODE_H + 10 + GAP_Y;
      continue;
    }

    // Default: simple step node
    const nodeId = `${pathPrefix}/${s.id}`;
    nodes.push({
      id: nodeId,
      type: "text",
      text: label,
      x: startX,
      y: curY,
      width: NODE_W,
      height: NODE_H,
      color: borderColor ?? cat.stroke,
      customData: { stepId: s.id, stepIndex: i },
    });

    // Sequential edge
    if (i > 0) {
      const prevNodeId = `${pathPrefix}/${steps[i - 1]!.id}`;
      const prevNode = nodes.find(
        (n) => n.id === prevNodeId || n.id === `${prevNodeId}__join`,
      );
      if (prevNode) {
        edges.push({
          id: `${prevNode.id}__to__${nodeId}`,
          fromNode: prevNode.id,
          toNode: nodeId,
        });
      }
    }

    curY += NODE_H + GAP_Y;
  }

  return {
    nodes,
    edges,
    width: maxWidth,
    height: curY - startY,
  };
}

/**
 * Convert a Flow (as returned by the API) into a CanvasData for system-canvas.
 * Optionally overlay run status from events.
 */
export function flowToCanvas(
  flow: FlowData,
  runEvents?: RunEventData[],
): CanvasData {
  const { nodes, edges } = layoutSteps(
    flow.steps,
    100,
    50,
    flow.name,
    runEvents,
  );

  return {
    nodes,
    edges,
    theme: {
      base: "midnight",
    },
  };
}
