import { createElement } from "preact";
import type {
  CanvasData,
  CanvasNode,
  CanvasEdge,
  CanvasTheme,
  SlotContext,
} from "system-canvas";
import { midnightTheme, resolveTheme } from "system-canvas";

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

// ── Step type colors (each becomes a theme category) ───────────────────────

const STEP_COLORS: Record<string, { fill: string; stroke: string }> = {
  http:     { fill: "rgba(6, 78, 59, 0.4)",   stroke: "#34d399" },
  log:      { fill: "rgba(30, 58, 138, 0.4)", stroke: "#60a5fa" },
  if:       { fill: "rgba(120, 53, 15, 0.4)", stroke: "#f59e0b" },
  loop:     { fill: "rgba(88, 28, 135, 0.4)", stroke: "#a78bfa" },
  parallel: { fill: "rgba(127, 29, 29, 0.4)", stroke: "#f87171" },
  subflow:  { fill: "rgba(21, 94, 117, 0.4)", stroke: "#22d3ee" },
  llm:      { fill: "rgba(76, 29, 149, 0.4)", stroke: "#c084fc" },
  wait:     { fill: "rgba(71, 85, 105, 0.4)", stroke: "#94a3b8" },
  default:  { fill: "rgba(38, 38, 38, 0.6)",  stroke: "#737373" },
};

function colorsFor(type: string) {
  return STEP_COLORS[type] ?? STEP_COLORS["default"]!;
}

// ── Status indicator (topRight custom slot) ────────────────────────────────

const STATUS_CHECK = "#22c55e";
const STATUS_ERROR = "#ef4444";
const STATUS_RUNNING = "#f59e0b";
const STATUS_PENDING = "#6b7689";

/**
 * Render a status indicator in the topRight slot region:
 *   - success  → green checkmark
 *   - error    → red X
 *   - running  → yellow pulsing dot
 *   - pending  → grey clock icon
 */
function renderStatusIndicator(ctx: SlotContext): unknown {
  const status = ctx.node.customData?.status as string | undefined;
  if (!status) return null;

  const { region } = ctx;
  const cx = region.x + region.width / 2;
  const cy = region.y + region.height / 2;
  const r = Math.min(region.width, region.height) / 2;

  if (status === "success") {
    // Green checkmark
    const s = r * 0.7;
    return createElement("g", { pointerEvents: "none" },
      createElement("path", {
        d: `M ${cx - s} ${cy} L ${cx - s * 0.2} ${cy + s * 0.7} L ${cx + s} ${cy - s * 0.5}`,
        stroke: STATUS_CHECK,
        strokeWidth: 2,
        fill: "none",
        strokeLinecap: "round",
        strokeLinejoin: "round",
      }),
    );
  }

  if (status === "error") {
    // Red X
    const s = r * 0.5;
    return createElement("g", { pointerEvents: "none" },
      createElement("line", {
        x1: cx - s, y1: cy - s, x2: cx + s, y2: cy + s,
        stroke: STATUS_ERROR, strokeWidth: 2, strokeLinecap: "round",
      }),
      createElement("line", {
        x1: cx + s, y1: cy - s, x2: cx - s, y2: cy + s,
        stroke: STATUS_ERROR, strokeWidth: 2, strokeLinecap: "round",
      }),
    );
  }

  if (status === "running") {
    // Yellow pulsing dot
    return createElement("g", { pointerEvents: "none" },
      createElement("circle", {
        cx, cy, r: r * 0.4,
        fill: STATUS_RUNNING, opacity: 0.9,
      }),
    );
  }

  // pending → grey clock icon
  const cr = r * 0.55;
  return createElement("g", { pointerEvents: "none" },
    createElement("circle", {
      cx, cy, r: cr,
      fill: "none", stroke: STATUS_PENDING, strokeWidth: 1.3,
    }),
    // hour hand (12 o'clock to 3 o'clock)
    createElement("line", {
      x1: cx, y1: cy, x2: cx, y2: cy - cr * 0.55,
      stroke: STATUS_PENDING, strokeWidth: 1.3, strokeLinecap: "round",
    }),
    createElement("line", {
      x1: cx, y1: cy, x2: cx + cr * 0.45, y2: cy,
      stroke: STATUS_PENDING, strokeWidth: 1.3, strokeLinecap: "round",
    }),
  );
}

// ── Build theme categories from step types ─────────────────────────────────

function buildStepCategory(type: string, colors: { fill: string; stroke: string }): any {
  return {
    fill: colors.fill,
    stroke: colors.stroke,
    cornerRadius: 8,
    defaultWidth: NODE_W,
    defaultHeight: NODE_H,
    type: "text" as const,
    slots: {
      header: {
        kind: "text",
        value: type.toUpperCase(),
        color: colors.stroke,
        fontSize: 10,
        fontWeight: 600,
      },
      topRight: {
        kind: "custom",
        render: renderStatusIndicator,
      },
    },
  };
}

function buildCategories(): Record<string, any> {
  const categories: Record<string, any> = {};
  for (const [type, colors] of Object.entries(STEP_COLORS)) {
    categories[`step-${type}`] = buildStepCategory(type, colors);
  }
  return categories;
}

// ── Build theme ────────────────────────────────────────────────────────────

const veinTheme: CanvasTheme = resolveTheme(
  {
    name: "vein",
    categories: buildCategories(),
  },
  midnightTheme,
);

// ── Convert Flow → CanvasData ──────────────────────────────────────────────

interface LayoutResult {
  nodes: CanvasNode[];
  edges: CanvasEdge[];
  width: number;
  height: number;
}

function stepStatus(
  stepPath: string,
  runEvents?: RunEventData[],
): string | undefined {
  if (!runEvents) return undefined;
  const stepEvts = runEvents.filter((e) => e.path === stepPath);
  if (stepEvts.length === 0) return "pending";
  const hasError = stepEvts.some((e) => e.type === "step.error");
  const hasEnd = stepEvts.some((e) => e.type === "step.end");
  if (hasError) return "error";
  if (hasEnd) return "success";
  return "running";
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
    const colors = colorsFor(s.type);
    const category = `step-${STEP_COLORS[s.type] ? s.type : "default"}`;
    const status = stepStatus(stepPath, runEvents);

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
        category,
        text: s.id,
        x: forkX,
        y: curY,
        width: NODE_W,
        height: NODE_H,
        customData: { stepId: s.id, stepIndex: i, status },
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
        category,
        text: `${s.id} (join)`,
        x: forkX,
        y: joinY,
        width: NODE_W,
        height: NODE_H / 2,
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
        category,
        text: s.id,
        x: startX,
        y: curY,
        width: NODE_W,
        height: NODE_H,
        customData: { stepId: s.id, stepIndex: i, status },
      });

      const thenStep = s.config.then as StepData | undefined;
      const elseStep = s.config.else as StepData | undefined;

      if (thenStep) {
        const thenX = startX - NODE_W / 2 - GAP_X / 2;
        const thenId = `${condId}/then/${thenStep.id}`;
        const thenCategory = `step-${STEP_COLORS[thenStep.type] ? thenStep.type : "default"}`;
        const thenStatus = stepStatus(thenId, runEvents);
        nodes.push({
          id: thenId,
          type: "text",
          category: thenCategory,
          text: thenStep.id,
          x: thenX,
          y: curY + NODE_H + GAP_Y,
          width: NODE_W,
          height: NODE_H,
          customData: { status: thenStatus },
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
        const elseCategory = `step-${STEP_COLORS[elseStep.type] ? elseStep.type : "default"}`;
        const elseStatus = stepStatus(elseId, runEvents);
        nodes.push({
          id: elseId,
          type: "text",
          category: elseCategory,
          text: elseStep.id,
          x: elseX,
          y: curY + NODE_H + GAP_Y,
          width: NODE_W,
          height: NODE_H,
          customData: { status: elseStatus },
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
      // Loop: show loop node with body info
      const loopId = `${pathPrefix}/${s.id}`;
      const bodyStep = s.config.body as StepData | undefined;
      const bodyLabel = bodyStep ? `${s.id} → ${bodyStep.id}` : s.id;
      nodes.push({
        id: loopId,
        type: "text",
        category,
        text: bodyLabel,
        x: startX,
        y: curY,
        width: NODE_W + 40,
        height: NODE_H + 10,
        customData: { stepId: s.id, stepIndex: i, status },
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
      category,
      text: s.id,
      x: startX,
      y: curY,
      width: NODE_W,
      height: NODE_H,
      customData: { stepId: s.id, stepIndex: i, status },
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
      base: "vein",
    },
  };
}

// ── Register the vein theme so system-canvas can resolve `base: "vein"` ────

export { veinTheme };
