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
 */
export interface StepData {
  id: string;
  type: string;
  config: Record<string, any>;
  depends?: string | string[];
  when?: boolean;
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

const NODE_W = 160;
const NODE_H = 60;
const GAP_Y = 50;
const GAP_X = 40;

// ── Step type colors (each becomes a theme category) ───────────────────────

const STEP_COLORS: Record<string, { fill: string; stroke: string }> = {
  http:     { fill: "rgba(6, 78, 59, 0.4)",   stroke: "#34d399" },
  log:      { fill: "rgba(30, 58, 138, 0.4)", stroke: "#60a5fa" },
  if:       { fill: "rgba(120, 53, 15, 0.4)", stroke: "#f59e0b" },
  loop:     { fill: "rgba(88, 28, 135, 0.4)", stroke: "#a78bfa" },
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
const STATUS_SKIPPED = "#4b5563";

function renderStatusIndicator(ctx: SlotContext): unknown {
  const status = ctx.node.customData?.status as string | undefined;
  if (!status) return null;

  const { region } = ctx;
  const cx = region.x + region.width / 2;
  const cy = region.y + region.height / 2;
  const r = Math.min(region.width, region.height) / 2;

  if (status === "success") {
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
    return createElement("g", { pointerEvents: "none" },
      createElement("circle", {
        cx, cy, r: r * 0.4,
        fill: STATUS_RUNNING, opacity: 0.9,
      }),
    );
  }

  if (status === "skipped") {
    return createElement("g", { pointerEvents: "none" },
      createElement("circle", {
        cx, cy, r: r * 0.3,
        fill: STATUS_SKIPPED, opacity: 0.7,
      }),
    );
  }

  // pending
  const cr = r * 0.55;
  return createElement("g", { pointerEvents: "none" },
    createElement("circle", {
      cx, cy, r: cr,
      fill: "none", stroke: STATUS_PENDING, strokeWidth: 1.3,
    }),
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

// ── DAG layout helpers ─────────────────────────────────────────────────────

/**
 * Get explicit deps for a step. If `depends` is set, use it.
 * Otherwise, implicit sequential: depends on previous step in array.
 */
function getDeps(step: StepData, index: number, steps: StepData[]): string[] {
  if (step.depends != null) {
    return Array.isArray(step.depends) ? step.depends : [step.depends];
  }
  if (index > 0) return [steps[index - 1]!.id];
  return [];
}

/**
 * Assign each step to a topological layer (depth).
 * Steps with no deps are layer 0. A step's layer = max(dep layers) + 1.
 */
function assignLayers(steps: StepData[]): Map<string, number> {
  const layers = new Map<string, number>();

  function getLayer(id: string): number {
    if (layers.has(id)) return layers.get(id)!;
    const idx = steps.findIndex((s) => s.id === id);
    if (idx === -1) return 0;
    const deps = getDeps(steps[idx]!, idx, steps);
    const depth = deps.length === 0 ? 0 : Math.max(...deps.map(getLayer)) + 1;
    layers.set(id, depth);
    return depth;
  }

  for (const s of steps) getLayer(s.id);
  return layers;
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
  const hasSkipped = stepEvts.some((e) => e.type === "step.skipped");
  if (hasError) return "error";
  if (hasSkipped) return "skipped";
  if (hasEnd) return "success";
  return "running";
}

// ── Convert Flow → CanvasData ──────────────────────────────────────────────

export function flowToCanvas(
  flow: FlowData,
  runEvents?: RunEventData[],
): CanvasData {
  const { steps } = flow;
  const nodes: CanvasNode[] = [];
  const edges: CanvasEdge[] = [];

  if (steps.length === 0) {
    return { nodes, edges, theme: { base: "vein" } };
  }

  // Build a quick lookup of step ids → step (for finding gates)
  const stepById = new Map<string, StepData>();
  for (const s of steps) stepById.set(s.id, s);

  // Assign layers
  const layers = assignLayers(steps);

  // Group steps by layer
  const maxLayer = Math.max(...Array.from(layers.values()), 0);
  const layerGroups: StepData[][] = Array.from({ length: maxLayer + 1 }, () => []);
  for (const s of steps) {
    layerGroups[layers.get(s.id) ?? 0]!.push(s);
  }

  // Layout: each layer is a row, nodes in same layer are side by side
  const nodePositions = new Map<string, { x: number; y: number }>();
  let curY = 50;

  for (let layer = 0; layer <= maxLayer; layer++) {
    const group = layerGroups[layer]!;
    const totalWidth = group.length * NODE_W + (group.length - 1) * GAP_X;
    const startX = 100 + (NODE_W - totalWidth) / 2;

    for (let i = 0; i < group.length; i++) {
      const s = group[i]!;
      const x = startX + i * (NODE_W + GAP_X);
      nodePositions.set(s.id, { x, y: curY });
    }

    curY += NODE_H + GAP_Y;
  }

  // Create nodes
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i]!;
    const pos = nodePositions.get(s.id)!;
    const stepPath = `${flow.name}/${s.id}`;
    const category = `step-${STEP_COLORS[s.type] ? s.type : "default"}`;
    const status = stepStatus(stepPath, runEvents);

    const w = s.type === "loop" ? NODE_W + 40 : NODE_W;
    const h = s.type === "loop" ? NODE_H + 10 : NODE_H;

    let text = s.id;
    if (s.type === "loop") {
      const bodyStep = s.config.body as StepData | undefined;
      if (bodyStep) text = `${s.id} → ${bodyStep.id}`;
    }

    nodes.push({
      id: `${flow.name}/${s.id}`,
      type: "text",
      category,
      text,
      x: pos.x,
      y: pos.y,
      width: w,
      height: h,
      customData: { stepId: s.id, stepIndex: i, status },
    });
  }

  // Create edges from depends. If the step has `when` and the dep is an
  // `if` gate, label the edge with "true" or "false".
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i]!;
    const deps = getDeps(s, i, steps);
    for (const dep of deps) {
      const fromId = `${flow.name}/${dep}`;
      const toId = `${flow.name}/${s.id}`;
      const depStep = stepById.get(dep);
      const isGateEdge = s.when != null && depStep?.type === "if";
      const edge: CanvasEdge = {
        id: `${fromId}__to__${toId}`,
        fromNode: fromId,
        toNode: toId,
      };
      if (isGateEdge) {
        edge.label = s.when ? "true" : "false";
      }
      edges.push(edge);
    }
  }

  return {
    nodes,
    edges,
    theme: { base: "vein" },
  };
}

// ── Register the vein theme so system-canvas can resolve `base: "vein"` ────

export { veinTheme };
