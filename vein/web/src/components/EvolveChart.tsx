import * as api from "../api";

// ── Evolve hill-climb chart ────────────────────────────────────────────────
//
// Renders the fitness trajectory of an eval/evolve-loop step from the run's
// event stream alone — no API beyond the events the panel already has. The
// loop emits synthetic per-generation events at `<path>#<gen>` whose
// step.end/step.replayed output carries { gen, fitness, bestFitness, bestGen,
// directive, version, knownCost, runs } (see eval/evolve-loop's emitGen), so
// the chart derives everything live: a dashed baseline, the best-so-far
// staircase, and one dot per generation (filled = new best, amber stroke =
// explore directive, ✕ = failed generation, pulsing = still running).
// Clicking a dot opens that generation's run via the same run-ref plumbing
// the event rows use.

interface GenPoint {
  gen: number;
  status: "done" | "replayed" | "error" | "pending";
  fitness?: number;
  bestFitness?: number;
  bestGen?: number;
  directive?: string;
  version?: string;
  knownCost?: number;
  run?: { workflow: string; runId: string };
  error?: string;
}

interface EvolveSeries {
  baseline?: number;
  points: GenPoint[];
}

/** Collect the evolve-loop's per-generation events into an ordered series.
 *  Returns null when the run has no evolve-loop generation events at all. */
export function evolveSeries(events: api.RunEvent[]): EvolveSeries | null {
  const byGen = new Map<number, GenPoint>();
  let baseline: number | undefined;
  // Multiple evolve loops in one run is theoretical; pin to the first path
  // prefix seen so a second loop can't interleave garbage into the chart.
  let prefix: string | undefined;

  for (const e of events) {
    if (e.stepType !== "eval/evolve-loop") continue;
    const hi = e.path.lastIndexOf("#");
    if (hi < 0) continue;
    const p = e.path.slice(0, hi);
    if (prefix === undefined) prefix = p;
    else if (p !== prefix) continue;
    const gen = Number(e.path.slice(hi + 1));
    if (!Number.isInteger(gen) || gen < 0) continue;

    const cur: GenPoint = byGen.get(gen) ?? { gen, status: "pending" };
    if (e.type === "step.start") {
      const inp = e.input as { directive?: string; bestFitness?: number } | undefined;
      if (typeof inp?.directive === "string") cur.directive = inp.directive;
      // Gen 0's start is briefed with bestFitness === the baseline fitness.
      if (gen === 0 && typeof inp?.bestFitness === "number") baseline = inp.bestFitness;
    } else if (e.type === "step.end" || e.type === "step.replayed") {
      const o = (e.output ?? {}) as Record<string, unknown>;
      cur.status = e.type === "step.end" ? "done" : "replayed";
      if (typeof o.fitness === "number") cur.fitness = o.fitness;
      if (typeof o.bestFitness === "number") cur.bestFitness = o.bestFitness;
      if (typeof o.bestGen === "number") cur.bestGen = o.bestGen;
      if (typeof o.directive === "string") cur.directive = o.directive;
      if (typeof o.version === "string") cur.version = o.version;
      if (typeof o.knownCost === "number") cur.knownCost = o.knownCost;
      const r = Array.isArray(o.runs) ? (o.runs[0] as Record<string, unknown> | undefined) : undefined;
      if (r && typeof r.workflow === "string" && typeof r.runId === "string") {
        cur.run = { workflow: r.workflow, runId: r.runId };
      }
      // A replayed gen 0 that had not yet beaten the baseline still tells us
      // the baseline (its bestFitness anchor is the baseline itself).
      if (gen === 0 && baseline === undefined && cur.bestGen === -1 && typeof cur.bestFitness === "number") {
        baseline = cur.bestFitness;
      }
    } else if (e.type === "step.error") {
      cur.status = "error";
      cur.error = e.error?.message;
      cur.fitness = 0;
    }
    byGen.set(gen, cur);
  }

  if (byGen.size === 0) return null;
  const points = [...byGen.values()].sort((a, b) => a.gen - b.gen);
  return { baseline, points };
}

const PAD_L = 34; // room for y labels
const PAD_R = 14;
const PAD_T = 12;
const PAD_B = 20; // room for gen labels
const X_STEP = 56;
const PLOT_H = 96;

export function EvolveChart(props: {
  events: api.RunEvent[];
  onOpenRun?: (workflow: string, runId: string) => void;
}) {
  const series = evolveSeries(props.events);
  if (!series) return null;
  const { baseline, points } = series;

  const H = PAD_T + PLOT_H + PAD_B;
  // x slot 0 is the baseline anchor; generations start at slot 1.
  const slots = points.length + 1;
  const W = PAD_L + (slots - 1) * X_STEP + PAD_R;
  const x = (slot: number) => PAD_L + slot * X_STEP;
  const y = (fitness: number) => PAD_T + (1 - Math.max(0, Math.min(1, fitness))) * PLOT_H;

  // Best-so-far staircase, anchored at the baseline. Each finished gen's
  // output.bestFitness is the loop's own computation — no re-deriving.
  let prevBest = baseline;
  const stair: string[] = [];
  if (baseline !== undefined) stair.push(`M ${x(0)} ${y(baseline)}`);
  points.forEach((p, i) => {
    const b = p.bestFitness ?? prevBest;
    if (b === undefined) return;
    const xi = x(i + 1);
    if (stair.length === 0) stair.push(`M ${xi} ${y(b)}`);
    else {
      if (prevBest !== undefined) stair.push(`L ${xi} ${y(prevBest)}`);
      stair.push(`L ${xi} ${y(b)}`);
    }
    prevBest = b;
  });

  const bestPoint = points.reduce<GenPoint | null>(
    (acc, p) => (p.fitness !== undefined && (!acc || p.fitness > (acc.fitness ?? 0)) ? p : acc),
    null,
  );
  const best = prevBest;
  const gridLines = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div class="evolve-chart">
      <div class="evolve-chart-header">
        <span class="evolve-chart-title">hill-climb</span>
        {baseline !== undefined && <span class="evolve-chart-stat">baseline {fmt(baseline)}</span>}
        {best !== undefined && (
          <span class="evolve-chart-stat evolve-chart-best">
            best {fmt(best)}
            {bestPoint && bestPoint.bestGen !== undefined && bestPoint.bestGen >= 0 ? ` (gen ${bestPoint.bestGen})` : ""}
          </span>
        )}
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} class="evolve-chart-svg">
        {gridLines.map((g) => (
          <g key={g}>
            <line x1={PAD_L - 4} y1={y(g)} x2={W - PAD_R} y2={y(g)} class="evolve-grid" />
            <text x={PAD_L - 8} y={y(g) + 3} class="evolve-ylabel">
              {g === 0 || g === 1 || g === 0.5 ? g : ""}
            </text>
          </g>
        ))}

        {baseline !== undefined && (
          <line x1={x(0)} y1={y(baseline)} x2={W - PAD_R} y2={y(baseline)} class="evolve-baseline" />
        )}
        {stair.length > 1 && <path d={stair.join(" ")} class="evolve-stair" />}

        {/* baseline anchor */}
        {baseline !== undefined && (
          <g>
            <circle cx={x(0)} cy={y(baseline)} r={3.5} class="evolve-dot-base" />
            <text x={x(0)} y={H - 6} class="evolve-xlabel">base</text>
          </g>
        )}

        {points.map((p, i) => {
          const xi = x(i + 1);
          const label = `gen ${p.gen}`;
          const clickable = p.run && props.onOpenRun;
          const open = () => clickable && props.onOpenRun!(p.run!.workflow, p.run!.runId);
          const tip = [
            label,
            p.version ? `version ${p.version}` : null,
            p.fitness !== undefined ? `fitness ${fmt(p.fitness)}` : null,
            best !== undefined && p.fitness !== undefined
              ? `Δ best ${fmt(p.fitness - (p.bestGen === p.gen ? (points[i - 1]?.bestFitness ?? baseline ?? 0) : (p.bestFitness ?? 0)))}`
              : null,
            p.directive ? `directive: ${p.directive}` : null,
            p.knownCost !== undefined ? `cost $${p.knownCost}` : null,
            p.status === "error" ? `FAILED: ${p.error ?? "unknown"}` : null,
            p.status === "pending" ? "running…" : null,
            clickable ? "click to open run" : null,
          ]
            .filter(Boolean)
            .join("\n");

          if (p.status === "error") {
            return (
              <g key={p.gen} class="evolve-pt" onClick={open}>
                <text x={xi} y={y(0) + 4} class="evolve-err">✕<title>{tip}</title></text>
                <text x={xi} y={H - 6} class="evolve-xlabel">{p.gen}</text>
              </g>
            );
          }
          if (p.status === "pending") {
            const yy = y(prevBestBefore(points, i, baseline));
            return (
              <g key={p.gen} class="evolve-pt">
                <circle cx={xi} cy={yy} r={4.5} class="evolve-dot-pending"><title>{tip}</title></circle>
                <text x={xi} y={H - 6} class="evolve-xlabel">{p.gen}</text>
              </g>
            );
          }
          const improved = p.bestGen === p.gen;
          const explore = p.directive === "explore";
          return (
            <g key={p.gen} class={`evolve-pt${clickable ? " evolve-pt-link" : ""}`} onClick={open}>
              <circle
                cx={xi}
                cy={y(p.fitness ?? 0)}
                r={improved ? 5 : 4}
                class={`evolve-dot ${improved ? "evolve-dot-best" : "evolve-dot-miss"}${explore ? " evolve-dot-explore" : ""}`}
              >
                <title>{tip}</title>
              </circle>
              <text x={xi} y={H - 6} class="evolve-xlabel">{p.gen}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Best fitness known just BEFORE generation i (where a pending dot hovers). */
function prevBestBefore(points: GenPoint[], i: number, baseline?: number): number {
  for (let j = i - 1; j >= 0; j--) {
    const b = points[j]?.bestFitness;
    if (typeof b === "number") return b;
  }
  return baseline ?? 0.5;
}

function fmt(n: number): string {
  return String(Math.round(n * 1000) / 1000);
}
