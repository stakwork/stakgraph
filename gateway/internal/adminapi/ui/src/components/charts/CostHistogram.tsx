// CostHistogram — stacked-area cost-by-time-by-dimension chart used
// on the Dashboard and AgentDetail pages. Thin shape-builder on top
// of UplotChart.
//
// Why client-side stacking
// ------------------------
// The backend (/_plugin/histogram/cost) returns one series per
// dimension value, NOT pre-stacked. uPlot's `bands` or `paths.bars`
// + a cumulative dataset produces a stacked area; doing the stacking
// in the browser keeps the response shape simple (one row per
// (dim, ts) pair) and means the same data can drive a stacked area
// or a multi-line view by swapping the series options.

import { useMemo } from "preact/hooks";
import type { Options, AlignedData } from "uplot";

import type { HistogramCostResponse } from "../../api/types";
import { UplotChart } from "./UplotChart";

// Palette: cycle through these for series; first ~6 align with the
// dark theme accent + warmer fallbacks. Beyond 8 the colours repeat
// — fine for v1 (more than 8 agents = the legend is unreadable
// anyway, fix in phase-9 with a "top N + other" rollup).
const PALETTE = [
  "#6ea8ff",
  "#69d6a4",
  "#f5b86c",
  "#ff6b6b",
  "#b189ff",
  "#5ed3e6",
  "#ffd166",
  "#ff8cb0",
];

interface Props {
  data: HistogramCostResponse;
  /** Window size in seconds, used to pin the x-axis range. Without
   *  this uPlot's auto-range looks fine for many data points but
   *  zooms out to a wild span when there's only one bucket (which
   *  is exactly what happens in a fresh dev install). Always pass
   *  it; pages know their window from the picker state. */
  windowSeconds: number;
  height?: number;
}

export function CostHistogram({ data, windowSeconds, height = 280 }: Props) {
  const { aligned, options } = useMemo(
    () => buildChartData(data, windowSeconds),
    [data, windowSeconds]
  );

  if (aligned[0].length === 0) {
    return <div class="empty">No cost data in the selected window.</div>;
  }
  return <UplotChart data={aligned} options={options} height={height} />;
}

function buildChartData(
  data: HistogramCostResponse,
  windowSeconds: number
): { aligned: AlignedData; options: Options } {
  // The x-axis spans exactly the query window — anchored to "now"
  // on the right. Pinning the range here means a single data point
  // (very common during onboarding) renders against a sensible
  // timeline instead of uPlot's auto-zoom across the epoch.
  const nowSec = Math.floor(Date.now() / 1000);
  const xMin = nowSec - windowSeconds;
  const xMax = nowSec;

  // Bucket size from the backend response — width of each cell
  // in the stacked bar / step chart.
  const bucketSec = data.bucket_size_seconds || 3600;

  // Collect every distinct bucket timestamp across all series and
  // sort once. uPlot requires aligned data: each series has the
  // same x[] (timestamps), with `null` where that series has no
  // value at a bucket.
  const tsSet = new Set<number>();
  for (const s of data.series) {
    for (const p of s.points) {
      tsSet.add(Math.floor(Date.parse(p.ts) / 1000));
    }
  }
  const xs = [...tsSet].sort((a, b) => a - b);
  if (xs.length === 0) {
    return {
      aligned: [[]],
      options: { width: 0, height: 0, series: [{}] },
    };
  }

  // Build the (series, x) matrix.
  const ySeries: (number | null)[][] = data.series.map((s) => {
    const lookup = new Map<number, number>();
    for (const p of s.points) {
      lookup.set(Math.floor(Date.parse(p.ts) / 1000), p.cost);
    }
    return xs.map((t) => (lookup.has(t) ? lookup.get(t)! : null));
  });

  const aligned: AlignedData = [xs, ...ySeries];

  const options: Options = {
    width: 0,
    height: 0,
    legend: { show: true },
    cursor: { drag: { x: false, y: false } },
    scales: {
      // Force the visible range to the query window. `time: true`
      // tells uPlot to format ticks as local-time dates rather
      // than raw numbers. The range-as-function form is documented
      // as the canonical way to pin (not just hint) the x extents
      // — uPlot still applies its data-driven auto-range when a
      // plain {min,max} object is passed.
      x: {
        time: true,
        auto: false,
        range: () => [xMin, xMax],
      },
      y: { auto: true },
    },
    axes: [
      {
        stroke: "#97a1b1",
        grid: { stroke: "#232c3b", width: 1 },
        ticks: { stroke: "#232c3b", width: 1 },
      },
      {
        stroke: "#97a1b1",
        grid: { stroke: "#232c3b", width: 1 },
        ticks: { stroke: "#232c3b", width: 1 },
        size: 64,
        values: (_self, splits) =>
          splits.map((v) => (v == null ? "" : fmtAxisUSD(v))),
      },
    ],
    series: [
      { label: "Time" },
      ...data.series.map((s, i) => ({
        label: s.dimension_value,
        stroke: PALETTE[i % PALETTE.length],
        width: 2,
        fill: PALETTE[i % PALETTE.length] + "22", // 13% alpha
        // Show points when there are very few buckets — a single
        // data point with no marker is invisible.
        points: { show: xs.length <= 3, size: 6 },
        value: (_u: unknown, v: number | null) =>
          v == null ? "—" : fmtSeriesUSD(v),
      })),
    ],
  };
  // bucketSec is referenced in case a future option wants to draw
  // bar widths — keep the value bound so the linter doesn't trim it.
  void bucketSec;
  return { aligned, options };
}

// Axis labels: tight, no padding, 4 decimals when the spend is in
// sub-cent territory so the gridlines actually read non-zero.
function fmtAxisUSD(v: number): string {
  if (v === 0) return "$0";
  const abs = Math.abs(v);
  if (abs < 0.01) return "$" + v.toFixed(4);
  if (abs < 1) return "$" + v.toFixed(3);
  return "$" + v.toFixed(2);
}

// Legend / tooltip values: higher precision so dev-scale spend is
// readable.
function fmtSeriesUSD(v: number): string {
  if (v === 0) return "$0.00";
  return Math.abs(v) < 0.01 ? "$" + v.toFixed(6) : "$" + v.toFixed(2);
}
