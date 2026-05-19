// Generic uPlot wrapper. uPlot is intentionally framework-agnostic
// (it doesn't ship a React/Preact wrapper) — this component bridges
// to Preact's lifecycle.
//
// Resize behaviour: we observe the parent's width via ResizeObserver
// and call `chart.setSize(w, h)`. uPlot otherwise locks to the
// initial size, which looks fine until someone resizes the browser
// or expands the sidebar.

import { useEffect, useRef } from "preact/hooks";
import uPlot from "uplot";
import type { AlignedData, Options } from "uplot";

interface Props {
  data: AlignedData;
  options: Options;
  /** Fixed pixel height. Width is computed from the parent. */
  height: number;
}

export function UplotChart({ data, options, height }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    // Tear down any previous chart on data/options change. uPlot
    // can update in-place via setData/setSize but the options
    // object is treated as immutable; rebuilding is the only
    // honest way to handle a series list change between renders.
    chartRef.current?.destroy();
    const opts: Options = {
      ...options,
      width: containerRef.current.clientWidth,
      height,
    };
    chartRef.current = new uPlot(opts, data, containerRef.current);

    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width ?? 0;
      if (w > 0) chartRef.current?.setSize({ width: w, height });
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [data, options, height]);

  return <div ref={containerRef} style={`height:${height}px`} />;
}
