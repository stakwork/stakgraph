import { z, defineStep } from "vein";

/**
 * Order the changes to process. This is the swappable "ordering strategy"
 * seam for experimentation — the default `chronological` reproduces the
 * existing oldest-first behavior. Swap the strategy (or this whole step)
 * to experiment with reverse / priority-based sweeps.
 */
export default defineStep({
  type: "concepts/prioritize-changes",
  description: "Order changes for processing. strategy: 'chronological' (oldest first, default) | 'reverse' (newest first). Output: { changes, count }.",
  input: z.object({
    changes: z.array(z.any()),
    strategy: z.enum(["chronological", "reverse"]).default("chronological"),
  }),
  output: z.any(),
  async run(cfg) {
    const changes = [...cfg.changes];
    const byDate = (a: any, b: any) =>
      new Date(a.date).getTime() - new Date(b.date).getTime();
    changes.sort(byDate);
    if (cfg.strategy === "reverse") changes.reverse();
    return { changes, count: changes.length };
  },
});
