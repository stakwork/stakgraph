import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";

/**
 * Plumbing over ctx.services.gaia.listTasks. Optionally filter by level
 * (1|2|3). Output: { tasks: [{ taskId, level, hasFile }], count, byLevel }.
 */
export default defineStep({
  type: "gaia/list-tasks",
  description:
    "List GAIA benchmark tasks via ctx.services.gaia.listTasks({ level? }). Config: level? (1|2|3, omit for all). Output: { tasks: [{ taskId, level, hasFile }], count, byLevel: { '1': n, '2': n, '3': n } }.",
  input: z.object({
    level: z.union([z.literal(1), z.literal(2), z.literal(3)]).optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const c = ctx as StepContext<VeinCapabilities & { gaia?: any }>;
    const gaia = c.services?.gaia;
    if (!gaia) throw new Error("gaia capability unavailable in this deployment");

    const tasks = await gaia.listTasks(cfg.level ? { level: cfg.level } : {});
    const byLevel: Record<string, number> = {};
    for (const t of tasks) {
      const key = String(t.level);
      byLevel[key] = (byLevel[key] || 0) + 1;
    }
    return { tasks, count: tasks.length, byLevel };
  },
});
