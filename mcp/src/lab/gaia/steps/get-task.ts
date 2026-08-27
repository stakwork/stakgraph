import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";
import { readFile } from "node:fs/promises";
import { basename } from "node:path";

/**
 * Plumbing over ctx.services.gaia.getTask, PLUS: when the task ships a
 * file, copy it (from the read-only filePath) into this run's artifacts
 * dir so agent steps (cwd = artifacts dir) can see it. Output adds
 * `stagedPath` (relative path under the run's artifacts dir, or null).
 */
export default defineStep({
  type: "gaia/get-task",
  description:
    "Fetch one GAIA task via ctx.services.gaia.getTask(taskId). If the task has an attached file, copies it into this run's artifacts dir (ctx.services.artifacts) and returns the staged RELATIVE path. Config: taskId. Output: { taskId, question, level, fileName, hasFile, stagedPath }.",
  input: z.object({
    taskId: z.string(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const c = ctx as StepContext<VeinCapabilities & { gaia?: any }>;
    const gaia = c.services?.gaia;
    const artifacts = c.services?.artifacts;
    if (!gaia) throw new Error("gaia capability unavailable in this deployment");
    if (!artifacts) throw new Error("artifacts capability unavailable in this deployment");

    const task = await gaia.getTask(cfg.taskId);
    if (!task) {
      throw new Error(
        `gaia.getTask(${cfg.taskId}) returned nothing — likely a bad taskId (check gaia/list-tasks for valid ids)`,
      );
    }

    let stagedPath: string | null = null;
    if (task.filePath) {
      let bytes: Buffer;
      try {
        bytes = await readFile(task.filePath);
      } catch (err: any) {
        throw new Error(
          `Task ${cfg.taskId} declares filePath "${task.filePath}" but it could not be read (${err.message}) — the file may have moved or permissions are wrong`,
        );
      }
      const name: string = task.fileName || basename(task.filePath);
      stagedPath = name;
      await artifacts.write(c.runId, name, bytes);
    }

    return {
      taskId: task.taskId,
      question: task.question,
      level: task.level,
      fileName: task.fileName ?? null,
      hasFile: Boolean(task.filePath),
      stagedPath,
    };
  },
});
