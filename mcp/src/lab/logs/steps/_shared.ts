import type { StepContext } from "strut";

/**
 * Shared by the logs/* steps (a seeded helper; the registry skips a leading
 * `_`). Every step is plumbing over `ctx.services.logs` — the lab's in-code
 * logs service (lab/logs/service.ts), which holds the `/logs/agent` fetches,
 * texts and time budget and the AWS client. A seeded step imports `strut`
 * alone, so the capability's shape is restated here.
 */
export interface LogsCapability {
  begin(
    runId: string,
    opts: { dir: string; prompt: string; swarmName?: string; poolName?: string; workspaceSlug?: string; budgetMinutes?: number },
  ): Promise<{ prompt: string; logsDir: string }>;
  fetchCloudwatch(runId: string, args: unknown, signal?: AbortSignal): Promise<string>;
  listGroups(runId: string, args: unknown): Promise<string>;
  fetchQuickwit(runId: string, args: unknown): Promise<string>;
  search(runId: string, args: unknown): Promise<string>;
  read(runId: string, args: unknown): Promise<string>;
  insights(runId: string, args: unknown, signal?: AbortSignal): Promise<string>;
}

export function logsOf(ctx: StepContext): LogsCapability {
  const logs = (ctx.services as { logs?: LogsCapability } | undefined)?.logs;
  if (!logs) throw new Error("services.logs is not here: the logs/* steps run in the lab, which builds it (lab/logs/service.ts)");
  return logs;
}

/** An AbortSignal that fires when the run starts cancelling, so a fetch or a
 *  query stops then rather than at its own wall-clock cap. Call `done()`. */
export function cancelSignal(ctx: StepContext): { signal: AbortSignal; done(): void } {
  const ac = new AbortController();
  const timer = setInterval(() => {
    if (ctx.control?.state === "cancelling") ac.abort();
  }, 500);
  return { signal: ac.signal, done: () => clearInterval(timer) };
}
