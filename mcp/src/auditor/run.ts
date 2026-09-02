import { prepareAuditor } from "./agent.js";
import { reportVerdict } from "./report.js";
import { AuditJob, Verdict } from "./types.js";

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf8");
}

function fallbackVerdict(
  taskId: string,
  error: string,
  startedAt: string,
): Verdict {
  return {
    taskId,
    overall: "unknown",
    claims: [],
    observations: [],
    summary: "The auditor could not complete.",
    startedAt,
    finishedAt: new Date().toISOString(),
    error,
  };
}

async function main(): Promise<void> {
  const startedAt = new Date().toISOString();
  let job: AuditJob | undefined;

  try {
    const raw = await readStdin();
    job = JSON.parse(raw) as AuditJob;
  } catch (err: any) {
    console.error(
      `[auditor] failed to read/parse job from stdin: ${err?.message ?? String(err)}`,
    );
    return;
  }

  let verdict: Verdict;
  try {
    verdict = await prepareAuditor(job).run();
  } catch (err: any) {
    verdict = fallbackVerdict(
      job.taskId,
      err?.message ?? String(err),
      startedAt,
    );
  }

  if (job.responseUrl && job.callbackApiKey) {
    await reportVerdict(job.responseUrl, job.callbackApiKey, verdict);
  } else {
    console.error("[auditor] no responseUrl/callbackApiKey — verdict not reported");
    console.log(JSON.stringify(verdict));
  }
}

main().catch((err) => {
  console.error(`[auditor] fatal: ${err?.message ?? String(err)}`);
});
