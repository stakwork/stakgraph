import { fetch } from "undici";
import { Verdict } from "./types.js";

const REPORT_ATTEMPTS =
  parseInt(process.env.AUDIT_REPORT_ATTEMPTS || "", 10) || 4;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function reportVerdict(
  responseUrl: string,
  callbackApiKey: string,
  verdict: Verdict,
): Promise<void> {
  for (let attempt = 1; attempt <= REPORT_ATTEMPTS; attempt++) {
    try {
      const resp = await fetch(responseUrl, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": callbackApiKey,
        },
        body: JSON.stringify(verdict),
      });
      if (resp.ok) {
        console.log(
          `[auditor] reported taskId=${verdict.taskId} overall=${verdict.overall} status=${resp.status}`,
        );
        return;
      }
      console.error(
        `[auditor] report non-ok taskId=${verdict.taskId} status=${resp.status} attempt=${attempt}/${REPORT_ATTEMPTS}`,
      );
    } catch (err: any) {
      console.error(
        `[auditor] report attempt ${attempt}/${REPORT_ATTEMPTS} failed taskId=${verdict.taskId}: ${err?.message ?? String(err)}`,
      );
    }
    if (attempt < REPORT_ATTEMPTS) await sleep(500 * 2 ** (attempt - 1));
  }
  console.error(
    `[auditor] report gave up taskId=${verdict.taskId} after ${REPORT_ATTEMPTS} attempts`,
  );
}
