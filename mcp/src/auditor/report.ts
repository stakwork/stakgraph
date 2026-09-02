import { fetch } from "undici";
import { Verdict } from "./types.js";

export async function reportVerdict(
  responseUrl: string,
  callbackApiKey: string,
  verdict: Verdict,
): Promise<void> {
  try {
    const resp = await fetch(responseUrl, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": callbackApiKey,
      },
      body: JSON.stringify(verdict),
    });
    console.log(
      `[auditor] reported taskId=${verdict.taskId} overall=${verdict.overall} status=${resp.status}`,
    );
  } catch (err: any) {
    console.error(
      `[auditor] report failed taskId=${verdict.taskId}: ${err?.message ?? String(err)}`,
    );
  }
}
