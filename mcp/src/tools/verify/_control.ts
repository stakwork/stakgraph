import express from "express";
import { runVerification } from "./agent.js";

const PORT = parseInt(process.env.CONTROL_PORT || "3457", 10);
const app = express();
app.use(express.json({ limit: "10mb" }));

app.get("/health", (_req, res) => res.json({ ok: true }));

app.post("/audit", async (req, res) => {
  const job = req.body || {};
  const { taskId, deck, model, responseUrl, callbackApiKey } = job;
  if (!taskId || !deck || !responseUrl) {
    return res.status(400).json({ error: "missing taskId/deck/responseUrl" });
  }
  console.error(`[control] received audit job taskId=${taskId} appUrl=${deck?.map?.appUrl}`);
  res.json({ accepted: true, taskId });

  (async () => {
    let verdict: any;
    try {
      verdict = await runVerification({
        taskId,
        taskPrompt: deck?.task?.prompt ?? "",
        diff: typeof deck?.diff === "string" ? deck.diff : JSON.stringify(deck?.diff ?? ""),
        appUrl: deck?.map?.appUrl ?? "http://localhost:3000",
        notes: deck?.map?.notes ?? null,
        model: model?.model ?? "anthropic/claude-opus-5",
        apiKey: model?.apiKey ?? process.env.ANTHROPIC_API_KEY ?? "",
        sessionId: `ctrl-${taskId}-${Date.now()}`,
      });
    } catch (err: any) {
      verdict = {
        taskId,
        overall: "unknown",
        claims: [],
        observations: [`control error: ${err?.message ?? String(err)}`],
        summary: "verification could not complete",
        evidence: [],
        startedAt: new Date().toISOString(),
        finishedAt: new Date().toISOString(),
        error: err?.message ?? String(err),
      };
    }
    try {
      const resp = await fetch(responseUrl, {
        method: "POST",
        headers: { "content-type": "application/json", "x-api-key": callbackApiKey ?? "" },
        body: JSON.stringify(verdict),
      });
      console.error(`[control] reported taskId=${taskId} overall=${verdict.overall} status=${resp.status}`);
    } catch (err: any) {
      console.error(`[control] report failed taskId=${taskId}: ${err?.message ?? String(err)}`);
    }
  })();
});

app.listen(PORT, () => console.error(`[control] listening on http://localhost:${PORT}`));
