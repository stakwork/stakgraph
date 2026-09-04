/**
 * HTTP surface over the STT service (plans/local-desktop-and-stt.md §4.4).
 * The WebSocket lives in ws.ts. Everything here is gated by `requireApiKey`
 * (permissive in dev, like the rest of vein).
 */
import type { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { requireApiKey } from "../auth.js";
import { findModel } from "./models.js";
import { SttUnavailableError, type SttService, type SttStreamOptions } from "./stt.js";

export function audioRoutes(app: Hono, stt: SttService): void {
  app.use("/audio/*", requireApiKey);

  app.get("/audio/models", async (c) => {
    return c.json({ available: await stt.available(), modelDir: stt.modelDir, models: await stt.models() });
  });

  // SSE: {"phase":"download","received","total"} … {"phase":"extract"} … {"phase":"done"}
  app.post("/audio/models/:id/download", async (c) => {
    const id = c.req.param("id");
    if (!findModel(id)) return c.json({ error: `unknown model ${JSON.stringify(id)}` }, 404);
    return streamSSE(c, async (s) => {
      let last = 0;
      try {
        await stt.ensureModel(id, (p) => {
          // Throttle byte progress to ~every 1 MB so the stream stays light.
          if (p.phase === "download" && p.received - last < 1_000_000 && p.received !== p.total) return;
          if (p.phase === "download") last = p.received;
          void s.writeSSE({ event: "progress", data: JSON.stringify(p) });
        });
        await s.writeSSE({ event: "progress", data: JSON.stringify({ phase: "done" }) });
      } catch (e) {
        await s.writeSSE({ event: "error", data: JSON.stringify({ error: (e as Error).message }) });
      }
    });
  });

  // Raw audio/wav body. Query: model, partialModel (ignored: batch is
  // single-recognizer), hotwords (list name), hotwordsScore, session.
  app.post("/audio/transcribe", async (c) => {
    const q = c.req.query();
    const opts: Omit<SttStreamOptions, "sampleRate"> = {};
    if (q["model"]) opts.model = q["model"];
    if (q["hotwords"]) opts.hotwords = q["hotwords"];
    if (q["hotwordsScore"]) opts.hotwordsScore = Number(q["hotwordsScore"]);
    if (q["session"]) opts.session = q["session"];
    const body = new Uint8Array(await c.req.arrayBuffer());
    if (body.byteLength < 44) return c.json({ error: "expected a WAV body" }, 400);
    try {
      return c.json(await stt.transcribe(body, opts));
    } catch (e) {
      return errorResponse(c, e);
    }
  });

  // ── Hotword lists (the dream cycle's promotion artifact) ────────────────

  app.get("/audio/hotwords", async (c) => c.json({ lists: await stt.hotwords.list() }));

  app.get("/audio/hotwords/:name", async (c) => {
    const text = await stt.hotwords.get(c.req.param("name"));
    if (text == null) return c.json({ error: "not found" }, 404);
    return c.text(text);
  });

  // Body: text/plain (one phrase per line, optional ` :score`) or JSON
  // { "phrases": ["…", …] } / { "text": "…" }.
  app.put("/audio/hotwords/:name", async (c) => {
    const name = c.req.param("name");
    let text: string;
    if ((c.req.header("content-type") ?? "").includes("application/json")) {
      const body = (await c.req.json()) as { phrases?: string[]; text?: string };
      text = body.text ?? (body.phrases ?? []).join("\n");
    } else {
      text = await c.req.text();
    }
    try {
      const list = await stt.hotwords.put(name, text);
      return c.json({ name, count: list.length });
    } catch (e) {
      return c.json({ error: (e as Error).message }, 400);
    }
  });

  app.delete("/audio/hotwords/:name", async (c) => {
    const ok = await stt.hotwords.delete(c.req.param("name"));
    return ok ? c.json({ ok: true }) : c.json({ error: "not found" }, 404);
  });

  // ── Sessions (finals + corrections) ─────────────────────────────────────

  app.get("/audio/sessions", async (c) => c.json({ sessions: await stt.sessions.list() }));

  app.get("/audio/sessions/:id", async (c) => {
    const entries = await stt.sessions.get(c.req.param("id"));
    if (!entries) return c.json({ error: "not found" }, 404);
    return c.json({ id: c.req.param("id"), entries });
  });

  app.post("/audio/sessions/:id/corrections", async (c) => {
    const body = (await c.req.json().catch(() => null)) as { index?: number; text?: string } | null;
    if (!body || typeof body.index !== "number" || typeof body.text !== "string") {
      return c.json({ error: "expected { index: number, text: string }" }, 400);
    }
    await stt.correct(c.req.param("id"), body.index, body.text);
    return c.json({ ok: true });
  });
}

function errorResponse(c: { json: (b: unknown, s: 400 | 501) => Response }, e: unknown): Response {
  if (e instanceof SttUnavailableError) return c.json({ error: e.message }, 501);
  return c.json({ error: (e as Error).message }, 400);
}
