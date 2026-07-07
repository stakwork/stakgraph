import { Request, Response } from "express";
import { randomUUID } from "crypto";
import { ModelName } from "../aieo/src/index.js";
import { SessionConfig } from "../repo/session.js";
import * as asyncReqs from "../graph/reqs.js";
import { startTracking, endTracking } from "../busy.js";
import {
  createBus,
  filterStepContent,
  signEventsToken,
  registerAbortController,
  unregisterAbortController,
  abortRequest,
} from "../repo/events.js";
import { get_context, stream_context } from "./agent.js";

// ── Shared request body parser ───────────────────────────────────────────

/**
 * Coerce a request-body `headers` value into a clean Record<string, string>.
 * Accepts a plain object whose values are strings/numbers/booleans; drops
 * non-string values and ignores anything else. Returns undefined when empty.
 */
function normalizeHeaders(input: unknown): Record<string, string> | undefined {
  if (!input || typeof input !== "object" || Array.isArray(input)) return undefined;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(input as Record<string, unknown>)) {
    if (typeof k !== "string" || !k) continue;
    if (typeof v === "string") out[k] = v;
    else if (typeof v === "number" || typeof v === "boolean") out[k] = String(v);
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function parseGraphAgentBody(req: Request) {
  const prompt = req.body.prompt as string | undefined;
  const modelName = req.body.model as ModelName | undefined;
  const apiKey = req.body.apiKey as string | undefined;
  const baseUrl = req.body.baseUrl as string | undefined;
  const sessionId = (req.body.sessionId as string | undefined) || randomUUID();
  const sessionConfig = req.body.sessionConfig as SessionConfig | undefined;
  const stream = req.body.stream as boolean | undefined;
  const maxTurns =
    typeof req.body.maxTurns === "number" ? req.body.maxTurns : undefined;

  // Forward only the caller's own Authorization header (L402 token) so the
  // tools can attach it when calling /v2/nodes on jarvis-boltwall.
  // We deliberately do NOT accept authToken from the request body — allowing a
  // body-supplied token would let any authenticated caller inject an arbitrary
  // credential and have graph tool calls authorized against a different
  // account's L402 balance (IDOR).
  const authToken = req.headers["authorization"] as string | undefined;

  const headers = normalizeHeaders(req.body.headers);

  const context = req.body.context as
    | { selectedRefId: string; nodeType: string; title?: string }
    | undefined;

  if (context) {
    console.log(
      `[graph_agent] context_present selectedRefId=${context.selectedRefId} nodeType=${context.nodeType}`,
    );
  }

  // Parse optional _metadata with a 64 KB size cap to prevent unbounded writes to SESSIONS_DIR.
  const MAX_METADATA_BYTES = 64 * 1024; // 64 KB
  let _metadata: unknown = undefined;
  if (req.body._metadata !== undefined) {
    const serialized = JSON.stringify(req.body._metadata);
    if (serialized.length > MAX_METADATA_BYTES) {
      // Return sentinel so the handler can reject with 400.
      return { prompt, modelName, apiKey, baseUrl, sessionId, sessionConfig, stream, maxTurns, authToken, headers, context, _metadata: null as unknown, _metadataOversized: true };
    }
    _metadata = req.body._metadata;
  }

  return { prompt, modelName, apiKey, baseUrl, sessionId, sessionConfig, stream, maxTurns, authToken, headers, context, _metadata, _metadataOversized: false };
}

// ── POST /graph_agent ────────────────────────────────────────────────────

/**
 * Main graph agent endpoint.
 *
 * - `stream: true`  → direct SSE (AI SDK UI message stream)
 * - `stream: false` → async job, returns { request_id, status, sessionId, events_token }
 *
 * Future v2 note: the request body reserves a `graphs[]` array field for
 * cross-graph queries (e.g. querying both knowledge graph and code graph).
 * This is NOT implemented in v1 but callers may include the field without
 * breaking anything.
 */
export async function graph_agent(req: Request, res: Response) {
  console.log("===> graph_agent", req.method, req.path, {
    hasPrompt: Boolean(req.body?.prompt),
    stream: Boolean(req.body?.stream),
    hasApiKey: Boolean(req.body?.apiKey),
    modelName: req.body?.model || "(none)",
    hasAuthToken: Boolean(req.headers["authorization"] || req.body?.authToken),
  });

  const body = parseGraphAgentBody(req);

  if (!body.prompt) {
    res.status(400).json({ error: "Missing prompt" });
    return;
  }

  if (body._metadataOversized) {
    res.status(400).json({ error: "_metadata exceeds maximum allowed size (64 KB)" });
    return;
  }

  // ── Streaming path: direct SSE response ─────────────────────────────
  if (body.stream) {
    const opId = startTracking("graph_agent_stream");
    const request_id = randomUUID();
    const abortController = registerAbortController(request_id);
    // Also register under sessionId so abort works with either key
    if (body.sessionId !== request_id) {
      registerAbortController(body.sessionId, abortController);
    }

    try {
      const { streamResult, finalizeSession } = await stream_context({
        prompt: body.prompt,
        modelName: body.modelName,
        apiKey: body.apiKey,
        baseUrl: body.baseUrl,
        sessionId: body.sessionId,
        sessionConfig: body.sessionConfig,
        maxTurns: body.maxTurns,
        authToken: body.authToken,
        abortSignal: abortController.signal,
        headers: body.headers,
        context: body.context,
        _metadata: body._metadata,
      });

      const streamResponse = streamResult.toUIMessageStreamResponse();

      res.status(streamResponse.status);
      res.setHeader("X-Request-Id", request_id);
      res.setHeader("X-Session-Id", body.sessionId);
      streamResponse.headers.forEach((value: string, key: string) => {
        res.setHeader(key, value);
      });

      const reader = streamResponse.body?.getReader();
      if (!reader) {
        res.status(500).json({ error: "No stream body" });
        endTracking(opId);
        return;
      }

      const onClientClose = () => {
        if (!abortController.signal.aborted) {
          console.log(`[graph_agent] Client disconnected; aborting request_id=${request_id} session=${body.sessionId}`);
          try { abortController.abort(); } catch (_) {}
        }
      };
      res.on("close", onClientClose);

      const pump = async () => {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          res.write(value);
        }
        res.end();
      };

      pump()
        .catch((err) => {
          console.error("[graph_agent] Stream error:", err);
          if (!res.headersSent) {
            res.status(500).json({ error: "Stream error" });
          } else {
            res.end();
          }
        })
        .finally(async () => {
          res.off("close", onClientClose);
          await finalizeSession();
          unregisterAbortController(request_id);
          if (body.sessionId !== request_id) {
            unregisterAbortController(body.sessionId);
          }
          endTracking(opId);
        });

      return;
    } catch (error: any) {
      console.error("[graph_agent] Stream setup error:", error);
      unregisterAbortController(request_id);
      if (body.sessionId !== request_id) {
        unregisterAbortController(body.sessionId);
      }
      endTracking(opId);
      if (!res.headersSent) {
        res.status(500).json({ error: error.message || "Internal server error" });
      }
      return;
    }
  }

  // ── Non-streaming path: async job with event bus ─────────────────────
  const request_id = asyncReqs.startReq();
  const opId = startTracking("graph_agent");

  const bus = createBus(request_id);

  const abortController = registerAbortController(request_id);
  if (body.sessionId && body.sessionId !== request_id) {
    registerAbortController(body.sessionId, abortController);
  }

  let events_token: string | undefined;
  try {
    events_token = signEventsToken(request_id);
  } catch (_) {
    // API_TOKEN not set — JWT-gated SSE won't be available
  }

  try {
    get_context({
      prompt: body.prompt,
      modelName: body.modelName,
      apiKey: body.apiKey,
      baseUrl: body.baseUrl,
      sessionId: body.sessionId,
      sessionConfig: body.sessionConfig,
      maxTurns: body.maxTurns,
      authToken: body.authToken,
      abortSignal: abortController.signal,
      headers: body.headers,
      context: body.context,
      _metadata: body._metadata,
      onStepEvent: (content) => {
        const events = filterStepContent(content);
        for (const ev of events) bus.emit(ev);
      },
    })
      .then((result) => {
        asyncReqs.finishReq(request_id, {
          success: true,
          answer: result.answer,
          cited_ref_ids: result.cited_ref_ids,
          usage: result.usage,
          sessionId: result.sessionId,
        });
        bus.emit({
          type: "done",
          result: {
            answer: result.answer,
            cited_ref_ids: result.cited_ref_ids,
            usage: result.usage,
          },
          timestamp: new Date().toISOString(),
        });
      })
      .catch((error) => {
        const aborted = abortController.signal.aborted;
        if (aborted) {
          console.log(`[graph_agent] Run aborted: ${request_id}`);
          asyncReqs.failReq(request_id, "aborted");
          bus.emit({
            type: "error",
            error: "aborted",
            timestamp: new Date().toISOString(),
          });
        } else {
          console.error("[graph_agent] Background work failed:", error);
          asyncReqs.failReq(request_id, error.message || error.toString());
          bus.emit({
            type: "error",
            error: error.message || error.toString(),
            timestamp: new Date().toISOString(),
          });
        }
      })
      .finally(() => {
        unregisterAbortController(request_id);
        if (body.sessionId && body.sessionId !== request_id) {
          unregisterAbortController(body.sessionId);
        }
        endTracking(opId);
      });

    res.json({
      request_id,
      status: "pending",
      sessionId: body.sessionId,
      ...(events_token && { events_token }),
    });
  } catch (error) {
    console.error("[graph_agent] Startup error:", error);
    asyncReqs.failReq(request_id, error);
    unregisterAbortController(request_id);
    if (body.sessionId && body.sessionId !== request_id) {
      unregisterAbortController(body.sessionId);
    }
    res.status(500).json({ error: "Internal server error" });
    endTracking(opId);
  }
}

// ── POST /graph_agent/abort ──────────────────────────────────────────────

export async function abort_graph_agent(req: Request, res: Response) {
  const request_id =
    (req.body?.request_id as string | undefined) ||
    (req.query?.request_id as string | undefined);
  const sessionId =
    (req.body?.sessionId as string | undefined) ||
    (req.body?.session_id as string | undefined) ||
    (req.query?.sessionId as string | undefined) ||
    (req.query?.session_id as string | undefined);
  const key = request_id || sessionId;
  console.log("===> POST /graph_agent/abort", { request_id, sessionId });
  if (!key) {
    res.status(400).json({ error: "Provide request_id or sessionId" });
    return;
  }
  const aborted = abortRequest(key);
  if (!aborted) {
    res.status(404).json({ aborted: false, error: "No active run found for the given key" });
    return;
  }
  res.json({ aborted: true, key });
}
