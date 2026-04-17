import { EventEmitter } from "events";
import jwt from "jsonwebtoken";
import { Response } from "express";

// ── JWT helpers ──────────────────────────────────────────────────────────

const JWT_EXPIRY = "1h";

function getSecret(): string {
  const secret = process.env.API_TOKEN;
  if (!secret) throw new Error("API_TOKEN is required for JWT signing");
  return secret;
}

export interface EventsTokenPayload {
  request_id: string;
  iat?: number;
  exp?: number;
}

/** Sign a short-lived JWT scoped to a single request_id. */
export function signEventsToken(request_id: string): string {
  return jwt.sign({ request_id } as EventsTokenPayload, getSecret(), {
    expiresIn: JWT_EXPIRY,
  });
}

/**
 * Verify a JWT and return the payload.
 * Throws if expired, malformed, or wrong secret.
 */
export function verifyEventsToken(token: string): EventsTokenPayload {
  return jwt.verify(token, getSecret()) as EventsTokenPayload;
}

// ── Generic API access tokens (for browser SPA) ──────────────────────────

export interface ApiTokenPayload {
  scope: "api";
  iat?: number;
  exp?: number;
}

/**
 * Sign a short-lived JWT granting general API access.
 * Injected into the SPA's index.html after Basic Auth / x-api-token passes.
 */
export function signApiToken(expiresIn: jwt.SignOptions["expiresIn"] = "1h"): string {
  return jwt.sign({ scope: "api" } as ApiTokenPayload, getSecret(), {
    expiresIn,
  });
}

/**
 * Verify an API access JWT. Throws if invalid/expired.
 */
export function verifyApiToken(token: string): ApiTokenPayload {
  const payload = jwt.verify(token, getSecret()) as ApiTokenPayload;
  if (payload.scope !== "api") {
    throw new Error("Invalid token scope");
  }
  return payload;
}

// ── Per-request event bus ────────────────────────────────────────────────

export type StepEventType = "tool_call" | "text" | "done" | "error";

export interface StepEvent {
  type: StepEventType;
  /** For tool_call: the tool name */
  toolName?: string;
  /** For tool_call: the tool input args */
  input?: unknown;
  /** For text: the LLM text chunk */
  text?: string;
  /** For error: the error message */
  error?: string;
  /** For done: the final result (same shape as asyncReqs result) */
  result?: unknown;
  /** ISO timestamp */
  timestamp: string;
}

class RequestEventBus {
  private emitter = new EventEmitter();
  /** Auto-cleanup timer */
  private timeout: ReturnType<typeof setTimeout> | null = null;
  private _ended = false;

  constructor(private ttlMs: number = 60 * 60 * 1000) {
    // Auto-destroy after TTL even if nobody listens
    this.timeout = setTimeout(() => this.destroy(), this.ttlMs);
  }

  get ended() {
    return this._ended;
  }

  emit(event: StepEvent) {
    if (this._ended) return;
    this.emitter.emit("step", event);
    if (event.type === "done" || event.type === "error") {
      this._ended = true;
      // Give listeners a moment to flush, then clean up
      setTimeout(() => this.destroy(), 5_000);
    }
  }

  /** Subscribe — returns an unsubscribe function. */
  subscribe(listener: (event: StepEvent) => void): () => void {
    this.emitter.on("step", listener);
    return () => this.emitter.removeListener("step", listener);
  }

  destroy() {
    if (this.timeout) clearTimeout(this.timeout);
    this.emitter.removeAllListeners();
  }
}

// Global registry keyed by request_id
const buses = new Map<string, RequestEventBus>();

export function createBus(request_id: string): RequestEventBus {
  const bus = new RequestEventBus();
  buses.set(request_id, bus);
  // Clean up map entry when bus is destroyed
  const origDestroy = bus.destroy.bind(bus);
  bus.destroy = () => {
    buses.delete(request_id);
    origDestroy();
  };
  return bus;
}

export function getBus(request_id: string): RequestEventBus | undefined {
  return buses.get(request_id);
}

// ── SSE helper ───────────────────────────────────────────────────────────

/**
 * Pipe a RequestEventBus into an Express SSE response.
 * Handles connection cleanup when the client disconnects.
 */
export function pipeToSSE(bus: RequestEventBus, res: Response) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no", // disable nginx buffering
  });
  res.flushHeaders();

  const unsub = bus.subscribe((event) => {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
    if (event.type === "done" || event.type === "error") {
      res.end();
    }
  });

  // If the bus already ended before we subscribed (race), close immediately
  if (bus.ended) {
    res.end();
    unsub();
    return;
  }

  res.on("close", () => {
    unsub();
  });
}

// ── Step content filter ──────────────────────────────────────────────────

/**
 * Extract SSE-worthy events from an onStepFinish content array.
 * Includes tool_call (name + input) and text, but NOT tool-results.
 */
export function filterStepContent(content: any[]): StepEvent[] {
  const events: StepEvent[] = [];
  const ts = new Date().toISOString();
  for (const item of content) {
    if (item.type === "tool-call") {
      events.push({
        type: "tool_call",
        toolName: item.toolName,
        input: item.input,
        timestamp: ts,
      });
    } else if (item.type === "text" && item.text) {
      events.push({
        type: "text",
        text: item.text,
        timestamp: ts,
      });
    }
    // Deliberately skip tool-result — those can be huge
  }
  return events;
}
