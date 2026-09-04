/**
 * `GET /audio/stream` — live dictation over a WebSocket
 * (plans/local-desktop-and-stt.md §4.4).
 *
 * Protocol (client → server):
 *   {"type":"start", model?, partialModel?, hotwords?, hotwordsScore?,
 *    sampleRate?, session?, endpoint?}          — SttStreamOptions, verbatim
 *   <binary>                                    — PCM16LE at sampleRate
 *   {"type":"end"}                              — flush; server closes after the final
 * Server → client:
 *   {"type":"ready", model, partialModel, hotwords}
 *   {"type":"partial", text}
 *   {"type":"final", index, text, words}
 *   {"type":"error", error}                     — then close
 *
 * Hono's node-ws adapter doesn't support @hono/node-server 2.x yet, and the
 * upgrade has to happen on the Node server anyway, so this hooks `ws`
 * straight onto the http.Server that `listen()` creates. Vein's first
 * client-to-server streaming route; the SSE-based rest is untouched.
 *
 * Auth: `Authorization: Bearer <VEIN_API_KEY>` or `?key=` — a browser's
 * WebSocket cannot set headers.
 */
import type { IncomingMessage, Server } from "node:http";
import type { Duplex } from "node:stream";
import { WebSocketServer, type WebSocket } from "ws";
import { apiKeyMatches } from "../auth.js";
import type { SttService, SttStream, SttStreamOptions } from "./stt.js";

export const AUDIO_STREAM_PATH = "/audio/stream";

export interface AttachOptions {
  /** Mount prefix when vein sits under a parent router (e.g. `/lab`). */
  basePath?: string;
  path?: string;
}

/** Attach the dictation socket to a Node http server. Returns a detach fn. */
export function attachAudioWebSocket(server: Server, stt: SttService, opts: AttachOptions = {}): () => void {
  const path = (opts.basePath ?? "") + (opts.path ?? AUDIO_STREAM_PATH);
  const wss = new WebSocketServer({ noServer: true });

  const onUpgrade = (req: IncomingMessage, socket: Duplex, head: Buffer) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (url.pathname !== path) {
      // Not ours. If nobody else handles upgrades the socket would hang
      // open forever, so answer 404 when we're the only listener.
      if (server.listenerCount("upgrade") === 1) {
        socket.write("HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n");
        socket.destroy();
      }
      return;
    }
    if (!apiKeyMatches(req.headers.authorization, url.searchParams.get("key"))) {
      socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return;
    }
    wss.handleUpgrade(req, socket, head, (ws) => handleSocket(ws, stt));
  };
  server.on("upgrade", onUpgrade);
  return () => {
    server.off("upgrade", onUpgrade);
    wss.close();
  };
}

/** Drive one socket. Exported for tests (any `ws`-shaped socket works). */
export function handleSocket(ws: WebSocket, stt: SttService): void {
  let stream: SttStream | null = null;
  let opening: Promise<void> | null = null;
  let finished = false;

  const send = (msg: Record<string, unknown>) => {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(msg));
  };
  const fail = (err: unknown) => {
    send({ type: "error", error: err instanceof Error ? err.message : String(err) });
    ws.close(1011, "error");
  };
  const emit = (events: ReturnType<SttStream["push"]>) => {
    for (const ev of events) send(ev);
  };

  ws.on("message", (data, isBinary) => {
    if (finished) return;
    if (isBinary) {
      if (!stream) {
        if (opening) {
          // Audio arriving before the recognizer is ready: wait, then feed.
          const chunk = toBytes(data);
          opening.then(() => stream && emit(stream.push(chunk))).catch(fail);
          return;
        }
        return fail(new Error('send {"type":"start"} before audio'));
      }
      try {
        emit(stream.push(toBytes(data)));
      } catch (e) {
        fail(e);
      }
      return;
    }

    let msg: { type?: string } & SttStreamOptions;
    try {
      msg = JSON.parse(toBytes(data).toString());
    } catch {
      return fail(new Error("expected JSON control message or binary PCM"));
    }
    if (msg.type === "start") {
      if (stream || opening) return fail(new Error("stream already started"));
      const { type: _t, ...o } = msg;
      opening = stt
        .openStream(o)
        .then((s) => {
          stream = s;
          send({ type: "ready", model: s.model, partialModel: s.partialModel, hotwords: s.hotwords });
        })
        .catch((e) => {
          opening = null;
          fail(e);
        });
      return;
    }
    if (msg.type === "end") {
      finished = true;
      const finish = async () => {
        if (stream) {
          try {
            emit(stream.end());
          } catch (e) {
            return fail(e);
          }
          // Session finals are on disk before the client can correct them.
          await stream.flush();
        }
        ws.close(1000, "done");
      };
      if (opening) opening.then(finish).catch(fail);
      else finish().catch(fail);
      return;
    }
    fail(new Error(`unknown message type ${JSON.stringify(msg.type)}`));
  });

  ws.on("close", () => {
    stream?.close();
    stream = null;
  });
}

function toBytes(data: unknown): Buffer {
  if (Buffer.isBuffer(data)) return data;
  if (Array.isArray(data)) return Buffer.concat(data as Buffer[]);
  if (data instanceof ArrayBuffer) return Buffer.from(data);
  return Buffer.from(String(data));
}
