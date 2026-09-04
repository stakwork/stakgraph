import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import WebSocket from "ws";
import { attachAudioWebSocket } from "./ws.js";
import { _resetAuthState } from "../auth.js";
import type { SttEvent, SttService, SttStream, SttStreamOptions } from "./stt.js";

/** A service whose streams echo frame counts as partials and finalize on end. */
function fakeService(overrides: Partial<SttService> = {}): SttService & { opened: SttStreamOptions[] } {
  const opened: SttStreamOptions[] = [];
  const svc = {
    opened,
    modelDir: "/nowhere",
    hotwords: {} as SttService["hotwords"],
    sessions: {} as SttService["sessions"],
    available: async () => true,
    models: async () => [],
    ensureModel: async () => "/nowhere",
    transcribe: async () => {
      throw new Error("unused");
    },
    correct: async () => {},
    async openStream(o: SttStreamOptions = {}): Promise<SttStream> {
      opened.push(o);
      let frames = 0;
      let bytes = 0;
      const stream: SttStream = {
        model: o.model ?? "fake",
        partialModel: o.partialModel ?? null,
        hotwords: typeof o.hotwords === "string" ? o.hotwords : null,
        push(b) {
          frames++;
          bytes += b.byteLength;
          const ev: SttEvent[] = [{ type: "partial", text: `frame ${frames}` }];
          if (frames === 3) ev.push({ type: "final", index: 0, text: "three frames", words: [] });
          return ev;
        },
        pushSamples: () => [],
        end: () => [{ type: "final", index: 1, text: `done ${bytes} bytes`, words: [] }],
        flush: async () => {},
        close() {},
      };
      return stream;
    },
    ...overrides,
  };
  return svc as SttService & { opened: SttStreamOptions[] };
}

async function listen(server: Server): Promise<number> {
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  return (server.address() as { port: number }).port;
}

function connect(url: string, headers?: Record<string, string>): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url, { headers });
    ws.once("open", () => resolve(ws));
    ws.once("error", reject);
    ws.once("unexpected-response", (_req, res) => reject(new Error(`HTTP ${res.statusCode}`)));
  });
}

/** Collect every JSON message until the socket closes. */
function collect(ws: WebSocket): Promise<Record<string, unknown>[]> {
  const out: Record<string, unknown>[] = [];
  return new Promise((resolve) => {
    ws.on("message", (d) => out.push(JSON.parse(d.toString())));
    ws.on("close", () => resolve(out));
  });
}

describe("/audio/stream websocket", () => {
  let server: Server;
  let port: number;
  let detach: () => void;
  let dir: string;
  const originalKey = process.env["VEIN_API_KEY"];

  beforeEach(async () => {
    _resetAuthState();
    delete process.env["VEIN_API_KEY"];
    dir = await mkdtemp(join(tmpdir(), "vein-ws-"));
    server = createServer((_req, res) => res.writeHead(404).end());
  });
  afterEach(async () => {
    detach?.();
    await new Promise<void>((r) => server.close(() => r()));
    await rm(dir, { recursive: true, force: true });
    if (originalKey === undefined) delete process.env["VEIN_API_KEY"];
    else process.env["VEIN_API_KEY"] = originalKey;
    _resetAuthState();
  });

  it("start → binary frames → end yields ready, partials, finals, then a clean close", async () => {
    const svc = fakeService();
    detach = attachAudioWebSocket(server, svc);
    port = await listen(server);
    const ws = await connect(`ws://127.0.0.1:${port}/audio/stream`);
    const done = collect(ws);
    ws.send(JSON.stringify({ type: "start", model: "m", hotwords: "team", sampleRate: 16000, session: "s" }));
    for (let i = 0; i < 3; i++) ws.send(Buffer.alloc(320));
    ws.send(JSON.stringify({ type: "end" }));
    const msgs = await done;
    assert.deepEqual(msgs, [
      { type: "ready", model: "m", partialModel: null, hotwords: "team" },
      { type: "partial", text: "frame 1" },
      { type: "partial", text: "frame 2" },
      { type: "partial", text: "frame 3" },
      { type: "final", index: 0, text: "three frames", words: [] },
      { type: "final", index: 1, text: "done 960 bytes", words: [] },
    ]);
    assert.deepEqual(svc.opened, [{ model: "m", hotwords: "team", sampleRate: 16000, session: "s" }]);
  });

  it("audio before start is an error; a bad list name surfaces as an error message", async () => {
    detach = attachAudioWebSocket(server, fakeService());
    port = await listen(server);
    const ws = await connect(`ws://127.0.0.1:${port}/audio/stream`);
    const done = collect(ws);
    ws.send(Buffer.alloc(10));
    const msgs = await done;
    assert.equal(msgs[0]?.type, "error");
    assert.match(String(msgs[0]?.error), /start/);

    const failing = fakeService({
      openStream: async () => {
        throw new Error("unknown hotwords list \"nope\"");
      },
    });
    detach();
    detach = attachAudioWebSocket(server, failing);
    const ws2 = await connect(`ws://127.0.0.1:${port}/audio/stream`);
    const done2 = collect(ws2);
    ws2.send(JSON.stringify({ type: "start", hotwords: "nope" }));
    const msgs2 = await done2;
    assert.deepEqual(msgs2, [{ type: "error", error: 'unknown hotwords list "nope"' }]);
  });

  it("leaves other upgrade paths alone", async () => {
    detach = attachAudioWebSocket(server, fakeService());
    port = await listen(server);
    await assert.rejects(() => connect(`ws://127.0.0.1:${port}/other`));
  });

  it("with VEIN_API_KEY set: rejects without a key, accepts Bearer or ?key=", async () => {
    process.env["VEIN_API_KEY"] = "sekret";
    detach = attachAudioWebSocket(server, fakeService());
    port = await listen(server);
    await assert.rejects(() => connect(`ws://127.0.0.1:${port}/audio/stream`), /HTTP 401/);
    await assert.rejects(() => connect(`ws://127.0.0.1:${port}/audio/stream?key=wrong`), /HTTP 401/);
    const a = await connect(`ws://127.0.0.1:${port}/audio/stream?key=sekret`);
    a.close();
    const b = await connect(`ws://127.0.0.1:${port}/audio/stream`, { authorization: "Bearer sekret" });
    b.close();
  });
});
