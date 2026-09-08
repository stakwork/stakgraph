// ── Browser dictation ──────────────────────────────────────────────────────
//
// Microphone → AudioWorklet (PCM16LE, ~100 ms frames) → `/audio/stream`
// WebSocket (src/audio/ws.ts). Partials arrive on every change, finals on
// endpoint detection and on stop. The browser is one of strut's audio
// clients (plans/local-desktop-and-stt.md §1); the desktop and mobile hosts
// capture natively and speak the same protocol.

import { sttStreamUrl } from "./api";

export interface DictationOptions {
  model: string;
  /** `null` = single recognizer (the finals model also produces partials). */
  partialModel: string | null;
  /** Inline phrases, one per line, optional ` :score` (see hotwords.ts). */
  hotwords?: string[];
  /** Server-side session id: finals are logged under it for the dream cycle. */
  session?: string;
  onReady?: (info: { model: string; partialModel: string | null; hotwords: string | null }) => void;
  onPartial: (text: string) => void;
  onFinal: (text: string, index: number) => void;
  onError: (message: string) => void;
  /** The socket closed (after the trailing final on stop, or on error). */
  onClose: () => void;
}

export interface Dictation {
  /** Send `end`: the server flushes, emits the trailing final, then closes. */
  stop(): void;
}

// Runs inside the AudioWorkletGlobalScope, where `sampleRate` is a global.
const WORKLET_SRC = `
class P extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.n = 0; this.frame = Math.round(sampleRate / 10); }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]; if (!ch) return true;
    this.buf.push(new Float32Array(ch)); this.n += ch.length;
    if (this.n >= this.frame) {
      const out = new Int16Array(this.n); let o = 0;
      for (const b of this.buf) for (let i = 0; i < b.length; i++) out[o++] = Math.max(-32768, Math.min(32767, Math.round(b[i] * 32768)));
      this.port.postMessage(out.buffer, [out.buffer]); this.buf = []; this.n = 0;
    }
    return true;
  }
}
registerProcessor("pcm16", P);
`;

export function dictationSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof AudioWorkletNode !== "undefined" &&
    typeof WebSocket !== "undefined"
  );
}

export async function startDictation(opts: DictationOptions): Promise<Dictation> {
  const media = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  // Ask for 16 kHz (the engine's rate); a browser that ignores it reports
  // its own rate and strut resamples. The rate must not change mid-stream.
  const ctx = new AudioContext({ sampleRate: 16000 });
  const sampleRate = ctx.sampleRate;
  let ws: WebSocket | null = null;
  let node: AudioWorkletNode | null = null;
  let closed = false;

  const teardown = () => {
    if (closed) return;
    closed = true;
    node?.disconnect();
    media.getTracks().forEach((t) => t.stop());
    void ctx.close();
    ws = null;
    opts.onClose();
  };

  try {
    await ctx.audioWorklet.addModule(
      URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" })),
    );
  } catch (e) {
    media.getTracks().forEach((t) => t.stop());
    void ctx.close();
    throw e;
  }

  const socket = new WebSocket(sttStreamUrl());
  ws = socket;
  socket.binaryType = "arraybuffer";
  socket.onopen = () => {
    socket.send(
      JSON.stringify({
        type: "start",
        sampleRate,
        model: opts.model,
        partialModel: opts.partialModel,
        hotwords: opts.hotwords && opts.hotwords.length > 0 ? opts.hotwords : undefined,
        session: opts.session,
      }),
    );
  };
  socket.onmessage = (ev) => {
    let m: { type: string; text?: string; index?: number; error?: string; model?: string; partialModel?: string | null; hotwords?: string | null };
    try {
      m = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (m.type === "ready") opts.onReady?.({ model: m.model!, partialModel: m.partialModel ?? null, hotwords: m.hotwords ?? null });
    else if (m.type === "partial") opts.onPartial(m.text ?? "");
    else if (m.type === "final") opts.onFinal(m.text ?? "", m.index ?? 0);
    else if (m.type === "error") opts.onError(m.error ?? "dictation error");
  };
  socket.onerror = () => opts.onError("dictation socket error");
  socket.onclose = teardown;

  node = new AudioWorkletNode(ctx, "pcm16", { numberOfInputs: 1, numberOfOutputs: 0 });
  node.port.onmessage = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data as ArrayBuffer);
  };
  ctx.createMediaStreamSource(media).connect(node);

  return {
    stop() {
      // Stop capturing right away; keep the socket open for the trailing final.
      node?.disconnect();
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "end" }));
      else teardown();
    },
  };
}
