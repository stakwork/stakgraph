/**
 * Speech-to-text service over sherpa-onnx (plans/local-desktop-and-stt.md §4).
 *
 * Owns: model download + verification, recognizer construction and caching,
 * the streaming session (PCM in → partial/final events out), and the batch
 * `transcribe` that runs the same streaming path over a WAV. Routes
 * (`routes.ts`) and the WebSocket (`ws.ts`) are thin over this.
 *
 * `sherpa-onnx-node` is an optionalDependency, imported lazily: a vein without
 * the addon boots, and `available()` says so. Tests inject a fake `engine`.
 *
 * Two-recognizer streams: when `partialModel` is set, a fast greedy model
 * produces the partials and the hotword-capable `model` produces finals and
 * owns endpoint detection (see §4.4 for why — sherpa's NeMo online
 * transducer is greedy-only, so speed and biasing come from different
 * families).
 */
import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { join } from "node:path";
import { pipeline } from "node:stream/promises";
import { Transform } from "node:stream";
import { randomUUID } from "node:crypto";
import { compileHotwords, parseHotwords, HotwordsStore, type Hotword } from "./hotwords.js";
import {
  DEFAULT_MODEL,
  DEFAULT_PARTIAL_MODEL,
  STT_MODELS,
  modelDirFromEnv,
  pickModelFiles,
  requireModel,
  sttModelPath,
  type SttModel,
} from "./models.js";
import { SessionStore } from "./sessions.js";

// ── Engine surface (what we use of sherpa-onnx-node; fakeable) ────────────

export interface EngineWaveform {
  samples: Float32Array;
  sampleRate: number;
}
export interface EngineResult {
  text: string;
  tokens?: string[];
  timestamps?: number[];
  start_time?: number;
}
export interface EngineStream {
  acceptWaveform(w: EngineWaveform): void;
  inputFinished(): void;
}
export interface EngineRecognizer {
  createStream(): EngineStream;
  isReady(s: EngineStream): boolean;
  decode(s: EngineStream): void;
  isEndpoint(s: EngineStream): boolean;
  reset(s: EngineStream): void;
  getResult(s: EngineStream): EngineResult;
}
export interface SttEngine {
  OnlineRecognizer: new (config: Record<string, unknown>) => EngineRecognizer;
  readWave(path: string): EngineWaveform;
}

/** Lazy sherpa import; null when the optional addon isn't installed. */
export async function loadSherpaEngine(): Promise<SttEngine | null> {
  try {
    const mod = (await import("sherpa-onnx-node")) as { default?: SttEngine } & SttEngine;
    return mod.default ?? mod;
  } catch {
    return null;
  }
}

// ── Public types ───────────────────────────────────────────────────────────

export interface SttWord {
  text: string;
  /** Seconds from the start of the stream. */
  start: number;
}

export type SttEvent =
  | { type: "partial"; text: string }
  | { type: "final"; index: number; text: string; words: SttWord[] };

export interface EndpointRules {
  /** Trailing silence (s) that ends a segment even with no speech yet. */
  rule1?: number;
  /** Trailing silence (s) after speech that ends a segment. */
  rule2?: number;
  /** Utterance length (s) after which a segment is cut regardless. */
  rule3?: number;
}

export interface SttStreamOptions {
  /** Finals model (hotword-capable). Default `VEIN_STT_MODEL` / catalog default. */
  model?: string;
  /** Fast model for partials; `null` for single-recognizer mode. Default
   *  `VEIN_STT_PARTIAL_MODEL` / catalog default. */
  partialModel?: string | null;
  /** A stored list name, or phrases inline. */
  hotwords?: string | readonly string[] | readonly Hotword[];
  /** Global per-token boost for phrases without their own `:score`. */
  hotwordsScore?: number;
  sampleRate?: number;
  /** Log finals + accept corrections under this id (`SessionStore`). */
  session?: string;
  endpoint?: EndpointRules;
}

export interface SttStream {
  readonly model: string;
  readonly partialModel: string | null;
  readonly hotwords: string | null;
  /** Feed PCM16LE bytes at the stream's sample rate. */
  push(pcm16le: Uint8Array): SttEvent[];
  /** Feed float32 samples (any rate; sherpa resamples). */
  pushSamples(samples: Float32Array, sampleRate: number): SttEvent[];
  /** Flush: pads silence so the last chunk decodes, emits the trailing final. */
  end(): SttEvent[];
  /** Resolves once every session-log write so far has landed — await it
   *  before telling a client it may correct the finals. */
  flush(): Promise<void>;
  close(): void;
}

export interface TranscribeResult {
  text: string;
  segments: { text: string; words: SttWord[] }[];
  model: string;
  hotwords: string | null;
  durationMs: number;
}

export type DownloadProgress =
  | { phase: "download"; received: number; total: number }
  | { phase: "extract" }
  | { phase: "done" };

export interface SttModelStatus extends SttModel {
  installed: boolean;
  /** Which entry env/default resolution picks. */
  default: "model" | "partialModel" | null;
}

export interface SttService {
  readonly modelDir: string;
  readonly hotwords: HotwordsStore;
  readonly sessions: SessionStore;
  /** Whether the sherpa addon loads in this process. */
  available(): Promise<boolean>;
  models(): Promise<SttModelStatus[]>;
  /** Download + verify + extract (idempotent, deduped). Resolves to the dir. */
  ensureModel(id: string, onProgress?: (p: DownloadProgress) => void): Promise<string>;
  openStream(opts?: SttStreamOptions): Promise<SttStream>;
  transcribe(wav: Uint8Array, opts?: Omit<SttStreamOptions, "sampleRate">): Promise<TranscribeResult>;
  /** Append a user's correction to a session's final. */
  correct(session: string, index: number, text: string): Promise<void>;
}

export interface SttServiceOptions {
  /** Local data dir (sessions, hotword lists, temp WAVs). */
  dataDir: string;
  modelDir?: string;
  engine?: () => Promise<SttEngine | null>;
  fetchImpl?: typeof fetch;
  env?: Record<string, string | undefined>;
  log?: (msg: string) => void;
}

// ── Implementation ─────────────────────────────────────────────────────────

const FRAME_SECONDS = 0.1;
const DEFAULT_HOTWORDS_SCORE = 2;
const DEFAULT_ENDPOINT: Required<EndpointRules> = { rule1: 2.4, rule2: 1.0, rule3: 20 };

interface Loaded {
  model: SttModel;
  dir: string;
  files: ReturnType<typeof pickModelFiles>;
}

export function createStt(opts: SttServiceOptions): SttService {
  const env = opts.env ?? process.env;
  const modelDir = opts.modelDir ?? modelDirFromEnv(env);
  const loadEngine = opts.engine ?? loadSherpaEngine;
  const fetchImpl = opts.fetchImpl ?? fetch;
  const log = opts.log ?? ((m: string) => console.log(`[stt] ${m}`));
  const hotwords = new HotwordsStore(opts.dataDir);
  const sessions = new SessionStore(opts.dataDir);

  let engineP: Promise<SttEngine | null> | undefined;
  const engine = () => (engineP ??= loadEngine());
  const requireEngine = async (): Promise<SttEngine> => {
    const e = await engine();
    if (!e) throw new SttUnavailableError();
    return e;
  };

  const downloads = new Map<string, Promise<string>>();
  const recognizers = new Map<string, Promise<EngineRecognizer>>();

  const defaultModelId = () => env["VEIN_STT_MODEL"] ?? DEFAULT_MODEL;
  const defaultPartialId = (): string | null => {
    const v = env["VEIN_STT_PARTIAL_MODEL"];
    if (v === "" || v === "none" || v === "off") return null;
    return v ?? DEFAULT_PARTIAL_MODEL;
  };

  async function installed(id: string): Promise<boolean> {
    try {
      await stat(join(sttModelPath(modelDir, id), ".ok"));
      return true;
    } catch {
      return false;
    }
  }

  async function ensureModel(id: string, onProgress?: (p: DownloadProgress) => void): Promise<string> {
    const model = requireModel(id);
    const dir = sttModelPath(modelDir, id);
    if (await installed(id)) return dir;
    let p = downloads.get(id);
    if (!p) {
      p = download(model, dir, onProgress).finally(() => downloads.delete(id));
      downloads.set(id, p);
    }
    return p;
  }

  async function download(model: SttModel, dir: string, onProgress?: (p: DownloadProgress) => void): Promise<string> {
    await mkdir(join(modelDir, "stt"), { recursive: true });
    const archive = `${dir}.tar.bz2.part`;
    log(`downloading ${model.id} (${Math.round(model.bytes / 1e6)} MB)`);
    const res = await fetchImpl(model.url);
    if (!res.ok || !res.body) throw new Error(`download failed for ${model.id}: HTTP ${res.status}`);
    const total = Number(res.headers.get("content-length") ?? model.bytes);
    const hash = createHash("sha256");
    let received = 0;
    const meter = new Transform({
      transform(chunk: Buffer, _enc, cb) {
        hash.update(chunk);
        received += chunk.length;
        onProgress?.({ phase: "download", received, total });
        cb(null, chunk);
      },
    });
    await pipeline(res.body as unknown as NodeJS.ReadableStream, meter, createWriteStream(archive));
    const digest = hash.digest("hex");
    if (digest !== model.sha256) {
      await rm(archive, { force: true });
      throw new Error(`sha256 mismatch for ${model.id}: expected ${model.sha256}, got ${digest}`);
    }
    onProgress?.({ phase: "extract" });
    const scratch = `${dir}.extract`;
    await rm(scratch, { recursive: true, force: true });
    await mkdir(scratch, { recursive: true });
    await untar(archive, scratch);
    await rm(dir, { recursive: true, force: true });
    await rename(join(scratch, model.archiveDir), dir);
    await rm(scratch, { recursive: true, force: true });
    await rm(archive, { force: true });
    await writeFile(join(dir, ".ok"), new Date().toISOString());
    onProgress?.({ phase: "done" });
    log(`installed ${model.id} → ${dir}`);
    return dir;
  }

  async function load(id: string): Promise<Loaded> {
    const model = requireModel(id);
    const dir = await ensureModel(id);
    const files = pickModelFiles(await readdir(dir));
    return { model, dir, files };
  }

  async function resolveHotwords(
    spec: SttStreamOptions["hotwords"],
  ): Promise<{ name: string | null; list: Hotword[] }> {
    if (spec == null) return { name: null, list: [] };
    if (typeof spec === "string") {
      const text = await hotwords.get(spec);
      if (text == null) throw new Error(`unknown hotwords list "${spec}"`);
      return { name: spec, list: parseHotwords(text) };
    }
    const list = spec.map((h) => (typeof h === "string" ? { phrase: h } : h)).filter((h) => h.phrase.trim());
    return { name: list.length ? "inline" : null, list };
  }

  async function recognizer(
    loaded: Loaded,
    list: readonly Hotword[],
    score: number,
    ep: Required<EndpointRules>,
  ): Promise<EngineRecognizer> {
    const e = await requireEngine();
    const biased = list.length > 0 && loaded.model.hotwords;
    const compiled = biased ? await compileHotwords(loaded.dir, list) : null;
    const key = [loaded.model.id, compiled?.hash ?? "-", biased ? score : "-", ep.rule1, ep.rule2, ep.rule3].join("|");
    let p = recognizers.get(key);
    if (!p) {
      const { dir, files } = loaded;
      const modelConfig: Record<string, unknown> = {
        transducer: { encoder: join(dir, files.encoder), decoder: join(dir, files.decoder), joiner: join(dir, files.joiner) },
        tokens: join(dir, files.tokens),
        numThreads: 2,
        provider: "cpu",
        debug: 0,
      };
      if (compiled) {
        // §4.2: without modelingUnit sherpa defaults to cjkchar and the list
        // silently does nothing.
        modelConfig["modelingUnit"] = "bpe";
        modelConfig["bpeVocab"] = compiled.vocab;
      }
      const config: Record<string, unknown> = {
        featConfig: { sampleRate: 16000, featureDim: 80 },
        modelConfig,
        decodingMethod: compiled ? "modified_beam_search" : "greedy_search",
        maxActivePaths: 4,
        enableEndpoint: true,
        rule1MinTrailingSilence: ep.rule1,
        rule2MinTrailingSilence: ep.rule2,
        rule3MinUtteranceLength: ep.rule3,
      };
      if (compiled) {
        config["hotwordsFile"] = compiled.file;
        config["hotwordsScore"] = score;
      }
      p = Promise.resolve().then(() => {
        const t0 = Date.now();
        const r = new e.OnlineRecognizer(config);
        log(`recognizer ${loaded.model.id}${compiled ? ` +hotwords(${compiled.hash})` : ""} ready in ${Date.now() - t0} ms`);
        return r;
      });
      p.catch(() => recognizers.delete(key));
      recognizers.set(key, p);
    }
    return p;
  }

  async function openStream(o: SttStreamOptions = {}): Promise<SttStream> {
    const modelId = o.model ?? defaultModelId();
    const partialId = o.partialModel === undefined ? defaultPartialId() : o.partialModel;
    const ep = { ...DEFAULT_ENDPOINT, ...stripUndefined(o.endpoint ?? {}) };
    const { name: hotwordsName, list } = await resolveHotwords(o.hotwords);
    const score = o.hotwordsScore ?? DEFAULT_HOTWORDS_SCORE;

    const main = await load(modelId);
    const mainRec = await recognizer(main, list, score, ep);
    let partialRec: EngineRecognizer | null = null;
    if (partialId && partialId !== modelId) {
      partialRec = await recognizer(await load(partialId), [], score, ep);
    }
    // Final indexes continue across connections that share a session, so a
    // correction's `index` is unambiguous within the session file.
    const firstIndex = o.session ? await sessions.nextIndex(o.session) : 0;
    return new Stream({
      model: main.model,
      partialModel: partialId && partialRec ? partialId : null,
      hotwords: hotwordsName,
      mainRec,
      partialRec,
      sampleRate: o.sampleRate ?? 16000,
      session: o.session ?? null,
      sessions,
      firstIndex,
    });
  }

  async function transcribe(wav: Uint8Array, o: Omit<SttStreamOptions, "sampleRate"> = {}): Promise<TranscribeResult> {
    const e = await requireEngine();
    const tmp = join(opts.dataDir, "audio", "tmp");
    await mkdir(tmp, { recursive: true });
    const path = join(tmp, `${randomUUID()}.wav`);
    await writeFile(path, wav);
    let wave: EngineWaveform;
    try {
      wave = e.readWave(path);
    } finally {
      await rm(path, { force: true });
    }
    // Batch runs single-recognizer: partials are irrelevant here.
    const stream = await openStream({ ...o, partialModel: null, sampleRate: wave.sampleRate });
    const t0 = Date.now();
    const finals: SttEvent[] = [];
    const frame = Math.max(1, Math.round(FRAME_SECONDS * wave.sampleRate));
    try {
      for (let i = 0; i < wave.samples.length; i += frame) {
        finals.push(...stream.pushSamples(wave.samples.subarray(i, i + frame), wave.sampleRate).filter((ev) => ev.type === "final"));
      }
      finals.push(...stream.end().filter((ev) => ev.type === "final"));
    } finally {
      stream.close();
    }
    const segments = finals.flatMap((ev) => (ev.type === "final" ? [{ text: ev.text, words: ev.words }] : []));
    return {
      text: segments.map((s) => s.text).join(" "),
      segments,
      model: stream.model,
      hotwords: stream.hotwords,
      durationMs: Date.now() - t0,
    };
  }

  return {
    modelDir,
    hotwords,
    sessions,
    available: async () => (await engine()) != null,
    async models() {
      const d = defaultModelId();
      const pd = defaultPartialId();
      return Promise.all(
        STT_MODELS.map(async (m) => ({
          ...m,
          installed: await installed(m.id),
          default: m.id === d ? ("model" as const) : m.id === pd ? ("partialModel" as const) : null,
        })),
      );
    },
    ensureModel,
    openStream,
    transcribe,
    correct: (session, index, text) => sessions.append(session, { type: "correction", t: new Date().toISOString(), index, text }),
  };
}

export class SttUnavailableError extends Error {
  constructor() {
    super("stt not available: sherpa-onnx-node is not installed for this platform");
    this.name = "SttUnavailableError";
  }
}

// ── The streaming session ──────────────────────────────────────────────────

interface StreamDeps {
  model: SttModel;
  partialModel: string | null;
  hotwords: string | null;
  mainRec: EngineRecognizer;
  partialRec: EngineRecognizer | null;
  sampleRate: number;
  session: string | null;
  sessions: SessionStore;
  firstIndex: number;
}

class Stream implements SttStream {
  readonly model: string;
  readonly partialModel: string | null;
  readonly hotwords: string | null;
  private readonly d: StreamDeps;
  private readonly main: EngineStream;
  private readonly partial: EngineStream | null;
  private lastPartial = "";
  private index = 0;
  private closed = false;
  private writes: Promise<void> = Promise.resolve();
  /** Rate of the last audio fed — sherpa aborts the process if a stream's
   *  rate changes, so the flush pad must match it. */
  private rate: number;
  private fed = false;

  constructor(d: StreamDeps) {
    this.d = d;
    this.model = d.model.id;
    this.partialModel = d.partialModel;
    this.hotwords = d.hotwords;
    this.main = d.mainRec.createStream();
    this.partial = d.partialRec ? d.partialRec.createStream() : null;
    this.index = d.firstIndex;
    this.rate = d.sampleRate;
  }

  push(pcm16le: Uint8Array): SttEvent[] {
    return this.pushSamples(pcm16ToFloat32(pcm16le), this.d.sampleRate);
  }

  pushSamples(samples: Float32Array, sampleRate: number): SttEvent[] {
    if (this.closed) throw new Error("stream is closed");
    this.feed(samples, sampleRate);
    const events: SttEvent[] = [];
    const text = this.currentPartialText();
    if (text !== this.lastPartial) {
      this.lastPartial = text;
      events.push({ type: "partial", text });
    }
    if (this.d.mainRec.isEndpoint(this.main)) {
      const fin = this.takeFinal();
      if (fin) events.push(fin);
      this.resetAll();
    }
    return events;
  }

  end(): SttEvent[] {
    if (this.closed) return [];
    const pad = Math.round((this.d.model.tailPadMs / 1000) * this.rate);
    this.feed(new Float32Array(pad), this.rate);
    this.main.inputFinished();
    while (this.d.mainRec.isReady(this.main)) this.d.mainRec.decode(this.main);
    const events: SttEvent[] = [];
    const fin = this.takeFinal();
    if (fin) events.push(fin);
    this.close();
    return events;
  }

  flush(): Promise<void> {
    return this.writes;
  }

  close(): void {
    this.closed = true;
  }

  private feed(samples: Float32Array, sampleRate: number): void {
    if (sampleRate !== this.rate) {
      if (this.rate !== this.d.sampleRate || samples.length) {
        // Changing rate mid-stream is fatal in sherpa (it exits the process).
        if (this.fed) throw new Error(`sample rate changed mid-stream (${this.rate} → ${sampleRate})`);
      }
      this.rate = sampleRate;
    }
    this.fed = true;
    const w = { samples, sampleRate };
    this.main.acceptWaveform(w);
    while (this.d.mainRec.isReady(this.main)) this.d.mainRec.decode(this.main);
    if (this.partial && this.d.partialRec) {
      this.partial.acceptWaveform(w);
      while (this.d.partialRec.isReady(this.partial)) this.d.partialRec.decode(this.partial);
    }
  }

  private currentPartialText(): string {
    const r = this.partial && this.d.partialRec ? this.d.partialRec.getResult(this.partial) : this.d.mainRec.getResult(this.main);
    return r.text.trim();
  }

  private takeFinal(): SttEvent | null {
    const r = this.d.mainRec.getResult(this.main);
    const text = r.text.trim();
    if (!text) return null;
    const ev: SttEvent = { type: "final", index: this.index++, text, words: wordsOf(r) };
    if (this.d.session) {
      const entry = {
        type: "final" as const,
        t: new Date().toISOString(),
        index: ev.index,
        text,
        words: ev.words,
        model: this.model,
        hotwords: this.hotwords,
      };
      const session = this.d.session;
      this.writes = this.writes
        .then(() => this.d.sessions.append(session, entry))
        .catch((e) => console.warn(`[stt] session log failed:`, e));
    }
    return ev;
  }

  private resetAll(): void {
    this.d.mainRec.reset(this.main);
    if (this.partial && this.d.partialRec) this.d.partialRec.reset(this.partial);
    this.lastPartial = "";
  }
}

/** Group sherpa's subword tokens into words. A token starting with `▁` (or
 *  a space, as the JSON renders it) begins a word. */
export function wordsOf(r: EngineResult): SttWord[] {
  const tokens = r.tokens ?? [];
  const ts = r.timestamps ?? [];
  const base = r.start_time ?? 0;
  const words: SttWord[] = [];
  tokens.forEach((tok, i) => {
    const starts = tok.startsWith("▁") || tok.startsWith(" ");
    const piece = starts ? tok.slice(1) : tok;
    const last = words[words.length - 1];
    if (starts || !last) words.push({ text: piece, start: round(base + (ts[i] ?? 0)) });
    else last.text += piece;
  });
  return words.filter((w) => w.text.length > 0);
}

export function pcm16ToFloat32(bytes: Uint8Array): Float32Array {
  const n = bytes.byteLength >> 1;
  const out = new Float32Array(n);
  const view = new DataView(bytes.buffer, bytes.byteOffset, n * 2);
  for (let i = 0; i < n; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

function round(n: number): number {
  return Math.round(n * 1000) / 1000;
}

function stripUndefined<T extends object>(o: T): Partial<T> {
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Partial<T>;
}

/** `tar xjf` — bsdtar (macOS, Windows 10+) and GNU tar both do bz2; Node's
 *  zlib has no bzip2. */
function untar(archive: string, into: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const p = spawn("tar", ["xjf", archive, "-C", into], { stdio: ["ignore", "ignore", "pipe"] });
    let err = "";
    p.stderr.on("data", (d) => (err += d));
    p.on("error", reject);
    p.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`tar exited ${code}: ${err.trim()}`))));
  });
}

