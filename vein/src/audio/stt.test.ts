import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createStt,
  pcm16ToFloat32,
  wordsOf,
  type EngineRecognizer,
  type EngineResult,
  type EngineStream,
  type SttEngine,
} from "./stt.js";
import { STT_MODELS, sttModelPath } from "./models.js";

/**
 * A scripted fake engine: each recognizer is fed a queue of "what to say
 * after N seconds of audio", so the stream logic (partials, endpoints,
 * two-recognizer mode, flush, session logging) is exercised with no addon.
 */
interface Script {
  /** text to report once at least `at` seconds of audio have been accepted */
  at: number;
  text: string;
  tokens?: string[];
  timestamps?: number[];
  /** report an endpoint once this text is reached */
  endpoint?: boolean;
}

class FakeStream implements EngineStream {
  seconds = 0;
  finished = false;
  acceptWaveform(w: { samples: Float32Array; sampleRate: number }): void {
    this.seconds += w.samples.length / w.sampleRate;
  }
  inputFinished(): void {
    this.finished = true;
  }
}

class FakeRecognizer implements EngineRecognizer {
  readonly config: Record<string, unknown>;
  readonly script: Script[];
  private offsets = new WeakMap<FakeStream, number>();
  constructor(config: Record<string, unknown>, script: Script[]) {
    this.config = config;
    this.script = script;
  }
  createStream(): EngineStream {
    return new FakeStream();
  }
  isReady(): boolean {
    return false;
  }
  decode(): void {}
  private current(s: EngineStream): Script | undefined {
    const fs = s as FakeStream;
    const local = fs.seconds - (this.offsets.get(fs) ?? 0);
    return [...this.script].reverse().find((x) => local >= x.at);
  }
  isEndpoint(s: EngineStream): boolean {
    return this.current(s)?.endpoint === true;
  }
  reset(s: EngineStream): void {
    const fs = s as FakeStream;
    this.offsets.set(fs, fs.seconds);
  }
  getResult(s: EngineStream): EngineResult {
    const cur = this.current(s);
    if (!cur) return { text: "" };
    return { text: cur.text, tokens: cur.tokens, timestamps: cur.timestamps, start_time: this.offsets.get(s as FakeStream) ?? 0 };
  }
}

function fakeEngine(scripts: Record<string, Script[]>, built: Record<string, unknown>[] = []): SttEngine {
  return {
    OnlineRecognizer: class {
      constructor(config: Record<string, unknown>) {
        built.push(config);
        const tokens = String((config["modelConfig"] as { tokens: string }).tokens);
        const id = STT_MODELS.find((m) => tokens.includes(`/stt/${m.id}/`))?.id ?? "?";
        return new FakeRecognizer(config, scripts[id] ?? []);
      }
    } as unknown as SttEngine["OnlineRecognizer"],
    readWave: () => ({ samples: new Float32Array(16000 * 3), sampleRate: 16000 }),
  };
}

/** Pretend `id` is installed under modelDir. */
async function installFake(modelDir: string, id: string): Promise<void> {
  const dir = sttModelPath(modelDir, id);
  await mkdir(dir, { recursive: true });
  for (const f of ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"]) await writeFile(join(dir, f), "");
  await writeFile(join(dir, "tokens.txt"), "<blk> 0\n▁S 1\nphinx 2\n");
  await writeFile(join(dir, ".ok"), "x");
}

const silence = (seconds: number) => new Uint8Array(16000 * 2 * seconds);

describe("stt service", () => {
  let root: string;
  let dataDir: string;
  let modelDir: string;
  const env = { VEIN_STT_MODEL: undefined, VEIN_STT_PARTIAL_MODEL: undefined } as Record<string, string | undefined>;

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "vein-stt-"));
    dataDir = join(root, "data");
    modelDir = join(root, "models");
    await installFake(modelDir, "zipformer-en-kroko");
    await installFake(modelDir, "nemo-fast-conformer-en-80ms");
  });
  afterEach(() => rm(root, { recursive: true, force: true }));

  it("reports availability from the engine loader and lists install state", async () => {
    const off = createStt({ dataDir, modelDir, env, engine: async () => null, log: () => {} });
    assert.equal(await off.available(), false);
    const on = createStt({ dataDir, modelDir, env, engine: async () => fakeEngine({}), log: () => {} });
    assert.equal(await on.available(), true);
    const models = await on.models();
    assert.equal(models.find((m) => m.id === "zipformer-en-kroko")?.installed, true);
    assert.equal(models.find((m) => m.id === "zipformer-en-kroko")?.default, "model");
    assert.equal(models.find((m) => m.id === "nemo-fast-conformer-en-80ms")?.default, "partialModel");
    assert.equal(models.find((m) => m.id === "nemo-fast-conformer-en-480ms")?.installed, false);
  });

  it("openStream throws a clear error without the addon", async () => {
    const stt = createStt({ dataDir, modelDir, env, engine: async () => null, log: () => {} });
    await assert.rejects(() => stt.openStream({ partialModel: null }), /stt not available/);
  });

  it("single recognizer: partials on change, final + reset on endpoint, flush on end", async () => {
    const engine = fakeEngine({
      "zipformer-en-kroko": [
        { at: 0.5, text: "hello", tokens: ["▁hello"], timestamps: [0.3] },
        { at: 1.0, text: "hello world", tokens: ["▁hello", "▁wor", "ld"], timestamps: [0.3, 0.8, 0.9], endpoint: true },
        { at: 1.5, text: "again", tokens: ["▁again"], timestamps: [0.1] },
      ],
    });
    const stt = createStt({ dataDir, modelDir, env, engine: async () => engine, log: () => {} });
    const s = await stt.openStream({ partialModel: null, session: "s1" });
    assert.equal(s.model, "zipformer-en-kroko");
    assert.equal(s.partialModel, null);
    const events = [];
    for (let i = 0; i < 12; i++) events.push(...s.push(silence(0.1)));
    events.push(...s.end());
    await s.flush();
    assert.deepEqual(events, [
      { type: "partial", text: "hello" },
      { type: "partial", text: "hello world" },
      { type: "final", index: 0, text: "hello world", words: [{ text: "hello", start: 0.3 }, { text: "world", start: 0.8 }] },
      // after the reset at ~1.1 s, the second script line is reached 1.5 s later — that's
      // during end()'s 2 s silence pad, so it surfaces as the trailing final only.
      { type: "final", index: 1, text: "again", words: [{ text: "again", start: 1.2 }] },
    ]);
    // Session log captured both finals.
    const entries = (await stt.sessions.get("s1")) ?? [];
    assert.deepEqual(entries.map((e) => (e.type === "final" ? e.text : e.type)), ["hello world", "again"]);
    await stt.correct("s1", 1, "again!");
    assert.equal(((await stt.sessions.get("s1")) ?? []).at(-1)?.type, "correction");
    // A later connection on the same session keeps counting.
    const s2 = await stt.openStream({ partialModel: null, session: "s1" });
    const later = [];
    for (let i = 0; i < 11; i++) later.push(...s2.push(silence(0.1)));
    assert.equal(later.find((e) => e.type === "final")?.index, 2);
  });

  it("two recognizers: partials from the fast model, finals + endpoint from the main one", async () => {
    const built: Record<string, unknown>[] = [];
    const engine = fakeEngine(
      {
        "zipformer-en-kroko": [{ at: 1.0, text: "Hello, world.", tokens: ["▁Hello", ","], timestamps: [0.2, 0.5], endpoint: true }],
        "nemo-fast-conformer-en-80ms": [
          { at: 0.2, text: "hel" },
          { at: 0.4, text: "hello" },
          { at: 0.8, text: "hello world" },
        ],
      },
      built,
    );
    const stt = createStt({ dataDir, modelDir, env, engine: async () => engine, log: () => {} });
    const s = await stt.openStream({ hotwords: ["Sphinx"], hotwordsScore: 3 });
    assert.equal(s.partialModel, "nemo-fast-conformer-en-80ms");
    assert.equal(s.hotwords, "inline");
    const events = [];
    for (let i = 0; i < 11; i++) events.push(...s.push(silence(0.1)));
    assert.deepEqual(events, [
      { type: "partial", text: "hel" },
      { type: "partial", text: "hello" },
      { type: "partial", text: "hello world" },
      { type: "final", index: 0, text: "Hello, world.", words: [{ text: "Hello,", start: 0.2 }] },
    ]);
    // Hotwords went to the main recognizer only, with bpe modeling unit + vocab.
    const main = built.find((c) => String((c["modelConfig"] as { tokens: string }).tokens).includes("zipformer-en-kroko"))!;
    const fast = built.find((c) => String((c["modelConfig"] as { tokens: string }).tokens).includes("nemo-fast"))!;
    assert.equal(main["decodingMethod"], "modified_beam_search");
    assert.equal(main["hotwordsScore"], 3);
    assert.equal((main["modelConfig"] as Record<string, unknown>)["modelingUnit"], "bpe");
    assert.match(String((main["modelConfig"] as Record<string, unknown>)["bpeVocab"]), /bpe\.vocab$/);
    assert.equal(fast["decodingMethod"], "greedy_search");
    assert.equal(fast["hotwordsFile"], undefined);
  });

  it("caches recognizers per (model, hotwords, score, endpoint) and resolves named lists", async () => {
    const built: Record<string, unknown>[] = [];
    const engine = fakeEngine({}, built);
    const stt = createStt({ dataDir, modelDir, env, engine: async () => engine, log: () => {} });
    await stt.hotwords.put("team", "Stakwork\nSphinx :3\n");
    const a = await stt.openStream({ partialModel: null, hotwords: "team" });
    const b = await stt.openStream({ partialModel: null, hotwords: "team" });
    const c = await stt.openStream({ partialModel: null, hotwords: "team", hotwordsScore: 4 });
    const d = await stt.openStream({ partialModel: null });
    assert.equal(a.hotwords, "team");
    assert.equal(b.hotwords, "team");
    assert.equal(c.hotwords, "team");
    assert.equal(d.hotwords, null);
    assert.equal(built.length, 3);
    await assert.rejects(() => stt.openStream({ partialModel: null, hotwords: "nope" }), /unknown hotwords list/);
  });

  it("ignores hotwords on models that cannot take them", async () => {
    const built: Record<string, unknown>[] = [];
    const stt = createStt({ dataDir, modelDir, env, engine: async () => fakeEngine({}, built), log: () => {} });
    const s = await stt.openStream({ model: "nemo-fast-conformer-en-80ms", partialModel: null, hotwords: ["x"] });
    assert.equal(s.model, "nemo-fast-conformer-en-80ms");
    assert.equal(built[0]!["decodingMethod"], "greedy_search");
  });

  it("transcribe runs the single-recognizer path over a WAV and joins the finals", async () => {
    const engine = fakeEngine({
      "zipformer-en-kroko": [
        { at: 1.0, text: "one", endpoint: true },
        { at: 1.0, text: "two", endpoint: true },
      ],
    });
    const stt = createStt({ dataDir, modelDir, env, engine: async () => engine, log: () => {} });
    const r = await stt.transcribe(new Uint8Array(64));
    assert.equal(r.model, "zipformer-en-kroko");
    assert.ok(r.segments.length >= 1);
    assert.equal(r.text, r.segments.map((s) => s.text).join(" "));
  });

  it("ensureModel rejects unknown ids and dedupes in-flight downloads", async () => {
    let fetches = 0;
    const fetchImpl = (async () => {
      fetches++;
      return new Response("not a tarball", { status: 200 });
    }) as unknown as typeof fetch;
    const stt = createStt({ dataDir, modelDir, env, engine: async () => fakeEngine({}), fetchImpl, log: () => {} });
    await assert.rejects(() => stt.ensureModel("nope"), /unknown stt model/);
    const p1 = stt.ensureModel("nemo-fast-conformer-en-480ms").catch((e: Error) => e.message);
    const p2 = stt.ensureModel("nemo-fast-conformer-en-480ms").catch((e: Error) => e.message);
    const [m1, m2] = await Promise.all([p1, p2]);
    assert.match(String(m1), /sha256 mismatch/);
    assert.equal(m1, m2);
    assert.equal(fetches, 1);
  });
});

describe("helpers", () => {
  it("wordsOf groups subword tokens on the ▁/space marker with segment offsets", () => {
    assert.deepEqual(
      wordsOf({ text: "", tokens: [" A", "s", "k", " not", "▁now"], timestamps: [0.3, 0.4, 0.5, 0.7, 1.0], start_time: 10 }),
      [
        { text: "Ask", start: 10.3 },
        { text: "not", start: 10.7 },
        { text: "now", start: 11 },
      ],
    );
  });

  it("pcm16ToFloat32 scales and honours the byte offset", () => {
    const buf = Buffer.alloc(8);
    buf.writeInt16LE(0, 0);
    buf.writeInt16LE(-32768, 2);
    buf.writeInt16LE(16384, 4);
    buf.writeInt16LE(32767, 6);
    const view = new Uint8Array(buf.buffer, buf.byteOffset + 2, 6);
    assert.deepEqual(Array.from(pcm16ToFloat32(view)), [-1, 0.5, 32767 / 32768]);
  });
});
