/**
 * STT model catalog — the one list desktop, server, and the dream cycle
 * agree on (plans/local-desktop-and-stt.md §4.5). Every entry is a sherpa-onnx
 * GitHub release asset with its sha256, so a download is verifiable and a
 * server image can pre-bake the same files.
 *
 * Only streaming transducers are listed: hotwords (§4.2) need a transducer,
 * and the NeMo online transducer in sherpa is greedy-only, so `hotwords`
 * records which entries can be biased.
 */
import { homedir } from "node:os";
import { join } from "node:path";

export interface SttModel {
  id: string;
  url: string;
  /** sha256 of the `.tar.bz2`. */
  sha256: string;
  bytes: number;
  /** Top-level directory inside the archive. */
  archiveDir: string;
  language: string;
  /** How often partials change, measured (ms). Baked into the export. */
  chunkMs: number;
  /** Accepts a hotwords list (`modified_beam_search` transducer). */
  hotwords: boolean;
  /** Emits casing and punctuation. */
  cased: boolean;
  /** Silence to feed after the last audio so the final chunk flushes (ms). */
  tailPadMs: number;
  description: string;
}

const RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models";

export const STT_MODELS: readonly SttModel[] = [
  {
    id: "zipformer-en-kroko",
    url: `${RELEASE}/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2`,
    sha256: "c8676e5ff9ac2a85296e53ee0fd4d5fb1db6770e7a7647166eeafe349ade6834",
    bytes: 57267600,
    archiveDir: "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
    language: "en",
    chunkMs: 1280,
    hotwords: true,
    cased: true,
    tailPadMs: 2000,
    description: "Streaming Zipformer (Banafo Kroko). Cased, punctuated, hotword-capable; partials every ~1.3 s. The finals model.",
  },
  {
    id: "nemo-fast-conformer-en-80ms",
    url: `${RELEASE}/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8.tar.bz2`,
    sha256: "7bd33a914e93370a1ba9c2066d9e841bdcad8613fa2a00537c1ae15d851a14d8",
    bytes: 102813625,
    archiveDir: "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8",
    language: "en",
    chunkMs: 80,
    hotwords: false,
    cased: false,
    tailPadMs: 1000,
    description: "NeMo streaming FastConformer transducer, 80 ms chunks (int8). Lowercase, greedy-only; partials every ~150 ms. The partials model.",
  },
  {
    id: "nemo-fast-conformer-en-480ms",
    url: `${RELEASE}/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8.tar.bz2`,
    sha256: "da93061cbf7b708b6b65976f70b29f519be29df750d8cdcabf98c65645930f13",
    bytes: 105913204,
    archiveDir: "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8",
    language: "en",
    chunkMs: 480,
    hotwords: false,
    cased: false,
    tailPadMs: 1000,
    description: "NeMo streaming FastConformer transducer, 480 ms chunks (int8). Lowercase, greedy-only; partials every ~570 ms at less CPU than the 80 ms variant.",
  },
  {
    id: "nemotron-speech-en-80ms",
    url: `${RELEASE}/sherpa-onnx-nemotron-speech-streaming-en-0.6b-80ms-int8-2026-04-25.tar.bz2`,
    sha256: "caaf92069dbd1ca054f8e17cab179813bc28b4585f5c392540357ece4722333d",
    bytes: 463945379,
    archiveDir: "sherpa-onnx-nemotron-speech-streaming-en-0.6b-80ms-int8-2026-04-25",
    language: "en",
    chunkMs: 80,
    hotwords: false,
    cased: true,
    tailPadMs: 1000,
    description: "NVIDIA Nemotron-speech streaming 0.6B, 80 ms chunks (int8). Cased, punctuated, the most accurate; greedy-only and ~0.5× real-time CPU on two threads.",
  },
];

export const DEFAULT_MODEL = "zipformer-en-kroko";
export const DEFAULT_PARTIAL_MODEL = "nemo-fast-conformer-en-80ms";

export function findModel(id: string): SttModel | undefined {
  return STT_MODELS.find((m) => m.id === id);
}

export function requireModel(id: string): SttModel {
  const m = findModel(id);
  if (!m) throw new Error(`unknown stt model "${id}" (known: ${STT_MODELS.map((x) => x.id).join(", ")})`);
  return m;
}

/** `VEIN_MODEL_DIR`, else the older `VEIN_MODEL_CACHE` (MiniLM's setting),
 *  else `~/.cache/vein-models`. STT models live under `<dir>/stt/<id>/`. */
export function modelDirFromEnv(env: Record<string, string | undefined> = process.env): string {
  return env["VEIN_MODEL_DIR"] ?? env["VEIN_MODEL_CACHE"] ?? join(homedir(), ".cache", "vein-models");
}

export function sttModelPath(modelDir: string, id: string): string {
  return join(modelDir, "stt", id);
}

/** Pick the model files inside an extracted dir. Prefers int8 exports. */
export function pickModelFiles(entries: readonly string[]): {
  encoder: string;
  decoder: string;
  joiner: string;
  tokens: string;
} {
  const pick = (re: RegExp) => {
    const hits = entries.filter((f) => re.test(f)).sort((a, b) => Number(b.includes("int8")) - Number(a.includes("int8")));
    if (!hits[0]) throw new Error(`model dir is missing a file matching ${re}`);
    return hits[0];
  };
  return {
    encoder: pick(/^encoder.*\.onnx$/),
    decoder: pick(/^decoder.*\.onnx$/),
    joiner: pick(/^joiner.*\.onnx$/),
    tokens: pick(/^tokens\.txt$/),
  };
}
