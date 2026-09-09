/**
 * Hotwords (contextual biasing) for sherpa-onnx transducer models — the
 * mechanism the dream cycle drives (plans/local-desktop-and-stt.md §4.2, §4.8).
 *
 * Two hard-won facts, both verified live:
 *   1. sherpa needs `modelingUnit: "bpe"`; left unset it defaults to `cjkchar`
 *      and splits every hotword into single characters the model never emits,
 *      so the list silently does nothing at any score.
 *   2. `bpe` mode needs a sentencepiece `bpe.vocab`, which the released models
 *      don't ship. We synthesize one from `tokens.txt`: every non-special
 *      token with the same score, so sherpa's unigram encoder picks the
 *      fewest-pieces segmentation (longest match). Good enough to turn
 *      "this FHIC swarm" into "the Sphinx swarm".
 *
 * This module has no sherpa dependency: it only writes the files the
 * recognizer config points at.
 */
import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";

export interface Hotword {
  phrase: string;
  /** Per-phrase boost; falls back to the recognizer's global score. */
  score?: number;
}

/** One phrase per line, optional trailing ` :score`, `#` comments, blanks
 *  ignored. The same text format sherpa reads, so a stored list is passed
 *  through verbatim. */
export function parseHotwords(text: string): Hotword[] {
  const out: Hotword[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const m = line.match(/^(.*?)\s*:\s*(-?\d+(?:\.\d+)?)$/);
    if (m && m[1]) out.push({ phrase: m[1].trim(), score: Number(m[2]) });
    else out.push({ phrase: line });
  }
  return out;
}

export function formatHotwords(list: readonly Hotword[]): string {
  return list.map((h) => (h.score == null ? h.phrase : `${h.phrase} :${h.score}`)).join("\n") + "\n";
}

/** Stable id for a compiled list: sha256 of its canonical text. */
export function hotwordsHash(list: readonly Hotword[]): string {
  return createHash("sha256").update(formatHotwords(list)).digest("hex").slice(0, 16);
}

/** Sentencepiece-style vocab (`piece\tscore`) from a sherpa `tokens.txt`
 *  (`piece id` per line). Special tokens (`<blk>`, `<sos/eos>`, `<unk>`) are
 *  replaced by the three sentencepiece specials. */
export function synthesizeBpeVocab(tokensTxt: string): string {
  const pieces = tokensTxt
    .split(/\r?\n/)
    .map((l) => l.trim().split(/\s+/)[0] ?? "")
    .filter((p) => p && !/^<.*>$/.test(p));
  return ["<unk>\t0", "<s>\t0", "</s>\t0", ...pieces.map((p) => `${p}\t-1`)].join("\n") + "\n";
}

export interface CompiledHotwords {
  hash: string;
  /** The list, one phrase per line (sherpa `hotwordsFile`). */
  file: string;
  /** The synthesized vocab (sherpa `bpeVocab`). */
  vocab: string;
}

/** Materialize a list beside a model: `<modelDir>/hotwords/<hash>.txt` plus
 *  `<modelDir>/bpe.vocab` (synthesized once). Idempotent. */
export async function compileHotwords(modelDir: string, list: readonly Hotword[]): Promise<CompiledHotwords> {
  const vocab = join(modelDir, "bpe.vocab");
  if (!(await exists(vocab))) {
    const tokens = await readFile(join(modelDir, "tokens.txt"), "utf-8");
    await writeFile(vocab, synthesizeBpeVocab(tokens));
  }
  const hash = hotwordsHash(list);
  const dir = join(modelDir, "hotwords");
  await mkdir(dir, { recursive: true });
  const file = join(dir, `${hash}.txt`);
  if (!(await exists(file))) await writeFile(file, formatHotwords(list));
  return { hash, file, vocab };
}

// ── Named lists (the dream cycle's promotion artifact) ─────────────────────

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

export interface HotwordsListInfo {
  name: string;
  count: number;
  updatedAt: string;
}

/** `<dataDir>/audio/hotwords/<name>.txt`. A workflow's `http` step PUTs a
 *  list here; a stream names it in `start.hotwords`. */
export class HotwordsStore {
  readonly dir: string;
  constructor(dataDir: string) {
    this.dir = join(dataDir, "audio", "hotwords");
  }

  private pathOf(name: string): string {
    if (!NAME_RE.test(name)) throw new Error(`invalid hotwords list name: ${JSON.stringify(name)}`);
    return join(this.dir, `${name}.txt`);
  }

  async list(): Promise<HotwordsListInfo[]> {
    let names: string[];
    try {
      names = (await readdir(this.dir)).filter((f) => f.endsWith(".txt"));
    } catch {
      return [];
    }
    const out: HotwordsListInfo[] = [];
    for (const f of names.sort()) {
      const p = join(this.dir, f);
      const [text, st] = await Promise.all([readFile(p, "utf-8"), stat(p)]);
      out.push({ name: f.slice(0, -4), count: parseHotwords(text).length, updatedAt: st.mtime.toISOString() });
    }
    return out;
  }

  /** The raw text, or null when the list doesn't exist. */
  async get(name: string): Promise<string | null> {
    try {
      return await readFile(this.pathOf(name), "utf-8");
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw e;
    }
  }

  async put(name: string, text: string): Promise<Hotword[]> {
    const p = this.pathOf(name);
    await mkdir(this.dir, { recursive: true });
    const list = parseHotwords(text);
    await writeFile(p, formatHotwords(list));
    return list;
  }

  async delete(name: string): Promise<boolean> {
    try {
      await rm(this.pathOf(name));
      return true;
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return false;
      throw e;
    }
  }
}

async function exists(p: string): Promise<boolean> {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}
