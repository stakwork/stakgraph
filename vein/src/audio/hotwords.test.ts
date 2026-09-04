import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  HotwordsStore,
  compileHotwords,
  formatHotwords,
  hotwordsHash,
  parseHotwords,
  synthesizeBpeVocab,
} from "./hotwords.js";

describe("hotwords list format", () => {
  it("parses phrases, per-phrase scores, comments and blanks", () => {
    const list = parseHotwords("# team lingo\nStakwork\nSphinx :3.5\n\n  vein workflow : 2\n");
    assert.deepEqual(list, [
      { phrase: "Stakwork" },
      { phrase: "Sphinx", score: 3.5 },
      { phrase: "vein workflow", score: 2 },
    ]);
  });

  it("round-trips through format", () => {
    const text = "Stakwork\nSphinx :3.5\n";
    assert.equal(formatHotwords(parseHotwords(text)), text);
  });

  it("hashes the canonical text, not the input spelling", () => {
    assert.equal(hotwordsHash(parseHotwords("a\nb :2")), hotwordsHash(parseHotwords("# x\n\na\n  b : 2 \n")));
    assert.notEqual(hotwordsHash(parseHotwords("a")), hotwordsHash(parseHotwords("a :2")));
  });
});

describe("synthesizeBpeVocab", () => {
  it("turns tokens.txt into a sentencepiece vocab with constant scores", () => {
    const vocab = synthesizeBpeVocab("<blk> 0\n<sos/eos> 1\n<unk> 2\ns 3\n▁the 4\n");
    assert.equal(vocab, "<unk>\t0\n<s>\t0\n</s>\t0\ns\t-1\n▁the\t-1\n");
  });
});

describe("compileHotwords", () => {
  let dir: string;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "vein-hotwords-"));
    await writeFile(join(dir, "tokens.txt"), "<blk> 0\n▁S 1\np 2\nhi 3\nn 4\nx 5\n");
  });
  afterEach(() => rm(dir, { recursive: true, force: true }));

  it("writes the list beside the model and synthesizes bpe.vocab once", async () => {
    const a = await compileHotwords(dir, [{ phrase: "Sphinx" }, { phrase: "Hive", score: 3 }]);
    assert.equal(a.vocab, join(dir, "bpe.vocab"));
    assert.equal(await readFile(a.file, "utf-8"), "Sphinx\nHive :3\n");
    assert.match(await readFile(a.vocab, "utf-8"), /^<unk>\t0\n<s>\t0\n<\/s>\t0\n▁S\t-1\n/);
    // Idempotent: same hash, same file, vocab not rewritten.
    await writeFile(a.vocab, "sentinel");
    const b = await compileHotwords(dir, [{ phrase: "Sphinx" }, { phrase: "Hive", score: 3 }]);
    assert.equal(b.file, a.file);
    assert.equal(await readFile(a.vocab, "utf-8"), "sentinel");
  });
});

describe("HotwordsStore", () => {
  let dir: string;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "vein-hotwords-store-"));
  });
  afterEach(() => rm(dir, { recursive: true, force: true }));

  it("put/get/list/delete a named list", async () => {
    const store = new HotwordsStore(dir);
    assert.deepEqual(await store.list(), []);
    assert.equal(await store.get("team"), null);
    const list = await store.put("team", "# lingo\nStakwork\nSphinx :3\n");
    assert.equal(list.length, 2);
    assert.equal(await store.get("team"), "Stakwork\nSphinx :3\n");
    const infos = await store.list();
    assert.equal(infos.length, 1);
    assert.equal(infos[0]!.name, "team");
    assert.equal(infos[0]!.count, 2);
    assert.equal(await store.delete("team"), true);
    assert.equal(await store.delete("team"), false);
  });

  it("rejects names that could escape the directory", async () => {
    const store = new HotwordsStore(dir);
    await assert.rejects(() => store.put("../x", "a"), /invalid hotwords list name/);
    await assert.rejects(() => store.get("a/b"), /invalid hotwords list name/);
  });
});
