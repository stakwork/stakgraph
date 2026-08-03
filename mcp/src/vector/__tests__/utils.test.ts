/**
 * chunkCode and weightedPooling sit under every embedding path
 * (vectorizeQuery / vectorizeBatch / vectorizeCodeDocument).
 *
 * createOverlappingChunks is not covered — it is marked "not used" and has no
 * call sites.
 */
import { test, expect } from "../../testkit.js";
import { chunkCode, weightedPooling } from "../utils.js";

test.describe("chunkCode", () => {
  test("keeps input shorter than the chunk size in one chunk", () => {
    expect(chunkCode("abc", 10)).toEqual(["abc"]);
  });

  test("keeps multiple short lines together and preserves newlines", () => {
    expect(chunkCode("aa\nbb\ncc", 10)).toEqual(["aa\nbb\ncc"]);
  });

  test("hard-splits a single line longer than the chunk size", () => {
    expect(chunkCode("abcdefghij", 4)).toEqual(["abcd", "efgh", "ij"]);
  });

  test("flushes the pending chunk before hard-splitting a long line", () => {
    expect(chunkCode("aa\nbbbbbbbbbb\ncc", 4)).toEqual([
      "aa",
      "bbbb",
      "bbbb",
      "bb",
      "cc",
    ]);
  });

  test("loses no characters when chunks are rejoined", () => {
    const source = "const a = 1;\n" + "x".repeat(37) + "\nreturn a;";
    const rejoined = chunkCode(source, 12).join("").replace(/\n/g, "");
    expect(rejoined).toBe(source.replace(/\n/g, ""));
  });

  test("returns a single empty chunk for empty input", () => {
    expect(chunkCode("", 10)).toEqual([""]);
  });
});

test.describe("weightedPooling", () => {
  test("returns a vector of the same dimensionality as its inputs", () => {
    expect(weightedPooling([[1, 2, 3], [4, 5, 6]], [1, 1])).toHaveLength(3);
  });

  test("uniform weights produce the arithmetic mean", () => {
    expect(weightedPooling([[1, 2], [3, 4]], [1, 1])).toEqual([2, 3]);
  });

  test("a zero weight excludes that vector entirely", () => {
    expect(weightedPooling([[0, 0], [10, 10]], [0, 1])).toEqual([10, 10]);
  });

  test("normalizes by total weight rather than vector count", () => {
    expect(weightedPooling([[2, 2], [4, 4]], [3, 1])).toEqual([2.5, 2.5]);
  });

  test("a single vector pools to itself regardless of its weight", () => {
    expect(weightedPooling([[5, 5]], [2])).toEqual([5, 5]);
  });
});
