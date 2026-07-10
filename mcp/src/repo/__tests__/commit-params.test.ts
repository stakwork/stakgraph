import { test, expect } from "../../testkit.js";
import { parseCommitList } from "../index.js";

// ---------------------------------------------------------------------------
// parseCommitList — unit tests
// ---------------------------------------------------------------------------

test.describe("parseCommitList", () => {
  test("comma-separated values → array of trimmed strings", () => {
    expect(parseCommitList("main,feature-branch")).toEqual(["main", "feature-branch"]);
  });

  test("single value → single-element array (backward compat)", () => {
    expect(parseCommitList("main")).toEqual(["main"]);
  });

  test("undefined → empty array", () => {
    expect(parseCommitList(undefined)).toEqual([]);
  });

  test("empty string → empty array", () => {
    expect(parseCommitList("")).toEqual([]);
  });

  test("values with extra whitespace are trimmed", () => {
    expect(parseCommitList("main , feature-branch ")).toEqual(["main", "feature-branch"]);
  });
});

// ---------------------------------------------------------------------------
// zipCommitsToUrls — pure helper exercised inline
// ---------------------------------------------------------------------------

/**
 * Pure helper extracted from the cloneMultipleRepos logic so it can be
 * tested without triggering real git operations.
 */
function zipCommitsToUrls(
  urls: string[],
  commits?: string[]
): Array<{ url: string; commit: string | undefined }> {
  return urls.map((url, idx) => ({
    url,
    commit: commits?.[idx] ?? commits?.[0],
  }));
}

test.describe("zipCommitsToUrls", () => {
  test("two repos each receive a different branch", () => {
    const result = zipCommitsToUrls(
      ["https://github.com/org/repoA", "https://github.com/org/repoB"],
      ["main", "feature-branch"]
    );
    expect(result[0].commit).toBe("main");
    expect(result[1].commit).toBe("feature-branch");
  });

  test("single commit falls back to all repos (backward compat)", () => {
    const result = zipCommitsToUrls(
      ["https://github.com/org/repoA", "https://github.com/org/repoB"],
      ["main"]
    );
    expect(result[0].commit).toBe("main");
    expect(result[1].commit).toBe("main");
  });

  test("no commits → all undefined", () => {
    const result = zipCommitsToUrls(
      ["https://github.com/org/repoA", "https://github.com/org/repoB"],
      undefined
    );
    expect(result[0].commit).toBeUndefined();
    expect(result[1].commit).toBeUndefined();
  });
});
