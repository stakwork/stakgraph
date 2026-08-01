import { test, expect } from "../../testkit.js";
import {
  normalizeRepoParam,
  parseNodeTypes,
  parseRefIds,
  parseSince,
  parseLimit,
  parseLimitMode,
  firstLines,
  buildGraphMeta,
  isTrue,
} from "../utils.js";
import type { NodeType } from "../types.js";

test.describe("isTrue", () => {
  test("accepts the three truthy spellings", () => {
    for (const v of ["true", "1", "True"]) expect(isTrue(v)).toBe(true);
  });

  test("rejects everything else", () => {
    for (const v of ["TRUE", "yes", "0", "", "false"])
      expect(isTrue(v)).toBe(false);
  });
});

test.describe("normalizeRepoParam", () => {
  test("returns undefined for empty and whitespace-only input", () => {
    expect(normalizeRepoParam(undefined)).toBeUndefined();
    expect(normalizeRepoParam("")).toBeUndefined();
    expect(normalizeRepoParam("   ")).toBeUndefined();
  });

  test("passes through a bare org/repo", () => {
    expect(normalizeRepoParam("org/repo")).toBe("org/repo");
  });

  test("strips a .git suffix", () => {
    expect(normalizeRepoParam("org/repo.git")).toBe("org/repo");
  });

  test("reduces an https url to org/repo", () => {
    expect(normalizeRepoParam("https://github.com/org/repo")).toBe("org/repo");
    expect(normalizeRepoParam("https://github.com/org/repo.git")).toBe(
      "org/repo"
    );
  });

  /**
   * Known bug: the bare-slug fast path `/^[^\s\/]+\/[^\s\/]+$/` matches
   * `git@github.com:org/repo` before the ssh branch below it can run, so the
   * ssh branch is unreachable for single-slash ssh urls and the host prefix
   * survives. Asserting current behavior so a fix is a deliberate change.
   */
  test("does NOT normalize a single-slash ssh url (known bug)", () => {
    expect(normalizeRepoParam("git@github.com:org/repo.git")).toBe(
      "git@github.com:org/repo"
    );
  });

  /**
   * Known bug: a url with a nested group keeps the first two path segments
   * and drops the repo name, so gitlab subgroups resolve to the wrong repo.
   */
  test("keeps only the first two path segments of a nested url (known bug)", () => {
    expect(normalizeRepoParam("http://gitlab.com/group/sub/repo")).toBe(
      "group/sub"
    );
  });

  test("falls back to the raw input when it cannot find two segments", () => {
    expect(normalizeRepoParam("just-one-word")).toBe("just-one-word");
    expect(normalizeRepoParam("https://github.com/org")).toBe(
      "https://github.com/org"
    );
  });
});

test.describe("query param parsers", () => {
  test("parseNodeTypes splits, trims, and dedupes", () => {
    expect(parseNodeTypes({ node_types: "Function, Class ,,Function" })).toEqual(
      ["Function", "Class"]
    );
  });

  test("parseNodeTypes falls back to the singular node_type key", () => {
    expect(parseNodeTypes({ node_type: "File" })).toEqual(["File"]);
  });

  test("parseNodeTypes returns an empty array when absent", () => {
    expect(parseNodeTypes({})).toEqual([]);
  });

  test("parseRefIds trims and drops empty segments but keeps duplicates", () => {
    expect(parseRefIds({ ref_ids: " a , b ,, a " })).toEqual(["a", "b", "a"]);
    expect(parseRefIds({})).toEqual([]);
  });

  test("parseSince returns undefined for absent and unparseable values", () => {
    expect(parseSince({})).toBeUndefined();
    expect(parseSince({ since: "abc" })).toBeUndefined();
  });

  test("parseSince keeps a fractional value and distinguishes 0 from absent", () => {
    expect(parseSince({ since: "1.5" })).toBe(1.5);
    expect(parseSince({ since: "0" })).toBe(0);
  });

  test("parseLimit returns undefined for absent and unparseable values", () => {
    expect(parseLimit({})).toBeUndefined();
    expect(parseLimit({ limit: "abc" })).toBeUndefined();
  });

  test("parseLimit truncates to an integer and distinguishes 0 from absent", () => {
    expect(parseLimit({ limit: "10" })).toBe(10);
    expect(parseLimit({ limit: "3.7" })).toBe(3);
    expect(parseLimit({ limit: "0" })).toBe(0);
  });

  test("parseLimit does not reject a negative limit", () => {
    expect(parseLimit({ limit: "-5" })).toBe(-5);
  });

  test("parseLimitMode only recognizes 'total', defaulting to per_type", () => {
    expect(parseLimitMode({})).toBe("per_type");
    expect(parseLimitMode({ limit_mode: "total" })).toBe("total");
    expect(parseLimitMode({ limit_mode: "bogus" })).toBe("per_type");
  });
});

test.describe("firstLines", () => {
  test("returns an empty string for undefined input", () => {
    expect(firstLines(undefined)).toBe("");
  });

  test("caps the number of lines", () => {
    expect(firstLines("a\nb\nc", 2)).toBe("a\nb");
  });

  test("caps the length of each line", () => {
    expect(firstLines("abcdef", 40, 3)).toBe("abc");
  });
});

test.describe("buildGraphMeta", () => {
  test("counts nodes per label and nulls out absent limit/since", () => {
    const labels = ["Function", "Class"] as NodeType[];
    const nodes = [
      { labels: ["Function"] },
      { labels: ["Function"] },
      { labels: ["Class"] },
    ];

    expect(buildGraphMeta(labels, nodes, undefined, "per_type", undefined)).toEqual({
      node_types: labels,
      limit: null,
      limit_mode: "per_type",
      since: null,
      counts: { Function: 2, Class: 1 },
    });
  });
});
