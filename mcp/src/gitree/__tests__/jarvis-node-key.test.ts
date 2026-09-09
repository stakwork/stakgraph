/**
 * Unit tests for jarvisConceptNodeKey.
 *
 * The function must reproduce jarvis's sanitize_node_key + _compose_node_key
 * (jarvis api/helper/schema_validation.py) byte-for-byte: the value is
 * matched by equality in jarvis's MERGE and covered by a
 * (node_key, namespace) uniqueness constraint, so any divergence either
 * re-forks duplicates or throws on write. The expected values below marked
 * "prod" are actual node_key values jarvis assigned on a production graph.
 */
import { test, expect } from "../../testkit.js";
import { jarvisConceptNodeKey } from "../store/utils.js";

test.describe("jarvisConceptNodeKey", () => {
  test("matches jarvis-assigned keys from prod (punctuation, ampersand, hyphens)", () => {
    // prod: jarvis wrote these exact keys for these exact names
    expect(
      jarvisConceptNodeKey("Change-of-Control & Anti-Assignment Provision Analysis")
    ).toBe("concept-changeofcontrolantiassignmentprovisionanalysis");
    expect(jarvisConceptNodeKey("treaty-silence-as-non-concurrency")).toBe(
      "concept-treatysilenceasnonconcurrency"
    );
    expect(jarvisConceptNodeKey("PLI: Return on Sales (ROS)")).toBe(
      "concept-plireturnonsalesros"
    );
  });

  test("strips spaces before lowercasing and drops unicode punctuation", () => {
    expect(jarvisConceptNodeKey("  Working Capital Pegs, Collars — True-Up  ")).toBe(
      "concept-workingcapitalpegscollarstrueup"
    );
  });

  test("keeps digits", () => {
    expect(jarvisConceptNodeKey("Section 382 Ownership Change")).toBe(
      "concept-section382ownershipchange"
    );
  });

  test("preserves non-space whitespace like jarvis does", () => {
    // jarvis removes only " " (str.replace) before stripping to
    // [a-zA-Z0-9\s] — a tab survives both steps. Bug-compatible on purpose.
    expect(jarvisConceptNodeKey("a\tb")).toBe("concept-a\tb");
  });

  test("hashes the value portion past 200 chars", () => {
    const name = "x".repeat(250);
    const key = jarvisConceptNodeKey(name);
    // hashlib.sha256(("x"*250).encode()).hexdigest()[:32], per _compose_node_key
    expect(key).toBe("concept-086d4a1c293bde318dc1fec9a21b9d82");
    expect(key.length).toBe(40);
  });

  test("does not hash at exactly 200 chars", () => {
    const name = "y".repeat(192); // "concept-" + 192 = 200
    expect(jarvisConceptNodeKey(name)).toBe(`concept-${"y".repeat(192)}`);
  });
});
