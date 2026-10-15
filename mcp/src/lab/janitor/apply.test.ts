import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, defineStep, fileArtifactsCapability, runWorkflow, standardServices } from "strut";
import { seedArtifactSteps } from "../artifacts/seed.js";
import { seedJanitorSteps, seedJanitorWorkflows } from "./seed.js";
import { planChanges, type Finding, type LiveNode } from "./steps/plan.js";
import { revertPlan, changeIds, type Applied } from "./steps/revert-plan.js";
import { verdicts, reportSection, type LiveState } from "./steps/readback.js";
import { pickBatch, renderBatch, byName, comparePath, placeTree, siblingsOf, PATH_SEP, MAX_SIBLINGS, type TreeNode } from "./steps/queue.js";

// ── the plan: findings × the live graph → what gets written ────────────────

const node = (ref_id: string, props: Record<string, unknown>, extra: Partial<LiveNode> = {}): LiveNode => ({
  ref_id,
  labels: ["Data_Bank", "Concept"],
  name: String(props.name ?? ref_id),
  properties: props,
  parents: [],
  deleted: false,
  stamp: null,
  ...extra,
});
const finding = (ref_id: string, action: Finding["action"], extra: Partial<Finding> = {}): Finding => ({
  ref_id,
  name: ref_id,
  issue: "vague",
  problem: "too vague",
  evidence: "old words",
  suggestion: "say more",
  severity: "medium",
  action,
  ...extra,
});
const OPTS = {
  visited: [{ ref_id: "law" }, { ref_id: "tort" }, { ref_id: "crime" }, { ref_id: "janitor" }, { ref_id: "mandate" }, { ref_id: "seeded" }],
  protect: ["janitor", "mandate"],
  parent_ref_id: "janitor",
  node_type: "Concept",
  properties: ["description", "docs"],
  min_severity: "low",
  max_changes: 20,
  seeded_prefix: "lab/",
};
const LIVE = new Map<string, LiveNode>([
  ["law", node("law", { name: "Law", description: "old words here" })],
  ["tort", node("tort", { name: "Tort", description: "A tort." }, { parents: ["law"] })],
  ["crime", node("crime", { name: "Crime", description: "A crime." }, { parents: ["law"] })],
  ["janitor", node("janitor", { name: "Janitor" })],
  ["mandate", node("mandate", { name: "Mandate", docs: "the mandate" }, { parents: ["janitor"] })],
  ["other-mandate", node("other-mandate", { name: "Other" }, { parents: ["janitor"] })],
  ["seeded", node("seeded", { name: "Job", description: "Seeded." }, { stamp: "lab/job/concepts/Job.md@abcdef123456" })],
]);

describe("planChanges", () => {
  it("plans an edit whose old_value matches the live value, in graph/edit-node's shape", () => {
    const p = planChanges([finding("law", { kind: "set_property", property: "description", old_value: "old words here", new_value: "New words." })], LIVE, OPTS);
    assert.deepEqual(p.skipped, []);
    assert.equal(p.edits.length, 1);
    const e = p.edits[0]!;
    assert.equal(e.id, "e1");
    assert.equal(e.kind, "edit");
    assert.equal(e.name, "Law");
    assert.equal(e.old_value, "old words here");
    assert.deepEqual(e.node_data, { description: "New words." });
    assert.deepEqual(e.properties_to_be_deleted, []);
    assert.deepEqual(e.finding, { issue: "vague", severity: "medium", problem: "too vague" });
  });

  it("accepts the evidence quote as proof of freshness when old_value is missing or truncated", () => {
    const p = planChanges([finding("law", { kind: "set_property", property: "description", new_value: "New." }, { evidence: "old words" })], LIVE, OPTS);
    assert.equal(p.edits.length, 1);
  });

  it("replace_text: resolved against the live value into a whole-value edit, with the fragment kept for the record", () => {
    const p = planChanges([finding("law", { kind: "replace_text", property: "description", find: "old words", replace: "new words" })], LIVE, OPTS);
    assert.deepEqual(p.skipped, []);
    const e = p.edits[0]!;
    assert.deepEqual([e.old_value, e.new_value, e.node_data, e.via], ["old words here", "new words here", { description: "new words here" }, { find: "old words", replace: "new words" }]);
    // "" removes the passage.
    const gone = planChanges([finding("law", { kind: "replace_text", property: "description", find: " here", replace: "" })], LIVE, OPTS);
    assert.equal(gone.edits[0]!.new_value, "old words");
  });

  it("replace_text without a property takes the one allowed property where find occurs exactly once; none or several is refused", () => {
    const live = new Map(LIVE);
    live.set("both", node("both", { name: "Both", description: "the rule here", docs: "the rule here, again" }));
    live.set("neither", node("neither", { name: "Neither", description: "x", docs: "y" }));
    const visited = [...OPTS.visited, { ref_id: "both" }, { ref_id: "neither" }];
    const p = planChanges(
      [
        finding("law", { kind: "replace_text", find: "old words", replace: "new words" }),
        finding("both", { kind: "replace_text", find: "the rule", replace: "a rule" }),
        finding("neither", { kind: "replace_text", find: "the rule", replace: "a rule" }),
        finding("law", { kind: "set_property", new_value: "x" }),
      ],
      live,
      { ...OPTS, visited },
    );
    assert.deepEqual(p.edits.map((e) => [e.ref_id, e.property, e.new_value, e.via]), [
      ["law", "description", "new words here", { find: "old words", replace: "new words", property_inferred: true }],
    ]);
    assert.deepEqual(p.skipped.map((s) => [s.ref_id, s.reason]), [["both", "invalid_action"], ["neither", "invalid_action"], ["law", "invalid_action"]]);
    assert.match(p.skipped[0]!.detail!, /in description and docs — say which/);
    assert.match(p.skipped[1]!.detail!, /in none of description, docs exactly once/);
    assert.match(p.skipped[2]!.detail!, /set_property without a property/);
    // A named property is never second-guessed: wrong one → find_not_once, as before.
    const named = planChanges([finding("both", { kind: "replace_text", property: "docs", find: "again", replace: "twice" })], live, { ...OPTS, visited });
    assert.deepEqual(named.edits.map((e) => [e.property, e.new_value, e.via?.property_inferred]), [["docs", "the rule here, twice", undefined]]);
  });

  it("replace_text: find must occur exactly once — zero (the text moved on) or several (ambiguous) is find_not_once; never emptied, never a no-op", () => {
    const live = new Map(LIVE);
    live.set("dup", node("dup", { name: "Dup", docs: "a rule. a rule." }));
    live.set("bare", node("bare", { name: "Bare" }));
    const p = planChanges(
      [
        finding("law", { kind: "replace_text", property: "description", find: "not there", replace: "x" }),
        finding("dup", { kind: "replace_text", property: "docs", find: "a rule.", replace: "x" }),
        finding("bare", { kind: "replace_text", property: "docs", find: "a", replace: "x" }),
        finding("law", { kind: "replace_text", property: "description", find: "old words here", replace: "  " }),
        finding("law", { kind: "replace_text", property: "description", find: "old", replace: "old" }),
        finding("law", { kind: "replace_text", property: "description", replace: "x" }),
        finding("law", { kind: "replace_text", property: "name", find: "Law", replace: "Laws" }),
      ],
      live,
      { ...OPTS, visited: [...OPTS.visited, { ref_id: "dup" }, { ref_id: "bare" }] },
    );
    assert.deepEqual(p.edits, []);
    assert.deepEqual(
      p.skipped.map((s) => s.reason),
      ["find_not_once", "find_not_once", "find_not_once", "empty_value", "no_change", "invalid_action", "property_not_allowed"],
    );
    assert.match(p.skipped[0]!.detail!, /not in description now/);
    assert.match(p.skipped[1]!.detail!, /occurs 2 times/);
  });

  it("skips as stale when neither old_value nor evidence matches what the node reads now", () => {
    const p = planChanges([finding("law", { kind: "set_property", property: "description", old_value: "something else", new_value: "New." }, { evidence: "not there" })], LIVE, OPTS);
    assert.deepEqual(p.edits, []);
    assert.equal(p.skipped[0]!.reason, "stale");
  });

  it("never writes Janitor, a mandate, anything hanging under Janitor, or a git-seeded Concept — whatever the agent says", () => {
    const set = (ref: string) => finding(ref, { kind: "set_property", property: "description", old_value: "", new_value: "Rewritten." });
    const p = planChanges([set("janitor"), set("mandate"), set("other-mandate"), set("seeded")], LIVE, {
      ...OPTS,
      visited: [...OPTS.visited, { ref_id: "other-mandate" }],
    });
    assert.deepEqual(p.edits, []);
    assert.deepEqual(
      p.skipped.map((s) => [s.ref_id, s.reason]),
      [
        ["janitor", "protected"],
        ["mandate", "protected"],
        ["other-mandate", "protected"],
        ["seeded", "git_seeded"],
      ],
    );
    assert.match(p.skipped[3]!.detail!, /lab\/job\/concepts\/Job\.md@/);
  });

  it("with targets (the run's batch): a finding outside it is not_in_batch; a target counts as visited; a move may go under any visited node", () => {
    const p = planChanges(
      [
        finding("tort", { kind: "set_property", property: "description", old_value: "A tort.", new_value: "A civil wrong." }),
        finding("law", { kind: "set_property", property: "description", old_value: "old words here", new_value: "Legal knowledge." }),
        finding("crime", { kind: "move", new_parent_ref_id: "law" }),
      ],
      LIVE,
      // tort and crime were handed to the agent; it recorded only law, which it read for context
      { ...OPTS, visited: [{ ref_id: "law" }], targets: ["tort", "crime"] },
    );
    assert.deepEqual(p.edits.map((e) => [e.ref_id, e.new_value]), [["tort", "A civil wrong."]]);
    assert.deepEqual(p.skipped.map((s) => [s.ref_id, s.reason]), [["law", "not_in_batch"], ["crime", "no_change"]]);
  });

  it("never renames, never writes an unlisted property, nothing outside visited, nothing that is not a Concept, nothing missing", () => {
    const live = new Map(LIVE);
    live.set("run", node("run", { name: "a run" }, { labels: ["Data_Bank", "StrutRun"] }));
    const p = planChanges(
      [
        finding("law", { kind: "set_property", property: "name", old_value: "Law", new_value: "Laws" }),
        finding("law", { kind: "set_property", property: "docs", old_value: "", new_value: "" }),
        finding("ghost", { kind: "set_property", property: "description", old_value: "", new_value: "x" }),
        finding("run", { kind: "set_property", property: "description", old_value: "", new_value: "x" }),
        finding("tort", { kind: "set_property", property: "description", old_value: "A tort.", new_value: "A tort." }),
        finding("tort", { kind: "teleport" } as never),
      ],
      live,
      { ...OPTS, visited: [...OPTS.visited, { ref_id: "run" }] },
    );
    assert.deepEqual(p.edits, []);
    assert.deepEqual(
      p.skipped.map((s) => s.reason),
      ["property_not_allowed", "empty_value", "not_visited", "wrong_type", "no_change", "invalid_action"],
    );
  });

  it("plans a move in graph/move-node's shape, from the node's one live parent; refuses no parent, two parents, a protected or unvisited target", () => {
    const live = new Map(LIVE);
    live.set("orphan", node("orphan", { name: "Orphan" }));
    live.set("twins", node("twins", { name: "Twins" }, { parents: ["law", "tort"] }));
    const visited = [...OPTS.visited, { ref_id: "orphan" }, { ref_id: "twins" }];
    const p = planChanges(
      [
        finding("crime", { kind: "move", new_parent_ref_id: "tort" }),
        finding("orphan", { kind: "move", new_parent_ref_id: "tort" }),
        finding("twins", { kind: "move", new_parent_ref_id: "crime" }),
        finding("tort", { kind: "move", new_parent_ref_id: "janitor" }),
        finding("tort", { kind: "move", new_parent_ref_id: "elsewhere" }),
        finding("tort", { kind: "move", new_parent_ref_id: "law" }),
        finding("tort", { kind: "move", new_parent_ref_id: "tort" }),
      ],
      live,
      { ...OPTS, visited },
    );
    assert.equal(p.moves.length, 1);
    const m = p.moves[0]!;
    assert.equal(m.id, "m1");
    assert.deepEqual([m.ref_id, m.edge_type, m.direction, m.from_ref_id, m.to_ref_id], ["crime", "PARENT_OF", "reverse", "law", "tort"]);
    assert.deepEqual(
      p.skipped.map((s) => s.reason),
      ["no_parent", "many_parents", "protected", "not_visited", "no_change", "invalid_action"],
    );
  });

  it("keeps the most severe changes under the cap, one change per node and property, and the rest as over_cap / duplicate", () => {
    const p = planChanges(
      [
        finding("tort", { kind: "set_property", property: "description", old_value: "A tort.", new_value: "Low." }, { severity: "low" }),
        finding("law", { kind: "set_property", property: "description", old_value: "old words here", new_value: "High." }, { severity: "high" }),
        finding("law", { kind: "set_property", property: "description", old_value: "old words here", new_value: "Again." }, { severity: "high" }),
        finding("crime", { kind: "set_property", property: "description", old_value: "A crime.", new_value: "Medium." }, { severity: "medium" }),
      ],
      LIVE,
      { ...OPTS, max_changes: 2 },
    );
    assert.deepEqual(p.edits.map((e) => [e.id, e.ref_id, e.new_value]), [["e1", "law", "High."], ["e2", "crime", "Medium."]]);
    assert.deepEqual(p.skipped.map((s) => [s.ref_id, s.reason]), [["law", "duplicate"], ["tort", "over_cap"]]);
  });

  it("filters by min_severity and by issue, and routes action none to manual", () => {
    const p = planChanges(
      [
        finding("law", { kind: "none" }, { suggestion: "merge with Legal" }),
        finding("law", undefined),
        finding("tort", { kind: "set_property", property: "description", old_value: "A tort.", new_value: "x" }, { severity: "low" }),
        finding("crime", { kind: "set_property", property: "description", old_value: "A crime.", new_value: "x" }, { severity: "high", issue: "other" }),
      ],
      LIVE,
      { ...OPTS, min_severity: "medium", issues: ["vague"] },
    );
    assert.deepEqual(p.manual, [
      { ref_id: "law", name: "law", issue: "vague", suggestion: "merge with Legal" },
      { ref_id: "law", name: "law", issue: "vague", suggestion: "say more" },
    ]);
    assert.deepEqual(p.skipped.map((s) => [s.ref_id, s.reason]), [["crime", "issue_not_applied"], ["tort", "below_min_severity"]]);
  });
});

// ── the readback: each write's result × the node as read back ─────────────

const state = (ref_id: string, props: Record<string, unknown>, parents: string[] = []): LiveState => ({ ref_id, name: String(props.name ?? ref_id), node_type: "Concept", properties: props, parents });
const EDIT = { id: "e1", kind: "edit" as const, ref_id: "law", name: "Law", property: "description", old_value: "old", new_value: "new", node_data: { description: "new" }, properties_to_be_deleted: [] };
const MOVE = { id: "m1", kind: "move" as const, ref_id: "crime", name: "Crime", from_ref_id: "law", to_ref_id: "tort", node_data: {}, properties_to_be_deleted: [] };

// ── the queue: which nodes a run reviews ──────────────────────────────────

describe("tree order", () => {
  // Medicine
  //   Cardio — Valve, Shared (also under Endocrine)
  //   Cardio Surgery
  //   Endocrine — Thyroid, Adrenal — Adrenal Type
  // plus a cycle back to the start and a node whose only parent is outside the tree.
  const tree: TreeNode[] = [
    { ref_id: "m", name: "Medicine", parents: ["t", "outside"] },
    { ref_id: "c", name: "Cardio", parents: ["m"] },
    { ref_id: "cs", name: "Cardio Surgery", parents: ["m"] },
    { ref_id: "e", name: "Endocrine", parents: ["m"] },
    { ref_id: "v", name: "Valve", parents: ["c"] },
    { ref_id: "s", name: "Shared", parents: ["e", "c"] },
    { ref_id: "t", name: "Thyroid", parents: ["e"] },
    { ref_id: "a", name: "Adrenal", parents: ["e"] },
    { ref_id: "at", name: "Adrenal Type", parents: ["a"] },
    { ref_id: "x", name: "Stray", parents: ["outside"] },
  ];
  const placed = placeTree(tree, "m");
  const paths = (xs: { path: string[] }[]) => xs.map((x) => x.path.join(PATH_SEP));

  it("a parent before its children, a subtree before the next sibling, siblings by name; a node with two parents at its first path", () => {
    assert.deepEqual(paths(placed), [
      "Medicine",
      "Medicine › Cardio",
      "Medicine › Cardio › Shared",
      "Medicine › Cardio › Valve",
      "Medicine › Cardio Surgery",
      "Medicine › Endocrine",
      "Medicine › Endocrine › Adrenal",
      "Medicine › Endocrine › Adrenal › Adrenal Type",
      "Medicine › Endocrine › Thyroid",
    ]);
    assert.deepEqual(
      placed.map((n) => [n.name, n.parent?.name ?? null]),
      [["Medicine", null], ["Cardio", "Medicine"], ["Shared", "Cardio"], ["Valve", "Cardio"], ["Cardio Surgery", "Medicine"], ["Endocrine", "Medicine"], ["Adrenal", "Endocrine"], ["Adrenal Type", "Adrenal"], ["Thyroid", "Endocrine"]],
    );
    assert.ok(!placed.some((n) => n.ref_id === "x"), "unreachable from the start: left out");
    assert.deepEqual(placeTree(tree, "nope"), []);
  });

  it("compares paths by segment, not as joined strings: Cardio's subtree before Cardio Surgery", () => {
    assert.ok(comparePath(["Medicine", "Cardio", "Valve"], ["Medicine", "Cardio Surgery"]) < 0);
    assert.ok("Medicine › Cardio › Valve" > "Medicine › Cardio Surgery", "the joined strings sort the other way");
    assert.ok(comparePath(["Medicine"], ["Medicine", "Cardio"]) < 0, "a prefix first");
    assert.equal(comparePath(["Medicine", "Cardio"], ["Medicine", "Cardio"]), 0);
  });

  it("siblings are the parent's other children, by name — a second parent's children too; the start has none", () => {
    assert.deepEqual(siblingsOf(tree, "e", "t"), ["Adrenal", "Shared"]);
    assert.deepEqual(siblingsOf(tree, "c", "s"), ["Valve"]);
    assert.deepEqual(siblingsOf(tree, undefined, "m"), []);
  });

  const take = (cursor: string | undefined, size = 3) => {
    const p = pickBatch(placed, cursor, size);
    return [p.batch.map((n) => n.name), p.next_cursor, p.from, p.wrapped];
  };

  it("no cursor: the start and the nodes after it; then each run the next ones after the last path taken", () => {
    assert.deepEqual(take(undefined), [["Medicine", "Cardio", "Shared"], "Medicine › Cardio › Shared", 0, false]);
    assert.deepEqual(take("Medicine › Cardio › Shared"), [["Valve", "Cardio Surgery", "Endocrine"], "Medicine › Endocrine", 3, false]);
  });

  it("wraps past the last node, so every node comes round once per ceil(total / size) runs", () => {
    assert.deepEqual(take("Medicine › Endocrine › Adrenal › Adrenal Type"), [["Thyroid", "Medicine", "Cardio"], "Medicine › Cardio", 8, true]);
    const seen = new Map<string, number>();
    let cursor: string | undefined;
    for (let run = 0; run < 9; run++) {
      const p = pickBatch(placed, cursor, 3);
      for (const n of p.batch) seen.set(n.name, (seen.get(n.name) ?? 0) + 1);
      cursor = p.next_cursor;
    }
    // 9 runs × 3 = 27 = 3 full passes over 9 nodes: each exactly three times
    assert.deepEqual([...seen.values()], [3, 3, 3, 3, 3, 3, 3, 3, 3]);
  });

  it("a cursor whose node is gone sorts where it was; one past every path starts over; a plain name (the old cursor) starts over", () => {
    assert.deepEqual(take("Medicine › Cardio › Bypass"), [["Shared", "Valve", "Cardio Surgery"], "Medicine › Cardio Surgery", 2, false]);
    assert.deepEqual(take("Zebra"), [["Medicine", "Cardio", "Shared"], "Medicine › Cardio › Shared", 0, true]);
    assert.deepEqual(take("Hematology"), [["Medicine", "Cardio", "Shared"], "Medicine › Cardio › Shared", 0, false]);
  });

  it("a batch no smaller than the tree is all of it, once; an empty tree is an empty batch", () => {
    assert.deepEqual(take("Medicine › Endocrine", 50)[0], ["Adrenal", "Adrenal Type", "Thyroid", "Medicine", "Cardio", "Shared", "Valve", "Cardio Surgery", "Endocrine"]);
    assert.deepEqual(pickBatch([], "x", 3), { batch: [], from: 0, wrapped: false, next_cursor: "x" });
  });

  it("name order is by code point, ties broken by ref_id: the same on every machine", () => {
    const xs = [{ ref_id: "b", name: "b" }, { ref_id: "z", name: "B" }, { ref_id: "a", name: "B" }, { ref_id: "c", name: "a" }];
    assert.deepEqual([...xs].sort(byName).map((x) => x.ref_id), ["a", "z", "c", "b"]);
  });

  it("renders where each node sits, then its text verbatim, fenced longer than any fence inside it", () => {
    const text = renderBatch([
      { ref_id: "r0", name: "Law", path: ["Law"], parent: null, siblings: [], description: "One line.", docs: "## Rules\n```\ncode\n````\nend" },
      { ref_id: "r1", name: "Tort", path: ["Law", "Tort"], parent: { ref_id: "r0", name: "Law" }, siblings: ["Contract", "Crime"], description: null, docs: null },
    ]);
    assert.match(text, /^### 1\. Law\nref_id: r0\npath: Law\nparent: none — the start of the sweep\nsiblings: none\n/);
    assert.ok(text.includes("### 2. Tort\nref_id: r1\npath: Law › Tort\nparent: Law (ref_id r0)\nsiblings: Contract, Crime\n"), text);
    assert.ok(text.includes("## Rules\n```\ncode\n````\nend"), "docs verbatim");
    assert.ok(text.includes("`````\n## Rules"), "a fence longer than the docs' own");
    const many = Array.from({ length: MAX_SIBLINGS + 3 }, (_, i) => `S${String(i).padStart(3, "0")}`);
    const big = renderBatch([{ ref_id: "r2", name: "Kid", path: ["Law", "Kid"], parent: { ref_id: "r0", name: "Law" }, siblings: many, description: "", docs: "" }]);
    assert.ok(big.includes(`S${String(MAX_SIBLINGS - 1).padStart(3, "0")} … and 3 more\n`), "past the cap: how many more");
  });
});

describe("verdicts", () => {
  it("applied only when the node reads back as planned; a string result is failed; a mismatch is not_verified", () => {
    const live = new Map([["law", state("law", { name: "Law", description: "new" })], ["crime", state("crime", { name: "Crime" }, ["tort"])]]);
    const ok = verdicts({ edits: [EDIT], moves: [MOVE] }, [{ status: "Success", ref_id: "law" }], [{ status: "Success" }], live);
    assert.deepEqual(ok.applied.map((c) => c.id), ["e1", "m1"]);
    assert.deepEqual([ok.failed, ok.not_verified], [[], []]);

    const failed = verdicts({ edits: [EDIT] }, ["graph/edit-node failed — UNKNOWN_ATTRIBUTE: nope"], [], live);
    assert.equal(failed.applied.length, 0);
    assert.match(failed.failed[0]!.error, /UNKNOWN_ATTRIBUTE/);

    const stale = new Map([["law", state("law", { name: "Law", description: "someone else's" })]]);
    const nv = verdicts({ edits: [EDIT] }, [{ status: "Success" }], [], stale);
    assert.deepEqual([nv.not_verified[0]!.expected, nv.not_verified[0]!.found], ["new", "someone else's"]);

    const never = verdicts({ edits: [EDIT] }, [], [], live);
    assert.match(never.failed[0]!.error, /never ran/);
  });

  it("a move that already hung there (Warning) counts as applied when the node reads back under the new parent", () => {
    const live = new Map([["crime", state("crime", { name: "Crime" }, ["tort"])]]);
    const v = verdicts({ moves: [MOVE] }, [], [{ status: "Warning", messages: ["Already hangs there — nothing moved"] }], live);
    assert.equal(v.applied.length, 1);
  });

  it("the report section names each change and how to revert the run", () => {
    const text = reportSection("123", { applied: [{ ...EDIT, result: {} }, { ...MOVE, result: {} }], failed: [], not_verified: [] }, { revert_workflow: "graph-janitor-revert", file: "applied.json" });
    assert.match(text, /^## Applied \(run 123\)/);
    assert.match(text, /- e1 · Law · description: "old" → "new"/);
    assert.match(text, /- m1 · Crime · moved from law to tort/);
    assert.match(text, /run `graph-janitor-revert` with input `\{ "run_id": "123" \}`/);
    assert.match(text, /"changes": \["e1", "m1"\]/);
    const nothing = reportSection("1", { applied: [], failed: [], not_verified: [] }, { revert_workflow: "x", file: "a" });
    assert.match(nothing, /Nothing was written/);
    assert.doesNotMatch(nothing, /To undo/);
  });
});

// ── the revert: the undo log × the live graph → the inverse plan ──────────

describe("revertPlan", () => {
  const applied: Applied[] = [
    { ...EDIT },
    { id: "e2", kind: "edit", ref_id: "tort", name: "Tort", property: "docs", old_value: null, new_value: "added docs", node_data: {}, properties_to_be_deleted: [] },
    { ...MOVE },
  ];
  it("puts the replaced value back, deletes a property that was absent, moves back — only where the janitor's write still stands", () => {
    const live = new Map([
      ["law", state("law", { name: "Law", description: "new" })],
      ["tort", state("tort", { name: "Tort", docs: "added docs" })],
      ["crime", state("crime", { name: "Crime" }, ["tort"])],
    ]);
    const p = revertPlan(applied, live);
    assert.deepEqual(p.skipped, []);
    assert.deepEqual(p.edits.map((e) => [e.id, e.reverts, e.node_data, e.properties_to_be_deleted, e.old_value, e.new_value]), [
      ["e1", "e1", { description: "old" }, [], "new", "old"],
      ["e2", "e2", {}, ["docs"], "added docs", null],
    ]);
    assert.deepEqual(p.moves.map((m) => [m.id, m.from_ref_id, m.to_ref_id, m.edge_type, m.direction]), [["m1", "tort", "law", "PARENT_OF", "reverse"]]);
  });

  it("skips a node someone changed or moved since (moved_on) and one that is gone (not_found)", () => {
    const live = new Map([
      ["law", state("law", { name: "Law", description: "a person's edit" })],
      ["crime", state("crime", { name: "Crime" }, ["law"])],
    ]);
    const p = revertPlan(applied, live);
    assert.deepEqual([p.edits, p.moves], [[], []]);
    assert.deepEqual(p.skipped.map((s) => [s.id, s.reason]), [["e1", "moved_on"], ["e2", "not_found"], ["m1", "moved_on"]]);
  });

  it("changeIds: a list, a comma-separated string, or nothing", () => {
    assert.deepEqual([...changeIds(["e1", " m2 "])!], ["e1", "m2"]);
    assert.deepEqual([...changeIds("e1,m2")!], ["e1", "m2"]);
    assert.equal(changeIds(undefined), null);
    assert.equal(changeIds([]), null);
  });
});

// ── the wiring, offline: the gate, the foreach bodies, what readback gets ──
// The engine over the seeded YAML with graph-get, the agent, the plan and
// readback steps and the two write steps swapped for fakes. The same branch
// over the REAL steps and a live graph is in seed.test.ts (one process: the
// live janitor suites share the Janitor tree, so they must not race).
describe("graph-janitor apply branch + graph-janitor-revert (offline: fakes)", () => {
  const MANDATE = { ref_id: "r-mandate", node_type: "Concept", name: "Some Mandate", properties: { docs: "THE MANDATE TEXT" }, edges: {} };
  const LAW = { ref_id: "r-law", node_type: "Concept", name: "Law", properties: { description: "Legal knowledge." }, edges: {} };
  const JANITOR = { ref_id: "r-janitor", node_type: "Concept", name: "Janitor", properties: {}, edges: {} };
  const kid = { ref_id: MANDATE.ref_id, node_type: "Concept", name: MANDATE.name, description: "What dirty means here." };
  const PLAN = {
    edits: [{ id: "e1", kind: "edit", ref_id: "r-law", name: "Law", property: "description", old_value: "Legal knowledge.", new_value: "The law.", node_data: { description: "The law." }, properties_to_be_deleted: [] }],
    moves: [{ id: "m1", kind: "move", ref_id: "r-tort", name: "Tort", edge_type: "PARENT_OF", direction: "reverse", from_ref_id: "r-law", to_ref_id: "r-crime", node_data: {}, properties_to_be_deleted: [] }],
    skipped: [{ ref_id: "r-janitor", name: "Janitor", issue: "vague", reason: "protected" }],
    manual: [],
    counts: { findings: 3, edits: 1, moves: 1, skipped: 1, manual: 0 },
  };
  const QUEUE = {
    batch: [{ ref_id: "r-law", name: "Law", node_type: "Concept", description: "Legal knowledge.", docs: null }],
    ref_ids: ["r-law"],
    visited: [{ ref_id: "r-law", name: "Law", node_type: "Concept" }],
    text: "### 1. Law",
    next_cursor: "Law",
    from: 0,
    total: 1,
    wrapped: false,
    truncated: false,
  };
  const calls: Record<string, Record<string, any>[]> = { queue: [], plan: [], edit: [], move: [], readback: [], revert: [] };
  const fake = (type: string, run: (cfg: any) => unknown) => defineStep({ type, input: z.object({}).passthrough(), output: z.any(), async run(cfg) { return run(cfg); } });
  const fakes = {
    "graph/graph-get": fake("graph/graph-get", (cfg) => {
      if (cfg.name === "Janitor") return { ...JANITOR, ...(cfg.children ? { children: [kid] } : {}) };
      if (cfg.name === MANDATE.name) return MANDATE;
      if (cfg.name === "Law") return LAW;
      return `node not found: Concept "${cfg.name}"`;
    }),
    agent: fake("agent", () => ({ result: "ok", object: { start_ref_id: LAW.ref_id, visited: [LAW], findings: [{ ref_id: "r-law", name: "Law", issue: "vague", problem: "p", evidence: "Legal", suggestion: "s", severity: "low", action: { kind: "set_property", property: "description", old_value: "Legal knowledge.", new_value: "The law." } }], summary: "one" }, steps: 1, usage: {}, cost: 0 })),
    "janitor/queue": fake("janitor/queue", (cfg) => (calls.queue.push(cfg), QUEUE)),
    "janitor/plan": fake("janitor/plan", (cfg) => (calls.plan.push(cfg), PLAN)),
    "janitor/revert-plan": fake("janitor/revert-plan", (cfg) => (calls.revert.push(cfg), { ...PLAN, revert_of: cfg.run_id, source: { run_id: cfg.run_id } })),
    "graph/edit-node": fake("graph/edit-node", (cfg) => (calls.edit.push(cfg), { status: "Success", ref_id: cfg.ref_id, updated: Object.keys(cfg.node_data ?? {}) })),
    "graph/move-node": fake("graph/move-node", (cfg) => (calls.move.push(cfg), { status: "Success", ref_id: cfg.ref_id, from_ref_id: cfg.from_ref_id, to_ref_id: cfg.to_ref_id })),
    "janitor/readback": fake("janitor/readback", (cfg) => (calls.readback.push(cfg), { applied: [...cfg.plan.edits, ...cfg.plan.moves], failed: [], not_verified: [], applied_json: "/artifacts/x/applied.json", counts: {} })),
  };
  let root: string;
  let ws: WorkspaceManager;
  let run: (name: string, input: Record<string, unknown>) => ReturnType<typeof runWorkflow>;
  before(async () => {
    root = await mkdtemp(join(tmpdir(), "janitor-apply-wf-"));
    ws = new WorkspaceManager(root);
    await seedArtifactSteps(ws);
    await seedJanitorSteps(ws);
    await seedJanitorWorkflows(ws);
    const { registry } = await buildRegistry(await ws.materializeCustomSteps());
    const dataDir = join(root, "data");
    const services = { ...standardServices({ secretsSource: {}, dataDir }), artifacts: fileArtifactsCapability(join(dataDir, "artifacts")) };
    run = async (name, input) => runWorkflow(await ws.getWorkflow(name), input, { ...registry, ...fakes } as typeof registry, { services });
  });
  after(() => rm(root, { recursive: true, force: true }));
  const reset = () => Object.values(calls).forEach((l) => (l.length = 0));

  it("a dry run plans with the findings, the Janitor node and the mandate, and writes nothing", async () => {
    reset();
    const res = await run("graph-janitor", { concept: MANDATE.name, start: "Law" });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    const cfg = calls.plan[0]!;
    assert.equal(cfg.findings.length, 1);
    // findings may only change the batch; what the agent read around it may be a move's new parent
    assert.deepEqual(cfg.targets, ["r-law"]);
    assert.deepEqual(cfg.visited, [LAW]);
    assert.equal(cfg.parent.ref_id, JANITOR.ref_id);
    assert.deepEqual(cfg.parent.children.map((c: any) => c.ref_id), [MANDATE.ref_id]);
    assert.equal(cfg.mandate_ref_id, MANDATE.ref_id);
    assert.deepEqual(cfg.properties, ["description", "docs"]);
    assert.equal(cfg.max_changes, 20);
    assert.deepEqual([calls.edit, calls.move, calls.readback], [[], [], []], "the apply branch never ran");
    const out = res.output as Record<string, any>;
    assert.equal(out.apply, false);
    assert.deepEqual(out.plan.counts, PLAN.counts);
    assert.deepEqual(out.plan.skipped, PLAN.skipped);
    assert.equal(out.applied, undefined);
  });

  it("apply=true writes each planned change through strut's own steps, in the plan's shape, and hands readback the aligned results", async () => {
    reset();
    const res = await run("graph-janitor", { concept: MANDATE.name, start: "Law", apply: true });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    assert.deepEqual(calls.edit, [{ ref_id: "r-law", node_data: { description: "The law." }, properties_to_be_deleted: [] }]);
    assert.deepEqual(calls.move, [{ ref_id: "r-tort", edge_type: "PARENT_OF", direction: "reverse", from_ref_id: "r-law", to_ref_id: "r-crime" }]);
    const rb = calls.readback[0]!;
    assert.equal(rb.workflow, "graph-janitor");
    assert.deepEqual(rb.plan.edits, PLAN.edits);
    assert.deepEqual(rb.edits.map((e: any) => e.status), ["Success"]);
    assert.deepEqual(rb.moves.map((m: any) => m.ref_id), ["r-tort"]);
    const out = res.output as Record<string, any>;
    assert.equal(out.apply, true);
    assert.deepEqual(out.applied.map((c: any) => c.id), ["e1", "m1"]);
    assert.equal(out.applied_json, "/artifacts/x/applied.json");
  });

  it("the revert workflow plans from the run id, writes the inverse through the same steps, and logs revert_of", async () => {
    reset();
    const res = await run("graph-janitor-revert", { run_id: "42", changes: ["e1"] });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    assert.deepEqual(calls.revert, [{ run_id: "42", changes: ["e1"] }]);
    assert.equal(calls.edit.length, 1);
    assert.equal(calls.move.length, 1);
    const rb = calls.readback[0]!;
    assert.deepEqual([rb.workflow, rb.revert_of], ["graph-janitor-revert", "42"]);
    const out = res.output as Record<string, any>;
    assert.equal(out.revert_of, "42");
    assert.deepEqual(out.applied.map((c: any) => c.id), ["e1", "m1"]);
  });

  it("both workflows are seeded under graph-maintenance; the revert's changes input is optional", async () => {
    const list = await ws.listWorkflows();
    for (const n of ["graph-janitor", "graph-janitor-revert"]) assert.equal(list.find((w) => w.name === n)?.category, "graph-maintenance", n);
    const flow = (await ws.getWorkflow("graph-janitor-revert")) as any;
    assert.deepEqual(Object.keys(flow.inputBlock).sort(), ["changes", "run_id"]);
    assert.equal(flow.inputBlock.changes.required, false);
  });
});
