/**
 * Smoke test for the GAIA LAB scoring service — no numpy, no dataset
 * download, no LLM. Uses a REAL throwaway git repo as a fake dataset
 * checkout (fixture metadata.jsonl + a stub scorer.py) and REAL python3 for
 * the driver, so the actual subprocess plumbing is exercised end-to-end.
 * Verifies:
 *   - dataset integrity enforcement: missing dir, dirty tree, missing
 *     scorer, sha-pin mismatch all refuse; untracked scorer.py is tolerated
 *   - getTask/listTasks return question/level/file and STRIP the gold
 *   - score() runs the pinned scorer via the python driver, aggregates
 *     accuracy + byLevel, flags unknown task ids, and stamps
 *     benchmarkRev/scorerSha256
 *
 * Run: npx tsx src/lab/gaia/smoke.ts
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildGaiaServices } from "./service.js";
import {
  ensureGaiaDataset,
  resetGaiaBootstrap,
  type BootstrapExecFn,
} from "./bootstrap.js";

function git(cwd: string, ...args: string[]): string {
  return execFileSync("git", args, { cwd, encoding: "utf-8" });
}

const T1 = "task-aaa";
const T2 = "task-bbb";
const T3 = "task-ccc";

/** Stub scorer.py: same signature as the real one, trivial logic (case-
 *  insensitive trimmed match; prints chatter to prove redirect_stdout works). */
const STUB_SCORER = `
def question_scorer(model_answer, ground_truth):
    print("chatter that must not corrupt the json")
    return model_answer.strip().lower() == ground_truth.strip().lower()
`;

/** The stub's hash — the pin is now ALWAYS enforced (service.ts defaults to
 *  the in-repo SCORER_SHA256), so fixture runs must pin the fixture. */
const STUB_SHA = createHash("sha256").update(STUB_SCORER).digest("hex");

function makeFakeDataset(dir: string): void {
  git(dir, "init", "-q");
  git(dir, "config", "user.email", "smoke@test");
  git(dir, "config", "user.name", "smoke");
  const split = join(dir, "2023", "validation");
  mkdirSync(split, { recursive: true });
  const rows = [
    { task_id: T1, Question: "What is 2+2?", Level: 1, file_name: "", "Final answer": "4" },
    { task_id: T2, Question: "Read the sheet.", Level: 2, file_name: "sheet.xlsx", "Final answer": "blue" },
    { task_id: T3, Question: "Hardest one.", Level: 3, file_name: "", "Final answer": "42" },
  ];
  writeFileSync(join(split, "metadata.jsonl"), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  writeFileSync(join(split, "sheet.xlsx"), "fake-xlsx-bytes");
  git(dir, "add", "-A");
  git(dir, "commit", "-qm", "init");
  // scorer.py arrives UNTRACKED — from the leaderboard Space, not this repo.
  writeFileSync(join(dir, "scorer.py"), STUB_SCORER);
}

async function main() {
  const dir = mkdtempSync(join(tmpdir(), "gaia-smoke-"));
  try {
    makeFakeDataset(dir);

    // Fixture runs: auto-setup OFF (verifyDataset is under test in isolation)
    // and the stub scorer pinned explicitly.
    const svc = (o: Record<string, unknown> = {}) =>
      buildGaiaServices({ dir, autoSetup: false, scorerSha256: STUB_SHA, ...o });

    // ── missing dir refuses ──
    await assert.rejects(
      () => svc({ dir: join(dir, "nope") }).verifyDataset(),
      /does not exist/,
    );

    // ── happy verify: untracked scorer.py tolerated, rev + sha reported ──
    const gaia = svc();
    const v = await gaia.verifyDataset();
    assert.equal(v.rev.length, 40);
    assert.match(v.scorerSha256, /^[0-9a-f]{64}$/);

    // ── sha pin mismatch refuses ──
    await assert.rejects(
      () => svc({ scorerSha256: "f".repeat(64) }).verifyDataset(),
      /does not match pinned/,
    );

    // ── listTasks / getTask: levels, files, and NO gold ──
    const tasks = await gaia.listTasks();
    assert.equal(tasks.length, 3);
    assert.deepEqual(await gaia.listTasks({ level: 2 }), [
      { taskId: T2, level: 2, hasFile: true },
    ]);
    const t2 = await gaia.getTask(T2);
    assert.equal(t2.question, "Read the sheet.");
    assert.equal(t2.fileName, "sheet.xlsx");
    assert.ok(t2.filePath?.endsWith(join("2023", "validation", "sheet.xlsx")));
    assert.ok(!JSON.stringify(t2).includes("blue"), "gold leaked from getTask");
    assert.ok(!JSON.stringify(tasks).includes("4"), "gold leaked from listTasks");
    await assert.rejects(() => gaia.getTask("nope"), /not in the validation split/);

    // ── score(): real python3 driver + stub scorer ──
    const report = await gaia.score([
      { taskId: T1, answer: " 4 " }, // correct after normalization
      { taskId: T2, answer: "red" }, // wrong
      { taskId: T3, answer: "42" }, // correct
      { taskId: "ghost", answer: "x" }, // unknown id
    ]);
    assert.equal(report.correct, 2);
    assert.equal(report.total, 4);
    assert.equal(report.accuracy, 2 / 3); // unknown id excluded from denominator
    assert.deepEqual(report.byLevel, {
      "1": { correct: 1, total: 1 },
      "2": { correct: 0, total: 1 },
      "3": { correct: 1, total: 1 },
    });
    const ghost = report.results.find((r) => r.taskId === "ghost");
    assert.match(ghost?.error ?? "", /unknown task_id/);
    assert.equal(report.benchmarkRev, v.rev);
    assert.equal(report.scorerSha256, v.scorerSha256);

    // ── empty pairs refuses ──
    await assert.rejects(() => gaia.score([]), /no pairs/);

    // ── dirty tree refuses (tracked file modified) ──
    writeFileSync(join(dir, "2023", "validation", "metadata.jsonl"), "{}\n");
    await assert.rejects(() => gaia.score([{ taskId: T1, answer: "4" }]), /local modifications/);
    git(dir, "checkout", "--", ".");

    // ── missing scorer refuses ──
    rmSync(join(dir, "scorer.py"));
    await assert.rejects(() => svc().verifyDataset(), /scorer\.py not found/);

    // ══ bootstrap (auto-setup) — no network, no real git ═════════════════
    await bootstrapSmoke();

    console.log("gaia smoke: ALL PASS");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}


/**
 * Auto-setup paths, fully offline: `exec` is faked (no git, no git-lfs, no
 * 210MB fetch) and `fetchText` returns the stub scorer instead of hitting the
 * leaderboard Space. Models the REAL step sequence — init, remote add,
 * `fetch --depth 1 <pinned sha>`, `checkout FETCH_HEAD` — because the pinned
 * revision is not a branch tip (see DATASET_REV).
 */
async function bootstrapSmoke(): Promise<void> {
  const box = mkdtempSync(join(tmpdir(), "gaia-boot-"));
  const log = () => {};
  try {
    const fakeExec =
      (
        opts: {
          lfs?: boolean;
          fetchErr?: string;
          lfsPointers?: boolean;
          noMeta?: boolean;
          seenAuth?: string[];
        } = {},
      ): BootstrapExecFn =>
      async (_cmd, args, o) => {
        if (args[0] === "lfs") {
          return opts.lfs === false
            ? { code: 1, stdout: "", stderr: "git: 'lfs' is not a git command" }
            : { code: 0, stdout: "git-lfs/3.4.0\n", stderr: "" };
        }
        // Record whether a credential helper was attached, so the token-optional
        // behaviour is asserted rather than assumed.
        if (opts.seenAuth && args.includes("fetch")) {
          opts.seenAuth.push(args.some((a) => a.startsWith("credential.helper=")) ? "auth" : "anon");
        }
        if (args.includes("fetch") && opts.fetchErr) {
          return { code: 128, stdout: "", stderr: opts.fetchErr };
        }
        if (args.includes("checkout")) {
          const split = join(o.cwd, "2023", "validation");
          mkdirSync(split, { recursive: true });
          if (!opts.noMeta) {
            writeFileSync(join(split, "metadata.jsonl"), JSON.stringify({ task_id: "x" }) + "\n");
          }
          writeFileSync(
            join(split, "attachment.xlsx"),
            opts.lfsPointers
              ? "version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\n"
              : "PK\u0003\u0004real-xlsx-bytes",
          );
          mkdirSync(join(o.cwd, ".git"), { recursive: true });
        }
        return { code: 0, stdout: "", stderr: "" };
      };
    const fetchStub = async () => STUB_SCORER;

    // ── NO token is fine: HF serves this repo's git endpoints anonymously ──
    resetGaiaBootstrap();
    const anon: string[] = [];
    const anonDir = join(box, "anon");
    const okAnon = await ensureGaiaDataset({
      dir: anonDir, hfToken: "", exec: fakeExec({ seenAuth: anon }),
      fetchText: fetchStub, scorerSha256: STUB_SHA, log,
    });
    assert.equal(okAnon.root, anonDir);
    assert.deepEqual(anon, ["anon"], "fetched with a credential helper despite no token");

    // ── a token, when present, IS attached to the fetch ──
    resetGaiaBootstrap();
    const withTok: string[] = [];
    await ensureGaiaDataset({
      dir: join(box, "tok"), hfToken: "hf_xxx", exec: fakeExec({ seenAuth: withTok }),
      fetchText: fetchStub, scorerSha256: STUB_SHA, log,
    });
    assert.deepEqual(withTok, ["auth"], "configured token was not attached to the fetch");

    // ── git-lfs missing refuses BEFORE fetching (silent pointer stubs) ──
    resetGaiaBootstrap();
    await assert.rejects(
      () =>
        ensureGaiaDataset({
          dir: join(box, "b"), exec: fakeExec({ lfs: false }), fetchText: fetchStub, log,
        }),
      /git-lfs is required/,
    );

    // ── HF's opaque gated failure ("expected 'packfile'") is recognised and,
    //    with no token configured, the message points at HF_TOKEN ──
    resetGaiaBootstrap();
    const gated = join(box, "c");
    await assert.rejects(
      () =>
        ensureGaiaDataset({
          dir: gated, hfToken: "",
          exec: fakeExec({ fetchErr: "fatal: expected 'packfile'" }),
          fetchText: fetchStub, log,
        }),
      /refused \(no HF token configured\)[\s\S]*Set HF_TOKEN/,
    );
    assert.ok(!existsSync(`${gated}.partial`), "staging dir left behind after a failed fetch");

    // ── same failure WITH a token blames the un-automatable click-through ──
    resetGaiaBootstrap();
    await assert.rejects(
      () =>
        ensureGaiaDataset({
          dir: join(box, "c2"), hfToken: "hf_xxx",
          exec: fakeExec({ fetchErr: "remote: Access to dataset is restricted. Please log in." }),
          fetchText: fetchStub, log,
        }),
      /not accepted GAIA's terms/,
    );

    // ── a revision without the metadata is rejected, not published ──
    //    (this is what a plain clone of `main` would produce today)
    resetGaiaBootstrap();
    const nometa = join(box, "nm");
    await assert.rejects(
      () =>
        ensureGaiaDataset({
          dir: nometa, exec: fakeExec({ noMeta: true }), fetchText: fetchStub,
          scorerSha256: STUB_SHA, log,
        }),
      /no 2023.*metadata\.jsonl[\s\S]*DATASET_REV/,
    );
    assert.ok(!existsSync(nometa), "a metadata-less revision was published anyway");

    // ── scorer hash mismatch refuses and writes NOTHING ──
    resetGaiaBootstrap();
    const mism = join(box, "d");
    await assert.rejects(
      () =>
        ensureGaiaDataset({
          dir: mism, exec: fakeExec(), fetchText: fetchStub, scorerSha256: "a".repeat(64), log,
        }),
      /refusing to install/,
    );
    assert.ok(!existsSync(join(mism, "scorer.py")), "unpinned scorer.py was installed anyway");

    // ── unresolved LFS pointers are caught, not silently graded on ──
    resetGaiaBootstrap();
    await assert.rejects(
      () =>
        ensureGaiaDataset({
          dir: join(box, "e"), exec: fakeExec({ lfsPointers: true }),
          fetchText: fetchStub, scorerSha256: STUB_SHA, log,
        }),
      /unresolved git-lfs pointers/,
    );

    // ── happy path: checkout + scorer installed, verified against the pin ──
    resetGaiaBootstrap();
    const ok = join(box, "f");
    const res = await ensureGaiaDataset({
      dir: ok, exec: fakeExec(), fetchText: fetchStub, scorerSha256: STUB_SHA, log,
    });
    assert.equal(res.root, ok);
    assert.ok(existsSync(join(ok, "2023", "validation", "metadata.jsonl")));
    assert.equal(readFileSync(join(ok, "scorer.py"), "utf-8"), STUB_SCORER);
    assert.ok(!existsSync(`${ok}.partial`), "staging dir survived a successful fetch");

    // ── idempotent: a populated dir short-circuits without touching git ──
    resetGaiaBootstrap();
    let execCalls = 0;
    const counting: BootstrapExecFn = async () => {
      execCalls += 1;
      return { code: 0, stdout: "", stderr: "" };
    };
    const again = await ensureGaiaDataset({
      dir: ok, exec: counting,
      fetchText: async () => { throw new Error("must not refetch scorer"); },
      scorerSha256: STUB_SHA, log,
    });
    assert.equal(again.root, ok);
    assert.equal(execCalls, 0, "populated checkout still shelled out to git");

    // ── a half-written checkout is re-fetched, not limped along ──
    resetGaiaBootstrap();
    const partial = join(box, "g");
    mkdirSync(partial, { recursive: true });
    writeFileSync(join(partial, "scorer.py"), STUB_SCORER); // scorer but no .git/meta
    const healed = await ensureGaiaDataset({
      dir: partial, exec: fakeExec(), fetchText: fetchStub, scorerSha256: STUB_SHA, log,
    });
    assert.equal(healed.root, partial);
    assert.ok(existsSync(join(partial, "2023", "validation", "metadata.jsonl")));

    console.log("gaia bootstrap smoke: PASS");
  } finally {
    resetGaiaBootstrap();
    rmSync(box, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
