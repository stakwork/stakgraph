import { z, defineStep } from "strut";
import { brief, podCall, POD_REF, podAccess, type PodCtx } from "./_shared.js";
import { File, capFiles } from "./diff.js";

export default defineStep({
  type: "pod/branch-diff",
  description:
    "The pod's branch against its base, across its repositories — committed work included — one entry per changed file (create | modify | delete) with its " +
    "content (long files cut in the middle). `base` is the branch to compare with; each repository's default branch when omitted. A repository sitting on its " +
    "base shows only its uncommitted work (what pod/diff shows); one whose base cannot be found is left out. Read it before pushing, and to describe a " +
    "pull request. Output: { files: [{ file, action, repoName, content }], total }.",
  input: z.object({
    ...POD_REF,
    base: z.string().min(1).optional().describe("The branch to diff against; the repository's default branch when omitted"),
    perFile: z.number().int().positive().default(6000).describe("Cap on one file's content"),
    maxFiles: z.number().int().positive().default(50).describe("How many files to list; `total` counts them all"),
  }),
  output: z.object({ files: z.array(File), total: z.number() }),
  async run(cfg, ctx: PodCtx) {
    const pod = await podAccess(ctx, cfg);
    const path = `/branch-diff${cfg.base ? `?base=${encodeURIComponent(cfg.base)}` : ""}`;
    const res = await podCall(ctx, pod.control, pod.sealed, path);
    if (!res.ok) throw new Error(`GET ${path}: ${res.status} — ${brief(res.body)}`);
    return capFiles(res.body, cfg.perFile, cfg.maxFiles);
  },
});
