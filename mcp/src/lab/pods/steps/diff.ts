import { z, defineStep } from "strut";
import { brief, capText, podCall, type PodCtx } from "./_shared.js";

export const File = z.object({
  file: z.string(),
  action: z.string(),
  repoName: z.string(),
  content: z.string(),
});

/** Staklink's file list, each file's content capped, the list bounded. */
export function capFiles(raw: unknown, perFile: number, maxFiles: number): { files: z.infer<typeof File>[]; total: number } {
  if (!Array.isArray(raw)) throw new Error(`the pod returned no file list: ${brief(raw)}`);
  const files = raw.map((f: any) => ({
    file: String(f?.file ?? ""),
    action: String(f?.action ?? ""),
    repoName: String(f?.repoName ?? ""),
    content: capText(String(f?.content ?? ""), perFile),
  }));
  return { files: files.slice(0, maxFiles), total: files.length };
}

export default defineStep({
  type: "pod/diff",
  description:
    "What is UNCOMMITTED in the pod's working trees, across its repositories: one entry per changed file (create | modify | delete) with its changed content " +
    "(long files cut in the middle). Empty when the working trees are clean. For the whole branch against its base — committed work included — use pod/branch-diff. " +
    "Output: { files: [{ file, action, repoName, content }], total }.",
  input: z.object({
    control: z.string().min(1).describe("The pod's control URL (pod/claim)"),
    sealed: z.string().min(1).describe("The pod's sealed password (pod/claim)"),
    perFile: z.number().int().positive().default(6000).describe("Cap on one file's content"),
    maxFiles: z.number().int().positive().default(50).describe("How many files to list; `total` counts them all"),
  }),
  output: z.object({ files: z.array(File), total: z.number() }),
  async run(cfg, ctx: PodCtx) {
    const res = await podCall(ctx, cfg.control, cfg.sealed, "/diff");
    if (!res.ok) throw new Error(`GET /diff: ${res.status} — ${brief(res.body)}`);
    return capFiles(res.body, cfg.perFile, cfg.maxFiles);
  },
});
