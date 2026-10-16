import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, normalize, relative, resolve } from "node:path";
import { z, defineStep, jobFileUrl, jobRoot } from "strut";
import { podBytes, type PodCtx } from "./_shared.js";

/** A relative path that stays inside its root; the normalized form, or an error. */
export function safeRel(p: string, what: string): string {
  const n = normalize(String(p).replace(/\\/g, "/"));
  if (!p || isAbsolute(p) || n === "." || n === ".." || n.startsWith("../") || n.split("/").includes("..") || p.includes("\0")) {
    throw new Error(`\`${what}\` must be a relative path that stays inside its directory, got ${JSON.stringify(p)}`);
  }
  return n;
}

export default defineStep({
  type: "pod/file",
  description:
    "Bring one file out of the pod into this job's directory (or, with no job, this run's artifacts): GET /file on the pod's control server, which serves " +
    "files from its staklink files dir (default /tmp/staklink-files), up to 10 MB. Use it for something a pod's browser or script wrote there, such as a cropped captcha image. " +
    "`path` is relative to that dir; `to` is the relative path to write here (e.g. captcha/round-1.png). " +
    "Output: { path, bytes, url } — `path` is `to`, `url` the served link.",
  input: z.object({
    control: z.string().min(1).describe("The pod's control URL (pod/claim)"),
    sealed: z.string().min(1).describe("The pod's sealed password (pod/claim)"),
    path: z.string().min(1).describe("File path relative to the pod's staklink files dir"),
    to: z.string().min(1).describe("Relative path to write in this job's directory (or run's artifacts)"),
  }),
  output: z.object({ path: z.string(), bytes: z.number(), url: z.string() }),
  async run(cfg, ctx: PodCtx) {
    const to = safeRel(cfg.to, "to");
    safeRel(cfg.path, "path");
    const route = `/file?path=${encodeURIComponent(cfg.path)}`;
    const res = await podBytes(ctx, cfg.control, cfg.sealed, route);
    if (res.status === 400) throw new Error(`GET /file: 400 — the pod refused the path ${JSON.stringify(cfg.path)} (it must be relative to its files dir)`);
    if (res.status === 404) throw new Error(`GET /file: 404 — no file ${JSON.stringify(cfg.path)} in the pod's files dir`);
    if (res.status === 413) throw new Error(`GET /file: 413 — ${JSON.stringify(cfg.path)} is over the pod's 10 MB limit`);
    if (!res.ok) throw new Error(`GET /file: ${res.status} — ${res.text}`);

    if (ctx.job) {
      const root = resolve(jobRoot(ctx.services?.dataDir ?? join(tmpdir(), "strut"), ctx.job));
      const dest = resolve(root, to);
      const rel = relative(root, dest);
      if (!rel || rel.startsWith("..") || isAbsolute(rel)) throw new Error(`\`to\` escapes the job directory: ${JSON.stringify(cfg.to)}`);
      await mkdir(dirname(dest), { recursive: true });
      await writeFile(dest, res.bytes);
      return { path: to, bytes: res.bytes.length, url: jobFileUrl(ctx.job, to) };
    }
    const artifacts = ctx.services?.artifacts;
    if (!artifacts) throw new Error("pod/file: no job and no `artifacts` capability on ctx.services");
    await artifacts.write(ctx.runId, to, res.bytes);
    return { path: to, bytes: res.bytes.length, url: `/artifacts/${ctx.runId}/${to}` };
  },
});
