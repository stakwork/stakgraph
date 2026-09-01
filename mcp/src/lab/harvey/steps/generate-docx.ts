import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdir, readFile, writeFile, unlink } from "node:fs/promises";
import { isAbsolute, join, resolve, sep, dirname } from "node:path";
import { tmpdir } from "node:os";

const execFileAsync = promisify(execFile);

/**
 * The `generate_docx` equivalent for the harvey-deliver pipeline (equivalent
 * functionality to the repo agent's docgen tool, pandoc-only — no templates,
 * no paraId stamping). Converts markdown (inline, or a file the agent built
 * incrementally) to a real .docx at a path INSIDE this run's artifacts dir.
 * Grant via agentTools to the drafter/aggregator agents.
 */
async function artifactsPath(ctx: StepContext | undefined, filename: string): Promise<string> {
  const c = ctx as StepContext<VeinCapabilities> | undefined;
  const artifacts = c?.services?.artifacts;
  if (!artifacts) throw new Error("artifacts capability unavailable — is this the lab vein?");
  const base = await artifacts.dir(c!.runId);
  if (isAbsolute(filename) || filename.split(/[\\/]/).includes("..")) {
    throw new Error(`filename must be a relative path without '..': "${filename}"`);
  }
  const path = resolve(join(base, filename));
  if (path !== base && !path.startsWith(base + sep)) {
    throw new Error(`filename escapes the run's artifacts dir: "${filename}"`);
  }
  await mkdir(dirname(path), { recursive: true });
  return path;
}

export default defineStep({
  type: "harvey/generate-docx",
  description:
    "Generate a real .docx file from markdown, written INSIDE this run's artifacts directory (your " +
    "working directory). `filename` is relative to it — e.g. 'draft_1/memo.docx' or 'output/memo.docx' " +
    "(parent dirs are created). Provide EITHER `markdown` (inline content — fine for short documents) OR " +
    "`markdownPath` (path to a .md file you built incrementally — prefer this for long documents, since " +
    "one tool call must carry the whole string). Conversion is pandoc; the file is verified non-empty. " +
    "Returns { path, filename, bytes }.",
  input: z.object({
    filename: z
      .string()
      .min(1)
      .describe("Output .docx path RELATIVE to the run's artifacts dir, e.g. 'output/memo.docx'."),
    markdown: z.string().optional().describe("Inline markdown content to convert."),
    markdownPath: z
      .string()
      .optional()
      .describe("Path to an existing markdown file to convert instead of inline content."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    if (!cfg.markdown && !cfg.markdownPath) {
      return "harvey/generate-docx failed: provide either `markdown` (inline) or `markdownPath` (path to a .md file).";
    }
    try {
      const outFile = await artifactsPath(ctx as StepContext, cfg.filename);
      let srcPath = cfg.markdownPath;
      let tmp: string | undefined;
      if (!srcPath) {
        tmp = join(tmpdir(), `gen-docx-${Date.now()}-${Math.random().toString(36).slice(2)}.md`);
        await writeFile(tmp, cfg.markdown!, "utf-8");
        srcPath = tmp;
      }
      try {
        await execFileAsync("pandoc", [srcPath, "-o", outFile], { timeout: 60_000 });
      } finally {
        if (tmp) await unlink(tmp).catch(() => {});
      }
      const bytes = (await readFile(outFile)).length;
      if (bytes === 0) return `harvey/generate-docx failed: pandoc produced an empty file at ${cfg.filename}`;
      return { path: outFile, filename: cfg.filename, bytes };
    } catch (err: any) {
      // Teaching string, never a throw at the LLM.
      return `harvey/generate-docx failed: ${err?.message ?? String(err)}`;
    }
  },
});
