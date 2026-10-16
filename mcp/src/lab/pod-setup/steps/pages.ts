import { z, defineStep } from "strut";

/**
 * pod-setup/pages — Concept pages as text a prompt reads. A template cannot
 * hold a newline (a `{{ }}` spanning lines is not a template), so a list of
 * pages joined with blank lines is made here: `pages` (graph/graph-get
 * outputs) each under its name with its docs, or `children` (a graph-get's
 * table of contents) one line each, name and description. `names` is what
 * an enum offers. Self-contained: imports only `strut`.
 */

export function renderPages(pages: Array<{ name?: unknown; properties?: Record<string, unknown> }>): string {
  return pages
    .map((p) => {
      const docs = String(p.properties?.docs ?? p.properties?.description ?? "").trim();
      return docs ? `## ${String(p.name)}\n\n${docs}` : `## ${String(p.name)}`;
    })
    .join("\n\n");
}

export function renderMenu(children: Array<{ name?: unknown; description?: unknown }>): string {
  return children.map((c) => `- ${String(c.name)}${c.description ? `: ${String(c.description)}` : ""}`).join("\n");
}

export default defineStep({
  type: "pod-setup/pages",
  description:
    "Render Concept pages as text a prompt reads: `pages` (graph/graph-get outputs) each under its name as a heading with its docs, or `children` " +
    "(a graph-get's `children`) one line each with its description. Output: { names, text }.",
  input: z
    .object({
      pages: z.array(z.object({ name: z.string() }).passthrough()).optional().describe("graph/graph-get outputs"),
      children: z.array(z.object({ name: z.string() }).passthrough()).optional().describe("a graph/graph-get's `children`"),
    })
    .refine((c) => !!c.pages !== !!c.children, "give `pages` or `children`"),
  output: z.object({ names: z.array(z.string()), text: z.string() }),
  async run(cfg) {
    const items = (cfg.pages ?? cfg.children)!;
    return {
      names: items.map((p) => p.name),
      text: cfg.pages ? renderPages(cfg.pages as any) : renderMenu(cfg.children as any),
    };
  },
});
