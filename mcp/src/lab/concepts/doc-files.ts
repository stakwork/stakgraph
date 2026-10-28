/**
 * The files that describe a repository AS A WHOLE — what it is, how it is
 * run, how to work in it — as the bootstrap agent names them from a clone
 * (strut's plans/concept-tree.md: the pick is recorded per repo in
 * ConceptsMetadata, and `concepts/pages` distils the repository's page from
 * exactly those files on every later sweep, so the page's inputs stay
 * deterministic and the hash rule holds). Pure: the agent's prompt, its
 * schema, and the clean-up of what it answered.
 */

export const MAX_DOC_FILES = 8;

export const DOC_FILES_DESCRIPTION =
  "The files that describe the repository AS A WHOLE — what it is, how it is run, how to work in it: a README, AGENTS.md or CLAUDE.md, a docs index, an architecture or contributing guide. Paths from the repository root. Never a feature's own file, never source. At most 8; empty when there is none.";

export const IDENTIFY_DOCS_SYSTEM =
  "You find the files that describe a repository as a whole. Use the provided tools to look at the repository root and its docs folder, open what looks like a guide, and answer with paths only.";

export const IDENTIFY_DOCS_PROMPT = `List the files in this repository that describe it as a whole — what it is and is for, how it is run and tested, how to work in it. Think README, AGENTS.md, CLAUDE.md, CONTRIBUTING, an ARCHITECTURE or design doc, a docs index. Paths relative to the repository root. Never a feature's own doc, never source code, never a changelog. At most ${MAX_DOC_FILES}, the most useful first; an empty list when the repository has none.`;

export const IDENTIFY_DOCS_SCHEMA = {
  type: "object",
  properties: {
    docFiles: { type: "array", items: { type: "string" }, description: DOC_FILES_DESCRIPTION },
  },
  required: ["docFiles"],
};

/** The model's list, cleaned: strings, relative, no `..`, deduped, capped, and only those `exists` vouches for. */
export function normalizeDocPaths(paths: unknown, exists: (p: string) => boolean): string[] {
  if (!Array.isArray(paths)) return [];
  const out: string[] = [];
  for (const raw of paths) {
    if (typeof raw !== "string") continue;
    const p = raw.trim().replace(/^(\.\/|\/)+/, "").replace(/\/+$/, "");
    if (!p || p.split("/").includes("..") || out.includes(p)) continue;
    if (!exists(p)) continue;
    out.push(p);
    if (out.length >= MAX_DOC_FILES) break;
  }
  return out;
}
