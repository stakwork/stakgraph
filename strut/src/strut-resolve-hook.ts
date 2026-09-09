/**
 * Module customization hook (node:module `register`): resolve the bare
 * specifier `"strut"` to the *running* strut's own entry module.
 *
 * Why: custom steps are user files under the workspace
 * (`<workspace>/steps/custom/<name>.ts`) that `import { defineStep } from
 * "strut"`. Node resolves a bare specifier by walking up from the importing
 * file, so that only works when the workspace happens to sit inside a
 * directory tree that can see a `strut` package (the dev checkout via package
 * self-reference, mcp via its vendored copy). A desktop install keeps the
 * workspace in Application Support, nowhere near the bundle, and a server can
 * point STRUT_WORKSPACE anywhere. This hook removes the dependence on
 * placement and guarantees the step gets the same module instance the server
 * runs, not a second copy.
 *
 * Registered once by `registerStrutResolver()` (strut-resolver.ts). Runs on
 * Node's hooks thread; keep it dependency-free. Everything that isn't exactly
 * `"strut"` falls through to the next resolver (tsx in dev, Node's default in
 * prod), so subpaths and every other package behave as before.
 *
 * `"vein"` is the pre-rename package name (#1664) and resolves the same way:
 * step versions published before the rename live in the graph verbatim —
 * their content hash is their identity — and still `import "vein"`.
 */
let entry = "";

const SELF_SPECIFIERS = new Set(["strut", "vein"]);

export function initialize(data: { entry: string }): void {
  entry = data.entry;
}

type ResolveContext = { parentURL?: string; conditions: string[]; importAttributes: Record<string, string> };
type ResolveResult = { url: string; format?: string | null; shortCircuit?: boolean };
type NextResolve = (specifier: string, context: ResolveContext) => Promise<ResolveResult>;

export async function resolve(specifier: string, context: ResolveContext, next: NextResolve): Promise<ResolveResult> {
  if (entry && SELF_SPECIFIERS.has(specifier)) return { url: entry, shortCircuit: true };
  return next(specifier, context);
}
