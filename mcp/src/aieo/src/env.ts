/**
 * Where aieo reads the environment — `process.env` unless the host hands it
 * another source with `useEnv`. A host that empties `process.env` after
 * boot (stakgraph's mcp, once its lab strut scrubs it so a workflow step
 * cannot read the deployment's keys) passes the copy it kept, and aieo's
 * per-call reads (`LLM_PROVIDER`, the provider keys, `LLM_GATEWAY_URL`,
 * `MAX_OUTPUT_TOKENS`, …) keep resolving. Reads go through `env`, never
 * `process.env`, so one call covers them all.
 */

let source: NodeJS.ProcessEnv = process.env;

/** Read the environment from `source` from now on (the host's own copy). */
export function useEnv(next: NodeJS.ProcessEnv): void {
  source = next;
}

/** The environment as aieo sees it. */
export const env: NodeJS.ProcessEnv = new Proxy({} as NodeJS.ProcessEnv, {
  get: (_t, k) => (typeof k === "string" ? source[k] : undefined),
  has: (_t, k) => typeof k === "string" && k in source,
  ownKeys: () => Reflect.ownKeys(source),
  getOwnPropertyDescriptor: (_t, k) => {
    const d = Object.getOwnPropertyDescriptor(source, k);
    return d && { ...d, configurable: true };
  },
});
