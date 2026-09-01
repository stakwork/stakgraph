import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";

/** Resolve the Jarvis base URL + auth via the secrets capability (secret
 *  store → env fallback). Duplicated in every jarvis/* step — see _shared.ts. */
async function jarvisCtx(ctx?: StepContext<VeinCapabilities>) {
  const http = ctx?.services?.http;
  if (!http) throw new Error("jarvis: ctx.services.http unavailable — run with a services bag");
  const secrets = ctx?.services?.secrets;
  const base = (await secrets?.get("JARVIS_URL"))?.replace(/\/+$/, "");
  if (!base) throw new Error("jarvis: JARVIS_URL not configured (set it in the mcp env or the vein secret store)");
  const token = (await secrets?.get("API_TOKEN")) ?? "";
  const rawTimeout = Number(await secrets?.get("JARVIS_HTTP_TIMEOUT_MS"));
  const timeout = Number.isFinite(rawTimeout) && rawTimeout > 0 ? rawTimeout : 180_000;
  return { base, http, timeout, headers: { "X-Api-Token": token } };
}

export default defineStep({
  type: "jarvis/register-namespace",
  description:
    "Register (create) a Jarvis NAMESPACE — a named data partition that scopes node/edge writes " +
    "(pass the same name as the `namespace` config of the jarvis write steps). Idempotent: registering " +
    "a namespace that already exists is a success. Namespaces are not an access-control boundary.",
  input: z.object({
    namespace: z
      .string()
      .min(1)
      .describe("Namespace name to register, e.g. a task slug. Reuse the exact same string in later jarvis write steps."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);
    const res = await http(`${base}/namespace`, {
      method: "POST",
      headers,
      timeout,
      body: { namespace: cfg.namespace },
    });
    if (res.ok) return { namespace: cfg.namespace, registered: true };

    // A non-2xx may just mean "already registered" — confirm against the list
    // before reporting failure, so reruns of a pipeline never die here.
    const list = await http(`${base}/namespace`, { headers, timeout });
    const names = (list.body as any)?.data?.namespace;
    if (Array.isArray(names) && names.includes(cfg.namespace)) {
      return { namespace: cfg.namespace, registered: true, alreadyExisted: true };
    }
    return `jarvis/register-namespace failed — HTTP ${res.status}: ${
      typeof res.body === "string" ? res.body : JSON.stringify(res.body)
    }`;
  },
});
