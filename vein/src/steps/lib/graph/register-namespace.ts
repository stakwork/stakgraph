import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText } from "./_shared.js";
export default defineStep({
  type: "graph/register-namespace",
  description:
    "Register (create) a graph NAMESPACE — a named data partition that scopes node/edge writes " +
    "(pass the same name as the `namespace` config of the graph write steps). Idempotent: registering " +
    "a namespace that already exists is a success. Names are lowercased. Namespaces are not an access-control boundary.",
  input: z.object({
    namespace: z
      .string()
      .min(1)
      .describe("Namespace name to register, e.g. a task slug. Reuse the exact same string in later graph write steps."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const r = await b.reader.registerNamespace(cfg.namespace);
      return { namespace: r.namespace, registered: true, ...(r.created ? {} : { alreadyExisted: true }) };
    } catch (e) {
      return errText("graph/register-namespace", e);
    }
  },
});
