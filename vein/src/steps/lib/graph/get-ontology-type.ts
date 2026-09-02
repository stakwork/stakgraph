import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText } from "./_shared.js";
export default defineStep({
  type: "graph/get-ontology-type",
  description:
    "Fetch the attribute schema for a SINGLE ontology node type. Returns exactly " +
    "one field — `attributes` — and nothing else; for a type's domain, parent or " +
    "description, use graph_get_ontology. " +
    "Each attribute value is a type string (e.g. 'string', 'int'); a `?` prefix " +
    "(e.g. '?string') means the attribute is OPTIONAL, no prefix means REQUIRED. " +
    "`attributes` is complete: it already includes everything inherited from parent types. " +
    "Lookup is case-insensitive. NODE types only — edge type names (e.g. 'IN_RUN') are not " +
    "schema nodes. Call graph_get_ontology first if you don't already know the exact type name.",
  input: z.object({
    type: z.string().describe("The node type name, e.g. 'VeinRun' (case-insensitive)."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const schema = await b.reader.getSchema(cfg.type);
      if (!schema) return `graph/get-ontology-type: unknown type ${cfg.type}`;
      return { attributes: schema.attributes };
    } catch (e) {
      return errText("graph/get-ontology-type", e);
    }
  },
});
