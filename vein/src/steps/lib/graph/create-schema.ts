import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText } from "./_shared.js";

const EXAMPLE = `- id: evidence_type
  type: graph/create-schema
  config:
    type: Evidence
    parent: Thing
    attributes:
      description: string
      content: "?string"
      evidence_status: "?string"
    node_key: description
    title_key: description
    description_key: content`;

export default defineStep({
  type: "graph/create-schema",
  description:
    "Register a NODE TYPE in the vein knowledge graph's ontology (a `:Schema` node), or add attributes to an existing one. " +
    "graph/create-node and graph/create-triplet refuse a node whose type has no schema, or whose properties the schema " +
    "does not declare — call this first when the type (or attribute) you need is missing from graph_get_ontology. " +
    "`attributes` maps attribute name → type: string | boolean | int | float | complex | datetime | list, with a `?` prefix " +
    "for OPTIONAL (e.g. '?float'); no prefix = REQUIRED on every node. `name` (string) is inherited from Thing and always " +
    "available. `node_key` is the identity: `-`-joined attribute names (default 'name') — two nodes with the same node_key " +
    "values in a namespace are the same node. `index` (default: the node_key attributes) lists the searchable attributes. " +
    "If the type ALREADY EXISTS this is add-only: attributes it lacks are added, nothing existing changes, and " +
    "parent/node_key/index are left alone (status 'Warning' with `added_attributes`). Vein's own types (VeinRun, " +
    "VeinWorkflow, …) are a closed registry and cannot be created or extended. Edge types between node types are NOT " +
    "schemas — create them with graph/create-triplet's create_schema_if_missing.\n\n" +
    EXAMPLE,
  input: z.object({
    type: z.string().describe("The node type label to register, e.g. 'Evidence'. Letters/digits/underscore, starts with a letter."),
    parent: z.string().optional().default("Thing").describe("Parent type in the ontology (must exist). Attributes are inherited from it. Default 'Thing'."),
    attributes: z
      .record(z.string(), z.string())
      .describe("Attribute name → type string ('string', '?float', 'list', …). '?' prefix = optional. May be {} to only rely on inherited attributes."),
    node_key: z
      .string()
      .optional()
      .describe("Identity key: '-'-joined attribute names, e.g. 'name' or 'claim_text-speaker_name'. Every token must be a declared (or inherited) attribute. Default 'name'."),
    index: z.array(z.string()).optional().describe("Searchable attributes (fulltext). Default: the node_key attributes."),
    title_key: z.string().optional().describe("Attribute shown as a node's title in search results (default: name)."),
    description_key: z.string().optional().describe("Attribute shown as a node's description in search results."),
    domain: z.string().optional().describe("Domain the type belongs to (a `Domain_<domain>` label on its nodes). Default 'entity'."),
    type_description: z.string().optional().describe("Human/LLM-facing description of what the type represents."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const { createNodeSchema } = await import("../../../graph/schema-crud.js");
      const r = await createNodeSchema(b.bolt, b.schemas, {
        type: cfg.type,
        parent: cfg.parent,
        attributes: cfg.attributes ?? {},
        node_key: cfg.node_key,
        index: cfg.index,
        title_key: cfg.title_key,
        description_key: cfg.description_key,
        domain: cfg.domain,
        type_description: cfg.type_description,
      });
      // No provenance marker: a Schema node is ontology, not data (not a
      // Data_Bank node), so there is nothing for an ACCESSED edge to land on.
      return {
        status: r.created ? "Success" : "Warning",
        created: r.created,
        type: r.type,
        ref_id: r.ref_id,
        parent: r.parent,
        node_key: r.node_key,
        ...(r.created ? {} : { added_attributes: r.added, messages: [r.added.length ? `Schema already existed — added ${r.added.join(", ")}` : "Schema already existed — nothing to add"] }),
        attributes: r.attributes,
      };
    } catch (e) {
      return errText("graph/create-schema", e);
    }
  },
});
