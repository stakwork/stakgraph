import { z, defineStep } from "vein";

/**
 * Boolean GATE for the ingest-doc workflow: does this Document node still
 * need ingestion? Returns a BARE boolean so downstream steps can use it
 * directly as a `when:` gate (the runner treats a boolean dep output as the
 * gate value — no separate `if` step needed).
 *
 * The dedupe contract (deliberately NOT the stakwork one): a Document node's
 * existence only proves the create step ran — an ingestion agent that died
 * mid-run leaves the node behind with no entities. So the skip keys on a
 * COMPLETION MARKER instead: `status === "ingested"`, written by
 * graph/edit-node only after the ingestion agent finishes successfully.
 * Reruns re-ingest partially-failed documents instead of silently skipping
 * them (create-or-merge on source_link makes the re-run safe).
 *
 * Defensive on purpose: graph read steps return an error STRING on
 * failure, and the template language cannot probe optional properties without
 * throwing — so the tolerant lookup lives here in TS.
 */
export default defineStep({
  type: "harvey/ingest-state",
  description:
    "Gate: true when a Document node still NEEDS ingestion (no `status: ingested` completion marker), " +
    "false when a prior run completed it. Pass graph/graph-get's output as `node`; tolerant of error " +
    "strings and missing properties (unknown state = needs ingestion).",
  input: z.object({
    node: z.any().describe("graph/graph-get output for the Document node (any shape tolerated)."),
  }),
  output: z.boolean(),
  async run(cfg) {
    const n = cfg.node;
    if (!n || typeof n !== "object") return true;
    const props = (n as Record<string, any>).properties;
    const status = props && typeof props === "object" ? props.status : undefined;
    return status !== "ingested";
  },
});
