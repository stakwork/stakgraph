/**
 * LIVE preflight for the jarvis/* steps — the "can the lab actually talk to
 * jarvis" check to run before kicking off a real pipeline run. Unlike
 * smoke.ts (offline, fake http) this exercises the SAME step code the
 * workflows run against a REAL jarvis over the same env resolution
 * (JARVIS_URL + API_TOKEN via the secrets fallback). Writes go to a
 * throwaway `preflight-check` namespace.
 *
 * Run: JARVIS_URL=http://localhost:5001 API_TOKEN=<STAKWORK_SECRET> \
 *        npx tsx src/lab/jarvis/preflight.ts
 */
import { httpCapability, type StepContext } from "vein";
import registerNamespace from "./steps/register-namespace.js";
import getOntology from "./steps/get-ontology.js";
import createNode from "./steps/create-node.js";
import graphGet from "./steps/graph-get.js";
import editNode from "./steps/edit-node.js";
import graphSearch from "./steps/graph-search.js";
import createTriplet from "./steps/create-triplet.js";

const ctx = {
  runId: "preflight",
  path: "preflight",
  scope: {},
  input: undefined,
  emit: async () => {},
  services: {
    http: httpCapability(),
    secrets: { get: async (n: string) => process.env[n] },
  },
} as unknown as StepContext;

const NS = "preflight-check";
type StepDef = { input: { parse(v: unknown): unknown }; run(cfg: unknown, ctx: StepContext): Promise<unknown> };
const run = async (def: unknown, input: unknown): Promise<any> =>
  (def as StepDef).run((def as StepDef).input.parse(input), ctx);
const fail = (label: string, out: unknown): never => {
  console.error(`✗ ${label}:`, typeof out === "string" ? out : JSON.stringify(out).slice(0, 400));
  process.exit(1);
};

async function main() {
  // 1. register namespace
  let out = await run(registerNamespace, { namespace: NS });
  if (typeof out === "string" || !out.registered) fail("register-namespace", out);
  console.log(`✔ register-namespace (${NS}${out.alreadyExisted ? ", existed" : ""})`);

  // 2. ontology has the types the deliver pipeline writes
  out = await run(getOntology, {});
  const text = JSON.stringify(out);
  for (const t of [
    "EvalSet", "EvalRequirement", "EvalTrigger", "EvalTriggerOutput", "CriterionResult",
    "Document", "ComputedFigure", "FormulaComponent", "ScratchpadEntry",
  ]) {
    if (!text.includes(t)) fail(`ontology missing type ${t}`, "(see get-ontology output)");
  }
  console.log("✔ get-ontology (eval chain + legal extraction types present)");

  // 3. create a Document node in the namespace
  out = await run(createNode, {
    node_type: "Document",
    namespace: NS,
    node_data: { source_link: "/tmp/preflight/doc.txt", title: "preflight-doc" },
  });
  if (typeof out === "string" || !out.ref_id) fail("create-node", out);
  const refId = out.ref_id as string;
  console.log(`✔ create-node (Document ${refId})`);

  // 4. read it back
  out = await run(graphGet, { ref_id: refId, namespace: NS });
  if (typeof out === "string" || out.ref_id !== refId) fail("graph-get", out);
  console.log("✔ graph-get");

  // 5. completion marker write (the ingest dedupe path)
  out = await run(editNode, { ref_id: refId, namespace: NS, node_data: { status: "ingested" } });
  if (typeof out === "string" && out.includes("failed")) fail("edit-node", out);
  out = await run(graphGet, { ref_id: refId, namespace: NS });
  if (out?.properties?.status !== "ingested") fail("edit-node readback (status)", out);
  console.log("✔ edit-node (status=ingested marker round-trips)");

  // 6. search finds it
  out = await run(graphSearch, { q: "preflight-doc", namespace: NS });
  if (typeof out === "string") fail("graph-search", out);
  console.log(`✔ graph-search (${Array.isArray(out) ? out.length : "?"} hit(s))`);

  // 7. eval-chain triplet (EvalSet -HAS_REQUIREMENT-> EvalRequirement)
  out = await run(createTriplet, {
    source_type: "EvalSet",
    source_data: { id: "preflight-check" },
    target_type: "EvalRequirement",
    target_data: { id: "preflight-check-c1", name: "test criterion", contested: false },
    edge_type: "HAS_REQUIREMENT",
    namespace: NS,
  });
  if (typeof out === "string" || !out.edge_ref_id) fail("create-triplet", out);
  console.log(`✔ create-triplet (HAS_REQUIREMENT edge ${out.edge_ref_id})`);

  console.log("\nPREFLIGHT PASSED — the lab's jarvis steps talk to this jarvis end-to-end.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
