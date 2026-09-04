import { z, defineStep } from "vein";

/**
 * 58313's wfbench_check_input_keys.py, for vein: a vein workflow declares
 * no input schema in YAML — its input contract is the set of `input.<key>`
 * references inside its {{ }} templates. Compare those against the task's
 * workflow_input keys:
 *   missing = referenced by the workflow but absent from the payload → the
 *             run would resolve undefined → HARNESS ERROR, do not launch.
 *   unused  = payload keys the workflow never reads → warning only.
 * An empty body (nothing published) is also a harness error.
 */
const TEMPLATE_RE = /\{\{([\s\S]*?)\}\}/g;
const DOT_RE = /\binput\.([A-Za-z_$][\w$]*)/g;
const BRACKET_RE = /\binput\[\s*["']([^"']+)["']\s*\]/g;

export function referencedInputKeys(yaml: string): string[] {
  const keys = new Set<string>();
  for (const m of yaml.matchAll(TEMPLATE_RE)) {
    const body = m[1] ?? "";
    for (const k of body.matchAll(DOT_RE)) keys.add(k[1]!);
    for (const k of body.matchAll(BRACKET_RE)) keys.add(k[1]!);
  }
  return [...keys].sort();
}

export default defineStep({
  type: "wfbench/check-input-keys",
  description:
    "Gate before launching the produced workflow: the `input.<key>` references in its YAML templates vs the task's workflow_input keys. keys_match=false (harness error, no launch) when the body is empty or it references keys the payload lacks. Output: { keys_match, referenced_keys, payload_keys, missing, unused, launch_payload, error_type, error }.",
  input: z.object({
    workflow_yaml: z.string().describe("The produced workflow's YAML ('' when nothing was published)."),
    workflow_input: z.record(z.string(), z.any()).describe("The launch payload (normalize-task's workflow_input)."),
  }),
  output: z.any(),
  async run(cfg) {
    const payload_keys = Object.keys(cfg.workflow_input).sort();
    if (!cfg.workflow_yaml.trim()) {
      return {
        keys_match: false,
        referenced_keys: [],
        payload_keys,
        missing: [],
        unused: payload_keys,
        launch_payload: cfg.workflow_input,
        error_type: "no_workflow_produced",
        error: "the author published no workflow body",
      };
    }
    const referenced_keys = referencedInputKeys(cfg.workflow_yaml);
    const missing = referenced_keys.filter((k) => !payload_keys.includes(k));
    const unused = payload_keys.filter((k) => !referenced_keys.includes(k));
    const keys_match = missing.length === 0;
    return {
      keys_match,
      referenced_keys,
      payload_keys,
      missing,
      unused,
      launch_payload: cfg.workflow_input,
      error_type: keys_match ? null : "input_keys_mismatch",
      error: keys_match
        ? null
        : `produced workflow reads input key(s) the task payload lacks: ${missing.join(", ")} (payload keys: ${payload_keys.join(", ") || "none"})`,
    };
  },
});
