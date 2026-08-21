import { tool, Tool, ModelMessage } from "ai";
import { z } from "zod";
import { randomUUID } from "crypto";
import { writeFileSync, unlinkSync } from "node:fs";
import os from "node:os";
import {
  getRepoMap,
  getFileSummary,
  runStakgraph,
  fulltextSearch,
  executeBashCommand,
} from "./bash.js";
import { textEdit, TextEditInput } from "./textEdit.js";
import { runDocx, runXlsx } from "./docgen.js";
import { AGENT_ARTIFACTS_DIR } from "./artifacts.js";
import { getProviderTool, Provider, ModelName, getGatewayBaseURL } from "../aieo/src/index.js";
import { log_agent_context } from "../log/agent.js";
import { createRunLogsDir, cleanupRunLogsDir } from "../log/utils.js";
import { RepoAnalyzer } from "gitsee/server";
import {
  listConcepts,
  getConceptDocumentation,
  createProposal,
  listProposals,
  HttpError,
} from "../gitree/service.js";
import { GraphStorage as GitreeGraphStorage } from "../gitree/store/index.js";
import { scanSkills, listPack, loadSkill, type IndexEntry } from "./skills.js";
import { db } from "../graph/neo4j.js";
import { callRemoteAgent, subAgentRepoNames, type SubAgent } from "./subagent.js";
import { registerJarvisTools } from "./toolsJarvis.js";
import { registerStakworkTools, type StakworkToolsOptions } from "./toolsStakwork.js";
import {
  registerGoogleSheetsTools,
  GOOGLE_SHEETS_TOOL_NAMES,
  type GoogleSheetsToolsOptions,
  type GoogleSheetsToolOverrides,
} from "./toolsGoogleSheets.js";
import * as stak from "../tools/stakgraph/index.js";
import { search as graphSearch, searchWithProvenance } from "../graph/graph.js";
import type { SearchProvenance } from "../graph/graph.js";
import { relevant_node_types } from "../graph/types.js";
import {
  landChange,
  type LandChangeResult,
  type WorktreeHandle,
  type AgentIdentity,
  type GitHubClient,
} from "./git_pr.js";

/**
 * Allowed write roots for the text-editor tool. The tool sandbox refuses any
 * path outside these (see resolveInCwd in textEdit.ts): the cloned repo, the
 * OS temp dir (scratch), and — when configured — the durable artifacts dir.
 *
 * When `strict` is true (create_pr path), the OS temp dir is omitted so
 * str_replace_based_edit_tool / apply_patch cannot write into the shared
 * /tmp/<owner>/<repo> checkout when a worktree is active.
 */
export function editorRoots(repoPath: string, strict?: boolean): string[] {
  if (strict) {
    const roots = [repoPath];
    if (AGENT_ARTIFACTS_DIR) roots.push(AGENT_ARTIFACTS_DIR);
    return roots;
  }
  const roots = [repoPath, os.tmpdir()];
  if (AGENT_ARTIFACTS_DIR) roots.push(AGENT_ARTIFACTS_DIR);
  return roots;
}

export interface ProvenanceEntry {
  tool_call_id?: string;
  tool_name: string;
  timestamp: string;
  provenance: SearchProvenance;
}

export interface ProvenanceCollector {
  entries: ProvenanceEntry[];
}

export interface PrCollector {
  result: LandChangeResult | undefined;
}

export interface GgnnTool {
  name: string;          // tool name, e.g. "ggnn_check"
  endpoint: string;      // e.g. "/check", "/predict", "/score-plan"
  description: string;   // tool description for the LLM
  bodyType: "predict" | "check" | "score-plan";
}

export interface GgnnConfig {
  url: string;           // base URL, e.g. "https://ggnn.sphinx.chat"
  apiKey: string;        // bearer token
  languages?: string[];  // e.g. ["typescript"]
  tools: GgnnTool[];
}

export interface MessagesRef {
  current: ModelMessage[];
}

export interface GetToolsOptions {
  /** create_pr path: collector that receives the landChange result */
  prCollector?: PrCollector;
  /** create_pr path: worktree + identity handle for landChange */
  prMode?: {
    handle: WorktreeHandle;
    identity: AgentIdentity;
    env: NodeJS.ProcessEnv;
    githubClient: GitHubClient;
  };
  /** create_pr path: base checkout path (shared clone) to reject in bash commands */
  baseCheckoutPath?: string;
  /**
   * Ephemeral (throwaway-worktree) run: arms the same bash/editor confinement
   * as prMode — no git write commands, no ambient GitHub tokens, strict editor
   * roots — without a create_pr tool. Used by read-only preview runs.
   */
  ephemeral?: boolean;
}

/** Whether this run is confined to a per-run worktree (PR run or ephemeral preview). */
function confinedRun(options?: GetToolsOptions): boolean {
  return Boolean(options?.prMode || options?.ephemeral);
}

/**
 * Bash confinement for worktree-confined runs. Returns a rejection message,
 * or undefined when the command is allowed.
 *
 * A guard, not a sandbox — the backstops are the withheld GitHub tokens and,
 * on the create_pr path, base-dirty detection. `git push` must be blocked even
 * with tokens withheld because the shared clone's origin URL embeds the PAT
 * (clone.ts inlines credentials), and a worktree shares the base repo's
 * remote config.
 */
export function confinedBashRejection(
  command: string,
  options?: GetToolsOptions,
): string | undefined {
  if (options?.baseCheckoutPath && command.includes(options.baseCheckoutPath)) {
    return (
      `bash command rejected: references the shared checkout '${options.baseCheckoutPath}'. ` +
      `Your current working directory is an isolated worktree of the same repo — rerun the command there using relative paths.`
    );
  }
  if (!confinedRun(options)) return undefined;
  // [regex, label]: the char class stops at command separators so a match
  // never spans two piped/chained commands ("git log | grep push" is fine —
  // each chained command gets its own `\bgit\b … \bpush\b` evaluation).
  const blocked: Array<[RegExp, string]> = [
    [/\bgit\b[^\n|;&]*\bpush\b/, "git push"],
    [/\bgit\b[^\n|;&]*\bcommit\b/, "git commit"],
    [/\bgit\b[^\n|;&]*\bremote\b/, "git remote"],
    [/\bgh\s+(pr|api|repo|release)\b/, "gh write commands"],
  ];
  for (const [re, label] of blocked) {
    if (re.test(command)) {
      const hint = options?.prMode
        ? "Commit/push/PR creation happen exclusively through the create_pr tool."
        : "This is a read-only preview run — no remote writes.";
      return `bash command rejected: ${label} is not allowed on this run. ${hint}`;
    }
  }
  return undefined;
}

type ToolName =
  | "repo_overview"
  | "file_summary"
  | "recent_commits"
  | "recent_contributions"
  | "fulltext_search"
  | "web_search"
  | "bash"
  | "final_answer"
  | "ask_clarifying_questions"
  | "list_concepts"
  | "learn_concept"
  | "learn_concepts"
  | "propose_concept_change"
  | "list_concept_proposals"
  | "list_skills"
  | "load_skill"
  | "list_workflows"
  | "learn_workflow"
  | "read_workflow_json"
  | "vector_search"
  | "stakgraph_search"
  | "stakgraph_map"
  | "stakgraph_code"
  | "jarvis"
  | "graph_sub_agent"
  | "ontology_edit"
  | "create_triplet"
  | "create_node"
  | "edit_node"
  | "logs_agent"
  | "str_replace_based_edit_tool"
  | "apply_patch"
  | "generate_docx"
  | "generate_xlsx"
  | "generate_xlsx_computed"
  | "stakwork_run_step"
  | "sheets_create_spreadsheet"
  | "sheets_update_values"
  | "sheets_batch_update_values"
  | "sheets_get_values"
  | "sheets_add_sheet"
  | "sheets_import_spreadsheet"
  | "create_pr";

/**
 * Object form of a per-tool config value. Lets a caller pass a description
 * override AND tool-specific options at once (instead of just a bare string).
 * Unknown/extra fields are ignored by tools that don't read them.
 */
export interface ToolConfigObject {
  /** Explicitly enable/disable the tool. Defaults to true when the object is present. */
  enabled?: boolean;
  /** Override the tool's description shown to the LLM. */
  description?: string;
  /** graph_sub_agent: max recursion depth for nested sub-agents. */
  maxDepth?: number;
  /** graph_sub_agent: max tool-loop steps per sub-agent run. */
  maxSteps?: number;
}

/**
 * A per-tool config value. Backwards compatible:
 *   - `string`  → enable + use as description override
 *   - `boolean` → enable/disable
 *   - `null`    → keep default
 *   - object    → enable (unless `enabled: false`) + carry description/options
 */
export type ToolConfigValue = string | boolean | null | ToolConfigObject;

/**
 * Keys a caller may nest inside `toolsConfig` that configure a tool *family*
 * rather than one tool. These are not tool names — they never reach the tool
 * selection loop — and they may carry secrets, so anything added here must be
 * handled by `redactToolsConfig` before the config is persisted or served.
 */
export interface ToolsConfigExtras {
  /** Google Sheets credentials, as an alternative to the top-level `googleSheets` body field. */
  google_sheets?: GoogleSheetsToolsOptions;
  /** camelCase alias of `google_sheets`. */
  googleSheets?: GoogleSheetsToolsOptions;
}

export type ToolsConfig = Partial<Record<ToolName, ToolConfigValue>> & ToolsConfigExtras;

/** Keys of ToolsConfig that are family config, not per-tool config. */
const TOOLS_CONFIG_EXTRA_KEYS: (keyof ToolsConfigExtras)[] = ["google_sheets", "googleSheets"];

/**
 * Google Sheets credentials for this run. The top-level `googleSheets` option
 * wins; `toolsConfig.google_sheets` / `toolsConfig.googleSheets` is a fallback
 * for callers that can only send a single config blob.
 */
export function resolveGoogleSheetsOptions(
  googleSheets?: GoogleSheetsToolsOptions,
  toolsConfig?: ToolsConfig
): GoogleSheetsToolsOptions | undefined {
  if (googleSheets?.serviceAccount) return googleSheets;
  for (const key of TOOLS_CONFIG_EXTRA_KEYS) {
    const v = toolsConfig?.[key];
    if (isToolConfigObject(v) && (v as GoogleSheetsToolsOptions).serviceAccount) {
      const { serviceAccount, driveFolderId } = v as GoogleSheetsToolsOptions;
      return {
        serviceAccount,
        driveFolderId: typeof driveFolderId === "string" && driveFolderId ? driveFolderId : undefined,
      };
    }
  }
  return undefined;
}

/**
 * Strip secrets from a toolsConfig before it is written to the session config
 * sidecar (which GET /repo/agent/session serves back to callers). Non-secret
 * fields are kept so the persisted config still describes the run.
 */
export function redactToolsConfig(toolsConfig?: ToolsConfig): ToolsConfig | undefined {
  if (!toolsConfig) return toolsConfig;
  let redacted: ToolsConfig | undefined;
  for (const [key, value] of Object.entries(toolsConfig)) {
    if (!isToolConfigObject(value) || !("serviceAccount" in value)) continue;
    redacted = redacted ?? { ...toolsConfig };
    const { serviceAccount, ...rest } = value as Record<string, unknown>;
    (redacted as any)[key] = rest;
  }
  return redacted ?? toolsConfig;
}

/** Narrow a config value to its object form. */
export function isToolConfigObject(v: unknown): v is ToolConfigObject {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * Whether the graph DATA write tool family (create_triplet,
 * create_batch_triplet, create_node, edit_node) is enabled. Any one of the
 * family's toolsConfig keys turns on the whole family — they share one
 * registration gate.
 */
export function graphWriteEnabled(toolsConfig?: ToolsConfig): boolean {
  return (
    toolConfigEnabled(toolsConfig?.create_triplet) ||
    toolConfigEnabled(toolsConfig?.create_node) ||
    toolConfigEnabled(toolsConfig?.edit_node)
  );
}

/** Whether a tool-config value should count as "on" (for opt-in tools). */
export function toolConfigEnabled(v: ToolConfigValue | undefined): boolean {
  if (v === undefined || v === null) return false;
  if (typeof v === "boolean") return v;
  if (typeof v === "string") return v.length > 0;
  return v.enabled !== false; // object present ⇒ enabled unless explicitly false
}

/** Description override carried by a tool-config value, if any. */
export function toolConfigDescription(v: ToolConfigValue | undefined): string | undefined {
  if (typeof v === "string" && v.length > 0) return v;
  if (isToolConfigObject(v) && typeof v.description === "string" && v.description.length > 0) {
    return v.description;
  }
  return undefined;
}

const TOOL_NAMES: Set<string> = new Set<string>([
  "repo_overview", "file_summary", "recent_commits", "recent_contributions",
  "fulltext_search", "web_search", "bash", "final_answer",
  "ask_clarifying_questions", "list_concepts", "learn_concept",
  "learn_concepts", "propose_concept_change", "list_concept_proposals",
  "list_skills", "load_skill",
  "list_workflows", "learn_workflow", "read_workflow_json",
  "vector_search", "stakgraph_search", "stakgraph_map", "stakgraph_code",
  "graph_sub_agent", "ontology_edit", "create_triplet", "create_node", "edit_node",
  "str_replace_based_edit_tool", "apply_patch",
  "generate_docx", "generate_xlsx", "generate_xlsx_computed",
  "stakwork_run_step",
  "sheets_create_spreadsheet", "sheets_update_values", "sheets_batch_update_values",
  "sheets_get_values", "sheets_add_sheet", "sheets_import_spreadsheet",
  "create_pr",
]);

export type SkillsConfig = Partial<Record<string, boolean>>;

const DEFAULT_DESCRIPTIONS: Record<ToolName, string> = {
  repo_overview:
    "Get a high-level view of the codebase architecture and structure. Use this to understand the project layout and identify where specific functionality might be located. Call this when you need to: 1) Orient yourself in an unfamiliar codebase, 2) Locate which directories/files might contain relevant code for a user's question, 3) Understand the overall project structure before diving deeper. Don't call this if you already know which specific files you need to examine.",
  file_summary:
    "Get a summary of what a specific file contains and its role in the codebase. Use this when you have identified a potentially relevant file and need to understand: 1) What functions/components it exports, 2) What its main responsibility is, 3) Whether it's worth exploring further for the user's question. Call this with a hypothesis like 'This file probably handles user authentication' or 'This looks like the main dashboard component'. Don't call this to browse random files.",
  recent_commits:
    "Query a repo for recent commits. The output is a list of recent commits.",
  recent_contributions:
    "Query a repo for recent PRs by a specific contributor. Input is the contributor's GitHub login. The output is a list of their most recent contributions, including PR titles, issue titles, commit messages, and code review comments.",
  fulltext_search:
    "Search the entire codebase for a specific term, using ripgrep (rg). Use this when you need to find a specific function, component, or file. Call this when the user provided specific text that might be present in the codebase. For example, if the query is 'Add a subtitle to the User Journeys page', you could call this with the query \"User Journeys\". Don't call this if you do not have specific text to search for",
  web_search: "Search the web for information",
  bash:
    "Execute a bash command in the working directory. " +
    "Keep the whole command under ~40KB (roughly 6,000 words of text). A single command over " +
    "~128KB is rejected by the OS with `spawn E2BIG` before it runs — nothing executes and " +
    "everything you generated for it is lost. When writing a large file, split it across " +
    "several commands: `cat > file` for the first chunk, `cat >> file` for each one after.",
  final_answer: `Provide the final answer to the user. YOU CAN CALL THIS TOOL AT THE END OF YOUR EXPLORATION.
CRITICAL: Put your ENTIRE response inside the 'answer' parameter as a well-formatted string. Do NOT call this tool with an empty object or without the answer field.

Example usage:
final_answer({ 
  "answer": "Based on my exploration of the codebase, here's how authentication works:\n\n1. Users authenticate via...\n2. The auth flow is handled by...\n3. Key files include..."
})`,
  ask_clarifying_questions: `Ask clarifying questions to the user when you need clarification about design or implementation choices. Call this at the end of your exploration.

QUESTION TYPES:
- single_choice: User picks one option
- multiple_choice: User picks one or more options

BASIC QUESTION EXAMPLE: (simple string options):
[{
  "question": "What type of app?",
  "type": "single_choice",
  "options": ["Web", "Mobile", "Desktop"]
}]

COLOR PICKER EXAMPLE: (use for brand colors, themes, UI colors):
[{
  "question": "Which primary color for your brand?",
  "type": "single_choice",
  "options": ["Sky Blue", "Purple", "Emerald"],
  "questionArtifact": {
    "type": "color_swatch",
    "data": [
      {"label": "Sky Blue", "value": "#0EA5E9"},
      {"label": "Purple", "value": "#8B5CF6"},
      {"label": "Emerald", "value": "#10B981"}
    ]
  }
}]

DIAGRAM QUESTION EXAMPLE: (use to confirm flows, architecture, data models):
[{
  "question": "Does this authentication flow look correct?",
  "type": "single_choice",
  "options": ["Yes, proceed", "No, needs changes"],
  "questionArtifact": {"type": "mermaid", "data": "graph TD\\n  A[Login]-->B{Valid?}\\n  B-->|Yes|C[Dashboard]\\n  B-->|No|D[Error]"}
}]

COMPARISON TABLE EXAMPLE: (use when comparing multiple approaches/technologies):
[{
  "question": "Which real-time approach should we use?",
  "type": "single_choice",
  "options": ["SSE", "WebSockets", "Polling"],
  "questionArtifact": {
    "type": "comparison_table",
    "data": {
      "columns": ["Pros", "Cons"],
      "rows": [
        {"label": "SSE", "description": "Server-Sent Events", "cells": {"Pros": ["Simple", "Auto-reconnect"], "Cons": ["Server→Client only"]}},
        {"label": "WebSockets", "description": "Full duplex", "cells": {"Pros": ["Bi-directional", "Low latency"], "Cons": ["Complex"]}},
        {"label": "Polling", "description": "HTTP requests", "cells": {"Pros": ["Works everywhere"], "Cons": ["High latency"]}}
      ]
    }
  }
}]

Rules:
- Maximum 3 questions
- Use mermaid questionArtifact to visualize and confirm flows before implementing
- Use comparison_table when comparing multiple approaches with pros/cons
- Use color_swatch when asking about colors/themes
- Can combine: use questionArtifact to show a diagram AND rich options for choices`,
  list_concepts:
    "List all high-level concepts (features) that have been learned about this codebase. Use this to discover what areas of functionality have been documented and understand the main components of the system. Returns a list of concepts with their ids, names, and descriptions. Use learn_concept to get full details on a specific concept.",
  learn_concept:
    "Get detailed information about a specific concept (feature) including its full documentation, associated PRs with summaries, and commits. Use this when you need deep understanding of how a particular feature was implemented and evolved over time.",
  learn_concepts: '', // this is just for naming, to enable the above 2.
  propose_concept_change:
    "Propose a change to the Concept knowledge base: create a new concept, update or delete an existing one, or merge two duplicate concepts. The change is NOT applied — it goes into a review queue where a human accepts or rejects it, with a diff of the documentation. For update/merge, first read the current docs with learn_concept and submit the FULL revised documentation (it replaces the whole body). Always give a rationale — the reviewer sees it. Check list_concept_proposals first so you don't file a duplicate of a proposal that is already pending.",
  list_concept_proposals:
    "List proposed changes to the Concept knowledge base (pending review by default). Use this before propose_concept_change to avoid filing a duplicate of a proposal that is already awaiting review.",
  list_skills:
    "List available skills. Called with no argument, returns every skill and skill pack installed — including packs not listed in your system prompt. Pass `pack` to expand one pack into its individual skills with their descriptions. This returns names and descriptions only; use load_skill to read a skill's actual instructions.",
  load_skill:
    "Load a skill's full instructions into context. Call this BEFORE doing work the skill covers — a skill's description says what area it covers, not how to do it, so acting on the description alone means working without the guidance. Pass the name exactly as listed in your system prompt or by list_skills; skills belonging to a pack are addressed as \"pack/skill\" (e.g. \"commercial-legal/nda-review\"). Loading a pack skill also pulls in that pack's shared context the first time.",
  list_workflows: 'List all Workflow nodes in the graph, returning workflow_name and node_key for each.',
  learn_workflow: 'Get the generated documentation for a specific workflow by its node_key.',
  read_workflow_json: 'Read the raw workflow_json property of a Workflow node by its node_key.',
  vector_search: 'Search for code nodes by semantic similarity using vector embeddings. Use this when you want to find code related to a concept or description, even if the exact terms are not present in the code. Returns the node name, type, filename, and line number for each match.',
  stakgraph_search:
    "Search the code graph by keyword, semantic meaning, or hybrid. Returns compact results with name, file, ref_id, and description. Use stakgraph_code with a ref_id to read full source.",
  stakgraph_map: "Trace relationships from a node in the code graph. Use direction 'up' for callers and 'down' for callees.",
  stakgraph_code:
    "Retrieve actual source code for a specific node. Use ref_id from search results, or name+node_type to identify the node. Defaults to depth 1 (just the node itself).",
  jarvis: '', // deprecated: Jarvis tools now auto-register whenever JARVIS_URL is set.
  graph_sub_agent: '', // default lives in toolsJarvis.ts; string value here overrides it.
  ontology_edit: '', // group gate: registers the ontology write tools (defaults live in toolsJarvis.ts).
  create_triplet: '', // gate: registers the graph data-write tools (defaults live in toolsJarvis.ts). Any one of create_triplet/create_node/edit_node enables the whole family.
  create_node: '', // gate: same graph data-write family as create_triplet.
  edit_node: '', // gate: same graph data-write family as create_triplet.
  logs_agent:
    "Query runtime logs (CloudWatch / Quickwit). Use when the user asks about errors, performance, or runtime behaviour. Pass a focused, specific question.",
  str_replace_based_edit_tool:
    "View and edit files inside the cloned repo (sandboxed to working dir). " +
    "Commands: view (path, optional view_range [start,end]), create (path, file_text), " +
    "str_replace (path, old_str — must match exactly once, new_str), " +
    "insert (path, insert_line — 0 = top, insert_text).",
  apply_patch:
    "Apply a unified-diff patch string to the cloned repo via `git apply`. " +
    "Patch must be valid unified-diff format. Returns success message or error details.",
  generate_docx:
    "Generate a Word (.docx) document from Markdown using Pandoc. " +
    "Input: { markdown?: string; markdownPath?: string; template?: string } — provide exactly one " +
    "of 'markdown' (inline content) or 'markdownPath' (path to a .md file; relative paths resolve " +
    "against the repo directory). " +
    "For anything longer than a couple of pages, ALWAYS use markdownPath: a large inline 'markdown' " +
    "argument has to fit in one message and will be cut off. " +
    "Build the .md file with a series of bash writes — `cat > file` for the first chunk, `cat >> file` " +
    "for each one after — keeping every command under the ~40KB bash limit, then convert it. That " +
    "applies to the FIRST write exactly as much as to the appends; do not emit a whole document in " +
    "one call. The byte count `wc -c` returns is the file's cumulative size, not the size of your " +
    "last command — a growing file is fine, an oversized single command is not. " +
    "'template' is an optional bundled reference-doc name for styling. Writes the file to the durable " +
    "artifacts directory and returns a download path: 'Generated: /repo/agent/file?path=...' " +
    "On failure returns a non-fatal 'generate_docx failed: ...' string.",
  generate_xlsx:
    "Generate an Excel (.xlsx) workbook from a structured definition using openpyxl. " +
    "Input: { filename?: string; sheets: Array<{ name: string; rows?: (string|number)[][]; " +
    "cells?: Array<{ ref: string; value?: string|number; formula?: string }> }> }. " +
    "Supports multiple sheets, formulas, and cross-sheet references (e.g. =Sheet2!B2). " +
    "Writes the file to the durable artifacts directory and returns a download path: " +
    "'Generated: /repo/agent/file?path=...' On failure returns a non-fatal 'generate_xlsx failed: ...' string.",
  generate_xlsx_computed:
    "Generate an Excel (.xlsx) workbook where computed cells (sums, percentages, ratios) are " +
    "evaluated in Python and written as literal numeric values — never formula strings. " +
    "Use this instead of generate_xlsx when the Harvey judge or any value-reader must see real numbers. " +
    "Input: { filename?: string; sheets: Array<{ name: string; rows?: (string|number)[][]; " +
    "cells?: Array<{ ref: string; value?: string|number }> (no formula field — literal values only); " +
    "computed?: Array<{ ref: string; op: 'sum'|'percent_of_total'|'ratio'; " +
    "range?: string; value_ref?: string; total_ref?: string; denominator_ref?: string; " +
    "decimals?: number; as_fraction?: boolean }> }> }. " +
    "Supported ops: 'sum' (column sum B2:B9 or row sum B2:E2), " +
    "'percent_of_total' (value_ref/total_ref × 100 by default; set as_fraction:true for raw 0–1 ratio; " +
    "for per-row '% of shares' over a column, emit one computed entry per destination row), " +
    "'ratio' (value_ref/denominator_ref). " +
    "Refs may be sheet-qualified: 'Sheet2!B2'. Computed entries are evaluated sequentially so earlier " +
    "results (e.g. a column sum) can feed later entries (e.g. percent_of_total). " +
    "Returns 'Generated: /repo/agent/file?path=...' on success. " +
    "On failure returns a non-fatal 'generate_xlsx_computed failed: ...' string.",
  stakwork_run_step: '', // opt-in; default lives in toolsStakwork.ts; string value here overrides it.
  // Gated on googleSheets credentials; defaults live in toolsGoogleSheets.ts.
  sheets_create_spreadsheet: '',
  sheets_update_values: '',
  sheets_batch_update_values: '',
  sheets_get_values: '',
  sheets_add_sheet: '',
  sheets_import_spreadsheet: '',
  // Gated inside toolsConfig branch only — never added to allTools unconditionally.
  create_pr:
    "Commit all staged changes and open a pull request on GitHub. " +
    "Input: { title: string, body: string }. " +
    "Returns the PR URL, branch, diff, and files-changed count on success, " +
    "or a classified failure mode with diff on error. " +
    "Call this at the end of your work when all file edits are complete.",
};

export async function get_tools(
  repoPath: string,
  apiKey: string,
  pat: string | undefined,
  toolsConfig?: ToolsConfig,
  provider?: Provider,
  repos?: string[],
  subAgents?: SubAgent[],
  ggnn?: GgnnConfig,
  messagesRef?: MessagesRef,
  provenanceCollector?: ProvenanceCollector,
  modelName?: ModelName,
  stakwork?: StakworkToolsOptions,
  googleSheets?: GoogleSheetsToolsOptions,
  skills?: SkillsConfig,
  ontologyDomains?: string,
  sessionId?: string,
  abortSignal?: AbortSignal,
  options?: GetToolsOptions,
) {
  const repoArr = repoPath.split("/");
  const isMultiRepo = repoPath === "/tmp";
  // For single repo, extract owner/name from path. For multi-repo, these will be empty.
  const repoOwner = isMultiRepo ? "" : repoArr[repoArr.length - 2];
  const repoName = isMultiRepo ? "" : repoArr[repoArr.length - 1];
  // Default content-search scope: constrain graph search to the session's repos
  // (owner/repo form) so results don't leak across repos on a multi-repo graph.
  // Returns undefined when no repos are known (single-repo deployments / no scope).
  const repoScopePatterns = (rs?: string[]): string[] | undefined =>
    rs && rs.length > 0 ? rs.map((r) => `${r}/**`) : undefined;
  // Resolve a repo owner/name pair from an optional "owner/name" string,
  // falling back to the single-repo values or the first entry in repos.
  function resolveRepo(repo?: string): { owner: string; name: string } | null {
    if (repo) {
      const parts = repo.split("/");
      if (parts.length === 2) return { owner: parts[0], name: parts[1] };
    }
    if (!isMultiRepo) return { owner: repoOwner, name: repoName };
    if (repos && repos.length > 0) {
      const parts = repos[0].split("/");
      if (parts.length === 2) return { owner: parts[0], name: parts[1] };
    }
    return null;
  }

  const web_search_tool =
    provider === "anthropic"
      ? getProviderTool(provider, apiKey, "webSearch")
      : undefined;
  const bash_tool =
    provider === "anthropic"
      ? getProviderTool(provider, apiKey, "bash")
      : undefined;

  console.log("===> web_search_tool type:", web_search_tool?.type);
  console.log(
    "===> web_search_tool structure:",
    JSON.stringify(web_search_tool, null, 2),
  );

  const defaultDescriptions: Record<ToolName, string> = {
    ...DEFAULT_DESCRIPTIONS,
    web_search: web_search_tool?.description || DEFAULT_DESCRIPTIONS.web_search,
    bash: bash_tool?.description || DEFAULT_DESCRIPTIONS.bash,
  };

  const allTools: Record<string, Tool<any, any>> = {
    repo_overview: tool({
      description: defaultDescriptions.repo_overview,
      inputSchema: z.object({}),
      execute: async () => {
        try {
          return await getRepoMap(repoPath, repos);
        } catch (e) {
          return "Could not retrieve repository map";
        }
      },
    }),
    file_summary: tool({
      description: defaultDescriptions.file_summary,
      inputSchema: z.object({
        file_path: z.string().describe("Path to the file to summarize"),
        hypothesis: z
          .string()
          .describe(
            "What you think this file might contain or handle, based on its name/location",
          ),
      }),
      execute: async ({ file_path }: { file_path: string }) => {
        try {
          return await runStakgraph(file_path, repoPath);
        } catch (e) {
          // stakgraph CLI not available or failed, fall back to reading first lines
          try {
            return getFileSummary(file_path, repoPath, 75);
          } catch (e) {
            return "Bad file path";
          }
        }
      },
    }),
    recent_commits: tool({
      description: defaultDescriptions.recent_commits,
      inputSchema: z.object({
        limit: z.number().optional().default(10),
        repo: z
          .string()
          .optional()
          .describe(
            "Repository in 'owner/name' format. Required for multi-repo contexts.",
          ),
      }),
      execute: async ({ limit, repo }: { limit?: number; repo?: string }) => {
        const resolved = resolveRepo(repo);
        if (!resolved)
          return "Could not determine repository. Provide a repo in 'owner/name' format.";
        try {
          const analyzer = new RepoAnalyzer({
            githubToken: pat,
          });
          const coms = await analyzer.getRecentCommitsWithFiles(
            resolved.owner,
            resolved.name,
            {
              limit: limit || 10,
            },
          );
          return coms;
        } catch (e) {
          console.error("Error retrieving recent commits:", e);
          return "Could not retrieve recent commits";
        }
      },
    }),
    recent_contributions: tool({
      description: defaultDescriptions.recent_contributions,
      inputSchema: z.object({
        user: z.string(),
        limit: z.number().optional().default(5),
        repo: z
          .string()
          .optional()
          .describe(
            "Repository in 'owner/name' format. Required for multi-repo contexts.",
          ),
      }),
      execute: async ({
        user,
        limit,
        repo,
      }: {
        user: string;
        limit?: number;
        repo?: string;
      }) => {
        const resolved = resolveRepo(repo);
        if (!resolved)
          return "Could not determine repository. Provide a repo in 'owner/name' format.";
        try {
          const analyzer = new RepoAnalyzer({
            githubToken: pat,
          });
          const output = await analyzer.getContributorPRs(
            resolved.owner,
            resolved.name,
            user,
            limit || 5,
          );
          return output;
        } catch (e) {
          console.error("Error retrieving recent contributions:", e);
          return "Could not retrieve repository map";
        }
      },
    }),
    fulltext_search: tool({
      description: defaultDescriptions.fulltext_search,
      inputSchema: z.object({
        query: z.string().describe("The term to search for"),
      }),
      execute: async ({ query }: { query: string }) => {
        try {
          return await fulltextSearch(query, repoPath, repos);
        } catch (e) {
          return `Search failed: ${e}`;
        }
      },
    }),
    // final_answer: tool({
    //   description: defaultDescriptions.final_answer,
    //   inputSchema: z.object({
    //     answer: z
    //       .string()
    //       .optional()
    //       .describe("Your complete final answer to the user's question."),
    //   }),
    //   execute: async (body: { answer?: string }) => {
    //     console.log("====> final_answer", JSON.stringify(body, null, 2));
    //     return body.answer || "";
    //   },
    // }),
  };

  // Skip provider-defined tools - not compatible with AI SDK v6 when mixed with regular tools???
  if (web_search_tool) {
    allTools.web_search = web_search_tool as any as Tool<any, any>;
    // Provider tools need to be added differently
  }

  // Per-request GitHub auth for the `gh` CLI (and any tool reading GH_TOKEN).
  // Scoped to this request's PAT so `gh` acts as the requesting user.
  //
  // Confined runs (create_pr worktree or ephemeral preview) get the tokens
  // BLANKED instead: executeBashCommand spreads process.env under this
  // overlay, so an empty string is the only way to also mask any ambient
  // container token. On the create_pr path landChange() supplies its own
  // credentialed env via gitEnv() — bash never needs a push token.
  const ghEnv: NodeJS.ProcessEnv | undefined = confinedRun(options)
    ? { GH_TOKEN: "", GITHUB_TOKEN: "" }
    : pat
      ? { GH_TOKEN: pat, GITHUB_TOKEN: pat }
      : undefined;

  const executeGuardedBash = async ({ command }: { command: string }) => {
    const rejection = confinedBashRejection(command, options);
    if (rejection) {
      console.log(`[bash-guard] rejected: ${command.slice(0, 200)}`);
      return rejection;
    }
    try {
      return await executeBashCommand(command, repoPath, 60000, ghEnv);
    } catch (e) {
      return `Command execution failed: ${e}`;
    }
  };

  // Always register bash tool — Anthropic uses native provider tool, others use executeBashCommand
  if (bash_tool) {
    // Anthropic: use native provider tool (existing behaviour)
    allTools.bash = tool({
      description: bash_tool.description || defaultDescriptions.bash,
      inputSchema: z.object({
        command: z.string().describe("The bash command to execute"),
      }),
      execute: executeGuardedBash,
    });
  } else {
    // Non-Anthropic: use executeBashCommand directly
    allTools.bash = tool({
      description: defaultDescriptions.bash,
      inputSchema: z.object({
        command: z.string().describe("The bash command to execute"),
      }),
      execute: executeGuardedBash,
    });
  }

  // Skills: registered only when the caller enabled at least one, so a run that
  // never asked for skills pays nothing — not the ~200 tokens of tool schema,
  // and not the risk of the model wandering into an unrelated pack.
  //
  // Registration is all-or-nothing rather than per-entry: once a caller is in,
  // these tools reach EVERYTHING installed, including packs absent from the
  // prompt index. That's the intended discovery path — the index is the
  // shortlist, not the allowlist.
  const anySkillEnabled = Object.values(skills || {}).some(Boolean);
  const installedSkills: IndexEntry[] = anySkillEnabled ? scanSkills() : [];
  if (installedSkills.length > 0) {
    // A pack's shared preamble is large (~9k tokens) and every skill in the
    // pack assumes the same one, so it's sent once per pack per request.
    const preamblesSent = new Set<string>();

    allTools.list_skills = tool({
      description: defaultDescriptions.list_skills,
      inputSchema: z.object({
        pack: z
          .string()
          .optional()
          .describe("Expand this pack into its individual skills. Omit to list everything installed."),
      }),
      execute: async ({ pack }: { pack?: string }) => {
        try {
          if (pack) {
            const skills = listPack(pack);
            if (!skills) {
              return {
                error: `No pack named "${pack}".`,
                packs: scanSkills()
                  .filter((e) => e.kind === "pack")
                  .map((e) => e.name),
              };
            }
            return {
              pack,
              skills: skills.map((s) => ({
                name: s.qualified,
                description: s.description,
              })),
            };
          }
          return {
            entries: scanSkills().map((e) =>
              e.kind === "pack"
                ? {
                    type: "pack" as const,
                    name: e.name,
                    description: e.description,
                    skill_count: e.skills.length,
                  }
                : {
                    type: "skill" as const,
                    name: e.qualified,
                    description: e.description,
                  },
            ),
          };
        } catch (e) {
          console.error("Error listing skills:", e);
          return "Could not list skills";
        }
      },
    });

    allTools.load_skill = tool({
      description: defaultDescriptions.load_skill,
      inputSchema: z.object({
        name: z
          .string()
          .describe('Skill name, pack-qualified when it belongs to a pack (e.g. "commercial-legal/nda-review")'),
      }),
      execute: async ({ name }: { name: string }) => {
        try {
          const result = loadSkill(name);
          if ("error" in result) return result;

          const sendPreamble =
            result.preamble && result.packName && !preamblesSent.has(result.packName);
          if (sendPreamble) preamblesSent.add(result.packName!);

          return {
            name: result.name,
            ...(sendPreamble
              ? {
                  pack: result.packName,
                  pack_context: result.preamble,
                  note: `Shared context for the "${result.packName}" pack — applies to every skill in it, sent once.`,
                }
              : {}),
            skill: result.body,
          };
        } catch (e) {
          console.error("Error loading skill:", e);
          return `Could not load skill "${name}"`;
        }
      },
    });
  }

  // Conditionally register workflow tools if Workflow nodes exist
  if (db) {
    const workflowCount = await db.count_workflow_nodes();
    if (workflowCount > 0) {
      console.log("===> workflow nodes found, registering workflow tools");
      allTools.list_workflows = tool({
        description: defaultDescriptions.list_workflows,
        inputSchema: z.object({}),
        execute: async () => {
          const workflows = await db.get_all_workflows();
          return workflows.map((w) => ({
            workflow_name: w.workflow_name,
            node_key: w.node_key,
          }));
        },
      });
      allTools.learn_workflow = tool({
        description: defaultDescriptions.learn_workflow,
        inputSchema: z.object({
          node_key: z.string().describe("node_key of the Workflow"),
        }),
        execute: async ({ node_key }) => {
          const doc = await db.get_workflow_documentation(node_key);
          if (!doc)
            return { error: "No documentation found for this workflow" };
          return { body: doc.body };
        },
      });
      allTools.read_workflow_json = tool({
        description: defaultDescriptions.read_workflow_json,
        inputSchema: z.object({
          node_key: z.string().describe("node_key of the Workflow"),
        }),
        execute: async ({ node_key }) => {
          const workflow = await db.get_workflow_by_key(node_key);
          if (!workflow) return { error: "Workflow not found" };
          return { workflow_json: workflow.workflow_json };
        },
      });
    } else {
      console.log("===> no workflow nodes found, skipping workflow tools");
    }
  } else {
    console.log("===> no db found, skipping workflow tools");
  }

  // Conditionally register vector_search tool if embeddings exist
  if (db) {
    const embeddingsCount = await db.count_nodes_with_embeddings();
    if (embeddingsCount > 0) {
      console.log(
        `===> ${embeddingsCount} nodes with embeddings found, registering vector_search tool`,
      );
      allTools.vector_search = tool({
        description: defaultDescriptions.vector_search,
        inputSchema: z.object({
          query: z
            .string()
            .describe(
              "A natural language description of the code you are looking for",
            ),
          limit: z
            .number()
            .optional()
            .default(10)
            .describe("Maximum number of results to return"),
        }),
        execute: async ({
          query,
          limit,
        }: {
          query: string;
          limit?: number;
        }) => {
          try {
            const codeNodeTypes = [
              "Function",
              "Class",
              "Endpoint",
              "Datamodel",
              "Request",
              "Page",
              "Trait",
              "Var",
            ];
            const results = await db.vectorSearch(
              query,
              limit || 10,
              codeNodeTypes as any,
              [],
              0,
              undefined,
              repoScopePatterns(repos),
            );
            return results.map((node) => ({
              name: node.properties.name,
              node_type:
                node.labels.find((l) => l !== "Data_Bank") || node.labels[0],
              file: node.properties.file,
              line: node.properties.start,
              score: node.score,
            }));
          } catch (e) {
            console.error("Error in vector_search:", e);
            return `Vector search failed: ${e}`;
          }
        },
      });
    } else {
      console.log(
        "===> no nodes with embeddings found, skipping vector_search tool",
      );
    }
  } else {
    console.log("===> no db found, skipping vector_search tool");
  }

  // Register stakgraph graph tools (requires Neo4j connection)
  if (db) {
    allTools.stakgraph_search = tool({
      description: defaultDescriptions.stakgraph_search,
      inputSchema: stak.SearchSchema,
      execute: async (args: z.infer<typeof stak.SearchSchema>) => {
        const valid = new Set(relevant_node_types());
        const filtered_node_types = (args.node_types ?? []).filter(t => valid.has(t as any)) as any[];
        const { results, provenance } = await searchWithProvenance(
          args.query,
          args.limit ?? 10,
          filtered_node_types,
          false,
          args.max_tokens ?? 15000,
          (args.method ?? "hybrid") as any,
          "json",
          [],
          args.language,
          "relevance",
          // Default to the session's repo scope so search doesn't leak across
          // repos on a multi-repo graph; the model may still override with a
          // narrower pattern.
          args.include_patterns ?? repoScopePatterns(repos),
          args.exclude_patterns,
        );
        if (provenanceCollector) {
          provenanceCollector.entries.push({
            tool_name: "stakgraph_search",
            timestamp: new Date().toISOString(),
            provenance,
          });
        }
        if (!Array.isArray(results)) return "No results";
        return JSON.stringify(
          results.map((node: any) => ({
            name: node.properties?.name,
            node_type: node.node_type,
            file: node.properties?.file,
            lines: `${node.properties?.start ?? "?"}-${node.properties?.end ?? "?"}`,
            ref_id: node.ref_id,
            description:
              node.properties?.description || node.properties?.docs || "",
          })),
        );
      },
    });
    allTools.stakgraph_map = tool({
      description: stak.GetMapTool.description || defaultDescriptions.stakgraph_map,
      inputSchema: stak.GetMapSchema,
      execute: async (args: z.infer<typeof stak.GetMapSchema>) => {
        // When exactly one repo is in scope and the model resolved the node by
        // name (no ref_id/file), constrain node resolution to that repo so a
        // same-named node in another repo isn't picked.
        const scoped =
          !args.ref_id && !args.file && repos && repos.length === 1
            ? { ...args, file: repos[0] }
            : args;
        const result = await stak.getMap(scoped);
        return result.content?.[0]?.text ?? "";
      },
    });
    allTools.stakgraph_code = tool({
      description: defaultDescriptions.stakgraph_code,
      inputSchema: stak.GetCodeSchema,
      execute: async (args: z.infer<typeof stak.GetCodeSchema>) => {
        const result = await stak.getCode(args);
        return result.content?.[0]?.text ?? "";
      },
    });
    console.log(
      "===> registered stakgraph graph tools: stakgraph_search, stakgraph_map, stakgraph_code",
    );
  }

  // Register Jarvis knowledge-graph tools (gated only by JARVIS_URL being set).
  // The recursive graph_sub_agent tool is opt-in via toolsConfig.graph_sub_agent
  // (a string value overrides its description); depth is capped in toolsJarvis.
  const graphSubAgentCfg = toolsConfig?.graph_sub_agent;
  registerJarvisTools(allTools, {
    subAgent: toolConfigEnabled(graphSubAgentCfg)
      ? {
          description: toolConfigDescription(graphSubAgentCfg),
          maxDepth: isToolConfigObject(graphSubAgentCfg)
            ? graphSubAgentCfg.maxDepth
            : undefined,
          maxSteps: isToolConfigObject(graphSubAgentCfg)
            ? graphSubAgentCfg.maxSteps
            : undefined,
          modelName,
          apiKey,
          // Links each sub-agent run back to the owning session so it appears
          // as its own inspectable session in /sessions.
          parentSessionId: sessionId,
          repo:
            repos && repos.length > 0
              ? repos.join(", ")
              : repoPath.replace(/\/+$/, "").split("/").slice(-2).join("/"),
        }
      : undefined,
    // Opt-in ontology write tools (create/update/delete node & edge types).
    ontologyEdit: toolConfigEnabled(toolsConfig?.ontology_edit),
    // Opt-in graph data-write tools (triplets + generic node create/edit).
    // Any one of the family's keys enables all of them.
    graphWrite: graphWriteEnabled(toolsConfig),
    // Scope get_ontology to the caller's `ontologyDomains`, unless the model
    // asks for specific `domains` itself. Unset => no filter, all domains.
    defaultDomains: ontologyDomains,
    // Thread the run's AbortSignal so Jarvis HTTP calls can be cancelled when
    // the busy safety-timeout or /repo/agent/abort fires.
    abortSignal,
  });

  // Register Stakwork run-research tools (read-only, gated on the caller
  // supplying a Stakwork API key — plumbed via the request body, never an
  // LLM-visible parameter). The execute tool stakwork_run_step additionally
  // requires an explicit truthy toolsConfig.stakwork_run_step opt-in, since
  // it launches real (billable) step executions.
  if (stakwork?.apiKey) {
    registerStakworkTools(allTools, {
      ...stakwork,
      runStep: toolConfigEnabled(toolsConfig?.stakwork_run_step),
    });
  }

  // Register Google Sheets tools (gated on the caller supplying service-account
  // credentials via the request body — never an LLM-visible parameter, and
  // never persisted; see redactToolsConfig). Credentials come from the
  // top-level `googleSheets` field or, as a fallback, from
  // `toolsConfig.google_sheets`. The family is on by default when credentials
  // are present; toolsConfig can disable individual tools
  // (`sheets_get_values: false`) or override their descriptions.
  const sheetsOptions = resolveGoogleSheetsOptions(googleSheets, toolsConfig);
  if (sheetsOptions) {
    const sheetsOverrides: GoogleSheetsToolOverrides = Object.fromEntries(
      GOOGLE_SHEETS_TOOL_NAMES.map((name) => {
        const v = toolsConfig?.[name];
        return [
          name,
          {
            disabled: v === false || (isToolConfigObject(v) && v.enabled === false),
            description: toolConfigDescription(v),
          },
        ];
      })
    );
    registerGoogleSheetsTools(allTools, sheetsOptions, sheetsOverrides);
  }

  // Register sub-agent tools (remote agent delegation)
  if (subAgents && subAgents.length > 0) {
    for (const subAgent of subAgents) {
      // Validate required fields
      if (!subAgent.name || !subAgent.url || !subAgent.apiToken) {
        console.warn(
          `[sub-agent] Skipping invalid sub-agent config: missing name, url, or apiToken/apiKey`,
          {
            name: subAgent.name,
            hasUrl: !!subAgent.url,
            hasToken: !!subAgent.apiToken,
          },
        );
        continue;
      }
      // Validate name is a safe tool identifier (alphanumeric + underscores)
      if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(subAgent.name)) {
        console.warn(
          `[sub-agent] Skipping sub-agent with invalid name "${subAgent.name}" — must be alphanumeric/underscores starting with a letter`,
        );
        continue;
      }
      // Validate URL is parseable
      try {
        new URL(subAgent.url);
      } catch {
        console.warn(
          `[sub-agent] Skipping sub-agent "${subAgent.name}" with invalid URL: ${subAgent.url}`,
        );
        continue;
      }

      const repoNames = subAgentRepoNames(subAgent);
      const reposPart = repoNames.length > 0
        ? ` Available repos in this sub-agent: ${repoNames.join(", ")}.`
        : "";
      const description =
        (subAgent.description ||
          `Ask the "${subAgent.name}" sub-agent a question. This delegates to a separate agent instance that may have different codebase context.`) +
        reposPart;

      // Capture subAgent in closure (name is guaranteed non-empty after validation above)
      const sa = subAgent;
      const saName = sa.name!;
      allTools[saName] = tool({
        description,
        inputSchema: z.object({
          prompt: z
            .string()
            .describe(
              "A focused, specific question or task to delegate to the sub-agent",
            ),
        }),
        execute: async ({ prompt }: { prompt: string }) => {
          console.log(
            `[sub-agent:${sa.name}] Executing with prompt: ${prompt.slice(0, 200)}...`,
          );
          const answer = await callRemoteAgent(sa, prompt);
          return answer;
        },
      });
      console.log(`===> registered sub-agent tool: ${sa.name}`);
    }
  }

  // Register GGNN tools if configured
  if (ggnn && ggnn.tools && ggnn.tools.length > 0) {
    const ggnnApiKey = ggnn.apiKey;
    const languages = ggnn.languages ?? [];
    const ggnnHeaders = {
      Authorization: `Bearer ${ggnnApiKey}`,
      "Content-Type": "application/json",
    };

    for (const gt of ggnn.tools) {
      if (!gt.name || !gt.endpoint) continue;
      const url = `${ggnn.url.replace(/\/$/, "")}${gt.endpoint}`;

      if (gt.bodyType === "predict") {
        // /predict — call before task starts: { task_description, languages }
        allTools[gt.name] = tool({
          description: gt.description,
          inputSchema: z.object({
            task_description: z
              .string()
              .describe("Description of the task about to be executed"),
          }),
          execute: async ({
            task_description,
          }: {
            task_description: string;
          }) => {
            console.log(`[ggnn:${gt.name}] POST ${url}`);
            const res = await fetch(url, {
              method: "POST",
              headers: ggnnHeaders,
              body: JSON.stringify({ task_description, languages }),
            });
            const data = await res.json();
            console.log(
              `[ggnn:${gt.name}] response:`,
              JSON.stringify(data, null, 2),
            );
            return data;
          },
        });
      } else if (gt.bodyType === "check") {
        // /check — call mid-execution: { task_description, languages, trace_so_far }
        allTools[gt.name] = tool({
          description: gt.description,
          inputSchema: z.object({
            task_description: z
              .string()
              .describe("Description of the current task"),
          }),
          execute: async ({
            task_description,
          }: {
            task_description: string;
          }) => {
            const trace = messagesRef?.current ?? [];
            console.log(
              `[ggnn:${gt.name}] POST ${url} (${trace.length} messages in trace)`,
            );
            const res = await fetch(url, {
              method: "POST",
              headers: ggnnHeaders,
              body: JSON.stringify({
                task_description,
                languages,
                trace_so_far: trace,
              }),
            });
            const data = await res.json();
            // Strip node_assessments to reduce token noise
            const { node_assessments, ...summary } = data;
            console.log(
              `[ggnn:${gt.name}] response:`,
              JSON.stringify(summary, null, 2),
            );
            return summary;
          },
        });
      } else if (gt.bodyType === "score-plan") {
        // /score-plan — call after planning: { task_description, languages, plan }
        allTools[gt.name] = tool({
          description: gt.description,
          inputSchema: z.object({
            task_description: z.string().describe("Description of the task"),
            plan: z.string().describe("The step-by-step plan to score"),
          }),
          execute: async ({
            task_description,
            plan,
          }: {
            task_description: string;
            plan: string;
          }) => {
            console.log(`[ggnn:${gt.name}] POST ${url}`);
            const res = await fetch(url, {
              method: "POST",
              headers: ggnnHeaders,
              body: JSON.stringify({ task_description, languages, plan }),
            });
            const data = await res.json();
            console.log(
              `[ggnn:${gt.name}] response:`,
              JSON.stringify(data, null, 2),
            );
            return data;
          },
        });
      }
      console.log(`===> registered ggnn tool: ${gt.name} -> ${url}`);
    }
  }

  // If no config, return all tools
  if (!toolsConfig) {
    return allTools;
  } else {
    // SPECIAL TOOLS: NOT included by default, but can be enabled with toolsConfig
    if (toolConfigEnabled(toolsConfig.ask_clarifying_questions)) {
      // Schema for artifact objects (color swatches, diagrams, tables, etc.)
      const artifactSchema = z.object({
        type: z.enum(["mermaid", "comparison_table", "color_swatch"]),
        data: z.record(z.string(), z.any()),
      });

      allTools.ask_clarifying_questions = tool({
        description: defaultDescriptions.ask_clarifying_questions,
        inputSchema: z.object({
          questions: z
            .array(
              z.object({
                question: z.string().describe("The question to ask the user"),
                type: z
                  .enum(["single_choice", "multiple_choice", "color_swatch"])
                  .describe("The type of question"),
                options: z
                  .array(z.string())
                  .optional()
                  .describe(
                    "Options - either simple strings or rich objects with artifacts",
                  ),
                questionArtifact: artifactSchema
                  .optional()
                  .describe(
                    "Artifact to display alongside the question (e.g., mermaid diagram)",
                  ),
              }),
            )
            .describe("The questions to ask the user (MAXIMUM 4 QUESTIONS)"),
        }),
        execute: async ({ questions }: { questions: any[] }) => {
          return questions;
        },
      });
    }
    if (toolConfigEnabled(toolsConfig.logs_agent)) {
      allTools.logs_agent = tool({
        description:
          toolConfigDescription(toolsConfig.logs_agent) ??
          DEFAULT_DESCRIPTIONS.logs_agent,
        inputSchema: z.object({
          prompt: z.string().describe("A focused question about runtime logs"),
        }),
        execute: async ({ prompt }: { prompt: string }) => {
          const sessionId = randomUUID();
          const logsDir = createRunLogsDir(randomUUID());
          try {
            const result = await log_agent_context(prompt, {
              modelName,
              apiKey,
              logsDir,
              sessionId,
              source: "logs_agent",
            });
            return result.final || "No result returned from logs agent.";
          } catch (e) {
            return `Logs agent error: ${e instanceof Error ? e.message : String(e)}`;
          } finally {
            cleanupRunLogsDir(logsDir);
          }
        },
      });
    }
    // str_replace_based_edit_tool
    if (toolConfigEnabled(toolsConfig.str_replace_based_edit_tool)) {
      if (provider === "anthropic") {
        // Native provider tool — model is specially trained on this schema
        const { createAnthropic } = await import("@ai-sdk/anthropic");
        const baseURL = getGatewayBaseURL("anthropic");
        const ant = createAnthropic({ apiKey, ...(baseURL && { baseURL }) });
        allTools.str_replace_based_edit_tool = ant.tools.textEditor_20250728({
          execute: async (input) => textEdit(input as TextEditInput, editorRoots(repoPath, confinedRun(options))),
        }) as any as Tool<any, any>;
      } else {
        // Generic fallback for OpenAI / other providers
        allTools.str_replace_based_edit_tool = tool({
          description:
            toolConfigDescription(toolsConfig.str_replace_based_edit_tool) ??
            defaultDescriptions.str_replace_based_edit_tool,
          inputSchema: z.object({
            command: z.enum(["view", "create", "str_replace", "insert"]),
            path: z.string(),
            file_text: z.string().optional(),
            old_str: z.string().optional(),
            new_str: z.string().optional(),
            insert_line: z.number().int().optional(),
            insert_text: z.string().optional(),
            view_range: z.array(z.number().int()).length(2).optional(),
          }),
          execute: async (input) => textEdit(input as TextEditInput, editorRoots(repoPath, confinedRun(options))),
        });
      }
    }
    // apply_patch — shells out to `git apply`; cloned repos are real git repos
    if (toolConfigEnabled(toolsConfig.apply_patch)) {
      allTools.apply_patch = tool({
        description:
          toolConfigDescription(toolsConfig.apply_patch) ??
          defaultDescriptions.apply_patch,
        inputSchema: z.object({
          patch: z.string().describe("Unified-diff patch string to apply"),
        }),
        execute: async ({ patch }: { patch: string }) => {
          const tmp = `/tmp/patch_${randomUUID()}.diff`;
          writeFileSync(tmp, patch);
          try {
            return await executeBashCommand(
              `git apply "${tmp}"`,
              repoPath,
              undefined,
              ghEnv,
            );
          } catch (e) {
            return `apply_patch failed: ${e}`;
          } finally {
            try {
              unlinkSync(tmp);
            } catch {}
          }
        },
      });
    }
    // create_pr — opt-in write path: stage, scan, commit, push, open PR.
    // Gated strictly inside the toolsConfig branch so flagless callers never
    // see it (the "if (!toolsConfig) return allTools" early-return above means
    // unconditional registrations in allTools would leak to every caller).
    if (toolConfigEnabled(toolsConfig.create_pr)) {
      const prOpts = options?.prMode;
      const prColl = options?.prCollector;
      allTools.create_pr = tool({
        description:
          toolConfigDescription(toolsConfig.create_pr) ??
          defaultDescriptions.create_pr,
        // No branch_hint: the branch is created (and named) at acquireWorktree
        // time, before the model runs, so a hint here could never take effect.
        inputSchema: z.object({
          title: z.string().describe("PR title"),
          body: z.string().describe("PR body / description"),
        }),
        execute: async ({
          title,
          body,
        }: {
          title: string;
          body: string;
        }) => {
          if (!prOpts) {
            console.error("[create_pr] invoked with no worktree handle — refusing");
            return "create_pr is not available: no worktree handle was resolved for this run";
          }
          console.log(
            `[create_pr] invoked: runId=${prOpts.handle.runId} branch=${prOpts.handle.branch} title=${JSON.stringify(title.slice(0, 120))}`,
          );
          const result = await landChange({
            handle: prOpts.handle,
            identity: prOpts.identity,
            env: prOpts.env,
            githubClient: prOpts.githubClient,
            title,
            body,
          });
          if (prColl) {
            prColl.result = result;
          }
          console.log(
            result.ok
              ? `[create_pr] landed: ${result.url} (${result.filesChanged} files)`
              : `[create_pr] failed: ${result.failure} — ${result.error.slice(0, 300)}`,
          );
          if (result.ok) {
            return (
              `PR created successfully!\n` +
              `URL: ${result.url}\n` +
              `Branch: ${result.branch}\n` +
              `Files changed: ${result.filesChanged}\n\n` +
              `Diff (first 4000 chars):\n${result.diff.slice(0, 4000)}`
            );
          } else {
            return (
              `PR creation failed (${result.failure}): ${result.error}\n\n` +
              `Diff (first 4000 chars):\n${result.diff.slice(0, 4000)}`
            );
          }
        },
      });
    }
    // generate_docx — Pandoc-based Word document generation
    if (toolConfigEnabled(toolsConfig.generate_docx)) {
      allTools.generate_docx = tool({
        description:
          toolConfigDescription(toolsConfig.generate_docx) ??
          defaultDescriptions.generate_docx,
        inputSchema: z.object({
          markdown: z.string().optional().describe("Inline Markdown content to convert to .docx (small documents only)"),
          markdownPath: z.string().optional().describe("Path to a Markdown file to convert (preferred for large documents — build it incrementally with bash first); relative paths resolve against the repo directory"),
          template: z.string().optional().describe("Optional bundled reference-doc template name for styling"),
        }),
        execute: async (input) => runDocx(input, repoPath),
      });
    }
    // generate_xlsx — openpyxl-based Excel workbook generation
    if (toolConfigEnabled(toolsConfig.generate_xlsx)) {
      allTools.generate_xlsx = tool({
        description:
          toolConfigDescription(toolsConfig.generate_xlsx) ??
          defaultDescriptions.generate_xlsx,
        inputSchema: z.object({
          filename: z.string().optional().describe("Base filename hint (without extension)"),
          sheets: z.array(z.object({
            name: z.string().describe("Sheet tab name"),
            rows: z.array(z.array(z.union([z.string(), z.number()]))).optional()
              .describe("Row-based data: each inner array is one row"),
            cells: z.array(z.object({
              ref: z.string().describe("Cell reference, e.g. A1 or B3"),
              value: z.union([z.string(), z.number()]).optional().describe("Cell value"),
              formula: z.string().optional().describe("Cell formula, e.g. =Sheet2!B2+1"),
            })).optional().describe("Fine-grained cell overrides (applied after rows)"),
          })).describe("Array of sheet definitions"),
        }),
        execute: async (input) => runXlsx(input),
      });
    }
    // generate_xlsx_computed — formula-free Excel workbook with server-side computed values
    if (toolConfigEnabled(toolsConfig.generate_xlsx_computed)) {
      allTools.generate_xlsx_computed = tool({
        description:
          toolConfigDescription(toolsConfig.generate_xlsx_computed) ??
          defaultDescriptions.generate_xlsx_computed,
        inputSchema: z.object({
          filename: z.string().optional().describe("Base filename hint (without extension)"),
          sheets: z.array(z.object({
            name: z.string().describe("Sheet tab name"),
            rows: z.array(z.array(z.union([z.string(), z.number()]))).optional()
              .describe("Row-based data: each inner array is one row"),
            cells: z.array(z.object({
              ref: z.string().describe("Cell reference, e.g. A1 or B3"),
              value: z.union([z.string(), z.number()]).optional().describe("Literal cell value (no formula field — this tool is formula-free)"),
            })).optional().describe("Fine-grained literal cell overrides (applied after rows)"),
            computed: z.array(z.object({
              ref: z.string().describe("Destination cell reference, e.g. B10"),
              op: z.enum(["sum", "percent_of_total", "ratio"]).describe(
                "'sum': sum a range (column B2:B9 or row B2:E2); " +
                "'percent_of_total': value_ref/total_ref × 100 (or raw ratio if as_fraction:true); " +
                "'ratio': value_ref/denominator_ref"
              ),
              range: z.string().optional().describe("Cell range for 'sum' op, e.g. B2:B9 or Sheet2!B2:B9"),
              value_ref: z.string().optional().describe("Numerator cell for 'percent_of_total' or 'ratio', e.g. B2 or Sheet2!B2"),
              total_ref: z.string().optional().describe("Denominator cell for 'percent_of_total' (also '% of shares'), e.g. B10"),
              denominator_ref: z.string().optional().describe("Denominator cell for 'ratio'"),
              decimals: z.number().optional().describe("Decimal places to round to (default 2)"),
              as_fraction: z.boolean().optional().describe("For 'percent_of_total': write raw 0–1 ratio instead of percent-scaled (×100)"),
            })).optional().describe(
              "Computed cells evaluated server-side and written as literal numbers. " +
              "Evaluated sequentially so earlier results feed later entries. " +
              "For per-row '% of shares', emit one entry per destination row."
            ),
          })).describe("Array of sheet definitions"),
        }),
        execute: async (input) => runXlsx(input, "generate_xlsx_computed"),
      });
    }
    // concepts
    if (
      toolConfigEnabled(toolsConfig.learn_concept) ||
      toolConfigEnabled(toolsConfig.list_concepts) ||
      toolConfigEnabled(toolsConfig.learn_concepts)
    ) {
      allTools.list_concepts = tool({
        description: defaultDescriptions.list_concepts,
        inputSchema: z.object({}),
        execute: async () => {
          try {
            const repo = isMultiRepo ? undefined : `${repoOwner}/${repoName}`;
            const result = await listConcepts(repo);
            return {
              concepts: result.concepts.map((f) => ({
                id: f.id,
                name: f.name,
                description: f.description,
              })),
              total: result.total,
              repo,
            };
          } catch (e) {
            console.error("Error listing concepts:", e);
            return "Could not retrieve concepts";
          }
        },
      });
      allTools.learn_concept = tool({
        description: defaultDescriptions.learn_concept,
        inputSchema: z.object({
          concept_id: z
            .string()
            .describe("The ID of the concept/feature to learn about"),
        }),
        execute: async ({ concept_id }: { concept_id: string }) => {
          try {
            const repo = isMultiRepo ? undefined : `${repoOwner}/${repoName}`;
            const doc = await getConceptDocumentation(concept_id, repo);
            if (!doc) {
              return { error: "Concept not found" };
            }
            return doc;
          } catch (e) {
            console.error("Error getting concept:", e);
            return "Could not retrieve concept";
          }
        },
      });
    }
    // concept proposals (human-reviewed writes; propose tool is opt-in)
    if (
      toolConfigEnabled(toolsConfig.propose_concept_change) ||
      toolConfigEnabled(toolsConfig.list_concept_proposals)
    ) {
      allTools.list_concept_proposals = tool({
        description: defaultDescriptions.list_concept_proposals,
        inputSchema: z.object({
          status: z
            .enum(["pending", "accepted", "rejected"])
            .optional()
            .default("pending")
            .describe("Filter by proposal status (default: pending)"),
        }),
        execute: async ({ status }) => {
          try {
            const repo = isMultiRepo ? undefined : `${repoOwner}/${repoName}`;
            const proposals = await listProposals(repo, status);
            return {
              proposals: proposals.map((p) => ({
                id: p.id,
                action: p.action,
                status: p.status,
                conceptId: p.conceptId,
                mergeIntoConceptId: p.mergeIntoConceptId,
                name: p.name,
                rationale: p.rationale,
                source: p.source,
                createdAt: p.createdAt,
              })),
              count: proposals.length,
              repo,
            };
          } catch (e) {
            console.error("Error listing concept proposals:", e);
            return "Could not retrieve concept proposals";
          }
        },
      });
    }
    if (toolConfigEnabled(toolsConfig.propose_concept_change)) {
      allTools.propose_concept_change = tool({
        description: defaultDescriptions.propose_concept_change,
        inputSchema: z.object({
          action: z
            .enum(["create", "update", "delete", "merge"])
            .describe(
              "create a new concept, update or delete an existing one, or merge concept_id into merge_into_concept_id"
            ),
          concept_id: z
            .string()
            .optional()
            .describe(
              "Target concept id (required for update/delete; for merge, the concept that will be absorbed and deleted). Use ids from list_concepts — never fabricate."
            ),
          merge_into_concept_id: z
            .string()
            .optional()
            .describe("merge only: the surviving concept's id"),
          name: z
            .string()
            .optional()
            .describe("create only: human-readable name for the new concept"),
          description: z
            .string()
            .optional()
            .describe("Optional one-line description (new or replacement)"),
          documentation: z
            .string()
            .optional()
            .describe(
              "Full markdown documentation (required for create/update/merge). Replaces the entire body — for updates, start from learn_concept's current docs."
            ),
          rationale: z
            .string()
            .describe(
              "Why this change should be made — shown to the human reviewer"
            ),
          pr_numbers: z
            .array(z.number())
            .optional()
            .describe("PR numbers that motivated this proposal (evidence)"),
        }),
        execute: async (input) => {
          try {
            const repo = isMultiRepo ? undefined : `${repoOwner}/${repoName}`;
            const storage = new GitreeGraphStorage();
            await storage.initialize();
            const proposal = await createProposal(storage, {
              action: input.action,
              repo,
              conceptId: input.concept_id,
              mergeIntoConceptId: input.merge_into_concept_id,
              name: input.name,
              description: input.description,
              documentation: input.documentation,
              rationale: input.rationale,
              source: "agent",
              prNumbers: input.pr_numbers,
            });
            return {
              status: "pending_review",
              proposalId: proposal.id,
              action: proposal.action,
              message:
                "Proposal filed. It will only take effect if a human reviewer accepts it.",
            };
          } catch (e: any) {
            if (e instanceof HttpError) {
              return { error: e.message, ...(e.extra || {}) };
            }
            console.error("Error proposing concept change:", e);
            return { error: e?.message || "Could not create proposal" };
          }
        },
      });
    }
  }

  // Start with all tools, then apply config to customize or exclude
  const selectedTools: Record<string, Tool<any, any>> = { ...allTools };

  for (const [toolName, config] of Object.entries(toolsConfig) as [
    ToolName,
    ToolConfigValue,
  ][]) {
    const originalTool = allTools[toolName];
    if (!originalTool) continue;

    // Exclude the tool when explicitly disabled (false, or object { enabled: false }).
    if (config === false || (isToolConfigObject(config) && config.enabled === false)) {
      delete selectedTools[toolName];
      continue;
    }
    // Apply a description override carried by a string or object config value.
    const description = toolConfigDescription(config);
    if (description) {
      selectedTools[toolName] = tool({
        description,
        inputSchema: (originalTool as any).inputSchema,
        execute: (originalTool as any).execute,
      }) as Tool<any, any>;
    }
    // Otherwise keep the default (already in selectedTools).
  }

  console.log("selectedTools", Object.keys(selectedTools));

  return selectedTools;
}

export function getDefaultToolDescriptions(): ToolsConfig {
  return DEFAULT_DESCRIPTIONS;
}

/** Accept ToolsConfig as an object or as a flat string like "key1 true key2 true" */
export function normalizeToolsConfig(
  raw: unknown
): ToolsConfig | undefined {
  if (raw == null) return undefined;
  if (typeof raw === "object" && !Array.isArray(raw)) return raw as ToolsConfig;
  if (typeof raw === "string") {
    const tokens = raw.trim().split(/\s+/);
    const config: ToolsConfig = {};
    let i = 0;
    while (i < tokens.length) {
      const key = tokens[i];
      if (!TOOL_NAMES.has(key)) {
        // Skip family-config extra keys silently; warn on truly unknown keys
        if (!(TOOLS_CONFIG_EXTRA_KEYS as string[]).includes(key)) {
          console.warn(`[normalizeToolsConfig] Unknown toolsConfig key dropped: "${key}"`);
        }
        i++;
        continue;
      }
      const next = tokens[i + 1];
      if (next === "true") {
        (config as any)[key] = true;
        i += 2;
      } else if (next === "false") {
        (config as any)[key] = false;
        i += 2;
      } else if (next === "null") {
        (config as any)[key] = null;
        i += 2;
      } else if (next !== undefined && !TOOL_NAMES.has(next)) {
        // treat as a custom string value
        (config as any)[key] = next;
        i += 2;
      } else {
        // no value follows, default to true
        (config as any)[key] = true;
        i++;
      }
    }
    return Object.keys(config).length > 0 ? config : undefined;
  }
  return undefined;
}