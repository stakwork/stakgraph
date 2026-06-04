// ── Public API ─────────────────────────────────────────────────────────────

// Re-export the engine's own zod so consumers define step schemas against
// the exact version `defineStep` and the schema-introspection helpers expect
// (avoids dual zod-version type/runtime mismatches in host apps).
export { z } from "zod";

// Core types and builders
export {
  flow,
  step,
  defineStep,
  type Flow,
  type Step,
  type StepDef,
  type AnyStepDef,
  type StepContext,
  type StepOptions,
  type StepRegistry,
  type RunEvent,
  type RunResult,
  type RunSummary,
  type RunEventType,
} from "./core.js";

// Runner
export { runWorkflow, type RunOptions } from "./runner.js";

// Expression engine
export {
  evaluateExpr,
  resolveTemplate,
  resolveConfig,
  hasTemplates,
  TemplateError,
} from "./expr.js";

// Persistence
export {
  type RunStore,
  FileRunStore,
  MemoryRunStore,
  generateRunId,
  tailJsonl,
} from "./store.js";

// Chat persistence (detached AI-builder background jobs)
export {
  type ChatStore,
  type ChatMeta,
  type ChatEvent,
  type ChatEventType,
  type ChatStatus,
  type StoredMessage,
  FileChatStore,
  MemoryChatStore,
  generateChatId,
  truncateToolMessages,
  isChatTerminal,
} from "./chat-store.js";

// Registry
export {
  buildRegistry,
  coreRegistry,
  createRegistry,
  type StepSource,
  type StepSources,
  type RegistryBundle,
} from "./steps/registry.js";

// Workspace
export {
  WorkspaceManager,
  type WorkflowMetadata,
  type WorkflowVersionInfo,
  type WorkflowListEntry,
  type StepDirMetadata,
  type StepInfo,
  type StepVersionInfo,
  type StepListEntry,
  type StepVersionsResult,
} from "./workspace.js";

// Content-hash versioning (internal dedup) + sequential version labels
export { contentHash, nextVersionLabel } from "./version.js";

// Vein factory — the primary entry point for library usage.
export {
  createVein,
  type Vein,
  type VeinOptions,
  type VeinRunOptions,
} from "./createVein.js";

// Default filesystem-backed server (a thin wrapper over createVein).
export { getApp, startServer } from "./server.js";
