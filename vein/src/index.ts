// ── Public API ─────────────────────────────────────────────────────────────

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
} from "./store.js";

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
  type StepListEntry,
} from "./workspace.js";

// Vein factory — the primary entry point for library usage.
export {
  createVein,
  type Vein,
  type VeinOptions,
  type VeinRunOptions,
} from "./createVein.js";

// Default filesystem-backed server (a thin wrapper over createVein).
export { getApp, startServer } from "./server.js";
