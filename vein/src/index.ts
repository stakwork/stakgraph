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
  type PromoteSpec,
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

// Run control — cancel / pause / resume for run trees (RUN_CONTROL_SPEC.md)
export {
  RunController,
  CancelledError,
  isCancelledError,
  type RunControl,
  type ControlState,
} from "./run-control.js";

// Resume journal — replay completed step outputs from the event log
export {
  buildJournal,
  invalidateFrom,
  readRunStart,
  transitiveDependents,
  type InvalidateResult,
} from "./journal.js";

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
  type TailOpts,
  type PartialRunSummary,
  FileRunStore,
  MemoryRunStore,
  generateRunId,
  tailJsonl,
  tailFromPolling,
  lastRunAtFromIds,
  summarizeFromEvents,
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
  type WorkspaceStore,
  FileWorkspaceStore,
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

// LLM token usage + cost (shared by the agent + lab eval/score steps)
export {
  TOKEN_PRICING,
  type LLMProvider,
  type TokenPricing,
  type TokenUsage,
  emptyUsage,
  addUsage,
  coerceUsage,
  usageFromResult,
  computeCost,
} from "./pricing.js";

// Standard capabilities — the http + secrets + artifacts services adapter steps build on.
export {
  standardServices,
  httpCapability,
  secretsCapability,
  fileArtifactsCapability,
  type ArtifactsCapability,
  type VeinCapabilities,
  type HttpCapability,
  type HttpRequestOptions,
  type HttpResponse,
  type SecretsCapability,
  type SecretReadable,
  type FetchLike,
} from "./capabilities.js";

// Secret store — deployment-scoped, encrypted credential persistence behind
// the `secrets` capability + the `/secrets` admin endpoints.
export {
  type SecretStore,
  type SecretInfo,
  FileSecretStore,
  MemorySecretStore,
  isValidSecretName,
} from "./secret-store.js";

// Record/replay for the services bag (the adapter "safe inner loop").
export {
  withCassette,
  emptyCassette,
  loadCassette,
  saveCassette,
  type Cassette,
  type CassetteEntry,
  type CassetteMode,
  type WithCassetteOptions,
} from "./cassette.js";

// Single-step runner (test one step in isolation, with optional cassette).
export {
  runSingleStep,
  cassettePath,
  type RunStepOptions,
  type RunStepResult,
} from "./run-step.js";

// Authoring — the workspace's author/test/inspect operations as one
// injectable service: what the meta/* steps are plumbing over. Auto-provided
// by createVein as `services.authoring`; embedders can build their own.
export {
  buildAuthoringCapability,
  AI_PUBLISHER,
  type AuthoringCapability,
  type AuthoringDeps,
  type StepPublishDeps,
  type StepPublishResult,
  type RunStepArgs,
} from "./authoring.js";

// Vein factory — the primary entry point for library usage.
export {
  createVein,
  type Vein,
  type VeinOptions,
  type VeinRunOptions,
} from "./createVein.js";

// Default filesystem-backed server (a thin wrapper over createVein).
export { getApp, startServer } from "./server.js";

// Graph backend — jarvis-compatible Neo4j writes/reads over bolt, no jarvis
// in the loop (plans/jarvis-graph-compat.md). Opt-in: nothing here runs
// unless a consumer opens a backend.
export {
  Bolt,
  graphConfigFromEnv,
  int as neo4jInt,
  type GraphConfig,
} from "./graph/bolt.js";
export {
  VEIN_SCHEMAS,
  VEIN_EDGES,
  VEIN_DOMAIN,
  VEIN_DOMAIN_LABEL,
  getVeinSchema,
  isVeinType,
  effectiveAttributes,
  typeLabelOf,
  type VeinSchema,
  type VeinEdgeDef,
  type AttrType,
} from "./graph/vein-schemas.js";
export { seedVeinDomain, type SeedReport } from "./graph/schema-seed.js";
export {
  NodeWriter,
  GraphValidationError,
  validateNode,
  composeNodeKey,
  buildSearchText,
  type Embedder,
  type NodeInput,
  type NodeWriteResult,
  type NodeUpdate,
  type NodeUpdateResult,
  type WriteMode,
  type WriteOptions,
  type GraphValidationCode,
} from "./graph/node-writer.js";
export { EdgeWriter, type EdgeInput, type EdgeWriteResult } from "./graph/edge-writer.js";
export { SchemaResolver, EDGE_TYPES_ALLOWLIST, type NodeSchema, type EdgeSchemaMatch } from "./graph/schema-resolver.js";
export { seedJarvisOntology, type OntologySeedReport } from "./graph/ontology-seed.js";
export { JARVIS_ONTOLOGY, type OntologyFixture } from "./graph/fixtures/jarvis-ontology.js";
export { MiniLMEmbedder, backfillEmbeddings, EMBEDDING_DIM, type BackfillReport } from "./graph/embeddings.js";
export {
  GraphReader,
  GraphReadError,
  type NodeEnvelope,
  type EdgeEnvelope,
  type SearchParams,
  type SearchResult,
  type NeighborsParams,
  type ConnectionCount,
  type SchemaEnvelope,
  type OntologyEdge,
} from "./graph/search.js";
export {
  openGraphBackend,
  openGraphBackendFromEnv,
  closeGraphBackends,
  type GraphBackend,
  type GraphBackendOptions,
} from "./graph/backend.js";
// Graph-backed workspace + the run/chat projector (plans/generic-storage.md §7).
export { Neo4jWorkspaceStore, type Neo4jWorkspaceStoreOptions } from "./graph/workspace-store.js";
export { graphWorkspaceFromEnv, graphWorkspaceRequested } from "./graph/wiring.js";
export {
  projectRuns,
  projectChats,
  projectAll,
  projectRunEvents,
  type ProjectRunsOptions,
  type ProjectReport,
} from "./graph/projector.js";
