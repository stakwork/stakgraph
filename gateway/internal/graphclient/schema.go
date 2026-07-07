package graphclient

// Label and relationship-type names for the agent catalog subgraph.
//
// The Hive-owned nodes (HiveAgent/HiveTool/HiveSkill/HiveSource) are
// PascalCase with a `Hive` prefix so they never collide with the
// generic Agent/Tool/Skill nodes other systems may push into the
// shared graph. They deliberately do NOT carry the `Data_Bank` label
// (that label drives mcp's code full-text + vector indexes; the
// catalog must not pollute code search).
//
// Prompts are the exception: a HiveAgent links to the *existing*
// generic `:Prompt` nodes (written by the Stakwork prompt workflow,
// keyed by `name`) via HAS_PROMPT. The gateway never creates or
// deletes Prompt nodes — it only wires/unwires the relationship — so
// there is no `HivePrompt` label.
//
// This is the one place label/edge names are defined — every Cypher
// string in this package interpolates these constants so a rename is a
// single-edit affair. See gateway/plans/agent-catalog.md "Data model".
const (
	LabelAgent  = "HiveAgent"
	LabelPrompt = "Prompt" // shared node authored elsewhere; linked, not owned
	LabelTool   = "HiveTool"
	LabelSkill  = "HiveSkill"
	LabelSource = "HiveSource"

	RelHasPrompt = "HAS_PROMPT"
	RelHasTool   = "HAS_TOOL"
	RelHasSkill  = "HAS_SKILL"
	RelDefinedBy = "DEFINED_BY"

	// ── Evals subgraph ────────────────────────────────────────────────
	//
	// The Eval* nodes are NOT gateway-owned: Jarvis (jarvis-backend)
	// authors them through its schema/node_key/Data_Bank machinery, so
	// the gateway only READS them (agent-detail Evals tab) and never
	// creates/updates them via raw Cypher — mutations are delegated to
	// Hive (see internal/adminapi/hivecallback.go). The one exception is
	// HAS_EVAL_SET: that edge anchors a gateway-owned :HiveAgent to a
	// Jarvis-owned :EvalSet, so the gateway MERGEs it directly (edge
	// only, no node schema involved). Jarvis stamps every Eval* node
	// with a `ref_id` (uuid4) which is how we address them.
	LabelEvalSet         = "EvalSet"
	LabelEvalRequirement = "EvalRequirement"
	LabelEvalTrigger     = "EvalTrigger"
	LabelEvalOutput      = "EvalTriggerOutput"

	RelHasEvalSet     = "HAS_EVAL_SET" // HiveAgent -> EvalSet (gateway-owned)
	RelHasRequirement = "HAS_REQUIREMENT"
	RelHasTrigger     = "HAS_TRIGGER"
	RelHasOutput      = "HAS_OUTPUT"
	RelAttributedTo   = "ATTRIBUTED_TO" // EvalTrigger -> HiveAgent

	// UNLINKED_EVAL_SET is the tombstone for an operator's explicit
	// unlink. HAS_EVAL_SET edges are auto-derived from EvalTrigger.agent
	// on every agent-evals read, so a bare edge delete would just get
	// re-derived on the next load; the tombstone records "the operator
	// said no" and blocks re-derivation until an explicit re-link (or
	// set deletion) clears it. Gateway-owned, like HAS_EVAL_SET.
	RelUnlinkedEvalSet = "UNLINKED_EVAL_SET" // HiveAgent -> EvalSet (gateway-owned)
)

// SchemaStatements are the constraint/index DDL the catalog needs.
// Every statement is `IF NOT EXISTS`, so running them on every cold
// start is idempotent and cheap. The write handler runs these once
// (guarded by a sync.Once) before its first upsert — "indexes on
// first write" rather than a migration step, because the gateway has
// no migration runner and neo4j may be unreachable at boot.
func SchemaStatements() []Statement {
	return []Statement{
		{Statement: "CREATE CONSTRAINT hive_agent_name IF NOT EXISTS " +
			"FOR (a:" + LabelAgent + ") REQUIRE a.name IS UNIQUE"},
		// No index on :Prompt here — those nodes (and their indexes) are
		// owned by the mcp/prompt-workflow writers; the catalog only
		// MATCHes them by `name` to link.
		{Statement: "CREATE INDEX hive_tool_key IF NOT EXISTS " +
			"FOR (t:" + LabelTool + ") ON (t.node_key)"},
		{Statement: "CREATE INDEX hive_skill_key IF NOT EXISTS " +
			"FOR (s:" + LabelSkill + ") ON (s.node_key)"},
	}
}
