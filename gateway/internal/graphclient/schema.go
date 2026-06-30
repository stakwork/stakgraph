package graphclient

// Label and relationship-type names for the agent catalog subgraph.
//
// All labels are PascalCase with a `Hive` prefix so they never collide
// with the generic Agent/Prompt/Tool/Skill nodes other systems may
// push into the shared graph. Hive catalog nodes deliberately do NOT
// carry the `Data_Bank` label (that label drives mcp's code full-text
// + vector indexes; the catalog must not pollute code search).
//
// This is the one place label/edge names are defined — every Cypher
// string in this package interpolates these constants so a rename is a
// single-edit affair. See gateway/plans/agent-catalog.md "Data model".
const (
	LabelAgent  = "HiveAgent"
	LabelPrompt = "HivePrompt"
	LabelTool   = "HiveTool"
	LabelSkill  = "HiveSkill"
	LabelSource = "HiveSource"

	RelHasPrompt = "HAS_PROMPT"
	RelHasTool   = "HAS_TOOL"
	RelHasSkill  = "HAS_SKILL"
	RelDefinedBy = "DEFINED_BY"
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
		{Statement: "CREATE INDEX hive_prompt_key IF NOT EXISTS " +
			"FOR (p:" + LabelPrompt + ") ON (p.node_key)"},
		{Statement: "CREATE INDEX hive_tool_key IF NOT EXISTS " +
			"FOR (t:" + LabelTool + ") ON (t.node_key)"},
		{Statement: "CREATE INDEX hive_skill_key IF NOT EXISTS " +
			"FOR (s:" + LabelSkill + ") ON (s.node_key)"},
	}
}
