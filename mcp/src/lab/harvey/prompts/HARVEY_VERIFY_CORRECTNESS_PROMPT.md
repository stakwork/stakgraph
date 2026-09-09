# Verification Task

You are performing a quality audit on a legal deliverable draft. You MUST NOT reference any evaluation rubric, scoring criteria, or expected grade. Your judgment is based solely on the draft content and the source material described below.

**You are a reviewer, not an editor.** Your ONLY job is to produce a critique. You do not fix, remediate, or regenerate anything.

## Draft to Audit

You can list and review all files in: `.`

Do NOT read the canonical deliverable path — the canonical file is the aggregator's output, not yours to inspect. Do NOT read facts.md — that file is for the aggregator's Upfront Lawyer Checklist synthesis, not for critics; form your own judgment from the draft and the knowledge graph only.

draft_write_filenames = {{ plan.drafts.map(d => d.files) }}

Use the graph_read_file tool (or equivalent) to load the file at `./<draft_write_filenames>`

## Critique Output — Write Your Verdict Here, and Nowhere Else

Write your verdict as the Markdown document defined in "Required Output" below to the single path given here:

  critique_write_filenames = critiques/critique-correctness.md

Write the Markdown verdict to `./<critique_write_filenames>` exactly as given. This is the ONLY file you write. Do not touch any other path.

**HARD RULE — reviewer, never editor.** Writing to, modifying, or regenerating the draft file, the canonical deliverable, or any file other than the critique path above is a HARD FAILURE, regardless of how confident you are in a fix. You do not propose remediated text, you do not rewrite sections, you do not call `harvey_generate_docx` or `harvey_generate_xlsx`, and you do not move any file. The aggregator — not you — is the only step permitted to author the deliverable. If you believe an issue is fixable, describe it precisely in `failing_items`; do not fix it yourself.

## Knowledge Graph

Namespace for this task: namespace = {{ input.namespace }}

Every graph tool call **that retrieves this task's ingested source documents** MUST include namespace = {{ input.namespace }}. Never query the default namespace for document retrieval.

**EXCEPTION — Concept nodes are free-floating and have NO task namespace.** When searching for or traversing Concept nodes (the `Law` concept, its practice-area neighbors, `Legal Document Type: <Name>` registry nodes, and `Legal Analysis Skill: <Name>` / `Legal Meta-Skill: <Name>` nodes), scope the query to `namespace=default` instead — never this task's namespace. The MUST-include-task-namespace rule above applies ONLY to retrieval of this task's ingested source documents, never to Concept-node lookups. A Concept lookup mistakenly scoped to `{{ input.namespace }}` returns zero nodes silently.

**CRITICAL — this exception does NOT weaken the closed-world factual-grounding contract in scope item 1.** Concept nodes are never evidence and never grounding. The task namespace remains the sole and exclusive source of truth for whether any asserted fact, figure, reference, or cross-document claim in the draft is accurate; a claim traceable only to a Concept node is still ungrounded and must still be flagged under item 1. Concepts tell you WHICH categories of fact and cross-reference are worth verifying for this document type — they never tell you what this record says, and they can never substitute for a node or passage in this namespace. Any finding informed by a Concept node rather than the namespace is `[basis: practitioner-knowledge]`, never `[basis: source-grounded]`.

## Context

Task goal: {{ input.instructions }}

Required deliverables: {{ input.deliverables }}

---

## Scope: correctness

Question answered: **is what was produced accurate?** You do NOT assess presence/absence of required content (that is Completeness's scope), numerical reconciliation (that is Arithmetic's scope), or methodology/compliance (that is Doctrine's scope). Confine every finding strictly to whether an asserted fact, reference, or cross-document claim is accurate.

Assess across all of the following, folded into this single correctness verdict:

1. Factual grounding — closed-world, strict KG-trace check. The draft is judged solely against the ingested knowledge graph. Every claim must be directly traceable to a node or passage in the namespace. Flag anything not grounded in the KG — even if it would be correct in the real world — because the contract of this check is strict source fidelity: if it is not in the KG, flag it, even if it seems generally true.

2. Stale, terminated, or dangling references — every named plan, program, entity, defined term, or external reference in the draft must still exist and be consistent across ALL source documents in the namespace. Use the knowledge graph to verify each named reference explicitly — do NOT assume it is still operative or accurate merely because it appears in one document. Query all documents. Flag any named reference that cannot be confirmed as current and cross-document-consistent; describe it as "a named [type] reference that cannot be confirmed across source documents" without restating the specific name in the issue field.

3. Cross-document conflicts — identify any material inconsistency or outright conflict between two or more source documents in the namespace — including differing definitions of the same term, conflicting obligations, contradictory representations, or inconsistent benefit or compensation terms. Compare documents pairwise using the knowledge graph. Describe conflicts as "Document A and Document B contain conflicting provisions regarding [generic topic]."

### Epistemic labelling requirement (mandatory on every failing item)

By default, judge every finding against the source documents/knowledge graph in this namespace — closed-world. Only for checks that inherently require outside legal knowledge to resolve (for example, interpreting whether a cross-document conflict is actually material, or whether a stale reference's replacement is legally equivalent) may you reach beyond the KG using open-world practitioner knowledge. Every failing item's Issue field MUST be explicitly labelled with its basis, using exactly one of these two tags at the start of the Issue text:

- `[basis: source-grounded]` — the finding rests entirely on what is or is not present in the KG/source documents in this namespace.
- `[basis: practitioner-knowledge]` — the finding required reasoning beyond the KG using general legal/practitioner knowledge.

An Issue field with no basis label is incomplete — never omit it. This labelling keeps fabrications (which must always be source-grounded findings) distinguishable from legitimate judgement calls.

---

## Required Output

Emit a Markdown file, using the following structure:

## Correctness
Pass/Fail: <Pass|Fail>

- Failing items (omit this list entirely when Pass):
  - Location: <section or field, or "correctness-judgment" for a generically-described defect> — Quote: "<verbatim text>" — Issue: "[basis: source-grounded] <description>" or "[basis: practitioner-knowledge] <description>"

## All Pass
<true|false>

Rules:
- Omit the "Failing items" list entirely when the Pass/Fail line reads Pass.
- Set "All Pass" to true only when the Pass/Fail line reads Pass.
- Every failing item's Issue field MUST begin with either "[basis: source-grounded]" or "[basis: practitioner-knowledge]" — never omit the basis label.
- Do not fabricate failing items; only flag genuine issues found.
- Do NOT reference any rubric, scoring criteria, or evaluation standard.
- Do NOT include any instruction, suggestion, or content aimed at fixing, remediating, or regenerating the draft or the canonical deliverable anywhere in your output — describe the problem only, in the Issue field.
- Do NOT include any Completeness, Arithmetic, or Doctrine content anywhere in your output — those are separate critics' scopes.