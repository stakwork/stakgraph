# Concept Lifecycle & Retrieval Improvements

Status: design notes, not yet implemented.

## Problem

The concept corpus only grows. The per-PR decision loop's action space is
`create_new | add_to_existing | ignore | update_description` — there is no merge,
deprecate, supersede, or concept-to-concept relation anywhere in the schema. At
dozens of repos and hundreds of concepts this produces:

- **Redundant concepts** — the writer can't spot the right existing concept, so it creates a new one.
- **Dead experiments** — concepts whose code no longer exists still rank in retrieval.
- **Overlapping variants** — e.g. Hive's three LLM coding assistants (AISDK/in-app,
  planner/external-server, Goose/cloud-sandbox) with nothing telling an agent which is which.
- **Wrong-place work** — agents retrieve a plausible-but-wrong concept and do accurate
  work in the wrong location.

Root cause is structural: redundancy/supersession/death are **corpus-level** properties,
invisible to any per-PR decision. They need a separate periodic consolidation process,
plus schema support for lifecycle and relations.

A second, compounding cause: `builder.ts` loads **all current concepts** into the
create-vs-update decision prompt. At hundreds of concepts the LLM misses the right
match in the list, creates a duplicate, the list grows, and the next miss gets more
likely. The writer suffers the same retrieval problem as the readers.

---

## Workstream 1: Fix the writer's own retrieval (highest leverage, smallest change)

**Where:** `builder.ts` context assembly (currently passes the full concept list to the decision prompt).

Replace "all concepts" with a **top-k shortlist** (k ≈ 15–25) of candidate concepts:

1. **File overlap** (near ground truth): which concepts `MODIFIES` the files this PR touches.
   One Cypher query over existing `(:Concept)-[:MODIFIES]->(:File)` edges.
2. **Embedding similarity**: PR title+body embedding vs. concept description embeddings.
   Requires giving concepts embeddings (see Workstream 2 prerequisites) — today only clues have them.
3. Union both lists, dedupe, cap at k. Always include concepts from the same repo touched recently.

Also add a `merge_into` hint field to `LLMDecision` (types.ts): when the decision LLM
notices two shortlisted concepts are obviously the same thing, it can flag the pair as a
merge **candidate** (not execute the merge — that's the consolidation job's call).

---

## Workstream 2: Schema additions

All additive; migration is trivial (default `status: "active"`, empty arrays).

### Concept fields (types.ts `Concept` interface)

```typescript
status: "active" | "experimental" | "deprecated" | "dead";
statusReason?: string;          // evidence: "no PR touches since 2026-01, 4/6 linked files deleted"
aliases: string[];              // org jargon that refers to this concept ("Falcon", "the rollup")
discriminator?: string;         // one sentence: how this differs from its siblings/near-matches
embedding?: number[];           // of name + description (+ aliases); concepts, not just clues
altitude: "product" | "capability" | "feature" | "implementation";
```

### Concept→Concept edges (new, Neo4j)

```
(:Concept)-[:PART_OF]->(:Concept)      // hierarchy: feature → capability → product
(:Concept)-[:VARIANT_OF]->(:Concept)   // siblings solving the same problem differently
(:Concept)-[:SUPERSEDES]->(:Concept)   // new implementation replaced old one
```

### Merge semantics

Merges must preserve provenance:
- Union `prNumbers`, `commitShas`, clues; keep earliest `createdAt`.
- Keep the losing concept's `id` as a redirect (e.g. a `mergedInto` property on a tombstone
  node, or an alias entry) so existing links/bookmarks resolve.
- Losing concept's name becomes an alias of the winner.

---

## Workstream 3: Consolidation job (nightly/weekly, corpus-level)

New endpoint (e.g. `POST /gitree/consolidate?repo=...`), runnable per-repo or org-wide.
Three passes. Candidate generation is **mechanical** (cheap, no LLM); only adjudication
spends tokens.

### Pass A: Merge / relate

1. Candidate pairs, generated mechanically:
   - description-embedding cosine > τ (start ~0.85), OR
   - file-set Jaccard overlap > τ' (start ~0.4) via `MODIFIES` edges, OR
   - shared `prNumbers` count ≥ 2.
2. LLM adjudicator per pair (or per connected cluster of pairs), given both concepts'
   full docs + evidence (PR lists, file lists, last-touch dates). Verdicts:
   - `merge` → execute merge semantics above.
   - `supersedes` → create edge, set loser `status: "deprecated"`.
   - `variant_of` → create edge, **write a discriminator sentence into BOTH concepts**
     ("unlike the Goose sandbox runner, this assistant runs in-app via AISDK…").
   - `distinct` → record the pair as adjudicated (don't re-ask every run); still write
     discriminators if the names/descriptions are confusable.

Discriminators are the point: retrieval failures between near-matches are fixed by
contrastive sentences, not longer descriptions.

### Pass B: Staleness

For each concept:
1. No PR/commit touches in N months (`lastUpdated`, `prNumbers`)?
2. Do its linked files still exist in HEAD (via `MODIFIES` edges + current graph)?
3. Mostly deleted → `status: "dead"`. Present but untouched → `status: "deprecated"` candidate
   (LLM confirms — some code is just stable, not dead; low churn + still imported = active).
4. Always set `statusReason` with the evidence.

Dead/deprecated concepts stay retrievable when explicitly asked, but are heavily
down-weighted in default ranking.

### Pass C: Hierarchy

Cluster concepts (embedding + shared-file + shared-repo signals) into parent
**capability** concepts; create `PART_OF` edges and set `altitude`. E.g. parent
"LLM coding assistants" (capability, Hive) with three `VARIANT_OF` children.
LLM proposes cluster names; run rarely (weekly/monthly) — hierarchy churn is confusing.

---

## Workstream 4: Retrieval redesign (read path)

Replace flat `getAllConcepts` (sorted by `lastUpdated`) with a two-stage read:

### Stage 1: cards

`GET /gitree/concepts/search?q=...&altitude=...&limit=...` returns lightweight cards:

```
{ id, name, oneLineDescription, status, discriminator, repo, altitude, score }
```

Ranking: `embedding_similarity × recency_decay(lastUpdated) × status_weight`
(status_weight: active 1.0, experimental 0.7, deprecated 0.2, dead 0.05).
Also do exact/fuzzy **alias matching** before embedding search — an alias hit is
near-certain intent and should rank first.

### Stage 2: full fetch

Existing `GET /gitree/concepts/:id` (docs + clues) only for the concept the agent picks.
Context stays small regardless of corpus size.

### Disambiguation as a first-class response

When top results are `VARIANT_OF` siblings (or near-tied scores under one parent),
return the parent + discriminators instead of N full concepts:

```json
{
  "disambiguation": true,
  "parent": "hive/llm-coding-assistants",
  "options": [
    { "id": "...", "discriminator": "in-app via AISDK/Vercel, receives full org context (Jamie)" },
    { "id": "...", "discriminator": "planner agent, structured workflows, external server" },
    { "id": "...", "discriminator": "cloud sandbox, clones repos, Goose agent, simple tasks" }
  ]
}
```

An agent with task context can almost always pick correctly. This single behavior is
the biggest fix for "accurate work in the wrong place."

### Altitude-aware retrieval for the agent hierarchy

Same corpus, filtered views by consumer tier:
- **Jamie** (high-level): `altitude=capability|product` — parents, summaries.
- **Planner**: `altitude=feature` — repos, entry points, discriminators.
- **Goose** (implementation): `altitude=implementation` + clues + file paths.

---

## Workstream 5: Usage logging (capture from day one, use later)

Log every retrieval: query text, cards returned, concept the agent expanded, and —
where observable — the files the agent subsequently touched vs. the concept's file set.

This yields, for free:
- **Ranking labels** (retrieved-and-used vs. retrieved-and-ignored).
- **Merge candidates** (always co-retrieved, never both used).
- **Training pairs** (utterance → concept/location) for a future fine-tuned retriever
  (small BGE-class embedding model, retrainable nightly on org jargon).

Deliberately deferred: don't build a learned retriever until the corpus is governed
(Workstreams 2–3). The current failure is ungoverned corpus, not embedding quality.

---

## Suggested build order

1. **W2 schema fields + concept embeddings** (prereq for everything).
2. **W1 writer shortlist** (stops the bleeding — duplicate creation rate drops immediately).
3. **W4 stage-1/stage-2 retrieval + alias match + status weighting.**
4. **W3 pass B (staleness)** — cheapest pass, immediately de-ranks dead experiments.
5. **W3 pass A (merge/relate + discriminators).**
6. **W4 disambiguation response + altitude filters.**
7. **W3 pass C (hierarchy)** and **W5 logging** (W5 can start anytime; it's just logging).

## Success metrics

- Duplicate-creation rate: % of new concepts later merged by Pass A (should trend → 0).
- Retrieval precision@3 against a small hand-labeled set of real agent queries
  (include jargon queries: "Falcon", "the rollup", "the coding assistant").
- Wrong-place rate: agent sessions whose touched files ⊄ retrieved concept's file
  neighborhood (proxy, from W5 logs).
- Corpus health: active vs. deprecated/dead ratio; % of confusable pairs with discriminators.
