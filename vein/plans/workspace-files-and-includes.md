# Workspace files + `@@include` — first-class prompt/asset storage

## Problem: the big-prompt problem

A workflow is published **by content** — one YAML string through
`publishWorkflowByContent` (and, for the AI builder, through the
`meta/publish-workflow` tool). That shape breaks down exactly where the lab
is heading: workflows whose step configs embed large verbatim prompts.

The concrete evidence is the harvey-deliver port (`mcp/src/lab/harvey/`):
14 production prompts, 8–73KB each, ~330KB total, that must land in step
configs **byte-for-byte** (the whole point of the port). Three stacked
failures make that impossible for the authoring agent today:

1. **Emission ceiling.** A tool-call argument is *output tokens*. Publishing
   a workflow containing a 70KB prompt means the model must emit 70KB
   verbatim in a single generation — at or past practical output limits.
   (Same failure mode that forced `markdownPath` onto `generate_docx` as an
   alternative to inline `markdown`.)
2. **Verbatim-copy corruption.** Even when the text arrives cheaply as
   *input* (the user pastes it into chat), the agent must copy it back out
   into a publish call. LLMs are unreliable long-verbatim copiers; a dropped
   line or silently "normalized" phrase degrades a production prompt with
   nothing diffing it against the original.
3. **Edits re-pay everything.** Changing one sentence of one prompt means
   re-emitting the entire multi-hundred-KB workflow YAML — per attempt. This
   is also what makes prompt *evolution* intractable at these sizes: an
   evolve-loop author burns its whole budget copying, not improving.

The harvey port dodged all three only because it was authored by an agent
with a filesystem: prompts live as `.md` files in `harvey/prompts/`, and a
**private ~25-line hack** in `mcp/src/lab/harvey/seed.ts` (`expandIncludes`)
splices them into the YAML at seed time via `@@include(FILE.md)` marker
lines. That hack is seed-side only: the engine, the UI, `meta/*`, and the
chat agent know nothing about it, and a UI edit to a seeded harvey workflow
edits the expanded blob (and is clobbered back at next boot). This plan
promotes the hack into the engine and deletes it.

## Design overview

Two pieces, both inside vein core:

1. A **`files/` area in the workspace** — versioned text assets, sibling to
   `workflows/` and `steps/`, with the same content-hash publish semantics.
2. **`@@include` expansion inside `publishWorkflowByContent`** — any
   published workflow may reference workspace files by marker; the engine
   splices them in at publish time. Every publisher (seeders, HTTP/UI,
   `meta/publish-workflow`, evolve authors) gets the mechanism for free.

The runner, registry, resolver, and versioning stay untouched: a published
workflow version is still one self-contained expanded YAML.

## 1. The `files/` area

Layout mirrors the existing steps conventions:

```
<workspace>/files/<path>                    # active content (what expansion reads)
<workspace>/files/_history/<path>/<vid>     # every version, immutable
<workspace>/files/_metadata.json            # { files: { <path>: { active, publisher?, versions: { vid: { hash, createdAt, description? } } } } }
```

- `<path>` is a relative path, nesting allowed (`prompts/HARVEY_DRAFT.md`),
  no `..`/absolute segments (same containment guard as `artifacts/dir`).
- **Text only** (utf-8) in v1, with a size cap (default 1MB/file) — this is
  a prompt/config store, not a blob store. Binary assets are out of scope.
- Content-hash versioned exactly like `publishStep`: republishing identical
  content is a no-op; identical-to-older re-activates it; changed content
  writes the next `vN`. `publisher` stamping follows the same rule as
  workflows (only applied when a version is actually written) so the meta
  surface's ai-stamp discipline extends to files: seeded files are
  unstamped and agent-published files are `publisher: "ai"`.

### WorkspaceManager API

```ts
publishFile(path, content, description?, publisher?): { version, changed }
appendFileDraft(path, chunk): { bytes }        // accumulate into files/_drafts/<path>
publishFileDraft(path, description?, publisher?): { version, changed }  // promote draft → version, clear draft
getFile(path, opts?: { version?, head? }): { content, version, bytes }
listFiles(filter?: { publisher?, prefix? }): [{ path, version, bytes, publisher }]
```

`appendFileDraft` is the load-bearing addition for the agent: large content
is built across **many small tool calls** into a draft, then promoted once —
no single call ever carries the whole file, and no version spam from
per-append publishes. (`putFile` alone would re-create the emission
ceiling.) Drafts are workspace-local scratch: not versioned, not readable by
expansion, cleared on promote.

No delete in v1 (matches workflows/steps — versions are retained; an
unused file is inert).

## 2. `@@include` expansion at publish time

`publishWorkflowByContent` gains one pass **before hashing**:

- A marker is a **full line** matching `^([ \t]*)@@include\(([^)]+)\)\s*$`.
  The captured path resolves against the workspace `files/` area (active
  version). Markers mid-line are not markers (no inline splicing).
- The file's content replaces the marker line, every non-empty line prefixed
  with the marker's own indentation; empty lines stay truly empty (no
  trailing whitespace). This is exactly the proven seed.ts algorithm — it
  makes a marker inside a YAML literal block (`prompt: |`) splice a
  multi-KB body in as valid YAML.
- **Missing file → the publish fails loudly.** A silently-unexpanded marker
  would seed a workflow whose prompt is the literal string
  `@@include(...)`.
- **No recursion in v1**: included content is not re-scanned for markers.
  Termination is trivial and one level covers every real use (prompt files
  are leaves).

### Versioning semantics: the hash covers the EXPANDED text

The stored `vN.yaml` is the expanded YAML — runtime truth, self-contained,
runnable even if the file store is later mangled. Consequences, all
deliberate:

- Editing a prompt file does **not** silently change what any pinned or
  active workflow version runs. Workflows pick up the new prompt on their
  next publish — explicit, diffable, consistent with vein's whole
  content-hash model. (The alternative — load-time expansion — makes "which
  prompt did this run actually use?" unanswerable from the version id.)
- Boot seeding keeps working unchanged: seeder publishes raw-with-markers →
  engine expands → hash differs iff a prompt file or the skeleton changed →
  re-seed. Same reconcile behavior the harvey hack has today.

### Round-tripping: keep the raw source alongside

If `get-workflow` returned only the expanded 200KB blob, the agent's *edit*
loop would re-inherit the emission problem. So when expansion changed
anything, the publish stores both:

```
workflows/<name>/<vid>.yaml       # expanded — what loadWorkflow/runner read (unchanged code path)
workflows/<name>/<vid>.src.yaml   # the pre-expansion source with markers
```

`getWorkflowYaml` / `meta/get-workflow` return the **src** form when it
exists (opt into `expanded: true` for the runtime text). The agent reads and
re-emits only the small skeleton; the version's `_metadata.json` entry
records the expansion provenance:

```json
"v3": { "hash": "…", "includes": { "prompts/HARVEY_DRAFT.md": "v2", … } }
```

`includes` answers "which workflows embed which file version" — the query
behind a future `republish workflows using file X` convenience and behind
honest evolve-loop lineage (a generation that only bumped a prompt file is
visible as exactly that).

## 3. Surfaces

- **meta tools** (granted to the lab chat agent alongside the existing
  authoring family): `meta/save-file` (small files, one call),
  `meta/append-file` + `meta/publish-file` (the draft path for big ones),
  `meta/get-file` (with `head` — read the first N chars for orientation
  without hauling 70KB into context), `meta/list-files`. Same ai-stamp
  rules as `meta/publish-workflow`.
- **HTTP routes** for the UI: list/get/put under `/files/*`. UI can grow a
  files tab later; not required for v1.
- **Chat attachments → files** (follow-up, its own small plan): a file
  attached to the lab chat lands directly in `files/_drafts/` verbatim —
  the zero-copy path where prompt text never passes through the model at
  all. V1 works without it (paste → append-file chunks), but this is the
  end state that fully kills failure mode #2.

## 4. Consumers / migration

- **harvey seed**: delete `expandIncludes` from `mcp/src/lab/harvey/seed.ts`.
  `seedHarveyWorkflows` first publishes `harvey/prompts/*.md` via
  `publishFile` (unstamped), then publishes the raw marker YAMLs; the engine
  expands. Seed ORDER matters: files before workflows (a missing include is
  a loud publish failure by design). `copy-lab-assets.mjs` keeps shipping
  the `prompts/` dir.
- **UI editors** get real diffs: a prompt tweak is a small file edit + a
  skeleton republish, not a blob rewrite.
- **Evolve loops**: an author agent publishes a new prompt-file version and
  republishes one small workflow — prompt evolution at harvey sizes becomes
  affordable per generation.
- **Authoring docs**: add the files/include surface (and the "prompts with
  runtime `{{ }}` templates belong in step CONFIG, not params — resolution
  is single-pass" rule the harvey port established) to the builder's docs.

## Non-goals (v1)

- Run-time / load-time expansion (see versioning rationale above).
- Includes in step SOURCE files (steps are code; imports already exist).
- Binary assets, and any overlap with the per-RUN `artifacts` capability —
  artifacts are what runs *produce*; workspace files are what definitions
  *reference*. Design-time vs run-time twins, kept separate.
- Cross-workspace file sharing / library files.

## Open questions

- Auto-republish dependents when a file changes (`includes` provenance makes
  it cheap to offer) — or keep republish explicit? Leaning explicit; an
  auto-cascade re-activating N workflows on one file save is surprising.
- Should `.src.yaml` be stored even when expansion is a no-op (uniformity)
  vs only-when-different (space)? Leaning only-when-different.
- Draft concurrency: two agents appending to the same draft path — last v1
  answer is "don't" (drafts are single-writer scratch); revisit if it bites.

## Implementation steps

1. `WorkspaceManager`: `files/` area (publish/append-draft/promote/get/list,
   metadata, history, containment + size guards). Unit tests mirror the
   `publishStep` suite (idempotence, reactivation, publisher stamping).
2. Expansion in `publishWorkflowByContent`: marker scan → files lookup →
   indent-aware splice → hash expanded → store `vid.yaml` +
   `vid.src.yaml` + `includes` provenance; loud failure on missing file.
   Tests: literal-block splice fidelity (the seed.ts cases), missing file,
   no-marker passthrough (zero behavior change), src round-trip via
   `getWorkflowYaml`.
3. `meta/*` file tools + HTTP routes; grant to the lab chat agent; extend
   the ai-stamp tests.
4. Migrate harvey seeding off `expandIncludes`; deliver-smoke keeps its
   "prompts spliced, no unexpanded markers" assertions (they should pass
   unmodified — same observable behavior, different owner).
5. Docs: vein AGENTS.md + authoring/builder docs; note in lab AGENTS.md
   that the harvey prompts are now workspace files.
