# Local desktop setup + speech-to-text (sherpa-onnx)

Context: we want vein to run as a **child process of a native desktop app**
(Swift on macOS, Kotlin on the JVM/Windows/Linux) with the existing web UI
shown in a webview, and we want **speech-to-text** that works the same way
whether vein is that local child process or a server that a mobile app talks
to. This doc covers both: what it takes to package vein for local use, and how
STT lands in vein via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

Status: nothing below is built. Findings are from reading the tree as of
2026-09-04.

---

## 0. Decisions up front

- **STT runs inside vein, not in the host app.** Vein is the process we
  package on every platform, so the recognizer lives there and native clients
  only capture audio and send it. The Swift/Kotlin sherpa bindings are
  reserved for a future offline-only mobile mode.
- **Streaming dictation is the product surface; there is no STT workflow
  step in v1.** Workflows sit *around* the recognizer as the learning loop
  ("dream cycles", §4.8), not in front of it. A batch `audio/transcribe`
  step can come later if a server workflow needs one.
- **Model family: Zipformer/NeMo transducers.** They are the only sherpa
  models that accept hotwords (contextual biasing), which is the mechanism
  the dream cycle drives. Whisper/Moonshine/SenseVoice are out (§4.2).
- **Desktop ships a Node runtime + a bundled `server.cjs` + `web/dist`**, not a
  single-file executable, for the first cut. Single-file (SEA / Bun / Deno
  compile) is a later optimization once the step loader no longer scans
  directories next to the running module (§2.2).
- **Desktop defaults to the filesystem workspace** (`VEIN_WORKSPACE_BACKEND=fs`).
  Graph features need a Neo4j the app doesn't ship (§3).
- **Native addons ship beside the binary, one platform each**, never
  embedded. This applies to `sherpa-onnx-node` and, if we keep it, to
  `onnxruntime-node`.

---

## 1. Topologies we must support

| Topology | Who runs vein | Audio path | Notes |
|---|---|---|---|
| Desktop app | child process, `127.0.0.1:<port>` | host captures mic → local HTTP/WS | webview loads `web/dist` from the same origin |
| Mobile app | remote server | app captures mic → HTTPS/WSS | same routes, different base URL |
| Server workflow (later) | server | files already on disk / artifacts | batch `audio/transcribe` step, not in v1 |
| Offline mobile (later) | none | on-device sherpa binding | out of scope; keep API shape identical |

The only client-side difference between rows 1 and 2 is the base URL and how
the API key is obtained.

---

## 2. Local (desktop) setup

### 2.1 What already works

- Server is Hono on `@hono/node-server`; `listen()` takes a port
  (`createVein.ts` ~L1892).
- `web/dist` is a static Vite bundle. The frontend calls the API through a
  relative `BASE`, so a webview pointed at `http://127.0.0.1:<port>` needs no
  frontend change.
- `VEIN_API_KEY` gating exists (`auth.ts`).
- Heavy deps (AI SDK providers, `@huggingface/transformers`, gdrive/octokit)
  are already `await import()`-ed lazily.

### 2.2 Blockers, in order of importance

1. **Neo4j is a boot dependency by default.** `graph/wiring.ts` verifies bolt
   at startup. Desktop launches with `VEIN_WORKSPACE_BACKEND=fs`. Consequence:
   `graph/*` steps and embeddings are unavailable locally unless the user
   points `NEO4J_*` at a reachable instance (allowed, just not default).
2. **Step discovery scans directories relative to `import.meta.url`**
   (`steps/registry.ts` `LIB_DIR` / `CORE_DIR` + `readdir` + dynamic
   `import()`). This is fine for "Node + bundle dir" packaging as long as the
   step files ship as real files. It breaks any single-file build. Fix when we
   go single-file: a build-time generated step manifest (static imports) with
   `readdir` only for custom steps.
3. **Custom steps are materialized to disk and `import "vein"`**
   (`graph/workspace-store.ts` `materializeCustomSteps`, and the fs workspace's
   `steps/custom`). Node resolves `vein` by walking up from the file. The app
   must ensure a resolvable `node_modules/vein` exists above the materialize
   dir — simplest is to make the data dir live under the app's bundle tree, or
   write a one-line shim package that re-exports from the running server.
4. **`web/dist` is resolved relative to the module** (`createVein.ts` ~L450).
   Add a `VEIN_WEB_DIST` override so the host can pass an absolute path.
5. **Bind address.** `serve({ fetch, port })` binds all interfaces. Add
   `VEIN_HOST` (default unchanged for servers; desktop passes `127.0.0.1`).
6. **Shell steps spawn `bash`** (`shell.ts`). macOS/Linux fine; Windows needs
   either Git-Bash detection or a `VEIN_SHELL` override. Not blocking for a
   macOS-first release.
7. **Native addons** (`onnxruntime-node` 211 MB all-platforms, `sharp` via
   transformers). Drop from the desktop build; MiniLM falls back to the WASM
   backend (§3).

### 2.3 Packaging: phase A (ship this first)

Inside the app bundle (macOS `Contents/Resources/vein/`, similar on others):

```
vein/
  node                     # official Node binary for the platform (~110 MB)
  server.cjs               # esbuild bundle of build/server.js + deps
  steps/                   # build/steps/** as files (registry scans these)
  web/dist/                # Vite output
  native/
    sherpa-onnx-<platform>/  # the one optionalDependency for this OS/arch
```

- esbuild: `platform=node`, `format=cjs`, mark `sherpa-onnx-node` and any
  `.node`-bearing package as `external`, and set `NODE_PATH` (or a small
  resolver shim) so `require("sherpa-onnx-node")` finds `native/`.
- Keep `steps/` as loose files so `LIB_DIR`/`CORE_DIR` scanning keeps working
  with zero code change. `import.meta.url` inside the bundle must resolve to
  the bundle's own dir; esbuild's `--inject` of an `import.meta.url` shim or
  `__dirname` replacement handles this.
- Size estimate: ~150 MB before models. Acceptable.

### 2.4 Packaging: phase B (single binary, later)

Only after §2.2 items 2–4 are done. Candidates: Node SEA (requires CJS entry +
`sea-config` assets), Bun `--compile`, Deno `compile` (already installed on
dev machines). All three still need native addons beside the binary. Do not
start this until phase A is in users' hands.

### 2.5 Host ↔ vein contract

Host spawns `node server.cjs` with env:

| Var | Desktop value |
|---|---|
| `VEIN_HOST` | `127.0.0.1` |
| `VEIN_PORT` | `0` (let the OS pick) |
| `VEIN_WORKSPACE` | app-support dir, e.g. `~/Library/Application Support/<App>/vein` |
| `VEIN_WORKSPACE_BACKEND` | `fs` |
| `VEIN_WEB_DIST` | absolute path to bundled `web/dist` |
| `VEIN_API_KEY` | random per launch, generated by the host |
| `VEIN_SECRET_KEY` | stable per install, stored in the OS keychain by the host |
| `VEIN_MODEL_DIR` | app-support `models/` (shared by MiniLM and sherpa, §4.5) |
| `ANTHROPIC_API_KEY` etc. | from the host's settings UI |

Protocol:

- vein prints one JSON line on stdout when ready:
  `{"event":"ready","port":51234}`. Today `listen()` logs a human string; add
  the structured line (keep the human one).
- Host loads the webview at `http://127.0.0.1:<port>/` and injects the API key
  (e.g. via a `?key=` on first load that the UI stores in `sessionStorage`,
  or a host-set cookie). Decide once; `?key=` is simplest.
- Host kills the child on quit. vein already handles `SIGTERM` via the run
  store's durable resume, so a hard kill is recoverable.
- Health: `GET /health` (add if missing) so the host can detect a crashed
  child and restart it.

Webview notes: WKWebView allows plain HTTP to localhost without ATS
exemptions. Compose Desktop has no built-in webview; use JCEF/KCEF (or JavaFX
WebView). Microphone capture stays in the host, not the webview (§4.6).

---

## 3. Embeddings without Neo4j (brief)

- Inference is not the problem. `@huggingface/transformers` runs MiniLM on
  the WASM backend (`onnxruntime-web`, already a dep) with identical vectors.
  Force `device: "wasm"` in `graph/embeddings.ts` when `onnxruntime-node`
  isn't present. Also make `allowLocalModels` configurable so a bundled model
  under `VEIN_MODEL_DIR` is used offline.
- **Storage is the open decision.** Vectors live in Neo4j vector indexes. With
  the fs backend there is nowhere to write them. Options, cheapest first:
  1. Desktop connects to a remote/Docker Neo4j (zero code).
  2. Small local vector store for the fs backend (sqlite-vec, or flat file +
     brute-force cosine; 384-dim MiniLM is fine to ~50k vectors).
  3. **Embedded graph DB (LadybugDB)** — a real local graph backend, not just
     vectors. See §3.1.
  4. Embed Neo4j + JRE in the app (heavy; last resort).
- Not required for STT. Defer until a desktop feature actually needs search.

### 3.1 LadybugDB as a local graph backend

Idea, not decided. Needs team discussion.

- **What it is.** [LadybugDB](https://github.com/LadybugDB/ladybug) is the
  maintained fork of Kuzu (Kuzu was acquired by Apple and archived late 2025;
  do not build on it). MIT, embedded, single file on disk, Cypher, Node
  bindings (`@ladybugdb/core`, native addon), disk-based HNSW vector index
  and full-text search as loadable extensions.
- **Why it's attractive.** One engine covers the graph workspace store, run
  projection, `graph/*` steps, and vector + fulltext search, all in a file
  under the app-support dir. No JVM, no daemon, no port. Strictly more than
  option 2 for similar packaging cost.
- **Why it's not a driver swap.** The seam is good: every query goes through
  `Bolt` (`graph/bolt.ts`), ~50 Cypher call sites across ~11 files, and the
  `GraphBackend` interface + storage conformance tests already exist. But
  the Cypher is Neo4j-shaped:
  - Ladybug is schema-first (`CREATE NODE TABLE`, typed columns, primary
    key). Vein does schema-less multi-label `MERGE` with labels computed at
    runtime (`node-writer.ts` ~L568: type + `Node` + `Data_Bank` +
    `Domain_*`). Ladybug's multi-label patterns are query-side, not a node
    carrying N labels.
  - `CREATE CONSTRAINT / VECTOR INDEX / FULLTEXT INDEX`, `CALL db.*`, and
    the three `apoc.*` uses all need rewriting to Ladybug's
    `CREATE_VECTOR_INDEX` / `CREATE_FTS_INDEX` and query functions.
  - Vector and FTS are separate shared libraries loaded at runtime. For an
    offline desktop they ship beside the binary, same rule as native addons
    (§0).
  - Variable-length paths, `UNWIND`, params, and most `MATCH`/`RETURN`
    shapes should survive with light edits.
- **Open design question: how much to conform.** Two positions, both live:
  - *Keep the jarvis-shaped logical model* (`node_key` + `namespace`
    identity, `Data_Bank` supertype, `Domain_*` labels as a labels column)
    and translate only inside the backend. Keeps writers/reader/conformance
    tests unchanged and leaves the door open to syncing a desktop graph to a
    swarm's Neo4j later.
  - *Ladybug-native schema.* Cheaper and cleaner if the local graph never
    needs to interoperate with jarvis; some of what the jarvis model carries
    (namespace, supertype labels) exists only for a shared server. It may be
    fine not to conform everything. Cost is a second logical model to
    maintain and no cheap sync path.
  - Lean: undecided. Decide with the team before any code.
- **Suggested order.** Ship phase A on the fs workspace first (graph steps on
  desktop say "connect a swarm"). Then a one-day spike: run vein's Cypher
  strings against Ladybug and count what breaks, which also informs the
  conform-or-not question. Only then a `LadybugBackend` behind
  `GraphBackend`, gated on a desktop feature that actually needs search or
  graph. Rough guess for the backend: two to three weeks.

---

## 4. Speech-to-text via sherpa-onnx

### 4.1 Facts (measured 2026-09-04 on an M-series Mac, Node 24, sherpa-onnx-node 1.13.7)

- Apache-2.0. CPU-only on desktop. npm `sherpa-onnx-node` + one optional
  platform package (`darwin-arm64`, `darwin-x64`, `linux-x64`,
  `linux-arm64`, `win-x64`, `win-ia32`; 23–34 MB each). Native addon; on
  macOS it loads with **no** `DYLD_LIBRARY_PATH` when installed normally.
- Streaming Zipformer (kroko, 57 MB): recognizer builds in ~0.9 s; decodes
  7 s of audio in ~100 ms (RTF ≈ 0.015). Output is cased and punctuated.
  Its exported chunk is 128 frames = **1.28 s**, so partials arrive every
  ~1.3 s — accurate but not "live". Chunk size is baked into the export.
- Streaming Zipformer 20M (2023-02, 128 MB): partials every **320 ms**, but
  uppercase, no punctuation, and noticeably worse accuracy on the same
  clip. Not worth shipping.
- NeMo streaming fast-conformer transducers (int8 ≈ 105 MB) measured on
  the same clip: **480 ms** variant → partials every ~570 ms, decode
  220 ms / 7 s; **80 ms** variant → partials every ~150–200 ms, decode
  560 ms / 7 s (RTF 0.08). Lowercase, no punctuation, accuracy on par with
  kroko ("stack work", "fink swarm", "vain work flow"). Load ≈ 0.4 s.
- **sherpa's NeMo online transducer is greedy-only** ("Unsupported decoding
  method: modified_beam_search"), so the NeMo streaming models cannot take
  hotwords. Fast partials and biased decoding come from different model
  families — hence the two-recognizer stream in §4.4.
- Nemotron-speech streaming 0.6B, 80 ms variant (int8, 463 MB): the
  accuracy ceiling — cased, punctuated, got "Sphinx" right unbiased — but
  decode is 3.9 s / 7 s on two threads (RTF ≈ 0.55), load ≈ 0.9 s, and it
  is the same NeMo implementation: **greedy-only, no hotwords**. An option
  for fast desktops that want a single recognizer without biasing, not the
  default.
- Bundles its own `libonnxruntime`. Loading it alongside `onnxruntime-node`
  in one process risks symbol clashes on Linux. Another reason to run
  MiniLM on WASM (§3) and drop `onnxruntime-node` from desktop builds.
- Models are separate `.tar.bz2` downloads. Node has no bzip2; extract by
  spawning `tar xjf` (bsdtar on macOS/Windows 10+, GNU tar on Linux all
  handle bz2).
- **Tail flush:** after the last audio the stream must be fed ~2 s of
  silence before `inputFinished()`, or the final chunk's words are lost
  (observed: "post the re" vs "post the results to hive").
- **A stream's sample rate must never change.** sherpa `exit(-1)`s the whole
  process on "You changed the input sampling rate" — no exception to catch.
  The service pads the flush at the stream's own rate and rejects a
  mid-stream rate change before it reaches the addon.
- **End to end over the WebSocket** (two-recognizer default, 100 ms frames
  sent at real-time pace): partials arrive 50–100 ms behind the audio
  position; the final lands ~300 ms after `end`; wall time ≈ audio length.
  First connection pays ~1.3 s for recognizer construction, later ones ~3 ms.

### 4.2 Hotwords — the mechanism the dream cycle drives

- Transducer-only, `modified_beam_search` only (greedy ignores them).
  Cost: decode went 80 → ~100 ms on the 7 s clip. Negligible.
- **`modelingUnit` must be `"bpe"`.** Unset, sherpa defaults to `cjkchar`
  and silently splits each hotword into single characters, which the model
  never emits — hotwords then have no effect at any score. This cost an
  hour; do not repeat it.
- `bpe` mode needs a `bpe.vocab` (sentencepiece vocab, `piece\tscore`).
  The released models don't ship one. **Synthesize it from `tokens.txt`**:
  every non-special token with a constant score of `-1`; sherpa's unigram
  encoder then picks the fewest-pieces segmentation. Verified: with this
  vocab, hotwords `sphinx`/`Sphinx` turned "on this FHIC swarm" into
  "on the Sphinx swarm" at score 1.5–3.
- Score is per matched token. 1.5–3 works; 5 over-biases ("Open the
  Stak | Graph, then …"). Per-phrase `:score` suffix is supported; the
  dream cycle should set it per term rather than one global number.
- Casing must match what the model would emit (kroko is cased: "Jarvis"
  came out capitalized unprompted, unknown names came out lowercase).
  Emit both forms for proper nouns.
- Limits: a hotword only helps when the acoustic path is already in the
  beam. "Stakwork" stayed "stackwork" at every score — the model hears
  "stack work" and the boosted token path is a different spelling. That is
  a post-correction case (§4.8 step 4), not a hotwords case.

### 4.3 Audio contract

- Engine input is 16 kHz mono float32. sherpa resamples other rates itself
  (verified with a 24 kHz WAV), so accept:
  - WAV files (any rate, 8/16/24-bit PCM) for the batch route;
  - raw PCM16LE frames with a declared `sampleRate` for streaming (what
    native mic capture produces cheaply on every platform).
- Not accepted in v1: compressed formats. Clients decode natively.

### 4.4 Surface

Everything lives in `src/audio/` and is exposed as `ctx.services.stt?`
(optional, like `artifacts`) so future steps can transcribe without
importing sherpa. All `sherpa-onnx-node` imports are lazy; without the
addon the routes return `501 { error: "stt not available" }` and vein boots
and passes every other test.

**`GET /audio/stream`** (WebSocket, `@hono/node-ws`) — live dictation:

- Client: `{"type":"start","model":"<id>","sampleRate":16000,"hotwords":"<list name>"|["phrase", …],"session":"<id>"}`
  then binary PCM16LE frames, then `{"type":"end"}`.
- Server: `{"type":"ready"}`, `{"type":"partial","text"}` on every change,
  `{"type":"final","text","words":[{w,start}]}` on endpoint detection and on
  `end`, `{"type":"error","error"}`.
- **Two recognizers per stream when `partialModel` is set**: the fast
  greedy NeMo model produces the partials, the hotword-capable Zipformer
  produces the finals and owns endpoint detection (both are reset on its
  endpoint). Partials feel live (~150 ms); finals are biased, cased and
  punctuated. Combined RTF ≈ 0.1 on M-series. With `partialModel` unset
  one recognizer does both.
- One recognizer per (model, hotwords hash, score, endpoint rules), cached;
  streams are cheap.
  Decoding is synchronous native work on the event loop, ~1–2 ms per
  100 ms frame on M-series; move to a worker thread only if a slow target
  shows it.
- Auth: `Authorization: Bearer` **or** `?key=` (a webview's WebSocket
  cannot set headers). HTTP routes below use the normal bearer middleware.
- This route needs vein's own `listen()` (WebSocket upgrade is injected
  into the Node server). It does not work through mcp's Express bridge,
  which is fine: mcp is not the desktop process.

**`POST /audio/transcribe`** — raw `audio/wav` body, `?model=`,
`?hotwords=`. Same output as a `final`. Push-to-talk and voice memos.

**`GET /audio/models`** — catalog entries with `installed`, size, chunk
latency, language. **`POST /audio/models/:id/download`** — SSE progress
(bytes, then extracting, then done); sha256 verified from the catalog.

**Sessions and hotword lists** (the dream-cycle seam, §4.8):
- `GET /audio/sessions`, `GET /audio/sessions/:id` — finals per session as
  JSON, written as `<dataDir>/audio/sessions/<id>.jsonl` by the stream.
- `POST /audio/sessions/:id/corrections { index, text }` — the user's edit
  of a final. Highest-signal training data.
- `GET/PUT/DELETE /audio/hotwords/:name` — a named phrase list (one per
  line, optional `:score`), stored under `<dataDir>/audio/hotwords/`. The
  stream's `start.hotwords` names one.

### 4.5 Model catalog (`src/audio/models.ts`)

id → GitHub release URL, sha256, size, file layout, chunk latency, casing.

| id | Size | Partials every | Hotwords | Notes |
|---|---|---|---|---|
| `zipformer-en-kroko` | 57 MB | 1.3 s | yes | cased + punctuated; finals model |
| `nemo-fast-conformer-en-80ms` | 103 MB | ~0.15 s | no (greedy-only) | partials model |
| `nemo-fast-conformer-en-480ms` | 106 MB | ~0.57 s | no (greedy-only) | middle ground if 80 ms costs too much CPU on a weak target |
| `nemotron-speech-en-80ms` | 463 MB | ~0.15 s | no (greedy-only) | accuracy ceiling; RTF ≈ 0.55 on 2 threads, cased + punctuated |

Defaults: `model` = `zipformer-en-kroko`, `partialModel` =
`nemo-fast-conformer-en-80ms`. Env `VEIN_STT_MODEL` / `VEIN_STT_PARTIAL_MODEL`
and per-call overrides. Dropped: the 2023-02 20M Zipformer (fast but weak).
`VEIN_MODEL_DIR` (default `~/.cache/vein-models`, alias of the existing
`VEIN_MODEL_CACHE`) holds `stt/<id>/`. Downloads happen on first use or via
the route, never at boot; server images pre-bake.

### 4.6 Client responsibilities (Swift / Kotlin)

- Capture the microphone natively (AVAudioEngine / AudioRecord), 16 kHz
  mono PCM16LE. Do **not** use `getUserMedia` inside the webview.
- Live: open `/audio/stream`, send ~100 ms frames, render partials, replace
  with finals. Push-to-talk: buffer, wrap as WAV, `POST /audio/transcribe`.
- Show the user's finals editable; send edits as corrections.

### 4.7 Testing

- Unit: catalog resolution, hotwords compile (bpe.vocab synthesis +
  file layout), the WebSocket protocol over a fake engine. No addon needed.
- Live (opt-in, `VEIN_TEST_STT=1`, like `VEIN_TEST_NEO4J_URI`): download
  kroko once into a temp model dir, stream its bundled `test_wavs/0.wav`
  as PCM frames, assert the final; then the same with a hotwords list.
  Skipped when the addon is missing.

### 4.8 Dream cycles — how recognition evolves with a user or company

The recognizer never retrains. What evolves is the **hotwords list** (an
environment artifact in `EVOLVE_SPEC.md` terms — layer 2: versioned,
reviewable, changes what every later run sees) and, optionally, a
correction glossary. Workflows produce both; nothing here is a new step type.

1. **Capture.** The stream logs every final to the session file; the UI
   sends the user's edits as corrections. Corrections say exactly which
   words the model gets wrong.
2. **Dream.** A scheduled workflow reads recent sessions and corrections,
   optionally company sources via the existing `gdrive/*`, `slack/*`,
   `github/*` steps, and an `llm` step extracts the lingo: names, product
   terms, acronyms, each with a suggested boost and both casings. It writes
   the list with an `http` step to `PUT /audio/hotwords/<name>`.
3. **Apply.** The next stream that names the list gets a recognizer built
   with it. Partials are biased directly; no latency cost.
4. **Correct.** For misses hotwords can't fix (the "stackwork" case), a
   glossary of `wrong → right` pairs applied to finals only, either as a
   string replacement or a small fast LLM pass. Finals can afford 100 ms.
5. **Measure.** Corrections per hundred words per list version. The eval
   substrate already knows how to score a versioned artifact; a list that
   makes things worse rolls back like a workflow version.

Open: whether the glossary lives in the same file as the hotwords (one
artifact per user/company) or beside it. Lean: same artifact, two sections.

## 5. Order of work

1. **STT core** (in progress on `vein-stt`): `src/audio/` service, catalog,
   hotwords compiler, `/audio/stream` WebSocket, `POST /audio/transcribe`,
   `/audio/models` + download, sessions + hotword lists. Testable on a
   server with no desktop work at all.
2. **Model bake-off**: measure the NeMo / Nemotron streaming variants for
   partial latency and accuracy on the same clips; pick the default.
3. **Server prerequisites for desktop**: `VEIN_HOST`, `VEIN_WEB_DIST`,
   structured `ready` line, `?key=` handoff in the UI, `VEIN_MODEL_DIR`
   alias (the last one lands with step 1).
4. **First dream cycle**: a sessions → llm → `PUT /audio/hotwords` workflow
   plus the corrections UI. Proves the loop before packaging.
5. **Phase A packaging**: esbuild bundle, Node binary, native dir, a macOS
   host proof-of-concept that spawns vein and streams the mic.
6. **Kotlin host**, Windows shell override.
7. Later: single-binary (phase B), local vector store or LadybugDB backend
   (§3, §3.1), batch `audio/transcribe` step, offline-mobile bindings.

## 6. Open questions

- Hotwords per user vs per company: lists are named, so both exist; the
  open part is whether a stream can name several and how they merge.
- Non-English dictation: kroko ships de/es/fr streaming variants and
  Nemotron 3.5 is multilingual. Catalog entries only; nothing else changes.
- Speaker diarization and VAD: sherpa ships silero-VAD and speaker
  embedding models. Not in v1; the streaming route's endpointing is enough.
- Decoding on the event loop vs a worker thread: fine on M-series; decide
  after measuring an older Intel laptop and Windows.
- LadybugDB local graph (§3.1): do it at all, and if so, jarvis-shaped
  logical model vs. Ladybug-native schema. Team discussion pending.

---

## 7. Decision rules

- Transcription logic lives in `src/audio/`, exposed as a service; routes
  are thin. No STT step in v1; workflows learn around the recognizer.
- Hotwords need `modelingUnit: "bpe"` + a synthesized `bpe.vocab`. Always.
- Everything sherpa is lazy-imported and optional. A vein without the addon
  must boot and run every non-audio test.
- Native addons are never embedded; ship one platform dir beside the binary.
- Clients capture audio, vein recognizes it. No second STT implementation in
  the host apps until offline mobile is actually scheduled.
