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

- **STT runs inside vein, not in the host app.** One implementation serves
  every topology (desktop-local, mobile→server, server-side workflows). Native
  clients only capture audio and send it. The Swift/Kotlin sherpa bindings are
  reserved for a future offline-only mobile mode, and even then the client must
  present the same API shape so callers can't tell local from remote.
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
| Server workflow | server | files already on disk / artifacts | `audio/transcribe` step |
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
  3. Embed Neo4j + JRE in the app (heavy; last resort).
- Not required for STT. Defer until a desktop feature actually needs search.

---

## 4. Speech-to-text via sherpa-onnx

### 4.1 Facts

- Apache-2.0. CPU-only on desktop (GPU is Jetson/CUDA-Linux only).
- npm: `sherpa-onnx-node` (currently 1.13.7) with one optional platform
  package each: `darwin-arm64`, `darwin-x64`, `linux-x64`, `linux-arm64`,
  `win-x64`, `win-ia32`; 23–34 MB unpacked each. Native addon.
- Both **offline** (whole-file) and **streaming** (partial results)
  recognizers. Model families: Whisper, SenseVoice, Zipformer (streaming +
  offline), Moonshine, Parakeet (NeMo TDT), Paraformer.
- Bundles its own `libonnxruntime`. Loading it alongside `onnxruntime-node`
  in one process risks symbol clashes on Linux. Another reason to run MiniLM
  on WASM (§3) and drop `onnxruntime-node` from desktop builds.
- Models are separate downloads (tens to hundreds of MB).

### 4.2 Dependency layout in vein

- Add `sherpa-onnx-node` as an **optionalDependency**. Import lazily
  (`await import("sherpa-onnx-node")`) inside the step's `run()` and the
  route handlers, matching the existing lazy-SDK pattern. If the import
  fails, routes return `501 { error: "stt not available" }` and the step
  fails with a clear message. Server installs without it still boot.
- New module: `src/audio/stt.ts` — owns recognizer construction, model
  resolution, and a tiny session registry for streaming. Steps and routes
  both go through it (AGENTS.md "step vs service": this is a service).
- Expose it as `ctx.services.stt?` on `VeinCapabilities` (optional, like
  `artifacts`) so custom steps can transcribe without importing sherpa.

### 4.3 Audio contract

- Input to the engine is **16 kHz mono float32 PCM**. Accept:
  - WAV files (any rate; sherpa's `readWave` handles 8/16/24-bit PCM; we
    resample to 16 kHz if needed via sherpa's built-in resampler).
  - Raw PCM16LE at 16 kHz for streaming frames (what native mic capture
    produces cheaply on every platform).
- Not accepted in v1: compressed formats (m4a/opus/mp3). Clients decode
  natively before sending. Revisit if mobile upload sizes hurt.

### 4.4 Surface

**Step** `audio/transcribe` (`src/steps/lib/audio/transcribe.ts`):

```yaml
- id: words
  type: audio/transcribe
  config:
    path: "{{ steps.download.output.path }}"   # WAV on disk or an artifact ref
    model: parakeet-tdt-0.6b-v3               # optional; default from settings
    language: auto                            # for multilingual models
```

Output: `{ text, segments: [{start, end, text}], language?, model, durationMs }`.
Segments come from the recognizer's timestamps where the model provides them
(Whisper/Parakeet/SenseVoice do; Zipformer streaming gives word times).

**Route** `POST /audio/transcribe` — multipart or raw `audio/wav` body,
optional `model`/`language` query. Same output as the step. Gated by
`VEIN_API_KEY` like everything else. This is what the desktop and mobile apps
call for push-to-talk and voice memos.

**Route** `GET /audio/stream` (WebSocket) — live dictation:

- Client sends `{"type":"start","model":"zipformer-streaming-en","sampleRate":16000}`
  then binary frames of PCM16LE, then `{"type":"end"}`.
- Server sends `{"type":"partial","text":...}` as the streaming recognizer
  updates and `{"type":"final","text":...,"segments":[...]}` on endpoint
  detection or `end`.
- One recognizer stream per socket; recognizer instances are shared per
  model and reused (they're expensive to construct; streams are cheap).
- Needs `@hono/node-ws`. This is vein's first client-to-server streaming
  route; keep it isolated in `src/audio/ws.ts` so the SSE-based rest of the
  server is untouched.

**Route** `GET /audio/models` — installed vs. available models, sizes, and
download state, so clients can show status and trigger downloads.
`POST /audio/models/:id/download` starts a download; progress via the
existing SSE event pattern.

### 4.5 Model management

- `VEIN_MODEL_DIR` (default `~/.cache/vein-models`, same dir MiniLM uses
  today via `VEIN_MODEL_CACHE` — rename/alias so there's one setting).
- A small **catalog** in `src/audio/models.ts`: id → download URL (GitHub
  releases of sherpa-onnx), archive layout, recognizer type, language list,
  streaming yes/no, approximate size. Start with four entries:

| id | Use | Streaming |
|---|---|---|
| `parakeet-tdt-0.6b-v3-int8` | English, best offline accuracy | no |
| `moonshine-base-en-int8` | English, small and fast | no |
| `sense-voice-multilingual-int8` | zh/en/ja/ko/yue offline | no |
| `zipformer-streaming-en-int8` | live dictation | yes |

- Default model: `parakeet-tdt-0.6b-v3-int8` for `transcribe`,
  `zipformer-streaming-en-int8` for `stream`. Overridable by env
  (`VEIN_STT_MODEL`, `VEIN_STT_STREAM_MODEL`) and per call.
- Downloads happen on first use or via the route; never at boot. Server
  images pre-bake models into `VEIN_MODEL_DIR`.
- Verify a checksum from the catalog after download; extract with Node's
  `zlib` + a minimal tar reader (no new dep) or `tar` package if we already
  pull one transitively.

### 4.6 Client responsibilities (Swift / Kotlin)

- Capture the microphone natively (AVAudioEngine / AudioRecord). Convert to
  16 kHz mono PCM16LE. Do **not** try to use `getUserMedia` inside the
  webview; permission and device handling are unreliable there.
- Push-to-talk: buffer, wrap as WAV, `POST /audio/transcribe`.
- Live: open `/audio/stream`, send frames every ~100 ms, render partials.
- On desktop the base URL is `http://127.0.0.1:<port>`; on mobile it's the
  server. Nothing else differs.

### 4.7 Testing

- Unit: `stt.ts` model resolution and catalog parsing without the addon
  (mock the import).
- Integration (opt-in, `VEIN_TEST_STT=1`, like `VEIN_TEST_NEO`): download the
  Moonshine model once into a temp dir, transcribe a checked-in 3-second WAV
  fixture, assert on normalized text. Skip when the addon is missing.
- Streaming: feed the same fixture as PCM frames over the WS route and assert
  the final matches the offline result within edit distance.

---

## 5. Order of work

1. **Server prerequisites** (small, ship together): `VEIN_HOST`,
   `VEIN_WEB_DIST`, structured `ready` line, `GET /health`, `?key=` handoff in
   the UI, `VEIN_MODEL_DIR` alias.
2. **STT core**: `src/audio/stt.ts`, catalog, `audio/transcribe` step,
   `POST /audio/transcribe`, `GET /audio/models`. Testable on a server with
   no desktop work at all.
3. **Streaming**: `@hono/node-ws`, `/audio/stream`.
4. **Phase A packaging**: esbuild bundle, Node binary, native dir, a macOS
   host proof-of-concept that spawns vein and shows the UI.
5. **Kotlin host**, Windows shell override.
6. Later: single-binary (phase B), local vector store (§3), offline-mobile
   bindings.

Steps 1–3 are pure vein work and are useful for the hosted product on their
own.

---

## 6. Open questions

- WebSocket auth: header (fine for native clients) vs. `?key=` (needed if the
  webview ever opens the socket). Lean: accept both, same as HTTP.
- Should `audio/transcribe` also accept an artifact ref from a previous step,
  or only a path? Lean: both; artifacts are the natural way a `gdrive`/`http`
  step hands a file forward.
- Whisper models in the catalog: they're popular but slow on CPU and lack
  Parakeet's accuracy in English. Include `whisper-small-int8` only if
  multilingual demand shows up beyond what SenseVoice covers.
- Speaker diarization and VAD: sherpa ships silero-VAD and speaker
  embedding models. Not in v1; the streaming route's endpointing is enough.

---

## 7. Decision rules

- Transcription logic lives in `src/audio/`, exposed as a service; steps and
  routes are thin.
- Everything sherpa is lazy-imported and optional. A vein without the addon
  must boot and run every non-audio test.
- Native addons are never embedded; ship one platform dir beside the binary.
- Clients capture audio, vein recognizes it. No second STT implementation in
  the host apps until offline mobile is actually scheduled.
