/**
 * `GET /audio/try` — a dependency-free page for trying dictation from a
 * browser: mic → PCM16 → `/audio/stream`, partials and finals on screen,
 * finals editable (edits go to `/audio/sessions/:id/corrections`).
 *
 * Dev/test surface only. The desktop and mobile hosts capture audio natively
 * (plans/local-desktop-and-stt.md §4.6); this page exists so the recognizer
 * can be exercised with nothing but vein and a browser. Inline HTML because
 * `tsc` doesn't copy assets. Served without the api-key middleware — the
 * page itself is inert; the socket and routes it calls still require the
 * key, which the page sends as `?key=` / `Bearer`.
 */
export const TRY_PAGE_HTML = String.raw`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vein · dictation</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.4 system-ui, sans-serif; margin: 0; padding: 20px; max-width: 860px; }
  h1 { font-size: 18px; margin: 0 0 12px; }
  fieldset { border: 1px solid #8884; border-radius: 6px; margin: 0 0 12px; padding: 10px 12px; }
  legend { padding: 0 6px; opacity: .7; }
  label { display: inline-block; margin: 4px 16px 4px 0; }
  input, select, textarea, button { font: inherit; }
  input[type=text] { width: 16em; }
  textarea { width: 100%; box-sizing: border-box; height: 4.5em; font-family: ui-monospace, monospace; }
  button { padding: 6px 14px; cursor: pointer; }
  #start { font-weight: 600; }
  #status { margin-left: 10px; opacity: .8; }
  #models div { margin: 2px 0; }
  #models button { padding: 1px 8px; font-size: 12px; margin-left: 6px; }
  #out { border: 1px solid #8884; border-radius: 6px; padding: 12px; min-height: 8em; font-size: 17px; }
  .final { display: block; margin: 0 0 4px; padding: 2px 4px; border-radius: 4px; outline: none; }
  .final:focus { background: #8882; }
  .final.edited::after { content: " ✎"; opacity: .5; font-size: 12px; }
  #partial { opacity: .55; }
  #log { font: 12px ui-monospace, monospace; opacity: .7; white-space: pre-wrap; max-height: 12em; overflow: auto; margin-top: 12px; }
</style>
</head>
<body>
<h1>vein · dictation test page</h1>

<fieldset>
  <legend>connection</legend>
  <label>api key <input type="text" id="key" placeholder="VEIN_API_KEY (blank in dev)"></label>
  <label>session <input type="text" id="session"></label>
</fieldset>

<fieldset>
  <legend>models <span id="avail"></span></legend>
  <div id="models">loading…</div>
  <label>finals model <select id="model"></select></label>
  <label>partials model <select id="partialModel"><option value="">none (single recognizer)</option></select></label>
</fieldset>

<fieldset>
  <legend>hotwords (one phrase per line, optional <code>:score</code>; or a stored list name on one line prefixed with <code>@</code>)</legend>
  <textarea id="hotwords" placeholder="Sphinx&#10;sphinx&#10;Stakwork :2"></textarea>
  <label>default score <input type="number" id="hotwordsScore" value="2" step="0.5" min="0" max="10" style="width:5em"></label>
</fieldset>

<p>
  <button id="start">● start</button>
  <button id="stop" disabled>■ stop</button>
  <span id="status">idle</span>
</p>

<div id="out"><span id="partial"></span></div>
<div id="log"></div>

<script>
const $ = (id) => document.getElementById(id);
const log = (m) => { $("log").textContent += m + "\n"; $("log").scrollTop = 1e9; };
const key = () => $("key").value.trim();
const headers = () => (key() ? { authorization: "Bearer " + key() } : {});
$("session").value = "try-" + new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
try { $("key").value = sessionStorage.getItem("vein-key") || new URLSearchParams(location.search).get("key") || ""; } catch {}
$("key").addEventListener("change", () => { try { sessionStorage.setItem("vein-key", key()); } catch {} });

// Vein may be mounted under a prefix (e.g. /lab): derive it from this page's path.
const BASE = location.pathname.replace(/\/audio\/try\/?$/, "");
const api = (p) => BASE + p;

async function loadModels() {
  const r = await fetch(api("/audio/models"), { headers: headers() });
  if (!r.ok) { $("models").textContent = "GET /audio/models → " + r.status + " " + (await r.text()); return; }
  const j = await r.json();
  $("avail").textContent = j.available ? "(sherpa addon loaded)" : "(sherpa addon NOT available — install sherpa-onnx-node)";
  $("models").innerHTML = "";
  for (const sel of ["model", "partialModel"]) {
    const s = $(sel); while (s.options.length > (sel === "partialModel" ? 1 : 0)) s.remove(s.options.length - 1);
  }
  for (const m of j.models) {
    const row = document.createElement("div");
    row.textContent = m.id + " — " + Math.round(m.bytes / 1e6) + " MB, partials ~" + m.chunkMs + " ms, hotwords: " + (m.hotwords ? "yes" : "no") + (m.installed ? " · installed" : " · not installed");
    if (!m.installed) {
      const b = document.createElement("button"); b.textContent = "download";
      b.onclick = () => download(m.id, b);
      row.appendChild(b);
    }
    $("models").appendChild(row);
    for (const sel of ["model", "partialModel"]) {
      if (sel === "model" && !m.hotwords) continue;
      const o = new Option(m.id + (m.installed ? "" : " (not installed)"), m.id, false, m.default === sel);
      $(sel).add(o);
    }
  }
}

async function download(id, btn) {
  btn.disabled = true;
  const r = await fetch(api("/audio/models/" + id + "/download"), { method: "POST", headers: headers() });
  const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
  while (true) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    for (const line of buf.split("\n")) if (line.startsWith("data:")) {
      try {
        const p = JSON.parse(line.slice(5));
        btn.textContent = p.phase === "download" ? Math.round(100 * p.received / p.total) + "%" : p.phase;
      } catch {}
    }
    buf = buf.slice(buf.lastIndexOf("\n") + 1);
  }
  await loadModels();
}

// ── capture ───────────────────────────────────────────────────────────────
let ws, ctx, node, mediaStream;

// AudioWorklet: batches ~100 ms of PCM16LE per message (sampleRate is the worklet global).
const WORKLET_SRC = String.raw${"`"}
    class P extends AudioWorkletProcessor {
      constructor() { super(); this.buf = []; this.n = 0; this.frame = Math.round(sampleRate / 10); }
      process(inputs) {
        const ch = inputs[0] && inputs[0][0]; if (!ch) return true;
        this.buf.push(new Float32Array(ch)); this.n += ch.length;
        if (this.n >= this.frame) {
          const out = new Int16Array(this.n); let o = 0;
          for (const b of this.buf) for (let i = 0; i < b.length; i++) out[o++] = Math.max(-32768, Math.min(32767, Math.round(b[i] * 32768)));
          this.port.postMessage(out.buffer, [out.buffer]); this.buf = []; this.n = 0;
        }
        return true;
      }
    }
    registerProcessor("pcm16", P);
  ${"`"};

function hotwords() {
  const t = $("hotwords").value.trim();
  if (!t) return undefined;
  if (t.startsWith("@") && !t.includes("\n")) return t.slice(1).trim();
  return t.split("\n").map((l) => l.trim()).filter(Boolean);
}

async function start() {
  $("start").disabled = true; $("stop").disabled = false;
  $("out").querySelectorAll(".final").forEach((e) => e.remove()); $("partial").textContent = "";
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (e) { return fail("microphone: " + e.message); }
  // Ask for 16 kHz; browsers that ignore it report their own rate and vein resamples.
  ctx = new AudioContext({ sampleRate: 16000 });
  const sampleRate = ctx.sampleRate;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = new URL(proto + "//" + location.host + api("/audio/stream"));
  if (key()) url.searchParams.set("key", key());
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    const startMsg = {
      type: "start", sampleRate,
      model: $("model").value, partialModel: $("partialModel").value || null,
      session: $("session").value.trim() || undefined,
      hotwords: hotwords(), hotwordsScore: Number($("hotwordsScore").value) || undefined,
    };
    ws.send(JSON.stringify(startMsg));
    log("→ start " + JSON.stringify(startMsg));
    status("loading recognizer… (first use downloads the model)");
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "ready") { status("listening @ " + sampleRate + " Hz · " + m.model + (m.partialModel ? " + " + m.partialModel : "") + (m.hotwords ? " · hotwords " + m.hotwords : "")); log("← ready " + JSON.stringify(m)); }
    else if (m.type === "partial") $("partial").textContent = m.text;
    else if (m.type === "final") { addFinal(m); $("partial").textContent = ""; log("← final #" + m.index + " " + JSON.stringify(m.text)); }
    else if (m.type === "error") fail(m.error);
  };
  ws.onclose = (ev) => { log("socket closed " + ev.code + " " + ev.reason); teardown(); status("idle"); };
  ws.onerror = () => fail("websocket error");

  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" })));
  node = new AudioWorkletNode(ctx, "pcm16", { numberOfInputs: 1, numberOfOutputs: 0 });
  node.port.onmessage = (e) => { if (ws && ws.readyState === 1) ws.send(e.data); };
  ctx.createMediaStreamSource(mediaStream).connect(node);
}

function stop() {
  $("stop").disabled = true;
  if (ws && ws.readyState === 1) { ws.send(JSON.stringify({ type: "end" })); log("→ end"); status("flushing…"); }
  else teardown();
}

function teardown() {
  node && node.disconnect(); node = null;
  mediaStream && mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null;
  ctx && ctx.close(); ctx = null;
  ws = null;
  $("start").disabled = false; $("stop").disabled = true;
}

function fail(msg) { log("error: " + msg); status("error: " + msg); if (ws) ws.close(); teardown(); }
function status(s) { $("status").textContent = s; }

function addFinal(m) {
  const el = document.createElement("span");
  el.className = "final"; el.contentEditable = "true"; el.textContent = m.text;
  el.title = "click to edit — the edit is sent as a correction";
  el.dataset.index = m.index; el.dataset.orig = m.text;
  el.addEventListener("blur", async () => {
    const text = el.textContent.trim();
    if (text === el.dataset.orig) return;
    const session = $("session").value.trim();
    const r = await fetch(api("/audio/sessions/" + encodeURIComponent(session) + "/corrections"), {
      method: "POST", headers: { "content-type": "application/json", ...headers() },
      body: JSON.stringify({ index: Number(el.dataset.index), text }),
    });
    log((r.ok ? "→ correction #" : "correction failed #") + el.dataset.index + " " + JSON.stringify(text));
    if (r.ok) { el.dataset.orig = text; el.classList.add("edited"); }
  });
  $("out").insertBefore(el, $("partial"));
}

$("start").onclick = start;
$("stop").onclick = stop;
loadModels().catch((e) => { $("models").textContent = String(e); });
</script>
</body>
</html>
`;
