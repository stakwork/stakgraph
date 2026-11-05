/**
 * custom-provider-e2e.ts — a custom OpenAI-compatible provider through the
 * real gateway image, priced from the runtime table, across a restart.
 *
 * Run (Docker up, `npm install` done in gateway/auth/ts):
 *
 *   make docker-build
 *   cd gateway/auth/ts && node --import tsx ../../scripts/custom-provider-e2e.ts
 *
 * IMAGE picks another image (default stakgraph-gateway:dev); KEEP=1 leaves
 * the containers up for poking at. Exits non-zero when any check fails.
 *
 * What it starts: redis, the gateway (macaroons and budgets enforced, the
 * datasheet fetch off), and, in this process, a fake OpenAI-compatible
 * upstream the container reaches at host.docker.internal: a private address,
 * so the provider needs `allow_private_network`, as `maple:8080` does on a
 * Stadeum box.
 *
 * What it proves:
 *   1. A provider added through bifrost's POST /api/providers with
 *      custom_provider_config.base_provider_type "openai" is routed for
 *      POST /openai/v1/chat/completions with model "maple/gemma4-31b": the
 *      upstream gets model "gemma4-31b", plain and streaming, and bifrost asks
 *      it for stream usage (stream_options.include_usage).
 *   2. The streamed answer carries the usage to the client, and the plugin
 *      accounts it.
 *   3. With no price anywhere the call counts $0, with the plugin's "no price"
 *      log line. After PUT /_plugin/pricing the same call is priced, and the
 *      run's cap refuses the next call with 402.
 *   4. When an upstream sends no `data: [DONE]`: a stream that closes needs no
 *      flag; one that parks after its usage holds every answer for the stream
 *      idle timeout unless the provider sets does_not_send_done_marker *and*
 *      wait_for_usage; the first alone drops the usage and bills $0.
 *   5. The price table, the custom providers, and a key that replaced a
 *      built-in provider's env.* slot all survive a container restart with the
 *      image's baked config.json.
 */

import { execFileSync, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const IMAGE = process.env.IMAGE ?? "stakgraph-gateway:dev";
const KEEP = process.env.KEEP === "1";
const NAME = `cp-e2e-${process.pid}`;
const ADMIN_USER = "admin";
const ADMIN_PASS = "e2e-admin-password";
const TOKEN = "e2e-provisioning-token";
const ORG_ID = "org_e2e";
// Fixture org pubkey #0, the key mint-macaroon.ts signs with.
const ORG_PUBKEY = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
const IDLE_TIMEOUT_S = 4;

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const TS_DIR = join(SCRIPT_DIR, "../auth/ts");
const MINT = join(SCRIPT_DIR, "mint-macaroon.ts");

// ─── checks ──────────────────────────────────────────────────────────

const failures: string[] = [];
function check(name: string, ok: boolean, detail = "") {
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name}${detail ? `  (${detail})` : ""}`);
  if (!ok) failures.push(name);
}
function section(title: string) {
  console.log(`\n=== ${title} ===`);
}

// ─── the fake upstream ───────────────────────────────────────────────
//
// The model id picks how a stream ends:
//   gemma4-31b  usage chunk, then `data: [DONE]`, then close (what Maple's
//               proxy forwards: it passes its backend's SSE through as is)
//   no-done     usage chunk, then close, no [DONE]
//   parks       usage chunk, then silence with the connection held open

interface Seen {
  model: string;
  stream: boolean;
  includeUsage: boolean;
  auth: string;
}
const seen: Seen[] = [];
const USAGE = { prompt_tokens: 1000, completion_tokens: 500, total_tokens: 1500 };

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let s = "";
    req.on("data", (c) => (s += c));
    req.on("end", () => resolve(s));
    req.on("error", reject);
  });
}

async function upstream(req: IncomingMessage, res: ServerResponse) {
  const body = await readBody(req);
  if (req.method === "GET" && req.url?.startsWith("/v1/models")) {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ object: "list", data: ["gemma4-31b", "no-done", "parks"].map((id) => ({ id, object: "model" })) }));
    return;
  }
  if (req.method !== "POST" || req.url !== "/v1/chat/completions") {
    res.writeHead(404).end();
    return;
  }
  const j = JSON.parse(body);
  seen.push({
    model: j.model,
    stream: !!j.stream,
    includeUsage: j.stream_options?.include_usage === true,
    auth: String(req.headers.authorization ?? ""),
  });
  const id = "chatcmpl-e2e";
  const created = Math.floor(Date.now() / 1000);
  if (!j.stream) {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({
      id, object: "chat.completion", created, model: j.model,
      choices: [{ index: 0, message: { role: "assistant", content: "hello" }, finish_reason: "stop" }],
      usage: USAGE,
    }));
    return;
  }
  res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
  const send = (o: unknown) => res.write(`data: ${JSON.stringify(o)}\n\n`);
  const chunk = (delta: object, finish: string | null) =>
    ({ id, object: "chat.completion.chunk", created, model: j.model, choices: [{ index: 0, delta, finish_reason: finish }] });
  send(chunk({ role: "assistant", content: "" }, null));
  send(chunk({ content: "hello" }, null));
  send(chunk({}, "stop"));
  if (j.stream_options?.include_usage) {
    send({ id, object: "chat.completion.chunk", created, model: j.model, choices: [], usage: USAGE });
  }
  if (j.model === "parks") {
    const t = setTimeout(() => res.end(), 30_000);
    req.on("close", () => clearTimeout(t));
    return;
  }
  if (j.model !== "no-done") res.write("data: [DONE]\n\n");
  res.end();
}

// ─── docker and http ─────────────────────────────────────────────────

function docker(...args: string[]): string {
  return execFileSync("docker", args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
}

// The container's output: the plugin logs to stderr, which `docker logs`
// passes through on its own stderr.
function containerLogs(name: string): string {
  const r = spawnSync("docker", ["logs", name], { encoding: "utf8" });
  return `${r.stdout}${r.stderr}`;
}

let GW = "";
let VK = ""; // the virtual key inference runs on, as a person's does
const basic = `Basic ${Buffer.from(`${ADMIN_USER}:${ADMIN_PASS}`).toString("base64")}`;

async function http(method: string, path: string, auth: string, body?: unknown) {
  const res = await fetch(`${GW}${path}`, {
    method,
    headers: { Authorization: auth, ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let json: any;
  try {
    json = JSON.parse(text);
  } catch {
    json = undefined;
  }
  return { status: res.status, text, json };
}
const admin = (m: string, p: string, b?: unknown) => http(m, p, basic, b);
const plugin = (m: string, p: string, b?: unknown) => http(m, p, `Bearer ${TOKEN}`, b);

// The host port is picked by docker, again on every restart.
async function locateGateway() {
  GW = `http://${docker("port", `${NAME}-gw`, "8181").split("\n")[0]}`;
  await waitHealthy();
}

async function waitHealthy() {
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    try {
      const [a, b] = await Promise.all([fetch(`${GW}/health`), fetch(`${GW}/_plugin/health`)]);
      if (a.ok && b.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`gateway not healthy in 90s:\n${containerLogs(`${NAME}-gw`).split("\n").slice(-40).join("\n")}`);
}

function mint(runId: string, maxCostUsd: number): string {
  return execFileSync(
    process.execPath,
    ["--import", "tsx", MINT, "--org-id", ORG_ID, "--user-id", "u_e2e", "--agent", "coder",
      "--run-id", runId, "--max-cost-usd", String(maxCostUsd), "--max-steps", "50"],
    { cwd: TS_DIR, encoding: "utf8" },
  ).trim();
}

interface Call {
  status: number;
  ms: number;
  body: string;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

let runSeq = 0;
function newRun(): string {
  return `r_cpe2e_${process.pid}_${++runSeq}`;
}

async function chat(model: string, runId: string, opts: { stream?: boolean; cap?: number; macaroon?: string } = {}): Promise<Call> {
  const macaroon = opts.macaroon ?? mint(runId, opts.cap ?? 1);
  const t0 = Date.now();
  const res = await fetch(`${GW}/openai/v1/chat/completions`, {
    method: "POST",
    headers: { "x-bf-vk": VK, "Content-Type": "application/json", "x-macaroon": macaroon },
    body: JSON.stringify({ model, stream: !!opts.stream, messages: [{ role: "user", content: "hi" }] }),
  });
  const body = await res.text();
  const ms = Date.now() - t0;
  let usage: Call["usage"];
  if (opts.stream) {
    for (const line of body.split("\n")) {
      if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
      try {
        const u = JSON.parse(line.slice(6)).usage;
        if (u) usage = u;
      } catch {
        // not JSON
      }
    }
  } else {
    try {
      usage = JSON.parse(body).usage;
    } catch {
      // error body
    }
  }
  return { status: res.status, ms, body, usage };
}

// The accumulator writes on a goroutine: wait for the call's step.
async function runState(runId: string): Promise<{ cost_usd: number; steps: number }> {
  const deadline = Date.now() + 5_000;
  for (;;) {
    const r = await plugin("GET", `/_plugin/runs/${runId}/state`);
    if (r.status === 200 && r.json.steps >= 1) return r.json;
    if (Date.now() > deadline) return r.json ?? { cost_usd: NaN, steps: 0 };
    await new Promise((res) => setTimeout(res, 100));
  }
}

const near = (a: number, b: number) => Math.abs(a - b) < 1e-9;

// POST /api/providers takes no keys: bifrost's create payload has no
// `keys` field (transports/v2.2.2 handlers/providers.go:125-136), so keys
// sent there are dropped without an error. The key is its own call.
function customProvider(name: string, extra: object = {}) {
  return {
    provider: name,
    network_config: {
      base_url: `http://host.docker.internal:${UPSTREAM_PORT}`,
      allow_private_network: true,
      stream_idle_timeout_in_seconds: IDLE_TIMEOUT_S,
    },
    custom_provider_config: { base_provider_type: "openai", ...extra },
  };
}

// ─── the run ─────────────────────────────────────────────────────────

let UPSTREAM_PORT = 0;

async function main() {
  const server = createServer((req, res) => {
    upstream(req, res).catch((e) => {
      console.error("upstream:", e);
      res.destroy();
    });
  });
  await new Promise<void>((r) => server.listen(0, "0.0.0.0", r));
  UPSTREAM_PORT = (server.address() as AddressInfo).port;

  try {
    section(`start redis + ${IMAGE}`);
    docker("network", "create", NAME);
    docker("volume", "create", `${NAME}-data`);
    docker("run", "-d", "--name", `${NAME}-redis`, "--network", NAME, "--network-alias", "redis", "redis:7-alpine");
    docker(
      "run", "-d", "--name", `${NAME}-gw`, "--network", NAME,
      "--add-host", "host.docker.internal:host-gateway",
      "-p", "127.0.0.1::8181",
      "-v", `${NAME}-data:/app/data`,
      "-e", "APP_HOST=127.0.0.1", "-e", "APP_PORT=8080",
      "-e", `BIFROST_ADMIN_USER=${ADMIN_USER}`, "-e", `BIFROST_ADMIN_PASS=${ADMIN_PASS}`,
      "-e", `BIFROST_PROVISIONING_TOKEN=${TOKEN}`,
      "-e", "BIFROST_PLUGIN_REDIS_URL=redis://redis:6379/0",
      "-e", "BIFROST_PLUGIN_ENFORCE_MACAROONS=true",
      "-e", "BIFROST_PLUGIN_ENFORCE_BUDGETS=true",
      "-e", "BIFROST_PLUGIN_PRICING_URL=off",
      IMAGE,
    );
    await locateGateway();
    console.log(`  gateway at ${GW}, fake upstream on :${UPSTREAM_PORT}`);

    const trust = await plugin("POST", "/_plugin/trust", {
      org_id: ORG_ID, pubkey: ORG_PUBKEY, issuer_url: "", revocation_poll_seconds: 60,
    });
    check("org registered in the trust registry", trust.status === 200, trust.text);

    section("custom providers through POST /api/providers");
    for (const [name, extra] of [
      ["maple", {}],
      ["maple-cut", { does_not_send_done_marker: true }],
      ["maple-wait", { does_not_send_done_marker: true, wait_for_usage: true }],
    ] as const) {
      const r = await admin("POST", "/api/providers", customProvider(name, extra));
      check(`POST /api/providers ${name}`, r.status === 200 || r.status === 201, `${r.status} ${r.text.slice(0, 200)}`);
      const k = await admin("POST", `/api/providers/${name}/keys`, {
        name: `${name}-key`, value: `${name}-fake-key`, models: ["*"], weight: 1,
      });
      check(`POST /api/providers/${name}/keys`, k.status === 200 || k.status === 201, `${k.status} ${k.text.slice(0, 200)}`);
    }
    const got = await admin("GET", "/api/providers/maple");
    check("GET /api/providers/maple: openai-based custom provider",
      got.json?.custom_provider_config?.base_provider_type === "openai",
      JSON.stringify(got.json?.custom_provider_config));
    const mapleKeys = await admin("GET", "/api/providers/maple/keys");
    check("maple has its key", (mapleKeys.json?.keys ?? []).length === 1, mapleKeys.text.slice(0, 200));

    // A customer and a virtual key allowed the three providers, the shape
    // stadeum gives each person.
    const customer = await admin("POST", "/api/governance/customers", {
      name: NAME, budget: { max_limit: 1000, reset_duration: "1d" },
    });
    const vk = await admin("POST", "/api/governance/virtual-keys", {
      name: NAME,
      customer_id: customer.json?.customer?.id,
      provider_configs: ["maple", "maple-cut", "maple-wait"].map((provider) => ({ provider, allowed_models: ["*"], key_ids: ["*"] })),
    });
    VK = vk.json?.virtual_key?.value ?? "";
    check("virtual key for the custom providers", !!VK, `${vk.status} ${vk.text.slice(0, 200)}`);

    section("unpriced: routed, usage reported, $0 with the log line");
    seen.length = 0;
    let run = newRun();
    let c = await chat("maple/gemma4-31b", run);
    check("plain call 200", c.status === 200, c.status === 200 ? "" : c.body.slice(0, 300));
    check("upstream got model \"gemma4-31b\" (the provider prefix stripped)", seen[0]?.model === "gemma4-31b", JSON.stringify(seen[0]));
    check("upstream got the provider's key as Bearer", seen[0]?.auth === "Bearer maple-fake-key", seen[0]?.auth);
    check("client got usage", c.usage?.total_tokens === 1500, JSON.stringify(c.usage));
    let st = await runState(run);
    check("plugin accounted one step at $0", st.steps === 1 && st.cost_usd === 0, JSON.stringify(st));

    const runStream = newRun();
    c = await chat("maple/gemma4-31b", runStream, { stream: true });
    check("streaming call 200", c.status === 200, c.status === 200 ? "" : c.body.slice(0, 300));
    check("bifrost asked the upstream for stream usage", seen[1]?.stream === true && seen[1]?.includeUsage === true, JSON.stringify(seen[1]));
    check("streamed answer carries usage to the client", c.usage?.total_tokens === 1500, JSON.stringify(c.usage));
    st = await runState(runStream);
    check("plugin accounted the stream: one step at $0", st.steps === 1 && st.cost_usd === 0, JSON.stringify(st));

    const logs = containerLogs(`${NAME}-gw`);
    const noPrice = logs.split("\n").filter((l) => l.includes(`no price for provider="maple" model="gemma4-31b"`));
    check("plugin logged the unpriced call (plain and stream)", noPrice.length >= 2, noPrice[0]?.replace(/^.*accounting: /, ""));

    section("PUT /_plugin/pricing, then priced");
    const table = {
      // Maple's price shape, ×1000 so one call is a round number of dollars.
      "maple/gemma4-31b": { input_per_mtok: 400, output_per_mtok: 1200 },
      // Bare keys price the model under any provider.
      "no-done": { input_per_mtok: 400, output_per_mtok: 1200 },
      parks: { input_per_mtok: 400, output_per_mtok: 1200 },
    };
    const put = await plugin("PUT", "/_plugin/pricing", table);
    check("PUT /_plugin/pricing 200", put.status === 200, put.text.slice(0, 200));
    const read = await plugin("GET", "/_plugin/pricing");
    check("GET /_plugin/pricing returns the table", read.json?.["maple/gemma4-31b"]?.output_per_mtok === 1200, read.text.slice(0, 200));
    const unauthed = await http("PUT", "/_plugin/pricing", "", {});
    check("PUT without the bearer is refused", unauthed.status === 401, String(unauthed.status));
    // 1000 × $400/M + 500 × $1200/M = $0.40 + $0.60.
    const COST = 1.0;

    run = newRun();
    c = await chat("maple/gemma4-31b", run, { cap: 5 });
    st = await runState(run);
    check("plain call priced at $1.00", c.status === 200 && near(st.cost_usd, COST), JSON.stringify(st));
    run = newRun();
    c = await chat("maple/gemma4-31b", run, { stream: true, cap: 5 });
    st = await runState(run);
    check("streaming call priced at $1.00", c.status === 200 && near(st.cost_usd, COST), JSON.stringify(st));

    section("the cap walk counts it");
    run = newRun();
    const mac = mint(run, 0.5);
    const first = await chat("maple/gemma4-31b", run, { macaroon: mac });
    st = await runState(run);
    check("first call on a $0.50 run: 200, $1.00 accumulated", first.status === 200 && near(st.cost_usd, COST), JSON.stringify(st));
    const second = await chat("maple/gemma4-31b", run, { macaroon: mac });
    check("second call refused 402 run_cost_exceeded", second.status === 402 && second.body.includes("run_cost_exceeded"),
      `${second.status} ${second.body.slice(0, 200)}`);

    section("streams without [DONE]");
    const variants: Array<[string, string, (c: Call, s: { cost_usd: number }) => boolean]> = [
      ["maple/no-done", "closes after usage, no flag: priced, prompt", (c, s) => near(s.cost_usd, COST) && c.ms < 2000],
      ["maple/parks", `parks after usage, no flag: priced, held for the ${IDLE_TIMEOUT_S}s idle timeout`,
        (c, s) => near(s.cost_usd, COST) && c.ms >= (IDLE_TIMEOUT_S - 1) * 1000],
      ["maple-cut/parks", "parks, does_not_send_done_marker alone: prompt, but usage dropped, $0",
        (c, s) => s.cost_usd === 0 && c.ms < 2000],
      ["maple-wait/parks", "parks, does_not_send_done_marker + wait_for_usage: priced, prompt",
        (c, s) => near(s.cost_usd, COST) && c.ms < 2000],
    ];
    for (const [model, what, ok] of variants) {
      run = newRun();
      c = await chat(model, run, { stream: true, cap: 5 });
      st = await runState(run);
      check(`${model}: ${what}`, c.status === 200 && ok(c, st),
        `${c.ms} ms, client usage ${c.usage?.total_tokens ?? "none"}, cost $${st.cost_usd}`);
    }

    section("restart with the baked config.json");
    const xaiKeys = await admin("GET", "/api/providers/xai/keys");
    const slot = (xaiKeys.json?.keys ?? [])[0];
    check("xai has its env.* key slot", slot?.value?.type === "env", JSON.stringify(slot?.value));
    const replaced = await admin("PUT", `/api/providers/xai/keys/${slot?.id}`, {
      name: slot?.name, value: "xai-e2e-fake-key", models: ["*"], weight: 1,
    });
    check("PUT replaces the xai slot", replaced.status === 200, `${replaced.status} ${replaced.text.slice(0, 200)}`);

    docker("restart", `${NAME}-gw`);
    await locateGateway();
    const after = await plugin("GET", "/_plugin/pricing");
    check("price table survived the restart", after.json?.["maple/gemma4-31b"]?.input_per_mtok === 400, after.text.slice(0, 200));
    for (const name of ["maple", "maple-cut", "maple-wait"]) {
      const p = await admin("GET", `/api/providers/${name}`);
      const k = await admin("GET", `/api/providers/${name}/keys`);
      check(`custom provider ${name} and its key survived the restart`,
        p.json?.custom_provider_config?.base_provider_type === "openai" && (k.json?.keys ?? []).length === 1,
        `${p.status}, keys ${k.text.slice(0, 160)}`);
    }
    const wait = await admin("GET", "/api/providers/maple-wait");
    check("its stream flags survived too",
      wait.json?.custom_provider_config?.does_not_send_done_marker === true && wait.json?.custom_provider_config?.wait_for_usage === true,
      JSON.stringify(wait.json?.custom_provider_config));
    const xaiAfter = (await admin("GET", "/api/providers/xai/keys")).json?.keys ?? [];
    check("the replaced xai key survived (same id, no longer env)",
      xaiAfter.length === 1 && xaiAfter[0].id === slot?.id && xaiAfter[0].value?.type !== "env",
      JSON.stringify(xaiAfter.map((k: any) => ({ id: k.id, name: k.name, type: k.value?.type }))));
    run = newRun();
    c = await chat("maple/gemma4-31b", run, { stream: true, cap: 5 });
    st = await runState(run);
    check("after the restart a Maple stream is routed and priced", c.status === 200 && near(st.cost_usd, COST), JSON.stringify(st));
  } finally {
    server.close();
    server.closeAllConnections();
    if (KEEP) {
      console.log(`\nKEEP=1: left ${NAME}-gw (${GW}) and ${NAME}-redis running`);
    } else {
      for (const args of [
        ["rm", "-f", `${NAME}-gw`, `${NAME}-redis`],
        ["network", "rm", NAME],
        ["volume", "rm", `${NAME}-data`],
      ]) {
        try {
          docker(...args);
        } catch {
          // already gone
        }
      }
    }
  }

  console.log(failures.length ? `\nFAILED: ${failures.length} check(s)` : "\nall checks passed");
  process.exit(failures.length ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
