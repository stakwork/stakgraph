import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import yaml from "js-yaml";
import { WorkspaceManager, buildRegistry, createStrut, defineStep, fileArtifactsCapability, runWorkflow, standardServices } from "strut";
import { seedPodSetupSteps, seedPodSetupWorkflows, POD_SETUP_CONCEPTS } from "./seed.js";
import { factsText, scanRepo } from "./steps/repo-facts.js";
import { finishSetup, type Draft } from "./steps/finish.js";

const front = (text: string) => {
  const m = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(text)!;
  return { meta: yaml.load(m[1]!) as Record<string, string>, body: text.slice(m[0].length).trim() };
};

/** A repository on disk, from a { path: content } map. */
async function repo(root: string, name: string, files: Record<string, string>): Promise<string> {
  const dir = join(root, name);
  for (const [p, c] of Object.entries(files)) {
    await mkdir(join(dir, p, ".."), { recursive: true });
    await writeFile(join(dir, p), c);
  }
  return dir;
}

describe("the Pod Setup pages", () => {
  it("are one tree under Pod Config: the root, a platform, Local Services, and a page per service under it; Onboarding is under Job", async () => {
    const files = await readdir(POD_SETUP_CONCEPTS.dir);
    const pages = new Map<string, { parent: string; description: string; body: string }>();
    for (const f of files) {
      const { meta, body } = front(await readFile(join(POD_SETUP_CONCEPTS.dir, f), "utf8"));
      assert.ok(meta.description && body, `${f}: a description and docs`);
      assert.ok(body.startsWith(meta.description), `${f}: the docs open with the description`);
      pages.set(f.replace(/\.md$/, ""), { parent: meta.parent!, description: meta.description, body });
    }
    assert.equal(pages.get("Pod Setup")?.parent, "Pod Config");
    assert.equal(pages.get("Pod Setup - Web")?.parent, "Pod Setup");
    assert.equal(pages.get("Pod Setup - Local Services")?.parent, "Pod Setup");
    const services = [...pages].filter(([, p]) => p.parent === "Pod Setup - Local Services").map(([n]) => n).sort();
    assert.deepEqual(services, [
      "Pod Setup - Elasticsearch", "Pod Setup - LiveKit", "Pod Setup - MySQL", "Pod Setup - Postgres",
      "Pod Setup - RabbitMQ", "Pod Setup - Redis", "Pod Setup - S3-compatible Storage", "Pod Setup - Supabase",
    ]);
    for (const [name, p] of pages) assert.ok(name === "Pod Setup" || pages.has(p.parent), `${name}: its parent is in the set`);
    // The pod is the app's container: no page asks for an `app` service.
    for (const [name, p] of pages) assert.ok(!/^\s+app:/m.test(p.body), `${name}: no app service`);
    // MinIO's images are gone from Docker Hub and Quay: a page naming one makes a compose file that cannot pull.
    for (const [name, p] of pages) assert.ok(!/image:\s*minio\//.test(p.body), `${name}: no MinIO image`);
    // The app signs with the secret the server is given: dev mode's own is `devkey: secret`.
    const livekit = pages.get("Pod Setup - LiveKit")!.body;
    const secret = /LIVEKIT_API_SECRET`\s+`([^`]+)`/.exec(livekit)?.[1];
    assert.equal(/LIVEKIT_KEYS:\s+"devkey: ([^"]+)"/.exec(livekit)?.[1], secret, "LIVEKIT_KEYS gives the server the app's secret");
    const onboarding = front(await readFile(join(POD_SETUP_CONCEPTS.dir, "..", "..", "job", "concepts", "Onboarding.md"), "utf8"));
    assert.equal(onboarding.meta.parent, "Job");
    for (const step of ["pod/claim", "identify-services", "pod/config-set", "wait: false", "pod/config-apply", "meta/verify-job", "fresh: true"]) {
      assert.ok(onboarding.body.includes(step), `Onboarding names ${step}`);
    }
  });
});

describe("pod-setup/repo-facts", () => {
  let root: string;
  before(async () => {
    root = await mkdtemp(join(tmpdir(), "repo-facts-"));
  });
  after(() => rm(root, { recursive: true, force: true }));

  it("finds package files with the manager their lockfile names, scripts, compose and env examples, and the env vars the code reads", async () => {
    const dir = await repo(root, "shop", {
      "package.json": JSON.stringify({ scripts: { dev: "next dev", build: "next build" } }),
      "yarn.lock": "",
      "apps/api/package.json": JSON.stringify({ scripts: { dev: "tsx watch src/main.ts" } }),
      "apps/api/src/db.ts": "const url = process.env.DATABASE_URL; const k = process.env['STRIPE_KEY'];",
      "web/src/config.ts": "export const api = import.meta.env.VITE_API_URL;",
      "worker/run.py": "import os\nos.getenv('REDIS_URL')\nos.environ['QUEUE']\n",
      ".env.example": "DATABASE_URL=postgres://...\nexport NEXTAUTH_SECRET=x\n# COMMENTED=1\n",
      "docker-compose.yml": "services: {}\n",
      "node_modules/left-pad/package.json": "{}",
      "node_modules/left-pad/index.js": "process.env.NOT_MINE",
      ".github/workflows/ci.yml": "env: { GH: 1 }",
      "logo.png": "\u0000PNG",
    });
    const facts = await scanRepo({ path: dir, repo: "acme/shop", url: "https://github.com/acme/shop" });
    assert.equal(facts.name, "shop");
    assert.deepEqual(
      facts.packageFiles.map((p) => [p.path, p.manager, p.lockfile]),
      [["package.json", "yarn", "yarn.lock"], ["apps/api/package.json", "yarn", "yarn.lock"]],
      "a monorepo package takes the root's lockfile; node_modules is never walked",
    );
    assert.deepEqual(facts.packageFiles[0]!.scripts, { dev: "next dev", build: "next build" });
    assert.deepEqual(facts.composeFiles, ["docker-compose.yml"]);
    assert.deepEqual(facts.envExamples, [{ file: ".env.example", names: ["DATABASE_URL", "NEXTAUTH_SECRET"] }]);
    assert.deepEqual(facts.envReads, [
      { file: "apps/api/src/db.ts", names: ["DATABASE_URL", "STRIPE_KEY"] },
      { file: "web/src/config.ts", names: ["VITE_API_URL"] },
      { file: "worker/run.py", names: ["QUEUE", "REDIS_URL"] },
    ]);
    const text = factsText([facts]);
    assert.match(text, /## shop — \/workspaces\/shop \(https:\/\/github.com\/acme\/shop\)/);
    assert.match(text, /package.json \(node; yarn, from yarn.lock\) scripts: dev="next dev", build="next build"/);
    assert.match(text, /Env example .env.example: DATABASE_URL, NEXTAUTH_SECRET/);
    assert.ok(!text.includes("NOT_MINE") && !text.includes("GH"));
  });

  it("refuses two repositories with one directory name", async () => {
    const { default: step } = await import("./steps/repo-facts.js");
    const a = await repo(root, "x/api", { "go.mod": "module a" });
    const b = await repo(root, "y/api", { "go.mod": "module b" });
    await assert.rejects(step.run(step.input.parse({ checkouts: [{ path: a }, { path: b }] }), {} as any), /share the directory name "api"/);
  });
});

describe("pod-setup/finish", () => {
  let root: string;
  let repos: Array<{ name: string; path: string }>;
  before(async () => {
    root = await mkdtemp(join(tmpdir(), "finish-"));
    repos = [
      { name: "shop", path: await repo(root, "shop", { "web/package.json": "{}", "api/package.json": "{}" }) },
      { name: "lib", path: await repo(root, "lib", { "package.json": "{}" }) },
    ];
  });
  after(() => rm(root, { recursive: true, force: true }));

  const draft = (over: Partial<Draft> = {}): Draft => ({
    frontend: {
      repo: "shop",
      subdir: "./web/",
      port: 3000,
      scripts: { start: "yarn dev --hostname 0.0.0.0", install: "yarn install" },
      env: {
        DATABASE_URL: "postgresql://postgres:password@postgres:5432/shop",
        REDIS_HOST: "redis",
        DB_TYPE: "postgres",
        PORT: "3000",
        PRE_START_COMMAND: "npx prisma migrate deploy",
        FLAG: true as unknown as string,
      },
    },
    services: [{ name: "api", repo: "shop", subdir: "api", port: 4000, scripts: { start: "yarn dev", install: "yarn install" } }],
    local_services: {
      app: { build: ".", command: "sleep infinity" },
      postgres: { image: "postgres:16", ports: ["5432:5432"], volumes: ["pg-data:/var/lib/postgresql/data", "./init.sql:/docker-entrypoint-initdb.d/init.sql"], networks: ["backend"] },
      redis: { image: "redis:7", ports: ["6379:6379"] },
    },
    ...over,
  });

  it("names the frontend, writes every cwd, moves config vars, makes service hosts 127.0.0.1, and assembles the compose file", async () => {
    const out = await finishSetup(draft(), repos);
    assert.deepEqual(out.services, [
      {
        name: "frontend",
        port: 3000,
        cwd: "/workspaces/shop/web",
        scripts: { start: "yarn dev --hostname 0.0.0.0", install: "yarn install", preStart: "npx prisma migrate deploy" },
        env: { DATABASE_URL: "postgresql://postgres:password@127.0.0.1:5432/shop", REDIS_HOST: "127.0.0.1", DB_TYPE: "postgres", FLAG: "true" },
      },
      { name: "api", port: 4000, cwd: "/workspaces/shop/api", scripts: { start: "yarn dev", install: "yarn install" } },
    ]);
    const compose = yaml.load(out.containerFiles["docker-compose.yml"]!) as any;
    assert.deepEqual(Object.keys(compose.services), ["postgres", "redis"], "no app service");
    assert.deepEqual(compose.volumes, { "pg-data": {} }, "the named volume declared, the bind mount not");
    assert.deepEqual(compose.networks, { backend: {} });
    assert.equal(out.containerFiles.Dockerfile, undefined, "no Dockerfile: hive's default stays");
    assert.ok(out.notes.some((n) => /dropped the compose service `app`/.test(n)));
    assert.ok(out.notes.some((n) => /moved env PRE_START_COMMAND/.test(n)));
    assert.ok(out.notes.some((n) => /env DATABASE_URL named a compose service as its host/.test(n)));
  });

  it("makes only a URL's host 127.0.0.1, never a user name or a value that only looks like one", async () => {
    const { localizeHosts } = await import("./steps/finish.js");
    const hosts = new Set(["postgres", "redis", "rustfs"]);
    assert.equal(localizeHosts("postgresql://postgres:pw@postgres:5432/db", hosts), "postgresql://postgres:pw@127.0.0.1:5432/db");
    assert.equal(localizeHosts("redis://redis:6379", hosts), "redis://127.0.0.1:6379");
    assert.equal(localizeHosts("http://rustfs/bucket?x=1", hosts), "http://127.0.0.1/bucket?x=1");
    assert.equal(localizeHosts("http://postgres.example.com:5432", hosts), "http://postgres.example.com:5432");
    assert.equal(localizeHosts("postgres", hosts), "postgres");
  });

  it("writes `services: {}` with no local services, and a Dockerfile only when it is not the default", async () => {
    const plain = await finishSetup(draft({ local_services: {}, dockerfile: "FROM ghcr.io/stakwork/staklink-js:latest" }), repos);
    assert.equal(plain.containerFiles["docker-compose.yml"], "services: {}\n");
    assert.equal(plain.containerFiles.Dockerfile, undefined);
    const other = await finishSetup(draft({ local_services: {}, dockerfile: "FROM ghcr.io/stakwork/staklink-android:latest" }), repos);
    assert.equal(other.containerFiles.Dockerfile, "FROM ghcr.io/stakwork/staklink-android:latest\n");
  });

  it("refuses what it cannot fix, naming it", async () => {
    const bad = async (d: Draft, re: RegExp) => assert.rejects(finishSetup(d, repos), re);
    const fe = draft().frontend;
    await bad(draft({ frontend: { ...fe, repo: "nope" } }), /repo "nope" is not one of the checked-out repositories \(shop, lib\)/);
    await bad(draft({ frontend: { ...fe, subdir: "missing" } }), /"missing" is not a directory in shop/);
    await bad(draft({ frontend: { ...fe, subdir: "../lib" } }), /leaves the repository/);
    await bad(draft({ frontend: { ...fe, port: 0 } }), /is not a port/);
    await bad(draft({ services: [{ name: "api", repo: "lib", port: 3000, scripts: { start: "x" } }] }), /port 3000 is also "frontend"'s/);
    await bad(draft({ services: [{ name: "Frontend", repo: "lib", port: 5000, scripts: { start: "x" } }] }), /the `frontend` field, not a service of that name/);
    await bad(draft({ services: [{ repo: "lib", port: 5000, scripts: { start: "x" } }] }), /needs a name/);
    await bad(draft({ frontend: { ...fe, scripts: { install: "yarn" } } }), /no start command/);
    await bad(draft({ frontend: { ...fe, env: { "BAD-NAME": "x" } } }), /"BAD-NAME" is not an env var name/);
  });
});

describe("identify-services (offline: fake agents, checkout and graph)", () => {
  const PAGES: Record<string, { description: string; docs: string; children?: string[] }> = {
    "Pod Setup": { description: "The contract.", docs: "ROOT DOCS", children: ["Pod Setup - Local Services", "Pod Setup - Web"] },
    "Pod Setup - Web": { description: "Web apps.", docs: "WEB DOCS" },
    "Pod Setup - Local Services": { description: "Beside the app.", docs: "LOCAL DOCS", children: ["Pod Setup - Postgres", "Pod Setup - Redis"] },
    "Pod Setup - Postgres": { description: "Postgres.", docs: "POSTGRES DOCS" },
    "Pod Setup - Redis": { description: "Redis.", docs: "REDIS DOCS" },
  };
  const gets: Record<string, any>[] = [];
  const fakeGet = defineStep({
    type: "graph/graph-get",
    input: z.object({}).passthrough(),
    output: z.any(),
    async run(cfg: any) {
      gets.push(cfg);
      const page = PAGES[cfg.name];
      if (!page) {
        const msg = `node not found: Concept "${cfg.name}"`;
        if (cfg.required) throw new Error(msg);
        return msg;
      }
      const out: any = { ref_id: `r-${cfg.name}`, node_type: "Concept", name: cfg.name, properties: { name: cfg.name, docs: page.docs }, edges: {} };
      if (cfg.children) {
        out.children = (page.children ?? []).map((n) => ({ ref_id: `r-${n}`, node_type: "Concept", name: n, description: PAGES[n]!.description }));
      }
      return out;
    },
  });
  const agentCalls: Record<string, any>[] = [];
  const PROFILE = {
    repos: [{ repo: "shop", role: "app", platform: "Pod Setup - Web", package_manager: "yarn", summary: "a Next.js shop" }],
    platforms: ["Pod Setup - Web"],
    local_services: ["Pod Setup - Postgres"],
  };
  const GENERATED: Draft = {
    frontend: { repo: "shop", port: 3000, scripts: { start: "yarn dev --hostname 0.0.0.0", install: "yarn install" }, env: { DATABASE_URL: "postgresql://u:p@postgres:5432/shop" } },
    services: [],
    local_services: { postgres: { image: "postgres:16", ports: ["5432:5432"] } },
  };
  const fakeAgent = defineStep({
    type: "agent",
    input: z.object({}).passthrough(),
    output: z.any(),
    async run(cfg: any) {
      agentCalls.push(cfg);
      const profile = (cfg.schema?.required ?? []).includes("platforms");
      return { result: "ok", object: profile ? PROFILE : GENERATED, steps: 3, usage: {}, cost: profile ? 0.01 : 0.02 };
    },
  });
  let root: string;
  let ws: WorkspaceManager;
  const fakeCheckout = defineStep({
    type: "git/checkout",
    input: z.object({}).passthrough(),
    output: z.any(),
    async run(cfg: any) {
      const name = String(cfg.repo).split("/").pop()!;
      const path = await repo(join(root, "co"), name, { "package.json": JSON.stringify({ scripts: { dev: "next dev" } }), "yarn.lock": "", "src/db.ts": "process.env.DATABASE_URL" });
      return { path, repo: `acme/${name}`, sha: "abc", ref: "main", branch: "main", host: "github.com", url: cfg.repo };
    },
  });
  let run: (input: Record<string, unknown>, registryOver?: Record<string, unknown>) => ReturnType<typeof runWorkflow>;
  before(async () => {
    root = await mkdtemp(join(tmpdir(), "identify-services-"));
    ws = new WorkspaceManager(join(root, "ws"));
    await seedPodSetupSteps(ws);
    await seedPodSetupWorkflows(ws);
    const flow = await ws.getWorkflow("identify-services");
    const { registry } = await buildRegistry(await ws.materializeCustomSteps());
    const dataDir = join(root, "data");
    const services = { ...standardServices({ secretsSource: {}, dataDir }), artifacts: fileArtifactsCapability(join(dataDir, "artifacts")) };
    run = (input, over = {}) =>
      runWorkflow(flow, input, { ...registry, agent: fakeAgent, "git/checkout": fakeCheckout, "graph/graph-get": fakeGet, ...over } as typeof registry, { services });
  });
  after(() => rm(root, { recursive: true, force: true }));

  it("validates statically against the real registry (what meta/validate-workflow runs)", async () => {
    const real = await createStrut({ workspace: ws, serveUi: false, enableChat: false, scheduler: false });
    const authoring = (real.services as any).authoring;
    const entry = (await ws.listWorkflows()).find((w) => w.name === "identify-services")!;
    assert.equal(entry.category, "pods");
    const source = await ws.getWorkflowSource("identify-services", entry.activeVersion);
    const v = await authoring.validateWorkflow(source, "identify-services");
    assert.equal(v.ok, true, JSON.stringify(v.errors, null, 2));
  });

  it("loads the root and the catalog, profiles from menus built from their children, loads the picked pages, and returns pod/config-set's input", async () => {
    gets.length = 0;
    agentCalls.length = 0;
    const r = await run({ repos: ["https://github.com/acme/shop"] });
    assert.equal(r.status, "success", JSON.stringify(r.error));
    const out = r.output as any;

    // Every load is a step, and every one is required.
    assert.deepEqual(gets.map((g) => [g.name, g.children ?? null, g.required]).sort((a, b) => String(a[0]).localeCompare(String(b[0]))), [
      ["Pod Setup", "PARENT_OF", true],
      ["Pod Setup - Local Services", "PARENT_OF", true],
      ["Pod Setup - Postgres", null, true],
      ["Pod Setup - Web", null, true],
    ]);

    const [profile, generate] = agentCalls;
    assert.deepEqual(profile!.toolFilter, ["repo_overview", "fulltext_search", "str_replace_based_edit_tool", "file_summary"], "no bash, no web");
    assert.match(profile!.prompt, /ROOT DOCS/);
    assert.match(profile!.prompt, /- Pod Setup - Web: Web apps\./, "the platform menu, one per line");
    assert.ok(!/- Pod Setup - Local Services:/.test(profile!.prompt), "the catalog is not a platform");
    assert.match(profile!.prompt, /- Pod Setup - Postgres: Postgres\.\n- Pod Setup - Redis: Redis\./);
    assert.match(profile!.prompt, /package.json \(node; yarn, from yarn.lock\) scripts: dev="next dev"/);
    const p = profile!.schema.properties;
    assert.deepEqual(p.repos.items.properties.repo.enum, ["shop"]);
    assert.deepEqual(p.repos.items.properties.platform.enum, ["Pod Setup - Web"]);
    assert.deepEqual(p.platforms.items.enum, ["Pod Setup - Web"]);
    assert.deepEqual(p.local_services.items.enum, ["Pod Setup - Postgres", "Pod Setup - Redis"]);

    assert.match(generate!.prompt, /## Pod Setup - Web\n\nWEB DOCS/, "the picked platform page, with real newlines");
    assert.match(generate!.prompt, /## Pod Setup - Postgres\n\nPOSTGRES DOCS/);
    assert.ok(!generate!.prompt.includes("REDIS DOCS"), "a page not picked is not loaded");
    assert.deepEqual(generate!.schema.properties.frontend.properties.repo.enum, ["shop"]);
    assert.deepEqual(generate!.schema.properties.services.items.properties.repo.enum, ["shop"], "the anchored service schema, resolved the same");

    assert.deepEqual(out.services, [
      { name: "frontend", port: 3000, cwd: "/workspaces/shop", scripts: { start: "yarn dev --hostname 0.0.0.0", install: "yarn install" }, env: { DATABASE_URL: "postgresql://u:p@127.0.0.1:5432/shop" } },
    ]);
    assert.deepEqual(Object.keys(out.containerFiles), ["docker-compose.yml"]);
    assert.deepEqual((yaml.load(out.containerFiles["docker-compose.yml"]) as any).services.postgres.image, "postgres:16");
    assert.deepEqual(out.pages, { platforms: ["Pod Setup - Web"], services: ["Pod Setup - Postgres"] });
    assert.equal(out.profile.repos[0].repo, "shop");
    assert.ok(Math.abs(out.cost - 0.03) < 1e-9);
  });

  it("fails the run, before any agent, when the root page is missing", async () => {
    agentCalls.length = 0;
    const missing = defineStep({
      type: "graph/graph-get",
      input: z.object({}).passthrough(),
      output: z.any(),
      async run(cfg: any) {
        if (cfg.name === "Pod Setup" && cfg.required) throw new Error('node not found: Concept "Pod Setup"');
        return fakeGet.run(cfg, {} as any);
      },
    });
    const r = await run({ repos: ["https://github.com/acme/shop"] }, { "graph/graph-get": missing });
    assert.equal(r.status, "error");
    assert.match(JSON.stringify(r.error), /node not found: Concept \\"Pod Setup\\"/);
    assert.equal(agentCalls.length, 0);
  });
});
