import { z, defineStep, dumpYaml, type StepContext, type StrutCapabilities } from "strut";
import { join, posix } from "node:path";

/**
 * pod-setup/finish — the generation agent's draft made into what hive's pool
 * config takes (`pod/config-set`'s input): `services` and `containerFiles`.
 * Whatever code can guarantee is guaranteed here, so the model supplies only
 * judgement (strut plans/onboarding.md §10):
 *   - the person-facing service is its own field, named `frontend` here —
 *     the name hive and staklink look the service up by;
 *   - a service names a repository (one of the checkouts) and a directory in
 *     it, which must exist; its cwd is written here, /workspaces/<repo>/<dir>;
 *   - names are unique pm2 names, ports unique integers;
 *   - an env var that is a service's CONFIG (PORT, *_COMMAND) moves to its
 *     port / scripts, since hive generates those;
 *   - a compose service's name used as a host in an env value becomes
 *     127.0.0.1 — not localhost, which can resolve to ::1 (where the port
 *     is not published), a MySQL client takes for its socket, and an S3 SDK
 *     makes `bucket.localhost`;
 *   - the compose file is assembled, not written: no `app` service (the pod
 *     is the app's container), every named volume and network declared,
 *     serialized by code so it always parses.
 * Every fix is listed in `notes`. What cannot be fixed fails, naming it.
 * Self-contained: imports only `strut` and Node builtins. The one look at
 * the disk — does a service's directory exist in its checkout, which is
 * outside the run's own files — goes through `ctx.services.shell`
 * (`test -d`), never `node:fs` (strut src/step-lint.ts).
 */

export const FRONTEND = "frontend";
export const DEFAULT_DOCKERFILE = "FROM ghcr.io/stakwork/staklink-js:latest";
/** Where the app reaches a compose service: the port it publishes, here. */
export const LOOPBACK = "127.0.0.1";
const SCRIPTS = ["start", "install", "build", "test", "e2eTest", "preStart", "postStart", "rebuild", "reset"] as const;
type Script = (typeof SCRIPTS)[number];
/** The env vars hive generates from a service's port and scripts. */
const CONFIG_VARS: Record<string, "port" | Script> = {
  PORT: "port",
  INSTALL_COMMAND: "install",
  BUILD_COMMAND: "build",
  TEST_COMMAND: "test",
  E2E_TEST_COMMAND: "e2eTest",
  PRE_START_COMMAND: "preStart",
  POST_START_COMMAND: "postStart",
  REBUILD_COMMAND: "rebuild",
  RESET_COMMAND: "reset",
};
const PM2_NAME = /^[A-Za-z0-9._-]{1,100}$/;
const ENV_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;
const VOLUME_NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;

export interface DraftService {
  name?: string;
  repo: string;
  subdir?: string;
  port: number;
  scripts: Partial<Record<Script, string>>;
  env?: Record<string, unknown>;
}

export interface Draft {
  frontend: DraftService;
  services?: DraftService[];
  local_services?: Record<string, unknown>;
  dockerfile?: string;
}

/** hive's ServiceConfig. */
export interface ServiceConfig {
  name: string;
  port: number;
  cwd: string;
  scripts: { start: string } & Partial<Record<Script, string>>;
  env?: Record<string, string>;
}

/** Every URL's host in `value` that is one of `hosts`, made 127.0.0.1: the
 *  authority after `//` is split at its last `@`, so a user name that
 *  happens to equal a service's (`postgres://postgres:pw@postgres:5432`) is
 *  left as it is. */
export function localizeHosts(value: string, hosts: Set<string>): string {
  return value.replace(/\/\/([^/?#\s"']*)/g, (whole, authority: string) => {
    const at = authority.lastIndexOf("@");
    const userinfo = at >= 0 ? authority.slice(0, at + 1) : "";
    const hostport = authority.slice(at + 1);
    const colon = hostport.indexOf(":");
    const host = colon >= 0 ? hostport.slice(0, colon) : hostport;
    return hosts.has(host) ? `//${userinfo}${LOOPBACK}${colon >= 0 ? hostport.slice(colon) : ""}` : whole;
  });
}

/** A directory inside a repository, as given: trimmed, no `./`, no slashes at the ends; never absolute or above the repository. */
function cleanSubdir(subdir: string | undefined, where: string): string {
  const s = String(subdir ?? "").trim().replace(/^\.\/+/, "").replace(/^\/+|\/+$/g, "");
  if (!s || s === ".") return "";
  const n = posix.normalize(s);
  if (n.startsWith("..") || posix.isAbsolute(n)) throw new Error(`${where}: subdir "${subdir}" leaves the repository`);
  return n;
}

/** Whether `path` is a directory — the caller's look at the disk. */
export type IsDir = (path: string) => Promise<boolean>;

export async function finishSetup(
  draft: Draft,
  repos: Array<{ name: string; path: string }>,
  isDir: IsDir,
): Promise<{ services: ServiceConfig[]; containerFiles: Record<string, string>; notes: string[] }> {
  const notes: string[] = [];
  if (!draft || typeof draft !== "object" || !draft.frontend) throw new Error("pod-setup/finish: the draft has no `frontend`");

  // ── the compose services: everything but the app itself ─────────────────
  const local: Record<string, Record<string, unknown>> = {};
  for (const [name, svc] of Object.entries(draft.local_services ?? {})) {
    if (name === "app") {
      notes.push("dropped the compose service `app`: the pod is the app's container");
      continue;
    }
    if (!svc || typeof svc !== "object" || Array.isArray(svc)) throw new Error(`pod-setup/finish: compose service "${name}" is not an object`);
    local[name] = svc as Record<string, unknown>;
  }
  const hosts = new Set<string>(Object.keys(local));
  for (const svc of Object.values(local)) {
    for (const k of ["container_name", "hostname"]) if (typeof svc[k] === "string" && svc[k]) hosts.add(String(svc[k]));
  }

  // ── the pm2 services ─────────────────────────────────────────────────────
  const byName = new Map(repos.map((r) => [r.name, r]));
  const drafts: Array<[string, DraftService]> = [
    [FRONTEND, draft.frontend],
    ...(draft.services ?? []).map((s, i): [string, DraftService] => [String(s?.name ?? "").trim() || `services[${i}]`, s]),
  ];
  const names = new Set<string>();
  const ports = new Map<number, string>();
  const services: ServiceConfig[] = [];
  for (const [i, [name, s]] of drafts.entries()) {
    const where = `pod-setup/finish: service "${name}"`;
    if (i > 0) {
      if (!s?.name || !PM2_NAME.test(name)) throw new Error(`${where}: a service needs a name of letters, digits, \`.\`, \`_\` or \`-\``);
      if (name.toLowerCase() === FRONTEND) throw new Error(`${where}: the person-facing service is the \`frontend\` field, not a service of that name`);
    }
    if (names.has(name)) throw new Error(`${where}: two services share the name`);
    names.add(name);

    const repo = byName.get(String(s?.repo ?? ""));
    if (!repo) throw new Error(`${where}: repo "${s?.repo}" is not one of the checked-out repositories (${[...byName.keys()].join(", ")})`);
    const subdir = cleanSubdir(s.subdir, where);
    if (subdir) {
      if (!(await isDir(join(repo.path, subdir)))) throw new Error(`${where}: "${subdir}" is not a directory in ${repo.name}`);
    }
    const port = Number(s.port);
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error(`${where}: port ${JSON.stringify(s.port)} is not a port`);
    if (ports.has(port)) throw new Error(`${where}: port ${port} is also "${ports.get(port)}"'s`);
    ports.set(port, name);

    const scripts: Partial<Record<Script, string>> = {};
    for (const k of SCRIPTS) {
      const v = s.scripts?.[k];
      if (typeof v === "string" && v.trim()) scripts[k] = v.trim();
    }
    const env: Record<string, string> = {};
    for (const [k, raw] of Object.entries(s.env ?? {})) {
      if (!ENV_NAME.test(k)) throw new Error(`${where}: "${k}" is not an env var name`);
      if (raw === null || raw === undefined) continue;
      let v = typeof raw === "string" ? raw : String(raw);
      const config = CONFIG_VARS[k];
      if (config === "port") {
        if (v !== String(port)) notes.push(`${name}: dropped env PORT=${v}; the service's port is ${port}`);
        continue;
      }
      if (config) {
        if (!scripts[config] && v.trim()) {
          scripts[config] = v.trim();
          notes.push(`${name}: moved env ${k} to its ${config} command`);
        } else {
          notes.push(`${name}: dropped env ${k}; its ${config} command is set`);
        }
        continue;
      }
      // A compose service's name as a host → 127.0.0.1 (the service publishes
      // its port there): in a URL, or as the whole value of a *_HOST var.
      const before = v;
      v = localizeHosts(v, hosts);
      if (/(^|_)(HOST|HOSTNAME)$/.test(k) && hosts.has(v)) v = LOOPBACK;
      if (v !== before) notes.push(`${name}: env ${k} named a compose service as its host; it is ${LOOPBACK}`);
      env[k] = v;
    }
    if (!scripts.start) throw new Error(`${where}: no start command`);
    services.push({
      name,
      port,
      cwd: `/workspaces/${repo.name}${subdir ? `/${subdir}` : ""}`,
      scripts: scripts as ServiceConfig["scripts"],
      ...(Object.keys(env).length ? { env } : {}),
    });
  }

  // ── the compose file ─────────────────────────────────────────────────────
  const volumes = new Set<string>();
  const networks = new Set<string>();
  for (const svc of Object.values(local)) {
    for (const v of Array.isArray(svc.volumes) ? svc.volumes : []) {
      const source = typeof v === "string" ? v.split(":")[0]! : v && typeof v === "object" && (v as any).type === "volume" ? String((v as any).source ?? "") : "";
      if (source && VOLUME_NAME.test(source)) volumes.add(source);
    }
    const nets = Array.isArray(svc.networks) ? svc.networks : svc.networks && typeof svc.networks === "object" ? Object.keys(svc.networks) : [];
    for (const n of nets) if (typeof n === "string" && n !== "default") networks.add(n);
  }
  const compose: Record<string, unknown> = { services: local };
  if (volumes.size) compose.volumes = Object.fromEntries([...volumes].sort().map((v) => [v, {}]));
  if (networks.size) compose.networks = Object.fromEntries([...networks].sort().map((n) => [n, {}]));
  const containerFiles: Record<string, string> = {
    "docker-compose.yml": Object.keys(local).length ? dumpYaml(compose) : "services: {}\n",
  };
  const dockerfile = String(draft.dockerfile ?? "").trim();
  if (dockerfile && dockerfile !== DEFAULT_DOCKERFILE) containerFiles.Dockerfile = `${dockerfile}\n`;
  return { services, containerFiles, notes };
}

export default defineStep({
  type: "pod-setup/finish",
  description:
    "Make a drafted pod setup into what hive's pool config takes: `services` (hive's ServiceConfig, the person-facing one named `frontend`, each " +
    "cwd /workspaces/<repo>/<dir> from a checked-out repository and a directory that exists in it) and `containerFiles` (docker-compose.yml " +
    "assembled from the drafted compose services, no `app` service, every named volume and network declared; a Dockerfile only when it is not " +
    "the default). Env vars that are a service's config move to its port and commands, and a compose service's name used as a host becomes " +
    "127.0.0.1. Output: { services, containerFiles, notes (each fix made) }; a draft it cannot fix fails, naming what is wrong.",
  input: z.object({
    repos: z.array(z.object({ name: z.string(), path: z.string() }).passthrough()).min(1).describe("pod-setup/repo-facts' repos"),
    raw: z.any().describe("the generation agent's object: { frontend, services?, local_services?, dockerfile? }"),
  }),
  output: z.object({ services: z.array(z.any()), containerFiles: z.record(z.string(), z.string()), notes: z.array(z.string()) }),
  async run(cfg, ctx) {
    const shell = (ctx as StepContext<StrutCapabilities>).services?.shell;
    if (!shell) throw new Error("shell capability unavailable");
    // `test -d` in the checkout: exit 0 is a directory; a checkout that is
    // not there at all fails the spawn, which is the same answer.
    const isDir: IsDir = (path) =>
      shell({ cmd: "test", args: ["-d", path], cwd: posix.dirname(path) })
        .then((r) => r.code === 0)
        .catch(() => false);
    return finishSetup(cfg.raw as Draft, cfg.repos, isDir);
  },
});
