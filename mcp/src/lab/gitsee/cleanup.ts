import { execSync } from "node:child_process";
import { readdirSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

/**
 * Manual cleanup for a gitsee boot-gate run that was KILLED (no teardown ran).
 * The verify-setup step tears down on a normal/errored finish, but if you stop
 * the whole process (Ctrl-C, kill the optimize), its `finally` never runs and the
 * booted stack is left up — a pm2 "frontend", our docker-compose services, and
 * any app-spawned stack like a `supabase start` CLI project (12 containers) or a
 * minio. This removes all of those. Leaves unrelated containers (Neo4j, etc.)
 * alone — it only touches pm2, supabase CLI stacks, and compose projects rooted
 * under the gitsee-lab tmp dir.
 *
 *   npx tsx src/lab/gitsee/cleanup.ts
 */
function sh(cmd: string): string {
  try {
    return execSync(cmd, { encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"] });
  } catch (e) {
    return (e as { stdout?: Buffer }).stdout?.toString() ?? "";
  }
}

const ids = (s: string) => s.split("\n").map((x) => x.trim()).filter(Boolean);

console.log("[gitsee-cleanup] pm2 delete all");
sh("npx -y pm2 delete all");

// 1. supabase CLI stacks (started by the app via `supabase start`)
const supa = ids(sh(`docker ps -aq --filter "label=com.supabase.cli.project"`));
if (supa.length) {
  console.log(`[gitsee-cleanup] removing ${supa.length} supabase container(s)`);
  sh(`docker rm -fv ${supa.join(" ")}`);
}

// 2. compose projects rooted under the gitsee-lab tmp dir (our docker-compose.yml)
const composeLines = sh(
  `docker ps -a --format '{{.ID}}|{{.Label "com.docker.compose.project.working_dir"}}'`,
)
  .split("\n")
  .filter(Boolean);
const ours = composeLines.filter((l) => l.includes("gitsee-lab")).map((l) => l.split("|")[0]);
if (ours.length) {
  console.log(`[gitsee-cleanup] removing ${ours.length} gitsee compose container(s)`);
  sh(`docker rm -fv ${ours.join(" ")}`);
}

// 3. prune the now-dangling networks/volumes those stacks created
sh("docker network prune -f");
const vols = ids(sh(`docker volume ls -q --filter "name=supabase_"`));
if (vols.length) sh(`docker volume rm ${vols.join(" ")}`);

// 4. strip files verify-setup staged into each reused clone (pm2.config.js,
//    docker-compose.yml, .pod-config) so a killed run doesn't leave a "prior
//    attempt" that distracts the next explore agent.
const labRoot = join(tmpdir(), "gitsee-lab");
if (existsSync(labRoot)) {
  for (const ws of readdirSync(labRoot)) {
    const wsDir = join(labRoot, ws);
    for (const f of ["pm2.config.js", "pm2.config.cjs", "docker-compose.yml"]) {
      try {
        rmSync(join(wsDir, f), { force: true });
      } catch {
        /* ignore */
      }
    }
    try {
      rmSync(join(wsDir, ".pod-config"), { recursive: true, force: true });
    } catch {
      /* ignore */
    }
  }
  console.log(`[gitsee-cleanup] stripped staged config files from ${labRoot}/*`);
}

console.log("[gitsee-cleanup] done. Remaining containers:");
console.log(sh("docker ps --format '{{.Names}}\\t{{.Ports}}'") || "(none)");
