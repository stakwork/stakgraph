import { execSync } from "node:child_process";

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

console.log("[gitsee-cleanup] done. Remaining containers:");
console.log(sh("docker ps --format '{{.Names}}\\t{{.Ports}}'") || "(none)");
