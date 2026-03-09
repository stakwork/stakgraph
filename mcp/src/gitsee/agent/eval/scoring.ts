/**
 * Scoring function for evaluating generated pm2/docker-compose configs
 * against gold-standard references.
 *
 * Uses structural comparison (parsed config analysis) rather than
 * string matching, so minor formatting differences don't tank scores.
 */

import type { ScoreBreakdown, TrainingExample } from "./types.js";

// ---------------------------------------------------------------------------
// Weights for each sub-score (must sum to 1.0)
// ---------------------------------------------------------------------------
const WEIGHTS = {
  format: 0.10,
  pm2_structure: 0.20,
  docker_structure: 0.20,
  env_vars: 0.15,
  app_service: 0.15,
  cwd: 0.10,
  frontend_naming: 0.10,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Try to extract a JS object from pm2.config.js content */
function parsePm2(content: string): any | null {
  try {
    // Strip module.exports = and trailing semicolons, then eval-ish parse
    let cleaned = content.trim();
    cleaned = cleaned.replace(/^module\.exports\s*=\s*/, "");
    cleaned = cleaned.replace(/;\s*$/, "");
    // Handle trailing commas (common in JS but invalid JSON)
    cleaned = cleaned.replace(/,(\s*[}\]])/g, "$1");
    // Try JSON parse first
    return JSON.parse(cleaned);
  } catch {
    // Fallback: try to extract apps array via regex
    try {
      const appsMatch = content.match(/apps\s*:\s*\[([\s\S]*)\]/);
      if (appsMatch) {
        // Extract service names at minimum
        const nameMatches = [...content.matchAll(/name\s*:\s*["']([^"']+)["']/g)];
        const scriptMatches = [...content.matchAll(/script\s*:\s*["']([^"']+)["']/g)];
        const cwdMatches = [...content.matchAll(/cwd\s*:\s*["']([^"']+)["']/g)];
        return {
          apps: nameMatches.map((m, i) => ({
            name: m[1],
            script: scriptMatches[i]?.[1] || "",
            cwd: cwdMatches[i]?.[1] || "",
            env: extractEnvBlock(content, m.index || 0),
          })),
        };
      }
    } catch {}
    return null;
  }
}

/** Extract env keys from a pm2 service block starting near `startIdx` */
function extractEnvBlock(content: string, startIdx: number): Record<string, string> {
  const envMatch = content.substring(startIdx).match(/env\s*:\s*\{([^}]+)\}/);
  if (!envMatch) return {};
  const envStr = envMatch[1];
  const pairs: Record<string, string> = {};
  for (const line of envStr.split("\n")) {
    const kv = line.match(/(\w+)\s*:\s*["']?([^"',\n]+)["']?/);
    if (kv) pairs[kv[1]] = kv[2].trim();
  }
  return pairs;
}

/** Parse docker-compose YAML loosely (we avoid a yaml dep - just extract services) */
function parseDockerCompose(content: string): {
  services: string[];
  hasAppService: boolean;
  appServiceCorrect: boolean;
  networks: string[];
  volumes: string[];
} {
  const services: string[] = [];
  const networks: string[] = [];
  const volumes: string[] = [];

  // Extract service names (lines that are indented exactly 2 spaces under "services:")
  const servicesSection = content.match(/services:\s*\n([\s\S]*?)(?=\nnetworks:|\nvolumes:|\n[a-z]|\s*$)/);
  if (servicesSection) {
    const serviceMatches = servicesSection[1].matchAll(/^  (\w[\w-]*):/gm);
    for (const m of serviceMatches) {
      services.push(m[1]);
    }
  }

  // Check if app service exists and has the required structure
  const hasAppService = services.includes("app");
  const appServiceCorrect =
    hasAppService &&
    content.includes("../..:/workspaces:cached") &&
    content.includes("sleep infinity") &&
    content.includes("app_network");

  // Extract network names
  const networksSection = content.match(/networks:\s*\n([\s\S]*?)(?=\nvolumes:|\nservices:|\s*$)/);
  if (networksSection) {
    const netMatches = networksSection[1].matchAll(/^  (\w[\w-]*):/gm);
    for (const m of netMatches) {
      networks.push(m[1]);
    }
  }

  // Extract volume names
  const volumesSection = content.match(/^volumes:\s*\n([\s\S]*?)$/m);
  if (volumesSection) {
    const volMatches = volumesSection[1].matchAll(/^  (\w[\w-]*):/gm);
    for (const m of volMatches) {
      volumes.push(m[1]);
    }
  }

  return { services, hasAppService, appServiceCorrect, networks, volumes };
}

/** Jaccard similarity between two sets */
function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  const intersection = new Set([...a].filter((x) => b.has(x)));
  const union = new Set([...a, ...b]);
  return intersection.size / union.size;
}

/** Fraction of expected items present in actual */
function recall(expected: Set<string>, actual: Set<string>): number {
  if (expected.size === 0) return 1;
  const found = [...expected].filter((x) => actual.has(x));
  return found.length / expected.size;
}

// ---------------------------------------------------------------------------
// Main scoring function
// ---------------------------------------------------------------------------

export function score(
  generated_pm2: string,
  generated_docker_compose: string,
  example: TrainingExample
): ScoreBreakdown {
  // ------- Format score -------
  const hasPm2 = generated_pm2.trim().length > 10;
  const hasDocker = generated_docker_compose.trim().length > 10;
  const format_score = (hasPm2 ? 0.5 : 0) + (hasDocker ? 0.5 : 0);

  // ------- Parse configs -------
  const genPm2 = parsePm2(generated_pm2);
  const goldPm2 = parsePm2(example.gold_pm2);
  const genDocker = parseDockerCompose(generated_docker_compose);
  const goldDocker = parseDockerCompose(example.gold_docker_compose);

  // ------- PM2 structure score -------
  let pm2_structure_score = 0;
  if (genPm2 && goldPm2) {
    const genNames = new Set<string>((genPm2.apps || []).map((a: any) => a.name as string));
    const goldNames = new Set<string>((goldPm2.apps || []).map((a: any) => a.name as string));
    // Service count similarity
    const countScore =
      genPm2.apps?.length && goldPm2.apps?.length
        ? 1 - Math.abs(genPm2.apps.length - goldPm2.apps.length) / Math.max(genPm2.apps.length, goldPm2.apps.length)
        : 0;
    // Service name overlap
    const nameScore = jaccard(genNames, goldNames);
    pm2_structure_score = 0.4 * countScore + 0.6 * nameScore;
  }

  // ------- Docker structure score -------
  const genDockerServices = new Set(genDocker.services);
  const goldDockerServices = new Set(goldDocker.services);
  const docker_structure_score = jaccard(genDockerServices, goldDockerServices);

  // ------- Env vars score -------
  let env_vars_score = 0;
  if (genPm2 && goldPm2) {
    let totalRecall = 0;
    let count = 0;
    for (const goldApp of goldPm2.apps || []) {
      const genApp = (genPm2.apps || []).find((a: any) => a.name === goldApp.name);
      if (genApp && goldApp.env && genApp.env) {
        const goldKeys = new Set(Object.keys(goldApp.env));
        const genKeys = new Set(Object.keys(genApp.env));
        // Check key recall (are all expected env vars present?)
        totalRecall += recall(goldKeys, genKeys);
        count++;
        // Also check critical env var values match
        const criticalKeys = ["PORT", "DATABASE_URL", "INSTALL_COMMAND"];
        let valueMatches = 0;
        let criticalCount = 0;
        for (const key of criticalKeys) {
          if (goldApp.env[key]) {
            criticalCount++;
            if (genApp.env[key] && normalizeValue(genApp.env[key]) === normalizeValue(goldApp.env[key])) {
              valueMatches++;
            }
          }
        }
        if (criticalCount > 0) {
          totalRecall = (totalRecall + valueMatches / criticalCount) / 2;
        }
      }
    }
    env_vars_score = count > 0 ? totalRecall / count : 0;
  }

  // ------- App service score -------
  const app_service_score = genDocker.appServiceCorrect ? 1 : genDocker.hasAppService ? 0.5 : 0;

  // ------- CWD score -------
  let cwd_score = 0;
  if (genPm2 && goldPm2) {
    let matches = 0;
    let total = 0;
    for (const goldApp of goldPm2.apps || []) {
      const genApp = (genPm2.apps || []).find((a: any) => a.name === goldApp.name);
      if (genApp && goldApp.cwd) {
        total++;
        if (genApp.cwd === goldApp.cwd) {
          matches++;
        } else if (genApp.cwd && goldApp.cwd && genApp.cwd.includes(goldApp.cwd.split("/").pop() || "")) {
          matches += 0.5; // partial credit for getting the subdir right
        }
      }
    }
    cwd_score = total > 0 ? matches / total : 0;
  }

  // ------- Frontend naming score -------
  let frontend_naming_score = 0;
  if (genPm2) {
    const hasFrontend = (genPm2.apps || []).some((a: any) => a.name === "frontend");
    frontend_naming_score = hasFrontend ? 1 : 0;
  }

  // ------- Weighted total -------
  const total =
    WEIGHTS.format * format_score +
    WEIGHTS.pm2_structure * pm2_structure_score +
    WEIGHTS.docker_structure * docker_structure_score +
    WEIGHTS.env_vars * env_vars_score +
    WEIGHTS.app_service * app_service_score +
    WEIGHTS.cwd * cwd_score +
    WEIGHTS.frontend_naming * frontend_naming_score;

  return {
    format_score,
    pm2_structure_score,
    docker_structure_score,
    env_vars_score,
    app_service_score,
    cwd_score,
    frontend_naming_score,
    total,
  };
}

function normalizeValue(v: string): string {
  return v.trim().replace(/["']/g, "").toLowerCase();
}

/** Pretty-print a score breakdown */
export function formatScore(s: ScoreBreakdown): string {
  return [
    `  format:           ${(s.format_score * 100).toFixed(0)}%`,
    `  pm2_structure:    ${(s.pm2_structure_score * 100).toFixed(0)}%`,
    `  docker_structure: ${(s.docker_structure_score * 100).toFixed(0)}%`,
    `  env_vars:         ${(s.env_vars_score * 100).toFixed(0)}%`,
    `  app_service:      ${(s.app_service_score * 100).toFixed(0)}%`,
    `  cwd:              ${(s.cwd_score * 100).toFixed(0)}%`,
    `  frontend_naming:  ${(s.frontend_naming_score * 100).toFixed(0)}%`,
    `  ────────────────────`,
    `  TOTAL:            ${(s.total * 100).toFixed(1)}%`,
  ].join("\n");
}
