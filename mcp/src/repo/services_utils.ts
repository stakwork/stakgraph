import { detectLanguagesAndPkgFiles, extractEnvVarsFromRepo } from "../graph/utils.js";
import { selectSetupHints } from "../gitsee/agent/prompts/services.js";
import { addUsage, AiUsage } from "../aieo/src/index.js";

export type SetupProfile = {
  package_manager: string;
  setup_hints: string[];
  required_local_services: string[];
  services: Array<{
    name: string;
    cwd: string;
    framework?: string;
    dev_script?: string;
    port?: number;
  }>;
};

export type RepoFacts = {
  factsBlock: string;
  envVarNames: string[];
};

export async function buildRepoFacts(repoDir: string): Promise<RepoFacts> {
  const [detected, envVarsByFile] = await Promise.all([
    detectLanguagesAndPkgFiles(repoDir),
    extractEnvVarsFromRepo(repoDir),
  ]);

  const langLines = detected.map(({ language, pkgFile }) => {
    const rel = pkgFile.replace(repoDir + "/", "");
    return `  ${language} (${rel})`;
  });

  const envLines: string[] = [];
  for (const [file, vars] of Object.entries(envVarsByFile)) {
    const rel = file.replace(repoDir + "/", "");
    const names = Array.from(vars);
    envLines.push(`  ${rel}: [${names.join(", ")}]`);
  }

  const envVarNames = Array.from(
    new Set(Object.values(envVarsByFile).flatMap((vars) => Array.from(vars)))
  );

  const factsBlock = [
    "REPO FACTS (pre-scanned, do not re-discover these — use them as ground truth):",
    langLines.length > 0 ? `Languages detected:\n${langLines.join("\n")}` : "Languages detected: none",
    envLines.length > 0 ? `Env vars by file:\n${envLines.join("\n")}` : "Env vars: none found",
  ].join("\n\n");

  return { factsBlock, envVarNames };
}

function normalizeHintKey(value: string): string {
  return value.toLowerCase().replace(/[.\s-]+/g, "_").replace(/^next_js$/, "nextjs");
}

const DEPENDENCY_HINT_KEYS = new Set([
  "elasticsearch",
  "livekit",
  "mysql",
  "postgres",
  "rabbitmq",
  "redis",
  "s3_compatible_storage",
  "supabase",
]);

export function collectHintKeys(profile: SetupProfile): string[] {
  const keys = new Set<string>();
  const profileValues = [
    profile.package_manager,
    ...profile.setup_hints,
    ...profile.services.flatMap((service) => [
      service.framework ?? "",
      service.dev_script ?? "",
    ]),
  ];

  for (const value of profileValues) {
    const key = normalizeHintKey(value);
    if (key && !DEPENDENCY_HINT_KEYS.has(key)) keys.add(key);
  }

  for (const value of profile.required_local_services ?? []) {
    const key = normalizeHintKey(value);
    if (key) keys.add(key);
  }

  return Array.from(keys);
}

export function buildSelectedHints(profile: SetupProfile): string {
  return selectSetupHints(collectHintKeys(profile));
}

export function combineUsage(
  first: Partial<AiUsage> & { model?: string; provider?: string },
  second: Partial<AiUsage> & { model?: string; provider?: string }
) {
  const usage = addUsage(first, second);
  return {
    ...usage,
    model: second.model ?? first.model,
    provider: second.provider ?? first.provider,
  };
}
