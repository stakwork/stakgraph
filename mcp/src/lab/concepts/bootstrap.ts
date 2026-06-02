import { Storage } from "./store/index.js";
import { Concept, Usage } from "./types.js";
import { generateSlug, makeRepoId } from "./store/utils.js";
import { get_context } from "../../repo/agent.js";
import { normalizeUsage } from "../../aieo/src/usage.js";
import { DOC_GUIDELINES } from "./llm.js";
import * as fs from "fs";
import * as path from "path";

const DEFAULT_BOOTSTRAP_LOOKBACK_DAYS = 10;

/**
 * Bootstrap result returned to the caller
 */
export interface BootstrapResult {
  features: Concept[];
  usage: Usage;
}

/**
 * Repo size category for calibrating feature count
 */
type RepoSize = "small" | "medium" | "large";

/**
 * Count source files in a repo to determine sizing hint
 */
function countSourceFiles(repoPath: string): number {
  const sourceExtensions = new Set([
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".py",
    ".go",
    ".rs",
    ".rb",
    ".swift",
    ".kt",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".vue",
    ".svelte",
  ]);

  let count = 0;

  function walk(dir: string) {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const name = entry.name;
      // Skip common non-source directories
      if (
        entry.isDirectory() &&
        !name.startsWith(".") &&
        name !== "node_modules" &&
        name !== "vendor" &&
        name !== "dist" &&
        name !== "build" &&
        name !== "target" &&
        name !== ".next" &&
        name !== "coverage" &&
        name !== "__pycache__" &&
        name !== "venv" &&
        name !== ".venv"
      ) {
        walk(path.join(dir, name));
      } else if (entry.isFile()) {
        const ext = path.extname(name).toLowerCase();
        if (sourceExtensions.has(ext)) {
          count++;
        }
      }
    }
  }

  walk(repoPath);
  return count;
}

/**
 * Classify repo size and determine target feature count range
 */
function classifyRepo(fileCount: number): {
  size: RepoSize;
  min: number;
  max: number;
} {
  if (fileCount <= 30) {
    return { size: "small", min: 2, max: 5 };
  } else if (fileCount <= 200) {
    return { size: "medium", min: 4, max: 10 };
  } else {
    return { size: "large", min: 6, max: 15 };
  }
}

/**
 * Bootstrap prompt configuration — the experiment surface. NOT baked in here:
 * the prompt text lives in the calling workflow's `params` block
 * (`workflows/bootstrap-then-process.yaml`) and is passed in via the
 * `concepts/bootstrap-explore` step config (mirrors how `concepts/decide`
 * sources its prompt). This keeps the big prompt visible + editable + tunable
 * in the vein UI rather than hidden in code.
 *
 *   - `system`   — the system override (no placeholders).
 *   - `template` — the exploration prompt, with `{slot}` placeholders the step
 *     fills with runtime-computed values: {owner}, {repo}, {size},
 *     {fileCount}, {min}, {max}.
 */
export interface BootstrapPromptConfig {
  system: string;
  template: string;
}

/**
 * Fill the `{slot}` placeholders in a bootstrap prompt template with the
 * runtime-computed sizing values. (Single-brace slots, distinct from vein's
 * `{{ }}` templates, so they survive config resolution untouched.)
 */
function fillBootstrapPrompt(
  template: string,
  vals: {
    owner: string;
    repo: string;
    size: RepoSize;
    fileCount: number;
    min: number;
    max: number;
  }
): string {
  return template
    .replaceAll("{owner}", vals.owner)
    .replaceAll("{repo}", vals.repo)
    .replaceAll("{size}", vals.size)
    .replaceAll("{fileCount}", String(vals.fileCount))
    .replaceAll("{min}", String(vals.min))
    .replaceAll("{max}", String(vals.max));
}

/**
 * JSON schema for the bootstrap agent's structured output
 */
const BOOTSTRAP_SCHEMA = {
  type: "object",
  properties: {
    features: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "Human-readable feature name",
          },
          description: {
            type: "string",
            description: "1-2 sentence description of the capability",
          },

          summary: {
            type: "string",
            description:
              "10-20 line high-level explanation of how the feature works",
          },
          files: {
            type: "array",
            items: { type: "string" },
            description:
              "Core source file paths relative to repo root (e.g. 'src/auth/login.ts')",
          },
        },
        required: ["name", "description", "summary", "files"],
      },
    },
  },
  required: ["features"],
};

/**
 * Bootstrap a brand-new repo by exploring the codebase and creating
 * an initial set of features. Uses get_context (agentic exploration)
 * with a structured schema to produce features in a single pass.
 *
 * Returns the created features and token usage.
 */
export async function bootstrapConcepts(
  owner: string,
  repo: string,
  repoPath: string,
  storage: Storage,
  promptConfig: BootstrapPromptConfig,
  sessionId?: string,
  lookbackDays: number = DEFAULT_BOOTSTRAP_LOOKBACK_DAYS
): Promise<BootstrapResult> {
  const repoId = `${owner}/${repo}`;

  console.log(`\n🚀 Bootstrap mode: exploring ${repoId} to seed initial features...`);

  // 1. Count source files for sizing hint
  const fileCount = countSourceFiles(repoPath);
  const sizing = classifyRepo(fileCount);
  console.log(
    `   📊 Found ${fileCount} source files (${sizing.size} repo, targeting ${sizing.min}-${sizing.max} features)`
  );

  // 2. Fill the prompt template (the experiment surface) and call get_context.
  const prompt = fillBootstrapPrompt(promptConfig.template, {
    owner,
    repo,
    size: sizing.size,
    fileCount,
    min: sizing.min,
    max: sizing.max,
  });

  const result = await get_context(prompt, repoPath, {
    schema: BOOTSTRAP_SCHEMA,
    systemOverride: promptConfig.system,
    sessionId,
    isolatedContext: true,
  });

  const decision = result.content as {
    features: Array<{
      name: string;
      description: string;
      summary: string;
      files: string[];
    }>;
  };

  // 3. Create and save features
  const now = new Date();
  const features: Concept[] = [];

  for (const f of decision.features || []) {
    const slug = generateSlug(f.name);
    const featureId = makeRepoId(repoId, slug);

    const feature: Concept = {
      id: featureId,
      repo: repoId,
      name: f.name,
      description: f.description,
      prNumbers: [],
      commitShas: [],
      createdAt: now,
      lastUpdated: now,
      documentation: f.summary,
    };

    await storage.saveConcept(feature);
    await storage.saveDocumentation(featureId, f.summary);
    features.push(feature);

    // Link core files identified by the LLM
    const files = f.files || [];
    if (files.length > 0) {
      const linked = await storage.linkConceptToFilesByPaths(featureId, files);
      console.log(`   ✅ Created feature: ${f.name} (${featureId}) — linked ${linked} files`);
    } else {
      console.log(`   ✅ Created feature: ${f.name} (${featureId})`);
    }
  }

  // 4. Set checkpoint to N days ago so processRepo replays recent changes
  //    This captures recent activity (e.g. someone forked a repo and added a few PRs)
  const lookback = new Date(now.getTime() - lookbackDays * 24 * 60 * 60 * 1000);
  await storage.setChronologicalCheckpoint(repoId, {
    lastProcessedTimestamp: lookback.toISOString(),
    processedAtTimestamp: [],
  });
  console.log(`   📌 Checkpoint set to ${lookbackDays} days ago — processRepo will pick up recent changes`);

  console.log(
    `\n🎯 Bootstrap complete: created ${features.length} features for ${repoId}`
  );

  return {
    features,
    usage: result.usage,
  };
}

/**
 * Explore a newly created feature to generate initial documentation.
 * Called when the incremental flow discovers a feature not in the bootstrap set.
 * Only runs if the feature has no existing documentation.
 */
export async function exploreNewConcept(
  feature: Concept,
  repoPath: string,
  storage: Storage,
  sessionId?: string
): Promise<Usage> {
  if (feature.documentation && feature.documentation.trim().length > 0) {
    return normalizeUsage();
  }

  console.log(`   🔍 Exploring codebase for new feature: ${feature.name}...`);

  const result = await get_context(
    `Generate SUCCINCT documentation for the "${feature.name}" feature in this codebase.

Description: ${feature.description}

${DOC_GUIDELINES.include}

${DOC_GUIDELINES.avoid}

Target length: 30-80 lines of markdown.`,
    repoPath,
    {
      systemOverride: `You are a software architect generating concise feature documentation. Use the provided tools to explore the repository and find the key files, components, models, and patterns related to this feature. Be thorough but focused.`,
      sessionId,
      isolatedContext: true,
    }
  );

  const documentation = typeof result.content === "string"
    ? result.content
    : result.final;

  feature.documentation = documentation;
  await storage.saveConcept(feature);
  await storage.saveDocumentation(feature.id, documentation);

  console.log(`   ✅ Documentation generated for: ${feature.name}`);

  return result.usage;
}
