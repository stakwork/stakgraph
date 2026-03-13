import { Storage } from "./store/index.js";
import { Feature, Usage } from "./types.js";
import { generateSlug, makeRepoId } from "./store/utils.js";
import { get_context } from "../repo/agent.js";
import { DOC_GUIDELINES } from "./llm.js";
import * as fs from "fs";
import * as path from "path";

/**
 * Bootstrap result returned to the caller
 */
export interface BootstrapResult {
  features: Feature[];
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
 * Build the bootstrap prompt with sizing hints
 */
function buildBootstrapPrompt(
  owner: string,
  repo: string,
  fileCount: number,
  sizing: { size: RepoSize; min: number; max: number }
): string {
  return `Explore the repository "${owner}/${repo}" and identify its core features.

This is a ${sizing.size} repository (~${fileCount} source files). Aim for ${sizing.min}-${sizing.max} features.

A "feature" is a distinct user-facing capability or business function — something a product owner or end user would recognize. Examples: "Authentication System", "Payment Processing", "Real-time Notifications", "Search and Filtering".

Do NOT create features for:
- Build tooling, CI/CD, or infrastructure
- Individual utility functions or helpers
- Code style, linting, or formatting
- Generic "bug fixes" or "refactoring"
- Testing infrastructure (unless it IS the product)

For each feature provide:
- **name**: A clear, non-technical name (no framework/library names)
- **description**: 1-2 sentences explaining what this capability does for users
- **summary**: SUCCINCT high-level documentation (30-80 lines markdown) for this feature's CURRENT state

**Summary requirements** — focus on what developers need to know to work on this feature:
${DOC_GUIDELINES.include}

${DOC_GUIDELINES.avoid}

Prefer fewer, broader features over many granular ones. Two closely related capabilities should be one feature, not two.`;
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
        },
        required: ["name", "description", "summary"],
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
export async function bootstrapFeatures(
  owner: string,
  repo: string,
  repoPath: string,
  storage: Storage
): Promise<BootstrapResult> {
  const repoId = `${owner}/${repo}`;

  console.log(`\n🚀 Bootstrap mode: exploring ${repoId} to seed initial features...`);

  // 1. Count source files for sizing hint
  const fileCount = countSourceFiles(repoPath);
  const sizing = classifyRepo(fileCount);
  console.log(
    `   📊 Found ${fileCount} source files (${sizing.size} repo, targeting ${sizing.min}-${sizing.max} features)`
  );

  // 2. Build prompt and call get_context with structured schema
  const prompt = buildBootstrapPrompt(owner, repo, fileCount, sizing);

  const result = await get_context(prompt, repoPath, {
    schema: BOOTSTRAP_SCHEMA,
    systemOverride: `You are a software architect analyzing a codebase to identify its core features. Use the provided tools to explore the repository structure, read key files (README, entry points, route definitions, main modules), and identify the distinct user-facing capabilities this software provides. Be thorough but focused — read enough to understand what each feature does, but don't try to read every file.`,
  });

  const decision = result.content as {
    features: Array<{
      name: string;
      description: string;
      summary: string;
    }>;
  };

  // 3. Create and save features
  const now = new Date();
  const features: Feature[] = [];

  for (const f of decision.features || []) {
    const slug = generateSlug(f.name);
    const featureId = makeRepoId(repoId, slug);

    const feature: Feature = {
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

    await storage.saveFeature(feature);
    await storage.saveDocumentation(featureId, f.summary);
    features.push(feature);

    console.log(`   ✅ Created feature: ${f.name} (${featureId})`);
  }

  // 4. Set checkpoint to now so processRepo skips historical replay
  await storage.setChronologicalCheckpoint(repoId, {
    lastProcessedTimestamp: now.toISOString(),
    processedAtTimestamp: [],
  });

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
export async function exploreNewFeature(
  feature: Feature,
  repoPath: string,
  storage: Storage
): Promise<Usage> {
  if (feature.documentation && feature.documentation.trim().length > 0) {
    return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
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
      systemOverride: `You are a software architect generating concise feature documentation. Use the provided tools to explore the repository and find the key files, components, and patterns related to this feature. Be thorough but focused.`,
    }
  );

  const documentation = typeof result.content === "string"
    ? result.content
    : result.final;

  feature.documentation = documentation;
  await storage.saveFeature(feature);
  await storage.saveDocumentation(feature.id, documentation);

  console.log(`   ✅ Documentation generated for: ${feature.name}`);

  return result.usage;
}
