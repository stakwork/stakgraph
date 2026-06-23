import { Storage } from "./store/index.js";
import { Concept, Usage } from "./types.js";
import { generateSlug, makeRepoId } from "./store/utils.js";
import { get_context } from "../repo/agent.js";
import { normalizeUsage } from "../aieo/src/usage.js";
import { DOC_GUIDELINES } from "./llm.js";
import * as fs from "fs";
import * as path from "path";

const BOOTSTRAP_LOOKBACK_DAYS = 10;

/**
 * Bootstrap result returned to the caller
 */
export interface BootstrapResult {
  concepts: Concept[];
  usage: Usage;
}

/**
 * Repo size category for calibrating concept count
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
 * Classify repo size and determine target concept count range
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
  return `Explore the repository "${owner}/${repo}" and identify its core concepts.

This is a ${sizing.size} repository (~${fileCount} source files). Aim for ${sizing.min}-${sizing.max} concepts.

A "concept" is a distinct user-facing capability or business function — something a product owner or end user would recognize. Examples: "Authentication System", "Payment Processing", "Real-time Notifications", "Search and Filtering".

Do NOT create concepts for:
- Build tooling, CI/CD, or infrastructure
- Individual utility functions or helpers
- Code style, linting, or formatting
- Generic "bug fixes" or "refactoring"
- Testing infrastructure (unless it IS the product)

For each concept provide:
- **name**: A clear, non-technical name (no framework/library names)
- **description**: 1-2 sentences explaining what this capability does for users
- **summary**: SUCCINCT high-level documentation (30-80 lines markdown) for this concept's CURRENT state
- **files**: List the core source files that implement this concept (relative paths from repo root, e.g. "src/auth/login.ts"). Include only the most important files — entry points, main modules, route definitions, core logic. Aim for 3-15 files per concept depending on scope.

**Summary requirements** — focus on what developers need to know to work on this concept:
${DOC_GUIDELINES.include}

${DOC_GUIDELINES.avoid}

Prefer fewer, broader concepts over many granular ones. Two closely related capabilities should be one concept, not two.`;
}

/**
 * JSON schema for the bootstrap agent's structured output
 */
const BOOTSTRAP_SCHEMA = {
  type: "object",
  properties: {
    concepts: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "Human-readable concept name",
          },
          description: {
            type: "string",
            description: "1-2 sentence description of the capability",
          },

          summary: {
            type: "string",
            description:
              "10-20 line high-level explanation of how the concept works",
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
  required: ["concepts"],
};

/**
 * Bootstrap a brand-new repo by exploring the codebase and creating
 * an initial set of concepts. Uses get_context (agentic exploration)
 * with a structured schema to produce concepts in a single pass.
 *
 * Returns the created concepts and token usage.
 */
export async function bootstrapConcepts(
  owner: string,
  repo: string,
  repoPath: string,
  storage: Storage,
  sessionId?: string
): Promise<BootstrapResult> {
  const repoId = `${owner}/${repo}`;

  console.log(`\n🚀 Bootstrap mode: exploring ${repoId} to seed initial concepts...`);

  // 1. Count source files for sizing hint
  const fileCount = countSourceFiles(repoPath);
  const sizing = classifyRepo(fileCount);
  console.log(
    `   📊 Found ${fileCount} source files (${sizing.size} repo, targeting ${sizing.min}-${sizing.max} concepts)`
  );

  // 2. Build prompt and call get_context with structured schema
  const prompt = buildBootstrapPrompt(owner, repo, fileCount, sizing);

  const result = await get_context(prompt, repoPath, {
    schema: BOOTSTRAP_SCHEMA,
    systemOverride: `You are a software architect analyzing a codebase to identify its core concepts. Use the provided tools to explore the repository structure, read key files (README, entry points, route definitions, main modules), and identify the distinct user-facing capabilities this software provides. Be thorough but focused — read enough to understand what each concept does, but don't try to read every file.`,
    sessionId,
    isolatedContext: true,
  });

  const decision = result.content as {
    concepts: Array<{
      name: string;
      description: string;
      summary: string;
      files: string[];
    }>;
  };

  // 3. Create and save concepts
  const now = new Date();
  const concepts: Concept[] = [];

  for (const f of decision.concepts || []) {
    const slug = generateSlug(f.name);
    const conceptId = makeRepoId(repoId, slug);

    const concept: Concept = {
      id: conceptId,
      repo: repoId,
      name: f.name,
      description: f.description,
      prNumbers: [],
      commitShas: [],
      createdAt: now,
      lastUpdated: now,
      documentation: f.summary,
    };

    await storage.saveConcept(concept);
    await storage.saveDocumentation(conceptId, f.summary);
    concepts.push(concept);

    // Link core files identified by the LLM
    const files = f.files || [];
    if (files.length > 0) {
      const linked = await storage.linkConceptToFilesByPaths(conceptId, files);
      console.log(`   ✅ Created concept: ${f.name} (${conceptId}) — linked ${linked} files`);
    } else {
      console.log(`   ✅ Created concept: ${f.name} (${conceptId})`);
    }
  }

  // 4. Set checkpoint to N days ago so processRepo replays recent changes
  //    This captures recent activity (e.g. someone forked a repo and added a few PRs)
  const lookback = new Date(now.getTime() - BOOTSTRAP_LOOKBACK_DAYS * 24 * 60 * 60 * 1000);
  await storage.setChronologicalCheckpoint(repoId, {
    lastProcessedTimestamp: lookback.toISOString(),
    processedAtTimestamp: [],
  });
  console.log(`   📌 Checkpoint set to ${BOOTSTRAP_LOOKBACK_DAYS} days ago — processRepo will pick up recent changes`);

  console.log(
    `\n🎯 Bootstrap complete: created ${concepts.length} concepts for ${repoId}`
  );

  return {
    concepts,
    usage: result.usage,
  };
}

/**
 * Explore a newly created concept to generate initial documentation.
 * Called when the incremental flow discovers a concept not in the bootstrap set.
 * Only runs if the concept has no existing documentation.
 */
export async function exploreNewConcept(
  concept: Concept,
  repoPath: string,
  storage: Storage,
  sessionId?: string
): Promise<Usage> {
  if (concept.documentation && concept.documentation.trim().length > 0) {
    return normalizeUsage();
  }

  console.log(`   🔍 Exploring codebase for new concept: ${concept.name}...`);

  const result = await get_context(
    `Generate SUCCINCT documentation for the "${concept.name}" concept in this codebase.

Description: ${concept.description}

${DOC_GUIDELINES.include}

${DOC_GUIDELINES.avoid}

Target length: 30-80 lines of markdown.`,
    repoPath,
    {
      systemOverride: `You are a software architect generating concise concept documentation. Use the provided tools to explore the repository and find the key files, components, and patterns related to this concept. Be thorough but focused.`,
      sessionId,
      isolatedContext: true,
    }
  );

  const documentation = typeof result.content === "string"
    ? result.content
    : result.final;

  concept.documentation = documentation;
  await storage.saveConcept(concept);
  await storage.saveDocumentation(concept.id, documentation);

  console.log(`   ✅ Documentation generated for: ${concept.name}`);

  return result.usage;
}
