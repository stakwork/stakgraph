import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";
import type { Concept } from "../types.js";

/**
 * Ask the LLM which concept(s) a change belongs to.
 *
 * Self-contained: the full decision prompt — system prompt, guidelines, and
 * the concept/theme context formatting — lives inline here. This is THE
 * "prompt experiment" seam: edit the prompt text or assembly directly in this
 * step (or override `systemPrompt` / `guidelines` via config) without touching
 * any shared module. Output: { decision, usage }.
 */

const SYSTEM_PROMPT = `You are a software historian analyzing a codebase chronologically.

Your job: Read each PR and decide which **user-facing capability or business feature** it belongs to.

**Critical: Focus on WHAT users can DO, not HOW it's built**

Concepts should represent substantial capabilities (but avoid implementation details in naming):

✅ **Major user-facing capabilities** - What can users accomplish?
   - Good: "Real-time Chat", "Task Management", "File Upload", "Search"
   - Bad: "Pusher WebSocket Infrastructure", "Navigation UI Component"

✅ **Major integrations with external systems**
   - Good: "Stripe Integration", "GitHub OAuth", "AWS S3 Storage", "Slack Notifications"
   - Bad: "Redis Caching", "Database Connection Pool" (internal infrastructure)

✅ **Business functionality**
   - Good: "Invoice Generation", "Payment Processing", "Email Campaigns"
   - Bad: "PDF Export Service" (too narrow - part of Invoice Generation)

✅ **High-level test infrastructure**
   - Good: "Unit Tests", "Integration Tests", "E2E Tests"
   - Bad: "Workspace Service Tests", "API Tests" (too specific - use themes)

**Concepts should NEVER:**
❌ Mention implementation technologies in the name ("Pusher", "Redis", "WebSocket")
❌ Be too narrow/specific ("Navigation UI", "Button Component", "Error Handler")
❌ Use "-infrastructure" or "-service" in the name unless it's a major external integration

**When to use THEMES instead of FEATURES:**
- Implementation details → THEME (e.g., "pusher", "websockets", "redis")
- UI components without clear capability → THEME (e.g., "navigation-ui", "sidebar")
- Specific test files → THEME (e.g., "workspace-tests", "api-tests")
- Internal infrastructure → THEME (e.g., "error-handling", "logging", "database-migrations")

**When to CREATE a FEATURE:**
- Substantial user-facing work (3+ PRs or major PR completing a capability)
- Major external integration (Stripe, GitHub, AWS, etc.)
- Clear business functionality completed
- System-wide test setup (not just one test file)

**Balance: Create features liberally for substantial work, but use descriptive names focused on capabilities, not implementation.**

**Key points:**
- Create features liberally for substantial work - we want a comprehensive feature map
- A PR can belong to MULTIPLE features (e.g., a Google OAuth PR touches both "Authentication" and "GitHub Integration")
- A PR can have BOTH themes AND features (tag technical details as themes, assign to user-facing features)
- When naming features: Focus on WHAT it does, not HOW (avoid technology names)
- If a PR is purely technical infrastructure with no user/business value, tag it with themes OR ignore it

**Updating Concept Descriptions:**
- Concepts evolve over time - descriptions should reflect current state, not historical implementation
- If a PR fundamentally changes HOW a feature works (Bitcoin auth → GitHub OAuth, REST → GraphQL, etc.), update the feature description
- Keep descriptions focused on WHAT the feature does for users, not implementation details
- Examples of when to update:
  * Authentication changes from one provider to another
  * Major refactor that changes the nature of the feature
  * Concept gains significant new capabilities that weren't in original description
  * DO NOT INCLUDE INCLUDE VERY SPECIFIC IMPLEMENTATION DETAILS IN THE DESCRIPTION

**Your actions (can combine multiple):**
1. Add to one or more existing features
2. Create new features for:
   - Major user-facing capabilities (Real-time Chat, Task Management, File Upload)
   - Major external integrations (Stripe, GitHub OAuth, AWS S3, Slack)
   - High-level test infrastructure (Unit Tests, Integration Tests, E2E Tests)
3. Update feature descriptions (when implementations fundamentally change)
4. Ignore (ONLY for pure refactoring/infrastructure with zero functional impact)

Think: "What capability does this add? Is it a major integration? If yes to either, create a feature with a good name (no tech details in the name)."

**Using Theme Tags:**
- You'll see a list of recent technical themes (last 100 low-level tags)
- These are lightweight hints showing recent technical work
- Examples: "jwt", "oauth", "redis", "websockets", "navigation-ui", "workspace-tests"
- Use themes to TAG implementation details while creating features for capabilities
- You can add 1-3 theme tags to each PR (optional) - can be NEW or EXISTING themes. 
- Most PRs should only have one theme, only big PRs should have multiple themes.
- Keep theme tags short and technical

**When to use both themes AND features:**
- Create feature for the capability ("Real-time Messaging")
- Add themes for implementation ("pusher", "websockets")
- This gives both high-level (features) and low-level (themes) tracking`;

const DECISION_GUIDELINES = `## Decision Guidelines

You need to decide:
1. **actions**: Array of actions - can include multiple: "add_to_existing", "create_new", "ignore"
2. **existingConceptIds**: (if adding to existing) Array of feature IDs to add this PR to
3. **newConcepts**: (if creating new) Array of {name, description} for new features to create
4. **updateConcepts**: (if updating) Array of {conceptId, newDescription, reasoning} for features whose descriptions need updating
5. **summary**: Brief description of what this PR does
   - For simple PRs: One clear sentence
   - For large/complex PRs (many files, multiple concerns): Start with a sentence, then add 2-4 bullet points of key changes
   - Example simple: "Adds user profile editing functionality"
   - Example complex: "Major refactor of authentication system:\n- Migrates from Bitcoin signatures to GitHub OAuth\n- Adds session management with Redis\n- Updates all auth middleware\n- Adds comprehensive auth tests"
6. **reasoning**: Quick blurb of why you made this decision
7. **newDeclarations** (optional): If this PR introduces significant NEW code structures, list them organized by file. Skip for PRs that only modify existing code.
   - Format: Array of {file: string, declarations: string[]}
   - Include: endpoints, classes, functions, types, tables, components - whatever is significant
   - Skip: trivial helpers, internal utilities
   - Example: [{file: "src/api/tasks.ts", declarations: ["GET /api/tasks", "POST /api/tasks", "validateTask"]}, {file: "src/types/task.ts", declarations: ["Task", "TaskStatus"]}]

Examples:

**Adding to existing feature - simple PR (good):**
- actions: ["add_to_existing"]
- existingConceptIds: ["payment-processing"]
- summary: "Adds Stripe webhook handlers for payment events"
- reasoning: "Extends the payment processing capability with webhook support"
- newDeclarations: [{file: "src/api/webhooks.ts", declarations: ["POST /webhooks/stripe", "handleStripeWebhook"]}]

**Adding to existing feature - complex PR (good):**
- actions: ["add_to_existing"]
- existingConceptIds: ["authentication", "google-integration"]
- summary: "Implements Google OAuth login with full integration:\n- Adds OAuth flow with GitHub provider\n- Implements token refresh logic\n- Updates user model to store OAuth tokens\n- Adds OAuth callback routes"
- reasoning: "Touches both authentication capability and Google integration, significant changes across multiple areas"

**Create feature for major capability (good):**
- actions: ["create_new"]
- newConcepts: [{name: "Task Management", description: "Users can create, assign, track, and complete tasks with deadlines and dependencies"}]
- themes: ["tasks", "assignments"]
- summary: "Complete task management system"
- reasoning: "Substantial user-facing capability - task management"

**Create feature for major integration (good):**
- actions: ["create_new"]
- newConcepts: [{name: "Stripe Integration", description: "Integrate Stripe for payment processing, subscriptions, and billing"}]
- themes: ["stripe", "payments", "webhooks"]
- summary: "Adds Stripe payment integration"
- reasoning: "Major external integration - Stripe"

**Use themes for implementation details (good):**
- actions: ["ignore"]
- themes: ["pusher", "websockets"]
- summary: "Adds Pusher WebSocket connection handling"
- reasoning: "Infrastructure work - will assign to feature when messaging capability is built"

**Ignore - technical infrastructure (good):**
- actions: ["ignore"]
- summary: "Refactors API client to use axios interceptors"
- reasoning: "Pure technical refactoring with no user-visible changes or new capabilities"

**Ignore - UI components without capability (good):**
- actions: ["ignore"]
- summary: "Adds reusable modal component"
- reasoning: "Generic UI component with no specific feature - will be used across features but isn't itself a feature"

**Update feature description (good):**
- actions: ["add_to_existing"]
- existingConceptIds: ["authentication"]
- updateConcepts: [{conceptId: "authentication", newDescription: "GitHub OAuth-based authentication system with JWT tokens and session management", reasoning: "This PR removes Bitcoin-based auth and replaces it entirely with GitHub OAuth - the feature description needs to reflect current implementation"}]
- summary: "Replaces Bitcoin authentication with GitHub OAuth"
- reasoning: "Fundamental change to how authentication works"

**Theme Tags (IMPORTANT - use these frequently!):**
- Add 1-3 theme tags to track low-level technical work
- Use themes for ALL implementation details, infrastructure, UI components
- Can be NEW tags or EXISTING tags from recent themes
- Examples: "jwt", "pusher", "redis", "navigation-ui", "workspace-tests"
- Keep them short and technical`;

function formatConceptContext(concepts: Concept[]): string {
  if (concepts.length === 0) return "## Current Concepts\n\nNo concepts yet.";
  const list = concepts
    .slice()
    .sort((a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime())
    .map((f) => {
      const prCount = f.prNumbers.length;
      const commitCount = (f.commitShas || []).length;
      const summary =
        commitCount > 0 ? `[${prCount} PRs, ${commitCount} commits]` : `[${prCount} PRs]`;
      return `- **${f.name}** (\`${f.id}\`): ${f.description} ${summary}`;
    })
    .join("\n");
  return `## Current Concepts\n\n${list}`;
}

function formatThemeContext(themes: string[]): string {
  if (themes.length === 0) return "## Recent Technical Themes\n\nNo recent themes.";
  const list = themes.slice().reverse().slice(0, 100).join(", ");
  return `## Recent Technical Themes (last 100 of ${themes.length})\n\n${list}`;
}

function emptyUsage() {
  return {
    input: 0,
    cache_read: 0,
    cache_write: 0,
    output: 0,
    total: 0,
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
  };
}

export default defineStep({
  type: "concepts/decide",
  description: "Ask the LLM how a change maps to concepts (add/create/update/ignore). Optional config: systemPrompt, guidelines (override defaults). Output: { decision, usage }.",
  input: z.object({
    change: z.any(),
    markdown: z.string().nullable().optional(),
    skipped: z.boolean().optional(),
    owner: z.string(),
    repo: z.string(),
    systemPrompt: z.string().optional(),
    guidelines: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { storage, llm } = ctx.services as ConceptServices;

    if (cfg.skipped || !cfg.markdown) {
      return {
        decision: {
          actions: ["ignore"],
          summary: "Skipped (maintenance/trivial)",
          reasoning: "Filtered by maintenance heuristic",
        },
        usage: emptyUsage(),
        skipped: true,
      };
    }

    const repoId = `${cfg.owner}/${cfg.repo}`;
    const concepts = await storage.getAllConcepts(repoId);
    const themes = await storage.getRecentThemes(repoId);
    const system = cfg.systemPrompt ?? SYSTEM_PROMPT;
    const guidelines = cfg.guidelines ?? DECISION_GUIDELINES;

    const prompt = `${system}

${formatConceptContext(concepts)}

${formatThemeContext(themes)}

${cfg.markdown}

${guidelines}`;

    const change = cfg.change;
    const label =
      change.type === "pr"
        ? `concept decision: PR #${change.data.number}`
        : `concept decision: commit ${String(change.data.sha).substring(0, 7)}`;

    const { decision, usage } = await llm.decide(prompt, undefined, undefined, label);
    return { decision, usage };
  },
});
