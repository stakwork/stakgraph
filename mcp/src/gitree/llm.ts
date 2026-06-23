import { z } from "zod";
import { callGenerateObject } from "../aieo/src/stream.js";
import { normalizeUsage } from "../aieo/src/usage.js";
import { Provider } from "../aieo/src/provider.js";
import { LLMDecision, Usage } from "./types.js";
import {
  appendMessages,
  appendStepMeta,
  loadSessionMessages,
  loadStepMeta,
} from "../repo/session.js";

/**
 * Shared documentation guidelines used by bootstrap, summarizer, and exploreNewConcept
 */
export const DOC_GUIDELINES = {
  include: `**What to include**:
- Brief overview (just a few sentences, including any major integrations or dependencies)
- List the 5-15 core files (just paths and 1-line purposes)
- Key concepts/components (high-level only)
- Main API endpoints/functions (names only, no implementations)
- Core data models (names only, brief purpose)
- Essential patterns and gotchas (briefly, if any)`,

  avoid: `**What to AVOID**:
- Code snippets or implementation details
- Long explanations of how things work internally
- Historical information about how it evolved
- Detailed API documentation
- Step-by-step flows unless absolutely essential`,
};

export interface GitreeSessionTracker {
  sessionId: string;
  step: number;
  turn: number;
  cumulativeInput: number;
  cumulativeOutput: number;
}

export function createGitreeSessionTracker(
  sessionId: string,
): GitreeSessionTracker {
  const stepMeta = loadStepMeta(sessionId);
  const lastMeta = stepMeta[stepMeta.length - 1];
  return {
    sessionId,
    step: stepMeta.length,
    turn:
      loadSessionMessages(sessionId).filter((m) => m.role === "user").length +
      1,
    cumulativeInput: lastMeta?.cumulativeInput ?? 0,
    cumulativeOutput: lastMeta?.cumulativeOutput ?? 0,
  };
}

export function appendGitreeLlmExchange(
  tracker: GitreeSessionTracker,
  prompt: string,
  response: string,
  usage: Usage,
  label?: string,
): void {
  const normalizedUsage = normalizeUsage(usage);
  appendMessages(tracker.sessionId, [
    { role: "user", content: prompt },
    { role: "assistant", content: response },
  ]);
  tracker.cumulativeInput += normalizedUsage.inputTokens || 0;
  tracker.cumulativeOutput += normalizedUsage.outputTokens || 0;
  appendStepMeta(tracker.sessionId, [
    {
      step: tracker.step,
      turn: tracker.turn,
      ...(label ? { label } : {}),
      usage: normalizedUsage,
      cumulativeInput: tracker.cumulativeInput,
      cumulativeOutput: tracker.cumulativeOutput,
      toolCalls: [],
      timestamp: new Date().toISOString(),
    },
  ]);
  tracker.step += 1;
  tracker.turn += 1;
}

/**
 * Schema for LLM decision using Zod
 */
const LLMDecisionSchema = z.object({
  actions: z.array(z.enum(["add_to_existing", "create_new", "ignore"])),
  existingConceptIds: z.array(z.string()).optional(),
  newConcepts: z
    .array(
      z.object({
        name: z.string(),
        description: z.string(),
      })
    )
    .optional(),
  updateConcepts: z
    .array(
      z.object({
        conceptId: z.string(),
        newDescription: z.string(),
        reasoning: z.string(),
      })
    )
    .optional(),
  themes: z.array(z.string()).optional(),
  summary: z.string(),
  reasoning: z.string(),
  newDeclarations: z
    .array(
      z.object({
        file: z.string(),
        declarations: z.array(z.string()),
      })
    )
    .optional(),
});

/**
 * LLM client for making decisions about PRs
 */
export class LLMClient {
  constructor(
    private provider: Provider,
    private apiKey: string,
    private sessionTracker?: GitreeSessionTracker,
  ) {}

  /**
   * Ask LLM to decide what to do with a PR
   */
  async decide(
    prompt: string,
    retries = 3,
    sessionId?: string,
    label?: string,
  ): Promise<{ decision: LLMDecision; usage: Usage }> {
    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const result = await callGenerateObject({
          provider: this.provider,
          apiKey: this.apiKey,
          prompt,
          schema: LLMDecisionSchema,
        });

        const response = JSON.stringify(result.object);
        if (this.sessionTracker) {
          appendGitreeLlmExchange(
            this.sessionTracker,
            prompt,
            response,
            result.usage,
            label,
          );
        } else if (sessionId) {
          appendMessages(sessionId, [
            { role: "user", content: prompt },
            { role: "assistant", content: response },
          ]);
        }

        return {
          decision: result.object as LLMDecision,
          usage: result.usage,
        };
      } catch (error) {
        lastError = error as Error;

        if (attempt < retries) {
          const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
          console.log(
            `   ⚠️  Attempt ${attempt} failed, retrying in ${delay / 1000}s...`,
          );
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    throw new Error(`Failed after ${retries} attempts: ${lastError?.message}`);
  }
}

/**
 * System prompt for the LLM
 */
export const SYSTEM_PROMPT = `You are a software historian analyzing a codebase chronologically.

Your job: Read each PR and decide which **user-facing capability or business concept** it belongs to.

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

**When to use THEMES instead of CONCEPTS:**
- Implementation details → THEME (e.g., "pusher", "websockets", "redis")
- UI components without clear capability → THEME (e.g., "navigation-ui", "sidebar")
- Specific test files → THEME (e.g., "workspace-tests", "api-tests")
- Internal infrastructure → THEME (e.g., "error-handling", "logging", "database-migrations")

**When to CREATE a CONCEPT:**
- Substantial user-facing work (3+ PRs or major PR completing a capability)
- Major external integration (Stripe, GitHub, AWS, etc.)
- Clear business functionality completed
- System-wide test setup (not just one test file)

**Balance: Create concepts liberally for substantial work, but use descriptive names focused on capabilities, not implementation.**

**Key points:**
- Create concepts liberally for substantial work - we want a comprehensive concept map
- A PR can belong to MULTIPLE concepts (e.g., a Google OAuth PR touches both "Authentication" and "GitHub Integration")
- A PR can have BOTH themes AND concepts (tag technical details as themes, assign to user-facing concepts)
- When naming concepts: Focus on WHAT it does, not HOW (avoid technology names)
- If a PR is purely technical infrastructure with no user/business value, tag it with themes OR ignore it

**Updating Concept Descriptions:**
- Concepts evolve over time - descriptions should reflect current state, not historical implementation
- If a PR fundamentally changes HOW a concept works (Bitcoin auth → GitHub OAuth, REST → GraphQL, etc.), update the concept description
- Keep descriptions focused on WHAT the concept does for users, not implementation details
- Examples of when to update:
  * Authentication changes from one provider to another
  * Major refactor that changes the nature of the concept
  * Concept gains significant new capabilities that weren't in original description
  * DO NOT INCLUDE INCLUDE VERY SPECIFIC IMPLEMENTATION DETAILS IN THE DESCRIPTION

**Your actions (can combine multiple):**
1. Add to one or more existing concepts
2. Create new concepts for:
   - Major user-facing capabilities (Real-time Chat, Task Management, File Upload)
   - Major external integrations (Stripe, GitHub OAuth, AWS S3, Slack)
   - High-level test infrastructure (Unit Tests, Integration Tests, E2E Tests)
3. Update concept descriptions (when implementations fundamentally change)
4. Ignore (ONLY for pure refactoring/infrastructure with zero functional impact)

Think: "What capability does this add? Is it a major integration? If yes to either, create a concept with a good name (no tech details in the name)."

**Using Theme Tags:**
- You'll see a list of recent technical themes (last 100 low-level tags)
- These are lightweight hints showing recent technical work
- Examples: "jwt", "oauth", "redis", "websockets", "navigation-ui", "workspace-tests"
- Use themes to TAG implementation details while creating concepts for capabilities
- You can add 1-3 theme tags to each PR (optional) - can be NEW or EXISTING themes. 
- Most PRs should only have one theme, only big PRs should have multiple themes.
- Keep theme tags short and technical

**When to use both themes AND concepts:**
- Create concept for the capability ("Real-time Messaging")
- Add themes for implementation ("pusher", "websockets")
- This gives both high-level (concepts) and low-level (themes) tracking`;

/**
 * Decision guidelines for the LLM
 */
export const DECISION_GUIDELINES = `## Decision Guidelines

You need to decide:
1. **actions**: Array of actions - can include multiple: "add_to_existing", "create_new", "ignore"
2. **existingConceptIds**: (if adding to existing) Array of concept IDs to add this PR to
3. **newConcepts**: (if creating new) Array of {name, description} for new concepts to create
4. **updateConcepts**: (if updating) Array of {conceptId, newDescription, reasoning} for concepts whose descriptions need updating
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

**Adding to existing concept - simple PR (good):**
- actions: ["add_to_existing"]
- existingConceptIds: ["payment-processing"]
- summary: "Adds Stripe webhook handlers for payment events"
- reasoning: "Extends the payment processing capability with webhook support"
- newDeclarations: [{file: "src/api/webhooks.ts", declarations: ["POST /webhooks/stripe", "handleStripeWebhook"]}]

**Adding to existing concept - complex PR (good):**
- actions: ["add_to_existing"]
- existingConceptIds: ["authentication", "google-integration"]
- summary: "Implements Google OAuth login with full integration:\n- Adds OAuth flow with GitHub provider\n- Implements token refresh logic\n- Updates user model to store OAuth tokens\n- Adds OAuth callback routes"
- reasoning: "Touches both authentication capability and Google integration, significant changes across multiple areas"

**Create concept for major capability (good):**
- actions: ["create_new"]
- newConcepts: [{name: "Task Management", description: "Users can create, assign, track, and complete tasks with deadlines and dependencies"}]
- themes: ["tasks", "assignments"]
- summary: "Complete task management system"
- reasoning: "Substantial user-facing capability - task management"

**Create concept for major integration (good):**
- actions: ["create_new"]
- newConcepts: [{name: "Stripe Integration", description: "Integrate Stripe for payment processing, subscriptions, and billing"}]
- themes: ["stripe", "payments", "webhooks"]
- summary: "Adds Stripe payment integration"
- reasoning: "Major external integration - Stripe"

**Use themes for implementation details (good):**
- actions: ["ignore"]
- themes: ["pusher", "websockets"]
- summary: "Adds Pusher WebSocket connection handling"
- reasoning: "Infrastructure work - will assign to concept when messaging capability is built"

**Ignore - technical infrastructure (good):**
- actions: ["ignore"]
- summary: "Refactors API client to use axios interceptors"
- reasoning: "Pure technical refactoring with no user-visible changes or new capabilities"

**Ignore - UI components without capability (good):**
- actions: ["ignore"]
- summary: "Adds reusable modal component"
- reasoning: "Generic UI component with no specific concept - will be used across concepts but isn't itself a concept"

**Update concept description (good):**
- actions: ["add_to_existing"]
- existingConceptIds: ["authentication"]
- updateConcepts: [{conceptId: "authentication", newDescription: "GitHub OAuth-based authentication system with JWT tokens and session management", reasoning: "This PR removes Bitcoin-based auth and replaces it entirely with GitHub OAuth - the concept description needs to reflect current implementation"}]
- summary: "Replaces Bitcoin authentication with GitHub OAuth"
- reasoning: "Fundamental change to how authentication works"

**BAD - Don't create concepts for UI components:**
❌ actions: ["create_new"]
❌ newConcepts: [{name: "Sidebar Navigation", description: "..."}]
Instead: Add to an existing concept that uses it, or ignore if it's generic

**BAD - Don't create concepts for infrastructure:**
❌ actions: ["create_new"]
❌ newConcepts: [{name: "Error Handling System", description: "..."}]
Instead: Ignore or add to relevant business concept if it improves error handling there

**BAD - Don't create concepts for code organization:**
❌ actions: ["create_new"]
❌ newConcepts: [{name: "TypeScript Migration", description: "..."}]
Instead: Ignore - this is pure technical work

**Theme Tags (IMPORTANT - use these frequently!):**
- Add 1-3 theme tags to track low-level technical work
- Use themes for ALL implementation details, infrastructure, UI components
- Can be NEW tags or EXISTING tags from recent themes
- Examples: "jwt", "pusher", "redis", "navigation-ui", "workspace-tests"
- Keep them short and technical
- Check recent themes - if you see related themes accumulating, consider creating a concept

**Example: Low-level work → Theme only**
{
  "actions": ["ignore"],
  "themes": ["pusher", "websockets"],
  "summary": "Adds Pusher WebSocket connection handling",
  "reasoning": "Infrastructure work - tracking with themes"
}

**Example: Theme + Concept**
{
  "actions": ["add_to_existing"],
  "existingConceptIds": ["real-time-chat"],
  "themes": ["pusher", "message-delivery"],
  "summary": "Improves real-time message delivery using Pusher",
  "reasoning": "Extends real-time chat capability; tracking implementation with themes"
}

**Example: Recognizing pattern from accumulated themes**
{
  "actions": ["create_new"],
  "newConcepts": [{
    "name": "Real-time Messaging",
    "description": "Users can send and receive messages in real-time across devices"
  }],
  "themes": ["real-time-complete"],
  "summary": "Completes real-time messaging system",
  "reasoning": "Recent themes show: pusher, websockets, message-delivery, presence - clear user capability now exists"
}

**BAD - Concept name has implementation details:**
❌ {
  "actions": ["create_new"],
  "newConcepts": [{name: "Pusher WebSocket Infrastructure", description: "Pusher-based real-time infrastructure..."}]
}
✅ Instead: {
  "actions": ["create_new"],
  "newConcepts": [{name: "Real-time Messaging", description: "Users can send and receive messages instantly"}],
  "themes": ["pusher", "websockets"]
}

**BAD - Concept too narrow/specific:**
❌ {
  "actions": ["create_new"],
  "newConcepts": [{name: "Navigation UI", description: "Collapsible sidebar navigation..."}]
}
✅ Instead: Add to broader concept or use themes: themes ["navigation-ui", "sidebar"]

**BAD - Test concept too specific:**
❌ {
  "actions": ["create_new"],
  "newConcepts": [{name: "Workspace Service Tests", description: "Tests for workspace service..."}]
}
✅ Instead: {
  "actions": ["add_to_existing"],
  "existingConceptIds": ["unit-tests"],
  "themes": ["workspace-tests"]
}`;
