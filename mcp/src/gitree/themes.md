# Themes - Emerging Feature Detection

## Problem

Currently, features are decided while reviewing a **single PR**. This causes:

1. **Premature feature creation**: LLM creates features too early (1-2 PRs)
2. **Missed emerging patterns**: Small related PRs get ignored until a "big enough" PR arrives
3. **No gradual recognition**: Can't accumulate weak signals across multiple PRs

For large repos with many contributors working on disparate features simultaneously, we need a way to **track emerging work** before promoting it to a full feature.

## Solution: Themes

Add a new entity called **Theme** - low-level building blocks that track implementation details and technical concepts before they coalesce into high-level features.

### Semantic Distinction: Themes vs Features

**Themes** = Low-level technical building blocks

- Can be implementation details ("jwt-tokens", "redis-caching", "stripe-webhooks")
- Can be protocols/technologies ("oauth", "websockets", "graphql")
- Can be architectural patterns ("event-sourcing", "pub-sub")
- Often too granular to be called a "feature"
- Multiple themes often combine to form one feature

**Features** = High-level user capabilities

- User-facing functionality ("Authentication System", "Payment Processing")
- Business capabilities ("Invoice Generation", "Real-time Notifications")
- Major integrations ("Stripe Integration", "Google OAuth")
- What users can DO, not HOW it's built

**Example**: Three themes ("oauth", "jwt", "sessions") → One feature ("Authentication System")

### Key Principles

1. **No hardcoded thresholds**: LLM decides when to promote based on context
2. **Themes are building blocks**: Track low-level work that isn't substantial enough for a feature yet
3. **LLM has full control**: Sees Features + Themes, decides everything
4. **Continuous processing**: Works for active repos with many simultaneous PRs
5. **Multiple themes coalesce**: Several related themes typically merge into one feature

## Data Structures

### Theme (New)

```typescript
interface Theme {
  id: string; // Slug from name (e.g., "jwt", "oauth", "redis-caching")
  name: string; // Human-readable (e.g., "JWT Tokens", "OAuth Flow", "Redis Caching")
  description: string; // Low-level technical details about this building block
  prNumbers: number[]; // PRs contributing to this theme
  createdAt: Date;
  lastUpdated: Date;
}
```

**Note**: Themes have the exact same structure as Features - the difference is purely semantic:

- **Themes**: Track low-level technical work (jwt, webhooks, caching strategies)
- **Features**: Track high-level capabilities (Authentication System, Payment Processing)

### Updated LLM Decision

```typescript
interface LLMDecision {
  actions: (
    | "add_to_existing"
    | "add_to_theme"
    | "create_new"
    | "create_theme"
    | "promote_theme"
    | "ignore"
  )[];

  // Add to existing features
  existingFeatureIds?: string[];

  // Add to or create themes (NEW)
  themeIds?: string[];
  newThemes?: Array<{
    name: string;
    description: string;
  }>;

  // Promote themes to features (NEW)
  promotions?: Array<{
    themeIds: string[]; // Can merge multiple themes into one feature!
    featureName: string;
    featureDescription: string;
    reasoning: string;
  }>;

  // Update features (existing)
  updateFeatures?: Array<{
    featureId: string;
    newDescription: string;
    reasoning: string;
  }>;

  summary: string;
  reasoning: string;
  newDeclarations?: Array<{
    file: string;
    declarations: string[];
  }>;
}
```

## LLM Context Changes

### Current Context (per PR)

```
System Prompt
Current Features: [list]
PR to Analyze: [PR content]
Decision Guidelines
```

### New Context (per PR)

```
System Prompt
Current Features: [list]
Emerging Themes: [list]  // NEW
PR to Analyze: [PR content]
Decision Guidelines (updated)
```

## Implementation Checklist

### 1. Update Type Definitions

- [ ] Add `Theme` interface to `types.ts` (same structure as `Feature`)
- [ ] Update `LLMDecision` interface with new actions and fields
- [ ] Update Zod schema in `llm.ts` to match new decision structure

### 2. Update Storage

- [ ] Add theme methods to abstract `Storage` class:

  ```typescript
  abstract saveTheme(theme: Theme): Promise<void>;
  abstract getTheme(id: string): Promise<Theme | null>;
  abstract getAllThemes(): Promise<Theme[]>;
  abstract deleteTheme(id: string): Promise<void>;
  ```

- [ ] Implement in `FileSystemStore`:

  - Store in `themes/` subdirectory (parallel to `features/`)
  - Same JSON format as features

- [ ] Implement in `GraphStorage`:
  - Add `:Theme` label (separate from `:Feature`)
  - Add indexes for theme lookups
  - Create `CONTRIBUTES_TO` relationships from PRs to Themes

### 3. Update Builder

- [ ] Modify `buildDecisionPrompt()` to include themes context:

  ```typescript
  private formatThemeContext(themes: Theme[]): string {
    if (themes.length === 0) {
      return "## Emerging Themes\n\nNo themes yet.";
    }

    const themeList = themes
      .sort((a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime())
      .map(t =>
        `- **${t.name}** (\`${t.id}\`): ${t.description} [${t.prNumbers.length} PRs]`
      )
      .join("\n");

    return `## Emerging Themes\n\n${themeList}`;
  }
  ```

- [ ] Update `applyDecision()` to handle new actions:

  ```typescript
  if (action === "add_to_theme") {
    // Add to existing theme(s)
  }

  if (action === "create_theme") {
    // Create new theme(s)
  }

  if (action === "promote_theme") {
    // Handle promotions (can merge multiple themes)
  }
  ```

- [ ] Implement `applyPromotion()` method:

  ```typescript
  private async applyPromotion(promotion: Promotion): Promise<void> {
    // Gather all PRs from all themes
    const allPRs = [];
    for (const themeId of promotion.themeIds) {
      const theme = await this.storage.getTheme(themeId);
      if (theme) {
        allPRs.push(...theme.prNumbers);
      }
    }

    // Create feature with merged PRs
    const feature: Feature = {
      id: this.generateFeatureId(promotion.featureName),
      name: promotion.featureName,
      description: promotion.featureDescription,
      prNumbers: [...new Set(allPRs)],  // dedupe
      createdAt: new Date(),
      lastUpdated: new Date(),
    };

    await this.storage.saveFeature(feature);

    // Delete all merged themes
    for (const themeId of promotion.themeIds) {
      await this.storage.deleteTheme(themeId);
      console.log(`   🗑️  Deleted theme: ${themeId}`);
    }

    console.log(`   ✨ Merged ${promotion.themeIds.length} theme(s) into: ${feature.name}`);
  }
  ```

### 4. Update LLM Prompts

- [ ] Update `SYSTEM_PROMPT` in `llm.ts` to explain Themes vs Features:

  ```typescript
  **Themes vs Features - Critical Distinction:**

  THEMES = Low-level technical building blocks
  - Implementation details: "jwt-tokens", "redis-caching", "stripe-webhooks"
  - Protocols/technologies: "oauth", "websockets", "graphql"
  - Architectural patterns: "event-sourcing", "pub-sub"
  - Too granular to be a feature on their own
  - Multiple themes typically combine to form one feature

  FEATURES = High-level user capabilities
  - User-facing functionality: "Authentication System", "Payment Processing"
  - Business capabilities: "Invoice Generation", "Real-time Notifications"
  - What users can DO, not HOW it's built
  - Result of multiple themes coalescing

  **Your decision process:**
  1. Is this PR significant enough for a FEATURE? (Adds major user capability)
     → Yes: add to existing feature or create new feature
     → No: Continue to step 2

  2. Is this PR doing low-level technical work? (Implementation detail, protocol, etc.)
     → Yes: add to existing theme or create new theme
     → No: Ignore (trivial/maintenance)

  3. Do we have enough related themes to form a feature? (Your judgment!)
     → Yes: PROMOTE multiple themes into one feature
     → Example: themes "oauth" + "jwt" + "sessions" → Feature "Authentication System"

  **When promoting:**
  - Merge MULTIPLE related themes into ONE high-level feature
  - Write a comprehensive feature description focused on WHAT it does (not HOW)
  - All PRs from all themes are included in the new feature
  - Promoted themes are automatically deleted
  ```

- [ ] Update `DECISION_GUIDELINES` with theme examples showing low-level vs high-level

### 5. Update CLI

- [ ] Add `list-themes` command:

  ```bash
  yarn gitree list-themes
  ```

- [ ] Add `show-theme <themeId>` command:

  ```bash
  yarn gitree show-theme authentication-work
  ```

- [ ] Update `show-pr` to display themes (in addition to features)

- [ ] Update `stats` to include theme count

### 6. Update API Routes

- [ ] Add `GET /gitree/themes` - list all themes
- [ ] Add `GET /gitree/themes/:id` - get specific theme with PRs
- [ ] Update `GET /gitree/prs/:number` to include themes

### 7. Update Store Utils

- [ ] Update `formatPRMarkdown()` to include themes:

  ```markdown
  _Part of features: `auth-system`, `google-integration`_
  _Part of themes: `authentication-work`_
  ```

- [ ] Update `parsePRMarkdown()` to parse themes section

## Example Workflows

### Scenario 1: Themes as Building Blocks → Feature

```
PR #101: "Add JWT token generation"
  → Decision: create_theme "jwt"
  → Theme "jwt" created (1 PR)
  → Reasoning: Low-level implementation detail, not a feature yet

PR #104: "Add OAuth callback handler"
  → Decision: create_theme "oauth"
  → Theme "oauth" created (1 PR)
  → Reasoning: Protocol implementation, building block

PR #107: "Add session middleware with Redis"
  → Decision: create_theme "sessions"
  → Theme "sessions" created (1 PR)
  → Reasoning: Technical mechanism, not user-facing yet

PR #110: "Connect JWT validation to OAuth flow"
  → Decision: add_to_theme ["jwt", "oauth"]
  → Both themes updated
  → Reasoning: Connects existing building blocks

PR #115: "Complete auth system with user management"
  → Decision: add_to_theme ["jwt", "oauth", "sessions"] + promote_theme
  → promotions: [{
      themeIds: ["jwt", "oauth", "sessions"],
      featureName: "Authentication System",
      featureDescription: "Complete OAuth-based authentication with JWT tokens and session management"
    }]
  → Feature "Authentication System" created (5 PRs)
  → All 3 themes deleted
  → Reasoning: Now substantial enough to be a user-facing capability

PR #120 → Decision: add_to_existing "authentication-system"
  → Normal feature operation
```

### Scenario 2: Multiple Low-Level Themes Coalesce

```
Current state:
  - Theme "stripe-webhooks" (2 PRs) - webhook handling
  - Theme "payment-intent" (2 PRs) - payment creation flow
  - Theme "subscription-api" (3 PRs) - recurring payments

PR #200: "Complete payment flow with subscriptions"
  → Decision: promote_theme
  → promotions: [{
      themeIds: ["stripe-webhooks", "payment-intent", "subscription-api"],
      featureName: "Payment Processing",
      featureDescription: "Stripe-based payment processing with one-time and recurring payments"
    }]
  → All 3 low-level themes merged into high-level Feature "Payment Processing" (7 PRs)
  → All 3 themes deleted
```

### Scenario 3: Mix of Themes and Features

```
Current state:
  - Feature "Authentication System" (12 PRs) - established high-level feature
  - Theme "websockets" (2 PRs) - low-level protocol implementation
  - Theme "push-notifications" (2 PRs) - low-level notification mechanism
  - Theme "presence-detection" (1 PR) - low-level technical detail

PR #300: "Real-time notification system with WebSocket support"
  → Decision: add_to_theme ["websockets", "push-notifications", "presence-detection"] + promote_theme
  → promotions: [{
      themeIds: ["websockets", "push-notifications", "presence-detection"],
      featureName: "Real-time Notifications",
      featureDescription: "WebSocket-based real-time notification system with presence detection"
    }]
  → 3 low-level themes promoted to Feature "Real-time Notifications" (5 PRs)
  → All 3 themes deleted
  → Authentication feature remains unchanged
  → Reasoning: The themes were building blocks that now form a complete user-facing capability
```

## Future Enhancements (Optional)

1. **Theme expiry**: Auto-delete themes with no activity for 90+ days
2. **Theme merging**: LLM can merge two themes without promoting to feature
3. **Theme splitting**: LLM can split one theme into two if scope diverges
4. **Confidence scores**: Track theme "strength" metric (number of PRs, recency, etc.)
5. **Cross-feature themes**: Allow themes to exist that support multiple features (e.g., "caching" theme used by multiple features)

## Success Metrics

After implementation, we should see:

- **Cleaner feature list**: Only high-level, user-facing capabilities become features
- **Better feature quality**: Features represent substantial capabilities, not implementation details
- **Low-level work tracked**: Technical building blocks captured as themes (jwt, redis, websockets, etc.)
- **Natural coalescence**: Multiple related themes merge into coherent features
- **No premature features**: Features only created when themes reach critical mass
- **Clear abstraction levels**: Themes = HOW (technical), Features = WHAT (capability)
