# Themes - Emerging Feature Detection

## Problem

Currently, features are decided while reviewing a **single PR**. This causes:

1. **Premature feature creation**: LLM creates features too early (1-2 PRs)
2. **Missed emerging patterns**: Small related PRs get ignored until a "big enough" PR arrives
3. **No gradual recognition**: Can't accumulate weak signals across multiple PRs

For large repos with many contributors working on disparate features simultaneously, we need a way to **track emerging work** before promoting it to a full feature.

## Solution: Themes

Add a new entity called **Theme** - exactly like a Feature, but represents **emerging work** that hasn't reached critical mass yet.

### Key Principles

1. **No hardcoded thresholds**: LLM decides when to promote based on context
2. **Minimal changes**: Themes are just Features in a different namespace
3. **LLM has full control**: Sees Features + Themes, decides everything
4. **Continuous processing**: Works for active repos with many simultaneous PRs

## Data Structures

### Theme (New)

```typescript
interface Theme {
  id: string; // Slug from name (e.g., "authentication-work")
  name: string; // Human-readable (e.g., "Authentication Work")
  description: string; // What this emerging work is about
  prNumbers: number[]; // PRs contributing to this theme
  createdAt: Date;
  lastUpdated: Date;
}
```

**Note**: Themes have the exact same structure as Features - just semantically different.

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
  **Themes vs Features:**
  - Use THEMES when work is emerging but not yet substantial
  - PROMOTE themes to features when they reach critical mass (your judgment!)
  - You can merge MULTIPLE related themes into ONE feature
  - Example: "oauth-work", "sessions", "jwt" → "Authentication System"
  - When promoting, all PRs from all themes are included in the new feature
  - Promoted themes are automatically deleted
  ```

- [ ] Update `DECISION_GUIDELINES` with theme examples

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

### Scenario 1: Simple Theme Promotion

```
PR #101 → Decision: create_theme "authentication-work"
  → Theme created (1 PR)

PR #104 → Decision: add_to_theme "authentication-work"
  → Theme updated (2 PRs)

PR #107 → Decision: add_to_theme "authentication-work"
  → Theme updated (3 PRs)

PR #110 → Decision: add_to_theme + promote_theme
  → Theme "authentication-work" promoted to Feature "Authentication System"
  → Theme deleted
  → Feature created (4 PRs)

PR #115 → Decision: add_to_existing "authentication-system"
  → Normal feature operation
```

### Scenario 2: Merging Multiple Themes

```
Current state:
  - Theme "oauth-work" (3 PRs)
  - Theme "session-handling" (2 PRs)
  - Theme "jwt-tokens" (2 PRs)

PR #120 → Decision: promote_theme
  → promotions: [{
      themeIds: ["oauth-work", "session-handling", "jwt-tokens"],
      featureName: "Authentication System",
      ...
    }]
  → All 3 themes merged into Feature "Authentication System" (7 PRs)
  → All 3 themes deleted
```

### Scenario 3: Simultaneous Multiple Promotions

```
Current state:
  - Theme "auth-work" (5 PRs)
  - Theme "payment-stripe" (4 PRs)
  - Theme "email-notifications" (3 PRs)

PR #200 → Decision: add_to_theme "email-notifications" + promote_theme
  → promotions: [
      { themeIds: ["auth-work"], featureName: "Authentication", ... },
      { themeIds: ["payment-stripe"], featureName: "Payment Processing", ... }
    ]
  → 2 features created
  → 2 themes deleted
  → 1 theme remains
```

## Future Enhancements (Optional)

1. **Theme expiry**: Auto-delete themes with no activity for 90+ days
2. **Theme merging**: LLM can merge two themes without promoting to feature
3. **Theme splitting**: LLM can split one theme into two if scope diverges
4. **Confidence scores**: Track theme "strength" metric (number of PRs, recency, etc.)

## Success Metrics

After implementation, we should see:

- Fewer premature features (features with < 3 PRs)
- Better feature quality (more comprehensive descriptions)
- Emerging patterns captured (themes accumulate related work)
- Cleaner feature list (only substantial work promoted)
