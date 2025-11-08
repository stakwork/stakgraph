# Themes - Lightweight Context Hints

## Problem

Currently, features are decided while reviewing a **single PR** with no memory of recent low-level technical work. This causes:

1. **No context**: LLM can't see patterns in recent technical work (jwt, oauth, redis, etc.)
2. **Premature feature creation**: Without seeing related work, features get created too early
3. **Missed patterns**: Related low-level work across PRs goes unrecognized

For large repos with many contributors, the LLM needs **lightweight context** about recent technical activity.

## Solution: Sliding Window of Theme Tags

Add a **sliding window** of the last 100 theme tags - simple strings that represent recent low-level technical work.

### Key Principles

1. **Ultra lightweight**: Themes are just strings, not entities
2. **Sliding window**: Only last 100 themes kept
3. **LRU behavior**: Reusing a theme moves it to the front (most recent)
4. **Context clues**: Helps LLM recognize patterns and make better feature decisions
5. **No complex logic**: No promotion, merging, or deletion logic needed

### What Are Themes?

**Themes** = Low-level technical tags that provide context

- Implementation details: "jwt", "redis-caching", "stripe-webhooks"
- Protocols/technologies: "oauth", "websockets", "graphql"
- Architectural patterns: "event-sourcing", "pub-sub"
- File patterns: "api-routes", "database-migrations"

**NOT tracked**: Theme descriptions, PR lists, creation dates, etc. Just the string!

### Themes vs Features

**Themes** = Lightweight context hints

- Just strings in a sliding window
- Shows what low-level work has been happening
- Provides "memory" of recent technical activity
- Example: ["jwt", "oauth", "redis", "websockets", "stripe-webhooks", ...]

**Features** = Tracked entities (unchanged from today)

- High-level capabilities with descriptions
- Track PRs, dates, documentation
- What users can DO
- Example: "Authentication System", "Payment Processing"

## Data Structures

### Metadata (Updated)

```typescript
interface Metadata {
  lastProcessedPR: number;
  recentThemes: string[]; // NEW: Sliding window, max 100 theme tags
}
```

### LLM Decision

```typescript
interface LLMDecision {
  actions: ("add_to_existing" | "create_new" | "ignore")[];

  // Add to existing features (unchanged)
  featureIds?: string[];

  // Create new features (unchanged)
  newFeatures?: Array<{
    name: string;
    description: string;
  }>;

  // Theme tags for this PR (NEW - super simple!)
  themes?: string[]; // 1-3 theme tags (can be new or existing)

  summary: string;
  reasoning: string;
  newDeclarations?: Array<{
    file: string;
    declarations: string[];
  }>;
}
```

**Note**: No complex promotion, merging, or theme management! LLM just picks 1-3 theme tags.

## LLM Context

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
Recent Technical Themes: jwt, oauth, redis, websockets, stripe-webhooks, graphql, ...
PR to Analyze: [PR content]
Decision Guidelines (updated)
```

## How It Works

### Theme Tag Behavior

```typescript
// PR #101: LLM adds "jwt"
recentThemes = ["jwt"];

// PR #102: LLM adds "oauth"
recentThemes = ["jwt", "oauth"];

// PR #103: LLM adds "redis"
recentThemes = ["jwt", "oauth", "redis"];

// PR #104: LLM reuses "jwt" (moves to front!)
recentThemes = ["oauth", "redis", "jwt"];

// PR #105: LLM adds "websockets"
recentThemes = ["oauth", "redis", "jwt", "websockets"];

// ... after 100 theme tags added ...
// Oldest themes automatically drop off
```

### LLM Decision Process

1. **Look at recent themes**: See what low-level work has been happening
2. **Decide on features**: Should this PR add to existing feature or create new one?
3. **Tag with themes**: Add 1-3 theme tags (new or existing) for future context

**Example**:

```json
{
  "actions": ["create_new"],
  "newFeatures": [
    {
      "name": "Authentication System",
      "description": "Complete OAuth-based authentication with JWT tokens and session management"
    }
  ],
  "themes": ["jwt", "oauth", "sessions"],
  "summary": "Complete authentication system",
  "reasoning": "Multiple auth-related PRs (see themes: jwt, oauth, sessions) - now substantial enough for a feature"
}
```

The LLM uses the recent themes as **hints** to recognize patterns!

## Implementation Checklist

### 1. Update Type Definitions

- [ ] Update `Metadata` interface in storage to include `recentThemes: string[]`
- [ ] Update `LLMDecision` interface to add optional `themes?: string[]`
- [ ] Update Zod schema in `llm.ts` to include themes field

### 2. Update Storage

- [ ] Add theme management to abstract `Storage` class:

  ```typescript
  abstract addThemes(themes: string[]): Promise<void>;
  abstract getRecentThemes(): Promise<string[]>;
  ```

- [ ] Implement in `FileSystemStore`:

  ```typescript
  async addThemes(themes: string[]): Promise<void> {
    const metadata = await this.getMetadata();

    // Remove themes if they already exist (LRU behavior)
    metadata.recentThemes = metadata.recentThemes.filter(t => !themes.includes(t));

    // Add to end (most recent)
    metadata.recentThemes.push(...themes);

    // Keep only last 100
    if (metadata.recentThemes.length > 100) {
      metadata.recentThemes = metadata.recentThemes.slice(-100);
    }

    await this.saveMetadata(metadata);
  }
  ```

- [ ] Implement in `GraphStorage` (similar logic in metadata node)

- [ ] Update metadata initialization to include `recentThemes: []`

### 3. Update Builder

- [ ] Modify `buildDecisionPrompt()` to include themes context:

  ```typescript
  private async formatThemeContext(): Promise<string> {
    const themes = await this.storage.getRecentThemes();

    if (themes.length === 0) {
      return "## Recent Technical Themes\n\nNo recent themes.";
    }

    // Show themes in reverse order (most recent first)
    const themeList = themes.slice().reverse().slice(0, 50).join(", ");

    return `## Recent Technical Themes (last 50 of ${themes.length})\n\n${themeList}`;
  }
  ```

- [ ] Update `applyDecision()` to save themes:
  ```typescript
  // After saving PR
  if (decision.themes && decision.themes.length > 0) {
    await this.storage.addThemes(decision.themes);
    console.log(`   🏷️  Tagged: ${decision.themes.join(", ")}`);
  }
  ```

### 4. Update LLM Prompts

- [ ] Update `SYSTEM_PROMPT` in `llm.ts`:

  ```typescript
  **Using Theme Tags:**
  - You'll see a list of recent technical themes (last 100 low-level tags)
  - These are lightweight context hints showing recent technical work
  - Examples: "jwt", "oauth", "redis", "webhooks", "graphql"
  - Use these as clues to recognize patterns across PRs
  - When you see related themes accumulating, it might be time to create a feature
  ```

- [ ] Update `DECISION_GUIDELINES`:

  ```typescript
  **Theme Tags (Optional):**
  - Add 1-3 theme tags to help track low-level technical work
  - Can be NEW tags or EXISTING tags from recent themes
  - Examples: "jwt", "oauth-flow", "redis-caching", "stripe-webhooks"
  - Keep them short and technical
  - These provide context for future PR decisions

  Example with themes:
  {
    "actions": ["add_to_existing"],
    "featureIds": ["authentication-system"],
    "themes": ["jwt", "session-management"],
    "summary": "Adds JWT refresh token logic",
    "reasoning": "Extends auth feature with refresh tokens"
  }

  Example recognizing pattern:
  {
    "actions": ["create_new"],
    "newFeatures": [{
      "name": "Authentication System",
      "description": "OAuth-based authentication with JWT and sessions"
    }],
    "themes": ["auth-complete"],
    "summary": "Completes authentication system",
    "reasoning": "Multiple auth PRs (themes show: jwt, oauth, sessions) - now ready for a feature"
  }
  ```

### 5. Update CLI

- [ ] Add `show-themes` command:

  ```bash
  yarn gitree show-themes
  # Shows recent themes in order
  ```

- [ ] Update `stats` to include theme count

### 6. Update API Routes

- [ ] Add `GET /gitree/themes` - get recent themes array

### 7. Update Store Utils

- [ ] Update `formatPRMarkdown()` to include themes if present:
  ```markdown
  _Part of features: `auth-system`, `google-integration`_
  _Themes: jwt, oauth, sessions_
  ```

## Example Workflow

```
PR #101: "Add JWT token generation"
  → Decision: create_new feature? No, too small
  → themes: ["jwt"]
  → Recent themes: ["jwt"]

PR #104: "Add OAuth callback"
  → Decision: create_new feature? No, too small
  → themes: ["oauth"]
  → Recent themes: ["jwt", "oauth"]

PR #107: "Add session storage with Redis"
  → Decision: create_new feature? No, too small
  → themes: ["sessions", "redis"]
  → Recent themes: ["jwt", "oauth", "sessions", "redis"]

PR #110: "Connect JWT to OAuth flow"
  → LLM sees themes: jwt, oauth, sessions, redis
  → Decision: Still building, not ready for feature
  → themes: ["jwt", "oauth"]  (reusing - bubbles to front!)
  → Recent themes: ["sessions", "redis", "jwt", "oauth"]

PR #115: "Complete auth with user management"
  → LLM sees themes: sessions, redis, jwt, oauth
  → Decision: PATTERN RECOGNIZED! Create feature!
  → actions: ["create_new"]
  → newFeatures: [{ name: "Authentication System", ... }]
  → themes: ["auth-complete"]
  → Recent themes: ["sessions", "redis", "jwt", "oauth", "auth-complete"]

PR #120: "Add password reset to auth"
  → Decision: add_to_existing "authentication-system"
  → themes: ["jwt", "sessions"]  (reusing - bubbles to front!)
  → Recent themes: ["redis", "oauth", "auth-complete", "jwt", "sessions"]
```

## Benefits

1. **Ultra simple**: Just an array of strings, no complex entities
2. **Natural memory**: LLM can see recent technical work
3. **Pattern recognition**: Related themes accumulate, signaling when to create features
4. **LRU behavior**: Frequently-used themes stay near the front
5. **No maintenance**: Automatic sliding window, no cleanup needed
6. **Lightweight**: No storage overhead, just metadata

## Success Metrics

After implementation, we should see:

- **Better feature timing**: LLM creates features when patterns emerge (evidenced by theme accumulation)
- **Context awareness**: LLM references themes in reasoning ("seeing jwt, oauth, sessions themes...")
- **Natural coalescence**: Related technical work (themes) leads to feature creation
- **Minimal overhead**: Just string array management, no complex logic
