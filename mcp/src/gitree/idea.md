The user wants:

1. Change "action" to use an array of actions instead of having a "both" option
2. Don't implement the regex parsing for PR markdown - mark it as TODO
3. Don't implement LLM calls - mark as TODO
4. Mark any other pieces that need implementation with TODO

So the action should be like:

```typescript
actions: ('add_to_existing' | 'create_new' | 'ignore')[]
```

This way you can have `['add_to_existing', 'create_new']` instead of 'both'.

Let me revise the markdown doc accordingly.

# GitHub Feature Knowledge Base - Implementation Spec (Simplified)

## Overview

A tool that processes GitHub PRs chronologically, using an LLM to organize them into conceptual features. A PR can belong to multiple features. A feature is just a conceptual grouping with a name, description, and list of related PRs.

**Key Principle**: Maximum simplicity. A graph where Features → PRs (many-to-many relationship).

## Core Concept

```
Feature = { name, description, prNumbers[] }
PR = { number, title, summary, mergedAt }

Relationships stored in Feature objects (not in PRs)
A PR can appear in multiple Features
```

---

## Core Types

```typescript
interface Feature {
  id: string; // Slug from name (e.g., "auth-system")
  name: string; // Human-readable (e.g., "Authentication System")
  description: string; // What this feature is about
  prNumbers: number[]; // All PRs that touched this feature
  createdAt: Date;
  lastUpdated: Date;
}

interface PRRecord {
  number: number;
  title: string;
  summary: string; // LLM-generated summary of what this PR does
  mergedAt: Date;
  url: string;
}

// No complex KnowledgeBase object needed!
// Just files on disk that reference each other
```

---

## Main Class: StreamingFeatureBuilder

```typescript
class StreamingFeatureBuilder {
  constructor(
    private storage: Storage,
    private llm: LLMClient,
    private github: GitHubClient
  ) {}

  /**
   * Main entry point: process a repo
   */
  async processRepo(owner: string, repo: string): Promise<void> {
    const lastProcessed = await this.storage.getLastProcessedPR();

    // TODO: Implement fetching PRs from GitHub
    const prs = await this.github.fetchPRs(owner, repo, {
      since: lastProcessed,
      state: "closed",
      sort: "created",
      direction: "asc",
    });

    console.log(
      `Processing ${prs.length} PRs starting from #${lastProcessed + 1}...`
    );

    for (const pr of prs) {
      await this.processPR(pr);
      await this.storage.setLastProcessedPR(pr.number);
      console.log(`✅ Processed PR #${pr.number}`);
    }

    const features = await this.storage.getAllFeatures();
    console.log(`\nDone! Total features: ${features.length}`);
  }

  /**
   * Process a single PR
   */
  private async processPR(pr: GitHubPR): Promise<void> {
    // Skip obvious noise
    if (this.shouldSkip(pr)) {
      console.log(`⏭️  Skipping #${pr.number}: ${pr.title}`);

      // Still save the PR record for completeness
      await this.storage.savePR({
        number: pr.number,
        title: pr.title,
        summary: "Skipped (maintenance/trivial)",
        mergedAt: pr.mergedAt,
        url: pr.url,
      });
      return;
    }

    // Get current features for context
    const features = await this.storage.getAllFeatures();

    // Ask LLM what to do
    const prompt = this.buildDecisionPrompt(pr, features);

    // TODO: Implement LLM call
    const decision = await this.llm.decide(prompt);

    // Apply decision
    await this.applyDecision(pr, decision);
  }

  /**
   * Quick heuristic filter (no LLM needed)
   */
  private shouldSkip(pr: GitHubPR): boolean {
    const skipPatterns = [
      /^bump/i,
      /^chore:/i,
      /dependabot/i,
      /^docs:/i,
      /typo/i,
      /^ci:/i,
    ];

    return skipPatterns.some((pattern) => pattern.test(pr.title));
  }

  /**
   * Build the decision prompt
   */
  private buildDecisionPrompt(pr: GitHubPR, features: Feature[]): string {
    return `${SYSTEM_PROMPT}

${this.formatFeatureContext(features)}

${this.formatPRDetails(pr)}

${DECISION_FORMAT}`;
  }

  /**
   * Format feature list for context
   */
  private formatFeatureContext(features: Feature[]): string {
    if (features.length === 0) {
      return "## Current Features\n\nNo features yet.";
    }

    const featureList = features
      .sort((a, b) => b.lastUpdated.getTime() - a.lastUpdated.getTime())
      .map(
        (f) =>
          `- **${f.name}** (\`${f.id}\`): ${f.description} [${f.prNumbers.length} PRs]`
      )
      .join("\n");

    return `## Current Features\n\n${featureList}`;
  }

  /**
   * Format PR details
   */
  private formatPRDetails(pr: GitHubPR): string {
    const files = pr.filesChanged.slice(0, 20).join("\n");
    const moreFiles =
      pr.filesChanged.length > 20
        ? `\n... and ${pr.filesChanged.length - 20} more files`
        : "";

    return `## PR to Analyze

**#${pr.number}**: ${pr.title}

**Description**:
${pr.body || "No description provided"}

**Files Changed** (${pr.filesChanged.length} total):
${files}${moreFiles}

**Stats**: +${pr.additions} -${pr.deletions} lines`;
  }

  /**
   * Apply LLM decision
   */
  private async applyDecision(
    pr: GitHubPR,
    decision: LLMDecision
  ): Promise<void> {
    // Save PR record
    const prRecord: PRRecord = {
      number: pr.number,
      title: pr.title,
      summary: decision.summary,
      mergedAt: pr.mergedAt,
      url: pr.url,
    };
    await this.storage.savePR(prRecord);

    // Process each action
    for (const action of decision.actions) {
      if (action === "ignore") {
        console.log(`   Ignored: ${decision.reasoning}`);
        continue;
      }

      if (action === "add_to_existing") {
        // Add to existing feature(s)
        if (
          decision.existingFeatureIds &&
          decision.existingFeatureIds.length > 0
        ) {
          for (const featureId of decision.existingFeatureIds) {
            const feature = await this.storage.getFeature(featureId);
            if (feature) {
              if (!feature.prNumbers.includes(pr.number)) {
                feature.prNumbers.push(pr.number);
                feature.lastUpdated = pr.mergedAt;
                await this.storage.saveFeature(feature);
                console.log(`   → Added to feature: ${feature.name}`);
              }
            }
          }
        }
      }

      if (action === "create_new") {
        // Create new feature(s)
        if (decision.newFeatures && decision.newFeatures.length > 0) {
          for (const newFeatureData of decision.newFeatures) {
            const newFeature: Feature = {
              id: this.generateFeatureId(newFeatureData.name),
              name: newFeatureData.name,
              description: newFeatureData.description,
              prNumbers: [pr.number],
              createdAt: pr.mergedAt,
              lastUpdated: pr.mergedAt,
            };
            await this.storage.saveFeature(newFeature);
            console.log(`   ✨ Created new feature: ${newFeature.name}`);
          }
        }
      }
    }
  }

  /**
   * Generate slug-style feature ID from name
   */
  private generateFeatureId(name: string): string {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
  }
}

interface LLMDecision {
  actions: ("add_to_existing" | "create_new" | "ignore")[]; // Can have multiple actions
  existingFeatureIds?: string[]; // Which features to add to
  newFeatures?: Array<{
    // Which features to create
    name: string;
    description: string;
  }>;
  summary: string; // Summary of the PR itself
  reasoning: string;
}
```

---

## Prompts

### System Prompt

```typescript
const SYSTEM_PROMPT = `You are a software historian analyzing a codebase chronologically.

Your job: Read each PR and decide which feature(s) it belongs to.

**Key points:**
- A PR can belong to MULTIPLE features (e.g., a Google OAuth PR touches both "Authentication" and "Google Integration")
- ONLY create new features for significant capabilities - be conservative!
- Most PRs should add to existing features
- When in doubt, add to existing rather than create new

**Your actions (can combine multiple):**
1. Add to one or more existing features
2. Create one or more new features (RARE!)
3. Ignore (if truly trivial - rare since we pre-filter)

Think: "What conceptual feature(s) does this PR contribute to?"`;
```

### Decision Format

```typescript
const DECISION_FORMAT = `## Your Decision

Respond with JSON only:

{
  "actions": ["add_to_existing" | "create_new" | "ignore"],  // Array of actions
  
  // If adding to existing (can be multiple):
  "existingFeatureIds": ["feature-id-1", "feature-id-2"],
  
  // If creating new (can be multiple):
  "newFeatures": [
    {
      "name": "Feature Name",
      "description": "2-3 sentences explaining what this feature is"
    }
  ],
  
  // Always include:
  "summary": "One sentence: what does this PR do?",
  "reasoning": "Why did you make this decision?"
}

Examples:

1. Adding to existing:
{
  "actions": ["add_to_existing"],
  "existingFeatureIds": ["payment-processing"],
  "summary": "Adds Stripe webhook handlers for payment events",
  "reasoning": "Extends existing payment processing with webhook support"
}

2. Multiple features:
{
  "actions": ["add_to_existing"],
  "existingFeatureIds": ["authentication", "google-integration"],
  "summary": "Implements Google OAuth login",
  "reasoning": "Touches both auth system and Google integration"
}

3. Create new:
{
  "actions": ["create_new"],
  "newFeatures": [
    {
      "name": "Real-time Notifications",
      "description": "WebSocket-based real-time notification system that pushes updates to users without polling. Includes presence detection and typing indicators."
    }
  ],
  "summary": "Initial WebSocket notification system",
  "reasoning": "This is a major new capability not covered by existing features"
}

4. Both add and create:
{
  "actions": ["add_to_existing", "create_new"],
  "existingFeatureIds": ["data-export"],
  "newFeatures": [
    {
      "name": "PDF Generation",
      "description": "Server-side PDF generation using Puppeteer. Converts HTML reports to downloadable PDFs with custom styling and pagination."
    }
  ],
  "summary": "Adds PDF export to existing data export feature",
  "reasoning": "Extends data export but PDF generation is significant enough to track separately"
}

5. Ignore:
{
  "actions": ["ignore"],
  "summary": "Updates development dependencies",
  "reasoning": "Pure maintenance work with no functional changes"
}`;
```

---

## Storage Abstraction

```typescript
/**
 * Abstract storage interface
 */
abstract class Storage {
  // Features
  abstract saveFeature(feature: Feature): Promise<void>;
  abstract getFeature(id: string): Promise<Feature | null>;
  abstract getAllFeatures(): Promise<Feature[]>;
  abstract deleteFeature(id: string): Promise<void>;

  // PRs
  abstract savePR(pr: PRRecord): Promise<void>;
  abstract getPR(number: number): Promise<PRRecord | null>;
  abstract getAllPRs(): Promise<PRRecord[]>;

  // Metadata
  abstract getLastProcessedPR(): Promise<number>;
  abstract setLastProcessedPR(number: number): Promise<void>;

  // Query helpers (derived from the graph)
  async getPRsForFeature(featureId: string): Promise<PRRecord[]> {
    const feature = await this.getFeature(featureId);
    if (!feature) return [];

    const prs: PRRecord[] = [];
    for (const prNumber of feature.prNumbers) {
      const pr = await this.getPR(prNumber);
      if (pr) prs.push(pr);
    }
    return prs;
  }

  async getFeaturesForPR(prNumber: number): Promise<Feature[]> {
    const allFeatures = await this.getAllFeatures();
    return allFeatures.filter((f) => f.prNumbers.includes(prNumber));
  }
}
```

---

## FileSystemStore Implementation

### Directory Structure

```
./knowledge-base/
  ├── metadata.json          # { lastProcessedPR: 123 }
  ├── features/
  │   ├── auth-system.json
  │   ├── google-integration.json
  │   └── payment-processing.json
  └── prs/
      ├── 1.md
      ├── 2.md
      └── 1034.md
```

### Feature JSON Format

```json
{
  "id": "auth-system",
  "name": "Authentication System",
  "description": "Complete authentication system with OAuth support, session management, and permission handling.",
  "prNumbers": [12, 15, 23, 45, 67, 89],
  "createdAt": "2023-01-15T10:30:00Z",
  "lastUpdated": "2023-03-20T14:22:00Z"
}
```

### PR Markdown Format

```markdown
# PR #1034: Add Google OAuth Support

**Merged**: 2023-02-10  
**URL**: https://github.com/owner/repo/pull/1034

## Summary

Integrates Google OAuth 2.0 into the authentication system. Users can now sign in with their Google accounts using the standard OAuth flow.

---

_Part of features: `auth-system`, `google-integration`_
```

### Implementation

```typescript
import fs from "fs/promises";
import path from "path";

class FileSystemStore extends Storage {
  private baseDir: string;
  private featuresDir: string;
  private prsDir: string;
  private metadataPath: string;

  constructor(baseDir: string = "./knowledge-base") {
    super();
    this.baseDir = baseDir;
    this.featuresDir = path.join(baseDir, "features");
    this.prsDir = path.join(baseDir, "prs");
    this.metadataPath = path.join(baseDir, "metadata.json");
  }

  /**
   * Initialize directory structure
   */
  async initialize(): Promise<void> {
    await fs.mkdir(this.featuresDir, { recursive: true });
    await fs.mkdir(this.prsDir, { recursive: true });

    try {
      await fs.access(this.metadataPath);
    } catch {
      await fs.writeFile(
        this.metadataPath,
        JSON.stringify({ lastProcessedPR: 0 }, null, 2)
      );
    }
  }

  // Features
  async saveFeature(feature: Feature): Promise<void> {
    const filePath = path.join(this.featuresDir, `${feature.id}.json`);
    await fs.writeFile(filePath, JSON.stringify(feature, null, 2));
  }

  async getFeature(id: string): Promise<Feature | null> {
    const filePath = path.join(this.featuresDir, `${id}.json`);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      return JSON.parse(content);
    } catch {
      return null;
    }
  }

  async getAllFeatures(): Promise<Feature[]> {
    try {
      const files = await fs.readdir(this.featuresDir);
      const features: Feature[] = [];

      for (const file of files) {
        if (file.endsWith(".json")) {
          const content = await fs.readFile(
            path.join(this.featuresDir, file),
            "utf-8"
          );
          features.push(JSON.parse(content));
        }
      }

      return features;
    } catch {
      return [];
    }
  }

  async deleteFeature(id: string): Promise<void> {
    const filePath = path.join(this.featuresDir, `${id}.json`);
    await fs.unlink(filePath);
  }

  // PRs
  async savePR(pr: PRRecord): Promise<void> {
    const filePath = path.join(this.prsDir, `${pr.number}.md`);
    const content = await this.formatPRMarkdown(pr);
    await fs.writeFile(filePath, content);
  }

  async getPR(number: number): Promise<PRRecord | null> {
    const filePath = path.join(this.prsDir, `${number}.md`);
    try {
      const content = await fs.readFile(filePath, "utf-8");
      // TODO: Implement parsing PR markdown back to PRRecord
      return this.parsePRMarkdown(number, content);
    } catch {
      return null;
    }
  }

  async getAllPRs(): Promise<PRRecord[]> {
    try {
      const files = await fs.readdir(this.prsDir);
      const prs: PRRecord[] = [];

      for (const file of files) {
        if (file.endsWith(".md")) {
          const prNumber = parseInt(file.replace(".md", ""));
          const pr = await this.getPR(prNumber);
          if (pr) prs.push(pr);
        }
      }

      return prs.sort((a, b) => a.number - b.number);
    } catch {
      return [];
    }
  }

  // Metadata
  async getLastProcessedPR(): Promise<number> {
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      const metadata = JSON.parse(content);
      return metadata.lastProcessedPR || 0;
    } catch {
      return 0;
    }
  }

  async setLastProcessedPR(number: number): Promise<void> {
    let metadata = { lastProcessedPR: 0 };
    try {
      const content = await fs.readFile(this.metadataPath, "utf-8");
      metadata = JSON.parse(content);
    } catch {
      // File doesn't exist yet
    }
    metadata.lastProcessedPR = number;
    await fs.writeFile(this.metadataPath, JSON.stringify(metadata, null, 2));
  }

  // Helpers
  private async formatPRMarkdown(pr: PRRecord): Promise<string> {
    // Get features this PR belongs to
    const features = await this.getFeaturesForPR(pr.number);
    const featureLinks =
      features.length > 0
        ? `\n---\n\n_Part of features: ${features
            .map((f) => `\`${f.id}\``)
            .join(", ")}_`
        : "";

    return `# PR #${pr.number}: ${pr.title}

**Merged**: ${pr.mergedAt.toISOString().split("T")[0]}  
**URL**: ${pr.url}

## Summary

${pr.summary}${featureLinks}
`.trim();
  }

  // TODO: Implement parsing markdown back to PRRecord
  private parsePRMarkdown(number: number, content: string): PRRecord {
    // This will parse the markdown format back into a PRRecord
    // You'll implement the actual parsing logic
    throw new Error("TODO: Implement parsePRMarkdown");
  }
}
```

---

## GitHub Client Interface

```typescript
// TODO: Implement GitHub client
interface GitHubClient {
  fetchPRs(
    owner: string,
    repo: string,
    options: {
      since: number;
      state: "open" | "closed" | "all";
      sort: "created" | "updated";
      direction: "asc" | "desc";
    }
  ): Promise<GitHubPR[]>;
}

interface GitHubPR {
  number: number;
  title: string;
  body: string | null;
  url: string;
  mergedAt: Date;
  additions: number;
  deletions: number;
  filesChanged: string[];
}
```

---

## LLM Client Interface

```typescript
// TODO: Implement LLM client
interface LLMClient {
  decide(prompt: string): Promise<LLMDecision>;
}
```

---

## Usage Example

```typescript
// Setup
const storage = new FileSystemStore("./my-repo-knowledge");
await storage.initialize();

// TODO: Implement these clients
const llm = new OpenAIClient({ apiKey: process.env.OPENAI_API_KEY });
const github = new GitHubClient({ token: process.env.GITHUB_TOKEN });

const builder = new StreamingFeatureBuilder(storage, llm, github);

// Process repo
await builder.processRepo("facebook", "react");

// Query: What features exist?
const features = await storage.getAllFeatures();
console.log(
  "Features:",
  features.map((f) => f.name)
);

// Query: What's in the auth feature?
const authPRs = await storage.getPRsForFeature("auth-system");
console.log(`Auth built over ${authPRs.length} PRs`);

// Query: What features does PR #123 touch?
const pr123Features = await storage.getFeaturesForPR(123);
console.log(
  "PR #123 touched:",
  pr123Features.map((f) => f.name)
);
```

---

## Implementation Checklist

- [ ] Define TypeScript interfaces
- [ ] Implement `Storage` abstract class with helper methods
- [ ] Implement `FileSystemStore`
  - [ ] All methods except `parsePRMarkdown` (marked TODO)
- [ ] TODO: Create GitHub client wrapper
- [ ] TODO: Create LLM client wrapper
- [ ] Implement `StreamingFeatureBuilder`
  - [ ] All logic except actual LLM/GitHub calls (marked TODO)
- [ ] Add CLI interface
- [ ] Test with small repo
- [ ] Add query commands (list features, show feature, etc.)
