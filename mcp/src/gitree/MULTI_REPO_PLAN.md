# Multi-Repo Support Implementation Plan

## Overview

This document outlines the implementation plan for adding multi-repo support to the gitree feature. Currently, Feature/PR/Commit nodes have no repo identifier, which means processing multiple repos would cause collisions (e.g., PR #1 from repo-A would overwrite PR #1 from repo-B).

## Current State

### Node Structure (No Repo Marker)

- **Feature**: `id`, `name`, `description`, `prNumbers`, `commitShas`, etc.
- **PRRecord**: `number`, `title`, `summary`, `url`, `files`, etc.
- **CommitRecord**: `sha`, `message`, `summary`, `url`, `files`, etc.
- **Clue**: `id`, `featureId`, `type`, `title`, `content`, etc.
- **FeaturesMetadata**: Global checkpoint with `namespace: "default"`

### Problems

1. PR numbers collide across repos (MERGE uses `{number}` as key)
2. Feature IDs collide across repos (MERGE uses `{id}` as key)
3. Checkpoints are global, not per-repo

### Existing Data That Helps

- `PRRecord.url` contains `https://github.com/owner/repo/pull/123`
- `CommitRecord.url` contains `https://github.com/owner/repo/commit/sha`
- We can parse repo from these URLs for migration!

---

## Design Decisions

### 1. Repo Identifier Format

Use `owner/repo` format (e.g., `"stakwork/hive"`).

### 2. ID Format

New IDs are repo-prefixed for global uniqueness:
- Feature: `owner/repo/feature-slug` (e.g., `stakwork/hive/auth-system`)
- PR: `owner/repo/pr-123` (e.g., `stakwork/hive/pr-456`)
- Commit: `owner/repo/commit-abc1234` (e.g., `stakwork/hive/commit-abc1234`)
- Clue: `owner/repo/clue-slug` (e.g., `stakwork/hive/jwt-token-utils`)

### 3. Query Behavior

**`repo` parameter is ALWAYS optional in queries.**

| Query | With `?repo=owner/repo` | Without `?repo=` |
|-------|-------------------------|------------------|
| List features | Filter to that repo | Return all features across all repos |
| Search features | Search within that repo | Search across all repos |
| Get PR by number | Filter to that repo | Return all PRs with that number (from all repos) |
| Get commit by SHA | Filter to that repo | Return commit (SHA is globally unique anyway) |
| List clues | Filter to that repo | Return all clues |
| Get stats | Stats for that repo | Aggregate stats across all repos |

### 4. Repo Parameter Flexibility

The `?repo=` query parameter should accept multiple formats:
- `?repo=owner/repo` (short form)
- `?repo=https://github.com/owner/repo` (full URL)
- `?repo=https://github.com/owner/repo.git` (with .git suffix)
- `?repo=git@github.com:owner/repo.git` (SSH format)

All formats are normalized to `owner/repo` internally.

### 5. Backwards Compatibility

- Existing nodes are migrated automatically on startup
- `repo` field is optional in TypeScript interfaces (for migration period)
- Legacy nodes without `repo` field still work (treated as "unknown" repo)

---

## Implementation Steps

### Step 1: Update Types (`types.ts`)

Add optional `repo` field to all relevant interfaces:

```typescript
export interface Feature {
  id: string; // Now: "owner/repo/feature-slug"
  repo?: string; // "owner/repo" - optional for backwards compat
  // ... rest unchanged
}

export interface PRRecord {
  number: number;
  repo?: string; // "owner/repo"
  // ... rest unchanged
}

export interface CommitRecord {
  sha: string;
  repo?: string; // "owner/repo"
  // ... rest unchanged
}

export interface Clue {
  id: string; // Now: "owner/repo/clue-slug"
  repo?: string; // "owner/repo"
  // ... rest unchanged
}
```

### Step 2: Add Utility Functions (`store/utils.ts`)

```typescript
/**
 * Parse "owner/repo" from various URL formats
 * Supports:
 * - https://github.com/owner/repo
 * - https://github.com/owner/repo/pull/123
 * - https://github.com/owner/repo/commit/sha
 * - git@github.com:owner/repo.git
 * - owner/repo
 */
export function parseRepoFromUrl(url: string): string | null {
  if (!url) return null;
  
  // Remove trailing .git if present
  let cleanUrl = url.replace(/\.git$/, "");
  
  // Handle SSH format (git@host:owner/repo)
  const sshMatch = cleanUrl.match(/git@[^:]+:(.+)/);
  if (sshMatch) {
    cleanUrl = sshMatch[1];
  }
  
  // Remove protocol and domain
  cleanUrl = cleanUrl.replace(/^https?:\/\//, "");
  cleanUrl = cleanUrl.replace(/^[^\/]+\//, ""); // Remove domain/host
  
  // Extract owner/repo (first two path segments)
  const pathParts = cleanUrl.split("/").filter((p) => p.length > 0);
  
  if (pathParts.length >= 2) {
    return `${pathParts[0]}/${pathParts[1]}`;
  }
  
  return null;
}

/**
 * Generate repo-prefixed ID
 */
export function makeRepoId(repo: string, slug: string): string {
  return `${repo}/${slug}`;
}

/**
 * Extract slug from repo-prefixed ID
 */
export function getSlugFromRepoId(repoId: string): string {
  const parts = repoId.split("/");
  return parts.length >= 3 ? parts.slice(2).join("/") : repoId;
}

/**
 * Extract repo from repo-prefixed ID
 */
export function getRepoFromRepoId(repoId: string): string | null {
  const parts = repoId.split("/");
  return parts.length >= 3 ? `${parts[0]}/${parts[1]}` : null;
}
```

### Step 3: Add Migration Logic (`store/graphStorage.ts`)

Add to `initialize()` method:

```typescript
async initialize(): Promise<void> {
  const session = this.driver.session();
  try {
    // Create indexes (existing code)
    // ...
    
    // Add new index on repo field
    await session.run(
      "CREATE INDEX feature_repo_index IF NOT EXISTS FOR (f:Feature) ON (f.repo)"
    );
    await session.run(
      "CREATE INDEX pr_repo_index IF NOT EXISTS FOR (p:PullRequest) ON (p.repo)"
    );
    await session.run(
      "CREATE INDEX commit_repo_index IF NOT EXISTS FOR (c:Commit) ON (c.repo)"
    );
    await session.run(
      "CREATE INDEX clue_repo_index IF NOT EXISTS FOR (c:Clue) ON (c.repo)"
    );
    
    // Run migration if needed
    await this.migrateToMultiRepo();
  } finally {
    await session.close();
  }
}

/**
 * Migrate existing data to multi-repo format
 * Only runs if migration is needed (checks first)
 */
private async migrateToMultiRepo(): Promise<void> {
  const session = this.driver.session();
  try {
    // Check if migration is needed
    const checkResult = await session.run(`
      MATCH (p:PullRequest) 
      WHERE p.repo IS NULL AND p.url IS NOT NULL
      RETURN count(p) as count
    `);
    
    const count = checkResult.records[0]?.get("count")?.toNumber() || 0;
    
    if (count === 0) {
      console.log("===> Multi-repo migration: Not needed (already migrated or no data)");
      return;
    }
    
    console.log(`===> Multi-repo migration: Migrating ${count} PRs and related nodes...`);
    
    // Step 1: Migrate PRs (parse repo from url)
    console.log("   Migrating PullRequest nodes...");
    await session.run(`
      MATCH (p:PullRequest)
      WHERE p.repo IS NULL AND p.url IS NOT NULL
      WITH p, 
           // Extract owner/repo from URL like https://github.com/owner/repo/pull/123
           split(replace(replace(p.url, 'https://github.com/', ''), '/pull/' + toString(p.number), ''), '/') as parts
      WHERE size(parts) >= 2
      SET p.repo = parts[0] + '/' + parts[1],
          p.legacyId = p.name,
          p.name = parts[0] + '/' + parts[1] + '/pr-' + toString(p.number)
    `);
    
    // Step 2: Migrate Commits (parse repo from url)
    console.log("   Migrating Commit nodes...");
    await session.run(`
      MATCH (c:Commit)
      WHERE c.repo IS NULL AND c.url IS NOT NULL
      WITH c,
           // Extract owner/repo from URL like https://github.com/owner/repo/commit/sha
           split(replace(p.url, 'https://github.com/', ''), '/') as parts
      WHERE size(parts) >= 2
      SET c.repo = parts[0] + '/' + parts[1],
          c.legacyId = c.name,
          c.name = parts[0] + '/' + parts[1] + '/commit-' + substring(c.sha, 0, 7)
    `);
    
    // Step 3: Migrate Features (infer repo from linked PRs)
    console.log("   Migrating Feature nodes...");
    await session.run(`
      MATCH (f:Feature)
      WHERE f.repo IS NULL
      OPTIONAL MATCH (p:PullRequest)-[:TOUCHES]->(f)
      WHERE p.repo IS NOT NULL
      WITH f, collect(DISTINCT p.repo)[0] as inferredRepo
      WHERE inferredRepo IS NOT NULL
      SET f.repo = inferredRepo,
          f.legacyId = f.id,
          f.id = inferredRepo + '/' + f.id
    `);
    
    // Step 4: Migrate Clues (infer repo from parent feature)
    console.log("   Migrating Clue nodes...");
    await session.run(`
      MATCH (c:Clue)
      WHERE c.repo IS NULL
      MATCH (f:Feature {id: c.featureId})
      WHERE f.repo IS NOT NULL
      SET c.repo = f.repo,
          c.legacyId = c.id,
          c.id = f.repo + '/' + c.id,
          c.featureId = f.id
    `);
    
    // Step 5: Migrate FeaturesMetadata to per-repo
    // Note: namespace stays "default" - we only add repo field for filtering
    console.log("   Migrating FeaturesMetadata nodes...");
    await session.run(`
      MATCH (m:FeaturesMetadata)
      WHERE m.repo IS NULL
      // Find most common repo from PRs to assign metadata to
      OPTIONAL MATCH (p:PullRequest)
      WHERE p.repo IS NOT NULL
      WITH m, collect(DISTINCT p.repo)[0] as inferredRepo
      WHERE inferredRepo IS NOT NULL
      SET m.repo = inferredRepo
    `);
    
    console.log("===> Multi-repo migration: Complete!");
    
  } catch (error) {
    console.error("===> Multi-repo migration: Error during migration:", error);
    // Don't throw - allow app to continue even if migration fails
  } finally {
    await session.close();
  }
}
```

### Step 4: Update Save Methods (`store/graphStorage.ts`)

Update all save methods to store `repo` and use prefixed IDs:

#### saveFeature()

```typescript
async saveFeature(feature: Feature): Promise<void> {
  const session = this.driver.session();
  try {
    const now = Math.floor(Date.now() / 1000);
    const dateTimestamp = Math.floor(feature.lastUpdated.getTime() / 1000);
    const cluesLastAnalyzedAtTimestamp = feature.cluesLastAnalyzedAt
      ? Math.floor(feature.cluesLastAnalyzedAt.getTime() / 1000)
      : null;

    await session.run(
      `
      MERGE (f:${Data_Bank}:Feature {id: $id})
      SET f.name = $name,
          f.repo = $repo,
          f.description = $description,
          f.prNumbers = $prNumbers,
          f.commitShas = $commitShas,
          f.date = $date,
          f.docs = $docs,
          f.cluesCount = $cluesCount,
          f.cluesLastAnalyzedAt = $cluesLastAnalyzedAt,
          f.namespace = $namespace,
          f.Data_Bank = $dataBankName,
          f.ref_id = COALESCE(f.ref_id, $refId),
          f.date_added_to_graph = COALESCE(f.date_added_to_graph, $dateAddedToGraph)
      RETURN f
      `,
      {
        id: feature.id, // Should already be repo-prefixed: "owner/repo/slug"
        repo: feature.repo || null,
        name: feature.name,
        // ... rest unchanged
      }
    );
    // ... rest unchanged
  } finally {
    await session.close();
  }
}
```

#### savePR()

```typescript
async savePR(pr: PRRecord): Promise<void> {
  const session = this.driver.session();
  try {
    // Generate repo-prefixed ID
    const prId = pr.repo ? `${pr.repo}/pr-${pr.number}` : `pr-${pr.number}`;
    
    await session.run(
      `
      MERGE (p:${Data_Bank}:PullRequest {id: $id})
      SET p.number = $number,
          p.repo = $repo,
          p.name = $name,
          p.title = $title,
          // ... rest unchanged
      `,
      {
        id: prId,
        number: pr.number,
        repo: pr.repo || null,
        name: prId,
        // ... rest unchanged
      }
    );
  } finally {
    await session.close();
  }
}
```

**IMPORTANT**: Change the MERGE key from `{number: $number}` to `{id: $id}` to support multi-repo!

#### saveCommit()

```typescript
async saveCommit(commit: CommitRecord): Promise<void> {
  const session = this.driver.session();
  try {
    // Generate repo-prefixed ID
    const commitId = commit.repo 
      ? `${commit.repo}/commit-${commit.sha.substring(0, 7)}` 
      : `commit-${commit.sha.substring(0, 7)}`;
    
    await session.run(
      `
      MERGE (c:${Data_Bank}:Commit {id: $id})
      SET c.sha = $sha,
          c.repo = $repo,
          c.name = $name,
          // ... rest unchanged
      `,
      {
        id: commitId,
        sha: commit.sha,
        repo: commit.repo || null,
        name: commitId,
        // ... rest unchanged
      }
    );
  } finally {
    await session.close();
  }
}
```

**IMPORTANT**: Change the MERGE key from `{sha: $sha}` to `{id: $id}` to support multi-repo!

#### saveClue()

Similar pattern - add `repo` field and use prefixed ID.

### Step 5: Update Query Methods (`store/graphStorage.ts`)

Add optional `repo` parameter to all query methods:

#### getAllFeatures()

```typescript
async getAllFeatures(repo?: string): Promise<Feature[]> {
  const session = this.driver.session();
  try {
    const result = await session.run(
      `
      MATCH (f:Feature)
      WHERE $repo IS NULL OR f.repo = $repo
      RETURN f
      ORDER BY f.date DESC
      `,
      { repo: repo || null }
    );

    return result.records.map((record) =>
      this.nodeToFeature(record.get("f"))
    );
  } finally {
    await session.close();
  }
}
```

#### getFeature()

```typescript
async getFeature(id: string, repo?: string): Promise<Feature | null> {
  const session = this.driver.session();
  try {
    // If repo provided and id doesn't have prefix, construct full id
    const fullId = repo && !id.includes('/') ? `${repo}/${id}` : id;
    
    const result = await session.run(
      `
      MATCH (f:Feature {id: $id})
      RETURN f
      `,
      { id: fullId }
    );

    if (result.records.length === 0) {
      return null;
    }

    return this.nodeToFeature(result.records[0].get("f"));
  } finally {
    await session.close();
  }
}
```

#### getPR()

```typescript
async getPR(number: number, repo?: string): Promise<PRRecord | null> {
  const session = this.driver.session();
  try {
    let query: string;
    let params: Record<string, any>;
    
    if (repo) {
      // Look up by repo-prefixed ID
      const prId = `${repo}/pr-${number}`;
      query = `MATCH (p:PullRequest {id: $id}) RETURN p`;
      params = { id: prId };
    } else {
      // Look up by number (may return multiple from different repos)
      query = `MATCH (p:PullRequest {number: $number}) RETURN p LIMIT 1`;
      params = { number };
    }
    
    const result = await session.run(query, params);

    if (result.records.length === 0) {
      return null;
    }

    return this.nodeToPR(result.records[0].get("p"));
  } finally {
    await session.close();
  }
}
```

#### getAllPRs()

```typescript
async getAllPRs(repo?: string): Promise<PRRecord[]> {
  const session = this.driver.session();
  try {
    const result = await session.run(
      `
      MATCH (p:PullRequest)
      WHERE $repo IS NULL OR p.repo = $repo
      RETURN p
      ORDER BY p.number ASC
      `,
      { repo: repo || null }
    );

    return result.records.map((record) => this.nodeToPR(record.get("p")));
  } finally {
    await session.close();
  }
}
```

Apply same pattern to:
- `getCommit()` / `getAllCommits()`
- `getClue()` / `getAllClues()`
- `getCluesForFeature()`
- `searchClues()`
- `getPRsForFeature()` / `getFeaturesForPR()`
- `getCommitsForFeature()` / `getFeaturesForCommit()`
- `linkFeaturesToFiles()`
- `getFilesForFeature()`
- `getAllFeaturesWithFilesAndContains()`
- `getProvenanceForConcepts()`

### Step 6: Update Storage Abstract Class (`storage.ts`)

Update method signatures to include optional `repo` parameter:

```typescript
export abstract class Storage {
  // Features
  abstract saveFeature(feature: Feature): Promise<void>;
  abstract getFeature(id: string, repo?: string): Promise<Feature | null>;
  abstract getAllFeatures(repo?: string): Promise<Feature[]>;
  abstract deleteFeature(id: string, repo?: string): Promise<void>;

  // PRs
  abstract savePR(pr: PRRecord): Promise<void>;
  abstract getPR(number: number, repo?: string): Promise<PRRecord | null>;
  abstract getAllPRs(repo?: string): Promise<PRRecord[]>;

  // Commits
  abstract saveCommit(commit: CommitRecord): Promise<void>;
  abstract getCommit(sha: string, repo?: string): Promise<CommitRecord | null>;
  abstract getAllCommits(repo?: string): Promise<CommitRecord[]>;

  // Clues
  abstract saveClue(clue: Clue): Promise<void>;
  abstract getClue(id: string, repo?: string): Promise<Clue | null>;
  abstract getAllClues(repo?: string): Promise<Clue[]>;
  abstract deleteClue(id: string, repo?: string): Promise<void>;
  abstract searchClues(
    query: string,
    embeddings: number[],
    featureId?: string,
    limit?: number,
    similarityThreshold?: number,
    repo?: string
  ): Promise<Array<Clue & { score: number; relevanceBreakdown?: any }>>;

  // Metadata - now per-repo
  abstract getLastProcessedPR(repo: string): Promise<number>;
  abstract setLastProcessedPR(repo: string, number: number): Promise<void>;
  abstract getLastProcessedCommit(repo: string): Promise<string | null>;
  abstract setLastProcessedCommit(repo: string, sha: string): Promise<void>;
  abstract getChronologicalCheckpoint(repo: string): Promise<ChronologicalCheckpoint | null>;
  abstract setChronologicalCheckpoint(repo: string, checkpoint: ChronologicalCheckpoint): Promise<void>;
  abstract getClueAnalysisCheckpoint(repo: string): Promise<ChronologicalCheckpoint | null>;
  abstract setClueAnalysisCheckpoint(repo: string, checkpoint: ChronologicalCheckpoint): Promise<void>;

  // Themes - per-repo
  abstract addThemes(repo: string, themes: string[]): Promise<void>;
  abstract getRecentThemes(repo: string): Promise<string[]>;

  // ... rest unchanged
}
```

### Step 7: Update Builder (`builder.ts`)

Update to pass repo to all save methods:

```typescript
export class StreamingFeatureBuilder {
  private repo: string; // Store repo identifier

  constructor(
    private storage: Storage,
    private llm: LLMClient,
    private octokit: Octokit,
    private repoPath?: string,
    private shouldAnalyzeClues: boolean = false
  ) {}

  async processRepo(owner: string, repo: string): Promise<{ usage: Usage; modifiedFeatureIds: Set<string> }> {
    // Set repo identifier
    this.repo = `${owner}/${repo}`;
    
    // Get checkpoint for THIS repo
    let checkpoint = await this.storage.getChronologicalCheckpoint(this.repo);
    
    // ... rest of processing
  }

  private async applyPrDecision(
    owner: string,
    repo: string,
    pr: GitHubPR,
    decision: LLMDecision,
    modifiedFeatureIds: Set<string>
  ): Promise<void> {
    // ... existing code ...

    // Save PR record with repo
    const prRecord: PRRecord = {
      number: pr.number,
      repo: this.repo, // Add repo!
      title: pr.title,
      summary: decision.summary,
      mergedAt: pr.mergedAt,
      url: pr.url,
      files: files.map((f) => f.filename),
      newDeclarations: decision.newDeclarations,
    };
    await this.storage.savePR(prRecord);

    // ... rest unchanged
  }

  private generateFeatureId(name: string): string {
    const slug = name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
    
    // Return repo-prefixed ID
    return `${this.repo}/${slug}`;
  }

  // Update all feature creation to include repo
  private async applyDecisionToFeatures(...) {
    // When creating new feature:
    const newFeature = config.createFeatureWith({
      ...baseFeature,
      repo: this.repo, // Add repo!
    });
  }
}
```

### Step 8: Update Routes (`routes.ts`)

Add repo parameter parsing to all routes:

```typescript
/**
 * Parse and normalize repo parameter from query string
 * Accepts: owner/repo, https://github.com/owner/repo, etc.
 */
function parseRepoParam(req: Request): string | undefined {
  const repoParam = req.query.repo as string | undefined;
  if (!repoParam) return undefined;
  
  // Use existing parseGitRepoUrl function
  const parsed = parseGitRepoUrl(repoParam);
  if (parsed) {
    return `${parsed.owner}/${parsed.repo}`;
  }
  
  // If it looks like owner/repo already, use as-is
  if (repoParam.match(/^[^\/]+\/[^\/]+$/)) {
    return repoParam;
  }
  
  return undefined;
}

/**
 * List all features
 * GET /gitree/features?repo=owner/repo (optional)
 */
export async function gitree_list_features(req: Request, res: Response) {
  try {
    const repo = parseRepoParam(req);
    const storage = new GraphStorage();
    await storage.initialize();

    const features = await storage.getAllFeatures(repo);
    const checkpoint = repo 
      ? await storage.getChronologicalCheckpoint(repo)
      : null;

    res.json({
      features: features.map((f) => ({
        id: f.id,
        repo: f.repo,
        ref_id: f.ref_id,
        name: f.name,
        description: f.description,
        prCount: f.prNumbers.length,
        commitCount: (f.commitShas || []).length,
        lastUpdated: f.lastUpdated.toISOString(),
        hasDocumentation: !!f.documentation,
      })),
      total: features.length,
      repo: repo || "all",
      lastProcessedTimestamp: checkpoint?.lastProcessedTimestamp || null,
      processing: isProcessing,
    });
  } catch (error: any) {
    console.error("Error listing features:", error);
    res.status(500).json({ error: error.message || "Failed to list features" });
  }
}

// Apply same pattern to all other routes:
// - gitree_get_feature
// - gitree_get_pr
// - gitree_get_commit
// - gitree_list_clues
// - gitree_get_clue
// - gitree_relevant_features
// - gitree_stats
// - gitree_all_features_graph
// - etc.
```

### Step 9: Update FeaturesMetadata to Per-Repo (`store/graphStorage.ts`)

Change metadata methods to use repo as an additional key (namespace stays "default"):

```typescript
async getChronologicalCheckpoint(repo: string): Promise<ChronologicalCheckpoint | null> {
  const session = this.driver.session();
  try {
    const result = await session.run(
      `
      MATCH (m:${Data_Bank}:FeaturesMetadata {namespace: $namespace, repo: $repo})
      RETURN m.chronologicalCheckpoint as checkpoint
      `,
      { namespace: "default", repo }
    );

    if (result.records.length === 0) {
      return null;
    }

    const checkpointStr = result.records[0].get("checkpoint");
    if (!checkpointStr) {
      return null;
    }

    return JSON.parse(checkpointStr);
  } catch (error) {
    console.error("Error reading chronological checkpoint:", error);
    return null;
  } finally {
    await session.close();
  }
}

async setChronologicalCheckpoint(repo: string, checkpoint: ChronologicalCheckpoint): Promise<void> {
  const session = this.driver.session();
  try {
    const now = Math.floor(Date.now() / 1000);

    await session.run(
      `
      MERGE (m:${Data_Bank}:FeaturesMetadata {namespace: $namespace, repo: $repo})
      SET m.chronologicalCheckpoint = $checkpoint,
          m.ref_id = COALESCE(m.ref_id, $refId),
          m.date_added_to_graph = COALESCE(m.date_added_to_graph, $dateAddedToGraph)
      RETURN m
      `,
      {
        namespace: "default",
        repo,
        checkpoint: JSON.stringify(checkpoint),
        refId: uuidv4(),
        dateAddedToGraph: now,
      }
    );
  } finally {
    await session.close();
  }
}

// Apply same pattern to:
// - getLastProcessedPR / setLastProcessedPR
// - getLastProcessedCommit / setLastProcessedCommit
// - getClueAnalysisCheckpoint / setClueAnalysisCheckpoint
// - addThemes / getRecentThemes
```

**IMPORTANT**: `namespace` always stays `"default"`. The `repo` field is added as a separate key for per-repo isolation.

---

## File Changes Summary

| File | Changes |
|------|---------|
| `types.ts` | Add `repo?: string` to Feature, PRRecord, CommitRecord, Clue |
| `store/utils.ts` | Add `parseRepoFromUrl()`, `makeRepoId()`, `getSlugFromRepoId()`, `getRepoFromRepoId()` |
| `store/storage.ts` | Update abstract method signatures with optional `repo` param |
| `store/graphStorage.ts` | Add migration, indexes, update saves/queries, per-repo metadata |
| `builder.ts` | Pass `repo` to all save methods, update ID generation |
| `routes.ts` | Add `parseRepoParam()`, add optional `?repo=` to all routes |
| `summarizer.ts` | Pass `repo` when calling storage methods |
| `fileLinker.ts` | Pass `repo` when calling storage methods |
| `clueAnalyzer.ts` | Pass `repo` when calling storage methods |
| `clueLinker.ts` | Pass `repo` when calling storage methods |

---

## Testing Checklist

- [ ] Migration runs successfully on existing data
- [ ] Migration is idempotent (safe to run multiple times)
- [ ] New repos are processed with correct repo-prefixed IDs
- [ ] Queries without `?repo=` return all data
- [ ] Queries with `?repo=` filter correctly
- [ ] `?repo=` accepts full URLs and normalizes them
- [ ] Checkpoints are isolated per-repo
- [ ] Processing repo A doesn't affect repo B's checkpoint
- [ ] Feature-PR-Commit relationships work correctly within a repo
- [ ] Cross-repo features don't collide (same name, different repos)

---

## Rollback Plan

If migration fails or causes issues:

1. The migration preserves old IDs in `legacyId` fields
2. To rollback, run:
```cypher
// Restore Feature IDs
MATCH (f:Feature) WHERE f.legacyId IS NOT NULL
SET f.id = f.legacyId
REMOVE f.repo, f.legacyId

// Restore PR names
MATCH (p:PullRequest) WHERE p.legacyId IS NOT NULL
SET p.name = p.legacyId
REMOVE p.repo, p.legacyId

// Restore Commit names
MATCH (c:Commit) WHERE c.legacyId IS NOT NULL
SET c.name = c.legacyId
REMOVE c.repo, c.legacyId

// Restore Clue IDs
MATCH (c:Clue) WHERE c.legacyId IS NOT NULL
SET c.id = c.legacyId
REMOVE c.repo, c.legacyId

// Restore Metadata
MATCH (m:FeaturesMetadata) WHERE m.repo IS NOT NULL
REMOVE m.repo
```
