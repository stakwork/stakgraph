import neo4j, { Driver, Session } from "neo4j-driver";
import { createNeo4jDriver, ResilientSession } from "../../../utils/neo4jRetry.js";
import { v4 as uuidv4 } from "uuid";
import { Storage } from "./storage.js";
import {
  Concept,
  PRRecord,
  CommitRecord,
  LinkResult,
  ChronologicalCheckpoint,
  Usage,
} from "../types.js";
import { formatPRMarkdown, formatCommitMarkdown, parseRepoFromUrl } from "./utils.js";
import { addUsage, normalizeUsage } from "../../../aieo/src/usage.js";

const Data_Bank = "Data_Bank";

function numberOrUndefined(value: any): number | undefined {
  if (value == null) return undefined;
  return value?.toNumber ? value.toNumber() : value;
}

function usageParams(usage?: Usage) {
  const normalized = usage ? normalizeUsage(usage) : undefined;
  return {
    input: normalized?.input ?? null,
    cacheRead: normalized?.cache_read ?? null,
    cacheWrite: normalized?.cache_write ?? null,
    inputTokens: normalized?.inputTokens ?? null,
    outputTokens: normalized?.outputTokens ?? null,
    totalTokens: normalized?.totalTokens ?? null,
  };
}

function usageFromProps(props: any): Usage | undefined {
  const input = numberOrUndefined(props.inputNoCacheTokens);
  const cacheRead = numberOrUndefined(props.cacheReadTokens);
  const cacheWrite = numberOrUndefined(props.cacheWriteTokens);
  const inputTokens = numberOrUndefined(props.inputTokens);
  const outputTokens = numberOrUndefined(props.outputTokens);
  const totalTokens = numberOrUndefined(props.totalTokens);

  if (
    input === undefined &&
    cacheRead === undefined &&
    cacheWrite === undefined &&
    inputTokens === undefined &&
    outputTokens === undefined &&
    totalTokens === undefined
  ) {
    return undefined;
  }

  return normalizeUsage({
    input,
    cache_read: cacheRead,
    cache_write: cacheWrite,
    inputTokens,
    outputTokens,
    totalTokens,
  });
}

/**
 * Neo4j graph-based storage implementation for features and PRs
 */
export class GraphStorage extends Storage {
  private driver: Driver;

  constructor() {
    super();
    this.driver = createNeo4jDriver();
    const host = process.env.NEO4J_HOST || "localhost:7687";
    const user = process.env.NEO4J_USER || "neo4j";
    console.log("===> GraphStorage connecting to", `bolt://${host}`, user);
  }

  private resilientSession(): ResilientSession {
    return new ResilientSession(() => this.driver, (d) => { this.driver = d; });
  }

  /**
   * Initialize indexes for better query performance
   */
  async initialize(): Promise<void> {
    const session = this.resilientSession();
    try {
        // Create indexes on id/number/sha for fast lookups
        await session.run(
          "CREATE INDEX concept_id_index IF NOT EXISTS FOR (f:Concept) ON (f.id)"
        );
        await session.run(
          "CREATE INDEX pr_number_index IF NOT EXISTS FOR (p:PullRequest) ON (p.number)"
        );
        await session.run(
          "CREATE INDEX commit_sha_index IF NOT EXISTS FOR (c:Commit) ON (c.sha)"
        );
        // Indexes for code nodes (for REFERENCES edges)
        await session.run(
          "CREATE INDEX function_name_index IF NOT EXISTS FOR (f:Function) ON (f.name)"
        );
        await session.run(
          "CREATE INDEX class_name_index IF NOT EXISTS FOR (c:Class) ON (c.name)"
        );
        await session.run(
          "CREATE INDEX endpoint_name_index IF NOT EXISTS FOR (e:Endpoint) ON (e.name)"
        );
        await session.run(
          "CREATE INDEX datamodel_name_index IF NOT EXISTS FOR (d:Datamodel) ON (d.name)"
        );
        await session.run(
          "CREATE INDEX var_name_index IF NOT EXISTS FOR (v:Var) ON (v.name)"
        );

        // Multi-repo indexes
        await session.run(
          "CREATE INDEX concept_repo_index IF NOT EXISTS FOR (f:Concept) ON (f.repo)"
        );
        await session.run(
          "CREATE INDEX pr_repo_index IF NOT EXISTS FOR (p:PullRequest) ON (p.repo)"
        );
        await session.run(
          "CREATE INDEX commit_repo_index IF NOT EXISTS FOR (c:Commit) ON (c.repo)"
        );
        await session.run(
          "CREATE INDEX pr_id_index IF NOT EXISTS FOR (p:PullRequest) ON (p.id)"
        );
        await session.run(
          "CREATE INDEX commit_id_index IF NOT EXISTS FOR (c:Commit) ON (c.id)"
        );
    } catch (error) {
      console.error("Error creating GraphStorage indexes:", error);
    } finally {
      await session.close();
    }
  }

  /**
   * Close the Neo4j driver connection
   */
  async close(): Promise<void> {
    await this.driver.close();
  }

  // Concepts

  async saveConcept(feature: Concept): Promise<void> {
    const session = this.resilientSession();
    try {
      const now = Math.floor(Date.now() / 1000);
      const dateTimestamp = Math.floor(feature.lastUpdated.getTime() / 1000);

      await session.run(
        `
        MERGE (f:${Data_Bank}:Concept {id: $id})
        SET f.name = $name,
            f.repo = $repo,
            f.description = $description,
            f.prNumbers = $prNumbers,
            f.commitShas = $commitShas,
            f.date = $date,
            f.docs = $docs,
            f.inputNoCacheTokens = $input,
            f.cacheReadTokens = $cacheRead,
            f.cacheWriteTokens = $cacheWrite,
            f.inputTokens = $inputTokens,
            f.outputTokens = $outputTokens,
            f.totalTokens = $totalTokens,
            f.namespace = $namespace,
            f.Data_Bank = $dataBankName,
            f.ref_id = COALESCE(f.ref_id, $refId),
            f.date_added_to_graph = COALESCE(f.date_added_to_graph, $dateAddedToGraph)
        RETURN f
        `,
        {
          id: feature.id, // Should be repo-prefixed: "owner/repo/slug"
          repo: feature.repo || null,
          name: feature.name,
          description: feature.description,
          prNumbers: feature.prNumbers,
          commitShas: feature.commitShas,
          date: dateTimestamp,
          docs: feature.documentation || "",
          ...usageParams(feature.usage),
          namespace: "default",
          dataBankName: feature.id,
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );

      // Create TOUCHES relationships from PRs to this Concept
      // Match PRs by repo-prefixed id or by number+repo combo
      if (feature.prNumbers.length > 0 && feature.repo) {
        await session.run(
          `
          MATCH (f:Concept {id: $featureId})
          UNWIND $prNumbers as prNumber
          MATCH (p:PullRequest)
          WHERE (p.repo = $repo AND p.number = prNumber) OR p.id = $repo + '/pr-' + toString(prNumber)
          MERGE (p)-[:TOUCHES]->(f)
          `,
          {
            featureId: feature.id,
            prNumbers: feature.prNumbers,
            repo: feature.repo,
          }
        );
      } else if (feature.prNumbers.length > 0) {
        // Legacy: no repo, match by number only
        await session.run(
          `
          MATCH (f:Concept {id: $featureId})
          UNWIND $prNumbers as prNumber
          MATCH (p:PullRequest {number: prNumber})
          MERGE (p)-[:TOUCHES]->(f)
          `,
          {
            featureId: feature.id,
            prNumbers: feature.prNumbers,
          }
        );
      }

      // Create TOUCHES relationships from Commits to this Concept
      const commitShas = feature.commitShas || [];
      if (commitShas.length > 0 && feature.repo) {
        await session.run(
          `
          MATCH (f:Concept {id: $featureId})
          UNWIND $commitShas as commitSha
          MATCH (c:Commit)
          WHERE (c.repo = $repo AND c.sha = commitSha) OR c.sha = commitSha
          MERGE (c)-[:TOUCHES]->(f)
          `,
          {
            featureId: feature.id,
            commitShas: commitShas,
            repo: feature.repo,
          }
        );
      } else if (commitShas.length > 0) {
        // Legacy: no repo, match by sha only
        await session.run(
          `
          MATCH (f:Concept {id: $featureId})
          UNWIND $commitShas as commitSha
          MATCH (c:Commit {sha: commitSha})
          MERGE (c)-[:TOUCHES]->(f)
          `,
          {
            featureId: feature.id,
            commitShas: commitShas,
          }
        );
      }
    } finally {
      await session.close();
    }
  }

  async getConcept(id: string, repo?: string): Promise<Concept | null> {
    const session = this.resilientSession();
    try {
      // If repo provided and id doesn't have prefix, construct full id
      const fullId = repo && !id.includes('/') ? `${repo}/${id}` : id;
      
      const result = await session.run(
        `
        MATCH (f:Concept {id: $id})
        RETURN f
        `,
        { id: fullId }
      );

      if (result.records.length === 0) {
        return null;
      }

      const node = result.records[0].get("f");
      return this.nodeToConcept(node);
    } finally {
      await session.close();
    }
  }

  async getAllConcepts(repo?: string): Promise<Concept[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(
        `
        MATCH (f:Concept)
        WHERE $repo IS NULL OR f.repo = $repo
        RETURN f
        ORDER BY f.date DESC
        `,
        { repo: repo || null }
      );

      return result.records.map((record) =>
        this.nodeToConcept(record.get("f"))
      );
    } finally {
      await session.close();
    }
  }

  async deleteConcept(id: string, repo?: string): Promise<void> {
    const session = this.resilientSession();
    try {
      // If repo provided and id doesn't have prefix, construct full id
      const fullId = repo && !id.includes('/') ? `${repo}/${id}` : id;
      
      await session.run(
        `
        MATCH (f:Concept {id: $id})
        DETACH DELETE f
        `,
        { id: fullId }
      );
    } finally {
      await session.close();
    }
  }

  // PRs

  async savePR(pr: PRRecord): Promise<void> {
    const session = this.resilientSession();
    try {
      const now = Math.floor(Date.now() / 1000);
      const dateTimestamp = Math.floor(pr.mergedAt.getTime() / 1000);
      const docs = await formatPRMarkdown(pr, this);
      
      // Generate repo-prefixed ID for multi-repo support
      const prId = pr.repo ? `${pr.repo}/pr-${pr.number}` : `pr-${pr.number}`;

      await session.run(
        `
        MERGE (p:${Data_Bank}:PullRequest {id: $id})
        SET p.number = $number,
            p.repo = $repo,
            p.name = $name,
            p.title = $title,
            p.summary = $summary,
            p.date = $date,
            p.url = $url,
            p.files = $files,
            p.newDeclarations = $newDeclarations,
            p.docs = $docs,
            p.namespace = $namespace,
            p.Data_Bank = $dataBankName,
            p.inputNoCacheTokens = $input,
            p.cacheReadTokens = $cacheRead,
            p.cacheWriteTokens = $cacheWrite,
            p.inputTokens = $inputTokens,
            p.outputTokens = $outputTokens,
            p.totalTokens = $totalTokens,
            p.ref_id = COALESCE(p.ref_id, $refId),
            p.date_added_to_graph = COALESCE(p.date_added_to_graph, $dateAddedToGraph)
        RETURN p
        `,
        {
          id: prId,
          number: pr.number,
          repo: pr.repo || null,
          name: prId,
          title: pr.title,
          summary: pr.summary,
          date: dateTimestamp,
          url: pr.url,
          files: pr.files,
          newDeclarations: pr.newDeclarations
            ? JSON.stringify(pr.newDeclarations)
            : null,
          docs,
          namespace: "default",
          dataBankName: `pull-request-${pr.number}`,
          ...usageParams(pr.usage),
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );

      // Note: TOUCHES relationships are created in saveConcept()
      // when features are updated with PR numbers
    } finally {
      await session.close();
    }
  }

  async getPR(number: number, repo?: string): Promise<PRRecord | null> {
    const session = this.resilientSession();
    try {
      let query: string;
      let params: Record<string, any>;
      
      if (repo) {
        // Look up by repo-prefixed ID
        const prId = `${repo}/pr-${number}`;
        query = `MATCH (p:PullRequest {id: $id}) RETURN p`;
        params = { id: prId };
      } else {
        // Look up by number (may return first found from any repo)
        query = `MATCH (p:PullRequest {number: $number}) RETURN p LIMIT 1`;
        params = { number };
      }
      
      const result = await session.run(query, params);

      if (result.records.length === 0) {
        return null;
      }

      const node = result.records[0].get("p");
      return this.nodeToPR(node);
    } finally {
      await session.close();
    }
  }

  async getAllPRs(repo?: string): Promise<PRRecord[]> {
    const session = this.resilientSession();
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

  // Commits

  async saveCommit(commit: CommitRecord): Promise<void> {
    const session = this.resilientSession();
    try {
      const now = Math.floor(Date.now() / 1000);
      const dateTimestamp = Math.floor(commit.committedAt.getTime() / 1000);
      const docs = await formatCommitMarkdown(commit, this);
      
      // Generate repo-prefixed ID for multi-repo support
      const commitId = commit.repo 
        ? `${commit.repo}/commit-${commit.sha.substring(0, 7)}` 
        : `commit-${commit.sha.substring(0, 7)}`;

      await session.run(
        `
        MERGE (c:${Data_Bank}:Commit {id: $id})
        SET c.sha = $sha,
            c.repo = $repo,
            c.name = $name,
            c.message = $message,
            c.summary = $summary,
            c.author = $author,
            c.date = $date,
            c.url = $url,
            c.files = $files,
            c.newDeclarations = $newDeclarations,
            c.docs = $docs,
            c.namespace = $namespace,
            c.Data_Bank = $dataBankName,
            c.inputNoCacheTokens = $input,
            c.cacheReadTokens = $cacheRead,
            c.cacheWriteTokens = $cacheWrite,
            c.inputTokens = $inputTokens,
            c.outputTokens = $outputTokens,
            c.totalTokens = $totalTokens,
            c.ref_id = COALESCE(c.ref_id, $refId),
            c.date_added_to_graph = COALESCE(c.date_added_to_graph, $dateAddedToGraph)
        RETURN c
        `,
        {
          id: commitId,
          sha: commit.sha,
          repo: commit.repo || null,
          name: commitId,
          message: commit.message,
          summary: commit.summary,
          author: commit.author,
          date: dateTimestamp,
          url: commit.url,
          files: commit.files,
          newDeclarations: commit.newDeclarations
            ? JSON.stringify(commit.newDeclarations)
            : null,
          docs,
          namespace: "default",
          dataBankName: `commit-${commit.sha.substring(0, 7)}`,
          ...usageParams(commit.usage),
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );

      // Note: TOUCHES relationships are created in saveConcept()
      // when features are updated with commit SHAs
    } finally {
      await session.close();
    }
  }

  async getCommit(sha: string, repo?: string): Promise<CommitRecord | null> {
    const session = this.resilientSession();
    try {
      let query: string;
      let params: Record<string, any>;
      
      if (repo) {
        // Look up by repo + sha (more specific)
        query = `MATCH (c:Commit) WHERE c.sha = $sha AND c.repo = $repo RETURN c`;
        params = { sha, repo };
      } else {
        // Look up by sha only (globally unique anyway)
        query = `MATCH (c:Commit {sha: $sha}) RETURN c LIMIT 1`;
        params = { sha };
      }
      
      const result = await session.run(query, params);

      if (result.records.length === 0) {
        return null;
      }

      const node = result.records[0].get("c");
      return this.nodeToCommit(node);
    } finally {
      await session.close();
    }
  }

  async getAllCommits(repo?: string): Promise<CommitRecord[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(
        `
        MATCH (c:Commit)
        WHERE $repo IS NULL OR c.repo = $repo
        RETURN c
        ORDER BY c.date ASC
        `,
        { repo: repo || null }
      );

      return result.records.map((record) => this.nodeToCommit(record.get("c")));
    } finally {
      await session.close();
    }
  }

  // Metadata - now per-repo

  async getLastProcessedPR(repo: string): Promise<number> {
    const session = this.resilientSession();
    try {
      const result = await session.run(
        `
        MATCH (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
        RETURN m.lastProcessedPR as lastProcessedPR
        `,
        { namespace: "default", repo }
      );

      if (result.records.length === 0) {
        console.log(`   No lastProcessedPR found for ${repo}, starting from 0`);
        return 0;
      }

      const value = result.records[0].get("lastProcessedPR");
      const lastPR =
        typeof value === "number"
          ? value
          : value?.toNumber
          ? value.toNumber()
          : 0;
      console.log(`   Resuming from PR #${lastPR} for ${repo}`);
      return lastPR;
    } catch (error) {
      console.error("   Error reading lastProcessedPR:", error);
      return 0;
    } finally {
      await session.close();
    }
  }

  async setLastProcessedPR(repo: string, number: number): Promise<void> {
    const session = this.resilientSession();
    try {
      const now = Math.floor(Date.now() / 1000);

      await session.run(
        `
        MERGE (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
        SET m.lastProcessedPR = $number,
            m.ref_id = COALESCE(m.ref_id, $refId),
            m.date_added_to_graph = COALESCE(m.date_added_to_graph, $dateAddedToGraph)
        RETURN m
        `,
        {
          namespace: "default",
          repo,
          number,
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );
    } finally {
      await session.close();
    }
  }

  async getLastProcessedCommit(repo: string): Promise<string | null> {
    const session = this.resilientSession();
    try {
      const result = await session.run(
        `
        MATCH (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
        RETURN m.lastProcessedCommit as lastProcessedCommit
        `,
        { namespace: "default", repo }
      );

      if (result.records.length === 0) {
        console.log(`   No lastProcessedCommit found for ${repo}, starting from beginning`);
        return null;
      }

      const value = result.records[0].get("lastProcessedCommit");
      if (value) {
        console.log(`   Resuming from commit ${value.substring(0, 7)} for ${repo}`);
      }
      return value || null;
    } catch (error) {
      console.error("   Error reading lastProcessedCommit:", error);
      return null;
    } finally {
      await session.close();
    }
  }

  async setLastProcessedCommit(repo: string, sha: string): Promise<void> {
    const session = this.resilientSession();
    try {
      const now = Math.floor(Date.now() / 1000);

      await session.run(
        `
        MERGE (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
        SET m.lastProcessedCommit = $sha,
            m.ref_id = COALESCE(m.ref_id, $refId),
            m.date_added_to_graph = COALESCE(m.date_added_to_graph, $dateAddedToGraph)
        RETURN m
        `,
        {
          namespace: "default",
          repo,
          sha,
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );
    } finally {
      await session.close();
    }
  }

  async getChronologicalCheckpoint(repo: string): Promise<ChronologicalCheckpoint | null> {
    const session = this.resilientSession();
    try {
      const result = await session.run(
        `
        MATCH (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
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

  async setChronologicalCheckpoint(
    repo: string,
    checkpoint: ChronologicalCheckpoint
  ): Promise<void> {
    const session = this.resilientSession();
    try {
      const now = Math.floor(Date.now() / 1000);

      await session.run(
        `
        MERGE (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
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

  // Themes - now per-repo

  async addThemes(repo: string, themes: string[]): Promise<void> {
    const session = this.resilientSession();
    try {
      const now = Math.floor(Date.now() / 1000);

      // Get current themes
      const result = await session.run(
        `
        MATCH (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
        RETURN m.recentThemes as recentThemes
        `,
        { namespace: "default", repo }
      );

      let recentThemes: string[] = [];
      if (result.records.length > 0) {
        recentThemes = result.records[0].get("recentThemes") || [];
      }

      // Remove themes if they already exist (LRU behavior)
      recentThemes = recentThemes.filter((t: string) => !themes.includes(t));

      // Add to end (most recent)
      recentThemes.push(...themes);

      // Keep only last 100
      if (recentThemes.length > 100) {
        recentThemes = recentThemes.slice(-100);
      }

      // Update metadata node
      await session.run(
        `
        MERGE (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
        SET m.recentThemes = $recentThemes,
            m.ref_id = COALESCE(m.ref_id, $refId),
            m.date_added_to_graph = COALESCE(m.date_added_to_graph, $dateAddedToGraph)
        RETURN m
        `,
        {
          namespace: "default",
          repo,
          recentThemes,
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );
    } finally {
      await session.close();
    }
  }

  async getRecentThemes(repo: string): Promise<string[]> {
    const session = this.resilientSession();
    try {
      const result = await session.run(
        `
        MATCH (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
        RETURN m.recentThemes as recentThemes
        `,
        { namespace: "default", repo }
      );

      if (result.records.length === 0) {
        return [];
      }

      return result.records[0].get("recentThemes") || [];
    } catch (error) {
      console.error("   Error reading recentThemes:", error);
      return [];
    } finally {
      await session.close();
    }
  }

  // Total Usage - cumulative token usage across all processing runs

  async getTotalUsage(repo: string): Promise<Usage> {
    const session = this.resilientSession();
    try {
      const result = await session.run(
        `
        MATCH (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
         RETURN m.totalInputNoCacheTokens as input,
           m.totalCacheReadTokens as cacheRead,
           m.totalCacheWriteTokens as cacheWrite,
           m.totalInputTokens as inputTokens,
               m.totalOutputTokens as outputTokens,
               m.totalTokens as totalTokens
        `,
        { namespace: "default", repo }
      );

      if (result.records.length === 0) {
        return normalizeUsage();
      }

      const record = result.records[0];
      const inputNoCacheRaw = record.get("input");
      const cacheReadRaw = record.get("cacheRead");
      const cacheWriteRaw = record.get("cacheWrite");
      const inputRaw = record.get("inputTokens");
      const outputRaw = record.get("outputTokens");
      const totalRaw = record.get("totalTokens");

      return normalizeUsage({
        input: numberOrUndefined(inputNoCacheRaw),
        cache_read: numberOrUndefined(cacheReadRaw),
        cache_write: numberOrUndefined(cacheWriteRaw),
        inputTokens: numberOrUndefined(inputRaw),
        outputTokens: numberOrUndefined(outputRaw),
        totalTokens: numberOrUndefined(totalRaw),
      });
    } catch (error) {
      console.error("   Error reading totalUsage:", error);
      return normalizeUsage();
    } finally {
      await session.close();
    }
  }

  async addToTotalUsage(repo: string, usage: Usage): Promise<void> {
    const session = this.resilientSession();
    try {
      const now = Math.floor(Date.now() / 1000);

      await session.run(
        `
        MERGE (m:${Data_Bank}:ConceptsMetadata {namespace: $namespace, repo: $repo})
        SET m.totalInputNoCacheTokens = COALESCE(m.totalInputNoCacheTokens, COALESCE(m.totalInputTokens, 0)) + $input,
          m.totalCacheReadTokens = COALESCE(m.totalCacheReadTokens, 0) + $cacheRead,
          m.totalCacheWriteTokens = COALESCE(m.totalCacheWriteTokens, 0) + $cacheWrite,
          m.totalInputTokens = COALESCE(m.totalInputTokens, 0) + $inputTokens,
            m.totalOutputTokens = COALESCE(m.totalOutputTokens, 0) + $outputTokens,
            m.totalTokens = COALESCE(m.totalTokens, 0) + $totalTokens,
            m.ref_id = COALESCE(m.ref_id, $refId),
            m.date_added_to_graph = COALESCE(m.date_added_to_graph, $dateAddedToGraph)
        RETURN m
        `,
        {
          namespace: "default",
          repo,
          input: usage.input,
          cacheRead: usage.cache_read,
          cacheWrite: usage.cache_write,
          inputTokens: usage.inputTokens,
          outputTokens: usage.outputTokens,
          totalTokens: usage.totalTokens,
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );
    } finally {
      await session.close();
    }
  }

  async getAggregatedMetadata(): Promise<{
    lastProcessedTimestamp: string | null;
    cumulativeUsage: Usage;
  }> {
    const session = this.resilientSession();
    try {
      const result = await session.run(
        `MATCH (m:${Data_Bank}:ConceptsMetadata)
         RETURN m.chronologicalCheckpoint as checkpoint,
              m.totalInputNoCacheTokens as input,
              m.totalCacheReadTokens as cacheRead,
              m.totalCacheWriteTokens as cacheWrite,
                m.totalInputTokens as inputTokens,
                m.totalOutputTokens as outputTokens,
                m.totalTokens as totalTokens`
      );

      let latestTimestamp: string | null = null;
    let cumulativeUsage = normalizeUsage();

      for (const record of result.records) {
        // Parse checkpoint to get timestamp
        const checkpointStr = record.get("checkpoint");
        if (checkpointStr) {
          try {
            const checkpoint = JSON.parse(checkpointStr);
            if (checkpoint.lastProcessedTimestamp) {
              if (!latestTimestamp || checkpoint.lastProcessedTimestamp > latestTimestamp) {
                latestTimestamp = checkpoint.lastProcessedTimestamp;
              }
            }
          } catch {}
        }

        // Sum usage
        const inputNoCacheRaw = record.get("input");
        const cacheReadRaw = record.get("cacheRead");
        const cacheWriteRaw = record.get("cacheWrite");
        const inputRaw = record.get("inputTokens");
        const outputRaw = record.get("outputTokens");
        const tokensRaw = record.get("totalTokens");

        cumulativeUsage = normalizeUsage(addUsage(cumulativeUsage, normalizeUsage({
          input: numberOrUndefined(inputNoCacheRaw),
          cache_read: numberOrUndefined(cacheReadRaw),
          cache_write: numberOrUndefined(cacheWriteRaw),
          inputTokens: numberOrUndefined(inputRaw),
          outputTokens: numberOrUndefined(outputRaw),
          totalTokens: numberOrUndefined(tokensRaw),
        })));
      }

      return {
        lastProcessedTimestamp: latestTimestamp,
        cumulativeUsage,
      };
    } finally {
      await session.close();
    }
  }

  // Documentation

  async saveDocumentation(
    featureId: string,
    documentation: string
  ): Promise<void> {
    const session = this.resilientSession();
    try {
      await session.run(
        `
        MATCH (f:Concept {id: $id})
        SET f.docs = $docs
        RETURN f
        `,
        {
          id: featureId,
          docs: documentation,
        }
      );
    } finally {
      await session.close();
    }
  }

  // Helper methods

  private nodeToConcept(node: any): Concept {
    const props = node.properties;
    
    return {
      id: props.id,
      repo: props.repo || undefined,
      ref_id: props.ref_id,
      name: props.name,
      description: props.description,
      prNumbers: props.prNumbers || [],
      commitShas: props.commitShas || [],
      createdAt: new Date(props.date * 1000),
      lastUpdated: new Date(props.date * 1000),
      documentation: props.docs || undefined,
      usage: usageFromProps(props),
    };
  }

  private nodeToPR(node: any): PRRecord {
    const props = node.properties;
    
    return {
      number: props.number.toNumber ? props.number.toNumber() : props.number,
      repo: props.repo || undefined,
      title: props.title,
      summary: props.summary,
      mergedAt: new Date(props.date * 1000),
      url: props.url,
      files: props.files || [],
      newDeclarations: props.newDeclarations
        ? JSON.parse(props.newDeclarations)
        : undefined,
      usage: usageFromProps(props),
    };
  }

  private nodeToCommit(node: any): CommitRecord {
    const props = node.properties;
    
    return {
      sha: props.sha,
      repo: props.repo || undefined,
      message: props.message,
      summary: props.summary,
      author: props.author,
      committedAt: new Date(props.date * 1000),
      url: props.url,
      files: props.files || [],
      newDeclarations: props.newDeclarations
        ? JSON.parse(props.newDeclarations)
        : undefined,
      usage: usageFromProps(props),
    };
  }

  // Concept-File Linking

  /**
   * Smart path matching to detect if a file path appears in documentation
   * Tries multiple variants: full path, filename only, path segments
   */
  private isFileInDocumentation(
    filePath: string,
    documentation: string
  ): boolean {
    if (!documentation) return false;

    const docLower = documentation.toLowerCase();

    // Extract filename
    const filename = filePath.split("/").pop() || "";

    // Try various path formats
    const pathsToCheck = [
      filePath, // Full path: "src/auth/auth.ts"
      filename, // Just filename: "auth.ts"
    ];

    // Add path segments (e.g., "auth/auth.ts" from "owner/repo/src/auth/auth.ts")
    const segments = filePath.split("/");
    for (let i = 1; i < segments.length; i++) {
      pathsToCheck.push(segments.slice(i).join("/"));
    }

    // Check if any variant exists in documentation
    return pathsToCheck.some((path) => docLower.includes(path.toLowerCase()));
  }

  async linkConceptsToFiles(featureId?: string, repo?: string): Promise<LinkResult> {
    const session = this.resilientSession();
    try {
      // Get features to process
      const features = featureId
        ? [await this.getConcept(featureId, repo)].filter(
            (f): f is Concept => f !== null
          )
        : await this.getAllConcepts(repo);

      if (features.length === 0) {
        return {
          featuresProcessed: 0,
          filesLinked: 0,
          filesInDocs: 0,
          filesNotInDocs: 0,
          featureFileLinks: [],
        };
      }

      const result: LinkResult = {
        featuresProcessed: features.length,
        filesLinked: 0,
        filesInDocs: 0,
        filesNotInDocs: 0,
        featureFileLinks: [],
      };

      // Process each feature
      for (const feature of features) {
        const documentation = feature.documentation || "";

        // Get all file paths from PRs with their frequency count
        const filePathsResult = await session.run(
          `
          MATCH (f:Concept {id: $featureId})

          // Collect files from PRs
          OPTIONAL MATCH (pr:PullRequest)-[:TOUCHES]->(f)
          WHERE pr.files IS NOT NULL
          WITH f, COLLECT(pr.files) as prFilesLists

          // Collect files from Commits
          OPTIONAL MATCH (c:Commit)-[:TOUCHES]->(f)
          WHERE c.files IS NOT NULL
          WITH f, prFilesLists, COLLECT(c.files) as commitFilesLists

          // Flatten both lists
          WITH f, prFilesLists + commitFilesLists as allFilesLists
          UNWIND allFilesLists as filesList
          UNWIND filesList as file

          // Count occurrences per file
          WITH file, COUNT(*) as changeCount
          RETURN file, changeCount
          `,
          { featureId: feature.id }
        );

        // Get total PR + commit count for this feature
        const totalChanges =
          feature.prNumbers.length + (feature.commitShas || []).length;

        if (filePathsResult.records.length === 0) {
          result.featureFileLinks.push({
            featureId: feature.id,
            filesLinked: 0,
            filesInDocs: 0,
            filesNotInDocs: 0,
          });
          continue;
        }

        // Process each file path with its PR count
        let linksCreatedForConcept = 0;
        let filesInDocsForConcept = 0;
        let filesNotInDocsForConcept = 0;

        for (const record of filePathsResult.records) {
          const filePath = record.get("file");
          const changeCountRaw = record.get("changeCount");
          const changeCount = changeCountRaw?.toNumber
            ? changeCountRaw.toNumber()
            : changeCountRaw || 0;

          // Calculate frequency score (0-1)
          const frequency = totalChanges > 0 ? changeCount / totalChanges : 0;

          // Check if file is in documentation
          const inDocs = this.isFileInDocumentation(filePath, documentation);

          // Calculate importance using two-tier system
          // Files in docs: 0.5-1.0, Files not in docs: 0.0-0.49
          const importance = inDocs ? 0.5 + frequency * 0.5 : frequency * 0.49;

          // Track statistics
          if (inDocs) {
            filesInDocsForConcept++;
          } else {
            filesNotInDocsForConcept++;
          }

          // Match File nodes where the file property ends with the PR file path
          // This handles cases like "owner/repo/src/me.ts" matching "src/me.ts"
          const linkResult = await session.run(
            `
            MATCH (f:Concept {id: $featureId})
            MATCH (file:File)
            WHERE file.file ENDS WITH $filePath
            MERGE (f)-[:MODIFIES {importance: $importance}]->(file)
            RETURN COUNT(file) as linkedCount
            `,
            {
              featureId: feature.id,
              filePath: filePath,
              importance: importance,
            }
          );

          const linkedCount = linkResult.records[0]?.get("linkedCount");
          const count = linkedCount?.toNumber
            ? linkedCount.toNumber()
            : linkedCount || 0;
          linksCreatedForConcept += count;
        }

        result.featureFileLinks.push({
          featureId: feature.id,
          filesLinked: linksCreatedForConcept,
          filesInDocs: filesInDocsForConcept,
          filesNotInDocs: filesNotInDocsForConcept,
        });
        result.filesLinked += linksCreatedForConcept;
        result.filesInDocs += filesInDocsForConcept;
        result.filesNotInDocs += filesNotInDocsForConcept;
      }

      return result;
    } finally {
      await session.close();
    }
  }

  /**
   * Link a feature to File nodes by explicit file paths.
   * Used by bootstrap to create MODIFIES edges from LLM-identified core files.
   * All files get importance 1.0 since they are directly identified as core files.
   * Returns the number of File nodes linked.
   */
  async linkConceptToFilesByPaths(featureId: string, filePaths: string[]): Promise<number> {
    if (filePaths.length === 0) return 0;

    const session = this.resilientSession();
    try {
      let totalLinked = 0;

      for (const filePath of filePaths) {
        const result = await session.run(
          `
          MATCH (f:Concept {id: $featureId})
          MATCH (file:File)
          WHERE file.file ENDS WITH $filePath
          MERGE (f)-[:MODIFIES {importance: 1.0}]->(file)
          RETURN COUNT(file) as linkedCount
          `,
          { featureId, filePath }
        );

        const linkedCount = result.records[0]?.get("linkedCount");
        totalLinked += linkedCount?.toNumber ? linkedCount.toNumber() : linkedCount || 0;
      }

      return totalLinked;
    } finally {
      await session.close();
    }
  }

  // Get Files for Concept
  // Supports expand options: CONTAINS, CALLS
  // Can be combined: expand=['CONTAINS', 'CALLS']

  async getFilesForConcept(
    featureId: string,
    expand?: string[]
  ): Promise<any[]> {
    const session = this.resilientSession();
    try {
      const shouldExpandContains = expand?.includes("CONTAINS") || false;
      const shouldExpandCalls = expand?.includes("CALLS") || false;

      let query = `
        MATCH (f:Concept {id: $featureId})-[r:MODIFIES]->(file:File)
      `;

      if (shouldExpandContains || shouldExpandCalls) {
        // Build optional matches based on what's being expanded
        // Use variable-length paths to capture nested structures
        if (shouldExpandContains) {
          query += `
        OPTIONAL MATCH (file)-[:CONTAINS*]->(contained)
          `;
        }
        if (shouldExpandCalls) {
          query += `
        OPTIONAL MATCH (file)-[:CONTAINS*]->(node)-[:CALLS]->(called)
          `;
        }

        // Collect the results
        query += `
        WITH file, r`;

        if (shouldExpandContains) {
          query += `,
             COLLECT(DISTINCT {
               name: contained.name,
               ref_id: contained.ref_id,
               node_type: [label IN labels(contained) WHERE label <> 'Data_Bank'][0]
             }) AS containedNodes`;
        }

        if (shouldExpandCalls) {
          query += `,
             COLLECT(DISTINCT {
               name: called.name,
               ref_id: called.ref_id,
               node_type: [label IN labels(called) WHERE label <> 'Data_Bank'][0]
             }) AS calledNodes`;
        }

        query += `
        RETURN file.name AS name,
               file.file AS file,
               file.ref_id AS ref_id,
               r.importance AS importance`;

        if (shouldExpandContains) {
          query += `,
               CASE WHEN SIZE(containedNodes) > 0 AND containedNodes[0].name IS NOT NULL
                 THEN containedNodes
                 ELSE []
               END AS contains`;
        }

        if (shouldExpandCalls) {
          query += `,
               CASE WHEN SIZE(calledNodes) > 0 AND calledNodes[0].name IS NOT NULL
                 THEN calledNodes
                 ELSE []
               END AS calls`;
        }

        query += `
        ORDER BY r.importance DESC
        `;
      } else {
        query += `
        RETURN file.name AS name,
               file.file AS file,
               file.ref_id AS ref_id,
               r.importance AS importance
        ORDER BY r.importance DESC
        `;
      }

      const result = await session.run(query, { featureId });

      return result.records.map((record) => {
        const fileData: any = {
          name: record.get("name"),
          file: record.get("file"),
          ref_id: record.get("ref_id"),
          importance: record.get("importance"),
        };

        if (shouldExpandContains) {
          fileData.contains = record.get("contains") || [];
        }

        if (shouldExpandCalls) {
          fileData.calls = record.get("calls") || [];
        }

        return fileData;
      });
    } finally {
      await session.close();
    }
  }

  /**
   * Get all features with their files and contained nodes
   * Returns structured data for building a flat graph
   */
  async getAllConceptsWithFilesAndContains(repo?: string): Promise<{
    features: any[];
    files: any[];
    containedNodes: any[];
    modifiesEdges: any[];
    containsEdges: any[];
  }> {
    const session = this.resilientSession();
    try {
      // Get all features (optionally filtered by repo)
      const featuresResult = await session.run(
        `MATCH (f:Concept) WHERE $repo IS NULL OR f.repo = $repo RETURN f ORDER BY f.date DESC`,
        { repo: repo || null }
      );
      const features = featuresResult.records.map((r) => r.get("f"));

      // Get all MODIFIES relationships and files
      const modifiesResult = await session.run(
        `
        MATCH (feature:Concept)-[m:MODIFIES]->(file:File)
        RETURN feature, m, file
        `
      );

      const filesMap = new Map();
      const modifiesEdges: any[] = [];

      for (const record of modifiesResult.records) {
        const featureNode = record.get("feature");
        const modifiesRel = record.get("m");
        const fileNode = record.get("file");

        // Add file
        if (fileNode && fileNode.properties && fileNode.properties.ref_id) {
          const fileRefId = fileNode.properties.ref_id;
          if (!filesMap.has(fileRefId)) {
            filesMap.set(fileRefId, fileNode);
          }
        }

        // Add MODIFIES edge
        if (featureNode && modifiesRel && fileNode) {
          modifiesEdges.push({
            edge_type: "MODIFIES",
            ref_id: modifiesRel.properties?.ref_id || "",
            source: featureNode.properties?.ref_id || "",
            target: fileNode.properties?.ref_id || "",
            properties: {
              importance: modifiesRel.properties?.importance,
            },
          });
        }
      }

      // Get file ref_ids to use in next query
      const fileRefIds = Array.from(filesMap.keys());

      // Get all contained nodes and CONTAINS edges starting from these files
      const containedNodesMap = new Map();
      const containsEdges: any[] = [];
      const processedEdges = new Set();

      if (fileRefIds.length > 0) {
        const containsResult = await session.run(
          `
          MATCH path = (file:File)-[:CONTAINS*]->(contained)
          WHERE file.ref_id IN $fileRefIds
          UNWIND relationships(path) AS rel
          WITH DISTINCT rel, startNode(rel) AS source, endNode(rel) AS target
          RETURN rel, source, target
          `,
          { fileRefIds }
        );

        for (const record of containsResult.records) {
          const rel = record.get("rel");
          const source = record.get("source");
          const target = record.get("target");

          // Add target node to containedNodesMap (skip Files)
          if (target && target.properties && target.properties.ref_id) {
            const isFile = target.labels && target.labels.includes("File");
            const refId = target.properties.ref_id;
            if (!isFile && !containedNodesMap.has(refId)) {
              containedNodesMap.set(refId, target);
            }
          }

          // Add source node if it's not a File (for nested CONTAINS)
          if (source && source.properties && source.properties.ref_id) {
            const isFile = source.labels && source.labels.includes("File");
            const refId = source.properties.ref_id;
            if (!isFile && !containedNodesMap.has(refId)) {
              containedNodesMap.set(refId, source);
            }
          }

          // Add CONTAINS edge
          if (
            rel &&
            source &&
            target &&
            source.properties &&
            target.properties
          ) {
            const sourceRefId = source.properties.ref_id || "";
            const targetRefId = target.properties.ref_id || "";
            const edgeKey = `${sourceRefId}:${targetRefId}`;

            if (!processedEdges.has(edgeKey) && sourceRefId && targetRefId) {
              processedEdges.add(edgeKey);
              containsEdges.push({
                edge_type: "CONTAINS",
                ref_id: rel.properties?.ref_id || "",
                source: sourceRefId,
                target: targetRefId,
                properties: rel.properties || {},
              });
            }
          }
        }
      }

      return {
        features,
        files: Array.from(filesMap.values()),
        containedNodes: Array.from(containedNodesMap.values()),
        modifiesEdges,
        containsEdges,
      };
    } finally {
      await session.close();
    }
  }

  /**
   * Get provenance data for multiple features by their IDs
   * Returns concepts with their files and filtered code entities
   * @param conceptIds - Array of feature IDs (can be repo-prefixed or not)
   * @param repo - Optional repo to filter/prefix IDs
   */
  async getProvenanceForConcepts(
    conceptIds: string[],
    repo?: string
  ): Promise<
    Array<{
      conceptId: string;
      name: string;
      description?: string;
      documentation?: string;
      files: Array<{
        refId: string;
        name: string;
        path: string;
        entities: Array<{
          refId: string;
          name: string;
          nodeType: string;
          file: string;
          start: number;
          end: number;
        }>;
      }>;
    }>
  > {
    const session = this.resilientSession();
    try {
      // If repo provided and IDs don't have prefix, construct full IDs
      const fullConceptIds = repo 
        ? conceptIds.map(id => id.includes('/') ? id : `${repo}/${id}`)
        : conceptIds;
      
      const result = await session.run(
        `
        // Match features by id
        MATCH (concept:Concept)
        WHERE concept.id IN $conceptIds

        // Get files connected via MODIFIES (with importance for sorting)
        OPTIONAL MATCH (concept)-[m:MODIFIES]->(file:File)

        // Get code entities via CONTAINS path (variable-length for nested structures)
        OPTIONAL MATCH (file)-[:CONTAINS*]->(entity)
        WHERE entity:Function OR entity:Page OR entity:Endpoint OR
              entity:Datamodel OR entity:UnitTest OR entity:IntegrationTest OR entity:E2etest

        // Return hierarchical data
        WITH concept, file, m.importance AS importance,
             COLLECT(DISTINCT {
               refId: entity.ref_id,
               name: entity.name,
               nodeType: [label IN labels(entity) WHERE label <> 'Data_Bank'][0],
               file: entity.file,
               start: toInteger(entity.start),
               end: toInteger(entity.end)
             }) AS entities

        // Apply performance limits: Max 20 files per concept
        WITH concept, file, importance, entities
        ORDER BY importance DESC
        LIMIT 20

        WITH concept,
             COLLECT({
               refId: file.ref_id,
               name: file.name,
               path: file.file,
               entities: entities
             }) AS files

        RETURN concept.id AS conceptId,
               concept.name AS name,
               concept.description AS description,
               concept.docs AS documentation,
               files
        `,
        { conceptIds: fullConceptIds }
      );

      const concepts: Array<{
        conceptId: string;
        name: string;
        description?: string;
        documentation?: string;
        files: Array<{
          refId: string;
          name: string;
          path: string;
          entities: Array<{
            refId: string;
            name: string;
            nodeType: string;
            file: string;
            start: number;
            end: number;
          }>;
        }>;
      }> = [];

      for (const record of result.records) {
        const conceptId = record.get("conceptId");
        const name = record.get("name");
        const description = record.get("description");
        const documentation = record.get("documentation") || "";
        const filesRaw = record.get("files");

        // Filter files to only those that have valid refId
        const files = filesRaw
          .filter((f: any) => f.refId)
          .map((file: any) => {
            // Filter entities where refId exists (not null from COLLECT)
            let entities = file.entities.filter((e: any) => e.refId);

            // Apply text matching filter if documentation exists
            if (documentation) {
              const docLower = documentation.toLowerCase();
              const matchedEntities = entities.filter((e: any) =>
                docLower.includes(e.name.toLowerCase())
              );

              // Use matched entities if found, otherwise return empty (accuracy over completeness)
              entities = matchedEntities;
            } else {
              // No documentation - return empty array
              entities = [];
            }

            // Apply performance limit: Max 50 entities per file
            entities = entities.slice(0, 50);

            return {
              refId: file.refId,
              name: file.name,
              path: file.path,
              entities: entities.map((e: any) => ({
                ...e,
                start: neo4j.isInt(e.start) ? e.start.toNumber() : e.start,
                end: neo4j.isInt(e.end) ? e.end.toNumber() : e.end,
              })),
            };
          });

        concepts.push({
          conceptId,
          name,
          description,
          documentation,
          files,
        });
      }

      return concepts;
    } finally {
      await session.close();
    }
  }
}
