import neo4j, { Driver, Session } from "neo4j-driver";
import { v4 as uuidv4 } from "uuid";
import { Storage } from "./storage.js";
import { Feature, PRRecord, LinkResult } from "../types.js";
import { formatPRMarkdown } from "./utils.js";

const Data_Bank = "Data_Bank";

/**
 * Neo4j graph-based storage implementation for features and PRs
 */
export class GraphStorage extends Storage {
  private driver: Driver;

  constructor() {
    super();
    const uri = `neo4j://${process.env.NEO4J_HOST || "localhost:7687"}`;
    const user = process.env.NEO4J_USER || "neo4j";
    const pswd = process.env.NEO4J_PASSWORD || "testtest";
    console.log("===> GraphStorage connecting to", uri, user);
    this.driver = neo4j.driver(uri, neo4j.auth.basic(user, pswd));
  }

  /**
   * Initialize indexes for better query performance
   */
  async initialize(): Promise<void> {
    const session = this.driver.session();
    try {
      // Create indexes on id/number for fast lookups
      await session.run(
        "CREATE INDEX feature_id_index IF NOT EXISTS FOR (f:Feature) ON (f.id)"
      );
      await session.run(
        "CREATE INDEX pr_number_index IF NOT EXISTS FOR (p:PullRequest) ON (p.number)"
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

  // Features

  async saveFeature(feature: Feature): Promise<void> {
    const session = this.driver.session();
    try {
      const now = Math.floor(Date.now() / 1000);
      const dateTimestamp = Math.floor(feature.lastUpdated.getTime() / 1000);

      await session.run(
        `
        MERGE (f:${Data_Bank}:Feature {id: $id})
        SET f.name = $name,
            f.description = $description,
            f.prNumbers = $prNumbers,
            f.date = $date,
            f.docs = $docs,
            f.namespace = $namespace,
            f.Data_Bank = $dataBankName,
            f.ref_id = COALESCE(f.ref_id, $refId),
            f.date_added_to_graph = COALESCE(f.date_added_to_graph, $dateAddedToGraph)
        RETURN f
        `,
        {
          id: feature.id,
          name: feature.name,
          description: feature.description,
          prNumbers: feature.prNumbers,
          date: dateTimestamp,
          docs: feature.documentation || "",
          namespace: "default",
          dataBankName: feature.id,
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );

      // Create TOUCHES relationships from PRs to this Feature
      if (feature.prNumbers.length > 0) {
        await session.run(
          `
          MATCH (f:Feature {id: $featureId})
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
    } finally {
      await session.close();
    }
  }

  async getFeature(id: string): Promise<Feature | null> {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `
        MATCH (f:Feature {id: $id})
        RETURN f
        `,
        { id }
      );

      if (result.records.length === 0) {
        return null;
      }

      const node = result.records[0].get("f");
      return this.nodeToFeature(node);
    } finally {
      await session.close();
    }
  }

  async getAllFeatures(): Promise<Feature[]> {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `
        MATCH (f:Feature)
        RETURN f
        ORDER BY f.date DESC
        `
      );

      return result.records.map((record) =>
        this.nodeToFeature(record.get("f"))
      );
    } finally {
      await session.close();
    }
  }

  async deleteFeature(id: string): Promise<void> {
    const session = this.driver.session();
    try {
      await session.run(
        `
        MATCH (f:Feature {id: $id})
        DETACH DELETE f
        `,
        { id }
      );
    } finally {
      await session.close();
    }
  }

  // PRs

  async savePR(pr: PRRecord): Promise<void> {
    const session = this.driver.session();
    try {
      const now = Math.floor(Date.now() / 1000);
      const dateTimestamp = Math.floor(pr.mergedAt.getTime() / 1000);
      const docs = await formatPRMarkdown(pr, this);

      await session.run(
        `
        MERGE (p:${Data_Bank}:PullRequest {number: $number})
        SET p.name = $name,
            p.title = $title,
            p.summary = $summary,
            p.date = $date,
            p.url = $url,
            p.files = $files,
            p.newDeclarations = $newDeclarations,
            p.docs = $docs,
            p.namespace = $namespace,
            p.Data_Bank = $dataBankName,
            p.ref_id = COALESCE(p.ref_id, $refId),
            p.date_added_to_graph = COALESCE(p.date_added_to_graph, $dateAddedToGraph)
        RETURN p
        `,
        {
          number: pr.number,
          name: `pr-${pr.number}`,
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
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );

      // Note: TOUCHES relationships are created in saveFeature()
      // when features are updated with PR numbers
    } finally {
      await session.close();
    }
  }

  async getPR(number: number): Promise<PRRecord | null> {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `
        MATCH (p:PullRequest {number: $number})
        RETURN p
        `,
        { number }
      );

      if (result.records.length === 0) {
        return null;
      }

      const node = result.records[0].get("p");
      return this.nodeToPR(node);
    } finally {
      await session.close();
    }
  }

  async getAllPRs(): Promise<PRRecord[]> {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `
        MATCH (p:PullRequest)
        RETURN p
        ORDER BY p.number ASC
        `
      );

      return result.records.map((record) => this.nodeToPR(record.get("p")));
    } finally {
      await session.close();
    }
  }

  // Metadata

  async getLastProcessedPR(): Promise<number> {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `
        MATCH (m:${Data_Bank}:FeaturesMetadata {namespace: $namespace})
        RETURN m.lastProcessedPR as lastProcessedPR
        `,
        { namespace: "default" }
      );

      if (result.records.length === 0) {
        console.log("   No lastProcessedPR found, starting from 0");
        return 0;
      }

      const value = result.records[0].get("lastProcessedPR");
      const lastPR = typeof value === 'number' ? value : (value?.toNumber ? value.toNumber() : 0);
      console.log(`   Resuming from PR #${lastPR}`);
      return lastPR;
    } catch (error) {
      console.error("   Error reading lastProcessedPR:", error);
      return 0;
    } finally {
      await session.close();
    }
  }

  async setLastProcessedPR(number: number): Promise<void> {
    const session = this.driver.session();
    try {
      const now = Math.floor(Date.now() / 1000);

      await session.run(
        `
        MERGE (m:${Data_Bank}:FeaturesMetadata {namespace: $namespace})
        SET m.lastProcessedPR = $number,
            m.ref_id = COALESCE(m.ref_id, $refId),
            m.date_added_to_graph = COALESCE(m.date_added_to_graph, $dateAddedToGraph)
        RETURN m
        `,
        {
          namespace: "default",
          number,
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );
    } finally {
      await session.close();
    }
  }

  // Themes

  async addThemes(themes: string[]): Promise<void> {
    const session = this.driver.session();
    try {
      const now = Math.floor(Date.now() / 1000);

      // Get current themes
      const result = await session.run(
        `
        MATCH (m:${Data_Bank}:FeaturesMetadata {namespace: $namespace})
        RETURN m.recentThemes as recentThemes
        `,
        { namespace: "default" }
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
        MERGE (m:${Data_Bank}:FeaturesMetadata {namespace: $namespace})
        SET m.recentThemes = $recentThemes,
            m.ref_id = COALESCE(m.ref_id, $refId),
            m.date_added_to_graph = COALESCE(m.date_added_to_graph, $dateAddedToGraph)
        RETURN m
        `,
        {
          namespace: "default",
          recentThemes,
          refId: uuidv4(),
          dateAddedToGraph: now,
        }
      );
    } finally {
      await session.close();
    }
  }

  async getRecentThemes(): Promise<string[]> {
    const session = this.driver.session();
    try {
      const result = await session.run(
        `
        MATCH (m:${Data_Bank}:FeaturesMetadata {namespace: $namespace})
        RETURN m.recentThemes as recentThemes
        `,
        { namespace: "default" }
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

  // Documentation

  async saveDocumentation(
    featureId: string,
    documentation: string
  ): Promise<void> {
    const session = this.driver.session();
    try {
      await session.run(
        `
        MATCH (f:Feature {id: $id})
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

  private nodeToFeature(node: any): Feature {
    const props = node.properties;
    return {
      id: props.id,
      name: props.name,
      description: props.description,
      prNumbers: props.prNumbers || [],
      createdAt: new Date(props.date * 1000),
      lastUpdated: new Date(props.date * 1000),
      documentation: props.docs || undefined,
    };
  }

  private nodeToPR(node: any): PRRecord {
    const props = node.properties;
    return {
      number: props.number.toNumber ? props.number.toNumber() : props.number,
      title: props.title,
      summary: props.summary,
      mergedAt: new Date(props.date * 1000),
      url: props.url,
      files: props.files || [],
      newDeclarations: props.newDeclarations
        ? JSON.parse(props.newDeclarations)
        : undefined,
    };
  }

  // Feature-File Linking

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

  async linkFeaturesToFiles(featureId?: string): Promise<LinkResult> {
    const session = this.driver.session();
    try {
      // Get features to process
      const features = featureId
        ? [await this.getFeature(featureId)].filter((f): f is Feature => f !== null)
        : await this.getAllFeatures();

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
          MATCH (f:Feature {id: $featureId})
          MATCH (pr:PullRequest)-[:TOUCHES]->(f)
          WHERE pr.files IS NOT NULL
          UNWIND pr.files as file
          RETURN file, COUNT(pr) as prCount
          `,
          { featureId: feature.id }
        );

        // Get total PR count for this feature
        const totalPRs = feature.prNumbers.length;

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
        let linksCreatedForFeature = 0;
        let filesInDocsForFeature = 0;
        let filesNotInDocsForFeature = 0;

        for (const record of filePathsResult.records) {
          const filePath = record.get("file");
          const prCountRaw = record.get("prCount");
          const prCount = prCountRaw?.toNumber ? prCountRaw.toNumber() : prCountRaw || 0;

          // Calculate frequency score (0-1)
          const frequency = totalPRs > 0 ? prCount / totalPRs : 0;

          // Check if file is in documentation
          const inDocs = this.isFileInDocumentation(filePath, documentation);

          // Calculate importance using two-tier system
          // Files in docs: 0.5-1.0, Files not in docs: 0.0-0.49
          const importance = inDocs
            ? 0.5 + frequency * 0.5
            : frequency * 0.49;

          // Track statistics
          if (inDocs) {
            filesInDocsForFeature++;
          } else {
            filesNotInDocsForFeature++;
          }

          // Match File nodes where the file property ends with the PR file path
          // This handles cases like "owner/repo/src/me.ts" matching "src/me.ts"
          const linkResult = await session.run(
            `
            MATCH (f:Feature {id: $featureId})
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
          const count = linkedCount?.toNumber ? linkedCount.toNumber() : linkedCount || 0;
          linksCreatedForFeature += count;
        }

        result.featureFileLinks.push({
          featureId: feature.id,
          filesLinked: linksCreatedForFeature,
          filesInDocs: filesInDocsForFeature,
          filesNotInDocs: filesNotInDocsForFeature,
        });
        result.filesLinked += linksCreatedForFeature;
        result.filesInDocs += filesInDocsForFeature;
        result.filesNotInDocs += filesNotInDocsForFeature;
      }

      return result;
    } finally {
      await session.close();
    }
  }

}
